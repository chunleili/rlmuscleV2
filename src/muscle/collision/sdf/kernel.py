"""Particle-vs-rigid-SDF collision in Warp — the simple soft-vs-rigid path.

For muscle-bone coupling the rigid body (bone, floor) is best represented as a
signed distance field: every soft-body vertex queries the SDF and, if it has
penetrated, is pushed back out along the SDF gradient. No broadphase, no
candidate pairs, no triangle winding, no self-collision filtering — interior
vertices have ``phi > 0`` and are automatic no-ops, so even surface extraction
is unnecessary. This is the classic PBD soft-vs-rigid contact (cf. Newton's
``eval_particle_body_contact`` and Bender's particle-rigid contact).

Pipeline (one ``wp.launch`` per substep):

  ``solve_sdf_contact_kernel``  per-vertex penetration push + contact flag

The kernel resolves penetration at the POSITION level (pushes ``pos`` and
``pprev`` by the same delta, so the push injects no velocity) and writes
``is_collide`` / ``contact_normal``. Normal restitution and tangential friction
are left to ``MuscleSim.update_velocities`` (its ``with_contact`` branch), the
standard PBD velocity solve — this file never touches velocity.

Shape types::

    0 = plane   phi = p.y - plane_y,        n = (0, 1, 0)
    1 = sphere  phi = |p - sphere_c| - r,   n = normalize(p - sphere_c)

All arithmetic is fp32.
"""
from __future__ import annotations

import warp as wp


@wp.func
def sdf_plane(p: wp.vec3, plane_y: wp.float32):
    """Horizontal plane at ``y = plane_y``, solid below. Outward normal +Y."""
    phi = p[1] - plane_y
    return phi, wp.vec3(0.0, 1.0, 0.0)


@wp.func
def sdf_sphere(p: wp.vec3, c: wp.vec3, r: wp.float32):
    """Solid sphere, centre ``c`` radius ``r``. Outward normal points away
    from the centre. Degenerate at the exact centre (``d≈0``) — fall back to
    +Y so the push is still well-defined."""
    d = p - c
    dn = wp.length(d)
    if dn < 1.0e-9:
        return -r, wp.vec3(0.0, 1.0, 0.0)
    return dn - r, d / dn


@wp.kernel
def solve_sdf_contact_kernel(
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    shape_type: wp.int32,            # 0 = plane, 1 = sphere
    plane_y: wp.float32,
    sphere_c: wp.vec3,
    sphere_r: wp.float32,
    contact_offset: wp.float32,      # push out until phi >= contact_offset
    # outputs
    is_collide: wp.array(dtype=wp.int32),
    contact_normal: wp.array(dtype=wp.vec3),
):
    i = wp.tid()

    # Pinned / infinite-mass verts (and rigid plane verts, if any) never move.
    # Non-contact verts are left untouched: the host zeroes is_collide /
    # contact_normal once per substep, so multiple SDF kernels can run in
    # sequence (e.g. ground + sphere) without one clearing the other's flags.
    if inv_mass[i] == 0.0:
        return

    p = pos[i]
    if shape_type == 1:
        phi, n = sdf_sphere(p, sphere_c, sphere_r)
    else:
        phi, n = sdf_plane(p, plane_y)

    if phi < contact_offset:
        # Push the vertex out to the (offset) surface. Move pprev by the same
        # amount so v = (pos - pprev)/dt is unchanged by the push — velocity
        # response is handled separately in update_velocities.
        dP = n * (contact_offset - phi)
        pos[i] = p + dP
        pprev[i] = pprev[i] + dP
        is_collide[i] = 1
        contact_normal[i] = n


@wp.kernel
def solve_sdf_contact_pos_kernel(
    pos: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    shape_type: wp.int32,            # 0 = plane, 1 = sphere
    plane_y: wp.float32,
    sphere_c: wp.vec3,
    sphere_r: wp.float32,
    contact_offset: wp.float32,
):
    """Position-only SDF push for quasi-static solvers (no velocity / pprev).

    For relaxation solvers like the GMG ``OneWaySim`` there is no velocity
    state — collision is just a positional projection: push every penetrating
    vertex out of the rigid SDF along the gradient. ``inv_mass == 0`` verts
    (e.g. bone-pinned attachment points) are skipped so the push never fights
    the pin-to-target overwrite.
    """
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    p = pos[i]
    if shape_type == 1:
        phi, n = sdf_sphere(p, sphere_c, sphere_r)
    else:
        phi, n = sdf_plane(p, plane_y)
    if phi < contact_offset:
        pos[i] = p + n * (contact_offset - phi)


@wp.kernel
def solve_mesh_contact_pos_kernel(
    pos: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    mesh: wp.uint64,                 # wp.Mesh.id of the rigid collider (e.g. bone)
    max_dist: wp.float32,            # closest-point search radius
    contact_offset: wp.float32,      # keep verts this far outside the surface
    max_push: wp.float32,            # reject contacts deeper than this (spurious)
):
    """Position-only push of verts out of a rigid TRIANGLE-MESH collider.

    The general-geometry counterpart of :func:`solve_sdf_contact_pos_kernel`:
    instead of an analytic SDF it queries a ``wp.Mesh`` (e.g. the skeleton)
    via ``mesh_query_point_sign_normal``. A vertex inside the mesh (or within
    ``contact_offset`` of it) is pushed back out along the surface normal. For
    quasi-static solvers — no velocity, position only. ``inv_mass == 0`` verts
    are skipped. The host owns the mesh and ``refit()``s it when the collider
    moves (e.g. per animation frame).

    ROBUSTNESS: ``mesh_query_point_sign_normal``'s sign is unreliable near thin
    bone features / sharp edges (the reason the bone winding is pre-flipped). A
    spurious sign flip reports a vert that is actually just OUTSIDE as deeply
    INSIDE relative to a far face, and the raw push ``x - n*err`` then catapults
    it cm-scale onto that face (single-vert spikes). Real per-substep penetration
    is sub-cm, so a contact reporting depth > ``max_push`` is rejected as a bad
    query. The degenerate ``|delta|≈0`` case (vert on the surface → unstable
    ``normalize``) is also skipped.
    """
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return
    x = pos[i]
    query = wp.mesh_query_point_sign_normal(mesh, x, max_dist)
    if query.result:
        cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        delta = x - cp
        dlen = wp.length(delta)
        if dlen < 1.0e-9:                          # on the surface: normal undefined
            return
        dist = dlen * query.sign                   # signed: < 0 inside the mesh
        err = dist - contact_offset
        if err < 0.0 and -err <= max_push:         # reject spurious deep (sign-flip)
            n = (delta / dlen) * query.sign        # outward surface normal
            pos[i] = x - n * err                    # push out to offset surface


@wp.kernel
def solve_sphere_soft_contact_kernel(
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    sphere_c: wp.vec3,
    sphere_r: wp.float32,
    inv_mass_sphere: wp.float32,     # 1/M of the rigid sphere (0 => kinematic)
    contact_offset: wp.float32,
    # outputs
    is_collide: wp.array(dtype=wp.int32),
    contact_normal: wp.array(dtype=wp.vec3),
    sphere_dC: wp.array(dtype=wp.vec3),   # 1-elem accumulator of sphere recoil
):
    """Two-way soft-vertex vs dynamic-sphere contact (position based).

    Splits each penetration correction between the soft vertex and the rigid
    sphere by inverse mass (Newton's third law in PBD form): the vertex moves
    out by ``w_i/(w_i+w_s)`` of the depth, the sphere recoils by
    ``w_s/(w_i+w_s)`` — accumulated into ``sphere_dC`` for the host to apply to
    the single sphere body. With ``inv_mass_sphere = 0`` the sphere is
    kinematic (infinite mass) and this reduces to the one-way push.
    """
    # Non-contact verts are left untouched (host zeroes is_collide /
    # contact_normal each substep) so this can run after the ground-plane
    # kernel without clearing its contact flags.
    i = wp.tid()
    if inv_mass[i] == 0.0:
        return

    d = pos[i] - sphere_c
    dist = wp.length(d)
    phi = dist - sphere_r
    if phi < contact_offset and dist > 1.0e-9:
        n = d / dist
        depth = contact_offset - phi
        w_i = inv_mass[i]
        w_s = inv_mass_sphere
        denom = w_i + w_s
        if denom <= 0.0:
            return
        dP = n * (depth * w_i / denom)
        pos[i] = pos[i] + dP
        pprev[i] = pprev[i] + dP
        is_collide[i] = 1
        contact_normal[i] = n
        # Sphere recoils opposite the push, weighted by its mass share.
        wp.atomic_add(sphere_dC, 0, n * (-depth * w_s / denom))
