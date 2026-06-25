from __future__ import annotations

import warp as wp


@wp.kernel
def scatter_target_kernel(
    pts: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3),
    rest_offset: wp.array(dtype=wp.vec3),
    dense_target: wp.array(dtype=wp.vec3),
):
    """Scatter the per-constraint bone target into a dense per-vertex array.

    ``dense_target[pts[i]] = target_pos[i] + rest_offset[i]``. For humanV2
    pts covers every muscle vertex, so dense_target is fully written; it
    feeds the Galerkin restrict to build the coarse full-pin target.
    """
    i = wp.tid()
    dense_target[pts[i]] = target_pos[i] + rest_offset[i]


@wp.kernel
def restrict_galerkin_scatter_kernel(
    cage_idx: wp.array(dtype=wp.int32),       # fine vert -> coarse tet (prolong P)
    uvw: wp.array(dtype=wp.vec3),
    coarse_tet: wp.array(dtype=wp.vec4i),
    dense_target: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),               # out: b = P^T T (atomic scatter)
):
    """b = P^T T_dense: scatter each fine vert's bone target into its coarse tet's
    4 corners weighted by the prolong barycentric weights. First half of the
    Galerkin restrict Cdef = (P^T P)^-1 (P^T T)."""
    i = wp.tid()
    tet = coarse_tet[cage_idx[i]]
    uv = uvw[i]
    a = 1.0 - uv[0] - uv[1] - uv[2]
    T = dense_target[i]
    wp.atomic_add(b, tet[0], a * T)
    wp.atomic_add(b, tet[1], uv[0] * T)
    wp.atomic_add(b, tet[2], uv[1] * T)
    wp.atomic_add(b, tet[3], uv[2] * T)


@wp.kernel
def block_solve_kernel(
    minv: wp.array(dtype=wp.float32),         # per-muscle pinv(P^T P), row-major blocks
    base: wp.array(dtype=wp.int32),           # per coarse vert: first vert of its block
    dim: wp.array(dtype=wp.int32),            # per coarse vert: block size
    matoff: wp.array(dtype=wp.int32),         # per coarse vert: block offset in minv
    b: wp.array(dtype=wp.vec3),
    coarse_target: wp.array(dtype=wp.vec3),   # out: Cdef = Minv b
):
    """Cdef[c] = sum_j Minv_block[c, j] * b[base+j]. Minv is block-diagonal per
    muscle (a fine vert only weights its own muscle's cage), so each coarse vert
    reads its block's row."""
    c = wp.tid()
    bs = base[c]
    d = dim[c]
    row = matoff[c] + (c - bs) * d
    acc = wp.vec3(0.0, 0.0, 0.0)
    for j in range(d):
        acc = acc + minv[row + j] * b[bs + j]
    coarse_target[c] = acc


@wp.kernel
def prolong_kernel(
    coarse_pos: wp.array(dtype=wp.vec3),
    coarse_tet: wp.array(dtype=wp.vec4i),
    cage_idx: wp.array(dtype=wp.int32),
    uvw: wp.array(dtype=wp.vec3),
    fine_pos: wp.array(dtype=wp.vec3),
):
    """fine.pos[i] = blend(coarse.pos at containing coarse tet).

    Direct (v1) prolongation — mirrors update_fine_mesh_mfree_kernel in
    realtimeMuscle/engine/arap_hpbd_nogui.py:253.
    """
    i = wp.tid()
    t = cage_idx[i]
    idx = coarse_tet[t]
    b = uvw[i]
    u = b[0]
    v = b[1]
    w = b[2]
    a = 1.0 - u - v - w
    fine_pos[i] = (a * coarse_pos[idx[0]] + u * coarse_pos[idx[1]]
                   + v * coarse_pos[idx[2]] + w * coarse_pos[idx[3]])


# ─── Kinematic pin-to-target (direct pos overwrite, NOT XPBD solve) ─────────
# These set pos[] directly each substep; they are not constraint-solve kernels
# (no dP/dPw accumulation) so they live here with the MG transfer kernels
# rather than in constraint/solve_kernels.py.

@wp.kernel
def pintotarget_overwrite_masked_kernel(
    pts: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3),
    rest_offset: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=wp.int32),     # 1 = apply pin, 0 = skip (freed vert)
    pos: wp.array(dtype=wp.vec3),
):
    """Hard-overwrite ``pos[pts[i]] = target_pos[i] + rest_offset[i]`` for pinned
    tendon verts; skip freed verts (active==0) so the fascia two-way glue +
    belly-attach can pull them inward."""
    i = wp.tid()
    if active[i] == 0:
        return
    pos[pts[i]] = target_pos[i] + rest_offset[i]


# ── Bone-surface target (target_mode="mesh", use_sliding gates re-projection) ─
# Each muscle vert binds to its nearest point on the bone SURFACE mesh. The
# binding (face + barycentric uv) may re-project ("slide") tangentially as the
# muscle moves, bounded to slide_range from the original anchor; the normal gap
# (rest distance from bone) is preserved so the muscle glides over the bone
# without penetrating or detaching.

@wp.kernel
def bone_bind_kernel(
    mesh: wp.uint64,
    rest_pos: wp.array(dtype=wp.vec3),
    max_dist: wp.float32,
    out_face: wp.array(dtype=wp.int32),
    out_uv: wp.array(dtype=wp.vec2),
    out_gap: wp.array(dtype=wp.float32),
):
    """Bind: closest point on the bone mesh to each muscle rest vert -> face,
    barycentric (u,v), and signed normal gap (rest - surface)·n."""
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, rest_pos[i], max_dist)
    if q.result:
        out_face[i] = q.face
        out_uv[i] = wp.vec2(q.u, q.v)
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        n = wp.mesh_eval_face_normal(mesh, q.face)
        out_gap[i] = wp.dot(rest_pos[i] - cp, n)
    else:
        out_face[i] = 0
        out_uv[i] = wp.vec2(0.0, 0.0)
        out_gap[i] = 0.0


@wp.kernel
def bone_slide_target_kernel(
    mesh: wp.uint64,                          # current frame's bone mesh
    muscle_pos: wp.array(dtype=wp.vec3),      # current muscle positions
    anchor_face: wp.array(dtype=wp.int32),    # leash origin (fixed bind) face
    anchor_uv: wp.array(dtype=wp.vec2),       # leash origin barycentric
    bind_face: wp.array(dtype=wp.int32),      # current binding face (in/out)
    bind_uv: wp.array(dtype=wp.vec2),         # current binding bary (in/out)
    normal_gap: wp.array(dtype=wp.float32),   # preserved rest normal distance
    slide_range: wp.float32,                  # max tangential drift from anchor
    reproject: wp.int32,                      # 1 = re-find closest (slide), else follow
    max_dist: wp.float32,
    target_out: wp.array(dtype=wp.vec3),      # out: per-vert bone target
):
    i = wp.tid()
    # Leash origin on the current (deformed) bone.
    anchor_p = wp.mesh_eval_position(mesh, anchor_face[i], anchor_uv[i][0], anchor_uv[i][1])
    if reproject == 1:
        q = wp.mesh_query_point_sign_winding_number(mesh, muscle_pos[i], max_dist)
        if q.result:
            cand = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
            # Accept the slide only if it stays within slide_range of the anchor;
            # otherwise keep the old binding (caps cumulative tangential drift).
            if wp.length(cand - anchor_p) <= slide_range:
                bind_face[i] = q.face
                bind_uv[i] = wp.vec2(q.u, q.v)
    f = bind_face[i]
    cp = wp.mesh_eval_position(mesh, f, bind_uv[i][0], bind_uv[i][1])
    n = wp.mesh_eval_face_normal(mesh, f)
    target_out[i] = cp + n * normal_gap[i]