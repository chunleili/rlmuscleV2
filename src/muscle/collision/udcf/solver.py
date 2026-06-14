"""UdcfDcdCollider — host orchestration for the Gaia DCD port.

Device kernels live in ``muscle.collision.udcf.kernel``; this module owns
broadphase hashgrid sizing, the per-iteration kernel launch order, and the
public ``UdcfParams`` / ``UdcfDcdCollider`` API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import warp as wp

from muscle.collision.udcf.kernel import (
    centroid_radius_kernel,
    face_normal_kernel,
    max_radius_reduce_kernel,
    broadphase_find_pairs_kernel,
    tri_aabb_kernel,
    broadphase_find_pairs_bvh_kernel,
    narrow_kernel,
    narrow_respond_gs_kernel,
    respond_kernel,
    apply_kernel,
    _clamp_contact_depth_kernel,
    _peak_update_kernel,
)


# ======================================================================
# Host orchestration
# ======================================================================

@dataclass
class UdcfParams:
    multiplier: float = 1.01          # collisionConstraintMultiplier
    friction_static: float = 0.5
    friction_dynamic: float = 0.4
    offset: float = 1.0e-6            # final accept threshold
    thickness: float = 0.0            # broadphase sphere slack
    cell_size: Optional[float] = None  # None → 2·max_radius + thickness (recomputed
                                       # every broadphase from a device readback).
                                       # If set manually the grid is built with that
                                       # cell size, but the query radius is
                                       # lower-bounded by 2·largest-triangle-radius
                                       # of the INITIAL geometry + thickness to
                                       # prevent silently dropping overlapping pairs.
                                       # If triangles grow significantly larger at
                                       # runtime, leave as None.
    max_pairs: int = 1 << 20
    max_contacts: int = 1 << 18
    allow_self: bool = False          # movable-vs-movable pairs
    grid_dim: int = 128
    normal_cull: float = 1.0          # skip self pairs with n_i·n_j > this
                                      # (co-oriented faces can't penetrate).
                                      # 1.0 disables; 0.0 culls same-hemisphere.
    use_bvh: bool = False             # broadphase backend.  False (default):
                                      # hashgrid — fastest here (~161ms on the
                                      # humanV2 muscle self set).  True: wp.Bvh
                                      # over per-triangle AABBs, rebuilt each
                                      # broadphase.  AABBs are tighter than the
                                      # hashgrid's single-cell sphere test, but
                                      # the per-broadphase tree REBUILD (~600ms
                                      # at 238k tris) costs more than it saves on
                                      # this mesh; kept as an option for meshes
                                      # with very non-uniform triangle sizes or
                                      # small tri counts where rebuild is cheap.
    jacobi_avg: bool = True           # True: dP/count (bounded, can under-
                                      # correct clusters); False: summed dP
                                      # (closer to the reference's serial
                                      # Gauss-Seidel accumulation, can
                                      # overshoot)
    gauss_seidel: bool = False        # True: faithful serial narrow+respond
                                      # (single-thread kernel; pair k detects
                                      # on positions corrected by pairs <k,
                                      # exactly like the reference loop).
                                      # Slower but stable under fold/press
                                      # loads where Jacobi tangles.
    max_push: float = 0.0             # >0: clamp the per-contact depth C at
                                      # this value (meters) before response.
                                      # Robustness guard ABSENT from the
                                      # reference: C1/C2 are plane distances,
                                      # not overlap depths, and steep small-
                                      # area crossings of large triangles can
                                      # report C far beyond the true overlap
                                      # — one substep then applies a huge
                                      # position jump (= velocity kick) and
                                      # the sim explodes. Clamping resolves
                                      # deep contacts over several substeps
                                      # instead. 0 disables (reference
                                      # behavior).


class CcdInterface(Protocol):
    """Reserved continuous-collision hook. Implementations get called once
    per solve() after the DCD pass with the collider as argument."""

    def solve(self, collider: "UdcfDcdCollider") -> None: ...


class UdcfDcdCollider:
    """Gaia_UDCF DCD between a movable vert set and an OPTIONAL static
    (per-frame animated) mesh, plus optional movable self-collision.

    Pass ``static_verts``/``static_tris`` for movable-vs-static (e.g. muscle
    vs animated bone).  Omit them (the default ``None``) with
    ``UdcfParams(allow_self=True)`` for self-collision only — no placeholder
    mesh, and the broadphase cell is sized off the movable triangles alone.

    ``pos`` is the live solver position array (e.g. ``OneWaySim.pos``) —
    the collider reads AND writes it in :meth:`solve`. Register on the
    solver's per-iteration hook::

        coll = UdcfDcdCollider(sim.pos, inv_mass, surf_tris, bone_v, bone_t)
        sim.collision_solvers_iter = [coll.solve]

    Static-mesh winding: triangle ``cross(e1, e2)`` normals must point OUT
    of the static body (SNU bone .geo files need the ``[:, [0, 2, 1]]``
    column swap, same as AnimatedBoneMeshCollider).
    """

    def __init__(self, pos: wp.array, inv_mass: wp.array,
                 surface_tris: np.ndarray,
                 static_verts: np.ndarray | None = None,
                 static_tris: np.ndarray | None = None,
                 params: UdcfParams | None = None,
                 device: str | None = None,
                 ccd: CcdInterface | None = None):
        self.params = params or UdcfParams()
        self.device = device or str(pos.device)
        self.pos = pos
        self.ccd = ccd
        p = self.params

        # Static body is OPTIONAL: pass None (the default) for self-collision
        # only (movable-vs-movable, needs allow_self=True).  No placeholder mesh.
        self.has_static = static_verts is not None and len(static_verts) > 0
        if not self.has_static and not p.allow_self:
            raise ValueError("UdcfDcdCollider: no static body and allow_self=False "
                             "-> nothing to collide; pass a static mesh or set "
                             "allow_self=True")

        self.n_mov = int(pos.shape[0])
        self.n_static = int(static_verts.shape[0]) if self.has_static else 0
        n_total = self.n_mov + self.n_static

        mov_tris = np.asarray(surface_tris, dtype=np.int32).reshape(-1, 3)
        if self.has_static:
            sta_tris = (np.asarray(static_tris, dtype=np.int32).reshape(-1, 3)
                        + self.n_mov)
            all_tris = np.vstack([mov_tris, sta_tris])
        else:
            all_tris = mov_tris
        self.n_tris = int(all_tris.shape[0])
        self.tris = wp.array2d(all_tris, dtype=wp.int32, device=self.device)
        group_np = np.zeros(self.n_tris, dtype=np.int32)
        group_np[mov_tris.shape[0]:] = 1          # no-op when there is no static
        self.tri_group = wp.array(group_np, dtype=wp.int32, device=self.device)

        im_np = inv_mass.numpy().astype(np.float32)
        if self.has_static:
            im_np = np.concatenate([im_np, np.zeros(self.n_static, dtype=np.float32)])
        self.inv_mass = wp.array(im_np, dtype=wp.float32, device=self.device)

        x_np = np.zeros((n_total, 3), dtype=np.float32)
        if self.has_static:
            x_np[self.n_mov:] = static_verts
        self.x = wp.array(x_np, dtype=wp.vec3, device=self.device)
        self.x_prev = wp.clone(self.x)

        self.centroids = wp.zeros(self.n_tris, dtype=wp.vec3, device=self.device)
        self.radii = wp.zeros(self.n_tris, dtype=wp.float32, device=self.device)
        self.normals = wp.zeros(self.n_tris, dtype=wp.vec3, device=self.device)
        self._max_radius = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.pair_a = wp.zeros(p.max_pairs, dtype=wp.int32, device=self.device)
        self.pair_b = wp.zeros(p.max_pairs, dtype=wp.int32, device=self.device)
        self.n_pairs = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.grid = wp.HashGrid(p.grid_dim, p.grid_dim, p.grid_dim,
                                device=self.device)
        # Optional BVH broadphase scratch (UdcfParams.use_bvh).
        self.bvh = None
        if p.use_bvh:
            self.aabb_lo = wp.zeros(self.n_tris, dtype=wp.vec3, device=self.device)
            self.aabb_hi = wp.zeros(self.n_tris, dtype=wp.vec3, device=self.device)
            wp.launch(tri_aabb_kernel, dim=self.n_tris,
                      inputs=[self.tris, self.x, wp.float32(p.thickness),
                              self.aabb_lo, self.aabb_hi], device=self.device)
            self.bvh = wp.Bvh(self.aabb_lo, self.aabb_hi)

        mc = p.max_contacts
        self.n_contacts = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.c_cur = wp.zeros((mc, 3), dtype=wp.int32, device=self.device)
        self.c_closest = wp.zeros((mc, 3), dtype=wp.int32, device=self.device)
        self.c_C = wp.zeros(mc, dtype=wp.float32, device=self.device)
        self.c_C1 = wp.zeros(mc, dtype=wp.float32, device=self.device)
        self.c_C2 = wp.zeros(mc, dtype=wp.float32, device=self.device)
        self.c_n = wp.zeros(mc, dtype=wp.vec3, device=self.device)
        self.c_b1 = wp.zeros(mc, dtype=wp.vec3, device=self.device)
        self.c_b2 = wp.zeros(mc, dtype=wp.vec3, device=self.device)
        # respond_kernel takes per-contact friction (golden replay needs
        # per-mesh values); production fills them with the params constants
        self.mu_s = wp.full(mc, p.friction_static, dtype=wp.float32,
                            device=self.device)
        self.mu_d = wp.full(mc, p.friction_dynamic, dtype=wp.float32,
                            device=self.device)

        self.dP = wp.zeros(n_total, dtype=wp.vec3, device=self.device)
        self.dPw = wp.zeros(n_total, dtype=wp.float32, device=self.device)

        # Try 1-D slice views; fall back to offset-based copies if unsupported
        try:
            self._x_mov = self.x[:self.n_mov]
            self._xp_mov = self.x_prev[:self.n_mov]
            self._x_sta = self.x[self.n_mov:]
            self._xp_sta = self.x_prev[self.n_mov:]
            # Verify slicing returns a proper wp.array (not None)
            assert self._x_mov is not None and self._x_sta is not None
            self._use_slice_views = True
        except Exception:
            self._x_mov = None
            self._xp_mov = None
            self._x_sta = None
            self._xp_sta = None
            self._use_slice_views = False

        self._overflow_warned = False
        # max n_contacts seen since last broadphase, tracked on device so
        # solve() never forces a host sync in the solver's hot loop
        self._peak_contacts = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._gs_dbg = wp.zeros(2, dtype=wp.float32, device=self.device)

        # One-time readback of the initial max triangle bounding radius.
        # Used in _broadphase to lower-bound the query_radius when cell_size
        # is fixed manually, preventing silently dropped overlapping pairs.
        if p.cell_size is not None:
            wp.launch(centroid_radius_kernel, dim=self.n_tris,
                      inputs=[self.tris, self.x, self.centroids, self.radii],
                      device=self.device)
            wp.launch(max_radius_reduce_kernel, dim=1,
                      inputs=[self.radii, self._max_radius], device=self.device)
            wp.synchronize()
            self._init_max_radius = float(self._max_radius.numpy()[0])
        else:
            self._init_max_radius = 0.0  # unused when cell_size is None

    def set_static_verts(self, verts: np.ndarray) -> None:
        """New static pose (per-frame bone animation). The previous pose
        becomes x_prev for friction's contact velocity."""
        if not self.has_static:
            raise RuntimeError("UdcfDcdCollider has no static body "
                               "(self-collision only) — set_static_verts invalid")
        if self._use_slice_views:
            wp.copy(self._xp_sta, self._x_sta)
            self._x_sta.assign(np.asarray(verts, dtype=np.float32))
        else:
            # Fallback: copy current static region to prev, then update
            wp.copy(self.x_prev, self.x,
                    dest_offset=self.n_mov, src_offset=self.n_mov,
                    count=self.n_static)
            v_wp = wp.array(np.asarray(verts, dtype=np.float32).reshape(-1, 3),
                            dtype=wp.vec3, device=self.device)
            wp.copy(self.x, v_wp,
                    dest_offset=self.n_mov, src_offset=0,
                    count=self.n_static)

    def _broadphase(self) -> None:
        p = self.params
        if p.use_bvh:
            self._broadphase_bvh()
            return
        wp.launch(centroid_radius_kernel, dim=self.n_tris,
                  inputs=[self.tris, self.x, self.centroids, self.radii],
                  device=self.device)
        if p.normal_cull < 1.0:
            wp.launch(face_normal_kernel, dim=self.n_tris,
                      inputs=[self.tris, self.x, self.normals], device=self.device)
        if p.cell_size is None:
            wp.launch(max_radius_reduce_kernel, dim=1,
                      inputs=[self.radii, self._max_radius], device=self.device)
            cell = max(2.0 * float(self._max_radius.numpy()[0]) + p.thickness,
                       1.0e-4)
            query_radius = cell
        else:
            cell = p.cell_size
            # Lower-bound the query radius by 2*initial_max_radius + thickness so
            # that a manually chosen cell_size smaller than the triangle extents
            # never silently drops overlapping pairs.  Grid build keeps the user's
            # cell for bucket granularity; only the neighbour-query reach is clamped.
            query_radius = max(cell,
                               2.0 * self._init_max_radius + p.thickness)
        self.grid.build(self.centroids, cell)
        self.n_pairs.zero_()
        wp.launch(broadphase_find_pairs_kernel, dim=self.n_tris,
                  inputs=[self.grid.id, self.centroids, self.radii, self.tris,
                          self.tri_group, self.normals, wp.float32(p.normal_cull),
                          wp.int32(int(p.allow_self)),
                          wp.float32(query_radius), wp.float32(p.thickness),
                          wp.int32(p.max_pairs),
                          self.pair_a, self.pair_b, self.n_pairs],
                  device=self.device)

    def _broadphase_bvh(self) -> None:
        """wp.Bvh broadphase (UdcfParams.use_bvh).  Refresh per-triangle AABBs,
        REBUILD the tree on the current boxes (refit alone keeps the stale tree
        topology and degrades query cost as the mesh deforms), then emit
        AABB-overlap pairs."""
        p = self.params
        wp.launch(tri_aabb_kernel, dim=self.n_tris,
                  inputs=[self.tris, self.x, wp.float32(p.thickness),
                          self.aabb_lo, self.aabb_hi], device=self.device)
        self.bvh = wp.Bvh(self.aabb_lo, self.aabb_hi)
        self.n_pairs.zero_()
        wp.launch(broadphase_find_pairs_bvh_kernel, dim=self.n_tris,
                  inputs=[self.bvh.id, self.aabb_lo, self.aabb_hi, self.tris,
                          self.tri_group, wp.int32(int(p.allow_self)),
                          wp.int32(p.max_pairs),
                          self.pair_a, self.pair_b, self.n_pairs],
                  device=self.device)

    def solve(self, i_iter: int = 0) -> None:
        """One DCD pass: refresh movable verts, (re)build pairs on the
        first iteration of each substep, then narrow → respond → apply."""
        p = self.params
        if self._use_slice_views:
            if i_iter == 0:
                wp.copy(self._xp_mov, self.pos)   # first-iteration snapshot
                                                   # (last stmt of prev iter
                                                   # via the solver hook, so
                                                   # this is the pre-DCD pos)
            wp.copy(self._x_mov, self.pos)
        else:
            if i_iter == 0:
                wp.copy(self.x_prev, self.pos,
                        dest_offset=0, src_offset=0, count=self.n_mov)
            wp.copy(self.x, self.pos,
                    dest_offset=0, src_offset=0, count=self.n_mov)
        if i_iter == 0:
            self._broadphase()
            self._peak_contacts.zero_()
        self.n_contacts.zero_()
        if p.gauss_seidel:
            wp.launch(narrow_respond_gs_kernel, dim=1,
                      inputs=[self.pair_a, self.pair_b, self.n_pairs,
                              wp.int32(p.max_pairs), self.tris,
                              self.x, self.x_prev, self.inv_mass,
                              wp.float32(p.offset), wp.float32(p.multiplier),
                              wp.float32(p.friction_static),
                              wp.float32(p.friction_dynamic),
                              wp.float32(p.max_push),
                              self.n_contacts, self._gs_dbg],
                      device=self.device)
            wp.launch(_peak_update_kernel, dim=1,
                      inputs=[self.n_contacts, self._peak_contacts],
                      device=self.device)
            # write corrected movable verts back to the live solver array
            if self._use_slice_views:
                wp.copy(self.pos, self._x_mov)
            else:
                wp.copy(self.pos, self.x,
                        dest_offset=0, src_offset=0, count=self.n_mov)
            if self.ccd is not None:
                self.ccd.solve(self)
            return
        wp.launch(narrow_kernel, dim=p.max_pairs,
                  inputs=[self.pair_a, self.pair_b, self.n_pairs, self.tris,
                          self.x, wp.float32(p.offset),
                          wp.int32(p.max_contacts),
                          self.n_contacts, self.c_cur, self.c_closest,
                          self.c_C, self.c_C1, self.c_C2,
                          self.c_n, self.c_b1, self.c_b2],
                  device=self.device)
        wp.launch(_peak_update_kernel, dim=1,
                  inputs=[self.n_contacts, self._peak_contacts],
                  device=self.device)
        if p.max_push > 0.0:
            wp.launch(_clamp_contact_depth_kernel, dim=p.max_contacts,
                      inputs=[self.n_contacts, self.c_C,
                              wp.float32(p.max_push)],
                      device=self.device)
        self.dP.zero_()
        self.dPw.zero_()
        wp.launch(respond_kernel, dim=p.max_contacts,
                  inputs=[self.n_contacts, self.c_cur, self.c_closest,
                          self.c_C, self.c_n, self.c_b1, self.c_b2,
                          self.x, self.x_prev, self.inv_mass,
                          wp.float32(p.multiplier), self.mu_s, self.mu_d,
                          self.dP, self.dPw],
                  device=self.device)
        wp.launch(apply_kernel, dim=self.n_mov,
                  inputs=[self.pos, self.dP, self.dPw,
                          wp.int32(1 if p.jacobi_avg else 0)],
                  device=self.device)
        if self.ccd is not None:
            self.ccd.solve(self)

    def stats(self) -> dict:
        """Host-side diagnostics (one device sync). Warns once per overflow.

        ``n_contacts`` is the peak value seen since the last broadphase
        (i_iter == 0) call, so it remains non-zero after the penetration
        is resolved within a substep.
        """
        np_ = int(self.n_pairs.numpy()[0])
        nc = int(self._peak_contacts.numpy()[0])
        p = self.params
        if (np_ > p.max_pairs or nc > p.max_contacts) and not self._overflow_warned:
            self._overflow_warned = True
            print(f"[udcf_dcd] WARNING: overflow pairs={np_}/{p.max_pairs} "
                  f"contacts={nc}/{p.max_contacts} — raise UdcfParams caps")
        return {"n_pairs": min(np_, p.max_pairs),
                "n_contacts": min(nc, p.max_contacts)}
