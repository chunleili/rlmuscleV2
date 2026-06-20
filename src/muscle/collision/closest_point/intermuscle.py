"""Inter-muscle collision via tet-barycentric inside test + closest-point push.

Resolve only muscle_A-vs-muscle_B penetration, NEVER muscle-vs-itself.  Across
two DISTINCT muscles every element of the other body is a valid target, so there
is no k-ring / self-adjacency filtering to do.

Inside / outside is decided by **tetrahedral barycentric containment**, not by a
surface pseudonormal sign.  ``mesh_query_point_sign_normal``'s angle-weighted
pseudonormal flips sign on thin / concave / near-non-manifold muscle surfaces
(exactly the densely-packed folded poses we care about), reporting verts that are
OUTSIDE as inside — pushing on those false positives manufactures real churn and
the penetration count never drops.  The muscles are tet meshes, so the robust
test is exact volumetric containment: a vertex ``x`` is inside muscle B iff it
falls inside one of B's tets (barycentric coords all >= 0, sum <= 1).  This holds
for thin / concave / even slightly-inverted tets, with no sign to flip.

Pipeline (everything on the GPU, one host sync per frame for the active mask):

  * a GLOBAL ``wp.Bvh`` over every muscle TET's AABB (tagged by muscle id) is the
    broadphase AND the inside test — ``bvh_query_aabb(x, x)`` returns the
    candidate tets containing ``x``; a barycentric test on the non-self ones
    decides containment.  Rebuilt each resolve pass (positions move).
  * the push TARGET is the nearest point on the containing muscle's boundary
    surface (its own per-muscle ``wp.Mesh``), found with a plain closest-point
    query.  Only the *geometric* closest point is used (face + bary), never the
    query's sign — so the surface search radius can be large enough to reach a
    deep vertex without re-introducing any sign-flip false positive.

  Decoupling the two — exact unbounded inside decision (tet) from geometric
  closest-surface push (mesh) — is what lets the count actually fall: the inside
  test no longer lies, and the push can see deep penetration.

muscle_id comes from the rest geo's per-tet ``muscle_id`` prim attr (199 strings
for humanV2); each muscle's surface is extracted from its own tet subset so the
shared interface between two touching muscles is a boundary for both.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from muscle.util.mesh_io import vectorized_surface_tris as _vectorized_surface_tris
from muscle.util.geo_io import GeoFile

_INF = wp.constant(1.0e30)


# ─── broadphase helpers for the per-muscle active mask (surface refit gating) ──

@wp.kernel
def _reset_aabb_kernel(lo: wp.array(dtype=wp.float32),
                       hi: wp.array(dtype=wp.float32)):
    i = wp.tid()
    lo[i] = _INF
    hi[i] = -_INF


@wp.kernel
def _muscle_aabb_kernel(pos: wp.array(dtype=wp.vec3),
                        vert_muscle: wp.array(dtype=wp.int32),
                        lo: wp.array(dtype=wp.float32),
                        hi: wp.array(dtype=wp.float32)):
    """Per-muscle AABB by atomic min/max of each muscle's vertices (lo/hi are
    flat ``[n_muscle * 3]``)."""
    i = wp.tid()
    m = vert_muscle[i]
    if m < 0:
        return
    p = pos[i]
    b = m * 3
    for c in range(3):
        wp.atomic_min(lo, b + c, p[c])
        wp.atomic_max(hi, b + c, p[c])


@wp.kernel
def _adjacency_kernel(lo: wp.array(dtype=wp.float32),
                      hi: wp.array(dtype=wp.float32),
                      margin: wp.float32, nm: wp.int32,
                      adj: wp.array(dtype=wp.int32),
                      active: wp.array(dtype=wp.int32),
                      n_pairs: wp.array(dtype=wp.int32)):
    """Mark overlapping muscle pairs (i < j) and flag both ``active``.  Used only
    to gate the per-muscle surface refit (refit a muscle only if it overlaps
    another this frame) and to report a pair count; the inside test itself uses
    the global tet BVH, not this matrix."""
    idx = wp.tid()
    i = idx / nm
    j = idx % nm
    if i >= j:
        return
    ov = True
    for c in range(3):
        if lo[i * 3 + c] > hi[j * 3 + c] + margin:
            ov = False
        if lo[j * 3 + c] > hi[i * 3 + c] + margin:
            ov = False
    if ov:
        adj[i * nm + j] = 1
        adj[j * nm + i] = 1
        active[i] = 1
        active[j] = 1
        wp.atomic_add(n_pairs, 0, 1)


# ─── tet BVH (inside test) ─────────────────────────────────────────────────

@wp.kernel
def _tet_aabb_kernel(pos: wp.array(dtype=wp.vec3),
                     tet: wp.array(dtype=wp.vec4i),
                     pad: wp.float32,
                     lo: wp.array(dtype=wp.vec3),
                     hi: wp.array(dtype=wp.vec3)):
    """Per-tet AABB (current positions) for the global inside-test BVH."""
    t = wp.tid()
    idx = tet[t]
    a = pos[idx[0]]
    b = pos[idx[1]]
    c = pos[idx[2]]
    d = pos[idx[3]]
    p = wp.vec3(pad, pad, pad)
    lo[t] = wp.min(wp.min(a, b), wp.min(c, d)) - p
    hi[t] = wp.max(wp.max(a, b), wp.max(c, d)) + p


@wp.func
def _tet_inside_score(x: wp.vec3, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3,
                      p3: wp.vec3) -> wp.float32:
    """Barycentric containment score: ``min(lambda)`` (> 0, larger = deeper) if
    ``x`` is inside the tet, else -1.  Orientation-independent (works for
    slightly-inverted tets); only degenerate tets are rejected."""
    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0
    Dm = wp.mat33(e1[0], e2[0], e3[0],
                  e1[1], e2[1], e3[1],
                  e1[2], e2[2], e3[2])
    det = wp.determinant(Dm)
    if wp.abs(det) < 1.0e-20:
        return -1.0
    lam = wp.inverse(Dm) * (x - p0)
    l1 = lam[0]
    l2 = lam[1]
    l3 = lam[2]
    l0 = 1.0 - l1 - l2 - l3
    if l0 >= 0.0 and l1 >= 0.0 and l2 >= 0.0 and l3 >= 0.0:
        return wp.min(wp.min(l0, l1), wp.min(l2, l3))
    return -1.0


@wp.func
def _inside_target(x: wp.vec3, m: wp.int32, bvh: wp.uint64,
                   tet: wp.array(dtype=wp.vec4i),
                   tet_muscle: wp.array(dtype=wp.int32),
                   pos: wp.array(dtype=wp.vec3)) -> wp.int32:
    """The non-self muscle ``x`` is most deeply inside (barycentric), or -1.

    Multiple tets of the same muscle containing ``x`` is normal (deep vertex);
    the deepest-score wins, so the result is a single target muscle.  When ``x``
    sits inside several DISTINCT muscles the deepest is taken this pass and the
    rest are cleared by later iterations."""
    best = float(0.0)
    best_mb = wp.int32(-1)
    q = wp.bvh_query_aabb(bvh, x, x)
    t = wp.int32(0)
    while wp.bvh_query_next(q, t):
        mb = tet_muscle[t]
        if mb != m:
            idx = tet[t]
            s = _tet_inside_score(x, pos[idx[0]], pos[idx[1]],
                                  pos[idx[2]], pos[idx[3]])
            if s > best:
                best = s
                best_mb = mb
    return best_mb


@wp.kernel
def _push_kernel(pos: wp.array(dtype=wp.vec3),
                 vert_muscle: wp.array(dtype=wp.int32),
                 inv_mass: wp.array(dtype=wp.float32),
                 bvh: wp.uint64,
                 tet: wp.array(dtype=wp.vec4i),
                 tet_muscle: wp.array(dtype=wp.int32),
                 mesh_ids: wp.array(dtype=wp.uint64),
                 surf_dist: wp.float32, offset: wp.float32,
                 push_cap: wp.float32):
    """One thread per muscle vertex: if it is inside another muscle (tet
    barycentric), move it to that muscle's nearest BOUNDARY surface point plus a
    small outward ``offset``.

    The surface point is the plain geometric closest point (face + bary only —
    the query's sign is never read), so ``surf_dist`` can be large enough to
    reach a deep vertex without any pseudonormal sign-flip.  ``push_cap`` clamps
    the per-pass step (0 = unlimited)."""
    i = wp.tid()
    m = vert_muscle[i]
    if m < 0 or inv_mass[i] == 0.0:
        return
    x = pos[i]
    mb = _inside_target(x, m, bvh, tet, tet_muscle, pos)
    if mb < 0:
        return
    q = wp.mesh_query_point_sign_normal(mesh_ids[mb], x, surf_dist)
    if not q.result:
        return
    cp = wp.mesh_eval_position(mesh_ids[mb], q.face, q.u, q.v)
    d = cp - x                                   # x is inside -> points outward
    dl = wp.length(d)
    if dl < 1.0e-9:
        return
    step = d + (d / dl) * offset                 # to surface, then offset beyond
    if push_cap > 0.0:
        mag = wp.length(step)
        if mag > push_cap:
            step = step * (push_cap / mag)
    pos[i] = x + step


@wp.kernel
def _count_inside_tet_kernel(pos: wp.array(dtype=wp.vec3),
                             vert_muscle: wp.array(dtype=wp.int32),
                             bvh: wp.uint64,
                             tet: wp.array(dtype=wp.vec4i),
                             tet_muscle: wp.array(dtype=wp.int32),
                             count: wp.array(dtype=wp.int32)):
    """Penetration METER: count each muscle vertex once if it is inside any other
    muscle (exact barycentric, no sign-flip false positives)."""
    i = wp.tid()
    m = vert_muscle[i]
    if m < 0:
        return
    if _inside_target(pos[i], m, bvh, tet, tet_muscle, pos) >= 0:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def _depth_inside_tet_kernel(pos: wp.array(dtype=wp.vec3),
                             vert_muscle: wp.array(dtype=wp.int32),
                             bvh: wp.uint64,
                             tet: wp.array(dtype=wp.vec4i),
                             tet_muscle: wp.array(dtype=wp.int32),
                             mesh_ids: wp.array(dtype=wp.uint64),
                             surf_dist: wp.float32,
                             depth: wp.array(dtype=wp.float32)):
    """Penetration-DEPTH meter: for each muscle vertex inside another muscle, the
    distance to that muscle's boundary surface (metres, 0 if outside all)."""
    i = wp.tid()
    depth[i] = 0.0
    m = vert_muscle[i]
    if m < 0:
        return
    x = pos[i]
    mb = _inside_target(x, m, bvh, tet, tet_muscle, pos)
    if mb < 0:
        return
    q = wp.mesh_query_point_sign_normal(mesh_ids[mb], x, surf_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh_ids[mb], q.face, q.u, q.v)
        depth[i] = wp.length(x - cp)


class InterMuscleCollider:
    """Resolve muscle-A-vs-muscle-B penetration with an exact tet-barycentric
    inside test and a geometric closest-surface push.

    A global ``wp.Bvh`` over all muscle tets (tagged by muscle id) is both the
    broadphase and the inside test; each muscle's ``wp.Mesh`` (points SHARED with
    ``sim.pos``) supplies the nearest boundary point for the push.  No host sync
    per resolve pass; one per frame to cache the active-muscle mask for refit.
    """

    def __init__(self, sim, rest_path: str, *, max_dist: float = 0.02,
                 contact_offset: float = 1e-4, device=None,
                 flip_winding: bool = True, iters: int = 1, substep_stride: int = 2,
                 push_cap: float = 0.0, surf_dist: float = None,
                 refit_stride: int = 1):
        self.sim = sim
        self.device = device or sim._device
        self.max_dist = float(max_dist)
        self.contact_offset = float(contact_offset)
        self.iters = max(1, int(iters))
        self.push_cap = float(push_cap)
        # Surface-search radius for the push / depth-meter closest-point query.
        # Independent of the inside test (which is the unbounded tet barycentric),
        # so it can be large to reach deep verts — only the geometric closest
        # point is used, never the query sign, so a large radius is safe.
        self.surf_dist = float(surf_dist) if surf_dist is not None \
            else max(0.2, 10.0 * self.max_dist)
        # Run the refit+push only every Nth substep (1 = every substep).  The
        # LAST substep always runs so the frame ends resolved.
        self.substep_stride = max(1, int(substep_stride))
        # Refit the per-muscle surface meshes only every Nth substep (1 = every
        # push). ~199 host-side refit() launches per substep dominate the cost;
        # the surface deforms little between substeps, so a slightly stale closest
        # point is fine for the iterative capped push.
        self.refit_stride = max(1, int(refit_stride))
        self._sub = 0
        self._nsub = int(getattr(sim, "nsubsteps", 1))

        gf = GeoFile(rest_path)
        tets = np.asarray(gf.tet_indices, dtype=np.int32)        # (NT, 4)
        if not gf.has_prim_attr("muscle_id"):
            raise ValueError(f"{rest_path} has no 'muscle_id' prim attr")
        mid = gf.prim_attr("muscle_id")                          # (NT,) int
        self.names = gf.prim_attr_strings("muscle_id")
        self.n_muscle = len(self.names)
        self.nvm = int(np.asarray(gf.positions).shape[0])        # muscle vert count
        nv = sim.pos.shape[0]

        # Per-vertex muscle id (-1 for non-muscle verts, e.g. fascia).  A vertex
        # shared by several muscles' tets gets the last one — interface verts are
        # rare and sit on the boundary either way.
        vert_muscle = np.full(nv, -1, dtype=np.int32)
        for k in range(4):
            vert_muscle[tets[:, k]] = mid
        self._vert_muscle = wp.array(vert_muscle, dtype=wp.int32, device=self.device)

        # Global tet BVH inputs: every muscle tet, tagged by muscle id.  Tet
        # indices reference global muscle verts (< nvm), so pos[idx] is valid.
        self._tet_idx = wp.array(tets, dtype=wp.vec4i, device=self.device)
        self._tet_muscle = wp.array(mid.astype(np.int32), dtype=wp.int32,
                                    device=self.device)
        self._n_tet = int(tets.shape[0])
        self._tet_lo = wp.zeros(self._n_tet, dtype=wp.vec3, device=self.device)
        self._tet_hi = wp.zeros(self._n_tet, dtype=wp.vec3, device=self.device)
        self._tet_bvh = None
        self._target = wp.zeros(self.nvm, dtype=wp.int32, device=self.device)

        # Per-muscle surface meshes (push target).  Each muscle's surface is
        # extracted from its OWN tet subset, so two touching muscles each own the
        # shared interface.  Points SHARED with sim.pos (refit tracks deformation;
        # winding does not matter — only the geometric closest point is used).
        self._meshes = []
        for m in range(self.n_muscle):
            mtets = tets[mid == m]
            if mtets.shape[0] == 0:
                self._meshes.append(None); continue
            surf = _vectorized_surface_tris(mtets).astype(np.int32)   # (k, 3)
            if flip_winding:
                surf = np.ascontiguousarray(surf[:, [0, 2, 1]])
            self._meshes.append(wp.Mesh(
                points=sim.pos,                          # SHARED with the sim
                indices=wp.array(surf.reshape(-1), dtype=wp.int32,
                                 device=self.device)))
        ids = np.array([m.id if m is not None else 0 for m in self._meshes],
                       dtype=np.uint64)
        self._mesh_ids = wp.array(ids, dtype=wp.uint64, device=self.device)

        self.inv_mass = sim.inv_mass
        nm = self.n_muscle
        self._lo = wp.zeros(nm * 3, dtype=wp.float32, device=self.device)
        self._hi = wp.zeros(nm * 3, dtype=wp.float32, device=self.device)
        self._adj = wp.zeros(nm * nm, dtype=wp.int32, device=self.device)
        self._active = wp.zeros(nm, dtype=wp.int32, device=self.device)
        self._n_pairs = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._depth = wp.zeros(self.nvm, dtype=wp.float32, device=self.device)
        self.last_pairs = 0
        self.last_active = 0
        self.refresh()

    def _build_tet_bvh(self):
        """Rebuild the global tet AABB BVH on the current positions (the inside
        test reads it; positions move every pass, so rebuild — a refit keeps the
        stale tree topology and degrades the query)."""
        wp.launch(_tet_aabb_kernel, dim=self._n_tet,
                  inputs=[self.sim.pos, self._tet_idx, wp.float32(0.0),
                          self._tet_lo, self._tet_hi], device=self.device)
        self._tet_bvh = wp.Bvh(self._tet_lo, self._tet_hi)

    def refresh(self):
        """Recompute the per-muscle active mask from the current positions — call
        once per frame.  Only used to gate the per-substep surface refit (refit a
        muscle only if it overlaps another this frame); the inside test uses the
        tet BVH, rebuilt per pass.  One host sync per frame for the 199-int
        active mask."""
        nm = self.n_muscle
        wp.launch(_reset_aabb_kernel, dim=nm * 3, inputs=[self._lo, self._hi],
                  device=self.device)
        wp.launch(_muscle_aabb_kernel, dim=self.sim.pos.shape[0],
                  inputs=[self.sim.pos, self._vert_muscle, self._lo, self._hi],
                  device=self.device)
        self._adj.zero_()
        self._active.zero_()
        self._n_pairs.zero_()
        # Tight margin: the active mask only gates the surface refit, and a muscle
        # can only be a push target if some vertex is INSIDE it (containment ->
        # tight AABB overlap). Using the large surf_dist here marked nearly every
        # muscle active (refit ~all 199 / substep, ~50% of cost). max_dist is a
        # small safety band around true overlap.
        margin = self.max_dist + self.contact_offset
        wp.launch(_adjacency_kernel, dim=nm * nm,
                  inputs=[self._lo, self._hi, wp.float32(margin), wp.int32(nm),
                          self._adj, self._active, self._n_pairs],
                  device=self.device)
        act = self._active.numpy()
        self.last_active = int(act.sum())
        self._active_meshes = [self._meshes[m] for m in range(nm)
                               if act[m] and self._meshes[m] is not None]
        self._sub = 0

    def __call__(self):
        # Run on every Nth substep AND always on the last one, so the frame ends
        # fully resolved.
        cur = self._sub
        last = self._nsub - 1
        run = (cur % self.substep_stride == 0) or (cur == last)
        self._sub += 1
        if not run:
            return
        # Refit gated separately from push: every refit_stride-th run, and always
        # on the last substep so the frame ends pushing against current geometry.
        # ~199 host-side refit() launches/substep dominate; the surface deforms
        # little between substeps so a stale closest point is fine.
        if cur % self.refit_stride == 0 or cur == last:
            for mesh in self._active_meshes:    # active only
                mesh.refit()
        # Build the tet BVH ONCE per call and reuse across iters. The capped push
        # moves verts at most push_cap/pass, so the AABBs go only mildly stale
        # within a call — far cheaper than rebuilding a 468k-tet tree every iter.
        self._build_tet_bvh()
        for _ in range(self.iters):
            wp.launch(_push_kernel, dim=self.nvm,
                      inputs=[self.sim.pos, self._vert_muscle, self.inv_mass,
                              self._tet_bvh.id, self._tet_idx, self._tet_muscle,
                              self._mesh_ids, wp.float32(self.surf_dist),
                              wp.float32(self.contact_offset),
                              wp.float32(self.push_cap)],
                      device=self.device)

    def pair_count(self) -> int:
        """Overlapping muscle pairs from the last ``refresh`` (host sync)."""
        self.last_pairs = int(self._n_pairs.numpy()[0])
        return self.last_pairs

    def inside_count(self) -> int:
        """Muscle vertices inside another muscle (exact barycentric meter).  Each
        vertex counted once.  Refreshes + rebuilds the tet BVH first."""
        self.refresh()
        self._build_tet_bvh()
        self._count.zero_()
        wp.launch(_count_inside_tet_kernel, dim=self.nvm,
                  inputs=[self.sim.pos, self._vert_muscle, self._tet_bvh.id,
                          self._tet_idx, self._tet_muscle, self._count],
                  device=self.device)
        self.pair_count()
        return int(self._count.numpy()[0])

    def inside_depth(self, meter_max_dist: float = None) -> np.ndarray:
        """Per-muscle-vertex inter-muscle penetration depth (metres, 0 if outside
        all neighbours): the distance from an inside vertex to the containing
        muscle's boundary surface.  ``meter_max_dist`` overrides the surface
        search radius (the inside test is the unbounded tet barycentric, so this
        only bounds how deep a surface point can be located)."""
        md = self.surf_dist if meter_max_dist is None else float(meter_max_dist)
        self.refresh()
        for mesh in self._active_meshes:
            mesh.refit()
        self._build_tet_bvh()
        wp.launch(_depth_inside_tet_kernel, dim=self.nvm,
                  inputs=[self.sim.pos, self._vert_muscle, self._tet_bvh.id,
                          self._tet_idx, self._tet_muscle, self._mesh_ids,
                          wp.float32(md), self._depth],
                  device=self.device)
        self.pair_count()
        return self._depth.numpy()
