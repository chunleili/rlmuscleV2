"""Inter-muscle collision via per-muscle closest-point projection.

The simplification over UDCF DCD self-collision: only resolve muscle_A-vs-muscle_B
penetration, NEVER muscle-vs-itself.  That removes the hard part — intra-body
self-collision needs k-ring/connectivity filtering so a vertex is not pushed
against its own neighbouring triangles.  Across two DISTINCT muscles every
triangle of the other body is a valid target, so the resolution is a plain
closest-point projection: treat each muscle as a closed watertight body, query
each candidate vertex against the OTHER muscle's surface with
``mesh_query_point_sign_normal``, and push the inside ones out to the closest
surface point.  (No precomputed signed distance field — the sign comes from the
mesh query — so this lives outside ``collision/sdf``; it is the same per-vertex
push as the SDF-bone collider but with a deformable closest-point target.)

This is also the physically meaningful case: a muscle folding into itself is
internal and invisible, whereas two packed muscles overlapping is the visible
penetration (199 muscles in humanV2).

Everything but the per-muscle BVH refit is on the GPU, with no host sync:

  * broadphase (``refresh``, once per frame): a per-muscle AABB kernel
    (atomic min/max into ``lo``/``hi``) then an N_muscle x N_muscle adjacency
    kernel that marks overlapping muscle pairs.
  * resolve (``__call__``, per substep): one push kernel, one thread per muscle
    vertex, that loops its muscle's overlapping neighbours, queries each
    neighbour's surface and applies the AVERAGE outward push (jacobi — the
    average is computed locally in the single owning thread, so no atomics and
    no race even when a vertex sits inside several muscles).

muscle_id comes from the rest geo's per-tet ``muscle_id`` prim attr (199 strings
for humanV2); each muscle's surface is extracted from its own tet subset so the
shared interface between two touching muscles is a surface for both.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from muscle.coupleOneWayGmg.example import _vectorized_surface_tris
from muscle.util.geo_io import GeoFile

_INF = wp.constant(1.0e30)


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
    """Mark overlapping muscle pairs (i < j) in the dense ``nm x nm`` matrix and
    flag both muscles ``active`` (only active muscles need a refit / push).

    Empty muscles keep ``lo = +inf`` so their overlap test always fails."""
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


@wp.func
def _push_against(mesh: wp.uint64, x: wp.vec3, max_dist: wp.float32,
                  offset: wp.float32):
    """Outward correction that moves ``x`` to ``mesh``'s surface if inside it
    (zero vec + 0 otherwise).  Returns (correction, hit)."""
    q = wp.mesh_query_point_sign_normal(mesh, x, max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        delta = x - cp
        dist = wp.length(delta) * q.sign          # < 0 inside the mesh
        err = dist - offset
        if err < 0.0:
            n = wp.normalize(delta) * q.sign
            return -n * err, 1                     # outward push (err < 0)
    return wp.vec3(0.0, 0.0, 0.0), 0


@wp.kernel
def _push_adj_kernel(pos: wp.array(dtype=wp.vec3),
                     vert_muscle: wp.array(dtype=wp.int32),
                     inv_mass: wp.array(dtype=wp.float32),
                     mesh_ids: wp.array(dtype=wp.uint64),
                     adj: wp.array(dtype=wp.int32), nm: wp.int32,
                     max_dist: wp.float32, offset: wp.float32,
                     push_cap: wp.float32):
    """One thread per muscle vertex: average the push out of every overlapping
    neighbour muscle it is inside, and apply it (clamped to ``push_cap``).

    The owning thread accumulates locally over its neighbour loop, so the jacobi
    average needs no atomics and cannot race even when the vertex sits inside
    several muscles.  ``x`` is read once, so within a launch the push is jacobi.

    ``push_cap`` limits the per-pass displacement (0 = unlimited).  A deep vertex
    is eased out a little each pass and re-queried, following the real exit path
    instead of teleporting straight to the nearest surface — which, in densely
    packed muscle, overshoots THROUGH the neighbour behind it and deepens the
    penetration.  Use several ``iters`` (and a small cap) to clear deep overlap."""
    i = wp.tid()
    m = vert_muscle[i]
    if m < 0 or inv_mass[i] == 0.0:
        return
    x = pos[i]
    acc = wp.vec3(0.0, 0.0, 0.0)
    c = int(0)
    row = m * nm
    for j in range(nm):
        if adj[row + j] != 0:
            d, hit = _push_against(mesh_ids[j], x, max_dist, offset)
            acc += d
            c += hit
    if c > 0:
        step = acc / wp.float32(c)
        if push_cap > 0.0:
            mag = wp.length(step)
            if mag > push_cap:
                step = step * (push_cap / mag)
        pos[i] = x + step


@wp.kernel
def _count_inside_adj_kernel(pos: wp.array(dtype=wp.vec3),
                             vert_muscle: wp.array(dtype=wp.int32),
                             mesh_ids: wp.array(dtype=wp.uint64),
                             adj: wp.array(dtype=wp.int32), nm: wp.int32,
                             max_dist: wp.float32,
                             count: wp.array(dtype=wp.int32)):
    """Penetration METER: count this vertex once per neighbour muscle it is
    inside (matches the directional inside-vertex tally)."""
    i = wp.tid()
    m = vert_muscle[i]
    if m < 0:
        return
    x = pos[i]
    row = m * nm
    for j in range(nm):
        if adj[row + j] != 0:
            q = wp.mesh_query_point_sign_normal(mesh_ids[j], x, max_dist)
            if q.result:
                cp = wp.mesh_eval_position(mesh_ids[j], q.face, q.u, q.v)
                if wp.length(x - cp) * q.sign < 0.0:
                    wp.atomic_add(count, 0, 1)


@wp.kernel
def _depth_inside_adj_kernel(pos: wp.array(dtype=wp.vec3),
                             vert_muscle: wp.array(dtype=wp.int32),
                             mesh_ids: wp.array(dtype=wp.uint64),
                             adj: wp.array(dtype=wp.int32), nm: wp.int32,
                             max_dist: wp.float32,
                             depth: wp.array(dtype=wp.float32)):
    """Penetration-DEPTH meter: per vertex, the deepest it sits inside any
    overlapping neighbour muscle (0 if outside all of them), metres."""
    i = wp.tid()
    depth[i] = 0.0
    m = vert_muscle[i]
    if m < 0:
        return
    x = pos[i]
    row = m * nm
    d = float(0.0)
    for j in range(nm):
        if adj[row + j] != 0:
            q = wp.mesh_query_point_sign_normal(mesh_ids[j], x, max_dist)
            if q.result:
                cp = wp.mesh_eval_position(mesh_ids[j], q.face, q.u, q.v)
                sd = wp.length(x - cp) * q.sign      # < 0 inside
                if -sd > d:
                    d = -sd
    depth[i] = d


class InterMuscleCollider:
    """Per-muscle closed-surface colliders that resolve only A-vs-B penetration.

    Each muscle's ``wp.Mesh`` shares ``sim.pos`` as its points array (no copy),
    so a single ``refit()`` per muscle per call tracks the deforming muscle; only
    the per-muscle BVHs are extra memory.  The broadphase and resolve are GPU
    kernels driven by a per-vertex ``vert_muscle`` map and a dense muscle
    adjacency matrix — no host sync per frame.
    """

    def __init__(self, sim, rest_path: str, *, max_dist: float = 0.02,
                 contact_offset: float = 1e-4, device=None,
                 flip_winding: bool = True, iters: int = 1, substep_stride: int = 2,
                 push_cap: float = 0.0):
        self.sim = sim
        self.device = device or sim._device
        self.max_dist = float(max_dist)
        self.contact_offset = float(contact_offset)
        self.iters = max(1, int(iters))
        self.push_cap = float(push_cap)
        # Run the refit+push only every Nth substep (1 = every substep).  The
        # ~10 per-frame refits are the cost; raising this trades resolution depth
        # for speed.  The LAST substep always runs so the frame ends resolved.
        self.substep_stride = max(1, int(substep_stride))
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
        # shared by several muscles' tets gets the last one — those interface
        # verts are rare and sit on the boundary either way.
        vert_muscle = np.full(nv, -1, dtype=np.int32)
        for k in range(4):
            vert_muscle[tets[:, k]] = mid
        self._vert_muscle = wp.array(vert_muscle, dtype=wp.int32, device=self.device)

        # Per-muscle surface meshes.  Each muscle's surface is extracted from its
        # OWN tet subset, so two touching muscles each own the shared interface
        # (both treat it as a closed boundary).
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

        # mesh.id per muscle, on the GPU (ids are stable across refit; 0 for an
        # empty muscle, which the adjacency matrix never references).
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

    def refresh(self):
        """Recompute the muscle adjacency from the current positions — call once
        per frame.  The overlapping-pair set is reused across the substep pushes;
        the surface itself is still refit every substep (intra-frame muscle
        deformation is large, so a frame-start surface is too stale to push
        against).  Reads back the 199-int ``active`` mask once so the per-substep
        refit can skip idle muscles (never queried this frame)."""
        nm = self.n_muscle
        wp.launch(_reset_aabb_kernel, dim=nm * 3, inputs=[self._lo, self._hi],
                  device=self.device)
        wp.launch(_muscle_aabb_kernel, dim=self.sim.pos.shape[0],
                  inputs=[self.sim.pos, self._vert_muscle, self._lo, self._hi],
                  device=self.device)
        self._adj.zero_()
        self._active.zero_()
        self._n_pairs.zero_()
        margin = self.max_dist + self.contact_offset
        wp.launch(_adjacency_kernel, dim=nm * nm,
                  inputs=[self._lo, self._hi, wp.float32(margin), wp.int32(nm),
                          self._adj, self._active, self._n_pairs],
                  device=self.device)
        # Cache the active meshes (one host sync/frame) so the per-substep refit
        # touches only muscles that overlap something this frame.
        act = self._active.numpy()
        self.last_active = int(act.sum())
        self._active_meshes = [self._meshes[m] for m in range(nm)
                               if act[m] and self._meshes[m] is not None]
        self._sub = 0

    def __call__(self):
        # Run on every Nth substep AND always on the last one, so the frame ends
        # fully resolved (skipping the final substep leaves the muscle wherever
        # the solver moved it — measured as no relief).
        run = (self._sub % self.substep_stride == 0) or (self._sub == self._nsub - 1)
        self._sub += 1
        if not run:
            return
        for mesh in self._active_meshes:    # per-substep refit, active only
            mesh.refit()
        for _ in range(self.iters):
            wp.launch(_push_adj_kernel, dim=self.nvm,
                      inputs=[self.sim.pos, self._vert_muscle, self.inv_mass,
                              self._mesh_ids, self._adj, wp.int32(self.n_muscle),
                              wp.float32(self.max_dist),
                              wp.float32(self.contact_offset),
                              wp.float32(self.push_cap)],
                      device=self.device)

    def pair_count(self) -> int:
        """Overlapping muscle pairs from the last ``refresh`` (host sync)."""
        self.last_pairs = int(self._n_pairs.numpy()[0])
        return self.last_pairs

    def inside_count(self) -> int:
        """Total inter-muscle inside-vertices (a penetration METER; counts each
        direction independently).  Refreshes the adjacency + refits the active
        muscles first; the meter only queries active muscles' surfaces."""
        self.refresh()
        for mesh in self._active_meshes:
            mesh.refit()
        self._count.zero_()
        wp.launch(_count_inside_adj_kernel, dim=self.nvm,
                  inputs=[self.sim.pos, self._vert_muscle, self._mesh_ids,
                          self._adj, wp.int32(self.n_muscle),
                          wp.float32(self.max_dist), self._count],
                  device=self.device)
        self.pair_count()
        return int(self._count.numpy()[0])

    def inside_depth(self, meter_max_dist: float = None) -> np.ndarray:
        """Per-muscle-vertex deepest inter-muscle penetration depth (metres, 0 if
        outside all neighbours).  ``meter_max_dist`` overrides the search radius
        for MEASUREMENT only (use a large value so deep penetration is not capped
        at the resolver's ``max_dist``).  Refreshes + refits active muscles."""
        md = self.max_dist if meter_max_dist is None else float(meter_max_dist)
        self.refresh()
        for mesh in self._active_meshes:
            mesh.refit()
        wp.launch(_depth_inside_adj_kernel, dim=self.nvm,
                  inputs=[self.sim.pos, self._vert_muscle, self._mesh_ids,
                          self._adj, wp.int32(self.n_muscle),
                          wp.float32(md), self._depth],
                  device=self.device)
        self.pair_count()
        return self._depth.numpy()
