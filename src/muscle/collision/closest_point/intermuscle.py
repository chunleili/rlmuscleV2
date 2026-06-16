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

Pipeline per call:
  1. refit each muscle's ``wp.Mesh`` (points shared with ``sim.pos``, BVH rebuilt)
  2. broadphase: per-muscle AABB, emit overlapping muscle pairs (sparse)
  3. for each pair (i, j): push verts of i inside j out to j's surface, and
     verts of j inside i out to i's surface

muscle_id comes from the rest geo's per-tet ``muscle_id`` prim attr (199 strings
for humanV2); each muscle's surface is extracted from its own tet subset so the
shared interface between two touching muscles is a surface for both.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from muscle.coupleOneWayGmg.example import _vectorized_surface_tris
from muscle.util.geo_io import GeoFile


@wp.kernel
def _accum_worklist_kernel(work_vidx: wp.array(dtype=wp.int32),
                           work_mesh: wp.array(dtype=wp.uint64),
                           pos: wp.array(dtype=wp.vec3),
                           inv_mass: wp.array(dtype=wp.float32),
                           max_dist: wp.float32,
                           contact_offset: wp.float32,
                           dP: wp.array(dtype=wp.vec3),
                           cnt: wp.array(dtype=wp.int32)):
    """Accumulate the inter-muscle push for one (vertex, target-mesh) work item.

    Broadphase emits one entry per (candidate vertex, other-muscle mesh) so the
    whole resolve is a SINGLE launch instead of one per muscle pair (~3000 pairs
    in humanV2 — per-pair launches were launch-overhead bound).  Corrections are
    atomic-accumulated (not written in place) so a vertex embedded in several
    muscles gets the AVERAGE of its per-target pushes — jacobi, race-free, so K
    passes converge instead of fighting (in-place writes diverged at K>1).
    """
    k = wp.tid()
    i = work_vidx[k]
    if inv_mass[i] == 0.0:
        return
    x = pos[i]
    query = wp.mesh_query_point_sign_normal(work_mesh[k], x, max_dist)
    if query.result:
        cp = wp.mesh_eval_position(work_mesh[k], query.face, query.u, query.v)
        delta = x - cp
        dist = wp.length(delta) * query.sign      # < 0 inside the mesh
        err = dist - contact_offset
        if err < 0.0:
            n = wp.normalize(delta) * query.sign
            wp.atomic_add(dP, i, -n * err)        # outward push (err < 0)
            wp.atomic_add(cnt, i, 1)


@wp.kernel
def _apply_accum_kernel(pos: wp.array(dtype=wp.vec3),
                        dP: wp.array(dtype=wp.vec3),
                        cnt: wp.array(dtype=wp.int32)):
    """Apply the averaged correction and clear the accumulators (jacobi)."""
    i = wp.tid()
    c = cnt[i]
    if c > 0:
        pos[i] = pos[i] + dP[i] / wp.float32(c)
        dP[i] = wp.vec3(0.0, 0.0, 0.0)
        cnt[i] = 0


@wp.kernel
def _count_inside_indexed_kernel(vidx: wp.array(dtype=wp.int32),
                                 pos: wp.array(dtype=wp.vec3),
                                 mesh: wp.uint64,
                                 max_dist: wp.float32,
                                 count: wp.array(dtype=wp.int32)):
    k = wp.tid()
    i = vidx[k]
    q = wp.mesh_query_point_sign_normal(mesh, pos[i], max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        if wp.length(pos[i] - cp) * q.sign < 0.0:
            wp.atomic_add(count, 0, 1)


class InterMuscleCollider:
    """Per-muscle closed-surface colliders that resolve only A-vs-B penetration.

    Each muscle's ``wp.Mesh`` shares ``sim.pos`` as its points array (no copy),
    so a single ``refit()`` per muscle per call tracks the deforming muscle; only
    the per-muscle BVHs are extra memory.
    """

    def __init__(self, sim, rest_path: str, *, max_dist: float = 0.02,
                 contact_offset: float = 1e-4, device=None,
                 flip_winding: bool = True, iters: int = 1):
        self.sim = sim
        self.device = device or sim._device
        self.max_dist = float(max_dist)
        self.contact_offset = float(contact_offset)
        self.iters = max(1, int(iters))

        gf = GeoFile(rest_path)
        tets = np.asarray(gf.tet_indices, dtype=np.int32)        # (NT, 4)
        if not gf.has_prim_attr("muscle_id"):
            raise ValueError(f"{rest_path} has no 'muscle_id' prim attr")
        mid = gf.prim_attr("muscle_id")                          # (NT,) int
        self.names = gf.prim_attr_strings("muscle_id")
        self.n_muscle = len(self.names)
        self.nvm = int(np.asarray(gf.positions).shape[0])        # muscle vert count

        # Per-muscle surface meshes + vertex index lists.  Each muscle's surface
        # is extracted from its OWN tet subset, so two touching muscles each own
        # the shared interface (both treat it as a closed boundary).
        self._meshes = []
        self._vidx = []        # wp.array(int32) of this muscle's vertices
        self._vidx_np = []     # host copy for the AABB broadphase
        for m in range(self.n_muscle):
            mtets = tets[mid == m]
            if mtets.shape[0] == 0:
                self._meshes.append(None); self._vidx.append(None)
                self._vidx_np.append(np.empty(0, np.int32)); continue
            surf = _vectorized_surface_tris(mtets).astype(np.int32)   # (k, 3)
            if flip_winding:
                surf = np.ascontiguousarray(surf[:, [0, 2, 1]])
            mesh = wp.Mesh(points=sim.pos,                  # SHARED with the sim
                           indices=wp.array(surf.reshape(-1), dtype=wp.int32,
                                            device=self.device))
            self._meshes.append(mesh)
            verts = np.unique(mtets.reshape(-1)).astype(np.int32)
            self._vidx.append(wp.array(verts, dtype=wp.int32, device=self.device))
            self._vidx_np.append(verts)

        # Per-muscle mesh.id, captured once (ids are stable across refit).
        self._mesh_ids = np.array(
            [m.id if m is not None else 0 for m in self._meshes], dtype=np.uint64)

        self.inv_mass = sim.inv_mass
        nv = sim.pos.shape[0]
        self._dP = wp.zeros(nv, dtype=wp.vec3, device=self.device)
        self._cnt = wp.zeros(nv, dtype=wp.int32, device=self.device)
        self._count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.last_pairs = 0
        # Fused work list (vertex, target-mesh), rebuilt once per frame.
        self._work_vidx = None
        self._work_mesh = None
        self.refresh()

    def _broadphase(self, pos_np: np.ndarray, margin: float):
        """Per-muscle AABB overlap → list of (i, j) muscle pairs (i < j)."""
        lo = np.full((self.n_muscle, 3), np.inf, np.float32)
        hi = np.full((self.n_muscle, 3), -np.inf, np.float32)
        for m, v in enumerate(self._vidx_np):
            if v.size:
                lo[m] = pos_np[v].min(0); hi[m] = pos_np[v].max(0)
        lo -= margin; hi += margin
        pairs = []
        for i in range(self.n_muscle):
            if not np.isfinite(lo[i]).all():
                continue
            # vectorised overlap of box i against all j > i
            j = np.arange(i + 1, self.n_muscle)
            ov = (np.all(lo[i] <= hi[j], axis=1) & np.all(hi[i] >= lo[j], axis=1)
                  & np.isfinite(lo[j]).all(axis=1))
            for jj in j[ov]:
                pairs.append((i, int(jj)))
        return pairs

    def refresh(self):
        """Rebuild the broadphase work list — call once per frame.

        Muscles deform little within a frame's substeps, so the overlapping-pair
        set and the candidate verts are reused across the substep pushes (the SDF
        query itself still runs against the refit surface every substep).  One
        host sync (``pos.numpy()``) per frame instead of per substep.
        """
        pos_np = self.sim.pos.numpy()
        pairs = self._broadphase(pos_np, self.max_dist + self.contact_offset)
        self.last_pairs = len(pairs)
        vparts, mparts = [], []
        for i, j in pairs:
            vparts.append(self._vidx_np[i]); mparts.append(np.full(
                self._vidx_np[i].size, self._mesh_ids[j], dtype=np.uint64))
            vparts.append(self._vidx_np[j]); mparts.append(np.full(
                self._vidx_np[j].size, self._mesh_ids[i], dtype=np.uint64))
        if vparts:
            self._work_vidx = wp.array(np.concatenate(vparts), dtype=wp.int32,
                                       device=self.device)
            self._work_mesh = wp.array(np.concatenate(mparts), dtype=wp.uint64,
                                       device=self.device)
        else:
            self._work_vidx = self._work_mesh = None

    def __call__(self):
        for mesh in self._meshes:
            if mesh is not None:
                mesh.refit()
        if self._work_vidx is not None:
            # K jacobi passes: accumulate averaged per-target pushes, then apply.
            # A vertex inside several muscles converges to a compromise position
            # over the passes (in-place gauss-seidel raced and diverged at K>1).
            # Work list reused; only the SDF queries re-run against the new pos.
            for _ in range(self.iters):
                wp.launch(_accum_worklist_kernel, dim=self._work_vidx.shape[0],
                          inputs=[self._work_vidx, self._work_mesh, self.sim.pos,
                                  self.inv_mass, wp.float32(self.max_dist),
                                  wp.float32(self.contact_offset),
                                  self._dP, self._cnt], device=self.device)
                wp.launch(_apply_accum_kernel, dim=self._dP.shape[0],
                          inputs=[self.sim.pos, self._dP, self._cnt],
                          device=self.device)

    def inside_count(self) -> int:
        """Total inter-muscle inside-vertices over all overlapping pairs (a
        penetration METER; counts each direction independently)."""
        for mesh in self._meshes:
            if mesh is not None:
                mesh.refit()
        pos_np = self.sim.pos.numpy()
        pairs = self._broadphase(pos_np, self.max_dist + self.contact_offset)
        self._count.zero_()
        for i, j in pairs:
            for vidx, mesh in ((self._vidx[i], self._meshes[j]),
                               (self._vidx[j], self._meshes[i])):
                wp.launch(_count_inside_indexed_kernel, dim=vidx.shape[0],
                          inputs=[vidx, self.sim.pos, mesh.id,
                                  wp.float32(self.max_dist), self._count],
                          device=self.device)
        return int(self._count.numpy()[0])
