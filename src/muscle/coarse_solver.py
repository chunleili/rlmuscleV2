"""GMG coarse-grid solver for muscle simulation.

Loads a coarse tet mesh from a Houdini .geo file, computes barycentric mappings
to/from the fine mesh, and runs a few elasticity (volume + tetarap) iterations
on the coarse mesh each substep. Implements realtimeMuscle's GMG cycle:

    restrict   coarse.pos += Σ_fine_in_coarse_cage barycentric · Δfine
    coarse_solve   K iters of XPBD volume + tetarap on coarse
    prolong   fine.pos = Σ_coarse_cage barycentric · coarse.pos

Reference: realtimeMuscle/engine/arap_hpbd.py + compute_R_acc.py.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.spatial
import warp as wp

from muscle.constraints import TETARAP, TETVOLUME
from muscle.geo import Geo
from muscle.muscle_warp import (
    Constraint,
    MuscleSim,
    apply_dP_kernel,
    clear_dP_kernel,
    solve_tetarap_kernel,
    solve_tetvolume_kernel,
)


# ---------------------------------------------------------------------------
# Restrict / prolong kernels
# ---------------------------------------------------------------------------

@wp.kernel
def restrict_kernel(
    coarse_pos: wp.array(dtype=wp.vec3),
    coarse_in_fine_idx: wp.array(dtype=wp.int32),
    coarse_in_fine_uvw: wp.array(dtype=wp.vec3),
    fine_tet_indices: wp.array(dtype=wp.vec4i),
    fine_pos: wp.array(dtype=wp.vec3),
    fine_pprev: wp.array(dtype=wp.vec3),
    fine_stopped: wp.array(dtype=wp.int32),
):
    """coarse.pos[i] += interp_in_fine_cage(fine.pos - fine.pprev).

    Stopped (kinematic) fine verts contribute zero delta — they don't
    represent free-body motion and would otherwise inject pintotarget's
    discontinuous bone-chasing jumps into coarse, destabilizing GMG.
    """
    i = wp.tid()
    fine_t = coarse_in_fine_idx[i]
    if fine_t < 0:
        return
    tet = fine_tet_indices[fine_t]
    uvw = coarse_in_fine_uvw[i]
    u = uvw[0]
    v = uvw[1]
    w = uvw[2]
    s0 = fine_stopped[tet[0]]
    s1 = fine_stopped[tet[1]]
    s2 = fine_stopped[tet[2]]
    s3 = fine_stopped[tet[3]]
    dp0 = wp.where(s0 != 0, wp.vec3(0.0, 0.0, 0.0),
                   fine_pos[tet[0]] - fine_pprev[tet[0]])
    dp1 = wp.where(s1 != 0, wp.vec3(0.0, 0.0, 0.0),
                   fine_pos[tet[1]] - fine_pprev[tet[1]])
    dp2 = wp.where(s2 != 0, wp.vec3(0.0, 0.0, 0.0),
                   fine_pos[tet[2]] - fine_pprev[tet[2]])
    dp3 = wp.where(s3 != 0, wp.vec3(0.0, 0.0, 0.0),
                   fine_pos[tet[3]] - fine_pprev[tet[3]])
    delta = (1.0 - u - v - w) * dp0 + u * dp1 + v * dp2 + w * dp3
    coarse_pos[i] = coarse_pos[i] + delta


@wp.kernel
def prolong_kernel(
    fine_pos: wp.array(dtype=wp.vec3),
    fine_in_coarse_idx: wp.array(dtype=wp.int32),
    fine_in_coarse_uvw: wp.array(dtype=wp.vec3),
    coarse_tet_indices: wp.array(dtype=wp.vec4i),
    coarse_pos: wp.array(dtype=wp.vec3),
    fine_stopped: wp.array(dtype=wp.int32),
    blend: float,
):
    """fine.pos[i] = blend * interp_in_coarse_cage(coarse.pos)
                   + (1-blend) * fine.pos[i].

    blend=1 fully overwrites (matches realtimeMuscle default; clobbers
    high-frequency fine modes each substep).  blend<1 nudges fine toward
    coarse interp without erasing local detail.  Stopped (kinematic)
    verts are skipped entirely — they're driven externally (e.g. by
    pintotarget) and prolonging them would clobber the kinematic chain.
    """
    i = wp.tid()
    if fine_stopped[i] != 0:
        return
    coarse_t = fine_in_coarse_idx[i]
    if coarse_t < 0:
        return
    tet = coarse_tet_indices[coarse_t]
    uvw = fine_in_coarse_uvw[i]
    u = uvw[0]
    v = uvw[1]
    w = uvw[2]
    interp = (
        (1.0 - u - v - w) * coarse_pos[tet[0]]
        + u * coarse_pos[tet[1]]
        + v * coarse_pos[tet[2]]
        + w * coarse_pos[tet[3]]
    )
    fine_pos[i] = blend * interp + (1.0 - blend) * fine_pos[i]


@wp.kernel
def copy_pprev_kernel(
    pprev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    pprev[i] = pos[i]


# ---------------------------------------------------------------------------
# Barycentric mapping (one-time precompute, host side)
# ---------------------------------------------------------------------------

def compute_barycentric_mapping(query_pos, cage_pos, cage_tets, k=20,
                                strict_inside=False):
    """For each query point, find which cage tet contains it + (u,v,w) bary.

    Bary at cage_tet[0] = 1-u-v-w; at [1],[2],[3] = u,v,w.

    strict_inside=False (default): out-of-cage queries fall back to the
    nearest tet (uvw may exceed [0,1] — extrapolation). Suitable for
    fine→coarse where the coarse cage covers everything.

    strict_inside=True: out-of-cage queries get idx=-1 and uvw=0 — caller
    must skip these. Required for coarse→fine when coarse covers gaps that
    fine doesn't (e.g. body envelope vs muscle volumes); otherwise small
    fine deltas extrapolate to huge coarse drift.
    """
    query_pos = np.ascontiguousarray(query_pos, dtype=np.float64)
    cage_pos = np.ascontiguousarray(cage_pos, dtype=np.float64)
    cage_tets = np.ascontiguousarray(cage_tets, dtype=np.int32)
    n_q = query_pos.shape[0]
    n_t = cage_tets.shape[0]

    centroids = cage_pos[cage_tets].mean(axis=1)
    tree = scipy.spatial.cKDTree(centroids)
    k_use = min(k, n_t)
    _, neighbor_tets = tree.query(query_pos, k=k_use)
    if k_use == 1:
        neighbor_tets = neighbor_tets[:, None]

    which_tet = np.full(n_q, -1, dtype=np.int32)
    bary = np.zeros((n_q, 3), dtype=np.float64)

    for j in range(k_use):
        unsolved_mask = which_tet < 0
        if not unsolved_mask.any():
            break
        cand = neighbor_tets[:, j]
        tets = cage_tets[cand]
        p_tet = cage_pos[tets]
        p0 = p_tet[:, 0]
        edge1 = p_tet[:, 1] - p0
        edge2 = p_tet[:, 2] - p0
        edge3 = p_tet[:, 3] - p0
        A = np.stack([edge1, edge2, edge3], axis=2)  # (n_q, 3, 3)
        rhs = (query_pos - p0)[..., None]            # (n_q, 3, 1)
        try:
            x = np.linalg.solve(A, rhs)[..., 0]
        except np.linalg.LinAlgError:
            x = np.full((n_q, 3), np.nan)
            for r in np.where(unsolved_mask)[0]:
                try:
                    x[r] = np.linalg.solve(A[r], rhs[r, :, 0])
                except np.linalg.LinAlgError:
                    pass
        sums = x.sum(axis=1)
        eps = 1e-6
        inside = (
            np.isfinite(x).all(axis=1)
            & (x[:, 0] >= -eps)
            & (x[:, 1] >= -eps)
            & (x[:, 2] >= -eps)
            & (sums <= 1.0 + eps)
        )
        new_match = inside & unsolved_mask
        if new_match.any():
            which_tet[new_match] = cand[new_match]
            bary[new_match] = np.clip(x[new_match], 0.0, 1.0)

    leftover = which_tet < 0
    if leftover.any() and not strict_inside:
        cand = neighbor_tets[:, 0]
        tets = cage_tets[cand]
        p_tet = cage_pos[tets]
        p0 = p_tet[:, 0]
        A = np.stack([p_tet[:, 1] - p0, p_tet[:, 2] - p0, p_tet[:, 3] - p0], axis=2)
        rhs = (query_pos - p0)[..., None]
        x = np.full((n_q, 3), 0.25, dtype=np.float64)
        for r in np.where(leftover)[0]:
            try:
                x[r] = np.linalg.solve(A[r], rhs[r, :, 0])
            except np.linalg.LinAlgError:
                pass
        which_tet[leftover] = cand[leftover]
        bary[leftover] = np.clip(x[leftover], -0.5, 1.5)
    # strict_inside=True leaves leftover with idx=-1, uvw=0 — caller skips them.

    return which_tet, bary.astype(np.float32)


# ---------------------------------------------------------------------------
# CoarseSolver
# ---------------------------------------------------------------------------

class CoarseSolver:
    """Coarse elasticity helper for GMG.

    Owns its own Warp state (pos/pprev/dP/etc.) plus barycentric maps to a
    fixed fine mesh. Each substep:
        1) restrict(fine.pos, fine.pprev) — push fine displacement into coarse
        2) solve(n_iters) — XPBD volume + tetarap
        3) prolong(fine.pos) — overwrite fine.pos by coarse barycentric interp
    """

    def __init__(
        self,
        coarse_geo_path: str,
        fine_pos_np: np.ndarray,
        fine_tet_np: np.ndarray,
        fine_tet_indices_field: wp.array,
        dt: float,
        density: float = 1000.0,
        volume_stiffness: float = 1e10,
        tetarap_stiffness: float = 1e10,
        dampingratio: float = 0.1,
    ):
        # 1. Load coarse mesh from .geo
        g = Geo(coarse_geo_path)
        self.pos0_np = np.asarray(g.positions, dtype=np.float32)
        self.tet_np = np.asarray(g.vert, dtype=np.int32).reshape(-1, 4)
        self.n_verts = self.pos0_np.shape[0]
        self.n_tets = self.tet_np.shape[0]
        print(f"[CoarseSolver] coarse mesh: {self.n_verts} verts, {self.n_tets} tets")

        self.dt = float(dt)
        self.density = float(density)
        self._fine_tet_indices_field = fine_tet_indices_field

        # 2. Compute mappings
        t0 = time.perf_counter()
        self.fine_in_coarse_idx_np, self.fine_in_coarse_uvw_np = compute_barycentric_mapping(
            fine_pos_np, self.pos0_np, self.tet_np, k=20)
        n_unmatched_f = int((self.fine_in_coarse_idx_np < 0).sum())
        print(f"[CoarseSolver] fine→coarse map: {len(fine_pos_np)} verts, "
              f"unmatched={n_unmatched_f}, {time.perf_counter()-t0:.2f}s")

        t0 = time.perf_counter()
        self.coarse_in_fine_idx_np, self.coarse_in_fine_uvw_np = compute_barycentric_mapping(
            self.pos0_np, fine_pos_np, fine_tet_np, k=30, strict_inside=True)
        n_unmatched_c = int((self.coarse_in_fine_idx_np < 0).sum())
        print(f"[CoarseSolver] coarse→fine map (strict): {self.n_verts} verts, "
              f"outside-fine-mesh={n_unmatched_c} (skip restrict for these), "
              f"{time.perf_counter()-t0:.2f}s")

        # 3. Rest matrices, volumes, mass
        self._compute_rest_data()

        # 4. Allocate Warp fields
        self._allocate_warp_fields()

        # 5. Build coarse Constraint AOS (volume + tetarap)
        self._build_constraints(volume_stiffness, tetarap_stiffness, dampingratio)

    # ---- precompute -------------------------------------------------------

    def _compute_rest_data(self):
        tet_pos = self.pos0_np[self.tet_np]
        cols = tet_pos[:, :3, :] - tet_pos[:, 3:4, :]
        M = np.transpose(cols, (0, 2, 1))
        dets = np.linalg.det(M)
        self.volumes = dets / 6.0
        valid = np.abs(dets) > 1e-30
        self.rest_matrices = np.zeros_like(M)
        if valid.any():
            self.rest_matrices[valid] = np.linalg.inv(M[valid])
        self.valid_tets = valid

        mass = np.zeros(self.n_verts, dtype=np.float64)
        avg = self.density * np.abs(self.volumes) / 4.0
        for k in range(4):
            np.add.at(mass, self.tet_np[:, k], avg)
        mass = np.maximum(mass, 1e-6)
        self.mass_np = mass.astype(np.float32)

    def _allocate_warp_fields(self):
        self.pos = wp.from_numpy(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.pprev = wp.from_numpy(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.dP = wp.zeros(self.n_verts, dtype=wp.vec3)
        self.dPw = wp.zeros(self.n_verts, dtype=wp.float32)
        self.mass = wp.from_numpy(self.mass_np, dtype=wp.float32)
        self.stopped = wp.zeros(self.n_verts, dtype=wp.int32)
        self.tet_indices = wp.from_numpy(self.tet_np.astype(np.int32), dtype=wp.vec4i)
        self.rest_volume = wp.from_numpy(np.abs(self.volumes).astype(np.float32),
                                         dtype=wp.float32)
        self.rest_matrix = wp.from_numpy(self.rest_matrices.astype(np.float32),
                                         dtype=wp.mat33)

        self.fine_in_coarse_idx = wp.from_numpy(self.fine_in_coarse_idx_np, dtype=wp.int32)
        self.fine_in_coarse_uvw = wp.from_numpy(self.fine_in_coarse_uvw_np, dtype=wp.vec3)
        self.coarse_in_fine_idx = wp.from_numpy(self.coarse_in_fine_idx_np, dtype=wp.int32)
        self.coarse_in_fine_uvw = wp.from_numpy(self.coarse_in_fine_uvw_np, dtype=wp.vec3)

    def _build_constraints(self, vol_k, arap_k, damping):
        """Build a Constraint AOS containing volume + tetarap entries."""
        valid = self.valid_tets
        valid_idx = np.where(valid)[0]
        n_valid = int(valid_idx.size)
        n_cons = 2 * n_valid
        if n_cons == 0:
            self.cons = wp.zeros(0, dtype=Constraint)
            self.cons_ranges = {}
            self.n_cons = 0
            return

        cons_np = np.zeros(n_cons, dtype=Constraint.numpy_dtype())
        rng_volume = (0, n_valid)
        rng_arap = (n_valid, n_valid)

        # Volume block
        cons_np["type"][:n_valid] = TETVOLUME
        cons_np["cidx"][:n_valid] = np.arange(n_valid, dtype=np.int32)
        cons_np["pts"][:n_valid] = self.tet_np[valid_idx]
        cons_np["stiffness"][:n_valid] = vol_k
        cons_np["dampingratio"][:n_valid] = damping
        cons_np["tetid"][:n_valid] = valid_idx
        cons_np["restlength"][:n_valid] = self.volumes[valid_idx].astype(np.float32)
        cons_np["compressionstiffness"][:n_valid] = -1.0

        # Tetarap block
        cons_np["type"][n_valid:] = TETARAP
        cons_np["cidx"][n_valid:] = np.arange(n_valid, 2 * n_valid, dtype=np.int32)
        cons_np["pts"][n_valid:] = self.tet_np[valid_idx]
        cons_np["stiffness"][n_valid:] = arap_k
        cons_np["dampingratio"][n_valid:] = damping
        cons_np["tetid"][n_valid:] = valid_idx
        cons_np["restlength"][n_valid:] = np.abs(self.volumes[valid_idx]).astype(np.float32)
        cons_np["restvector"][n_valid:, 3] = 1.0
        cons_np["compressionstiffness"][n_valid:] = -1.0

        self.cons = wp.array(cons_np, dtype=Constraint)
        self.cons_ranges = {TETVOLUME: rng_volume, TETARAP: rng_arap}
        self.n_cons = n_cons
        print(f"[CoarseSolver] built {n_cons} coarse constraints "
              f"(volume {n_valid} + tetarap {n_valid})")

    # ---- step interface ---------------------------------------------------

    def restrict(self, fine_pos_field: wp.array, fine_pprev_field: wp.array,
                 fine_stopped: wp.array):
        """coarse.pos += interp_in_fine_cage(fine.pos - fine.pprev). Also
        snapshots coarse.pos into coarse.pprev so the coarse XPBD solve sees a
        consistent prev/pos pair (matching fine's substep convention)."""
        wp.launch(restrict_kernel, dim=self.n_verts, inputs=[
            self.pos, self.coarse_in_fine_idx, self.coarse_in_fine_uvw,
            self._fine_tet_indices_field, fine_pos_field, fine_pprev_field,
            fine_stopped,
        ])
        wp.launch(copy_pprev_kernel, dim=self.n_verts,
                  inputs=[self.pprev, self.pos])

    def solve(self, n_iters: int):
        """Jacobi: clear → solve volume → solve tetarap → apply_dP, repeat."""
        if self.n_cons == 0 or n_iters <= 0:
            return
        for _ in range(n_iters):
            wp.launch(clear_dP_kernel, dim=self.n_verts,
                      inputs=[self.dP, self.dPw])
            offset, count = self.cons_ranges[TETVOLUME]
            wp.launch(solve_tetvolume_kernel, dim=count, inputs=[
                self.cons, self.pos, self.pprev,
                self.dP, self.dPw, self.mass, self.stopped,
                float(self.dt), 1, 0, offset])
            offset, count = self.cons_ranges[TETARAP]
            wp.launch(solve_tetarap_kernel, dim=count, inputs=[
                self.cons, self.pos, self.pprev,
                self.dP, self.dPw, self.mass, self.stopped,
                self.rest_matrix,
                float(self.dt), 1, offset])
            wp.launch(apply_dP_kernel, dim=self.n_verts,
                      inputs=[self.pos, self.dP, self.dPw])

    def prolong(self, fine_pos_field: wp.array, fine_stopped: wp.array,
                blend: float = 1.0):
        wp.launch(prolong_kernel, dim=fine_pos_field.shape[0], inputs=[
            fine_pos_field, self.fine_in_coarse_idx, self.fine_in_coarse_uvw,
            self.tet_indices, self.pos, fine_stopped, float(blend),
        ])


# ---------------------------------------------------------------------------
# MultigridMuscleSim — fine MuscleSim + CoarseSolver glue
# ---------------------------------------------------------------------------

class MultigridMuscleSim(MuscleSim):
    """MuscleSim with optional GMG coarse-grid correction in step().

    Activates when cfg.coarse_geo_path is set. Each substep:
        integrate(fine)
        restrict(fine→coarse)        # transfer fine displacement
        coarse.solve(coarse_iters)   # absorb low-frequency error
        prolong(coarse→fine)         # overwrite fine.pos by coarse interp
        fine constraint solve + apply_dP + post_smooth + repair + pin
        update_velocities(fine)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Mark pintotarget src verts as stopped (kinematic) at init time so
        # GMG never sees them drifting via integrate. Without this, the first
        # substep runs integrate before apply_pintotarget marks them, and the
        # restrict step picks up huge spurious deltas.
        if getattr(self, "pintotarget_constraints", None):
            self._mark_pintotarget_stopped()
            self._pintotarget_stopped_marked = True
        path = getattr(cfg, "coarse_geo_path", None)
        if not path:
            self.coarse = None
            return
        self.coarse_iterations = int(getattr(cfg, "coarse_iterations", 5))
        self.prolong_blend = float(getattr(cfg, "prolong_blend", 1.0))
        self.coarse = CoarseSolver(
            coarse_geo_path=path,
            fine_pos_np=self.pos0_np,
            fine_tet_np=self.tet_np,
            fine_tet_indices_field=self.tet_indices,
            dt=self.dt,
            density=float(getattr(cfg, "density", 1000.0)),
            volume_stiffness=float(getattr(cfg, "coarse_volume_stiffness",
                                           getattr(cfg, "coarse_stiffness", 1e10))),
            tetarap_stiffness=float(getattr(cfg, "coarse_tetarap_stiffness",
                                            getattr(cfg, "coarse_stiffness", 1e10))),
            dampingratio=float(getattr(cfg, "coarse_dampingratio", 0.1)),
        )

    def step(self):
        self.update_attach_targets()
        for _ in range(self.cfg.num_substeps):
            self.clear_forces()
            self.accumulate_active_fiber_force()
            self.integrate()
            if self.coarse is not None:
                self.coarse.restrict(self.pos, self.pprev, self.stopped)
                self.coarse.solve(self.coarse_iterations)
                self.coarse.prolong(self.pos, self.stopped,
                                    blend=self.prolong_blend)
            self.clear()
            self.solve_constraints()
            if self.use_jacobi:
                self.apply_dP()
            self.post_smooth()
            self.freeze_near_inverted()
            self.repair_inverted_tets()
            if hasattr(self, "apply_pintotarget"):
                self.apply_pintotarget()
            self.update_velocities()
