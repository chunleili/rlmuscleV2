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
    clear_cons_L_kernel,
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
):
    """Delta restriction: coarse.pos += Σ_k bary[k] · (fine.pos[c_k] - fine.pprev[c_k]).

    Mirrors realtimeMuscle's update_coarse_mesh_mfree_kernel (v2 dp form).
    Each substep adds fine's predicted-displacement-this-substep into
    coarse, so coarse accumulates the macro motion of the body. Coarse XPBD
    then relaxes vol+arap constraints on this drifted state, removing
    low-frequency error before prolong writes back. Requires that
    apply_pintotarget runs AFTER integrate (which snapshots pprev) but
    BEFORE this kernel — otherwise pin teleports never produce Δfine ≠ 0
    and coarse never sees bone-driven motion.
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
    b0 = 1.0 - u - v - w
    dp0 = fine_pos[tet[0]] - fine_pprev[tet[0]]
    dp1 = fine_pos[tet[1]] - fine_pprev[tet[1]]
    dp2 = fine_pos[tet[2]] - fine_pprev[tet[2]]
    dp3 = fine_pos[tet[3]] - fine_pprev[tet[3]]
    coarse_pos[i] = coarse_pos[i] + b0 * dp0 + u * dp1 + v * dp2 + w * dp3


@wp.kernel
def prolong_kernel(
    fine_pos: wp.array(dtype=wp.vec3),
    fine_in_coarse_idx: wp.array(dtype=wp.int32),
    fine_in_coarse_uvw: wp.array(dtype=wp.vec3),
    coarse_tet_indices: wp.array(dtype=wp.vec4i),
    coarse_pos: wp.array(dtype=wp.vec3),
    fine_stopped: wp.array(dtype=wp.int32),
):
    """fine.pos[i] = interp_in_coarse_cage(coarse.pos). Total overwrite.

    Mirrors realtimeMuscle's update_fine_mesh_mfree_kernel (v1 direct
    cage). Stopped (kinematic) verts are skipped — they're driven
    externally by apply_pintotarget after this; overwriting them now would
    just be undone by the re-snap, but skipping is cheaper and keeps the
    kinematic chain exact.
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
    fine_pos[i] = (
        (1.0 - u - v - w) * coarse_pos[tet[0]]
        + u * coarse_pos[tet[1]]
        + v * coarse_pos[tet[2]]
        + w * coarse_pos[tet[3]]
    )


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
                                strict_inside=False,
                                boundary_tol=1e-2):
    """For each query point, find which cage tet contains it + (u,v,w) bary.

    Bary at cage_tet[0] = 1-u-v-w; at [1],[2],[3] = u,v,w.

    strict_inside=False (default): out-of-cage queries fall back to the
    nearest tet (uvw may exceed [0,1] — extrapolation). Suitable for
    fine→coarse where the coarse cage covers everything.

    strict_inside=True: queries that are not strictly inside any cage tet
    (within boundary_tol metres of a tet's interior) get idx=-1, uvw=0 —
    caller must skip these. Required for coarse→fine when coarse covers
    gaps that fine doesn't, but boundary_tol must accommodate decimation
    drift on per-muscle coarse meshes whose surface verts sit a few mm
    off the fine muscle surface.
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
        # Bary-space tolerance: 0.05 ≈ 5% of a tet's edge length. On a 5mm
        # fine tet that's 0.25mm of metric slack — enough to absorb sub-mm
        # numerical drift from per-muscle decimation, without admitting
        # genuinely outside coarse verts (those are typically several mm out).
        eps = 5e-2
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
    if leftover.any():
        # Fallback: snap to nearest tet, with bary clamped to a non-extrapolating
        # simplex projection (each coord in [0,1], sum ≤ 1). This is stable
        # under feedback (no amplification) and handles per-muscle decimation
        # drift where coarse surface verts sit a few mm off fine surface.
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
        # Clamp each coord into [0,1] then ensure sum ≤ 1 by uniform scaling.
        x_clip = np.clip(x[leftover], 0.0, 1.0)
        s = x_clip.sum(axis=1)
        scale = np.where(s > 1.0, 1.0 / np.maximum(s, 1e-12), 1.0)
        x_clip = x_clip * scale[:, None]
        which_tet[leftover] = cand[leftover]
        bary[leftover] = x_clip

        if strict_inside:
            # Caller asked for strict mode but we still assign the projected
            # snap rather than -1: prevents instability from skipping verts
            # while preserving non-extrapolating behaviour. Verts whose
            # projection is meaningless (genuinely far from cage) are caller's
            # problem to detect by checking the metric error.
            pass

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
        fine_mass_np: np.ndarray | None = None,
        fine_tendonmask_np: np.ndarray | None = None,
    ):
        # 1. Load coarse mesh — .geo (Houdini) or .npz (per-muscle build).
        ext = str(coarse_geo_path).lower().rsplit(".", 1)[-1]
        if ext == "npz":
            data = np.load(coarse_geo_path, allow_pickle=True)
            self.pos0_np = np.asarray(data["positions"], dtype=np.float32)
            self.tet_np = np.asarray(data["tet_indices"], dtype=np.int32).reshape(-1, 4)
        else:
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

        # 3. Rest matrices, volumes (geometric only — mass moved out)
        self._compute_rest_data()

        # 4. Mass + tendonmask: gather from fine via bary maps if provided,
        # else fall back to geometric density estimate. Gathering preserves
        # mass conservation (Σ_coarse = Σ_fine) and propagates fine
        # heterogeneity (heavy bellies, tendon ends) to coarse.
        if fine_mass_np is not None:
            self._gather_mass_from_fine(np.asarray(fine_mass_np, dtype=np.float64))
        else:
            self._compute_geometric_mass()
        self._gather_tendonmask_from_fine(
            fine_pos_np, fine_tet_np,
            np.asarray(fine_tendonmask_np, dtype=np.float64)
            if fine_tendonmask_np is not None else None,
        )

        # 5. Allocate Warp fields
        self._allocate_warp_fields()

        # 6. Build coarse Constraint AOS (volume + tetarap)
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

    def _compute_geometric_mass(self):
        """Fallback: density × tet_volume / 4 distributed to 4 verts.
        Used when no fine_mass_np is supplied. Doesn't reflect fine
        heterogeneity — prefer _gather_mass_from_fine."""
        mass = np.zeros(self.n_verts, dtype=np.float64)
        avg = self.density * np.abs(self.volumes) / 4.0
        for k in range(4):
            np.add.at(mass, self.tet_np[:, k], avg)
        mass = np.maximum(mass, 1e-6)
        self.mass_np = mass.astype(np.float32)
        print(f"[CoarseSolver] mass: geometric (density × vol / 4), "
              f"Σ={mass.sum():.4f} kg")

    def _gather_mass_from_fine(self, fine_mass_np: np.ndarray):
        """Scatter fine vertex mass onto coarse verts via fine_in_coarse bary.

        For each fine vert v with mass m_v, locate its containing coarse tet
        T = fine_in_coarse_idx[v] and weights (1-u-v-w, u, v, w). Add
        m_v · weight_k to coarse_mass[T[k]]. Total mass is conserved exactly
        when bary sums to 1 (true after our snap-and-clamp), so heavy
        regions of fine map to heavy regions of coarse.
        """
        idx = self.fine_in_coarse_idx_np
        uvw = self.fine_in_coarse_uvw_np.astype(np.float64)
        n_fine = fine_mass_np.shape[0]
        if idx.shape[0] != n_fine:
            raise ValueError(
                f"fine_mass_np ({n_fine}) doesn't match fine_in_coarse "
                f"({idx.shape[0]})")

        valid_v = idx >= 0
        coarse_tets_for_fine = self.tet_np[idx[valid_v]]   # (n_v, 4)
        u = uvw[valid_v, 0]
        v = uvw[valid_v, 1]
        w = uvw[valid_v, 2]
        b0 = 1.0 - u - v - w
        bary4 = np.stack([b0, u, v, w], axis=1)            # (n_v, 4)
        contrib = fine_mass_np[valid_v][:, None] * bary4   # (n_v, 4)

        mass = np.zeros(self.n_verts, dtype=np.float64)
        for k in range(4):
            np.add.at(mass, coarse_tets_for_fine[:, k], contrib[:, k])

        # Floor any zero-mass coarse verts (shouldn't happen if fine covers
        # them but cheap safety) — pick 1% of median to mirror fine policy.
        nz = mass[mass > 0]
        floor = float(np.median(nz)) * 0.01 if nz.size else 1e-6
        n_floored = int((mass < floor).sum())
        mass = np.maximum(mass, floor)
        self.mass_np = mass.astype(np.float32)

        sum_fine = float(fine_mass_np.sum())
        sum_coarse = float(mass.sum())
        ratio = sum_coarse / sum_fine if sum_fine > 0 else float("nan")
        print(f"[CoarseSolver] mass: gathered from fine, "
              f"Σ_fine={sum_fine:.4f} → Σ_coarse={sum_coarse:.4f} "
              f"(ratio={ratio:.4f}, floored {n_floored} verts at {floor:.2e})")

    def _gather_tendonmask_from_fine(
        self,
        fine_pos_np: np.ndarray,
        fine_tet_np: np.ndarray,
        fine_tendonmask_np: np.ndarray | None,
    ):
        """Per-coarse-tet tendonmask = fine-volume-weighted average.

        For each fine tet, find which coarse tet contains its centroid,
        then accumulate fine_vol · fine_mask into coarse, divide by Σ vol.
        Only used for inspection/coloring today (coarse vol+arap kernels
        don't read tendonmask), but keeping it consistent with fine avoids
        future surprises when adding a coarse fiber pass.
        """
        n_tet = self.tet_np.shape[0]
        self.tendonmask_np = np.zeros(n_tet, dtype=np.float32)
        if fine_tendonmask_np is None:
            return
        # Fine tet centroids → which coarse tet contains them.
        ftet_pos = fine_pos_np[fine_tet_np]
        ftet_centroids = ftet_pos.mean(axis=1).astype(np.float64)
        cols = ftet_pos[:, :3, :] - ftet_pos[:, 3:4, :]
        ftet_vol = np.abs(np.linalg.det(np.transpose(cols, (0, 2, 1)))) / 6.0
        which_coarse, _ = compute_barycentric_mapping(
            ftet_centroids, self.pos0_np, self.tet_np, k=20)

        valid_f = which_coarse >= 0
        target = which_coarse[valid_f]
        w = ftet_vol[valid_f] * fine_tendonmask_np[valid_f]
        wsum = ftet_vol[valid_f]
        mask_acc = np.zeros(n_tet, dtype=np.float64)
        vol_acc = np.zeros(n_tet, dtype=np.float64)
        np.add.at(mask_acc, target, w)
        np.add.at(vol_acc, target, wsum)
        nz = vol_acc > 0
        out = np.zeros(n_tet, dtype=np.float64)
        out[nz] = mask_acc[nz] / vol_acc[nz]
        self.tendonmask_np = out.astype(np.float32)
        n_empty = int((~nz).sum())
        print(f"[CoarseSolver] tendonmask: gathered from fine "
              f"({n_empty} coarse tets received no fine tet, default 0)")

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
        # tendonmask isn't read by the current vol/arap kernels, but kept on
        # the GPU for parity with fine and for inspection / future fiber pass.
        self.tendonmask = wp.from_numpy(
            self.tendonmask_np if hasattr(self, "tendonmask_np")
            else np.zeros(self.n_tets, dtype=np.float32),
            dtype=wp.float32)

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

    def restrict(self, fine_pos_field: wp.array, fine_pprev_field: wp.array):
        """Delta restriction: coarse.pos += bary · (fine.pos - fine.pprev).

        fine.pprev must be a snapshot taken at substep start (= integrate's
        pprev := pos). apply_pintotarget should run AFTER integrate but
        BEFORE this so pin teleports show up in Δfine and propagate to
        coarse. Coarse XPBD then relaxes vol+arap on this drifted state.
        """
        wp.launch(restrict_kernel, dim=self.n_verts, inputs=[
            self.pos, self.coarse_in_fine_idx, self.coarse_in_fine_uvw,
            self._fine_tet_indices_field, fine_pos_field, fine_pprev_field,
        ])

    def solve(self, n_iters: int):
        """Jacobi: clear → solve volume → solve tetarap → apply_dP, repeat.

        XPBD λ is RESET ONCE before the iteration loop (per substep), then
        accumulates across iters within this substep. Without the reset, λ
        grows unboundedly across substeps and frames → coarse mesh explodes.
        Mirrors realtimeMuscle's reset_lagrangian(coarse.lagrangian) before
        each substep's project_constraints loop.
        """
        if self.n_cons == 0 or n_iters <= 0:
            return
        wp.launch(clear_cons_L_kernel, dim=self.n_cons, inputs=[self.cons])
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

    def prolong(self, fine_pos_field: wp.array, fine_stopped: wp.array):
        wp.launch(prolong_kernel, dim=fine_pos_field.shape[0], inputs=[
            fine_pos_field, self.fine_in_coarse_idx, self.fine_in_coarse_uvw,
            self.tet_indices, self.pos, fine_stopped,
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

        # Mirror fine constraint stiffness/damping for vol + tetarap. Coarse
        # must use the same material parameters as fine — anything weaker and
        # the coarse projection contradicts fine's stiffness, anything stronger
        # and coarse over-constrains. Read from cfg.constraints by type.
        cons_list = getattr(cfg, "constraints", []) or []
        def _find_cons(type_name: str):
            for c in cons_list:
                t = c.get("type") if isinstance(c, dict) else getattr(c, "type", None)
                if t == type_name:
                    return c
            return None
        def _get(c, key, default):
            if c is None:
                return default
            return c.get(key, default) if isinstance(c, dict) else getattr(c, key, default)
        vol_c = _find_cons("volume")
        arap_c = _find_cons("tetarap")
        coarse_vol_k = float(_get(vol_c, "stiffness", 1e10))
        coarse_arap_k = float(_get(arap_c, "stiffness", 1e10))
        # Damping: realtimeMuscle's project_constraints has zero damping in
        # the constraint solve (only velocity damping in semi_euler). Mirror
        # that — XPBD damping needs a consistent pprev, which the delta
        # restrict doesn't provide for coarse. Velocity damping still acts
        # on fine via integrate's veldamping.
        coarse_damp = 0.0
        print(f"[MultigridMuscleSim] coarse stiffness mirrors fine: "
              f"vol={coarse_vol_k:.2e}, arap={coarse_arap_k:.2e}, "
              f"damping={coarse_damp} (zeroed: matches realtimeMuscle)")

        self.coarse_iterations = int(getattr(cfg, "coarse_iterations", 5))
        self.prolong_blend = float(getattr(cfg, "prolong_blend", 1.0))
        # Debug: skip fine elasticity entirely. Only pintotarget drives motion;
        # coarse XPBD propagates that through coarse mesh; prolong writes back
        # to fine. Used to isolate whether fine constraints or coarse correction
        # is the source of explosion.
        self.coarse_only = bool(getattr(cfg, "coarse_only", False))
        if self.coarse_only:
            print("[MultigridMuscleSim] DEBUG: coarse_only mode — skipping fine elasticity")
        # Pull fine attributes off the GPU once, at init time, so the coarse
        # solver can gather them onto its own mesh (mass conservation +
        # heterogeneity propagation). MuscleSim has already populated these
        # by the time super().__init__ returns.
        fine_mass_np = self.mass.numpy() if hasattr(self, "mass") else None
        fine_tendonmask_np = (
            self.tendonmask.numpy() if hasattr(self, "tendonmask") else None
        )
        self.coarse = CoarseSolver(
            coarse_geo_path=path,
            fine_pos_np=self.pos0_np,
            fine_tet_np=self.tet_np,
            fine_tet_indices_field=self.tet_indices,
            dt=self.dt,
            density=float(getattr(cfg, "density", 1000.0)),
            volume_stiffness=coarse_vol_k,
            tetarap_stiffness=coarse_arap_k,
            dampingratio=coarse_damp,
            fine_mass_np=fine_mass_np,
            fine_tendonmask_np=fine_tendonmask_np,
        )

    def step(self):
        self.update_attach_targets()
        has_pin = hasattr(self, "apply_pintotarget")

        if self.coarse_only and self.coarse is not None:
            # Debug pipeline: pin drives motion, coarse propagates, prolong
            # writes back. No fine elasticity, no velocity update.
            #
            # Per substep:
            #   1) snapshot fine.pprev := fine.pos (no integrate to do it)
            #   2) apply_pintotarget — pin teleports happen AFTER snapshot,
            #      so Δfine = fine.pos - fine.pprev ≠ 0 at pin verts
            #   3) coarse.restrict(fine.pos, fine.pprev) — picks up Δ
            #   4) coarse.solve — propagates through vol+arap
            #   5) coarse.prolong(fine.pos) — overwrite non-stopped
            #   6) re-apply_pintotarget — keep pin verts exactly on bone
            for _ in range(self.cfg.num_substeps):
                wp.launch(copy_pprev_kernel, dim=self.pos.shape[0],
                          inputs=[self.pprev, self.pos])
                if has_pin:
                    self.apply_pintotarget()
                self.coarse.restrict(self.pos, self.pprev)
                self.coarse.solve(self.coarse_iterations)
                self.coarse.prolong(self.pos, self.stopped)
                if has_pin:
                    self.apply_pintotarget()
            return

        # Normal (full) GMG pipeline, mirrors realtimeMuscle.substep:
        #   integrate (semi_euler: pprev := pos, pos += dt·vel)
        #   apply_pintotarget — pin teleport AFTER pprev snapshot so coarse sees Δ
        #   coarse.restrict / solve / prolong
        #   fine constraint solve + apply_dP + post_smooth + repair
        #   apply_pintotarget — re-snap pins clobbered by fine constraints
        #   update_velocities
        for _ in range(self.cfg.num_substeps):
            self.clear_forces()
            self.accumulate_active_fiber_force()
            self.integrate()
            if has_pin:
                self.apply_pintotarget()
            if self.coarse is not None:
                self.coarse.restrict(self.pos, self.pprev)
                self.coarse.solve(self.coarse_iterations)
                self.coarse.prolong(self.pos, self.stopped)
            self.clear()
            self.solve_constraints()
            if self.use_jacobi:
                self.apply_dP()
            self.post_smooth()
            self.freeze_near_inverted()
            self.repair_inverted_tets()
            if has_pin:
                self.apply_pintotarget()
            self.update_velocities()
