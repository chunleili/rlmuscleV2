"""Warp ARAP-HPBD one-way pin-to-target solver.

Originally ported from realtimeMuscle lion sim; generic over any
restpos.geo + cons_<frame>.geo pipeline (lion, HumanV2, ...).

Algorithm-equivalent (not strict bit-identical) to
``realtimeMuscle/engine/arap_hpbd.py`` running the lion_hpbd.json config:
quasi-static, no gravity, no multigrid, ``nsubsteps=10``,
``fine_iterations=1``, SOR ``omega=0.1``, ``mu=1e9``.

This module is intentionally self-contained — it does NOT inherit from or
modify ``muscle.muscle_warp.MuscleSim`` so the fiber/Millard/DGF pipeline
stays untouched. The only shared symbol is ``ssvd_fn`` (oriented SVD)
imported from ``muscle.muscle_warp``.
"""
from __future__ import annotations

import warp as wp

from muscle.muscle_warp import ssvd_fn


@wp.kernel
def precompute_rest_kernel(
    pos_rest: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.vec4i),
    B: wp.array(dtype=wp.mat33),
    rest_volume: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    mu: float,
):
    """Compute B[t] = inv(D_m), V0[t] = |det(D_m)|/6, alpha[t] = 1/(mu*V0[t])."""
    t = wp.tid()
    idx = tet_indices[t]
    x0 = pos_rest[idx[0]]
    x1 = pos_rest[idx[1]]
    x2 = pos_rest[idx[2]]
    x3 = pos_rest[idx[3]]
    e1 = x1 - x0
    e2 = x2 - x0
    e3 = x3 - x0
    # mat33 is row-major; place e1,e2,e3 as columns.
    Dm = wp.mat33(
        e1[0], e2[0], e3[0],
        e1[1], e2[1], e3[1],
        e1[2], e2[2], e3[2],
    )
    det = wp.determinant(Dm)
    V0 = wp.abs(det) / 6.0
    B[t] = wp.inverse(Dm)
    rest_volume[t] = V0
    alpha[t] = 1.0 / (mu * V0)


@wp.kernel
def arap_svd_kernel(
    pos_mid: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.vec4i),
    B: wp.array(dtype=wp.mat33),
    inv_mass: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    L: wp.array(dtype=wp.float32),
    dt: float,
    omega: float,
    pos: wp.array(dtype=wp.vec3),
):
    """ARAP-SVD parallel Jacobi (atomic) constraint projection.

    Reads vertex positions from ``pos_mid`` (a snapshot taken before the
    iteration) and writes corrections atomically into ``pos`` — mirrors
    the parallel Jacobi pattern in ``project_constraints`` of
    ``realtimeMuscle/engine/arap_hpbd.py:494``.
    """
    t = wp.tid()
    idx = tet_indices[t]
    p0i = idx[0]
    p1i = idx[1]
    p2i = idx[2]
    p3i = idx[3]

    x0 = pos_mid[p0i]
    x1 = pos_mid[p1i]
    x2 = pos_mid[p2i]
    x3 = pos_mid[p3i]
    e1 = x1 - x0
    e2 = x2 - x0
    e3 = x3 - x0
    Ds = wp.mat33(
        e1[0], e2[0], e3[0],
        e1[1], e2[1], e3[1],
        e1[2], e2[2], e3[2],
    )
    Bt = B[t]
    F = Ds * Bt

    svd = ssvd_fn(F)
    sigma = svd.sigma
    s0 = sigma[0] - 1.0
    s1 = sigma[1] - 1.0
    s2 = sigma[2] - 1.0
    C2 = s0 * s0 + s1 * s1 + s2 * s2
    if C2 < 1.0e-12:
        return
    C = wp.sqrt(C2)
    dCdS = wp.vec3(s0 / C, s1 / C, s2 / C)

    diagS = wp.mat33(
        dCdS[0], 0.0, 0.0,
        0.0, dCdS[1], 0.0,
        0.0, 0.0, dCdS[2],
    )
    # dCdF = U @ diag(dCdS) @ V^T
    dCdF = svd.U * diagS * wp.transpose(svd.V)

    # Per-vertex gradients: g_{k+1} = dCdF · B[k,:]^T (ROW k of B).
    # Derivation: F = Ds·B with Ds_{i,m} = (x_{m+1}-x_0)_i, so
    # dF_{ik}/d(x_{m+1})_j = δ_{ij}·B_{m,k} → dC/d(x_{m+1})_j = (dCdF·B[m,:]^T)_j.
    row0 = wp.vec3(Bt[0, 0], Bt[0, 1], Bt[0, 2])
    row1 = wp.vec3(Bt[1, 0], Bt[1, 1], Bt[1, 2])
    row2 = wp.vec3(Bt[2, 0], Bt[2, 1], Bt[2, 2])
    g1 = dCdF * row0
    g2 = dCdF * row1
    g3 = dCdF * row2
    g0 = -(g1 + g2 + g3)

    w0 = inv_mass[p0i]
    w1 = inv_mass[p1i]
    w2 = inv_mass[p2i]
    w3 = inv_mass[p3i]
    denom = (
        w0 * wp.dot(g0, g0)
        + w1 * wp.dot(g1, g1)
        + w2 * wp.dot(g2, g2)
        + w3 * wp.dot(g3, g3)
    )
    alpha_tilde = alpha[t] / (dt * dt)
    dlambda = -(C + alpha_tilde * L[t]) / (denom + alpha_tilde)
    L[t] = L[t] + dlambda  # one slot per tet — no race

    f = omega * dlambda
    wp.atomic_add(pos, p0i, (f * w0) * g0)
    wp.atomic_add(pos, p1i, (f * w1) * g1)
    wp.atomic_add(pos, p2i, (f * w2) * g2)
    wp.atomic_add(pos, p3i, (f * w3) * g3)


@wp.kernel
def pintotarget_overwrite_kernel(
    pts: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3),
    rest_offset: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
):
    """Hard-overwrite ``pos[pts[i]] = target_pos[i] + rest_offset[i]``.

    Matches PinToTarget.solve / pintotarget_kernel in
    realtimeMuscle/engine/constraints/distance_constraints.py:124, where
    ``rest_offset[i] = restpos[pts[i]] - rest_target_pos[i]`` (computed
    once at init).
    """
    i = wp.tid()
    pid = pts[i]
    pos[pid] = target_pos[i] + rest_offset[i]


# ─── Python-side class glue ────────────────────────────────────────────────

import json
from pathlib import Path

import numpy as np

from muscle.util.geo_io import GeoFile, read_cons_frame


class OneWaySim:
    """Per-frame ARAP-HPBD driver for the pin-to-target workflow.

    Lifecycle (one frame):
        sim.load_frame_targets(frame)  # reads cons_<frame>.geo target_pos
        sim.apply_pintotarget()        # one-shot hard overwrite (per-frame)
        for _ in range(nsubsteps):     # 10 for lion
            sim.substep()              # ARAP, fine_iters=1 for lion
    """

    def __init__(self, cfg: dict, device: str | None = None):
        self.cfg = cfg
        self.mu = float(cfg["mu"])
        self.omega = float(cfg["omega"])
        self.dt_frame = float(cfg["delta_t"])
        self.nsubsteps = int(cfg["nsubsteps"])
        self.fine_iters = int(cfg["fine_iterations"])
        self.total_mass = float(cfg["total_mass"])
        self.geo_dir = str(cfg["geo_dir"])
        self.cons_dir = str(cfg.get("cons_dir", self.geo_dir))
        self.rest_path = str(cfg.get("rest_path", str(Path(self.geo_dir) / "restpos.geo")))
        self.dt = self.dt_frame / self.nsubsteps
        self.end_frame = int(cfg.get("end_frame", 90))
        self.start_frame = int(cfg.get("start_frame", 1))
        self._device = device

        rest = GeoFile(self.rest_path)
        self.NV = int(rest.positions.shape[0])
        self.NT = int(rest.tet_indices.shape[0])

        self.pos      = wp.array(rest.positions.copy(), dtype=wp.vec3, device=device)
        self.pos_mid  = wp.clone(self.pos)
        self.pos_rest = wp.array(rest.positions.copy(), dtype=wp.vec3, device=device)
        self.tet_idx  = wp.array(rest.tet_indices, dtype=wp.vec4i, device=device)

        self.B           = wp.zeros(self.NT, dtype=wp.mat33,    device=device)
        self.rest_volume = wp.zeros(self.NT, dtype=wp.float32, device=device)
        self.alpha       = wp.zeros(self.NT, dtype=wp.float32, device=device)
        self.L           = wp.zeros(self.NT, dtype=wp.float32, device=device)
        wp.launch(
            precompute_rest_kernel,
            dim=self.NT,
            inputs=[self.pos_rest, self.tet_idx, self.B,
                    self.rest_volume, self.alpha, self.mu],
            device=device,
        )

        # Per-vertex mass = sum over incident tets of (density * V0 / 4).
        V0 = self.rest_volume.numpy()
        density = self.total_mass / float(V0.sum())
        mass_v = np.zeros(self.NV, dtype=np.float32)
        per_tet_mass = density * V0 / 4.0
        for k in range(4):
            np.add.at(mass_v, rest.tet_indices[:, k], per_tet_mass)
        inv_mass_np = (1.0 / mass_v).astype(np.float32)
        self.inv_mass = wp.array(inv_mass_np, dtype=wp.float32, device=device)

        # Pin-to-target: topology (pts) is constant across all cons_<frame>.geo
        # files; only target_pos varies.
        cons0 = GeoFile(str(Path(self.cons_dir) / f"cons_{self.start_frame}.geo"))
        self.NCONS = int(cons0.pts.shape[0])
        self.pts = wp.array(cons0.pts, dtype=wp.int32, device=device)
        rest_offset_np = (rest.positions[cons0.pts] - cons0.target_pos).astype(np.float32)
        self.rest_offset = wp.array(rest_offset_np, dtype=wp.vec3, device=device)
        self.target_pos = wp.array(cons0.target_pos.copy(), dtype=wp.vec3, device=device)

    # ------------------------------------------------------------------

    def load_frame_targets(self, frame: int) -> None:
        targets = read_cons_frame(self.cons_dir, frame)
        # In-place copy preserves the GPU buffer identity / device.
        self.target_pos.assign(targets)

    def apply_pintotarget(self) -> None:
        wp.launch(
            pintotarget_overwrite_kernel,
            dim=self.NCONS,
            inputs=[self.pts, self.target_pos, self.rest_offset, self.pos],
            device=self._device,
        )

    def substep(self) -> None:
        # quasi_static=true, no_gravity, no_multigrid → ARAP only.
        self.L.zero_()
        for _ in range(self.fine_iters):
            wp.copy(self.pos_mid, self.pos)
            wp.launch(
                arap_svd_kernel,
                dim=self.NT,
                inputs=[self.pos_mid, self.tet_idx, self.B,
                        self.inv_mass, self.alpha, self.L,
                        self.dt, self.omega, self.pos],
                device=self._device,
            )

    def step_frame(self, frame: int) -> None:
        self.load_frame_targets(frame)
        self.apply_pintotarget()
        for _ in range(self.nsubsteps):
            self.substep()

    def positions_np(self) -> np.ndarray:
        return self.pos.numpy()


def load_config(config_path: str) -> dict:
    """Load a JSON scene config and resolve relative paths against cwd."""
    cfg = json.load(open(config_path))
    for key in ("geo_dir", "cons_dir", "rest_path"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(Path.cwd() / cfg[key])
    return cfg
