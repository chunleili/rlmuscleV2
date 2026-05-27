"""Warp port of the realtimeMuscle lion ARAP-HPBD solver.

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
