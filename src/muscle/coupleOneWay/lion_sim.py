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

    # Per-vertex gradients: g_{k+1} = dCdF @ B[:,k]; g_0 = -(g_1+g_2+g_3).
    col0 = wp.vec3(Bt[0, 0], Bt[1, 0], Bt[2, 0])
    col1 = wp.vec3(Bt[0, 1], Bt[1, 1], Bt[2, 1])
    col2 = wp.vec3(Bt[0, 2], Bt[1, 2], Bt[2, 2])
    g1 = dCdF * col0
    g2 = dCdF * col1
    g3 = dCdF * col2
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
