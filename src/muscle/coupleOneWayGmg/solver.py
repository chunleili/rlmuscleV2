"""Warp ARAP-HPBD one-way pin-to-target solver.

Originally ported from realtimeMuscle lion sim; generic over any
restpos.geo + cons_<frame>.geo pipeline (lion, minotaur, ...).

Algorithm-equivalent (not strict bit-identical) to
``realtimeMuscle/engine/arap_hpbd.py`` running the lion_nomg.json config:
quasi-static, no gravity, no multigrid, ``nsubsteps=10``,
``fine_iterations=1``, SOR ``omega=0.1``, ``mu=1e9``.

This module is intentionally self-contained — it does NOT inherit from or
modify ``muscle.muscle_warp.MuscleSim`` so the fiber/Millard/DGF pipeline
stays untouched. The only shared symbol is ``ssvd_fn`` (oriented SVD)
imported from ``muscle.muscle_warp``.
"""
from __future__ import annotations

import warp as wp

from muscle.muscle_warp import ssvd_fn, XpbdEngine


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
def precompute_rest_scaled_kernel(
    pos_rest: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.vec4i),
    scale: wp.array(dtype=wp.float32),      # per-tet shrink factor (1.0 = no shrink)
    stiffness: wp.array(dtype=wp.float32),  # per-tet ARAP stiffness (muscle: mu; fascia: solidshapestiffness)
    B: wp.array(dtype=wp.mat33),
    rest_volume: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
):
    """Like precompute_rest_kernel but per-tet: B = inv(scale*Dm),
    rest_volume from UNSCALED Dm (physical mass), alpha = 1/(stiffness*V0)."""
    t = wp.tid()
    idx = tet_indices[t]
    x0 = pos_rest[idx[0]]
    x1 = pos_rest[idx[1]]
    x2 = pos_rest[idx[2]]
    x3 = pos_rest[idx[3]]
    e1 = x1 - x0
    e2 = x2 - x0
    e3 = x3 - x0
    Dm = wp.mat33(
        e1[0], e2[0], e3[0],
        e1[1], e2[1], e3[1],
        e1[2], e2[2], e3[2],
    )
    det = wp.determinant(Dm)
    V0 = wp.abs(det) / 6.0
    k = stiffness[t]
    if V0 < 1.0e-12:
        # Degenerate tet: make it inert (identity rest, no volume, stiff-inert).
        B[t] = wp.identity(n=3, dtype=wp.float32)
        rest_volume[t] = 0.0
        alpha[t] = 1.0e20
        return
    s = scale[t]
    Dm_s = wp.mat33(
        s * e1[0], s * e2[0], s * e3[0],
        s * e1[1], s * e2[1], s * e3[1],
        s * e1[2], s * e2[2], s * e3[2],
    )
    B[t] = wp.inverse(Dm_s)
    rest_volume[t] = V0
    alpha[t] = 1.0 / (k * V0)


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
def solve_volume_kernel(
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
    """XPBD hydrostatic volume constraint ``C = det(F) - 1``, ``F = Ds·B``.

    Parallel-Jacobi (atomic) projection mirroring :func:`arap_svd_kernel`:
    reads positions from ``pos_mid`` (a snapshot) and accumulates corrections
    into ``pos``. ARAP-SVD matches shape only and does not strictly preserve
    per-tet volume; this adds the missing incompressibility term. ``alpha`` is
    the per-tet XPBD compliance ``1/(k_vol·V0)`` and ``L`` the per-tet
    Lagrange-multiplier accumulator (one slot per tet — no race).

    Gradient mapping is identical to the ARAP kernel: with ``F = Ds·B`` and
    ``Ds_{i,m} = (x_{m+1}-x_0)_i`` we have
    ``dC/d(x_{m+1}) = dCdF · B[m,:]^T``, where for the volume term
    ``dCdF = d det(F)/dF = det(F)·F^{-T}`` (the cofactor matrix).
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

    J = wp.determinant(F)
    C = J - 1.0
    if wp.abs(C) < 1.0e-9:
        return
    # dC/dF = det(F) · F^{-T}  (cofactor of F)
    dCdF = J * wp.transpose(wp.inverse(F))

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
    L[t] = L[t] + dlambda

    f = omega * dlambda
    wp.atomic_add(pos, p0i, (f * w0) * g0)
    wp.atomic_add(pos, p1i, (f * w1) * g1)
    wp.atomic_add(pos, p2i, (f * w2) * g2)
    wp.atomic_add(pos, p3i, (f * w3) * g3)


@wp.kernel
def solve_fibermillard_kernel(
    pos_mid: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.vec4i),
    B: wp.array(dtype=wp.mat33),
    inv_mass: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    L: wp.array(dtype=wp.float32),
    v_fiber_dir: wp.array(dtype=wp.vec3),     # per-vertex rest fiber direction
    activation: wp.array(dtype=wp.float32),   # per-tet activation a in [0,1]
    tendonmask: wp.array(dtype=wp.float32),   # per-tet tendon fraction in [0,1]
    contraction_ratio: float,
    dt: float,
    omega: float,
    pos: wp.array(dtype=wp.vec3),
):
    """STUB — FiberMillard active-contraction constraint (NOT yet enabled).

    Drives the fiber stretch ``s = |F·f0|`` toward an activation-dependent
    target ``s* = 1 - (1-tendonmask)·a·contraction_ratio`` (the Millard-style
    along-fiber shortening). Same XPBD Jacobi/atomic projection as the volume
    and ARAP kernels.

    *** UNTESTED / GATED OFF. *** It requires per-vertex fiber direction
    (``v_fiber_dir``) and per-tet ``activation`` arrays, which the humanV2 GMG
    asset (``restpos.geo``) does NOT currently carry — it has only
    ``tendonmask``. Until those attributes are baked into the geo (fiber field
    from the muscle mesh + activation from the RL trajectory) and the loader is
    extended, ``__init__`` raises on ``use_fiber=True``. Kept here so the
    constitutive model and wiring are documented in one place.
    """
    t = wp.tid()
    idx = tet_indices[t]
    p0i = idx[0]
    p1i = idx[1]
    p2i = idx[2]
    p3i = idx[3]

    f0 = (v_fiber_dir[p0i] + v_fiber_dir[p1i]
          + v_fiber_dir[p2i] + v_fiber_dir[p3i]) / 4.0
    fn = wp.length(f0)
    if fn < 1.0e-9:
        return
    f0 = f0 / fn

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

    d = F * f0                      # current fiber vector
    s = wp.length(d)                # current stretch ratio
    if s < 1.0e-9:
        return
    acti = activation[t]
    belly = 1.0 - tendonmask[t]
    s_target = 1.0 - belly * acti * contraction_ratio
    C = s - s_target

    # dC/dF = (1/s)·(F f0) f0^T  →  map to verts via dCdF·B[m,:]^T
    dCdF = wp.outer(d / s, f0)
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
    L[t] = L[t] + dlambda

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


@wp.kernel
def pintotarget_overwrite_masked_kernel(
    pts: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3),
    rest_offset: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=wp.int32),     # 1 = apply pin, 0 = skip (freed vert)
    pos: wp.array(dtype=wp.vec3),
):
    """Like pintotarget_overwrite_kernel but skips verts whose pin is freed.

    Used when fascia is active: muscle verts the fascia glues to (plus a
    dilated 1-ring) are freed (active==0) so the two-way glue can pull them
    inward instead of being hard-overwritten back to the bone target.
    """
    i = wp.tid()
    if active[i] == 0:
        return
    pos[pts[i]] = target_pos[i] + rest_offset[i]


@wp.kernel
def pintotarget_soft_kernel(
    pts: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3),
    rest_offset: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=wp.int32),     # 0 = freed vert (gets soft pull), 1 = skip
    stiffness: float,                     # in [0,1]: fraction toward bone target / substep
    pos: wp.array(dtype=wp.vec3),
):
    """Soft pin for FREED verts: blend pos toward the bone target.

    Freed verts (active==0) are not hard-overwritten by the masked pin, so
    they can yield to the fascia glue. But without any tie to the bone they
    drift catastrophically when the per-frame target jumps. This soft pull
    (a PBD-style partial step toward target) lets them track gross body
    motion while still yielding inward to the fascia squeeze.
    """
    i = wp.tid()
    if active[i] != 0:
        return
    pid = pts[i]
    tgt = target_pos[i] + rest_offset[i]
    pos[pid] = pos[pid] + stiffness * (tgt - pos[pid])


@wp.kernel
def solve_glue_kernel(
    pair_i: wp.array(dtype=wp.int32),    # fascia vertex (unique per pair)
    pair_j: wp.array(dtype=wp.int32),    # muscle vertex (may repeat across pairs)
    restlen: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),  # 1/stiffness
    inv_mass: wp.array(dtype=wp.float32),
    L: wp.array(dtype=wp.float32),
    dt: float,
    omega: float,
    pos: wp.array(dtype=wp.vec3),
):
    """XPBD two-way distance constraint between pos[pair_i] and pos[pair_j]."""
    c = wp.tid()
    i = pair_i[c]
    j = pair_j[c]
    n = pos[i] - pos[j]
    d = wp.length(n)
    if d < 1.0e-9:
        return
    dirn = n / d
    C = d - restlen[c]
    wi = inv_mass[i]
    wj = inv_mass[j]
    alpha_tilde = compliance[c] / (dt * dt)
    denom = wi + wj + alpha_tilde
    if denom < 1.0e-12:
        return
    dlambda = -(C + alpha_tilde * L[c]) / denom
    L[c] = L[c] + dlambda
    f = omega * dlambda
    wp.atomic_add(pos, i, (f * wi) * dirn)
    wp.atomic_add(pos, j, (-f * wj) * dirn)


@wp.kernel
def fascia_anchor_kernel(
    anchor_fascia: wp.array(dtype=wp.int32),
    anchor_muscle: wp.array(dtype=wp.int32),
    rest_offset: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
):
    """pos[anchor_fascia[i]] = pos[anchor_muscle[i]] + rest_offset[i]."""
    i = wp.tid()
    pos[anchor_fascia[i]] = pos[anchor_muscle[i]] + rest_offset[i]


@wp.kernel
def compute_barycentric_kernel(
    query_pos: wp.array(dtype=wp.vec3),
    cage_pos: wp.array(dtype=wp.vec3),
    cage_tet: wp.array(dtype=wp.vec4i),
    out_cage_idx: wp.array(dtype=wp.int32),
    out_bary: wp.array(dtype=wp.vec3),
):
    """For each query point, find a cage tet containing it (brute force).

    Mirrors `compute_barycentric_kernel` in
    realtimeMuscle/engine/compute_R_acc.py — falls back to nearest-vertex tet
    if no container is found. Brute force is fine: this runs once at init.
    """
    i = wp.tid()
    p = query_pos[i]
    n_cage = cage_tet.shape[0]

    found_idx = int(-1)
    found_bary = wp.vec3(0.0, 0.0, 0.0)
    min_dis = float(1.0e30)
    nearest_tet = int(0)
    nearest_bary = wp.vec3(0.0, 0.0, 0.0)

    for t in range(n_cage):
        idx = cage_tet[t]
        p0 = cage_pos[idx[0]]
        p1 = cage_pos[idx[1]]
        p2 = cage_pos[idx[2]]
        p3 = cage_pos[idx[3]]
        Am = wp.mat33(
            p1[0] - p0[0], p2[0] - p0[0], p3[0] - p0[0],
            p1[1] - p0[1], p2[1] - p0[1], p3[1] - p0[1],
            p1[2] - p0[2], p2[2] - p0[2], p3[2] - p0[2],
        )
        bv = p - p0
        x = wp.inverse(Am) * bv
        contained = (x[0] >= 0.0 and x[1] >= 0.0 and x[2] >= 0.0
                     and (x[0] + x[1] + x[2] <= 1.0))
        if contained and found_idx == -1:
            found_idx = t
            found_bary = x

        d0 = wp.length(p0 - p)
        d1 = wp.length(p1 - p)
        d2 = wp.length(p2 - p)
        d3 = wp.length(p3 - p)
        dm = wp.min(wp.min(d0, d1), wp.min(d2, d3))
        if dm < min_dis:
            min_dis = dm
            nearest_tet = t
            nearest_bary = x

    if found_idx >= 0:
        out_cage_idx[i] = found_idx
        out_bary[i] = found_bary
    else:
        out_cage_idx[i] = nearest_tet
        out_bary[i] = nearest_bary


@wp.kernel
def restrict_direct_kernel(
    fine_pos: wp.array(dtype=wp.vec3),
    fine_tet: wp.array(dtype=wp.vec4i),
    cage_idx_c2f: wp.array(dtype=wp.int32),
    uvw_c2f: wp.array(dtype=wp.vec3),
    coarse_pos: wp.array(dtype=wp.vec3),
):
    """coarse.pos[i] = blend(fine.pos at containing fine tet).

    Direct restriction. Stable V-cycle: coarse mirrors fine each substep,
    relaxes elasticity, prolongs back, fine re-pins and refines. Replaces
    the dp variant from arap_hpbd_nogui.py which is unstable under
    quasi-static (fine_old_pos never updated → cumulative coarse strain).
    """
    i = wp.tid()
    t = cage_idx_c2f[i]
    idx = fine_tet[t]
    uvw = uvw_c2f[i]
    u = uvw[0]
    v = uvw[1]
    w = uvw[2]
    a = 1.0 - u - v - w
    coarse_pos[i] = (a * fine_pos[idx[0]] + u * fine_pos[idx[1]]
                     + v * fine_pos[idx[2]] + w * fine_pos[idx[3]])


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


# ─── Python-side class glue ────────────────────────────────────────────────

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from muscle.util.geo_io import GeoFile, read_cons_frame


def _check_unzipped(path: str) -> None:
    """Raise if a required .geo asset is missing — model dirs ship as zips.

    Each case's geometry (restpos, restpos_lowres, cons_*.geo) lives inside
    ``data/<case>/model.zip`` to save LFS storage. Unzip once before running
    any sim. See data/README.md.
    """
    p = Path(path)
    if p.exists():
        return
    parts = p.parts
    if "data" in parts:
        i = parts.index("data")
        if i + 1 < len(parts):
            case_dir = Path(*parts[: i + 2])
            zip_path = case_dir / "model.zip"
            if zip_path.exists():
                raise FileNotFoundError(
                    f"missing asset: {path}\n"
                    f"model assets are zipped — extract first:\n"
                    f"    (cd {case_dir} && unzip -o model.zip)\n"
                    f"see data/README.md"
                )
    raise FileNotFoundError(f"missing asset: {path}")


@dataclass
class MergedState:
    pos: wp.array
    pos_mid: wp.array
    pos_rest: wp.array
    tet_idx: wp.array
    B: wp.array
    rest_volume: wp.array
    alpha: wp.array
    L: wp.array
    inv_mass: wp.array
    NT: int
    glue_fascia: wp.array
    glue_muscle: wp.array
    glue_restlen: wp.array
    glue_compliance: wp.array
    glue_L: wp.array
    n_glue: int
    anchor_fascia: wp.array
    anchor_muscle: wp.array
    anchor_offset: wp.array
    n_anchor: int


def build_merged_state(muscle_pos, muscle_tet, fascia_pos, fascia_tet,
                       fascia_rest_scale, muscle_stiffness, fascia_stiffness,
                       glue_fascia, glue_muscle, glue_restlen, glue_stiffness,
                       anchor_fascia, anchor_muscle, total_mass, device):
    """Concatenate muscle+fascia into one particle system and precompute rest.

    glue_fascia/anchor_fascia are LOCAL fascia vertex indices (offset by muscle
    vertex count here). glue_muscle/anchor_muscle are muscle indices.
    """
    nvm = muscle_pos.shape[0]
    pos_np = np.vstack([muscle_pos, fascia_pos]).astype(np.float32)
    tet_np = np.vstack([muscle_tet, fascia_tet + nvm]).astype(np.int32)
    nt_m = muscle_tet.shape[0]
    NT = tet_np.shape[0]
    NV = pos_np.shape[0]

    scale_np = np.ones(NT, dtype=np.float32)
    scale_np[nt_m:] = fascia_rest_scale
    stiff_np = np.empty(NT, dtype=np.float32)
    stiff_np[:nt_m] = muscle_stiffness
    stiff_np[nt_m:] = fascia_stiffness

    pos = wp.array(pos_np, dtype=wp.vec3, device=device)
    pos_mid = wp.clone(pos)
    pos_rest = wp.array(pos_np.copy(), dtype=wp.vec3, device=device)
    tet_idx = wp.array(tet_np, dtype=wp.vec4i, device=device)
    B = wp.zeros(NT, dtype=wp.mat33, device=device)
    rest_volume = wp.zeros(NT, dtype=wp.float32, device=device)
    alpha = wp.zeros(NT, dtype=wp.float32, device=device)
    L = wp.zeros(NT, dtype=wp.float32, device=device)
    scale = wp.array(scale_np, dtype=wp.float32, device=device)
    stiff = wp.array(stiff_np, dtype=wp.float32, device=device)
    wp.launch(precompute_rest_scaled_kernel, dim=NT,
              inputs=[pos_rest, tet_idx, scale, stiff, B, rest_volume, alpha],
              device=device)

    V0 = rest_volume.numpy()
    density = total_mass / float(V0.sum())
    mass_v = np.zeros(NV, dtype=np.float32)
    per_tet_mass = density * V0 / 4.0
    for k in range(4):
        np.add.at(mass_v, tet_np[:, k], per_tet_mass)
    inv_mass_np = np.where(mass_v > 0.0, 1.0 / np.maximum(mass_v, 1e-30), 0.0).astype(np.float32)
    inv_mass = wp.array(inv_mass_np, dtype=wp.float32, device=device)

    gf = (np.asarray(glue_fascia, np.int32) + nvm)
    gm = np.asarray(glue_muscle, np.int32)
    n_glue = len(gf)
    af = (np.asarray(anchor_fascia, np.int32) + nvm)
    am = np.asarray(anchor_muscle, np.int32)
    offset_np = (pos_np[af] - pos_np[am]).astype(np.float32) if len(af) else np.zeros((0, 3), np.float32)

    return MergedState(
        pos=pos, pos_mid=pos_mid, pos_rest=pos_rest, tet_idx=tet_idx,
        B=B, rest_volume=rest_volume, alpha=alpha, L=L, inv_mass=inv_mass, NT=NT,
        glue_fascia=wp.array(gf, dtype=wp.int32, device=device),
        glue_muscle=wp.array(gm, dtype=wp.int32, device=device),
        glue_restlen=wp.array(np.asarray(glue_restlen, np.float32), dtype=wp.float32, device=device),
        glue_compliance=wp.array((1.0 / np.asarray(glue_stiffness, np.float32)).astype(np.float32),
                                 dtype=wp.float32, device=device),
        glue_L=wp.zeros(max(n_glue, 1), dtype=wp.float32, device=device),
        n_glue=n_glue,
        anchor_fascia=wp.array(af, dtype=wp.int32, device=device),
        anchor_muscle=wp.array(am, dtype=wp.int32, device=device),
        anchor_offset=wp.array(offset_np, dtype=wp.vec3, device=device),
        n_anchor=len(af),
    )


def apply_anchor(state, device=None):
    if state.n_anchor:
        wp.launch(fascia_anchor_kernel, dim=state.n_anchor,
                  inputs=[state.anchor_fascia, state.anchor_muscle,
                          state.anchor_offset, state.pos], device=device)


def solve_glue(state, dt, omega, iters, device=None):
    for _ in range(iters):
        state.glue_L.zero_()
        if state.n_glue:
            wp.launch(solve_glue_kernel, dim=state.n_glue,
                      inputs=[state.glue_fascia, state.glue_muscle,
                              state.glue_restlen, state.glue_compliance,
                              state.inv_mass, state.glue_L, dt, omega, state.pos],
                      device=device)


def run_substeps(state, nsubsteps, fine_iters, glue_iters, dt, omega, device=None):
    """Standalone (no multigrid, no pin) loop — used by tests."""
    for _ in range(nsubsteps):
        apply_anchor(state, device)
        state.L.zero_()
        for _ in range(fine_iters):
            wp.copy(state.pos_mid, state.pos)
            wp.launch(arap_svd_kernel, dim=state.NT,
                      inputs=[state.pos_mid, state.tet_idx, state.B,
                              state.inv_mass, state.alpha, state.L, dt, omega, state.pos],
                      device=device)
        solve_glue(state, dt, omega, glue_iters, device)


class OneWaySim(XpbdEngine):
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
        # Master switch. When False the fascia layer is fully disabled and the
        # solver is byte-for-byte the pre-fascia pin-to-target sim (no merged
        # state, no pin_active mask, no anchor/glue). Defaults to False so any
        # config lacking the key reproduces the original behaviour exactly.
        self.use_fascia = bool(cfg.get("use_fascia", False))
        self.fascia_path = cfg.get("fascia_path") if self.use_fascia else None
        if self.fascia_path is not None and not Path(self.fascia_path).is_absolute():
            self.fascia_path = str(Path.cwd() / self.fascia_path)
        self.fascia_rest_scale = float(cfg.get("fascia_rest_scale", 0.95))
        self.fascia_glue_iters = int(cfg.get("fascia_glue_iterations", self.fine_iters))
        # Soft-pin stiffness for FREED muscle verts (fraction toward bone target
        # per substep). Keeps freed verts tracking gross body motion so they
        # don't tear loose when the per-frame target jumps, while still letting
        # the fascia glue pull them inward. 0 disables (pure free).
        self.fascia_soft_pin = float(cfg.get("fascia_soft_pin", 0.5))
        self.dt = self.dt_frame / self.nsubsteps
        self.end_frame = int(cfg.get("end_frame", 90))
        self.start_frame = int(cfg.get("start_frame", 1))
        self._device = device

        self.use_multigrid = bool(cfg.get("use_multigrid", False))
        self.coarse_iters = int(cfg.get("coarse_iterations", 0))
        # Accept either `coarse_rest_path` (preferred) or legacy
        # `coarse_model_path` (matches realtimeMuscle scene JSONs).
        self.coarse_rest_path = cfg.get("coarse_rest_path",
                                        cfg.get("coarse_model_path", None))
        if self.coarse_rest_path is not None:
            self.coarse_rest_path = str(self.coarse_rest_path)

        # Optional explicit hydrostatic volume constraint (incompressibility).
        # ARAP-SVD matches shape only; with use_volume the fine loop adds a
        # det(F)=1 XPBD constraint at stiffness volume_stiffness. Off by default
        # so existing configs reproduce the pure ARAP-HPBD behaviour exactly.
        self.use_volume = bool(cfg.get("use_volume", False))
        self.volume_stiffness = float(cfg.get("volume_stiffness", 1.0e10))

        # Optional FiberMillard active-contraction constraint. SCAFFOLD ONLY:
        # the humanV2 GMG asset lacks per-vertex fiber direction and per-tet
        # activation, so enabling this raises until that data is baked in (see
        # solve_fibermillard_kernel). Flags parsed here so configs/docs are ready.
        self.use_fiber = bool(cfg.get("use_fiber", False))
        self.fiber_model = str(cfg.get("fiber_model", "fibermillard"))
        self.fiber_stiffness = float(cfg.get("fiber_stiffness", 1.0e4))
        self.contraction_ratio = float(cfg.get("contraction_ratio", 0.4))
        if self.use_fiber:
            raise NotImplementedError(
                "use_fiber=true is not yet supported by the OneWayGmg solver: "
                "the FiberMillard constraint needs per-vertex fiber direction "
                "and per-tet activation, which the humanV2 asset (restpos.geo) "
                "does not carry (only tendonmask is present). Bake fiber "
                "direction (from the muscle fiber field) and activation (from "
                "the RL trajectory) into the geo, extend the loader to read "
                "them, then enable solve_fibermillard_kernel in substep()."
            )

        _check_unzipped(self.rest_path)
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

        # Per-tet compliance + Lagrange accumulator for the optional volume
        # constraint: alpha_vol = 1/(k_vol·V0). Degenerate tets (V0<=0) are made
        # inert with a large compliance so they contribute no correction.
        if self.use_volume:
            with np.errstate(divide="ignore"):
                vol_alpha_np = 1.0 / (self.volume_stiffness * V0)
            vol_alpha_np[~np.isfinite(vol_alpha_np)] = 1.0e30
            vol_alpha_np[V0 <= 0.0] = 1.0e30
            self.vol_alpha = wp.array(vol_alpha_np.astype(np.float32),
                                      dtype=wp.float32, device=device)
            self.vol_L = wp.zeros(self.NT, dtype=wp.float32, device=device)

        # Pin-to-target: topology (pts) is constant across all cons_<frame>.geo
        # files; only target_pos varies.
        cons0_path = str(Path(self.cons_dir) / f"cons_{self.start_frame}.geo")
        _check_unzipped(cons0_path)
        cons0 = GeoFile(cons0_path)
        self.NCONS = int(cons0.pts.shape[0])
        self.pts = wp.array(cons0.pts, dtype=wp.int32, device=device)
        self._pin_pts_np = np.asarray(cons0.pts, dtype=np.int64)
        self.pin_active = None  # set by _init_fascia when fascia is active
        rest_offset_np = (rest.positions[cons0.pts] - cons0.target_pos).astype(np.float32)
        self.rest_offset = wp.array(rest_offset_np, dtype=wp.vec3, device=device)
        self.target_pos = wp.array(cons0.target_pos.copy(), dtype=wp.vec3, device=device)

        if self.use_multigrid:
            if self.coarse_rest_path is None:
                raise ValueError(
                    "use_multigrid=true but no coarse_rest_path / "
                    "coarse_model_path in config"
                )
            self._init_coarse(rest.positions)

        if self.fascia_path:
            self._init_fascia(rest.positions.copy())

    # ------------------------------------------------------------------

    def _init_fascia(self, rest_muscle_pos):
        from scipy.spatial import cKDTree
        import logging
        log = logging.getLogger(__name__)

        _check_unzipped(self.fascia_path)
        fg = GeoFile(self.fascia_path)
        fascia_pos = fg.positions
        fascia_tet = fg.tet_indices

        mrest = GeoFile(self.rest_path)

        # glue_scale is needed by both pairing modes (surface uses it for the
        # uniform stiffness); read it here so _pair_surface can see it.
        glue_scale = float(self.cfg.get("fascia_glue_stiffness_scale", 0.001))

        glue_mode = str(self.cfg.get("fascia_glue_mode", "groups"))
        if glue_mode == "surface":
            glue_fascia, glue_muscle, glue_restlen, glue_stiff = self._pair_surface(
                fg, fascia_pos, mrest, rest_muscle_pos, glue_scale)
        else:
            surf = mrest.point_group("surface_points")
            surf_tree = cKDTree(rest_muscle_pos[surf])

            def pair(group_name, dist_attr, stiff_attr):
                if group_name not in fg._groups:
                    return (np.zeros(0, np.int32), np.zeros(0, np.int32),
                            np.zeros(0, np.float32), np.zeros(0, np.float32))
                fids = fg.point_group(group_name)
                d, loc = surf_tree.query(fascia_pos[fids], k=1)
                mids = surf[loc].astype(np.int32)
                want = fg.point_attr(dist_attr)[fids].astype(np.float32)
                restlen = np.minimum(want, d.astype(np.float32))  # don't yank distant pts
                stiff = fg.point_attr(stiff_attr)[fids].astype(np.float32)
                return fids.astype(np.int32), mids, restlen, stiff

            gi_f, gi_m, gi_r, gi_s = pair("tissue_inner_points",
                                          "tissuetomuscledistance", "tissuetomusclestiffness")
            gb_f, gb_m, gb_r, gb_s = pair("tissue_bone_points",
                                          "tissuetobonedistance", "tissuetobonestiffness")
            glue_fascia = np.concatenate([gi_f, gb_f])
            glue_muscle = np.concatenate([gi_m, gb_m])
            glue_restlen = np.concatenate([gi_r, gb_r])
            glue_stiff = np.concatenate([gi_s, gb_s]) * glue_scale

        if "rigid" in fg._groups:
            rid = fg.point_group("rigid")
            full_tree = cKDTree(rest_muscle_pos)
            _, aloc = full_tree.query(fascia_pos[rid], k=1)
            anchor_fascia = rid.astype(np.int32)
            anchor_muscle = aloc.astype(np.int32)
        else:
            anchor_fascia = np.zeros(0, np.int32)
            anchor_muscle = np.zeros(0, np.int32)

        log.info(f"fascia: {fascia_pos.shape[0]} verts, {fascia_tet.shape[0]} tets; "
                 f"glue mode={glue_mode}, glue pairs={len(glue_fascia)}; "
                 f"anchors={len(anchor_fascia)}; rest_scale={self.fascia_rest_scale}")

        # --- Free the muscle verts the fascia glues to (+dilated ring) from the
        # pin overwrite, and soften the surface muscle tets so the glue can pull
        # them inward. See Task 9. ---
        free_ring = int(self.cfg.get("fascia_free_ring", 1))
        surface_mu = float(self.cfg.get("fascia_surface_mu", 1.0e5))

        muscle_tet = mrest.tet_indices  # (nt_m, 4)
        nvm = rest_muscle_pos.shape[0]
        free_mask = np.zeros(nvm, dtype=bool)
        seed = np.unique(glue_muscle.astype(np.int64))
        free_mask[seed[(seed >= 0) & (seed < nvm)]] = True
        # Dilate by `free_ring` rings via muscle-tet adjacency.
        if free_ring > 0:
            from collections import defaultdict
            v2t = defaultdict(list)
            for ti, tet in enumerate(muscle_tet):
                for v in tet:
                    v2t[int(v)].append(ti)
            for _ in range(free_ring):
                cur = np.where(free_mask)[0]
                for v in cur:
                    for ti in v2t[int(v)]:
                        for w in muscle_tet[ti]:
                            free_mask[int(w)] = True
        free_ids = np.where(free_mask)[0].astype(np.int32)

        # Per-muscle-tet stiffness: soft where a tet touches a freed vert.
        muscle_stiff = np.full(muscle_tet.shape[0], self.mu, dtype=np.float32)
        tet_touches_free = free_mask[muscle_tet].any(axis=1)
        muscle_stiff[tet_touches_free] = surface_mu

        self.nvm = rest_muscle_pos.shape[0]
        self.merged = build_merged_state(
            muscle_pos=rest_muscle_pos, muscle_tet=mrest.tet_indices,
            fascia_pos=fascia_pos, fascia_tet=fascia_tet,
            fascia_rest_scale=self.fascia_rest_scale,
            muscle_stiffness=muscle_stiff,
            fascia_stiffness=float(fg.point_attr("solidshapestiffness").mean()),
            glue_fascia=glue_fascia, glue_muscle=glue_muscle,
            glue_restlen=glue_restlen, glue_stiffness=glue_stiff,
            anchor_fascia=anchor_fascia, anchor_muscle=anchor_muscle,
            total_mass=self.total_mass, device=self._device)

        # Build the pin_active mask over the existing pin pts: free verts in free_ids.
        active = np.ones(self._pin_pts_np.shape[0], dtype=np.int32)
        active[np.isin(self._pin_pts_np, free_ids)] = 0
        self.pin_active = wp.array(active, dtype=wp.int32, device=self._device)
        log.info(f"fascia: freed {len(free_ids)} muscle verts from pin "
                 f"(ring={free_ring}); {int(active.sum())}/{len(active)} pins active; "
                 f"surface_mu={surface_mu}, glue_scale={glue_scale}")
        self.pos = self.merged.pos
        self.pos_mid = self.merged.pos_mid
        self.tet_idx = self.merged.tet_idx
        self.B = self.merged.B
        self.alpha = self.merged.alpha
        self.L = self.merged.L
        self.inv_mass = self.merged.inv_mass
        self.NT = self.merged.NT
        self.NV = self.pos.shape[0]

    # ------------------------------------------------------------------

    def _pair_surface(self, fg, fascia_pos, mrest, rest_muscle_pos, glue_scale):
        """Build glue pairs from the MUSCLE side (full-surface glue mode).

        Instead of pairing the small tissue_inner/tissue_bone fascia groups to
        their nearest muscle vert (groups mode, ~4252 pairs), this pairs EVERY
        muscle surface vertex to its nearest fascia vertex (~119482 pairs). This
        un-dilutes the global mean-radius squeeze metric by gluing the whole
        surface, not just a small patch. See Task 10.

        Returns (glue_fascia LOCAL ids, glue_muscle GLOBAL ids, glue_restlen,
        glue_stiff) — same convention as groups mode; build_merged_state offsets
        glue_fascia by nvm.
        """
        from scipy.spatial import cKDTree

        surf = mrest.point_group("surface_points")  # global muscle vert ids
        fascia_tree = cKDTree(fascia_pos)
        d, floc = fascia_tree.query(rest_muscle_pos[surf], k=1)  # floc: LOCAL fascia ids

        want = float(self.cfg.get("fascia_glue_target_distance", 0.01))
        glue_muscle = surf.astype(np.int32)
        glue_fascia = floc.astype(np.int32)
        glue_restlen = np.minimum(want, d.astype(np.float32)).astype(np.float32)

        base_stiff = float(self.cfg.get(
            "fascia_glue_stiffness",
            float(fg.point_attr("tissuetomusclestiffness").mean())))
        glue_stiff = np.full(len(surf), base_stiff * glue_scale, dtype=np.float32)
        return glue_fascia, glue_muscle, glue_restlen, glue_stiff

    # ------------------------------------------------------------------

    def _init_coarse(self, fine_rest_positions: np.ndarray) -> None:
        """Load coarse mesh, allocate per-coarse fields, build f↔c barycentric map.

        Both mappings are computed in rest pose: fine vert i → containing
        coarse tet (cage_idx, uvw) for prolongation; coarse vert i →
        containing fine tet (cage_idx_c2f, uvw_c2f) for restriction.
        """
        device = self._device
        _check_unzipped(self.coarse_rest_path)
        coarse = GeoFile(self.coarse_rest_path)
        self.coarse_NV = int(coarse.positions.shape[0])
        self.coarse_NT = int(coarse.tet_indices.shape[0])

        self.coarse_pos     = wp.array(coarse.positions.copy(), dtype=wp.vec3, device=device)
        self.coarse_pos_mid = wp.clone(self.coarse_pos)
        self.coarse_tet_idx = wp.array(coarse.tet_indices, dtype=wp.vec4i, device=device)

        self.coarse_B           = wp.zeros(self.coarse_NT, dtype=wp.mat33,   device=device)
        self.coarse_rest_volume = wp.zeros(self.coarse_NT, dtype=wp.float32, device=device)
        self.coarse_alpha       = wp.zeros(self.coarse_NT, dtype=wp.float32, device=device)
        self.coarse_L           = wp.zeros(self.coarse_NT, dtype=wp.float32, device=device)
        wp.launch(
            precompute_rest_kernel,
            dim=self.coarse_NT,
            inputs=[self.coarse_pos, self.coarse_tet_idx, self.coarse_B,
                    self.coarse_rest_volume, self.coarse_alpha, self.mu],
            device=device,
        )
        V0c = self.coarse_rest_volume.numpy()
        density_c = self.total_mass / float(V0c.sum())
        mass_c = np.zeros(self.coarse_NV, dtype=np.float32)
        per_tet_mass_c = density_c * V0c / 4.0
        for k in range(4):
            np.add.at(mass_c, coarse.tet_indices[:, k], per_tet_mass_c)
        self.coarse_inv_mass = wp.array((1.0 / mass_c).astype(np.float32),
                                        dtype=wp.float32, device=device)

        # Rest-pose barycentric maps (one-shot brute force).
        fine_pos_rest = wp.array(fine_rest_positions.copy(), dtype=wp.vec3, device=device)

        # fine → coarse: each fine vert i lives in some coarse tet
        self.cage_idx = wp.zeros(self.NV, dtype=wp.int32, device=device)
        self.uvw      = wp.zeros(self.NV, dtype=wp.vec3,  device=device)
        wp.launch(
            compute_barycentric_kernel,
            dim=self.NV,
            inputs=[fine_pos_rest, self.coarse_pos, self.coarse_tet_idx,
                    self.cage_idx, self.uvw],
            device=device,
        )

        # coarse → fine: each coarse vert i lives in some fine tet
        self.cage_idx_c2f = wp.zeros(self.coarse_NV, dtype=wp.int32, device=device)
        self.uvw_c2f      = wp.zeros(self.coarse_NV, dtype=wp.vec3,  device=device)
        wp.launch(
            compute_barycentric_kernel,
            dim=self.coarse_NV,
            inputs=[self.coarse_pos, fine_pos_rest, self.tet_idx,
                    self.cage_idx_c2f, self.uvw_c2f],
            device=device,
        )
        wp.synchronize()

        # Capture the MUSCLE-only fine vertex count for multigrid prolongation.
        # _init_coarse runs before _init_fascia, so self.NV is still muscle-only
        # here; _init_fascia later mutates self.NV to the merged total. The
        # cage_idx/uvw maps only have this many entries, so prolong must launch
        # with mg_fine_nv (not the merged self.NV) to stay in bounds.
        self.mg_fine_nv = self.NV

    # ------------------------------------------------------------------

    def load_frame_targets(self, frame: int) -> None:
        targets = read_cons_frame(self.cons_dir, frame)
        # In-place copy preserves the GPU buffer identity / device.
        self.target_pos.assign(targets)

    def apply_pintotarget(self) -> None:
        if getattr(self, "pin_active", None) is not None:
            wp.launch(
                pintotarget_overwrite_masked_kernel,
                dim=self.NCONS,
                inputs=[self.pts, self.target_pos, self.rest_offset,
                        self.pin_active, self.pos],
                device=self._device,
            )
            # Soft-pull the freed verts toward their bone target so they track
            # gross body motion (per-frame target jumps) while still yielding
            # inward to the fascia squeeze.
            if self.fascia_soft_pin > 0.0:
                wp.launch(
                    pintotarget_soft_kernel,
                    dim=self.NCONS,
                    inputs=[self.pts, self.target_pos, self.rest_offset,
                            self.pin_active, self.fascia_soft_pin, self.pos],
                    device=self._device,
                )
        else:
            wp.launch(
                pintotarget_overwrite_kernel,
                dim=self.NCONS,
                inputs=[self.pts, self.target_pos, self.rest_offset, self.pos],
                device=self._device,
            )

    def build_program(self):
        s = self

        def pre_phase():
            if s.use_multigrid:
                # Direct V-cycle: restrict fine→coarse, relax coarse, prolong
                # back, re-apply pin-to-target (prolong overwrites pinned verts).
                wp.launch(
                    restrict_direct_kernel,
                    dim=s.coarse_NV,
                    inputs=[s.pos, s.tet_idx,
                            s.cage_idx_c2f, s.uvw_c2f, s.coarse_pos],
                    device=s._device,
                )
                s.coarse_L.zero_()
                for _ in range(s.coarse_iters):
                    wp.copy(s.coarse_pos_mid, s.coarse_pos)
                    wp.launch(
                        arap_svd_kernel,
                        dim=s.coarse_NT,
                        inputs=[s.coarse_pos_mid, s.coarse_tet_idx,
                                s.coarse_B, s.coarse_inv_mass,
                                s.coarse_alpha, s.coarse_L,
                                s.dt, s.omega, s.coarse_pos],
                        device=s._device,
                    )
                wp.launch(
                    prolong_kernel,
                    dim=getattr(s, "mg_fine_nv", s.NV),
                    inputs=[s.coarse_pos, s.coarse_tet_idx,
                            s.cage_idx, s.uvw, s.pos],
                    device=s._device,
                )
                s.apply_pintotarget()
            if getattr(s, "merged", None) is not None:
                apply_anchor(s.merged, s._device)
            s.L.zero_()
            if s.use_volume:
                s.vol_L.zero_()

        def iter_phase():
            for it in range(s.fine_iters):
                wp.copy(s.pos_mid, s.pos)
                wp.launch(
                    arap_svd_kernel,
                    dim=s.NT,
                    inputs=[s.pos_mid, s.tet_idx, s.B,
                            s.inv_mass, s.alpha, s.L,
                            s.dt, s.omega, s.pos],
                    device=s._device,
                )
                # Optional incompressibility: project det(F)=1 from the same
                # pos_mid snapshot (parallel-Jacobi over constraint types).
                if s.use_volume:
                    wp.launch(
                        solve_volume_kernel,
                        dim=s.NT,
                        inputs=[s.pos_mid, s.tet_idx, s.B,
                                s.inv_mass, s.vol_alpha, s.vol_L,
                                s.dt, s.omega, s.pos],
                        device=s._device,
                    )
                # Per-iteration collision hook (UDCF DCD): interleaves contact
                # resolution with ARAP/volume the way the reference PBD loop
                # interleaves narrowPhase() with projectConstraints().
                for fn in getattr(s, "collision_solvers_iter", None) or []:
                    fn(it)

        def post_phase():
            if getattr(s, "merged", None) is not None:
                solve_glue(s.merged, s.dt, s.omega,
                           s.fascia_glue_iters, s._device)
            # Optional collision pass (e.g. muscle vs rigid SDF). Runs AFTER ARAP /
            # glue each substep so it pushes against the relaxed geometry; pinned
            # verts are protected by the collider's own inv_mass=0 mask.
            for fn in getattr(s, "collision_solvers", None) or []:
                fn()

        self.program = [pre_phase, iter_phase, post_phase]

    def substep(self) -> None:
        if not hasattr(self, "program"):
            self.build_program()
        for phase in self.program:
            phase()

    def step_frame(self, frame: int) -> None:
        self.load_frame_targets(frame)
        self.apply_pintotarget()
        if getattr(self, "merged", None) is not None:
            apply_anchor(self.merged, self._device)
        for _ in range(self.nsubsteps):
            self.substep()

    def positions_np(self) -> np.ndarray:
        return self.pos.numpy()


def load_config(config_path: str) -> dict:
    """Load a JSON scene config and resolve relative paths against cwd."""
    cfg = json.load(open(config_path))
    for key in ("geo_dir", "cons_dir", "rest_path", "fascia_path"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(Path.cwd() / cfg[key])
    return cfg
