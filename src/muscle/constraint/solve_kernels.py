"""Constraint-solve Warp kernels for the XPBD muscle solver.

Holds every ``solve_*`` constraint kernel plus the ``@wp.func`` device helpers
and structs they (and a few host-side kernels in ``engine.py``) build on.
Split out of ``engine.py``; depends only on ``muscle.defs`` and ``muscle.fiber``
so the layering stays acyclic (engine imports this, never the reverse).
"""

from __future__ import annotations

import warp as wp

from muscle.fiber.dgf import (
    dgf_active_force_length_wp,
    dgf_passive_force_length_wp,
    dgf_force_velocity_wp,
)
from muscle.fiber.millard import (
    millard_eval_wp,
    millard_energy_eval_wp,
)
from muscle.defs import ConstraintType, FemFlag


@wp.struct
class SVDResult:
    U: wp.mat33
    sigma: wp.vec3
    V: wp.mat33

@wp.struct
class PolarResult:
    S: wp.mat33
    R: wp.mat33

@wp.func
def get_L_component(L: wp.vec3, loff: int) -> float:
    return wp.where(loff == 0, L[0], wp.where(loff == 1, L[1], L[2]))

@wp.func
def set_L_component(L: wp.vec3, loff: int, val: float) -> wp.vec3:
    return wp.vec3(
        wp.where(loff == 0, val, L[0]),
        wp.where(loff == 1, val, L[1]),
        wp.where(loff == 2, val, L[2]))


# ---------------------------------------------------------------------------
# Math @wp.func
# ---------------------------------------------------------------------------

@wp.func
def update_dP(dP: wp.array(dtype=wp.vec3), dPw: wp.array(dtype=wp.float32),
              dp: wp.vec3, pt: int):
    wp.atomic_add(dP, pt, dp)
    wp.atomic_add(dPw, pt, 1.0)

@wp.func
def fem_flags_fn(ctype: int) -> int:
    flags = 0
    if ctype == ConstraintType.TETARAP or ctype == ConstraintType.TETARAPNORM:
        flags = flags | FemFlag.LINEARENERGY
    if ctype == ConstraintType.TETARAPNORM or ctype == ConstraintType.TETFIBERNORM or ctype == ConstraintType.TETFIBERDGF:
        flags = flags | FemFlag.NORMSTIFFNESS
    return flags

@wp.func
def project_to_line_fn(p: wp.vec3, orig: wp.vec3, direction: wp.vec3) -> wp.vec3:
    return orig + direction * wp.dot(p - orig, direction)

@wp.func
def get_inv_mass_fn(idx: int, mass: wp.array(dtype=wp.float32),
                    stopped: wp.array(dtype=wp.int32)) -> float:
    res = float(0.0)
    if stopped[idx] != 0:
        res = 0.0
    else:
        m = mass[idx]
        res = wp.where(m > 0.0, 1.0 / m, 0.0)
    return res




@wp.func
def ssvd_fn(F: wp.mat33) -> SVDResult:
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sigma = wp.vec3(0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wp.svd3(F, U, sigma, V)

    detU = wp.determinant(U)
    if detU < 0.0:
        U = wp.mat33(
            U[0, 0], U[0, 1], -U[0, 2],
            U[1, 0], U[1, 1], -U[1, 2],
            U[2, 0], U[2, 1], -U[2, 2])
        sigma = wp.vec3(sigma[0], sigma[1], -sigma[2])

    detV = wp.determinant(V)
    if detV < 0.0:
        V = wp.mat33(
            V[0, 0], V[0, 1], -V[0, 2],
            V[1, 0], V[1, 1], -V[1, 2],
            V[2, 0], V[2, 1], -V[2, 2])
        sigma = wp.vec3(sigma[0], sigma[1], -sigma[2])

    result = SVDResult()
    result.U = U
    result.sigma = sigma
    result.V = V
    return result


@wp.func
def polar_decomposition_fn(F: wp.mat33) -> PolarResult:
    svd = ssvd_fn(F)
    U = svd.U
    sigma_vec = svd.sigma
    V = svd.V

    R = U * wp.transpose(V)
    sig_mat = wp.mat33(
        sigma_vec[0], 0.0, 0.0,
        0.0, sigma_vec[1], 0.0,
        0.0, 0.0, sigma_vec[2])
    S = V * sig_mat * wp.transpose(V)

    result = PolarResult()
    result.S = S
    result.R = R
    return result

@wp.func
def squared_norm3_fn(a: wp.mat33) -> float:
    s = float(0.0)
    for i in range(3):
        for j in range(3):
            s = s + a[i, j] * a[i, j]
    return s

@wp.func
def mat3_to_quat_fn(m: wp.mat33) -> wp.vec4f:
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    qw = float(0.0)
    qx = float(0.0)
    qy = float(0.0)
    qz = float(0.0)
    if trace > 0.0:
        s = wp.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = wp.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = wp.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = wp.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return wp.vec4f(qx, qy, qz, qw)


@wp.func
def transfer_tension_fn(muscletension: float, tendonmask: float,
                        minfiberscale: float, maxfiberscale: float) -> float:
    fiberscale = minfiberscale + (1.0 - tendonmask) * muscletension * (maxfiberscale - minfiberscale)
    return fiberscale


# ---------------------------------------------------------------------------
# Constraint solver @wp.func
# ---------------------------------------------------------------------------

@wp.func
def tet_volume_update_xpbd_fn(
    use_jacobi: int,
    cidx: int,
    c_L: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    restlength: float,
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    tetid: int,
    pts: wp.vec4i,
    dt: float,
    stiffness: float,
    kstiffcompress: float,
    kdampratio: float,
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    has_compress_stiffness: int,
):
    inv_mass0 = get_inv_mass_fn(pts[0], mass, stopped)
    inv_mass1 = get_inv_mass_fn(pts[1], mass, stopped)
    inv_mass2 = get_inv_mass_fn(pts[2], mass, stopped)
    inv_mass3 = get_inv_mass_fn(pts[3], mass, stopped)

    p0 = pos[pts[0]]
    p1 = pos[pts[1]]
    p2 = pos[pts[2]]
    p3 = pos[pts[3]]

    # pts[3] is the reference vertex (matches precompute_rest_kernel and
    # the ARAP/fiber kernels). With d_k = p_k - p_3 for k=0,1,2,
    # signed volume V = det[d_0 | d_1 | d_2] / 6, so the per-vertex gradients
    # follow from d/dd_k det = (cross of the other two), and the reference
    # vertex grad_3 = -(grad_0 + grad_1 + grad_2).
    d0 = p0 - p3
    d1 = p1 - p3
    d2 = p2 - p3
    grad0 = wp.cross(d1, d2) / 6.0
    grad1 = wp.cross(d2, d0) / 6.0
    grad2 = wp.cross(d0, d1) / 6.0
    grad3 = -(grad0 + grad1 + grad2)

    w_sum = (inv_mass0 * wp.length_sq(grad0) +
             inv_mass1 * wp.length_sq(grad1) +
             inv_mass2 * wp.length_sq(grad2) +
             inv_mass3 * wp.length_sq(grad3))

    if w_sum > 1e-9:
        volume = wp.dot(wp.cross(d0, d1), d2) / 6.0

        comp = 0
        if has_compress_stiffness != 0:
            if volume < restlength:
                comp = 1
        kstiff = wp.where(comp != 0, kstiffcompress, stiffness)
        loff = wp.min(comp, 2)

        if kstiff != 0.0:
            l = get_L_component(c_L[cidx], loff)

            alpha = 1.0 / kstiff
            alpha = alpha / (dt * dt)

            C = volume - restlength

            dsum = float(0.0)
            gamma = float(1.0)
            if kdampratio > 0.0:
                prev0 = pprev[pts[0]]
                prev1 = pprev[pts[1]]
                prev2 = pprev[pts[2]]
                prev3 = pprev[pts[3]]
                beta = kstiff * kdampratio * dt * dt
                gamma = alpha * beta / dt
                dsum = (wp.dot(grad0, pos[pts[0]] - prev0) +
                        wp.dot(grad1, pos[pts[1]] - prev1) +
                        wp.dot(grad2, pos[pts[2]] - prev2) +
                        wp.dot(grad3, pos[pts[3]] - prev3))
                dsum = dsum * gamma
                gamma = gamma + 1.0

            dlambda = (-C - alpha * l - dsum) / (gamma * w_sum + alpha)

            if use_jacobi != 0:
                update_dP(dP, dPw, dlambda * inv_mass0 * grad0, pts[0])
                update_dP(dP, dPw, dlambda * inv_mass1 * grad1, pts[1])
                update_dP(dP, dPw, dlambda * inv_mass2 * grad2, pts[2])
                update_dP(dP, dPw, dlambda * inv_mass3 * grad3, pts[3])
            else:
                pos[pts[0]] = pos[pts[0]] + dlambda * inv_mass0 * grad0
                pos[pts[1]] = pos[pts[1]] + dlambda * inv_mass1 * grad1
                pos[pts[2]] = pos[pts[2]] + dlambda * inv_mass2 * grad2
                pos[pts[3]] = pos[pts[3]] + dlambda * inv_mass3 * grad3
                c_L[cidx] = set_L_component(c_L[cidx], loff, get_L_component(c_L[cidx], loff) + dlambda)


@wp.func
def tet_fiber_update_xpbd_fn(
    use_jacobi: int,
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    cidx: int,
    c_L: wp.array(dtype=wp.vec3),
    pts: wp.vec4i,
    dt: float,
    fiber: wp.vec3,
    stiffness: float,
    Dminv: wp.mat33,
    kdampratio: float,
    restlength: float,
    restvector: wp.vec4f,
    acti: float,
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    target_stretch: float,
):
    inv_mass0 = get_inv_mass_fn(pts[0], mass, stopped)
    inv_mass1 = get_inv_mass_fn(pts[1], mass, stopped)
    inv_mass2 = get_inv_mass_fn(pts[2], mass, stopped)
    inv_mass3 = get_inv_mass_fn(pts[3], mass, stopped)

    l = c_L[cidx][0]
    alpha = 1.0 / stiffness
    alpha = alpha / restlength
    alpha = alpha / (dt * dt)
    grad_scale = float(1.0)
    psi = float(0.0)

    c0 = pos[pts[0]] - pos[pts[3]]
    c1 = pos[pts[1]] - pos[pts[3]]
    c2 = pos[pts[2]] - pos[pts[3]]
    _Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])

    wTDminvT = wp.vec3(restvector[0], restvector[1], restvector[2])
    FwT = _Ds * wTDminvT
    psi = 0.5 * wp.length_sq(FwT)

    if psi > 1e-9:
        psi_sqrt = wp.sqrt(2.0 * psi)
        grad_scale = 1.0 / psi_sqrt
        psi = psi_sqrt

    Ht = wp.outer(wTDminvT, FwT)

    grad0 = grad_scale * wp.vec3(Ht[0, 0], Ht[0, 1], Ht[0, 2])
    grad1 = grad_scale * wp.vec3(Ht[1, 0], Ht[1, 1], Ht[1, 2])
    grad2 = grad_scale * wp.vec3(Ht[2, 0], Ht[2, 1], Ht[2, 2])
    grad3 = -grad0 - grad1 - grad2

    w_sum = (inv_mass0 * wp.length_sq(grad0) +
             inv_mass1 * wp.length_sq(grad1) +
             inv_mass2 * wp.length_sq(grad2) +
             inv_mass3 * wp.length_sq(grad3))

    if w_sum > 1e-9:
        dsum = float(0.0)
        gamma = float(1.0)
        if kdampratio > 0.0:
            beta = stiffness * kdampratio * dt * dt
            beta = beta * restlength
            gamma = alpha * beta / dt

            dsum = (wp.dot(grad0, pos[pts[0]] - pprev[pts[0]]) +
                    wp.dot(grad1, pos[pts[1]] - pprev[pts[1]]) +
                    wp.dot(grad2, pos[pts[2]] - pprev[pts[2]]) +
                    wp.dot(grad3, pos[pts[3]] - pprev[pts[3]]))
            dsum = dsum * gamma
            gamma = gamma + 1.0

        C = psi - target_stretch
        dL = (-C - alpha * l - dsum) / (gamma * w_sum + alpha)
        if use_jacobi != 0:
            update_dP(dP, dPw, dL * inv_mass0 * grad0, pts[0])
            update_dP(dP, dPw, dL * inv_mass1 * grad1, pts[1])
            update_dP(dP, dPw, dL * inv_mass2 * grad2, pts[2])
            update_dP(dP, dPw, dL * inv_mass3 * grad3, pts[3])
        else:
            pos[pts[0]] = pos[pts[0]] + dL * inv_mass0 * grad0
            pos[pts[1]] = pos[pts[1]] + dL * inv_mass1 * grad1
            pos[pts[2]] = pos[pts[2]] + dL * inv_mass2 * grad2
            pos[pts[3]] = pos[pts[3]] + dL * inv_mass3 * grad3
            c_L[cidx] = wp.vec3(c_L[cidx][0] + dL, c_L[cidx][1], c_L[cidx][2])


@wp.func
def distance_pos_update_xpbd_fn(
    use_jacobi: int,
    cidx: int,
    c_L: wp.array(dtype=wp.vec3),
    pt0: int,
    pt1: int,
    p0: wp.vec3,
    p1: wp.vec3,
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    restlength: float,
    kstiff: float,
    kdampratio: float,
    kstiffcompress: float,
    dt: float,
    has_compress_stiffness: int,
):
    invmass0 = float(0.0)
    invmass1 = get_inv_mass_fn(pt1, mass, stopped)

    if pt0 >= 0:
        invmass0 = get_inv_mass_fn(pt0, mass, stopped)

    wsum = invmass0 + invmass1

    if wsum != 0.0:
        p1_current = pos[pt1]
        p0_current = p0
        if pt0 >= 0:
            p0_current = pos[pt0]

        n = p1_current - p0_current
        d = wp.length(n)
        if d >= 1e-6:
            comp = 0
            if has_compress_stiffness != 0:
                if d < restlength:
                    comp = 1
            kstiff_val = wp.where(comp != 0, kstiffcompress, kstiff)
            loff = wp.min(comp, 2)
            if kstiff_val != 0.0:
                l = get_L_component(c_L[cidx], loff)

                alpha = 1.0 / kstiff_val
                alpha = alpha / (dt * dt)

                C = d - restlength
                n = n / d
                gradC = n

                dsum = float(0.0)
                gamma = float(1.0)
                if kdampratio > 0.0:
                    if pt0 >= 0:
                        prev0 = pprev[pt0]
                        prev1 = pprev[pt1]
                        beta = kstiff_val * kdampratio * dt * dt
                        gamma = alpha * beta / dt
                        dsum = gamma * (-wp.dot(gradC, p0_current - prev0) + wp.dot(gradC, p1_current - prev1))
                    else:
                        prev1 = pprev[pt1]
                        beta = kstiff_val * kdampratio * dt * dt
                        gamma = alpha * beta / dt
                        dsum = gamma * wp.dot(gradC, p1_current - prev1)
                    gamma = gamma + 1.0

                dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                dp = n * (-dL)

                if use_jacobi != 0:
                    if pt0 >= 0:
                        update_dP(dP, dPw, invmass0 * dp, pt0)
                    update_dP(dP, dPw, -invmass1 * dp, pt1)
                else:
                    if pt0 >= 0:
                        pos[pt0] = pos[pt0] + invmass0 * dp
                    pos[pt1] = pos[pt1] - invmass1 * dp
                    c_L[cidx] = set_L_component(c_L[cidx], loff, get_L_component(c_L[cidx], loff) + dL)


@wp.func
def attach_bilateral_update_fn(
    use_jacobi: int,
    cidx: int,
    c_L: wp.array(dtype=wp.vec3),
    pt1: int,
    p0: wp.vec3,
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    restlength: float,
    kstiff: float,
    kdampratio: float,
    kstiffcompress: float,
    dt: float,
    has_compress_stiffness: int,
    reaction_accum: wp.array(dtype=wp.vec3),
):
    invmass1 = get_inv_mass_fn(pt1, mass, stopped)
    wsum = invmass1

    if wsum != 0.0:
        p1_current = pos[pt1]
        p0_current = p0

        n = p1_current - p0_current
        d = wp.length(n)
        if d >= 1e-6:
            comp = 0
            if has_compress_stiffness != 0:
                if d < restlength:
                    comp = 1
            kstiff_val = wp.where(comp != 0, kstiffcompress, kstiff)
            loff = wp.min(comp, 2)
            if kstiff_val != 0.0:
                l = get_L_component(c_L[cidx], loff)

                alpha = 1.0 / kstiff_val
                alpha = alpha / (dt * dt)

                C = d - restlength
                n = n / d

                dsum = float(0.0)
                gamma = float(1.0)
                if kdampratio > 0.0:
                    prev1 = pprev[pt1]
                    beta = kstiff_val * kdampratio * dt * dt
                    gamma = alpha * beta / dt
                    dsum = gamma * wp.dot(n, p1_current - prev1)
                    gamma = gamma + 1.0

                dL = (-C - alpha * l - dsum) / (gamma * wsum + alpha)
                dp = n * (-dL)

                if use_jacobi != 0:
                    update_dP(dP, dPw, -invmass1 * dp, pt1)
                else:
                    pos[pt1] = pos[pt1] - invmass1 * dp
                    c_L[cidx] = set_L_component(c_L[cidx], loff, get_L_component(c_L[cidx], loff) + dL)

                wp.atomic_add(reaction_accum, cidx, C * n)


# ---------------------------------------------------------------------------
# Simulation @wp.kernel
# ---------------------------------------------------------------------------

@wp.func
def mat33_inverse_fn(M: wp.mat33) -> wp.mat33:
    """Compute 3x3 matrix inverse via cofactor expansion."""
    d = wp.determinant(M)
    if wp.abs(d) < 1.0e-30:
        return wp.identity(n=3, dtype=float)
    inv_d = 1.0 / d
    return wp.mat33(
        (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]) * inv_d,
        (M[0, 2] * M[2, 1] - M[0, 1] * M[2, 2]) * inv_d,
        (M[0, 1] * M[1, 2] - M[0, 2] * M[1, 1]) * inv_d,
        (M[1, 2] * M[2, 0] - M[1, 0] * M[2, 2]) * inv_d,
        (M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0]) * inv_d,
        (M[0, 2] * M[1, 0] - M[0, 0] * M[1, 2]) * inv_d,
        (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]) * inv_d,
        (M[0, 1] * M[2, 0] - M[0, 0] * M[2, 1]) * inv_d,
        (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) * inv_d,
    )


@wp.kernel
def solve_tetvolume_kernel(
    c_stiffness: wp.array(dtype=wp.float32),
    c_compressionstiffness: wp.array(dtype=wp.float32),
    c_pts: wp.array(dtype=wp.vec4i),
    c_tetid: wp.array(dtype=wp.int32),
    c_restlength: wp.array(dtype=wp.float32),
    c_dampingratio: wp.array(dtype=wp.float32),
    c_L: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    dt: float,
    use_jacobi: int,
    has_compress_stiffness: int,
    offset: int,
):
    c = offset + wp.tid()
    cstiffness = c_stiffness[c]
    if cstiffness <= 0.0:
        return
    kstiffcompress = c_compressionstiffness[c]
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, cstiffness)
    pts = c_pts[c]
    tetid = c_tetid[c]
    tet_volume_update_xpbd_fn(
        use_jacobi, c, c_L, pos, pprev,
        c_restlength[c], dP, dPw, tetid, pts,
        dt, cstiffness, kstiffcompress, c_dampingratio[c],
        mass, stopped, has_compress_stiffness)


@wp.kernel
def solve_tetfibernorm_kernel(
    c_stiffness: wp.array(dtype=wp.float32),
    c_pts: wp.array(dtype=wp.vec4i),
    c_tetid: wp.array(dtype=wp.int32),
    c_dampingratio: wp.array(dtype=wp.float32),
    c_restlength: wp.array(dtype=wp.float32),
    c_restvector: wp.array(dtype=wp.vec4f),
    c_L: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    rest_matrix: wp.array(dtype=wp.mat33),
    v_fiber_dir: wp.array(dtype=wp.vec3),
    activation: wp.array(dtype=wp.float32),
    tendonmask: wp.array(dtype=wp.float32),
    dt: float,
    use_jacobi: int,
    contraction_ratio: float,
    fiber_stiffness_scale: float,
    offset: int,
):
    c = offset + wp.tid()
    cstiffness = c_stiffness[c]
    if cstiffness <= 0.0:
        return
    pts = c_pts[c]
    tetid = c_tetid[c]
    fiber_dir = (v_fiber_dir[pts[0]] + v_fiber_dir[pts[1]] +
                 v_fiber_dir[pts[2]] + v_fiber_dir[pts[3]]) / 4.0
    Dminv = rest_matrix[tetid]
    acti = activation[tetid]
    _tendonmask = tendonmask[tetid]
    belly_factor = 1.0 - _tendonmask

    fiberscale = transfer_tension_fn(acti, _tendonmask, 0.0, 10.0)
    stiffness_val = cstiffness * fiberscale * fiber_stiffness_scale
    if stiffness_val > 0.0:
        target_stretch = 1.0 - belly_factor * acti * contraction_ratio

        tet_fiber_update_xpbd_fn(
            use_jacobi, pos, pprev, dP, dPw,
            c, c_L, pts, dt, fiber_dir, stiffness_val, Dminv,
            c_dampingratio[c], c_restlength[c], c_restvector[c],
            acti, mass, stopped, target_stretch)


@wp.kernel
def solve_tetfiberdgf_kernel(
    c_stiffness: wp.array(dtype=wp.float32),
    c_pts: wp.array(dtype=wp.vec4i),
    c_tetid: wp.array(dtype=wp.int32),
    c_restlength: wp.array(dtype=wp.float32),
    c_restvector: wp.array(dtype=wp.vec4f),
    c_restdir: wp.array(dtype=wp.vec3),
    c_dampingratio: wp.array(dtype=wp.float32),
    c_L: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    rest_matrix: wp.array(dtype=wp.mat33),
    v_fiber_dir: wp.array(dtype=wp.vec3),
    activation: wp.array(dtype=wp.float32),
    tendonmask: wp.array(dtype=wp.float32),
    dt: float,
    use_jacobi: int,
    fiber_stiffness_scale: float,
    offset: int,
):
    """XPBD fiber constraint with DGF force-length curves (quasi-static).

    Uses DGF active f_L(lm_tilde) and passive f_PE(lm_tilde) to modulate
    constraint stiffness. target_stretch = 0 so equilibrium is naturally
    determined by where DGF contractile force equals external load.
    No force-velocity curve (quasi-static assumption).
    """
    c = offset + wp.tid()
    cstiffness = c_stiffness[c]
    if cstiffness <= 0.0:
        return
    pts = c_pts[c]
    tetid = c_tetid[c]
    fiber_dir = (v_fiber_dir[pts[0]] + v_fiber_dir[pts[1]] +
                 v_fiber_dir[pts[2]] + v_fiber_dir[pts[3]]) / 4.0
    Dminv = rest_matrix[tetid]
    acti = activation[tetid]

    # Compute current fiber stretch from deformation gradient
    c0 = pos[pts[0]] - pos[pts[3]]
    c1 = pos[pts[1]] - pos[pts[3]]
    c2 = pos[pts[2]] - pos[pts[3]]
    _Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])
    wTDminvT = wp.vec3(c_restvector[c][0], c_restvector[c][1], c_restvector[c][2])
    FwT = _Ds * wTDminvT
    lm_tilde = wp.sqrt(wp.max(wp.length_sq(FwT), 1.0e-12))

    # DGF force-length: stiffness modulated by f_L(lm) for correct force profile.
    # restdir[0] = sigma0 (peak isometric stress)
    # restdir[1] = contraction_factor (from DGF curve inversion, set per-step)
    sigma0 = c_restdir[c][0]
    contraction_factor = c_restdir[c][1]
    f_L_val = dgf_active_force_length_wp(lm_tilde)
    f_PE_val = dgf_passive_force_length_wp(lm_tilde)
    f_total = acti * f_L_val + f_PE_val
    # Stiffness follows DGF curve; fiber_stiffness_scale calibrates magnitude
    fiberscale = wp.max(f_total, 0.01)
    stiffness_val = cstiffness * fiberscale * fiber_stiffness_scale

    if stiffness_val > 0.0:
        # Target from DGF equilibrium: lm_eq ≈ 1 - a * contraction_factor
        target_stretch = 1.0 - acti * contraction_factor

        tet_fiber_update_xpbd_fn(
            use_jacobi, pos, pprev, dP, dPw,
            c, c_L, pts, dt, fiber_dir, stiffness_val, Dminv,
            c_dampingratio[c], c_restlength[c], c_restvector[c],
            acti, mass, stopped, target_stretch)


@wp.kernel
def solve_tetfibermillard_kernel(
    c_stiffness: wp.array(dtype=wp.float32),
    c_pts: wp.array(dtype=wp.vec4i),
    c_tetid: wp.array(dtype=wp.int32),
    c_restlength: wp.array(dtype=wp.float32),
    c_restvector: wp.array(dtype=wp.vec4f),
    c_restdir: wp.array(dtype=wp.vec3),
    c_dampingratio: wp.array(dtype=wp.float32),
    c_L: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    rest_matrix: wp.array(dtype=wp.mat33),
    v_fiber_dir: wp.array(dtype=wp.vec3),
    activation: wp.array(dtype=wp.float32),
    tendonmask: wp.array(dtype=wp.float32),
    dt: float,
    use_jacobi: int,
    millard_stiffness_scale: float,
    offset: int,
    # Millard active f_L curve data (force)
    fl_x_coeffs: wp.array(dtype=wp.float32),
    fl_y_coeffs: wp.array(dtype=wp.float32),
    fl_seg_bounds: wp.array(dtype=wp.float32),
    fl_n_segments: int,
    fl_x_lo: float, fl_x_hi: float,
    fl_y_lo: float, fl_y_hi: float,
    fl_dydx_lo: float, fl_dydx_hi: float,
    # Millard active f_L curve data (energy)
    fl_F_coeffs: wp.array(dtype=wp.float32),
    fl_energy_offset: float,
    # Millard passive f_PE curve data (unused for now, kept for API compat)
    fpe_x_coeffs: wp.array(dtype=wp.float32),
    fpe_y_coeffs: wp.array(dtype=wp.float32),
    fpe_seg_bounds: wp.array(dtype=wp.float32),
    fpe_n_segments: int,
    fpe_x_lo: float, fpe_x_hi: float,
    fpe_y_lo: float, fpe_y_hi: float,
    fpe_dydx_lo: float, fpe_dydx_hi: float,
    fiber_skip_detF: float,
):
    """XPBD fiber constraint with energy-based Millard 2012 formulation.

    C = sqrt(2 * sigma0 * a * Psi_L(lm))
    gradC_i = (dC/dlm) * (dlm/dx_i)
    where dC/dlm = sigma0 * a * f_L / C
    """
    c = offset + wp.tid()
    cstiffness = c_stiffness[c]
    if cstiffness <= 0.0:
        return
    pts = c_pts[c]
    tetid = c_tetid[c]
    acti = activation[tetid]
    sigma0 = c_restdir[c][0]

    # Active force only in the muscle belly: scale by belly_factor = 1 - tendonmask
    # (matches solve_tetfibernorm_kernel). Without this, tendon-end tets get full
    # active stress and over-contract against the rigid end attach -> distortion.
    acti = acti * (1.0 - tendonmask[tetid])

    # Skip if no activation (no energy → C=0)
    if acti < 1.0e-6:
        return

    # ── Deformation gradient: lm and dlm/dx_i ──
    c0 = pos[pts[0]] - pos[pts[3]]
    c1 = pos[pts[1]] - pos[pts[3]]
    c2 = pos[pts[2]] - pos[pts[3]]
    _Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])

    # Skip fiber solve for near-inverted tets to prevent further distortion
    if fiber_skip_detF > 0.0:
        F_full = _Ds * rest_matrix[tetid]
        J = wp.determinant(F_full)
        if J < fiber_skip_detF:
            return

    wTDminvT = wp.vec3(c_restvector[c][0], c_restvector[c][1], c_restvector[c][2])
    FwT = _Ds * wTDminvT  # = F * w (fiber direction in deformed space)
    lm_sq = wp.length_sq(FwT)
    if lm_sq < 1.0e-12:
        return
    lm = wp.sqrt(lm_sq)

    # ── Evaluate Millard energy and force ──
    psi_L = millard_energy_eval_wp(lm,
        fl_x_coeffs, fl_F_coeffs, fl_seg_bounds, fl_n_segments,
        fl_x_lo, fl_x_hi, fl_y_lo, fl_y_hi, fl_dydx_lo, fl_dydx_hi)
    psi_L = psi_L + fl_energy_offset  # shift so Psi > 0 for all physical lm

    f_L_val = millard_eval_wp(lm,
        fl_x_coeffs, fl_y_coeffs, fl_seg_bounds, fl_n_segments,
        fl_x_lo, fl_x_hi, fl_y_lo, fl_y_hi, fl_dydx_lo, fl_dydx_hi)

    # ── Energy constraint: C = sqrt(2 * sigma0 * a * (Psi_L + offset)) ──
    C_energy = wp.sqrt(2.0 * sigma0 * acti * wp.max(psi_L, 1.0e-20))
    dC_dlm = sigma0 * acti * f_L_val / C_energy

    # ── Geometric gradients: dlm/dx_i ──
    # Ht = wTDminvT ⊗ FwT, grad_lm_i = (1/lm) * row_i(Ht)
    inv_lm = 1.0 / lm
    Ht = wp.outer(wTDminvT, FwT)
    grad_lm0 = inv_lm * wp.vec3(Ht[0, 0], Ht[0, 1], Ht[0, 2])
    grad_lm1 = inv_lm * wp.vec3(Ht[1, 0], Ht[1, 1], Ht[1, 2])
    grad_lm2 = inv_lm * wp.vec3(Ht[2, 0], Ht[2, 1], Ht[2, 2])
    grad_lm3 = -grad_lm0 - grad_lm1 - grad_lm2

    # ── Chain rule: gradC_i = dC/dlm * dlm/dx_i ──
    grad0 = dC_dlm * grad_lm0
    grad1 = dC_dlm * grad_lm1
    grad2 = dC_dlm * grad_lm2
    grad3 = dC_dlm * grad_lm3

    # ── XPBD update ──
    inv_mass0 = get_inv_mass_fn(pts[0], mass, stopped)
    inv_mass1 = get_inv_mass_fn(pts[1], mass, stopped)
    inv_mass2 = get_inv_mass_fn(pts[2], mass, stopped)
    inv_mass3 = get_inv_mass_fn(pts[3], mass, stopped)

    w_sum = (inv_mass0 * wp.length_sq(grad0) +
             inv_mass1 * wp.length_sq(grad1) +
             inv_mass2 * wp.length_sq(grad2) +
             inv_mass3 * wp.length_sq(grad3))

    if w_sum < 1.0e-9:
        return

    # Compliance: alpha = 1/(k * V0 * dt^2)
    # Uniform stiffness (no tendonmask scaling): softening tendon-junction tets made
    # them floppy and over-distort, so keep full stiffness everywhere. Active FORCE is
    # already belly-only via acti *= (1 - tendonmask) above.
    stiffness_val = cstiffness * millard_stiffness_scale
    restlength = c_restlength[c]  # V0 (tet volume)
    alpha = 1.0 / stiffness_val
    alpha = alpha / restlength
    alpha = alpha / (dt * dt)

    l = c_L[c][0]  # accumulated Lagrange multiplier
    kdampratio = c_dampingratio[c]

    # Damping
    dsum = float(0.0)
    gamma = float(1.0)
    if kdampratio > 0.0:
        beta = stiffness_val * kdampratio * dt * dt
        beta = beta * restlength
        gamma = alpha * beta / dt
        dsum = (wp.dot(grad0, pos[pts[0]] - pprev[pts[0]]) +
                wp.dot(grad1, pos[pts[1]] - pprev[pts[1]]) +
                wp.dot(grad2, pos[pts[2]] - pprev[pts[2]]) +
                wp.dot(grad3, pos[pts[3]] - pprev[pts[3]]))
        dsum = dsum * gamma
        gamma = gamma + 1.0

    dL = (-C_energy - alpha * l - dsum) / (gamma * w_sum + alpha)
    if use_jacobi != 0:
        update_dP(dP, dPw, dL * inv_mass0 * grad0, pts[0])
        update_dP(dP, dPw, dL * inv_mass1 * grad1, pts[1])
        update_dP(dP, dPw, dL * inv_mass2 * grad2, pts[2])
        update_dP(dP, dPw, dL * inv_mass3 * grad3, pts[3])
    else:
        pos[pts[0]] = pos[pts[0]] + dL * inv_mass0 * grad0
        pos[pts[1]] = pos[pts[1]] + dL * inv_mass1 * grad1
        pos[pts[2]] = pos[pts[2]] + dL * inv_mass2 * grad2
        pos[pts[3]] = pos[pts[3]] + dL * inv_mass3 * grad3
        c_L[c] = wp.vec3(c_L[c][0] + dL, c_L[c][1], c_L[c][2])


@wp.kernel
def solve_attach_kernel(
    c_stiffness: wp.array(dtype=wp.float32),
    c_compressionstiffness: wp.array(dtype=wp.float32),
    c_pts: wp.array(dtype=wp.vec4i),
    c_restlength: wp.array(dtype=wp.float32),
    c_restvector: wp.array(dtype=wp.vec4f),
    c_dampingratio: wp.array(dtype=wp.float32),
    c_L: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    reaction_accum: wp.array(dtype=wp.vec3),
    dt: float,
    use_jacobi: int,
    has_compress_stiffness: int,
    offset: int,
):
    c = offset + wp.tid()
    cstiffness = c_stiffness[c]
    if cstiffness <= 0.0:
        return
    kstiffcompress = c_compressionstiffness[c]
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, cstiffness)
    pts = c_pts[c]
    pt_src = pts[0]
    rv = c_restvector[c]
    p0_target = wp.vec3(rv[0], rv[1], rv[2])
    attach_bilateral_update_fn(
        use_jacobi, c, c_L, pt_src, p0_target,
        pos, pprev, dP, dPw, mass, stopped,
        c_restlength[c], cstiffness, c_dampingratio[c],
        kstiffcompress, dt, has_compress_stiffness, reaction_accum)


@wp.kernel
def solve_pin_kernel(
    c_stiffness: wp.array(dtype=wp.float32),
    c_compressionstiffness: wp.array(dtype=wp.float32),
    c_pts: wp.array(dtype=wp.vec4i),
    c_restlength: wp.array(dtype=wp.float32),
    c_restvector: wp.array(dtype=wp.vec4f),
    c_dampingratio: wp.array(dtype=wp.float32),
    c_L: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    dt: float,
    use_jacobi: int,
    has_compress_stiffness: int,
    offset: int,
):
    c = offset + wp.tid()
    cstiffness = c_stiffness[c]
    if cstiffness <= 0.0:
        return
    kstiffcompress = c_compressionstiffness[c]
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, cstiffness)
    pts = c_pts[c]
    pt_src = pts[0]
    rv = c_restvector[c]
    p0_target = wp.vec3(rv[0], rv[1], rv[2])
    p1_src = pos[pt_src]
    distance_pos_update_xpbd_fn(
        use_jacobi, c, c_L, -1, pt_src,
        p0_target, p1_src,
        pos, pprev, dP, dPw, mass, stopped,
        c_restlength[c], cstiffness, c_dampingratio[c],
        kstiffcompress, dt, has_compress_stiffness)


@wp.kernel
def solve_pintotarget_kernel(
    c_pts: wp.array(dtype=wp.vec4i),
    c_restvector: wp.array(dtype=wp.vec4f),
    c_restdir: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    offset: int,
):
    """Hard pin: pos[src] = bone_target + rest_offset (no XPBD, direct overwrite).

    restvector = live bone_pos[tgt] (refreshed by update_attach_targets_kernel).
    restdir    = rest_offset = pos0[src] - bone_pos0[tgt] (captured at build).
    Each src appears in at most one pintotarget constraint, so the parallel
    overwrites do not race.
    """
    c = offset + wp.tid()
    pts = c_pts[c]
    pt_src = pts[0]
    if pt_src < 0:
        return
    rv = c_restvector[c]
    rd = c_restdir[c]
    pos[pt_src] = wp.vec3(rv[0] + rd[0], rv[1] + rd[1], rv[2] + rd[2])


@wp.kernel
def solve_distanceline_kernel(
    c_stiffness: wp.array(dtype=wp.float32),
    c_compressionstiffness: wp.array(dtype=wp.float32),
    c_pts: wp.array(dtype=wp.vec4i),
    c_restlength: wp.array(dtype=wp.float32),
    c_restvector: wp.array(dtype=wp.vec4f),
    c_restdir: wp.array(dtype=wp.vec3),
    c_dampingratio: wp.array(dtype=wp.float32),
    c_L: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    dt: float,
    use_jacobi: int,
    has_compress_stiffness: int,
    offset: int,
):
    c = offset + wp.tid()
    cstiffness = c_stiffness[c]
    if cstiffness <= 0.0:
        return
    kstiffcompress = c_compressionstiffness[c]
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, cstiffness)
    pts = c_pts[c]
    pt_src = pts[0]
    p_src = pos[pt_src]
    rv = c_restvector[c]
    line_origin = wp.vec3(rv[0], rv[1], rv[2])
    line_dir = c_restdir[c]

    p_projected = project_to_line_fn(p_src, line_origin, line_dir)

    distance_pos_update_xpbd_fn(
        use_jacobi, c, c_L, -1, pt_src,
        p_projected, p_src,
        pos, pprev, dP, dPw, mass, stopped,
        c_restlength[c], cstiffness, c_dampingratio[c],
        kstiffcompress, dt, has_compress_stiffness)


# Polar-decomposition ARAP (RLMuscle parity).
# C = ||F - R||_F where R = polar(F).rotation. With FemFlag.LINEARENERGY flag (set
# for ConstraintType.TETARAP via fem_flags_fn), C is the linear norm and gradscale = 1/psi.
# Only penalises non-rotational deformation, so fiber contraction stays free
# (unlike the SVD form which also locks volume + isotropy).
@wp.func
def tet_arap_update_xpbd_fn(
    use_jacobi: int,
    cidx: int,
    c_restvector: wp.array(dtype=wp.vec4f),
    c_L: wp.array(dtype=wp.vec3),
    pts: wp.vec4i,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    restlength: float,
    restmatrix: wp.mat33,
    kstiff: float,
    kdampratio: float,
    flags: int,
):
    pt0 = pts[0]
    pt1 = pts[1]
    pt2 = pts[2]
    pt3 = pts[3]
    p0 = pos[pt0]
    p1 = pos[pt1]
    p2 = pos[pt2]
    p3 = pos[pt3]

    invmass0 = get_inv_mass_fn(pt0, mass, stopped)
    invmass1 = get_inv_mass_fn(pt1, mass, stopped)
    invmass2 = get_inv_mass_fn(pt2, mass, stopped)
    invmass3 = get_inv_mass_fn(pt3, mass, stopped)

    c0 = p0 - p3
    c1 = p1 - p3
    c2 = p2 - p3
    Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])

    F = Ds * restmatrix

    polar = polar_decomposition_fn(F)
    R = polar.R
    d = F - R

    c_restvector[cidx] = mat3_to_quat_fn(R)

    psi = squared_norm3_fn(d)
    gradscale = float(2.0)
    if (flags & (FemFlag.LINEARENERGY | FemFlag.NORMSTIFFNESS)) != 0:
        psi = wp.sqrt(psi)
        gradscale = 1.0 / psi

    if psi >= 1e-6:
        Ht = restmatrix * wp.transpose(d)
        grad0 = gradscale * wp.vec3(Ht[0, 0], Ht[0, 1], Ht[0, 2])
        grad1 = gradscale * wp.vec3(Ht[1, 0], Ht[1, 1], Ht[1, 2])
        grad2 = gradscale * wp.vec3(Ht[2, 0], Ht[2, 1], Ht[2, 2])
        grad3 = -grad0 - grad1 - grad2

        wsum = (invmass0 * wp.dot(grad0, grad0) +
                invmass1 * wp.dot(grad1, grad1) +
                invmass2 * wp.dot(grad2, grad2) +
                invmass3 * wp.dot(grad3, grad3))

        if wsum != 0.0:
            alpha = 1.0 / kstiff
            if (flags & FemFlag.NORMSTIFFNESS) != 0:
                alpha = alpha / restlength
            alpha = alpha / (dt * dt)

            dsum = float(0.0)
            gamma = float(1.0)
            if kdampratio > 0.0:
                prev0 = pprev[pt0]
                prev1 = pprev[pt1]
                prev2 = pprev[pt2]
                prev3 = pprev[pt3]
                beta = kstiff * kdampratio * dt * dt
                if (flags & FemFlag.NORMSTIFFNESS) != 0:
                    beta = beta * restlength
                gamma = alpha * beta / dt
                dsum = (wp.dot(grad0, p0 - prev0) +
                        wp.dot(grad1, p1 - prev1) +
                        wp.dot(grad2, p2 - prev2) +
                        wp.dot(grad3, p3 - prev3))
                dsum = dsum * gamma
                gamma = gamma + 1.0

            C = psi
            dL = (-C - alpha * c_L[cidx][0] - dsum) / (gamma * wsum + alpha)
            if use_jacobi != 0:
                update_dP(dP, dPw, dL * invmass0 * grad0, pt0)
                update_dP(dP, dPw, dL * invmass1 * grad1, pt1)
                update_dP(dP, dPw, dL * invmass2 * grad2, pt2)
                update_dP(dP, dPw, dL * invmass3 * grad3, pt3)
            else:
                pos[pt0] = pos[pt0] + dL * invmass0 * grad0
                pos[pt1] = pos[pt1] + dL * invmass1 * grad1
                pos[pt2] = pos[pt2] + dL * invmass2 * grad2
                pos[pt3] = pos[pt3] + dL * invmass3 * grad3
                c_L[cidx] = wp.vec3(c_L[cidx][0] + dL, c_L[cidx][1], c_L[cidx][2])


@wp.kernel
def solve_tetarap_kernel(
    c_stiffness: wp.array(dtype=wp.float32),
    c_pts: wp.array(dtype=wp.vec4i),
    c_tetid: wp.array(dtype=wp.int32),
    c_type: wp.array(dtype=wp.int32),
    c_restlength: wp.array(dtype=wp.float32),
    c_dampingratio: wp.array(dtype=wp.float32),
    c_restvector: wp.array(dtype=wp.vec4f),
    c_L: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    rest_matrix: wp.array(dtype=wp.mat33),
    dt: float,
    use_jacobi: int,
    offset: int,
):
    c = offset + wp.tid()
    cstiffness = c_stiffness[c]
    if cstiffness <= 0.0:
        return
    pts = c_pts[c]
    tetid = c_tetid[c]
    ctype = c_type[c]
    flags = fem_flags_fn(ctype)
    tet_arap_update_xpbd_fn(
        use_jacobi, c, c_restvector, c_L, pts, dt, pos, pprev, dP, dPw,
        mass, stopped, c_restlength[c], rest_matrix[tetid],
        cstiffness, c_dampingratio[c], flags)


# SVD-based ARAP (canonical PBD-XPBD form, ported from
# realtimeMuscle/engine/arap_hpbd.py:project_constraints).
# C = ||S - I||_F where F = D_s · B and B = restmatrix.
# Gradient: dC/dF = sum_k dC/dS_k · U[:,k] · V[:,k]^T. Sign-fixed SVD via
# ssvd_fn keeps the gradient pointing back to det F > 0 even when F is
# inverted, which the polar-form variant could not do.
# NOTE: penalises ALL singular-value deviation (volume + isotropy + shear),
# which over-stiffens fiber contraction. Kept for reference; dispatch uses
# the polar form above.
@wp.kernel
def solve_tetarap_svd_kernel(
    c_stiffness: wp.array(dtype=wp.float32),
    c_pts: wp.array(dtype=wp.vec4i),
    c_tetid: wp.array(dtype=wp.int32),
    c_L: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    rest_matrix: wp.array(dtype=wp.mat33),
    dt: float,
    use_jacobi: int,
    offset: int,
):
    c = offset + wp.tid()
    cstiffness = c_stiffness[c]
    if cstiffness <= 0.0:
        return
    pts = c_pts[c]
    tetid = c_tetid[c]
    pt0 = pts[0]
    pt1 = pts[1]
    pt2 = pts[2]
    pt3 = pts[3]
    p0 = pos[pt0]
    p1 = pos[pt1]
    p2 = pos[pt2]
    p3 = pos[pt3]

    invmass0 = get_inv_mass_fn(pt0, mass, stopped)
    invmass1 = get_inv_mass_fn(pt1, mass, stopped)
    invmass2 = get_inv_mass_fn(pt2, mass, stopped)
    invmass3 = get_inv_mass_fn(pt3, mass, stopped)

    # V2 convention: pt3 is the reference (= x0 in realtimeMuscle).
    c0 = p0 - p3
    c1 = p1 - p3
    c2 = p2 - p3
    Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])
    B = rest_matrix[tetid]
    F = Ds * B

    svd = ssvd_fn(F)
    U = svd.U
    sigma = svd.sigma
    V = svd.V

    s_minus_1 = wp.vec3(sigma[0] - 1.0, sigma[1] - 1.0, sigma[2] - 1.0)
    sum_sq = (s_minus_1[0] * s_minus_1[0]
              + s_minus_1[1] * s_minus_1[1]
              + s_minus_1[2] * s_minus_1[2])
    if sum_sq < 1.0e-12:
        return
    Cval = wp.sqrt(sum_sq)
    inv_C = 1.0 / Cval
    dCdS0 = inv_C * s_minus_1[0]
    dCdS1 = inv_C * s_minus_1[1]
    dCdS2 = inv_C * s_minus_1[2]

    # dCdF[i,j] = sum_k dCdS[k] * U[i,k] * V[j,k]
    dCdF00 = U[0, 0] * V[0, 0] * dCdS0 + U[0, 1] * V[0, 1] * dCdS1 + U[0, 2] * V[0, 2] * dCdS2
    dCdF01 = U[0, 0] * V[1, 0] * dCdS0 + U[0, 1] * V[1, 1] * dCdS1 + U[0, 2] * V[1, 2] * dCdS2
    dCdF02 = U[0, 0] * V[2, 0] * dCdS0 + U[0, 1] * V[2, 1] * dCdS1 + U[0, 2] * V[2, 2] * dCdS2
    dCdF10 = U[1, 0] * V[0, 0] * dCdS0 + U[1, 1] * V[0, 1] * dCdS1 + U[1, 2] * V[0, 2] * dCdS2
    dCdF11 = U[1, 0] * V[1, 0] * dCdS0 + U[1, 1] * V[1, 1] * dCdS1 + U[1, 2] * V[1, 2] * dCdS2
    dCdF12 = U[1, 0] * V[2, 0] * dCdS0 + U[1, 1] * V[2, 1] * dCdS1 + U[1, 2] * V[2, 2] * dCdS2
    dCdF20 = U[2, 0] * V[0, 0] * dCdS0 + U[2, 1] * V[0, 1] * dCdS1 + U[2, 2] * V[0, 2] * dCdS2
    dCdF21 = U[2, 0] * V[1, 0] * dCdS0 + U[2, 1] * V[1, 1] * dCdS1 + U[2, 2] * V[1, 2] * dCdS2
    dCdF22 = U[2, 0] * V[2, 0] * dCdS0 + U[2, 1] * V[2, 1] * dCdS1 + U[2, 2] * V[2, 2] * dCdS2

    # grad_ptk[a] = sum_j dCdF[a,j] * B[k,j]   (k=0,1,2 for non-reference verts)
    grad0 = wp.vec3(
        dCdF00 * B[0, 0] + dCdF01 * B[0, 1] + dCdF02 * B[0, 2],
        dCdF10 * B[0, 0] + dCdF11 * B[0, 1] + dCdF12 * B[0, 2],
        dCdF20 * B[0, 0] + dCdF21 * B[0, 1] + dCdF22 * B[0, 2])
    grad1 = wp.vec3(
        dCdF00 * B[1, 0] + dCdF01 * B[1, 1] + dCdF02 * B[1, 2],
        dCdF10 * B[1, 0] + dCdF11 * B[1, 1] + dCdF12 * B[1, 2],
        dCdF20 * B[1, 0] + dCdF21 * B[1, 1] + dCdF22 * B[1, 2])
    grad2 = wp.vec3(
        dCdF00 * B[2, 0] + dCdF01 * B[2, 1] + dCdF02 * B[2, 2],
        dCdF10 * B[2, 0] + dCdF11 * B[2, 1] + dCdF12 * B[2, 2],
        dCdF20 * B[2, 0] + dCdF21 * B[2, 1] + dCdF22 * B[2, 2])
    grad3 = -grad0 - grad1 - grad2

    wsum = (invmass0 * wp.dot(grad0, grad0)
            + invmass1 * wp.dot(grad1, grad1)
            + invmass2 * wp.dot(grad2, grad2)
            + invmass3 * wp.dot(grad3, grad3))
    if wsum == 0.0:
        return

    alpha = 1.0 / cstiffness
    alpha_tilde = alpha / (dt * dt)
    L0 = c_L[c][0]
    dL = -(Cval + alpha_tilde * L0) / (wsum + alpha_tilde)

    if use_jacobi != 0:
        update_dP(dP, dPw, dL * invmass0 * grad0, pt0)
        update_dP(dP, dPw, dL * invmass1 * grad1, pt1)
        update_dP(dP, dPw, dL * invmass2 * grad2, pt2)
        update_dP(dP, dPw, dL * invmass3 * grad3, pt3)
    else:
        pos[pt0] = pos[pt0] + dL * invmass0 * grad0
        pos[pt1] = pos[pt1] + dL * invmass1 * grad1
        pos[pt2] = pos[pt2] + dL * invmass2 * grad2
        pos[pt3] = pos[pt3] + dL * invmass3 * grad3
        c_L[c] = wp.vec3(L0 + dL, c_L[c][1], c_L[c][2])


@wp.kernel
def solve_tet_aniso_arap_kernel(
    c_stiffness: wp.array(dtype=wp.float32),
    c_pts: wp.array(dtype=wp.vec4i),
    c_tetid: wp.array(dtype=wp.int32),
    c_L: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    rest_matrix: wp.array(dtype=wp.mat33),
    v_fiber_dir: wp.array(dtype=wp.vec3),
    activation: wp.array(dtype=wp.float32),
    tendonmask: wp.array(dtype=wp.float32),
    dt: float,
    use_jacobi: int,
    contraction_ratio: float,
    fiber_stiffness_scale: float,
    offset: int,
):
    """SoA anisotropic (along-fiber) ARAP, passive HOBAK fiber energy.

    SoA / dP-dPw port of ``solve_aniso_arap_kernel`` (the HPBD pos_mid version):
    quadratic fiber energy ``psi = 0.5*(s - SI4*fiberscale)^2`` with along-fiber
    stretch ``s = |F*f0|``, ``SI4 = sign(det F)`` (along-fiber inversion flag),
    and active ``fiberscale = 1 - belly*a*ratio`` (a=0 -> passive). Uses the V2
    rest convention (pt3 = reference, ``F = Ds*rest_matrix[tetid]``) so it shares
    the SoA ``rest_matrix`` with the isotropic ARAP-SVD kernel.
    """
    c = offset + wp.tid()
    cstiffness = c_stiffness[c]
    if cstiffness <= 0.0:
        return
    pts = c_pts[c]
    tetid = c_tetid[c]
    pt0 = pts[0]
    pt1 = pts[1]
    pt2 = pts[2]
    pt3 = pts[3]

    f0 = (v_fiber_dir[pt0] + v_fiber_dir[pt1]
          + v_fiber_dir[pt2] + v_fiber_dir[pt3]) / 4.0
    fn = wp.length(f0)
    if fn < 1.0e-9:
        return
    f0 = f0 / fn

    c0 = pos[pt0] - pos[pt3]
    c1 = pos[pt1] - pos[pt3]
    c2 = pos[pt2] - pos[pt3]
    Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])
    B = rest_matrix[tetid]
    F = Ds * B

    d = F * f0
    s = wp.length(d)
    if s < 1.0e-9:
        return
    # Skip degenerate / inverted / over-stretched tets (folded poses reach s>10
    # and yank shared verts cm-scale). s>1.5 = abnormal.
    detF = wp.determinant(F)
    if detF <= 0.0 or s > 1.5:
        return
    SI4 = wp.where(detF < 0.0, -1.0, 1.0)
    belly = 1.0 - tendonmask[tetid]
    fiberscale = 1.0 - belly * activation[tetid] * contraction_ratio
    diff = s - SI4 * fiberscale
    Cval = 0.5 * diff * diff
    # dpsi/dF = (diff/s)*d*f0^T mapped to vertex gradients via B (pt3 reference).
    gradscale = diff / s
    dCdF = gradscale * wp.outer(d, f0)
    grad0 = wp.vec3(
        dCdF[0, 0] * B[0, 0] + dCdF[0, 1] * B[0, 1] + dCdF[0, 2] * B[0, 2],
        dCdF[1, 0] * B[0, 0] + dCdF[1, 1] * B[0, 1] + dCdF[1, 2] * B[0, 2],
        dCdF[2, 0] * B[0, 0] + dCdF[2, 1] * B[0, 1] + dCdF[2, 2] * B[0, 2])
    grad1 = wp.vec3(
        dCdF[0, 0] * B[1, 0] + dCdF[0, 1] * B[1, 1] + dCdF[0, 2] * B[1, 2],
        dCdF[1, 0] * B[1, 0] + dCdF[1, 1] * B[1, 1] + dCdF[1, 2] * B[1, 2],
        dCdF[2, 0] * B[1, 0] + dCdF[2, 1] * B[1, 1] + dCdF[2, 2] * B[1, 2])
    grad2 = wp.vec3(
        dCdF[0, 0] * B[2, 0] + dCdF[0, 1] * B[2, 1] + dCdF[0, 2] * B[2, 2],
        dCdF[1, 0] * B[2, 0] + dCdF[1, 1] * B[2, 1] + dCdF[1, 2] * B[2, 2],
        dCdF[2, 0] * B[2, 0] + dCdF[2, 1] * B[2, 1] + dCdF[2, 2] * B[2, 2])
    grad3 = -grad0 - grad1 - grad2

    invmass0 = get_inv_mass_fn(pt0, mass, stopped)
    invmass1 = get_inv_mass_fn(pt1, mass, stopped)
    invmass2 = get_inv_mass_fn(pt2, mass, stopped)
    invmass3 = get_inv_mass_fn(pt3, mass, stopped)
    wsum = (invmass0 * wp.dot(grad0, grad0)
            + invmass1 * wp.dot(grad1, grad1)
            + invmass2 * wp.dot(grad2, grad2)
            + invmass3 * wp.dot(grad3, grad3))
    if wsum < 1.0e-20:
        return

    stiffness_val = cstiffness * fiber_stiffness_scale
    alpha_tilde = (1.0 / stiffness_val) / (dt * dt)
    L0 = c_L[c][0]
    dL = -(Cval + alpha_tilde * L0) / (wsum + alpha_tilde)

    if use_jacobi != 0:
        update_dP(dP, dPw, dL * invmass0 * grad0, pt0)
        update_dP(dP, dPw, dL * invmass1 * grad1, pt1)
        update_dP(dP, dPw, dL * invmass2 * grad2, pt2)
        update_dP(dP, dPw, dL * invmass3 * grad3, pt3)
    else:
        pos[pt0] = pos[pt0] + dL * invmass0 * grad0
        pos[pt1] = pos[pt1] + dL * invmass1 * grad1
        pos[pt2] = pos[pt2] + dL * invmass2 * grad2
        pos[pt3] = pos[pt3] + dL * invmass3 * grad3
        c_L[c] = wp.vec3(L0 + dL, c_L[c][1], c_L[c][2])


@wp.func
def tet_snh_update_xpbd_fn(
    use_jacobi: int,
    cidx: int,
    c_L: wp.array(dtype=wp.vec3),
    pts: wp.vec4i,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    restvolume: float,
    restmatrix: wp.mat33,
    mu: float,
    lam: float,
    kdampratio: float,
):
    """Stable Neo-Hookean XPBD constraint (coupled deviatoric + volumetric).

    Deviatoric: C_D = ||F||_F - sqrt(3)
    Volumetric: C_H = J - alpha  where alpha = 1 + mu/lam
    Coupled 2x2 solve for both Lagrange multipliers.
    Reference: Macklin "A Constraint-based Formulation of Stable Neo-Hookean Materials"
    """
    pt0 = pts[0]
    pt1 = pts[1]
    pt2 = pts[2]
    pt3 = pts[3]
    p0 = pos[pt0]
    p1 = pos[pt1]
    p2 = pos[pt2]
    p3 = pos[pt3]

    invmass0 = get_inv_mass_fn(pt0, mass, stopped)
    invmass1 = get_inv_mass_fn(pt1, mass, stopped)
    invmass2 = get_inv_mass_fn(pt2, mass, stopped)
    invmass3 = get_inv_mass_fn(pt3, mass, stopped)

    # Deformation gradient F = Ds * Dm^-1
    c0 = p0 - p3
    c1 = p1 - p3
    c2 = p2 - p3
    Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])
    F = Ds * restmatrix

    # --- Deviatoric constraint: C_D = ||F||_F - sqrt(3) ---
    Ic = (F[0, 0] * F[0, 0] + F[0, 1] * F[0, 1] + F[0, 2] * F[0, 2] +
          F[1, 0] * F[1, 0] + F[1, 1] * F[1, 1] + F[1, 2] * F[1, 2] +
          F[2, 0] * F[2, 0] + F[2, 1] * F[2, 1] + F[2, 2] * F[2, 2])
    r_s = wp.sqrt(wp.max(Ic, 1.0e-12))
    dC = r_s - 1.7320508075688772  # sqrt(3)

    if r_s < 1.0e-6:
        return

    # Deviatoric gradient: dC_D/dF = F/||F||, projected to position space
    inv_rs = 1.0 / r_s
    Fnorm = F * inv_rs
    Ht = restmatrix * wp.transpose(Fnorm)
    dgrad0 = wp.vec3(Ht[0, 0], Ht[0, 1], Ht[0, 2])
    dgrad1 = wp.vec3(Ht[1, 0], Ht[1, 1], Ht[1, 2])
    dgrad2 = wp.vec3(Ht[2, 0], Ht[2, 1], Ht[2, 2])
    dgrad3 = -dgrad0 - dgrad1 - dgrad2

    dsum_w = (invmass0 * wp.dot(dgrad0, dgrad0) +
              invmass1 * wp.dot(dgrad1, dgrad1) +
              invmass2 * wp.dot(dgrad2, dgrad2) +
              invmass3 * wp.dot(dgrad3, dgrad3))
    if dsum_w == 0.0:
        return

    # --- Volumetric constraint: C_H = J - alpha ---
    # Clamped: only resist volume expansion (J > alpha)
    alpha_snh = 1.0 + mu / lam
    J = wp.determinant(F)
    hC = J - alpha_snh

    # Volume gradients: dJ/dx_i via cross products of edge vectors
    inv_detDm = 1.0 / wp.max(wp.abs(restvolume) * 6.0, 1.0e-20)
    hgrad0 = wp.cross(c1, c2) * inv_detDm
    hgrad1 = wp.cross(c2, c0) * inv_detDm
    hgrad2 = wp.cross(c0, c1) * inv_detDm
    hgrad3 = -hgrad0 - hgrad1 - hgrad2

    hsum_w = (invmass0 * wp.dot(hgrad0, hgrad0) +
              invmass1 * wp.dot(hgrad1, hgrad1) +
              invmass2 * wp.dot(hgrad2, hgrad2) +
              invmass3 * wp.dot(hgrad3, hgrad3))
    if hsum_w == 0.0:
        return

    # Coupled cross-term
    dhsum = (invmass0 * wp.dot(dgrad0, hgrad0) +
             invmass1 * wp.dot(dgrad1, hgrad1) +
             invmass2 * wp.dot(dgrad2, hgrad2) +
             invmass3 * wp.dot(dgrad3, hgrad3))

    # XPBD compliances: alpha = 1/(stiffness * volume * dt^2)
    abs_vol = wp.max(wp.abs(restvolume), 1.0e-20)
    dalpha = 1.0 / (mu * abs_vol * dt * dt)
    halpha = 1.0 / (lam * abs_vol * dt * dt)

    # Damping
    ddamp = float(0.0)
    dgamma = float(1.0)
    hdamp = float(0.0)
    hgamma = float(1.0)
    if kdampratio > 0.0:
        prev0 = pprev[pt0]
        prev1 = pprev[pt1]
        prev2 = pprev[pt2]
        prev3 = pprev[pt3]
        dbeta = mu * abs_vol * kdampratio * dt * dt
        hbeta = lam * abs_vol * kdampratio * dt * dt
        dgamma = dalpha * dbeta / dt
        hgamma = halpha * hbeta / dt
        ddamp = (wp.dot(dgrad0, p0 - prev0) + wp.dot(dgrad1, p1 - prev1) +
                 wp.dot(dgrad2, p2 - prev2) + wp.dot(dgrad3, p3 - prev3))
        hdamp = (wp.dot(hgrad0, p0 - prev0) + wp.dot(hgrad1, p1 - prev1) +
                 wp.dot(hgrad2, p2 - prev2) + wp.dot(hgrad3, p3 - prev3))
        ddamp = ddamp * dgamma
        hdamp = hdamp * hgamma
        dgamma = dgamma + 1.0
        hgamma = hgamma + 1.0

    dL = c_L[cidx][0]
    hL = c_L[cidx][2]

    # Coupled 2x2 linear solve for both Lagrange multipliers
    Axx = dgamma * dsum_w + dalpha
    Ayy = hgamma * hsum_w + halpha
    det_A = Axx * Ayy - dhsum * dhsum
    if det_A < 1.0e-8:
        return
    inv_det = 1.0 / det_A
    db = -dC - dalpha * dL - ddamp
    hb = -hC - halpha * hL - hdamp
    ddL = inv_det * (Ayy * db - dhsum * hb)
    hdL = inv_det * (Axx * hb - dhsum * db)

    if use_jacobi != 0:
        update_dP(dP, dPw, invmass0 * (ddL * dgrad0 + hdL * hgrad0), pt0)
        update_dP(dP, dPw, invmass1 * (ddL * dgrad1 + hdL * hgrad1), pt1)
        update_dP(dP, dPw, invmass2 * (ddL * dgrad2 + hdL * hgrad2), pt2)
        update_dP(dP, dPw, invmass3 * (ddL * dgrad3 + hdL * hgrad3), pt3)
    else:
        pos[pt0] = p0 + invmass0 * (ddL * dgrad0 + hdL * hgrad0)
        pos[pt1] = p1 + invmass1 * (ddL * dgrad1 + hdL * hgrad1)
        pos[pt2] = p2 + invmass2 * (ddL * dgrad2 + hdL * hgrad2)
        pos[pt3] = p3 + invmass3 * (ddL * dgrad3 + hdL * hgrad3)
        c_L[cidx] = wp.vec3(dL + ddL, c_L[cidx][1], hL + hdL)


@wp.kernel
def solve_tetsnh_kernel(
    c_stiffness: wp.array(dtype=wp.float32),
    c_pts: wp.array(dtype=wp.vec4i),
    c_tetid: wp.array(dtype=wp.int32),
    c_restdir: wp.array(dtype=wp.vec3),
    c_restlength: wp.array(dtype=wp.float32),
    c_dampingratio: wp.array(dtype=wp.float32),
    c_L: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    rest_matrix: wp.array(dtype=wp.mat33),
    dt: float,
    use_jacobi: int,
    offset: int,
):
    c = offset + wp.tid()
    mu = c_stiffness[c]
    if mu <= 0.0:
        return
    pts = c_pts[c]
    tetid = c_tetid[c]
    lam = c_restdir[c][0]
    tet_snh_update_xpbd_fn(
        use_jacobi, c, c_L, pts, dt, pos, pprev, dP, dPw,
        mass, stopped, c_restlength[c], rest_matrix[tetid],
        mu, lam, c_dampingratio[c])


# ---------------------------------------------------------------------------
# MassSpring constraint solvers (Gaia S01 U-drop material)
# ---------------------------------------------------------------------------

@wp.kernel
def solve_distance_kernel(
    c_stiffness: wp.array(dtype=wp.float32),
    c_compressionstiffness: wp.array(dtype=wp.float32),
    c_pts: wp.array(dtype=wp.vec4i),
    c_restlength: wp.array(dtype=wp.float32),
    c_dampingratio: wp.array(dtype=wp.float32),
    c_L: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    dt: float,
    use_jacobi: int,
    has_compress_stiffness: int,
    offset: int,
):
    """Two-vertex XPBD edge spring.  Reuses ``distance_pos_update_xpbd_fn``
    (the engine distance constraint math) with both endpoints dynamic — the
    counterpart of MassSpring's spring_kernel."""
    c = offset + wp.tid()
    cstiffness = c_stiffness[c]
    if cstiffness <= 0.0:
        return
    kstiffcompress = c_compressionstiffness[c]
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, cstiffness)
    pts = c_pts[c]
    pt0 = pts[0]
    pt1 = pts[1]
    distance_pos_update_xpbd_fn(
        use_jacobi, c, c_L, pt0, pt1,
        pos[pt0], pos[pt1],
        pos, pprev, dP, dPw, mass, stopped,
        c_restlength[c], cstiffness, c_dampingratio[c],
        kstiffcompress, dt, has_compress_stiffness)


@wp.kernel
def solve_tetvolumehard_kernel(
    c_pts: wp.array(dtype=wp.vec4i),
    c_tetid: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    dm_inv: wp.array(dtype=wp.mat33),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    offset: int,
):
    """Hard PBD volume preservation C=det(F)-1, dlambda=-C/w (no compliance).
    Migrated from MassSpring volume_kernel; Gauss-Seidel direct write.
    ``dm_inv`` is the rest DmInv with vertex 0 as reference."""
    c = offset + wp.tid()
    pts = c_pts[c]
    i0 = pts[0]
    i1 = pts[1]
    i2 = pts[2]
    i3 = pts[3]
    w0 = get_inv_mass_fn(i0, mass, stopped)
    w1 = get_inv_mass_fn(i1, mass, stopped)
    w2 = get_inv_mass_fn(i2, mass, stopped)
    w3 = get_inv_mass_fn(i3, mass, stopped)
    v0 = pos[i0]
    P = wp.matrix_from_cols(pos[i1] - v0, pos[i2] - v0, pos[i3] - v0)
    iDS = dm_inv[c_tetid[c]]
    F = P * iDS
    f0 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f1 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f2 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    dF0 = wp.cross(f1, f2)
    dF1 = wp.cross(f2, f0)
    dF2 = wp.cross(f0, f1)
    g1 = dF0 * iDS[0, 0] + dF1 * iDS[0, 1] + dF2 * iDS[0, 2]
    g2 = dF0 * iDS[1, 0] + dF1 * iDS[1, 1] + dF2 * iDS[1, 2]
    g3 = dF0 * iDS[2, 0] + dF1 * iDS[2, 1] + dF2 * iDS[2, 2]
    g0 = -g1 - g2 - g3
    C = wp.determinant(F) - 1.0
    if C == 0.0:
        return
    w = (wp.dot(g0, g0) * w0 + wp.dot(g1, g1) * w1
         + wp.dot(g2, g2) * w2 + wp.dot(g3, g3) * w3)
    if w == 0.0:
        return
    dlam = -C / w
    pos[i0] = pos[i0] + g0 * (dlam * w0)
    pos[i1] = pos[i1] + g1 * (dlam * w1)
    pos[i2] = pos[i2] + g2 * (dlam * w2)
    pos[i3] = pos[i3] + g3 * (dlam * w3)


@wp.kernel
def solve_tetinversion_kernel(
    c_pts: wp.array(dtype=wp.vec4i),
    c_tetid: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    rest_vol: wp.array(dtype=wp.float32),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    multiplier: float,
    offset: int,
):
    """Untangle inverted tets (signed vol<=0).  Migrated directly from
    MassSpring inversion_kernel (no engine counterpart); GS direct write."""
    c = offset + wp.tid()
    pts = c_pts[c]
    i0 = pts[0]
    i1 = pts[1]
    i2 = pts[2]
    i3 = pts[3]
    e1 = pos[i1] - pos[i0]
    e2 = pos[i2] - pos[i0]
    e3 = pos[i3] - pos[i0]
    vol = wp.dot(e1, wp.cross(e2, e3)) / 6.0
    if vol > 0.0:
        return
    w0 = get_inv_mass_fn(i0, mass, stopped)
    w1 = get_inv_mass_fn(i1, mass, stopped)
    w2 = get_inv_mass_fn(i2, mass, stopped)
    w3 = get_inv_mass_fn(i3, mass, stopped)
    gr0 = wp.cross(e3 - e1, e2 - e1)
    gr1 = wp.cross(e2, e3)
    gr2 = wp.cross(e3, e1)
    gr3 = wp.cross(e1, e2)
    C = 6.0 * (vol - rest_vol[c_tetid[c]] * multiplier)
    if C == 0.0:
        return
    a0 = gr0 * w0
    a1 = gr1 * w1
    a2 = gr2 * w2
    a3 = gr3 * w3
    w = wp.dot(a0, a0) + wp.dot(a1, a1) + wp.dot(a2, a2) + wp.dot(a3, a3)
    if w == 0.0:
        return
    dlam = -C / w
    pos[i0] = pos[i0] + gr0 * (dlam * w0)
    pos[i1] = pos[i1] + gr1 * (dlam * w1)
    pos[i2] = pos[i2] + gr2 * (dlam * w2)
    pos[i3] = pos[i3] + gr3 * (dlam * w3)


@wp.kernel
def solve_aniso_arap_kernel(
    pos_mid: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.vec4i),
    B: wp.array(dtype=wp.mat33),
    fiber_dir: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    L: wp.array(dtype=wp.float32),
    activation: wp.array(dtype=wp.float32),  # per-tet a in [0,1] (0 = passive)
    tendonmask: wp.array(dtype=wp.float32),  # per-tet tendon fraction in [0,1]
    contraction_ratio: float,
    dt: float,
    omega: float,
    pos: wp.array(dtype=wp.vec3),
):
    """Anisotropic ARAP fiber constraint (Houdini, HOBAK / 8.4.2 in
    *Dynamic Deformables*).

    Quadratic fiber energy ``psi = 0.5·(sqrt(I5) - SI4·fiberscale)^2`` with the
    along-fiber invariant ``I5 = |F·f0|^2`` (so ``sqrt(I5) = s``), ``fiberscale
    = 1`` (passive, no contraction), and ``SI4 = sign(det F)`` flagging
    along-fiber inversion (cheap branch of Houdini ``calcI4``). Quadratic
    HOBAK fiber energy projected via parallel-Jacobi XPBD.
    """
    t = wp.tid()
    idx = tet_indices[t]
    p0i = idx[0]
    p1i = idx[1]
    p2i = idx[2]
    p3i = idx[3]

    f0 = (fiber_dir[p0i] + fiber_dir[p1i] + fiber_dir[p2i] + fiber_dir[p3i]) / 4.0
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

    d = F * f0
    s = wp.length(d)
    if s < 1.0e-9:
        return
    # Skip degenerate / inverted / over-stretched tets:
    # folded-pose regions reach s>10 and yank shared verts cm-scale. s>1.5 = abnormal.
    if wp.determinant(F) <= 0.0 or s > 1.5:
        return
    SI4 = wp.where(wp.determinant(F) < 0.0, -1.0, 1.0)
    # Active contraction: fiberscale = s* = 1 - belly·a·ratio (a=0 ⇒ passive,
    # fiberscale=1). Target along-fiber stretch is SI4·fiberscale.
    belly = 1.0 - tendonmask[t]
    fiberscale = 1.0 - belly * activation[t] * contraction_ratio
    diff = s - SI4 * fiberscale
    C = 0.5 * diff * diff
    # dpsi/dF = (diff/s)·d·f0^T  (gradscale = diff/s = 1 - SI4/s).
    gradscale = diff / s
    dCdF = gradscale * wp.outer(d, f0)
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
    if denom < 1.0e-20:
        return
    alpha_tilde = alpha[t] / (dt * dt)
    dlambda = -(C + alpha_tilde * L[t]) / (denom + alpha_tilde)
    L[t] = L[t] + dlambda

    f = omega * dlambda
    wp.atomic_add(pos, p0i, (f * w0) * g0)
    wp.atomic_add(pos, p1i, (f * w1) * g1)
    wp.atomic_add(pos, p2i, (f * w2) * g2)
    wp.atomic_add(pos, p3i, (f * w3) * g3)


# ---------------------------------------------------------------------------
# MuscleSim (Warp backend)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public surface: re-export the solve kernels, @wp.func helpers and structs so
# host modules can ``from muscle.constraint.solve_kernels import *``. Underscore
# names (none here) would need listing; module-level imports are excluded.
# ---------------------------------------------------------------------------
_IMPORTED = {
    'wp', 'annotations', 'ConstraintType', 'FemFlag',
    'dgf_active_force_length_wp', 'dgf_passive_force_length_wp', 'dgf_force_velocity_wp',
    'millard_eval_wp', 'millard_energy_eval_wp',
}
__all__ = [_n for _n in dir()
           if not _n.startswith('__') and _n != '_IMPORTED' and _n not in _IMPORTED]
