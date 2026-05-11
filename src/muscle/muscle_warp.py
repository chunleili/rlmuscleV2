import os
import time
from pathlib import Path

import numpy as np
import warp as wp
import warp.render

from muscle.config import SimConfig
from muscle.constraints import (
    PIN, ATTACH, PINTOTARGET, TETVOLUME, TETFIBERNORM, DISTANCELINE,
    TETARAP, TETARAPNORM, TETFIBERDGF, TETSNH, TETFIBERMILLARD,
    LINEARENERGY, NORMSTIFFNESS,
    ConstraintBuilderMixin,
)
from muscle.mesh_io import build_surface_tris, load_mesh, load_bone_mesh
from muscle.vis_warp import WarpRenderer


def activation_ramp(t: float) -> float:
    """Piecewise activation over normalized time [0,1]: 0→0.5→1.0→0.7→0.3→0."""
    if t <= 0.2:
        return 0.0
    elif t <= 0.3:
        return 0.5
    elif t <= 0.5:
        return 1.0
    elif t <= 0.7:
        return 0.7
    elif t <= 0.8:
        return 0.3
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Utility @wp.kernel for filling arrays
# ---------------------------------------------------------------------------

@wp.kernel
def fill_float_kernel(arr: wp.array(dtype=wp.float32), val: wp.float32):
    i = wp.tid()
    arr[i] = val

@wp.kernel
def fill_vec3_kernel(arr: wp.array(dtype=wp.vec3), val: wp.vec3):
    i = wp.tid()
    arr[i] = val


# ---------------------------------------------------------------------------
# @wp.struct definitions
# ---------------------------------------------------------------------------

@wp.struct
class SVDResult:
    U: wp.mat33
    sigma: wp.vec3
    V: wp.mat33

@wp.struct
class PolarResult:
    S: wp.mat33
    R: wp.mat33

@wp.struct
class Constraint:
    type: wp.int32
    cidx: wp.int32
    pts: wp.vec4i
    stiffness: wp.float32
    dampingratio: wp.float32
    tetid: wp.int32
    L: wp.vec3
    restlength: wp.float32
    restvector: wp.vec4f
    restdir: wp.vec3
    compressionstiffness: wp.float32


@wp.kernel
def update_cons_restdir1_kernel(
    cons: wp.array(dtype=Constraint),
    val: float,
    ctype_match: int,
    offset: int,
    count: int,
):
    """Update restdir[1] for constraints of a given type."""
    idx = offset + wp.tid()
    if cons[idx].type == ctype_match:
        rd = cons[idx].restdir
        cons[idx].restdir = wp.vec3(rd[0], val, rd[2])


# ---------------------------------------------------------------------------
# Utility @wp.func helpers
# ---------------------------------------------------------------------------

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
    if ctype == TETARAP or ctype == TETARAPNORM:
        flags = flags | LINEARENERGY
    if ctype == TETARAPNORM or ctype == TETFIBERNORM or ctype == TETFIBERDGF:
        flags = flags | NORMSTIFFNESS
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


# ---------------------------------------------------------------------------
# DGF (DeGroote-Fregly 2016) muscle curve @wp.func
# Constants mirror dgf_curves.py
# ---------------------------------------------------------------------------

@wp.func
def dgf_active_force_length_wp(lm_tilde: float) -> float:
    """Active force-length curve: 3 Gaussian-like terms."""
    b11 = 0.815;  b21 = 1.055;  b31 = 0.162;  b41 = 0.063
    b12 = 0.433;  b22 = 0.717;  b32 = -0.030; b42 = 0.200
    b13 = 0.1;    b23 = 1.0;    b33 = 0.354
    EPS = 1.0e-6

    d1 = wp.max(wp.abs(b31 + b41 * lm_tilde), EPS)
    g1 = b11 * wp.exp(-0.5 * ((lm_tilde - b21) / d1) * ((lm_tilde - b21) / d1))

    d2 = wp.max(wp.abs(b32 + b42 * lm_tilde), EPS)
    g2 = b12 * wp.exp(-0.5 * ((lm_tilde - b22) / d2) * ((lm_tilde - b22) / d2))

    g3 = b13 * wp.exp(-0.5 * ((lm_tilde - b23) / b33) * ((lm_tilde - b23) / b33))
    return g1 + g2 + g3


@wp.func
def dgf_passive_force_length_wp(lm_tilde: float) -> float:
    """Passive force-length curve: exponential."""
    KPE = 4.0
    E0 = 0.6
    LM_MIN = 0.2

    offset = wp.exp(KPE * (LM_MIN - 1.0) / E0)
    denom = wp.exp(KPE) - offset
    arg = wp.clamp(KPE * (lm_tilde - 1.0) / E0, -50.0, 50.0)
    result = (wp.exp(arg) - offset) / denom
    return wp.max(result, 0.0)


@wp.func
def dgf_force_velocity_wp(v_norm: float) -> float:
    """Force-velocity curve: hyperbolic (asinh-based)."""
    D1 = -0.3211346127989808
    D2 = -8.149
    D3 = -0.374
    D4 = 0.8825327733249912

    x = D2 * v_norm + D3
    return D1 * wp.log(x + wp.sqrt(x * x + 1.0)) + D4


# ---------------------------------------------------------------------------
# Millard 2012 quintic Bezier curve evaluation (GPU)
# Coefficients in power basis: f(u) = c0 + c1*u + c2*u^2 + ... + c5*u^5
# Energy integral F(u) = ∫₀ᵘ y(t)·x'(t)dt is degree-10 polynomial.
# ---------------------------------------------------------------------------

@wp.func
def horner10_wp(u: float, c0: float, c1: float, c2: float, c3: float,
                c4: float, c5: float, c6: float, c7: float, c8: float,
                c9: float, c10: float) -> float:
    """Evaluate degree-10 polynomial via Horner's method."""
    return c0 + u * (c1 + u * (c2 + u * (c3 + u * (c4 + u * (
           c5 + u * (c6 + u * (c7 + u * (c8 + u * (c9 + u * c10)))))))))


@wp.func
def horner5_wp(u: float, c0: float, c1: float, c2: float,
               c3: float, c4: float, c5: float) -> float:
    """Evaluate degree-5 polynomial via Horner's method."""
    return c0 + u * (c1 + u * (c2 + u * (c3 + u * (c4 + u * c5))))


@wp.func
def horner5_deriv_wp(u: float, c1: float, c2: float,
                     c3: float, c4: float, c5: float) -> float:
    """Evaluate derivative of degree-5 polynomial: c1 + 2c2*u + ... + 5c5*u^4."""
    return c1 + u * (2.0 * c2 + u * (3.0 * c3 + u * (4.0 * c4 + u * 5.0 * c5)))


@wp.func
def millard_eval_wp(
    x_target: float,
    x_coeffs: wp.array(dtype=wp.float32),
    y_coeffs: wp.array(dtype=wp.float32),
    seg_bounds: wp.array(dtype=wp.float32),
    n_segments: int,
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    dydx_lo: float, dydx_hi: float,
) -> float:
    """Evaluate Millard piecewise quintic Bezier curve y(x) on GPU.

    Args:
        x_target: Input value (normalized fiber length).
        x_coeffs: Flattened power-basis coefficients for x(u), shape (n_seg*6,).
        y_coeffs: Flattened power-basis coefficients for y(u), shape (n_seg*6,).
        seg_bounds: Segment x-boundaries, shape (n_seg+1,): [x_start_0, x_end_0, ...].
        n_segments: Number of Bezier segments.
        x_lo, x_hi: Domain boundaries.
        y_lo, y_hi, dydx_lo, dydx_hi: Boundary values for linear extrapolation.
    """
    # Linear extrapolation outside domain
    if x_target < x_lo:
        return y_lo + dydx_lo * (x_target - x_lo)
    if x_target > x_hi:
        return y_hi + dydx_hi * (x_target - x_hi)

    # Find segment via linear scan (no break — Warp limitation)
    seg = int(n_segments - 1)
    found = int(0)
    for i in range(n_segments):
        if found == 0 and x_target <= seg_bounds[i + 1]:
            seg = i
            found = 1

    # Read x(u) coefficients for this segment
    off = seg * 6
    xc0 = float(x_coeffs[off])
    xc1 = float(x_coeffs[off + 1])
    xc2 = float(x_coeffs[off + 2])
    xc3 = float(x_coeffs[off + 3])
    xc4 = float(x_coeffs[off + 4])
    xc5 = float(x_coeffs[off + 5])

    # Newton iteration: x(u) = x_target → solve for u
    x_at_1 = xc0 + xc1 + xc2 + xc3 + xc4 + xc5
    denom = x_at_1 - xc0
    u = float(0.5)
    if wp.abs(denom) > 1.0e-20:
        u = (x_target - xc0) / denom
    u = wp.clamp(u, 0.0, 1.0)

    for _iter in range(5):
        xu = horner5_wp(u, xc0, xc1, xc2, xc3, xc4, xc5)
        dxu = horner5_deriv_wp(u, xc1, xc2, xc3, xc4, xc5)
        if wp.abs(dxu) > 1.0e-20:
            u = u - (xu - x_target) / dxu
            u = wp.clamp(u, -0.05, 1.05)

    # Read y(u) coefficients and evaluate
    yc0 = float(y_coeffs[off])
    yc1 = float(y_coeffs[off + 1])
    yc2 = float(y_coeffs[off + 2])
    yc3 = float(y_coeffs[off + 3])
    yc4 = float(y_coeffs[off + 4])
    yc5 = float(y_coeffs[off + 5])

    return horner5_wp(u, yc0, yc1, yc2, yc3, yc4, yc5)


@wp.func
def millard_energy_eval_wp(
    x_target: float,
    x_coeffs: wp.array(dtype=wp.float32),
    F_coeffs: wp.array(dtype=wp.float32),
    seg_bounds: wp.array(dtype=wp.float32),
    n_segments: int,
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    dydx_lo: float, dydx_hi: float,
) -> float:
    """Evaluate cumulative energy integral Psi(x) = integral_{x_lo}^{x} y(s) ds.

    Uses precomputed degree-10 polynomial F_coeffs per segment.
    F_coeffs layout: (n_seg*11,) flat, F_coeffs[seg*11 : seg*11+11].
    """
    # Left extrapolation: quadratic (integral of linear extrapolation)
    if x_target <= x_lo:
        dx = x_target - x_lo
        return y_lo * dx + 0.5 * dydx_lo * dx * dx

    # Accumulate complete segments + partial segment
    total = float(0.0)

    # Warp has no break — use done flag
    done = int(0)
    for i in range(n_segments):
        if done == 0:
            if x_target >= seg_bounds[i + 1]:
                # Complete segment: add F(1)
                off = i * 11
                total += horner10_wp(
                    1.0,
                    float(F_coeffs[off]), float(F_coeffs[off + 1]),
                    float(F_coeffs[off + 2]), float(F_coeffs[off + 3]),
                    float(F_coeffs[off + 4]), float(F_coeffs[off + 5]),
                    float(F_coeffs[off + 6]), float(F_coeffs[off + 7]),
                    float(F_coeffs[off + 8]), float(F_coeffs[off + 9]),
                    float(F_coeffs[off + 10]),
                )
            else:
                # Partial segment: Newton x→u, then F(u)
                xoff = i * 6
                xc0 = float(x_coeffs[xoff])
                xc1 = float(x_coeffs[xoff + 1])
                xc2 = float(x_coeffs[xoff + 2])
                xc3 = float(x_coeffs[xoff + 3])
                xc4 = float(x_coeffs[xoff + 4])
                xc5 = float(x_coeffs[xoff + 5])

                x_at_1 = xc0 + xc1 + xc2 + xc3 + xc4 + xc5
                denom = x_at_1 - xc0
                u = float(0.5)
                if wp.abs(denom) > 1.0e-20:
                    u = (x_target - xc0) / denom
                u = wp.clamp(u, 0.0, 1.0)

                for _iter in range(5):
                    xu = horner5_wp(u, xc0, xc1, xc2, xc3, xc4, xc5)
                    dxu = horner5_deriv_wp(u, xc1, xc2, xc3, xc4, xc5)
                    if wp.abs(dxu) > 1.0e-20:
                        u = u - (xu - x_target) / dxu
                        u = wp.clamp(u, -0.05, 1.05)

                off = i * 11
                total += horner10_wp(
                    u,
                    float(F_coeffs[off]), float(F_coeffs[off + 1]),
                    float(F_coeffs[off + 2]), float(F_coeffs[off + 3]),
                    float(F_coeffs[off + 4]), float(F_coeffs[off + 5]),
                    float(F_coeffs[off + 6]), float(F_coeffs[off + 7]),
                    float(F_coeffs[off + 8]), float(F_coeffs[off + 9]),
                    float(F_coeffs[off + 10]),
                )
                done = 1

    # Right extrapolation: quadratic
    if x_target > x_hi:
        dx = x_target - x_hi
        total += y_hi * dx + 0.5 * dydx_hi * dx * dx

    return total


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
    cons: wp.array(dtype=Constraint),
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
            l = get_L_component(cons[cidx].L, loff)

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
                cons[cidx].L = set_L_component(cons[cidx].L, loff, get_L_component(cons[cidx].L, loff) + dlambda)


@wp.func
def tet_fiber_update_xpbd_fn(
    use_jacobi: int,
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    cidx: int,
    cons: wp.array(dtype=Constraint),
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

    l = cons[cidx].L[0]
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
            cons[cidx].L = wp.vec3(cons[cidx].L[0] + dL, cons[cidx].L[1], cons[cidx].L[2])


@wp.func
def distance_pos_update_xpbd_fn(
    use_jacobi: int,
    cidx: int,
    cons: wp.array(dtype=Constraint),
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
                l = get_L_component(cons[cidx].L, loff)

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
                    cons[cidx].L = set_L_component(cons[cidx].L, loff, get_L_component(cons[cidx].L, loff) + dL)


@wp.func
def attach_bilateral_update_fn(
    use_jacobi: int,
    cidx: int,
    cons: wp.array(dtype=Constraint),
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
                l = get_L_component(cons[cidx].L, loff)

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
                    cons[cidx].L = set_L_component(cons[cidx].L, loff, get_L_component(cons[cidx].L, loff) + dL)

                wp.atomic_add(reaction_accum, cidx, C * n)


# ---------------------------------------------------------------------------
# Simulation @wp.kernel
# ---------------------------------------------------------------------------

@wp.kernel
def precompute_rest_kernel(
    pos0: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.vec4i),
    rest_volume: wp.array(dtype=wp.float32),
    rest_matrix: wp.array(dtype=wp.mat33),
    mass: wp.array(dtype=wp.float32),
    density: float,
    total_rest_volume: wp.array(dtype=wp.float32),
):
    c = wp.tid()
    pts = tet_indices[c]
    c0 = pos0[pts[0]] - pos0[pts[3]]
    c1 = pos0[pts[1]] - pos0[pts[3]]
    c2 = pos0[pts[2]] - pos0[pts[3]]
    Dm = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])
    vol = wp.abs(wp.determinant(Dm)) / 6.0
    rest_volume[c] = vol
    wp.atomic_add(total_rest_volume, 0, vol)
    mass_contrib = vol * density / 4.0
    for i in range(4):
        wp.atomic_add(mass, pts[i], mass_contrib)
    rest_matrix[c] = wp.inverse(Dm)


@wp.kernel
def _clamp_force_by_accel_kernel(
    force: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=wp.float32),
    max_accel: float,
):
    """Clamp force magnitude so that |f|/m <= max_accel."""
    i = wp.tid()
    m = mass[i]
    if m < 1.0e-12:
        force[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    f = force[i]
    f_mag = wp.length(f)
    f_limit = m * max_accel
    if f_mag > f_limit and f_mag > 1.0e-20:
        force[i] = f * (f_limit / f_mag)


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
def _svd_clamp_accumulate_kernel(
    tet_indices: wp.array(dtype=wp.vec4i),
    pos: wp.array(dtype=wp.vec3),
    rest_matrix: wp.array(dtype=wp.mat33),
    stopped: wp.array(dtype=wp.int32),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    sigma_min: float,
):
    """SVD-based inversion prevention: accumulate Jacobi-style position
    corrections that project tets toward non-inverted configurations.
    All 4 vertices are corrected symmetrically via centroid-preserving
    projection."""
    c = wp.tid()
    pts = tet_indices[c]
    p0 = pos[pts[0]]; p1 = pos[pts[1]]; p2 = pos[pts[2]]; p3 = pos[pts[3]]
    c0 = p0 - p3; c1 = p1 - p3; c2 = p2 - p3
    Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])
    Dm_inv = rest_matrix[c]
    F = Ds * Dm_inv

    # Signed SVD: F = U * diag(sigma) * V^T
    svd = ssvd_fn(F)
    sig = svd.sigma

    # Only act if any singular value is below threshold
    if sig[0] >= sigma_min and sig[1] >= sigma_min and sig[2] >= sigma_min:
        return

    # Clamp singular values
    sig_c = wp.vec3(
        wp.max(sig[0], sigma_min),
        wp.max(sig[1], sigma_min),
        wp.max(sig[2], sigma_min))
    S_c = wp.mat33(
        sig_c[0], 0.0, 0.0,
        0.0, sig_c[1], 0.0,
        0.0, 0.0, sig_c[2])

    # Target deformation gradient with clamped singular values
    F_target = svd.U * S_c * wp.transpose(svd.V)

    # Reconstruct target edge matrix: Ds_target = F_target * Dm
    Dm = mat33_inverse_fn(Dm_inv)
    Ds_target = F_target * Dm

    # Compute per-vertex corrections (relative to reference vertex p3)
    # Ds columns = [p0-p3, p1-p3, p2-p3], p3 is reference → no raw correction
    d0 = wp.vec3(Ds_target[0, 0] - c0[0], Ds_target[1, 0] - c0[1], Ds_target[2, 0] - c0[2])
    d1 = wp.vec3(Ds_target[0, 1] - c1[0], Ds_target[1, 1] - c1[1], Ds_target[2, 1] - c1[2])
    d2 = wp.vec3(Ds_target[0, 2] - c2[0], Ds_target[1, 2] - c2[1], Ds_target[2, 2] - c2[2])
    # Distribute symmetrically: shift centroid so sum of corrections = 0
    shift = (d0 + d1 + d2) * 0.25
    d0 = d0 - shift
    d1 = d1 - shift
    d2 = d2 - shift
    d3 = -shift

    # Accumulate corrections via Jacobi averaging (same as XPBD dP/dPw)
    if stopped[pts[0]] == 0:
        wp.atomic_add(dP, pts[0], d0)
        wp.atomic_add(dPw, pts[0], 1.0)
    if stopped[pts[1]] == 0:
        wp.atomic_add(dP, pts[1], d1)
        wp.atomic_add(dPw, pts[1], 1.0)
    if stopped[pts[2]] == 0:
        wp.atomic_add(dP, pts[2], d2)
        wp.atomic_add(dPw, pts[2], 1.0)
    if stopped[pts[3]] == 0:
        wp.atomic_add(dP, pts[3], d3)
        wp.atomic_add(dPw, pts[3], 1.0)


@wp.kernel
def _apply_svd_clamp_kernel(
    pos: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    alpha: float,
):
    """Apply Jacobi-averaged SVD clamp corrections with blending factor."""
    i = wp.tid()
    w = dPw[i]
    if w > 0.0:
        pos[i] = pos[i] + alpha * dP[i] / w


@wp.kernel
def _freeze_near_inverted_kernel(
    tet_indices: wp.array(dtype=wp.vec4i),
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    rest_matrix: wp.array(dtype=wp.mat33),
    stopped: wp.array(dtype=wp.int32),
    detF_threshold: float,
):
    """Revert vertices of near-inverted tets (det(F) < threshold) to pprev."""
    c = wp.tid()
    pts = tet_indices[c]
    p0 = pos[pts[0]]; p1 = pos[pts[1]]; p2 = pos[pts[2]]; p3 = pos[pts[3]]
    c0 = p0 - p3; c1 = p1 - p3; c2 = p2 - p3
    Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])
    F = Ds * rest_matrix[c]
    J = wp.determinant(F)
    if J < detF_threshold:
        for i in range(4):
            vi = pts[i]
            if stopped[vi] == 0:
                pos[vi] = pprev[vi]


@wp.kernel
def integrate_kernel(
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),
    gravity_vec: wp.vec3,
    veldamping: float,
    dt: float,
):
    i = wp.tid()
    if stopped[i] != 0:
        pprev[i] = pos[i]
        vel[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    extacc = gravity_vec
    m = mass[i]
    if m > 1.0e-12:
        extacc = extacc + force[i] / m
    pprev[i] = pos[i]
    v = (1.0 - veldamping) * vel[i]
    v = v + dt * extacc
    vel[i] = v
    pos[i] = pos[i] + dt * v


@wp.kernel
def accumulate_active_fiber_force_kernel(
    tet_indices: wp.array(dtype=wp.vec4i),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3),
    rest_matrix: wp.array(dtype=wp.mat33),
    rest_volume: wp.array(dtype=wp.float32),
    v_fiber_dir: wp.array(dtype=wp.vec3),
    activation: wp.array(dtype=wp.float32),
    sigma0: float,
    lambda_opt: float,
    v_max_norm: float,
    d_damp: float,
    # Millard f_L curve data (force evaluation only)
    fl_x_coeffs: wp.array(dtype=wp.float32),
    fl_y_coeffs: wp.array(dtype=wp.float32),
    fl_seg_bounds: wp.array(dtype=wp.float32),
    fl_n_segments: int,
    fl_x_lo: float,
    fl_x_hi: float,
    fl_y_lo: float,
    fl_y_hi: float,
    fl_dydx_lo: float,
    fl_dydx_hi: float,
):
    """Explicit active fiber force per tet, scattered to vertex force field.

    f_j = -V0 * tau * w[j] * n   (j=0,1,2)
    f_3 = -(f_0 + f_1 + f_2)
    where tau = sigma0 * (a * f_FL(r) * f_V(v_norm) + d_damp * v_norm)
    r = lambda_f / lambda_opt, v_norm = dlm_dt / v_max_norm
    """
    tid = wp.tid()
    pts = tet_indices[tid]
    acti = activation[tid]
    if acti < 1.0e-6 and d_damp < 1.0e-6:
        return

    p0 = pos[pts[0]]
    p1 = pos[pts[1]]
    p2 = pos[pts[2]]
    p3 = pos[pts[3]]

    c0 = p0 - p3
    c1 = p1 - p3
    c2 = p2 - p3
    _Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])

    Dminv = rest_matrix[tid]
    V0 = rest_volume[tid]

    # Per-tet fiber direction (average vertex dirs, normalized)
    d_raw = (v_fiber_dir[pts[0]] + v_fiber_dir[pts[1]]
             + v_fiber_dir[pts[2]] + v_fiber_dir[pts[3]])
    d_len = wp.length(d_raw)
    if d_len < 1.0e-8:
        return
    d = d_raw / d_len

    # w = Dminv @ d
    w = Dminv * d

    # Fd = F @ d = Ds @ w
    Fd = _Ds * w
    lm_raw = wp.length(Fd)
    if lm_raw < 1.0e-8:
        return

    # Stretch ratio alignment: r = lambda_f / lambda_opt
    r = lm_raw / lambda_opt

    # Current fiber unit direction
    n = Fd / lm_raw

    # --- Fiber velocity from vertex velocities ---
    dv0 = vel[pts[0]] - vel[pts[3]]
    dv1 = vel[pts[1]] - vel[pts[3]]
    dv2 = vel[pts[2]] - vel[pts[3]]
    Ds_dot = wp.mat33(
        dv0[0], dv1[0], dv2[0],
        dv0[1], dv1[1], dv2[1],
        dv0[2], dv1[2], dv2[2])
    Fd_dot = Ds_dot * w
    dlm_dt = wp.dot(n, Fd_dot)

    v_norm = 0.0
    if v_max_norm > 1.0e-8:
        v_norm = wp.clamp(dlm_dt / v_max_norm, -1.0, 1.0)

    # Evaluate Millard f_FL(r) and DGF f_V(v_norm)
    f_FL_val = millard_eval_wp(
        r, fl_x_coeffs, fl_y_coeffs, fl_seg_bounds, fl_n_segments,
        fl_x_lo, fl_x_hi, fl_y_lo, fl_y_hi, fl_dydx_lo, fl_dydx_hi)
    f_V_val = dgf_force_velocity_wp(v_norm)

    # tau = sigma0 * (a * f_FL * f_V + d_damp * v_norm)
    tau = sigma0 * (acti * f_FL_val * f_V_val + d_damp * v_norm)

    # Scatter node forces: f_j = -V0 * tau * w[j] * n
    s = -V0 * tau
    f0 = s * w[0] * n
    f1 = s * w[1] * n
    f2 = s * w[2] * n
    f3 = -(f0 + f1 + f2)

    wp.atomic_add(force, pts[0], f0)
    wp.atomic_add(force, pts[1], f1)
    wp.atomic_add(force, pts[2], f2)
    wp.atomic_add(force, pts[3], f3)


@wp.kernel
def update_velocities_kernel(
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    vel[i] = (pos[i] - pprev[i]) / dt


@wp.kernel
def update_velocities_with_contact_kernel(
    pos: wp.array(dtype=wp.vec3),
    pprev: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    dt: float,
    is_collide: wp.array(dtype=wp.int32),
    contact_normal: wp.array(dtype=wp.vec3),
    restitution_n: float,
    restitution_t: float,
):
    i = wp.tid()
    v = (pos[i] - pprev[i]) / dt
    if is_collide[i] == 0:
        vel[i] = v
        return
    n = contact_normal[i]
    n_sq = wp.dot(n, n)
    if n_sq < 1.0e-12:
        vel[i] = v
        return
    n_hat = n / wp.sqrt(n_sq)
    vn = wp.dot(v, n_hat)
    v_t = v - vn * n_hat
    if vn < 0.0:
        vn_new = -restitution_n * vn
    else:
        vn_new = vn
    v_t_new = restitution_t * v_t
    vel[i] = vn_new * n_hat + v_t_new


@wp.kernel
def lbs_apply_kernel(
    pos: wp.array(dtype=wp.vec3),
    pos0: wp.array(dtype=wp.vec3),
    vertex_bone: wp.array(dtype=wp.int32),
    T_rel_flat: wp.array(dtype=wp.float32),
    blend: float,
):
    # Row-vector convention: q = p @ R + t, where each bone's 12-float row is
    # laid out as [r00,r01,r02, r10,r11,r12, r20,r21,r22, t0,t1,t2].
    i = wp.tid()
    b = vertex_bone[i]
    if b < 0:
        return
    p = pos0[i]
    base = b * 12
    qx = p[0] * T_rel_flat[base + 0] + p[1] * T_rel_flat[base + 3] + p[2] * T_rel_flat[base + 6] + T_rel_flat[base + 9]
    qy = p[0] * T_rel_flat[base + 1] + p[1] * T_rel_flat[base + 4] + p[2] * T_rel_flat[base + 7] + T_rel_flat[base + 10]
    qz = p[0] * T_rel_flat[base + 2] + p[1] * T_rel_flat[base + 5] + p[2] * T_rel_flat[base + 8] + T_rel_flat[base + 11]
    target = wp.vec3(qx, qy, qz)
    cur = pos[i]
    pos[i] = (1.0 - blend) * cur + blend * target


@wp.kernel
def clear_dP_kernel(
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    dP[i] = wp.vec3(0.0)
    dPw[i] = 0.0

@wp.kernel
def clear_cons_L_kernel(
    cons: wp.array(dtype=Constraint),
):
    i = wp.tid()
    cons[i].L = wp.vec3(0.0)


@wp.kernel
def clear_reaction_kernel(
    reaction_accum: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    reaction_accum[i] = wp.vec3(0.0)


@wp.kernel
def apply_dP_kernel(
    pos: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
):
    idx = wp.tid()
    w = dPw[idx]
    if w > 1e-9:
        pos[idx] = pos[idx] + dP[idx] / w


@wp.kernel
def calc_vol_error_kernel(
    tet_indices: wp.array(dtype=wp.vec4i),
    pos: wp.array(dtype=wp.vec3),
    rest_volume: wp.array(dtype=wp.float32),
    total_vol_out: wp.array(dtype=wp.float32),
):
    c = wp.tid()
    pts = tet_indices[c]
    p0 = pos[pts[0]]
    p1 = pos[pts[1]]
    p2 = pos[pts[2]]
    p3 = pos[pts[3]]
    d1 = p1 - p0
    d2 = p2 - p0
    d3 = p3 - p0
    volume = wp.dot(wp.cross(d2, d1), d3) / 6.0
    wp.atomic_add(total_vol_out, 0, volume)


@wp.kernel
def compute_cell_tendon_mask_kernel(
    tet_indices: wp.array(dtype=wp.vec4i),
    v_tendonmask: wp.array(dtype=wp.float32),
    tendonmask: wp.array(dtype=wp.float32),
):
    c = wp.tid()
    pts = tet_indices[c]
    sum_mask = float(0.0)
    for i in range(4):
        sum_mask = sum_mask + v_tendonmask[pts[i]]
    tendonmask[c] = sum_mask / 4.0


@wp.kernel
def update_attach_targets_kernel(
    cons: wp.array(dtype=Constraint),
    bone_pos_field: wp.array(dtype=wp.vec3),
    n_cons: int,
):
    c = wp.tid()
    if c >= n_cons:
        return
    ctype = cons[c].type
    if ctype == ATTACH:
        tgt_idx = cons[c].pts[2]
        if tgt_idx >= 0:
            target_pos = bone_pos_field[tgt_idx]
            rv = cons[c].restvector
            cons[c].restvector = wp.vec4f(target_pos[0], target_pos[1], target_pos[2], rv[3])
    elif ctype == DISTANCELINE:
        tgt_idx = cons[c].pts[2]  # pts layout: [src_idx, -1, tgt_idx, -1]
        if tgt_idx >= 0:
            target_pos = bone_pos_field[tgt_idx]
            rv = cons[c].restvector
            cons[c].restvector = wp.vec4f(target_pos[0], target_pos[1], target_pos[2], rv[3])
    elif ctype == PINTOTARGET:
        tgt_idx = cons[c].pts[2]
        if tgt_idx >= 0:
            target_pos = bone_pos_field[tgt_idx]
            rv = cons[c].restvector
            cons[c].restvector = wp.vec4f(target_pos[0], target_pos[1], target_pos[2], rv[3])


# ---------------------------------------------------------------------------
# Per-type constraint solver @wp.kernel
# ---------------------------------------------------------------------------

@wp.kernel
def solve_tetvolume_kernel(
    cons: wp.array(dtype=Constraint),
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
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    kstiffcompress = cons[c].compressionstiffness
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, cstiffness)
    pts = cons[c].pts
    tetid = cons[c].tetid
    tet_volume_update_xpbd_fn(
        use_jacobi, c, cons, pos, pprev,
        cons[c].restlength, dP, dPw, tetid, pts,
        dt, cstiffness, kstiffcompress, cons[c].dampingratio,
        mass, stopped, has_compress_stiffness)


@wp.kernel
def solve_tetfibernorm_kernel(
    cons: wp.array(dtype=Constraint),
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
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    pts = cons[c].pts
    tetid = cons[c].tetid
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
            c, cons, pts, dt, fiber_dir, stiffness_val, Dminv,
            cons[c].dampingratio, cons[c].restlength, cons[c].restvector,
            acti, mass, stopped, target_stretch)


@wp.kernel
def solve_tetfiberdgf_kernel(
    cons: wp.array(dtype=Constraint),
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
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    pts = cons[c].pts
    tetid = cons[c].tetid
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
    wTDminvT = wp.vec3(cons[c].restvector[0], cons[c].restvector[1], cons[c].restvector[2])
    FwT = _Ds * wTDminvT
    lm_tilde = wp.sqrt(wp.max(wp.length_sq(FwT), 1.0e-12))

    # DGF force-length: stiffness modulated by f_L(lm) for correct force profile.
    # restdir[0] = sigma0 (peak isometric stress)
    # restdir[1] = contraction_factor (from DGF curve inversion, set per-step)
    sigma0 = cons[c].restdir[0]
    contraction_factor = cons[c].restdir[1]
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
            c, cons, pts, dt, fiber_dir, stiffness_val, Dminv,
            cons[c].dampingratio, cons[c].restlength, cons[c].restvector,
            acti, mass, stopped, target_stretch)


@wp.kernel
def solve_tetfibermillard_kernel(
    cons: wp.array(dtype=Constraint),
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
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    pts = cons[c].pts
    tetid = cons[c].tetid
    acti = activation[tetid]
    sigma0 = cons[c].restdir[0]

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

    wTDminvT = wp.vec3(cons[c].restvector[0], cons[c].restvector[1], cons[c].restvector[2])
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
    stiffness_val = cstiffness * fiber_stiffness_scale
    restlength = cons[c].restlength  # V0 (tet volume)
    alpha = 1.0 / stiffness_val
    alpha = alpha / restlength
    alpha = alpha / (dt * dt)

    l = cons[c].L[0]  # accumulated Lagrange multiplier
    kdampratio = cons[c].dampingratio

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
        cons[c].L = wp.vec3(cons[c].L[0] + dL, cons[c].L[1], cons[c].L[2])


@wp.kernel
def solve_attach_kernel(
    cons: wp.array(dtype=Constraint),
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
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    kstiffcompress = cons[c].compressionstiffness
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, cstiffness)
    pts = cons[c].pts
    pt_src = pts[0]
    rv = cons[c].restvector
    p0_target = wp.vec3(rv[0], rv[1], rv[2])
    attach_bilateral_update_fn(
        use_jacobi, c, cons, pt_src, p0_target,
        pos, pprev, dP, dPw, mass, stopped,
        cons[c].restlength, cstiffness, cons[c].dampingratio,
        kstiffcompress, dt, has_compress_stiffness, reaction_accum)


@wp.kernel
def solve_pin_kernel(
    cons: wp.array(dtype=Constraint),
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
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    kstiffcompress = cons[c].compressionstiffness
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, cstiffness)
    pts = cons[c].pts
    pt_src = pts[0]
    rv = cons[c].restvector
    p0_target = wp.vec3(rv[0], rv[1], rv[2])
    p1_src = pos[pt_src]
    distance_pos_update_xpbd_fn(
        use_jacobi, c, cons, -1, pt_src,
        p0_target, p1_src,
        pos, pprev, dP, dPw, mass, stopped,
        cons[c].restlength, cstiffness, cons[c].dampingratio,
        kstiffcompress, dt, has_compress_stiffness)


@wp.kernel
def solve_pintotarget_kernel(
    cons: wp.array(dtype=Constraint),
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
    pts = cons[c].pts
    pt_src = pts[0]
    if pt_src < 0:
        return
    rv = cons[c].restvector
    rd = cons[c].restdir
    pos[pt_src] = wp.vec3(rv[0] + rd[0], rv[1] + rd[1], rv[2] + rd[2])


@wp.kernel
def solve_distanceline_kernel(
    cons: wp.array(dtype=Constraint),
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
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    kstiffcompress = cons[c].compressionstiffness
    kstiffcompress = wp.where(kstiffcompress >= 0.0, kstiffcompress, cstiffness)
    pts = cons[c].pts
    pt_src = pts[0]
    p_src = pos[pt_src]
    rv = cons[c].restvector
    line_origin = wp.vec3(rv[0], rv[1], rv[2])
    line_dir = cons[c].restdir

    p_projected = project_to_line_fn(p_src, line_origin, line_dir)

    distance_pos_update_xpbd_fn(
        use_jacobi, c, cons, -1, pt_src,
        p_projected, p_src,
        pos, pprev, dP, dPw, mass, stopped,
        cons[c].restlength, cstiffness, cons[c].dampingratio,
        kstiffcompress, dt, has_compress_stiffness)


# SVD-based ARAP (canonical PBD-XPBD form, ported from
# realtimeMuscle/engine/arap_hpbd.py:project_constraints).
# C = ||S - I||_F where F = D_s · B and B = restmatrix.
# Gradient: dC/dF = sum_k dC/dS_k · U[:,k] · V[:,k]^T. Sign-fixed SVD via
# ssvd_fn keeps the gradient pointing back to det F > 0 even when F is
# inverted, which the polar-form variant could not do.
@wp.kernel
def solve_tetarap_kernel(
    cons: wp.array(dtype=Constraint),
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
    cstiffness = cons[c].stiffness
    if cstiffness <= 0.0:
        return
    pts = cons[c].pts
    tetid = cons[c].tetid
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
    L0 = cons[c].L[0]
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
        cons[c].L = wp.vec3(L0 + dL, cons[c].L[1], cons[c].L[2])


@wp.func
def tet_snh_update_xpbd_fn(
    use_jacobi: int,
    cidx: int,
    cons: wp.array(dtype=Constraint),
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

    dL = cons[cidx].L[0]
    hL = cons[cidx].L[2]

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
        cons[cidx].L = wp.vec3(dL + ddL, cons[cidx].L[1], hL + hdL)


@wp.kernel
def solve_tetsnh_kernel(
    cons: wp.array(dtype=Constraint),
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
    mu = cons[c].stiffness
    if mu <= 0.0:
        return
    pts = cons[c].pts
    tetid = cons[c].tetid
    lam = cons[c].restdir[0]
    tet_snh_update_xpbd_fn(
        use_jacobi, c, cons, pts, dt, pos, pprev, dP, dPw,
        mass, stopped, cons[c].restlength, rest_matrix[tetid],
        mu, lam, cons[c].dampingratio)


# ---------------------------------------------------------------------------
# MuscleSim (Warp backend)
# ---------------------------------------------------------------------------

class MuscleSim(ConstraintBuilderMixin):
    """Warp-backed muscle simulator.

    Holds init + a hook-free step loop alongside all Warp kernels.
    Bone-coupled scenarios (LBS predictor → constraint corrector) compose
    their own per-substep loop using the public methods (integrate,
    solve_constraints, …) — see coupleOneWay/example_drive.py.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._init_backend()

        self.constraint_configs = self.cfg.constraints if self.cfg.constraints else []

        print("Loading mesh from:", cfg.geo_path)
        mesh_data = load_mesh(
            cfg.geo_path,
            prim_path=getattr(cfg, "geo_prim_path", None),
            prim_root=getattr(cfg, "geo_prim_root", None),
        )
        self.pos0_np, self.tet_np, self.v_fiber_np, self.v_tendonmask_np, self.geo = mesh_data
        self.n_verts = self.pos0_np.shape[0]
        print(f"Loaded mesh: {self.n_verts} vertices, {self.tet_np.shape[0]} tetrahedra.")

        print("Loading bone mesh:", cfg.bone_geo_path)
        self._load_bone(cfg.bone_geo_path)

        print("Allocating&initializing fields...")
        self._allocate_fields()
        self._init_fields()
        self._precompute_rest()
        self._build_surface_tris()
        # GPU parallelism: auto-enable Jacobi to avoid race conditions on CUDA
        is_gpu = self.pos.device.is_cuda
        self.use_jacobi = getattr(self.cfg, 'use_jacobi', is_gpu)
        self.use_colored_gs = getattr(self.cfg, 'use_colored_gs', False)
        self.build_constraints()
        self.contraction_ratio = getattr(self.cfg, 'contraction_ratio', 0.4)
        self.fiber_stiffness_scale = getattr(self.cfg, 'fiber_stiffness_scale', 10000.0)
        self.has_compressstiffness = getattr(self.cfg, 'HAS_compressstiffness', False)
        self.dt = self.cfg.dt / self.cfg.num_substeps
        self.step_cnt = 0

        self._init_renderer()
        print("All initialization done.")

    def _load_bone(self, target_path):
        """Load bone mesh data and create GPU fields."""
        if Path(target_path).exists():
            bone_root = getattr(self.cfg, "bone_root", "/character/bone")
            self.bone_geo, self.bone_pos, self.bone_indices_np, self.bone_muscle_ids = \
                load_bone_mesh(target_path, bone_root=bone_root)
            self._create_bone_fields()
        else:
            self.bone_geo = None
            self.bone_pos = np.zeros((0, 3), dtype=np.float32)
            self.bone_indices_np = np.zeros(0, dtype=np.int32)
            self.bone_muscle_ids = {}

    def load_bone_geo(self, target_path):
        """Return (bone_geo, bone_pos). Loads once, then returns cached data."""
        if not hasattr(self, 'bone_pos'):
            self._load_bone(target_path)
        return self.bone_geo, self.bone_pos

    def step(self):
        """Run one frame. Standalone (no bone coupling) substep loop."""
        self.update_attach_targets()
        debug_per_sub = bool(getattr(self.cfg, "debug_per_substep", False))
        n_sub = self.cfg.num_substeps
        for sub_i in range(n_sub):
            self.clear_forces()
            self.accumulate_active_fiber_force()
            self.integrate()
            # Hard-pin snap before solve so ARAP/vol see kinematic verts at
            # their new bone-driven position.
            if hasattr(self, "apply_pintotarget"):
                self.apply_pintotarget()
            self.clear()
            self.solve_constraints(run_extras=False)
            if self.use_jacobi:
                self.apply_dP()
            # Extras (collision) AFTER apply_dP — see _capturable_substep.
            self.solve_extras()
            self.post_smooth()
            self.freeze_near_inverted()
            self.repair_inverted_tets()
            if hasattr(self, "apply_pintotarget"):
                self.apply_pintotarget()
            self.update_velocities()
            if debug_per_sub:
                self._debug_check_substep(sub_i, n_sub)

    def _debug_check_substep(self, sub_i, n_sub):
        """Per-substep mesh-quality + state logging. cfg.debug_per_substep."""
        import logging
        from muscle.couple.mesh_utils import (
            check_mesh_quality, MeshDistortionError,
        )
        log = logging.getLogger(__name__)
        pos_np = self.pos.numpy()
        # rest_matrix was built against tet_np[:, [3,0,1,2]] ordering — reuse
        # so det(F) here matches the solver's internal F.
        tet_idx = self.tet_np[:, [3, 0, 1, 2]]
        tet_poses = self.rest_matrix.numpy()
        step_id = getattr(self, "step_cnt", -1)
        tag = f"step={step_id} sub={sub_i + 1}/{n_sub}"
        try:
            detF = check_mesh_quality(pos_np, tet_idx, tet_poses, step=None)
            detF_lo = float(detF.min())
            detF_hi = float(detF.max())
        except MeshDistortionError as exc:
            log.error(f"{tag} MeshDistortionError: {exc}")
            return
        centroid = pos_np.mean(axis=0)
        try:
            vol_err = float(self.calc_vol_error())
        except Exception:
            vol_err = float("nan")
        log.info(
            f"{tag}  centroid=({centroid[0]:+.4f},{centroid[1]:+.4f},{centroid[2]:+.4f})  "
            f"vol_err={vol_err:+.4f}  detF=[{detF_lo:.4f},{detF_hi:.4f}]"
        )

    def get_fps(self):
        if not hasattr(self, 'step_start_time') or not hasattr(self, 'step_end_time'):
            return 0.0
        dur = self.step_end_time - self.step_start_time
        return 1.0 / dur if dur != 0 else 0.0

    @classmethod
    def from_procedural(cls, vertices, tets, fiber_dirs_per_tet,
                        bone_targets=None, *, constraint_configs,
                        dts, num_substeps=1, device="cpu",
                        density=1060.0, veldamping=0.02, gravity=0.0,
                        sigma0=0.0, lambda_opt=1.0,
                        contraction_ratio=0.0, fiber_stiffness_scale=1.0,
                        v_max_norm=0.0, d_damp=0.0):
        """Create a MuscleSim from procedural mesh data (no file loading).

        Args:
            vertices: Vertex positions (N, 3) float.
            tets: Tet connectivity (M, 4) int.
            fiber_dirs_per_tet: Per-tet fiber directions (M, 3) float.
            bone_targets: Optional attachment target positions (K, 3) float.
            constraint_configs: List of constraint config dicts, e.g.
                [{"type": "volume", "stiffness": 1e6, "dampingratio": 0.1}, ...].
            dts: Frame timestep [s] (substep dt = dts / num_substeps).
            num_substeps: Number of substeps per frame.
            device: Warp device string ("cpu" or "cuda:0").
            density: Material density [kg/m^3].
            veldamping: Velocity damping coefficient.
            gravity: Gravity magnitude [m/s^2].
            sigma0: Peak isometric stress [Pa] for explicit active fiber force.
            lambda_opt: Optimal fiber stretch ratio (l_opt / L_ref).
            contraction_ratio: Initial contraction ratio.
            fiber_stiffness_scale: Scaling factor for fiber constraint stiffness.
            v_max_norm: Normalizer for fiber velocity (V_max * lambda_opt).
            d_damp: Linear damping coefficient for fiber velocity term.

        Returns:
            Initialized MuscleSim instance.
        """
        from types import SimpleNamespace

        vertices = np.asarray(vertices, dtype=np.float32)
        tets = np.asarray(tets, dtype=np.int32)
        n_v = len(vertices)

        # Average per-tet fiber dirs to per-vertex
        fiber_dirs_per_tet = np.asarray(fiber_dirs_per_tet, dtype=np.float32)
        vf = np.zeros((n_v, 3), dtype=np.float32)
        vc = np.zeros(n_v, dtype=np.float32)
        for e, t in enumerate(tets):
            for vi in t:
                vf[vi] += fiber_dirs_per_tet[e]; vc[vi] += 1.0
        vc = np.maximum(vc, 1.0)
        vf /= vc[:, None]
        vf /= np.maximum(np.linalg.norm(vf, axis=1, keepdims=True), 1e-8)

        cfg = SimpleNamespace(
            geo_path="<procedural>", bone_geo_path="<none>",
            gui=False, render_mode="none",
            constraints=list(constraint_configs),
            dt=dts, num_substeps=num_substeps,
            gravity=gravity, density=density, veldamping=veldamping,
            contraction_ratio=contraction_ratio,
            fiber_stiffness_scale=fiber_stiffness_scale,
            HAS_compressstiffness=False, arch=device,
            save_image=False, pause=False, reset=False,
            show_wireframe=False,
            render_fps=24, color_bones=False, color_muscles="tendonmask",
            activation=0.0, nsteps=1,
            sigma0=sigma0, lambda_opt=lambda_opt,
            v_max_norm=v_max_norm, d_damp=d_damp,
        )

        wp.set_device(device)
        sim = object.__new__(cls)
        sim.cfg = cfg
        sim.constraint_configs = cfg.constraints

        sim.pos0_np = vertices
        sim.tet_np = tets
        sim.v_fiber_np = vf
        sim.v_tendonmask_np = None
        sim.geo = SimpleNamespace()
        sim.n_verts = n_v

        if bone_targets is not None:
            sim.bone_pos = np.asarray(bone_targets, dtype=np.float32)
        else:
            sim.bone_pos = np.zeros((0, 3), dtype=np.float32)
        sim.bone_geo = None
        sim.bone_indices_np = np.zeros(0, dtype=np.int32)
        sim.bone_muscle_ids = {}

        wp.init()
        sim._init_backend()
        sim._allocate_fields()
        sim._init_fields()
        sim._precompute_rest()
        sim._build_surface_tris()
        sim._create_bone_fields()

        is_gpu = not device.startswith("cpu")
        sim.use_jacobi = is_gpu
        sim.use_colored_gs = False
        sim.contraction_ratio = contraction_ratio
        sim.fiber_stiffness_scale = fiber_stiffness_scale
        sim.has_compressstiffness = False
        sim.dt = dts / num_substeps
        sim.step_cnt = 0
        sim.renderer = None

        sim.build_constraints()
        return sim

    def rebuild_constraints(self, extra_constraints=None):
        """Rebuild constraint arrays, optionally appending extra constraints.

        Useful for adding ATTACH constraints after initial build.

        Args:
            extra_constraints: Optional list of constraint dicts to append.
        """
        all_cons = list(self.raw_constraints)
        if extra_constraints:
            all_cons.extend(extra_constraints)

        all_cons.sort(key=lambda c: c["type"])
        n = len(all_cons)
        self.n_cons = n
        self.cons_ranges = {}
        if n == 0:
            self.cons = wp.zeros(0, dtype=Constraint)
            self.reaction_accum = wp.zeros(1, dtype=wp.vec3)
            return

        prev, start = None, 0
        for i, c in enumerate(all_cons):
            c["cidx"] = i
            t = c["type"]
            if t != prev:
                if prev is not None:
                    self.cons_ranges[prev] = (start, i - start)
                start, prev = i, t
        self.cons_ranges[prev] = (start, n - start)

        dt_np = Constraint.numpy_dtype()
        arr = np.zeros(n, dtype=dt_np)
        for key in ("type", "pts", "stiffness", "dampingratio", "tetid",
                    "L", "restlength", "restvector", "restdir",
                    "compressionstiffness"):
            arr[key] = np.array([c[key] for c in all_cons])
        arr["cidx"] = np.arange(n, dtype=np.int32)
        self.cons = wp.array(arr, dtype=Constraint)
        self.reaction_accum = wp.zeros(max(n, 1), dtype=wp.vec3)

    def _init_backend(self):
        wp.init()

    def _build_surface_tris(self):
        self.surface_tris = build_surface_tris(self.tet_np)
        self.n_tris = self.surface_tris.shape[0]
        self.surface_tris_flat = self.surface_tris.reshape(-1).astype(np.int32)

    def _allocate_fields(self):
        n_v = self.n_verts
        n_tet = self.tet_np.shape[0]

        self.pos = wp.zeros(n_v, dtype=wp.vec3)
        self.pprev = wp.zeros(n_v, dtype=wp.vec3)
        self.pos0 = wp.zeros(n_v, dtype=wp.vec3)
        self.vel = wp.zeros(n_v, dtype=wp.vec3)
        self.force = wp.zeros(n_v, dtype=wp.vec3)
        self.mass = wp.zeros(n_v, dtype=wp.float32)
        self.stopped = wp.zeros(n_v, dtype=wp.int32)
        self.v_fiber_dir = wp.zeros(n_v, dtype=wp.vec3)
        self.dP = wp.zeros(n_v, dtype=wp.vec3)
        self.dPw = wp.zeros(n_v, dtype=wp.float32)
        self.tet_indices = wp.zeros(n_tet, dtype=wp.vec4i)

        self.rest_volume = wp.zeros(n_tet, dtype=wp.float32)
        self.rest_matrix = wp.zeros(n_tet, dtype=wp.mat33)
        self.activation = wp.zeros(n_tet, dtype=wp.float32)

        # LBS optional: filled by LBSDriver via set_lbs_T_rel / apply_lbs.
        self._lbs_T_rel_field = None

    def _create_bone_fields(self):
        if self.bone_pos.shape[0] > 0:
            self.bone_pos_field = wp.from_numpy(self.bone_pos.astype(np.float32), dtype=wp.vec3)
            if self.bone_indices_np.shape[0] > 0:
                self.bone_indices_field = wp.from_numpy(self.bone_indices_np.astype(np.int32), dtype=wp.int32)
            else:
                self.bone_indices_field = None
        else:
            self.bone_pos_field = None
            self.bone_indices_field = None

    def build_constraints(self):
        print("Building constraints...")
        all_constraints, _dt_collect = self._collect_raw_constraints()

        # Sort by type for per-type kernel dispatch
        all_constraints.sort(key=lambda c: c['type'])

        n_cons = len(all_constraints)
        self.n_cons = n_cons
        self.cons_ranges = {}
        if n_cons > 0:
            prev_type = None
            start_idx = 0
            for i, c in enumerate(all_constraints):
                c['cidx'] = i
                ctype = c['type']
                if ctype != prev_type:
                    if prev_type is not None:
                        self.cons_ranges[prev_type] = (start_idx, i - start_idx)
                    start_idx = i
                    prev_type = ctype
            if prev_type is not None:
                self.cons_ranges[prev_type] = (start_idx, n_cons - start_idx)

            type_arr = np.array([c['type'] for c in all_constraints], dtype=np.int32)
            cidx_arr = np.arange(n_cons, dtype=np.int32)
            pts_arr = np.array([c['pts'] for c in all_constraints], dtype=np.int32)
            stiffness_arr = np.array([c['stiffness'] for c in all_constraints], dtype=np.float32)
            dampingratio_arr = np.array([c['dampingratio'] for c in all_constraints], dtype=np.float32)
            tetid_arr = np.array([c['tetid'] for c in all_constraints], dtype=np.int32)
            L_arr = np.array([c['L'] for c in all_constraints], dtype=np.float32)
            restlength_arr = np.array([c['restlength'] for c in all_constraints], dtype=np.float32)
            restvector_arr = np.array([c['restvector'] for c in all_constraints], dtype=np.float32)
            restdir_arr = np.array([c['restdir'] for c in all_constraints], dtype=np.float32)
            compressionstiffness_arr = np.array([c['compressionstiffness'] for c in all_constraints], dtype=np.float32)

            cons_np = np.zeros(n_cons, dtype=Constraint.numpy_dtype())
            cons_np['type'] = type_arr
            cons_np['cidx'] = cidx_arr
            cons_np['pts'] = pts_arr
            cons_np['stiffness'] = stiffness_arr
            cons_np['dampingratio'] = dampingratio_arr
            cons_np['tetid'] = tetid_arr
            cons_np['L'] = L_arr
            cons_np['restlength'] = restlength_arr
            cons_np['restvector'] = restvector_arr
            cons_np['restdir'] = restdir_arr
            cons_np['compressionstiffness'] = compressionstiffness_arr

            self.cons = wp.array(cons_np, dtype=Constraint)
        else:
            self.cons = wp.zeros(0, dtype=Constraint)
            self.raw_constraints = []

        print(f"Built {n_cons} constraints total. [{_dt_collect*1000:.0f}ms]")

        self.reaction_accum = wp.zeros(max(n_cons, 1), dtype=wp.vec3)

        # Initialize Millard curve arrays if needed (for constraints or active force)
        needs_millard = (TETFIBERMILLARD in self.cons_ranges
                         or getattr(self.cfg, 'sigma0', 0.0) > 0)
        if needs_millard:
            self._init_millard_curves()

        # Initialize explicit active fiber force if sigma0 is configured
        sigma0 = getattr(self.cfg, 'sigma0', 0.0)
        if sigma0 > 0:
            self.sigma0 = sigma0
            self.lambda_opt = getattr(self.cfg, 'lambda_opt', 1.0)
            self.v_max_norm = getattr(self.cfg, 'v_max_norm', 0.0)
            self.d_damp = getattr(self.cfg, 'd_damp', 0.0)
            self._has_active_fiber_force = True
            print(f"  Active fiber force: sigma0={sigma0:.0f} Pa, "
                  f"lambda_opt={self.lambda_opt:.3f}, "
                  f"v_max_norm={self.v_max_norm:.3f}, d_damp={self.d_damp:.2f}")
        else:
            self.v_max_norm = 0.0
            self.d_damp = 0.0
            self._has_active_fiber_force = False

        # Build colored constraint groups for parallel Gauss-Seidel
        if self.use_colored_gs and n_cons > 0:
            self._build_colored_gs(all_constraints)

    def _init_millard_curves(self):
        """Initialize Millard 2012 curve coefficient arrays for GPU evaluation."""
        from muscle.millard_curves import MillardCurves

        mc = MillardCurves()
        self._millard_curves = mc  # keep reference for CPU-side equilibrium solve

        def _upload_curve(curve):
            """Upload MillardCurve data to Warp arrays."""
            n_seg = len(curve.segments)
            x_coeffs = np.zeros((n_seg, 6), dtype=np.float32)
            y_coeffs = np.zeros((n_seg, 6), dtype=np.float32)
            F_coeffs = np.zeros((n_seg, 11), dtype=np.float32)
            seg_bounds = np.zeros(n_seg + 1, dtype=np.float32)

            for i, seg in enumerate(curve.segments):
                x_coeffs[i] = seg.x_coeffs.astype(np.float32)
                y_coeffs[i] = seg.y_coeffs.astype(np.float32)
                F_coeffs[i] = seg.F_coeffs.astype(np.float32)
                seg_bounds[i] = np.float32(seg.x_start)
            seg_bounds[n_seg] = np.float32(curve.segments[-1].x_end)

            return {
                'x_coeffs': wp.from_numpy(x_coeffs.flatten(), dtype=wp.float32),
                'y_coeffs': wp.from_numpy(y_coeffs.flatten(), dtype=wp.float32),
                'F_coeffs': wp.from_numpy(F_coeffs.flatten(), dtype=wp.float32),
                'seg_bounds': wp.from_numpy(seg_bounds, dtype=wp.float32),
                'n_segments': n_seg,
                'x_lo': curve.x_lo,
                'x_hi': curve.x_hi,
                'y_lo': curve.y_lo,
                'y_hi': curve.y_hi,
                'dydx_lo': curve.dydx_lo,
                'dydx_hi': curve.dydx_hi,
            }

        fl = _upload_curve(mc.fl)
        self._millard_fl_x_coeffs = fl['x_coeffs']
        self._millard_fl_y_coeffs = fl['y_coeffs']
        self._millard_fl_F_coeffs = fl['F_coeffs']
        # Energy offset: -eval_integral(0) so that Psi(lm) + offset > 0 for all lm > 0
        self._millard_fl_energy_offset = float(-mc.fl.eval_integral(0.0))
        self._millard_fl_seg_bounds = fl['seg_bounds']
        self._millard_fl_n_segments = fl['n_segments']
        self._millard_fl_x_lo = fl['x_lo']
        self._millard_fl_x_hi = fl['x_hi']
        self._millard_fl_y_lo = fl['y_lo']
        self._millard_fl_y_hi = fl['y_hi']
        self._millard_fl_dydx_lo = fl['dydx_lo']
        self._millard_fl_dydx_hi = fl['dydx_hi']

        fpe = _upload_curve(mc.fpe)
        self._millard_fpe_x_coeffs = fpe['x_coeffs']
        self._millard_fpe_y_coeffs = fpe['y_coeffs']
        self._millard_fpe_F_coeffs = fpe['F_coeffs']
        self._millard_fpe_seg_bounds = fpe['seg_bounds']
        self._millard_fpe_n_segments = fpe['n_segments']
        self._millard_fpe_x_lo = fpe['x_lo']
        self._millard_fpe_x_hi = fpe['x_hi']
        self._millard_fpe_y_lo = fpe['y_lo']
        self._millard_fpe_y_hi = fpe['y_hi']
        self._millard_fpe_dydx_lo = fpe['dydx_lo']
        self._millard_fpe_dydx_hi = fpe['dydx_hi']

        print(f"  Millard curves initialized: f_L ({mc.fl.segments.__len__()} seg), "
              f"f_PE ({mc.fpe.segments.__len__()} seg)")

    def _build_colored_gs(self, all_constraints):
        """Reorder constraints by (color, type) and build per-color dispatch ranges."""
        from .constraints import build_constraint_color_groups
        import time

        t0 = time.perf_counter()
        color_groups = build_constraint_color_groups(all_constraints)
        n_colors = len(color_groups)
        print(f"  Graph coloring: {n_colors} colors, "
              f"sizes: {[len(g) for g in color_groups]} "
              f"[{(time.perf_counter()-t0)*1000:.0f}ms]")

        # Reorder constraints: (color, type) sorted
        reordered = []
        self.color_type_ranges = []
        global_offset = 0
        for color_idx, group in enumerate(color_groups):
            color_cons = [all_constraints[i] for i in group]
            color_cons.sort(key=lambda c: c['type'])

            type_ranges = {}
            prev_type = None
            start = 0
            for i, c in enumerate(color_cons):
                ctype = c['type']
                if ctype != prev_type:
                    if prev_type is not None:
                        type_ranges[prev_type] = (global_offset + start, i - start)
                    start = i
                    prev_type = ctype
            if prev_type is not None:
                type_ranges[prev_type] = (global_offset + start, len(color_cons) - start)

            self.color_type_ranges.append(type_ranges)
            reordered.extend(color_cons)
            global_offset += len(color_cons)

        # Rebuild constraint array with new order
        n_cons = len(reordered)
        for i, c in enumerate(reordered):
            c['cidx'] = i

        cons_np = np.zeros(n_cons, dtype=Constraint.numpy_dtype())
        cons_np['type'] = np.array([c['type'] for c in reordered], dtype=np.int32)
        cons_np['cidx'] = np.arange(n_cons, dtype=np.int32)
        cons_np['pts'] = np.array([c['pts'] for c in reordered], dtype=np.int32)
        cons_np['stiffness'] = np.array([c['stiffness'] for c in reordered], dtype=np.float32)
        cons_np['dampingratio'] = np.array([c['dampingratio'] for c in reordered], dtype=np.float32)
        cons_np['tetid'] = np.array([c['tetid'] for c in reordered], dtype=np.int32)
        cons_np['L'] = np.array([c['L'] for c in reordered], dtype=np.float32)
        cons_np['restlength'] = np.array([c['restlength'] for c in reordered], dtype=np.float32)
        cons_np['restvector'] = np.array([c['restvector'] for c in reordered], dtype=np.float32)
        cons_np['restdir'] = np.array([c['restdir'] for c in reordered], dtype=np.float32)
        cons_np['compressionstiffness'] = np.array([c['compressionstiffness'] for c in reordered], dtype=np.float32)
        self.cons = wp.array(cons_np, dtype=Constraint)

    def _init_fields(self):
        self.pos0 = wp.from_numpy(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.pos = wp.from_numpy(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.vel.zero_()
        self.force.zero_()
        self.mass.zero_()
        self.stopped.zero_()
        self.tet_indices = wp.from_numpy(self.tet_np.astype(np.int32), dtype=wp.vec4i)

        if self.v_fiber_np is None:
            self.v_fiber_np = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (self.n_verts, 1))
        self.v_fiber_dir = wp.from_numpy(self.v_fiber_np.astype(np.float32), dtype=wp.vec3)

        n_tet = self.tet_np.shape[0]
        self.tendonmask = wp.zeros(n_tet, dtype=wp.float32)
        if self.v_tendonmask_np is not None:
            v_tendonmask_wp = wp.from_numpy(self.v_tendonmask_np.astype(np.float32), dtype=wp.float32)
            wp.launch(compute_cell_tendon_mask_kernel, dim=n_tet,
                      inputs=[self.tet_indices, v_tendonmask_wp, self.tendonmask])

        self.activation.zero_()
        self.total_rest_volume = wp.zeros(1, dtype=wp.float32)

        print("Initialized fields done.")

    def _precompute_rest(self):
        n_tet = self.tet_np.shape[0]
        self.total_rest_volume.zero_()
        self.mass.zero_()
        wp.launch(precompute_rest_kernel, dim=n_tet,
                  inputs=[self.pos0, self.tet_indices, self.rest_volume,
                          self.rest_matrix, self.mass, self.cfg.density,
                          self.total_rest_volume])
        # Clamp vertex mass to prevent instability from degenerate tets.
        mass_floor = getattr(self.cfg, 'mass_floor', 0.0)
        if mass_floor <= 0.0 and getattr(self.cfg, 'sigma0', 0.0) > 0.0:
            mass_np = self.mass.numpy()
            mass_floor = float(np.median(mass_np[mass_np > 0])) * 0.01
        if mass_floor > 0.0:
            mass_np = self.mass.numpy()
            n_clamped = int((mass_np < mass_floor).sum())
            if n_clamped > 0:
                mass_np = np.maximum(mass_np, mass_floor)
                self.mass = wp.from_numpy(mass_np, dtype=wp.float32)
                print(f"  Mass floor applied: {mass_floor:.2e} kg ({n_clamped} vertices clamped)")

    def _init_renderer(self):
        self.renderer = None
        if self.cfg.gui and self.cfg.render_mode in ("human", "usd"):
            if self.cfg.render_mode == "usd":
                stage_path = getattr(self.cfg, "output_path", None) or "muscle_sim.usd"
                os.makedirs(os.path.dirname(stage_path) or ".", exist_ok=True)
            else:
                stage_path = None
            self.renderer = WarpRenderer(mode=self.cfg.render_mode, stage_path=stage_path)

    def reset(self):
        self.pos = wp.from_numpy(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.pprev = wp.from_numpy(self.pos0_np.astype(np.float32), dtype=wp.vec3)
        self.vel.zero_()
        self.force.zero_()
        self.activation.zero_()
        self.clear()
        self.step_cnt = 1

    def update_attach_targets(self):
        if (
            hasattr(self, "bone_pos_field")
            and self.bone_pos_field is not None
            and self.bone_pos_field.shape[0] > 0
            and self.n_cons > 0
            and (
                len(getattr(self, "attach_constraints", []) or []) > 0
                or len(getattr(self, "pintotarget_constraints", []) or []) > 0
                or len(getattr(self, "distanceline_constraints", []) or []) > 0
            )
        ):
            wp.launch(update_attach_targets_kernel, dim=self.n_cons,
                      inputs=[self.cons, self.bone_pos_field, self.n_cons])

    def integrate(self):
        gravity_vec = getattr(self, '_gravity_vec', None)
        if gravity_vec is None:
            # Default: Y-axis gravity (legacy behavior)
            gravity_vec = wp.vec3(0.0, self.cfg.gravity, 0.0)
        wp.launch(integrate_kernel, dim=self.n_verts,
                  inputs=[self.pos, self.pprev, self.vel,
                          self.force, self.mass, self.stopped,
                          gravity_vec, self.cfg.veldamping, self.dt])

    def clear_forces(self):
        """Zero the per-vertex force accumulator."""
        self.force.zero_()

    def accumulate_active_fiber_force(self):
        """Compute explicit active fiber force per tet, scatter to vertex force field.

        Uses tau = sigma0 * (a * f_FL(r) * f_V(v_norm) + d_damp * v_norm)
        where r = lambda_f / lambda_opt, v_norm = dlm_dt / v_max_norm.
        See docs/notes/xpbd_fem_fiber_lm.md for derivation.
        """
        if not getattr(self, '_has_active_fiber_force', False):
            return
        n_tet = self.tet_np.shape[0]
        wp.launch(accumulate_active_fiber_force_kernel, dim=n_tet, inputs=[
            self.tet_indices, self.pos, self.vel, self.force,
            self.rest_matrix, self.rest_volume,
            self.v_fiber_dir, self.activation,
            self.sigma0, self.lambda_opt,
            self.v_max_norm, self.d_damp,
            self._millard_fl_x_coeffs, self._millard_fl_y_coeffs,
            self._millard_fl_seg_bounds, self._millard_fl_n_segments,
            self._millard_fl_x_lo, self._millard_fl_x_hi,
            self._millard_fl_y_lo, self._millard_fl_y_hi,
            self._millard_fl_dydx_lo, self._millard_fl_dydx_hi,
        ])
        # Clamp per-vertex acceleration to prevent instability on degenerate tets.
        max_accel = getattr(self.cfg, 'max_accel', 2000.0)
        if max_accel > 0.0:
            wp.launch(_clamp_force_by_accel_kernel, dim=self.n_verts,
                      inputs=[self.force, self.mass, wp.float32(max_accel)])

    def save_predicted_positions(self, vertex_ids):
        """Save predicted positions after integrate() and before solve()."""
        self._predicted_pos = self.pos.numpy()[vertex_ids].copy()
        self._predicted_ids = vertex_ids

    def get_insertion_force(self, insertion_vids, dts):
        """Get net muscle force on bone from insertion constraint correction.

        Requires save_predicted_positions() called after integrate().
        """
        pos_np = self.pos.numpy()
        corrected = pos_np[insertion_vids]
        predicted = self._predicted_pos

        mass_np = self.mass.numpy()
        masses = mass_np[insertion_vids]
        dts2 = dts * dts

        corrections = corrected.astype(np.float64) - predicted.astype(np.float64)
        forces = masses[:, None].astype(np.float64) * corrections / dts2

        return (-forces.sum(axis=0)).astype(np.float32)

    def update_velocities(self):
        is_collide = getattr(self, "is_collide", None)
        contact_normal = getattr(self, "contact_normal", None)
        if is_collide is None or contact_normal is None:
            wp.launch(update_velocities_kernel, dim=self.n_verts,
                      inputs=[self.pos, self.pprev, self.vel, self.dt])
            return
        rn = float(getattr(self, "restitution_n", 0.0))
        rt = float(getattr(self, "restitution_t", 1.0))
        wp.launch(update_velocities_with_contact_kernel, dim=self.n_verts,
                  inputs=[self.pos, self.pprev, self.vel, self.dt,
                          is_collide, contact_normal, rn, rt])

    def set_lbs_T_rel(self, T_rel_flat_np):
        """Upload (n_bones * 12,) float32 row-major T_rel buffer to GPU.

        Allocates the Warp array on first call.
        """
        if getattr(self, "_lbs_T_rel_field", None) is None:
            self._lbs_T_rel_field = wp.from_numpy(
                T_rel_flat_np.astype(np.float32), dtype=wp.float32)
        else:
            self._lbs_T_rel_field.assign(T_rel_flat_np.astype(np.float32))

    def apply_lbs(self, vertex_bone_field, blend: float = 1.0):
        """Overwrite pos with LBS target = pos0 @ T_rel[vertex_bone[v]],
        lerped from current pos by ``blend``.

        Caller must have called ``set_lbs_T_rel`` first.
        """
        if self._lbs_T_rel_field is None:
            return
        wp.launch(lbs_apply_kernel, dim=self.n_verts,
                  inputs=[self.pos, self.pos0, vertex_bone_field,
                          self._lbs_T_rel_field, float(blend)])

    def repair_inverted_tets(self):
        """SVD-based inversion repair: clamp F singular values and project positions.

        Uses Jacobi-style accumulation (like XPBD) to avoid race conditions
        when multiple tets share vertices.  Each iteration: clear dP/dPw →
        accumulate SVD clamp corrections → apply weighted average.
        """
        repair_alpha = getattr(self.cfg, 'repair_alpha', 0.0)
        repair_iters = int(getattr(self.cfg, 'repair_iters', 0))
        sigma_min = getattr(self.cfg, 'repair_sigma_min', 0.1)
        if repair_alpha <= 0.0 or repair_iters <= 0:
            return
        n_tet = self.tet_np.shape[0]
        for _ in range(repair_iters):
            # Clear accumulation buffers (reuse XPBD dP/dPw)
            wp.launch(clear_dP_kernel, dim=self.n_verts,
                      inputs=[self.dP, self.dPw])
            # Accumulate SVD clamp corrections
            wp.launch(_svd_clamp_accumulate_kernel, dim=n_tet,
                      inputs=[self.tet_indices, self.pos, self.rest_matrix,
                              self.stopped, self.dP, self.dPw,
                              wp.float32(sigma_min)])
            # Apply weighted average with blending factor alpha
            wp.launch(_apply_svd_clamp_kernel, dim=self.n_verts,
                      inputs=[self.pos, self.dP, self.dPw,
                              wp.float32(repair_alpha)])

    def calc_vol_error(self):
        total_vol = wp.zeros(1, dtype=wp.float32)
        n_tet = self.tet_np.shape[0]
        wp.launch(calc_vol_error_kernel, dim=n_tet,
                  inputs=[self.tet_indices, self.pos, self.rest_volume, total_vol])
        total_vol_np = total_vol.numpy()[0]
        total_rest_np = self.total_rest_volume.numpy()[0]
        if total_rest_np == 0.0:
            return 0.0
        return (total_vol_np - total_rest_np) / total_rest_np

    def clear(self):
        wp.launch(clear_dP_kernel, dim=self.n_verts,
                  inputs=[self.dP, self.dPw])
        if self.n_cons > 0:
            wp.launch(clear_cons_L_kernel, dim=self.n_cons,
                      inputs=[self.cons])

    def clear_reaction(self):
        if self.n_cons > 0:
            wp.launch(clear_reaction_kernel, dim=max(self.n_cons, 1),
                      inputs=[self.reaction_accum])

    def apply_dP(self):
        wp.launch(apply_dP_kernel, dim=self.n_verts,
                  inputs=[self.pos, self.dP, self.dPw])

    def apply_pintotarget(self):
        """Hard pin pass: directly snap pinned source verts to bone target.

        Runs AFTER apply_dP so the snap is final — apply_dP would otherwise
        add accumulated Jacobi deltas on top and undo the pin (NaN on GPU).
        """
        if not getattr(self, "pintotarget_constraints", None):
            return
        rng = self.cons_ranges.get(PINTOTARGET) if hasattr(self, "cons_ranges") else None
        if rng is None:
            return
        offset, count = rng
        if count <= 0:
            return
        if not getattr(self, "_pintotarget_stopped_marked", False):
            self._mark_pintotarget_stopped()
            self._pintotarget_stopped_marked = True
        wp.launch(solve_pintotarget_kernel, dim=count,
                  inputs=[self.cons, self.pos, offset])

    def _mark_pintotarget_stopped(self):
        """Mark all pintotarget source verts as `stopped=1`.

        Effect: integrate_kernel skips them (no velocity, no pos drift), and
        constraint solvers see inv_mass=0 (they act as kinematic anchors).
        Position is then driven *only* by solve_pintotarget_kernel each substep.
        Without this, apply_pintotarget creates huge artificial velocities
        (pinned vert jumps several cm per substep) which integrate then over-
        shoots — fatal under GMG (restrict feeds the overshoot into coarse).
        """
        src_idx = np.array([c["pts"][0] for c in self.pintotarget_constraints],
                           dtype=np.int32)
        if src_idx.size == 0:
            return
        stopped_np = self.stopped.numpy()
        stopped_np[src_idx] = 1
        self.stopped = wp.from_numpy(stopped_np, dtype=wp.int32)

    def _dispatch_constraints(self, type_ranges):
        """Dispatch constraint solver kernels for a given {type: (offset, count)} mapping."""
        has_compress = 1 if self.has_compressstiffness else 0
        use_jacobi_int = 1 if self.use_jacobi else 0
        for ctype, (offset, count) in type_ranges.items():
            if ctype == TETVOLUME:
                wp.launch(solve_tetvolume_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.dt, use_jacobi_int, has_compress, offset])
            elif ctype == TETFIBERNORM:
                wp.launch(solve_tetfibernorm_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix, self.v_fiber_dir,
                    self.activation, self.tendonmask,
                    self.dt, use_jacobi_int,
                    self.contraction_ratio, self.fiber_stiffness_scale,
                    offset])
            elif ctype == ATTACH:
                wp.launch(solve_attach_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.reaction_accum,
                    self.dt, use_jacobi_int, has_compress, offset])
            elif ctype == PIN:
                wp.launch(solve_pin_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.dt, use_jacobi_int, has_compress, offset])
            elif ctype == PINTOTARGET:
                # Skipped here — pintotarget is a kinematic post-pass run by
                # apply_pintotarget() AFTER apply_dP, so Jacobi accumulation
                # cannot undo the snap.
                continue
            elif ctype == DISTANCELINE:
                wp.launch(solve_distanceline_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.dt, use_jacobi_int, has_compress, offset])
            elif ctype == TETARAP:
                wp.launch(solve_tetarap_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix,
                    self.dt, use_jacobi_int, offset])
            elif ctype == TETFIBERDGF:
                wp.launch(solve_tetfiberdgf_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix, self.v_fiber_dir,
                    self.activation, self.tendonmask,
                    self.dt, use_jacobi_int,
                    self.fiber_stiffness_scale,
                    offset])
            elif ctype == TETFIBERMILLARD:
                wp.launch(solve_tetfibermillard_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix, self.v_fiber_dir,
                    self.activation, self.tendonmask,
                    self.dt, use_jacobi_int,
                    self.fiber_stiffness_scale,
                    offset,
                    # Millard f_L curve data (force + energy)
                    self._millard_fl_x_coeffs, self._millard_fl_y_coeffs,
                    self._millard_fl_seg_bounds, self._millard_fl_n_segments,
                    self._millard_fl_x_lo, self._millard_fl_x_hi,
                    self._millard_fl_y_lo, self._millard_fl_y_hi,
                    self._millard_fl_dydx_lo, self._millard_fl_dydx_hi,
                    self._millard_fl_F_coeffs,
                    self._millard_fl_energy_offset,
                    # Millard f_PE curve data
                    self._millard_fpe_x_coeffs, self._millard_fpe_y_coeffs,
                    self._millard_fpe_seg_bounds, self._millard_fpe_n_segments,
                    self._millard_fpe_x_lo, self._millard_fpe_x_hi,
                    self._millard_fpe_y_lo, self._millard_fpe_y_hi,
                    self._millard_fpe_dydx_lo, self._millard_fpe_dydx_hi,
                    float(getattr(self.cfg, 'fiber_skip_detF', 0.0)),
                    ])
            elif ctype == TETSNH:
                wp.launch(solve_tetsnh_kernel, dim=count, inputs=[
                    self.cons, self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix,
                    self.dt, use_jacobi_int, offset])

    def solve_constraints(self, run_extras: bool = True):
        """Dispatch built-in constraints; optionally also run registered
        ``constraint_solvers`` extras.

        ``run_extras=False`` is used when the caller wants the extras
        (typically collision) to run AFTER ``apply_dP`` so they push against
        the post-ARAP geometry rather than competing with the stale
        pre-solve gradient. The caller should follow up with
        ``solve_extras()``.
        """
        extras = getattr(self, "constraint_solvers", None)
        if self.n_cons == 0 and (not run_extras or not extras):
            return
        if self.n_cons > 0:
            if self.use_colored_gs and hasattr(self, 'color_type_ranges'):
                for type_ranges in self.color_type_ranges:
                    self._dispatch_constraints(type_ranges)
            else:
                self._dispatch_constraints(self.cons_ranges)
        if run_extras and extras:
            for fn in extras:
                fn()
            if not self.use_jacobi:
                self.apply_dP()

    def solve_extras(self):
        """Run only the registered ``constraint_solvers`` (e.g., collision).
        Intended to be called AFTER ``apply_dP`` so extras see post-ARAP
        positions. In GS mode (``use_jacobi=False``) the built-in path's
        apply_dP has already been invoked by the caller, so here we just
        run extras and (for GS) flush any extras-generated dP again.
        """
        extras = getattr(self, "constraint_solvers", None)
        if not extras:
            return
        for fn in extras:
            fn()
        if not self.use_jacobi:
            self.apply_dP()

    # Fiber constraint types to exclude during post-smoothing
    _FIBER_TYPES = frozenset([TETFIBERNORM, TETFIBERDGF, TETFIBERMILLARD])

    def post_smooth(self):
        """Jacobi post-smoothing pass: solve only non-fiber constraints
        (volume, ARAP, attach) to recover mesh quality after fiber contraction.
        Runs post_smooth_iters iterations, each doing clear→solve→apply.

        On GPU, forces Jacobi mode to avoid race conditions (post_smooth does
        NOT use colored GS).  On CPU, respects the global use_jacobi setting.
        """
        n_iters = int(getattr(self.cfg, 'post_smooth_iters', 0))
        if n_iters <= 0 or self.n_cons == 0:
            return
        # Build filtered type_ranges once and cache
        if not hasattr(self, '_smooth_ranges'):
            self._smooth_ranges = {
                k: v for k, v in self.cons_ranges.items()
                if k not in self._FIBER_TYPES
            }
        # Force Jacobi on GPU to prevent race conditions
        need_jacobi = self.use_jacobi or self.pos.device.is_cuda
        saved_jacobi = self.use_jacobi
        if need_jacobi:
            self.use_jacobi = True
        for _ in range(n_iters):
            self.clear()
            self._dispatch_constraints(self._smooth_ranges)
            if self.use_jacobi:
                self.apply_dP()
            self.apply_pintotarget()
        self.use_jacobi = saved_jacobi

    def freeze_near_inverted(self):
        """Revert vertices of near-inverted tets to their pre-substep positions.
        Controlled by cfg.freeze_detF_threshold (default 0 = disabled).

        WARNING: Not GPU-safe (race condition on shared vertices). Skipped on
        CUDA devices with a one-time warning.
        """
        threshold = getattr(self.cfg, 'freeze_detF_threshold', 0.0)
        if threshold <= 0.0:
            return
        device = self.pos.device
        if device.is_cuda:
            if not getattr(self, '_freeze_gpu_warned', False):
                import logging
                logging.getLogger(__name__).warning(
                    "freeze_near_inverted skipped on GPU (race condition on shared vertices)")
                self._freeze_gpu_warned = True
            return
        n_tet = self.tet_np.shape[0]
        wp.launch(_freeze_near_inverted_kernel, dim=n_tet,
                  inputs=[self.tet_indices, self.pos, self.pprev,
                          self.rest_matrix, self.stopped,
                          wp.float32(threshold)])

    def _capturable_substep(self):
        """One muscle PBD substep — all Warp kernels, no CPU sync.

        Safe for CUDA graph capture: no .numpy(), no wp.from_numpy(), no
        wp.array() allocations. All conditionals (use_jacobi, use_colored_gs,
        post_smooth_iters, etc.) are resolved at capture time.
        """
        self.clear_forces()
        # self.accumulate_active_fiber_force()
        self.integrate()
        self.clear()
        self.update_attach_targets()
        # self.clear_reaction()
        self.solve_constraints(run_extras=False)
        if self.use_jacobi:
            self.apply_dP()
        # Extras (collision) AFTER apply_dP so ARAP/volume settles first;
        # otherwise the stale pre-solve ARAP gradient gets added on top of
        # the collision push and pulls penetrated verts back inward.
        self.solve_extras()
        self.post_smooth()
        # self.freeze_near_inverted()
        # self.repair_inverted_tets()
        self.update_velocities()

    def render(self):
        if self.renderer is None:
            return
        pos_np = self.pos.numpy()
        bone_pos_np = None
        bone_idx = None
        if hasattr(self, 'bone_pos_field') and self.bone_pos_field is not None:
            bone_pos_np = self.bone_pos_field.numpy()
            bone_idx = self.bone_indices_np if hasattr(self, 'bone_indices_np') else None
        sim_time = self.step_cnt * self.cfg.dt
        self.renderer.update(sim_time, pos_np, self.surface_tris_flat, bone_pos_np, bone_idx)

    def run(self):
        self.step_cnt = 1

        if self.renderer is not None:
            def on_key(symbol, modifiers):
                import pyglet
                if symbol == pyglet.window.key.R:
                    self.reset()
                    return pyglet.event.EVENT_HANDLED
                if symbol == pyglet.window.key.SPACE:
                    self.cfg.pause = not self.cfg.pause
                    return pyglet.event.EVENT_HANDLED
            self.renderer.register_key_press_callback(on_key)

        while self.step_cnt <= self.cfg.nsteps:
            if self.renderer is not None and not self.renderer.is_running():
                break
            if self.cfg.reset:
                self.reset()
                self.step_cnt = 1
                self.cfg.reset = False
            if not self.cfg.pause:
                wp.launch(fill_float_kernel, dim=self.activation.shape[0],
                          inputs=[self.activation, self.cfg.activation])
                self.step_start_time = time.perf_counter()
                self.step()
                self.step_cnt += 1
                self.step_end_time = time.perf_counter()
            self.render()

        if self.renderer is not None:
            self.renderer.save()


def main():
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "data/muscle/config/bicep.json"
    print("Using config:", config_path)
    cfg = load_config(config_path)
    cfg.gui = False
    sim = MuscleSim(cfg)
    print("Running for", cfg.nsteps, "steps.")
    sim.run()


if __name__ == "__main__":
    main()
