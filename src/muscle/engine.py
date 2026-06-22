"""XPBD muscle solver core: host helpers, non-solve Warp kernels, and the XpbdEngine base class.

Constraint-solve kernels live in ``muscle.constraint.solve_kernels`` (imported via *).
MuscleSim / MassSpringSim live in ``muscle.musclesim`` / ``muscle.massspringsim``.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import warp as wp
import warp.render

from muscle.util.util import SimConfig
from muscle.util.mesh_io import build_surface_tris, load_mesh, load_bone_mesh
from muscle.fiber.dgf import (
    dgf_active_force_length_wp,
    dgf_passive_force_length_wp,
    dgf_force_velocity_wp,
)
from muscle.fiber.millard import (
    MillardCurve,
    MillardCurves,
    millard_equilibrium_fiber_length,
    millard_eval_wp,
    millard_energy_eval_wp,
)

from muscle.defs import ConstraintType, FemFlag
from muscle.constraint.builder import ConstraintBuilderMixin
from muscle.constraint.solve_kernels import *


def build_constraint_color_groups(all_constraints):
    """Build graph-colored constraint groups for parallel Gauss-Seidel.

    Two constraints that share any vertex (pts[i] >= 0) are adjacent in the
    dual graph and cannot be in the same color group. Returns a list of
    np.ndarray, each containing constraint indices of one color.
    """
    import warp as wp
    from collections import defaultdict
    from newton._src.sim.graph_coloring import ColoringAlgorithm, color_graph

    n_cons = len(all_constraints)
    if n_cons == 0:
        return []

    # vertex -> constraint id inverted index
    vertex_to_cons = defaultdict(list)
    for cid, c in enumerate(all_constraints):
        for v in c['pts']:
            if v >= 0:
                vertex_to_cons[v].append(cid)

    # build dual graph edges
    edge_set = set()
    for cids in vertex_to_cons.values():
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = cids[i], cids[j]
                edge_set.add((min(a, b), max(a, b)))

    if not edge_set:
        return [np.arange(n_cons, dtype=np.int32)]

    edges = np.array(sorted(edge_set), dtype=np.int32)
    edge_wp = wp.array(edges, dtype=int, device="cpu")
    color_groups = color_graph(
        n_cons, edge_wp,
        balance_colors=True,
        target_max_min_color_ratio=1.1,
        algorithm=ColoringAlgorithm.MCS,
    )
    return color_groups




# ===========================================================================
# Warp renderer (moved from vis_warp.py)
# ===========================================================================

class WarpRenderer:
    """Wraps wp.render.OpenGLRenderer or wp.render.UsdRenderer."""

    def __init__(self, mode: str = "human", stage_path: str = "muscle_sim.usd"):
        self.mode = mode
        self.renderer = None
        if mode == "human":
            self.renderer = wp.render.OpenGLRenderer(vsync=True)
        elif mode == "usd":
            self.renderer = wp.render.UsdRenderer(stage_path)
        self.sim_time = 0.0

    def is_running(self) -> bool:
        if self.renderer is None:
            return True
        if self.mode == "human":
            return self.renderer.is_running()
        return True

    def update(self, sim_time: float, pos_np: np.ndarray, tri_indices: np.ndarray,
               bone_pos_np: np.ndarray = None, bone_indices: np.ndarray = None):
        if self.renderer is None:
            return
        self.sim_time = sim_time

        self.renderer.begin_frame(sim_time)
        self.renderer.render_mesh(
            name="muscle",
            points=pos_np,
            indices=tri_indices,
            colors=(0.8, 0.3, 0.2),
        )
        if bone_pos_np is not None and bone_indices is not None and len(bone_indices) > 0:
            self.renderer.render_mesh(
                name="bone",
                points=bone_pos_np,
                indices=bone_indices,
                colors=(0.9, 0.9, 0.85),
            )
        self.renderer.end_frame()

    def save(self):
        if self.renderer is not None and self.mode == "usd":
            self.renderer.save()

    def register_key_press_callback(self, callback):
        if self.mode == "human" and self.renderer is not None:
            self.renderer.register_key_press_callback(callback)


# ===========================================================================
# MuscleSim core
# ===========================================================================

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

@wp.kernel
def update_cons_restdir1_kernel(
    c_type: wp.array(dtype=wp.int32),
    c_restdir: wp.array(dtype=wp.vec3),
    val: float,
    ctype_match: int,
    offset: int,
    count: int,
):
    """Update restdir[1] for constraints of a given type."""
    idx = offset + wp.tid()
    if c_type[idx] == ctype_match:
        rd = c_restdir[idx]
        c_restdir[idx] = wp.vec3(rd[0], val, rd[2])


# ---------------------------------------------------------------------------
# Utility @wp.func helpers
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
    exp_damp: float,
    const_damp: float,
):
    i = wp.tid()
    v = (pos[i] - pprev[i]) / dt
    # Gaia velocity damping (updateVelocities:3559-3595): vMag scaled by
    # exp_damp then offset by const_damp.  Defaults (exp=1, const=0) -> no-op.
    if const_damp != 0.0 or exp_damp != 1.0:
        vmag = wp.length(v)
        if vmag > 1.0e-6:
            vnew = vmag * exp_damp - const_damp
            if vnew < 1.0e-6:
                vnew = 0.0
            vel[i] = v * (vnew / vmag)
        else:
            vel[i] = wp.vec3(0.0, 0.0, 0.0)
    else:
        vel[i] = v


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
    c_L: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    c_L[i] = wp.vec3(0.0)


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
    c_type: wp.array(dtype=wp.int32),
    c_pts: wp.array(dtype=wp.vec4i),
    c_restvector: wp.array(dtype=wp.vec4f),
    bone_pos_field: wp.array(dtype=wp.vec3),
    n_cons: int,
):
    c = wp.tid()
    if c >= n_cons:
        return
    ctype = c_type[c]
    if ctype == ConstraintType.ATTACH:
        tgt_idx = c_pts[c][2]
        if tgt_idx >= 0:
            target_pos = bone_pos_field[tgt_idx]
            rv = c_restvector[c]
            c_restvector[c] = wp.vec4f(target_pos[0], target_pos[1], target_pos[2], rv[3])
    elif ctype == ConstraintType.DISTANCELINE:
        tgt_idx = c_pts[c][2]  # pts layout: [src_idx, -1, tgt_idx, -1]
        if tgt_idx >= 0:
            target_pos = bone_pos_field[tgt_idx]
            rv = c_restvector[c]
            c_restvector[c] = wp.vec4f(target_pos[0], target_pos[1], target_pos[2], rv[3])
    elif ctype == ConstraintType.PINTOTARGET:
        tgt_idx = c_pts[c][2]
        if tgt_idx >= 0:
            target_pos = bone_pos_field[tgt_idx]
            rv = c_restvector[c]
            c_restvector[c] = wp.vec4f(target_pos[0], target_pos[1], target_pos[2], rv[3])


# ---------------------------------------------------------------------------
# Per-type constraint solver @wp.kernel
# ---------------------------------------------------------------------------

def build_unique_edges(tets: np.ndarray) -> np.ndarray:
    """All unique tet edges (the MassSpring spring set)."""
    es = set()
    for t in tets:
        for a in range(4):
            for b in range(a + 1, 4):
                i, j = int(t[a]), int(t[b])
                es.add((i, j) if i < j else (j, i))
    return np.asarray(sorted(es), dtype=np.int32)





class XpbdEngine(ConstraintBuilderMixin):
    """Shared XPBD substep engine for the muscle/oneway/massspring sims.

    Owns the state-independent substep primitives: ``integrate``,
    ``clear``, ``apply_dP``, ``solve_constraints`` /
    ``_dispatch_constraints``, ``solve_extras``, ``post_smooth``,
    ``freeze_near_inverted``, ``repair_inverted_tets``,
    ``apply_pintotarget``, ``update_velocities``, and the program runner
    ``run_program``.  Subclasses own their loaders/kernels and define
    ``build_program()`` to set ``self.program`` — an ordered list of
    bound-method phase closures whose default reproduces that sim's
    substep order."""

    def integrate(self):
        gravity_vec = getattr(self, '_gravity_vec', None)
        if gravity_vec is None:
            # Default: Y-axis gravity (legacy behavior)
            gravity_vec = wp.vec3(0.0, self.cfg.gravity, 0.0)
        wp.launch(integrate_kernel, dim=self.n_verts,
                  inputs=[self.pos, self.pprev, self.vel,
                          self.force, self.mass, self.stopped,
                          gravity_vec, self.cfg.veldamping, self.dt])

    def update_velocities(self):
        is_collide = getattr(self, "is_collide", None)
        contact_normal = getattr(self, "contact_normal", None)
        if is_collide is None or contact_normal is None:
            exp_damp = float(getattr(self, "_vel_exp_damp", 1.0))
            const_damp = float(getattr(self, "_vel_const_damp", 0.0))
            wp.launch(update_velocities_kernel, dim=self.n_verts,
                      inputs=[self.pos, self.pprev, self.vel, self.dt,
                              exp_damp, const_damp])
            return
        rn = float(getattr(self, "restitution_n", 0.0))
        rt = float(getattr(self, "restitution_t", 1.0))
        wp.launch(update_velocities_with_contact_kernel, dim=self.n_verts,
                  inputs=[self.pos, self.pprev, self.vel, self.dt,
                          is_collide, contact_normal, rn, rt])

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

    def clear(self):
        wp.launch(clear_dP_kernel, dim=self.n_verts,
                  inputs=[self.dP, self.dPw])
        if self.n_cons > 0:
            wp.launch(clear_cons_L_kernel, dim=self.n_cons,
                      inputs=[self.c_L])

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
        rng = self.cons_ranges.get(ConstraintType.PINTOTARGET) if hasattr(self, "cons_ranges") else None
        if rng is None:
            return
        offset, count = rng
        if count <= 0:
            return
        if not getattr(self, "_pintotarget_stopped_marked", False):
            self._mark_pintotarget_stopped()
            self._pintotarget_stopped_marked = True
        wp.launch(solve_pintotarget_kernel, dim=count,
                  inputs=[self.c_pts, self.c_restvector, self.c_restdir,
                          self.pos, offset])

    def _dispatch_constraints(self, type_ranges):
        """Dispatch constraint solver kernels for a given {type: (offset, count)} mapping."""
        has_compress = 1 if self.has_compressstiffness else 0
        use_jacobi_int = 1 if self.use_jacobi else 0
        for ctype, (offset, count) in type_ranges.items():
            if ctype == ConstraintType.TETVOLUME:
                wp.launch(solve_tetvolume_kernel, dim=count, inputs=[
                    self.c_stiffness, self.c_compressionstiffness, self.c_pts,
                    self.c_tetid, self.c_restlength, self.c_dampingratio, self.c_L,
                    self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.dt, use_jacobi_int, has_compress, offset])
            elif ctype == ConstraintType.TETFIBERNORM:
                wp.launch(solve_tetfibernorm_kernel, dim=count, inputs=[
                    self.c_stiffness, self.c_pts, self.c_tetid,
                    self.c_dampingratio, self.c_restlength, self.c_restvector, self.c_L,
                    self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix, self.v_fiber_dir,
                    self.activation, self.tendonmask,
                    self.dt, use_jacobi_int,
                    self.contraction_ratio, self.fiber_stiffness_scale,
                    offset])
            elif ctype == ConstraintType.ATTACH:
                wp.launch(solve_attach_kernel, dim=count, inputs=[
                    self.c_stiffness, self.c_compressionstiffness, self.c_pts,
                    self.c_restlength, self.c_restvector, self.c_dampingratio, self.c_L,
                    self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.reaction_accum,
                    self.dt, use_jacobi_int, has_compress, offset])
            elif ctype == ConstraintType.PIN:
                wp.launch(solve_pin_kernel, dim=count, inputs=[
                    self.c_stiffness, self.c_compressionstiffness, self.c_pts,
                    self.c_restlength, self.c_restvector, self.c_dampingratio, self.c_L,
                    self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.dt, use_jacobi_int, has_compress, offset])
            elif ctype == ConstraintType.PINTOTARGET:
                # Skipped here — pintotarget is a kinematic post-pass run by
                # apply_pintotarget() AFTER apply_dP, so Jacobi accumulation
                # cannot undo the snap.
                continue
            elif ctype == ConstraintType.DISTANCELINE:
                wp.launch(solve_distanceline_kernel, dim=count, inputs=[
                    self.c_stiffness, self.c_compressionstiffness, self.c_pts,
                    self.c_restlength, self.c_restvector, self.c_restdir,
                    self.c_dampingratio, self.c_L,
                    self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.dt, use_jacobi_int, has_compress, offset])
            elif ctype == ConstraintType.DISTANCE:
                wp.launch(solve_distance_kernel, dim=count, inputs=[
                    self.c_stiffness, self.c_compressionstiffness, self.c_pts,
                    self.c_restlength, self.c_dampingratio, self.c_L,
                    self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.dt, use_jacobi_int, has_compress, offset])
            elif ctype == ConstraintType.TETVOLUMEHARD:
                wp.launch(solve_tetvolumehard_kernel, dim=count, inputs=[
                    self.c_pts, self.c_tetid, self.pos, self.ms_dm_inv,
                    self.mass, self.stopped, offset])
            elif ctype == ConstraintType.TETINVERSION:
                wp.launch(solve_tetinversion_kernel, dim=count, inputs=[
                    self.c_pts, self.c_tetid, self.pos, self.ms_rest_vol,
                    self.mass, self.stopped,
                    wp.float32(self.ms_inversion_mult), offset])
            elif ctype == ConstraintType.TETARAP:
                wp.launch(solve_tetarap_kernel, dim=count, inputs=[
                    self.c_stiffness, self.c_pts, self.c_tetid, self.c_type,
                    self.c_restlength, self.c_dampingratio, self.c_restvector, self.c_L,
                    self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix,
                    self.dt, use_jacobi_int, offset])
            elif ctype == ConstraintType.TETARAPSVD:
                wp.launch(solve_tetarap_svd_kernel, dim=count, inputs=[
                    self.c_stiffness, self.c_pts, self.c_tetid, self.c_L,
                    self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix,
                    self.dt, use_jacobi_int, offset])
            elif ctype == ConstraintType.TETARAPANISO:
                wp.launch(solve_tet_aniso_arap_kernel, dim=count, inputs=[
                    self.c_stiffness, self.c_pts, self.c_tetid, self.c_L,
                    self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix, self.v_fiber_dir,
                    self.activation, self.tendonmask,
                    self.dt, use_jacobi_int,
                    self.contraction_ratio, self.fiber_stiffness_scale,
                    offset])
            elif ctype == ConstraintType.TETFIBERDGF:
                wp.launch(solve_tetfiberdgf_kernel, dim=count, inputs=[
                    self.c_stiffness, self.c_pts, self.c_tetid, self.c_restlength,
                    self.c_restvector, self.c_restdir, self.c_dampingratio, self.c_L,
                    self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix, self.v_fiber_dir,
                    self.activation, self.tendonmask,
                    self.dt, use_jacobi_int,
                    self.fiber_stiffness_scale,
                    offset])
            elif ctype == ConstraintType.TETFIBERMILLARD:
                wp.launch(solve_tetfibermillard_kernel, dim=count, inputs=[
                    self.c_stiffness, self.c_pts, self.c_tetid, self.c_restlength,
                    self.c_restvector, self.c_restdir, self.c_dampingratio, self.c_L,
                    self.pos, self.pprev,
                    self.dP, self.dPw, self.mass, self.stopped,
                    self.rest_matrix, self.v_fiber_dir,
                    self.activation, self.tendonmask,
                    self.dt, use_jacobi_int,
                    self.millard_stiffness_scale,
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
            elif ctype == ConstraintType.TETSNH:
                wp.launch(solve_tetsnh_kernel, dim=count, inputs=[
                    self.c_stiffness, self.c_pts, self.c_tetid, self.c_restdir,
                    self.c_restlength, self.c_dampingratio, self.c_L,
                    self.pos, self.pprev,
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
    _FIBER_TYPES = frozenset([ConstraintType.TETFIBERNORM, ConstraintType.TETFIBERDGF, ConstraintType.TETFIBERMILLARD])

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

    def run_program(self):
        """Execute self.program (list of zero-arg phase closures) once per
        substep for cfg.num_substeps. Subclasses set self.program in
        build_program(); self._program_sub holds the current substep index
        for phases that need it (e.g. debug)."""
        for _sub in range(self.cfg.num_substeps):
            self._program_sub = _sub
            for phase in self.program:
                phase()

    def _build_colored_gs(self, all_constraints):
        """Reorder constraints by (color, type) and build per-color dispatch ranges."""
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

        self._upload_cons(reordered)

    def _upload_cons(self, cons_list):
        """Upload constraint dicts as struct-of-arrays Warp buffers.

        Replaces the old array-of-``Constraint``-structs layout: each kernel
        now takes only the per-field arrays it actually reads/writes.
        """
        n = len(cons_list)
        if n == 0:
            self.c_type = wp.zeros(0, dtype=wp.int32)
            self.c_pts = wp.zeros(0, dtype=wp.vec4i)
            self.c_stiffness = wp.zeros(0, dtype=wp.float32)
            self.c_compressionstiffness = wp.zeros(0, dtype=wp.float32)
            self.c_dampingratio = wp.zeros(0, dtype=wp.float32)
            self.c_tetid = wp.zeros(0, dtype=wp.int32)
            self.c_L = wp.zeros(0, dtype=wp.vec3)
            self.c_restlength = wp.zeros(0, dtype=wp.float32)
            self.c_restvector = wp.zeros(0, dtype=wp.vec4f)
            self.c_restdir = wp.zeros(0, dtype=wp.vec3)
            return

        def _col(key, np_dtype):
            return np.array([c[key] for c in cons_list], dtype=np_dtype)

        self.c_type = wp.array(_col('type', np.int32), dtype=wp.int32)
        self.c_pts = wp.array(_col('pts', np.int32), dtype=wp.vec4i)
        self.c_stiffness = wp.array(_col('stiffness', np.float32), dtype=wp.float32)
        self.c_compressionstiffness = wp.array(_col('compressionstiffness', np.float32), dtype=wp.float32)
        self.c_dampingratio = wp.array(_col('dampingratio', np.float32), dtype=wp.float32)
        self.c_tetid = wp.array(_col('tetid', np.int32), dtype=wp.int32)
        self.c_L = wp.array(_col('L', np.float32), dtype=wp.vec3)
        self.c_restlength = wp.array(_col('restlength', np.float32), dtype=wp.float32)
        self.c_restvector = wp.array(_col('restvector', np.float32), dtype=wp.vec4f)
        self.c_restdir = wp.array(_col('restdir', np.float32), dtype=wp.vec3)


def main():
    import sys
    from muscle.musclesim import MuscleSim
    config_path = sys.argv[1] if len(sys.argv) > 1 else "data/muscle/config/bicep.json"
    print("Using config:", config_path)
    cfg = load_config(config_path)
    cfg.gui = False
    sim = MuscleSim(cfg)
    print("Running for", cfg.nsteps, "steps.")
    sim.run()


if __name__ == "__main__":
    main()
