"""Batched DART position differences on GPU via NVIDIA Warp.

Mirrors `rl_mass.joint_utils.dart_position_differences` but processes N worlds
in parallel. For each rotational joint (FreeJoint rot + every BallJoint):
    R1 = expMap(q1[joint]); R2 = expMap(q2[joint])
    R_diff = R1.T @ R2
    R_proj = svd3(R_diff)  -> closest proper rotation
    diff   = logMap(R_proj)
FreeJoint translation: R1.T @ (t2 - t1).
RevoluteJoints: q2 - q1 (direct copy).

This module also provides `WarpMujocoToDart`, the GPU sibling of
`rl_mass.joint_utils.mujoco_qpos_batch_to_dart`. It eliminates the
CPU↔GPU round-trip inside the muscle inner loop's SPD controller.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from rl_mass.joint_utils import MASS_JOINT_TYPES, _build_index_tables, _MJ_QPOS


@wp.func
def _rodrigues(e: wp.vec3) -> wp.mat33:
    theta = wp.sqrt(wp.dot(e, e))
    if theta < 1.0e-8:
        return wp.mat33(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )
    k = e / theta
    K = wp.mat33(
        0.0, -k[2], k[1],
        k[2], 0.0, -k[0],
        -k[1], k[0], 0.0,
    )
    I = wp.mat33(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    )
    s = wp.sin(theta)
    c = wp.cos(theta)
    return I + s * K + (1.0 - c) * (K * K)


@wp.func
def _logmap(R: wp.mat33) -> wp.vec3:
    # cosθ = (trace - 1)/2
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    c = 0.5 * (tr - 1.0)
    if c > 1.0:
        c = 1.0
    if c < -1.0:
        c = -1.0
    theta = wp.acos(c)
    if theta < 1.0e-8:
        # First-order: 0.5 * (R - R^T) reads off the skew vector.
        return wp.vec3(
            0.5 * (R[2, 1] - R[1, 2]),
            0.5 * (R[0, 2] - R[2, 0]),
            0.5 * (R[1, 0] - R[0, 1]),
        )
    s = wp.sin(theta)
    factor = theta / (2.0 * s)
    return wp.vec3(
        factor * (R[2, 1] - R[1, 2]),
        factor * (R[0, 2] - R[2, 0]),
        factor * (R[1, 0] - R[0, 1]),
    )


@wp.kernel
def _rot_diff_kernel(
    dart_q1: wp.array2d(dtype=wp.float32),
    dart_q2: wp.array2d(dtype=wp.float32),
    rot_starts: wp.array(dtype=wp.int32),
    out_diff: wp.array2d(dtype=wp.float32),
):
    w, rj = wp.tid()
    s = rot_starts[rj]
    e1 = wp.vec3(dart_q1[w, s], dart_q1[w, s + 1], dart_q1[w, s + 2])
    e2 = wp.vec3(dart_q2[w, s], dart_q2[w, s + 1], dart_q2[w, s + 2])
    R1 = _rodrigues(e1)
    R2 = _rodrigues(e2)
    R_diff = wp.transpose(R1) * R2

    # warp.svd3 needs in/out vars; project R_diff to closest proper rotation.
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sigma = wp.vec3(0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wp.svd3(R_diff, U, sigma, V)
    R_proj = U * wp.transpose(V)
    # Enforce det = +1 (flip last column of V if det(U*V^T) = -1).
    if wp.determinant(R_proj) < 0.0:
        Vf = wp.mat33(
            V[0, 0], V[0, 1], -V[0, 2],
            V[1, 0], V[1, 1], -V[1, 2],
            V[2, 0], V[2, 1], -V[2, 2],
        )
        R_proj = U * wp.transpose(Vf)

    d = _logmap(R_proj)
    out_diff[w, s + 0] = d[0]
    out_diff[w, s + 1] = d[1]
    out_diff[w, s + 2] = d[2]


@wp.kernel
def _free_trans_kernel(
    dart_q1: wp.array2d(dtype=wp.float32),
    dart_q2: wp.array2d(dtype=wp.float32),
    free_di: wp.int32,
    out_diff: wp.array2d(dtype=wp.float32),
):
    w = wp.tid()
    e1 = wp.vec3(dart_q1[w, free_di], dart_q1[w, free_di + 1], dart_q1[w, free_di + 2])
    R1 = _rodrigues(e1)
    dt = wp.vec3(
        dart_q2[w, free_di + 3] - dart_q1[w, free_di + 3],
        dart_q2[w, free_di + 4] - dart_q1[w, free_di + 4],
        dart_q2[w, free_di + 5] - dart_q1[w, free_di + 5],
    )
    r = wp.transpose(R1) * dt
    out_diff[w, free_di + 3] = r[0]
    out_diff[w, free_di + 4] = r[1]
    out_diff[w, free_di + 5] = r[2]


@wp.kernel
def _rev_diff_kernel(
    dart_q1: wp.array2d(dtype=wp.float32),
    dart_q2: wp.array2d(dtype=wp.float32),
    rev_idx: wp.array(dtype=wp.int32),
    out_diff: wp.array2d(dtype=wp.float32),
):
    w, r = wp.tid()
    i = rev_idx[r]
    out_diff[w, i] = dart_q2[w, i] - dart_q1[w, i]


class WarpDartDiff:
    """Batched DART position differences (nworld, n_dart_v) on GPU."""

    def __init__(self, num_worlds: int, joint_types=None, device: str | None = None):
        if joint_types is None:
            joint_types = MASS_JOINT_TYPES
        idx = _build_index_tables(joint_types)

        self.num_worlds = int(num_worlds)
        self.device = device or wp.get_preferred_device().alias

        # Rotational joints: FreeJoint rot (if present) then BallJoints. Each
        # contributes 3 DART slots starting at rot_starts[r].
        rot_starts_list: list[int] = []
        self.free_di = int(idx["free_dart_idx"])
        if self.free_di >= 0:
            rot_starts_list.append(self.free_di)
        rot_starts_list.extend(int(s) for s in idx["ball_dart_starts"])

        with wp.ScopedDevice(self.device):
            self.rot_starts = wp.array(
                np.asarray(rot_starts_list, dtype=np.int32), dtype=wp.int32
            )
            self.rev_idx = wp.array(
                np.asarray(idx["rev_dart_indices"], dtype=np.int32), dtype=wp.int32
            )

        self.n_rot = int(len(rot_starts_list))
        self.n_rev = int(len(idx["rev_dart_indices"]))
        self.dart_q_size = int(
            sum({"Free": 6, "Ball": 3, "Revolute": 1}[jt] for jt in joint_types)
        )

    def compute(self, q1: wp.array, q2: wp.array, out: wp.array) -> None:
        """All inputs are wp.array2d(dtype=wp.float32) shape (nworld, n_dart)."""
        with wp.ScopedDevice(self.device):
            if self.n_rot > 0:
                wp.launch(
                    _rot_diff_kernel,
                    dim=(self.num_worlds, self.n_rot),
                    inputs=[q1, q2, self.rot_starts],
                    outputs=[out],
                )
            if self.free_di >= 0:
                wp.launch(
                    _free_trans_kernel,
                    dim=self.num_worlds,
                    inputs=[q1, q2, self.free_di],
                    outputs=[out],
                )
            if self.n_rev > 0:
                wp.launch(
                    _rev_diff_kernel,
                    dim=(self.num_worlds, self.n_rev),
                    inputs=[q1, q2, self.rev_idx],
                    outputs=[out],
                )


# ---------------------------------------------------------------------------
# MuJoCo qpos -> DART qpos on GPU (eliminates the scipy round-trip in SPD)
# ---------------------------------------------------------------------------

@wp.func
def _quat_wxyz_to_expmap(qw: wp.float32, qx: wp.float32, qy: wp.float32, qz: wp.float32) -> wp.vec3:
    # Pick the w>=0 hemisphere so the rotvec stays in [-pi, pi]
    # and matches scipy.Rotation.from_quat(...).as_rotvec().
    if qw < 0.0:
        qw = -qw
        qx = -qx
        qy = -qy
        qz = -qz
    n = wp.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)
    qw = qw / n
    qx = qx / n
    qy = qy / n
    qz = qz / n
    v_norm = wp.sqrt(qx * qx + qy * qy + qz * qz)
    if v_norm < 1.0e-8:
        return wp.vec3(0.0, 0.0, 0.0)
    theta = 2.0 * wp.atan2(v_norm, qw)
    f = theta / v_norm
    return wp.vec3(qx * f, qy * f, qz * f)


@wp.kernel
def _free_mj_to_dart_kernel(
    mj_q: wp.array2d(dtype=wp.float32),
    free_mi: wp.int32,
    free_di: wp.int32,
    root_offset: wp.vec3,
    dart_q: wp.array2d(dtype=wp.float32),
):
    w = wp.tid()
    # MuJoCo: trans (3) then quat_wxyz (4) starting at free_mi.
    # DART:   expmap_3 then trans_3 starting at free_di.
    t = wp.vec3(
        mj_q[w, free_mi + 0] - root_offset[0],
        mj_q[w, free_mi + 1] - root_offset[1],
        mj_q[w, free_mi + 2] - root_offset[2],
    )
    em = _quat_wxyz_to_expmap(
        mj_q[w, free_mi + 3],
        mj_q[w, free_mi + 4],
        mj_q[w, free_mi + 5],
        mj_q[w, free_mi + 6],
    )
    dart_q[w, free_di + 0] = em[0]
    dart_q[w, free_di + 1] = em[1]
    dart_q[w, free_di + 2] = em[2]
    dart_q[w, free_di + 3] = t[0]
    dart_q[w, free_di + 4] = t[1]
    dart_q[w, free_di + 5] = t[2]


@wp.kernel
def _ball_mj_to_dart_kernel(
    mj_q: wp.array2d(dtype=wp.float32),
    ball_mj_starts: wp.array(dtype=wp.int32),
    ball_dart_starts: wp.array(dtype=wp.int32),
    dart_q: wp.array2d(dtype=wp.float32),
):
    w, k = wp.tid()
    mi = ball_mj_starts[k]
    di = ball_dart_starts[k]
    em = _quat_wxyz_to_expmap(
        mj_q[w, mi + 0],
        mj_q[w, mi + 1],
        mj_q[w, mi + 2],
        mj_q[w, mi + 3],
    )
    dart_q[w, di + 0] = em[0]
    dart_q[w, di + 1] = em[1]
    dart_q[w, di + 2] = em[2]


@wp.kernel
def _rev_mj_to_dart_kernel(
    mj_q: wp.array2d(dtype=wp.float32),
    rev_mj_idx: wp.array(dtype=wp.int32),
    rev_dart_idx: wp.array(dtype=wp.int32),
    dart_q: wp.array2d(dtype=wp.float32),
):
    w, k = wp.tid()
    dart_q[w, rev_dart_idx[k]] = mj_q[w, rev_mj_idx[k]]


class WarpMujocoToDart:
    """GPU sibling of `mujoco_qpos_batch_to_dart`. Avoids the per-iter
    scipy Rotation round-trip inside the muscle inner loop's SPD controller.

    Input/output are ``wp.array2d(dtype=wp.float32)``. Output buffer is
    allocated and owned by the caller (passed via ``compute``).
    """

    def __init__(
        self,
        num_worlds: int,
        joint_types: list[str] | None = None,
        root_offset: np.ndarray | None = None,
        device: str | None = None,
    ):
        if joint_types is None:
            joint_types = MASS_JOINT_TYPES
        idx = _build_index_tables(joint_types)

        self.num_worlds = int(num_worlds)
        self.device = device or wp.get_preferred_device().alias

        self.free_mi = int(idx["free_mj_idx"])
        self.free_di = int(idx["free_dart_idx"])
        ro = np.zeros(3, dtype=np.float32) if root_offset is None else (
            np.asarray(root_offset, dtype=np.float32).reshape(3)
        )
        self.root_offset = wp.vec3(float(ro[0]), float(ro[1]), float(ro[2]))

        ball_mj = np.asarray(idx["ball_mj_starts"], dtype=np.int32)
        ball_dart = np.asarray(idx["ball_dart_starts"], dtype=np.int32)
        rev_mj = np.asarray(idx["rev_mj_indices"], dtype=np.int32)
        rev_dart = np.asarray(idx["rev_dart_indices"], dtype=np.int32)

        with wp.ScopedDevice(self.device):
            self.ball_mj_starts = wp.array(ball_mj, dtype=wp.int32)
            self.ball_dart_starts = wp.array(ball_dart, dtype=wp.int32)
            self.rev_mj_idx = wp.array(rev_mj, dtype=wp.int32)
            self.rev_dart_idx = wp.array(rev_dart, dtype=wp.int32)
        self.n_ball = int(len(ball_mj))
        self.n_rev = int(len(rev_mj))

        self.mj_q_size = int(sum(_MJ_QPOS[jt] for jt in joint_types))
        self.dart_q_size = int(
            sum({"Free": 6, "Ball": 3, "Revolute": 1}[jt] for jt in joint_types)
        )

    def compute(self, mj_q: wp.array, dart_q: wp.array) -> None:
        """``mj_q`` is (nworld, mj_q_size) float32; ``dart_q`` is
        (nworld, dart_q_size) float32. Output buffer is overwritten."""
        with wp.ScopedDevice(self.device):
            if self.free_di >= 0:
                wp.launch(
                    _free_mj_to_dart_kernel,
                    dim=self.num_worlds,
                    inputs=[mj_q, self.free_mi, self.free_di, self.root_offset],
                    outputs=[dart_q],
                )
            if self.n_ball > 0:
                wp.launch(
                    _ball_mj_to_dart_kernel,
                    dim=(self.num_worlds, self.n_ball),
                    inputs=[mj_q, self.ball_mj_starts, self.ball_dart_starts],
                    outputs=[dart_q],
                )
            if self.n_rev > 0:
                wp.launch(
                    _rev_mj_to_dart_kernel,
                    dim=(self.num_worlds, self.n_rev),
                    inputs=[mj_q, self.rev_mj_idx, self.rev_dart_idx],
                    outputs=[dart_q],
                )
