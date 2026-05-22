"""Batched DART position differences on GPU via NVIDIA Warp.

Mirrors `rl_mass.joint_utils.dart_position_differences` but processes N worlds
in parallel. For each rotational joint (FreeJoint rot + every BallJoint):
    R1 = expMap(q1[joint]); R2 = expMap(q2[joint])
    R_diff = R1.T @ R2
    R_proj = svd3(R_diff)  -> closest proper rotation
    diff   = logMap(R_proj)
FreeJoint translation: R1.T @ (t2 - t1).
RevoluteJoints: q2 - q1 (direct copy).
"""

from __future__ import annotations

import numpy as np
import warp as wp

from rl_mass.joint_utils import MASS_JOINT_TYPES, _build_index_tables


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
