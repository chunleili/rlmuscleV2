"""Character class ported from MASS (external/MASS/core/Character.cpp).

Provides:
  - BVH-based target position/velocity generation with cyclic blending
  - SPD (Stable PD) controller for computing desired torques

All computations match DART semantics. The internal representation uses
DART exp-map format for transforms; MuJoCo conversion happens at the
boundary.

Usage:
    from rl_mass.engine import MuJoCoEngine
    from rl_mass.character import Character

    engine = MuJoCoEngine("data/rl_mass/humanoid.xml", timestep=1/600)
    char = Character(engine)
    char.load_bvh("external/MASS/data/motion/walk.bvh", cyclic=True)
    char.reset()
    target_q = char.get_target_positions_mujoco(t=0.0, dt=1/30)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from rl_mass.bvh import (
    BVH,
    _expmap_to_rotmat,
    _logmap,
    _make_transform,
    _transform_inv,
)
from rl_mass.engine import PhysicsEngine
from rl_mass.joint_utils import (
    DART_ROOT_OFFSET,
    MASS_BODY_NAMES,
    MASS_JOINT_TYPES,
    dart_qpos_to_mujoco,
    expmap_to_quat_wxyz,
    mujoco_qpos_to_dart,
    quat_wxyz_to_expmap,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DART DOFs and MuJoCo qpos sizes per joint type
_DART_DOF = {"Free": 6, "Ball": 3, "Revolute": 1}
_MJ_QPOS = {"Free": 7, "Ball": 4, "Revolute": 1}
_MJ_DOF = {"Free": 6, "Ball": 3, "Revolute": 1}  # MuJoCo nv per joint

DART_QPOS_SIZE = sum(_DART_DOF[jt] for jt in MASS_JOINT_TYPES)  # 56


# ---------------------------------------------------------------------------
# Character class
# ---------------------------------------------------------------------------

class Character:
    """Character with BVH-driven target generation and SPD controller.

    This class operates in DART qpos space internally for exact matching
    with the reference MASS implementation. Conversion to/from MuJoCo
    format is done at the API boundary.

    Args:
        engine: Physics engine instance.
        kp: SPD proportional gain (default 300.0).
        kv: SPD derivative gain (default sqrt(2*kp)).
    """

    def __init__(
        self,
        engine: PhysicsEngine,
        kp: float = 300.0,
        kv: Optional[float] = None,
    ):
        self.engine = engine
        self.bvh: Optional[BVH] = None

        # Tc: cumulative root transform for cyclic motion (4x4 homogeneous)
        self.Tc = np.eye(4, dtype=np.float64)

        # SPD parameters (DART dof-space, nv = 56)
        nv = engine.nv
        if kv is None:
            kv = math.sqrt(2.0 * kp)
        self.Kp = np.full(nv, kp, dtype=np.float64)
        self.Kv = np.full(nv, kv, dtype=np.float64)
        # Zero gains for root DOFs (first 6 for FreeJoint)
        # (applied in GetSPDForces by zeroing tau[0:6])

    def load_bvh(
        self,
        filepath: str,
        cyclic: bool = True,
        bvh_map: Optional[dict[str, str]] = None,
    ) -> None:
        """Load and parse a BVH file.

        bvh_map: body_name -> BVH joint name. None uses the default Mixamo-style
        MASS_BVH_MAP. Pass joint_utils.MASS_NATIVE_BVH_MAP for BVH files that
        already use MASS body names (no Character1_* layer).
        """
        self.bvh = BVH(bvh_map=bvh_map)
        self.bvh.parse(filepath, cyclic)

    def reset(self) -> None:
        """Reset Tc to T0 with y-translation zeroed. Port of Character::Reset."""
        assert self.bvh is not None, "Load BVH first"
        self.Tc = self.bvh.T0.copy()
        self.Tc[1, 3] = 0.0  # translation[1] = 0 (y = height)

    def get_target_positions_dart(self, t: float, dt: float) -> np.ndarray:
        """Compute target positions in DART qpos space.

        Port of Character::GetTargetPositions. This may mutate self.Tc
        when the cycle wraps.

        Args:
            t: Current time.
            dt: Control time step (1/control_hz).

        Returns:
            DART qpos vector (56,).
        """
        assert self.bvh is not None

        # 1. Get raw BVH motion in DART format
        p = self.bvh.get_motion_dart(t)

        # 2. Extract root transform from p[0:6]
        #    DART FreeJoint: [logMap(3), translation(3)]
        R_current = _expmap_to_rotmat(p[:3])
        t_current = p[3:6].copy()
        T_current = _make_transform(R_current, t_current)

        # 3. Remove T0: T_current = T0^-1 * T_current
        T0_inv = _transform_inv(self.bvh.T0)
        T_current = T0_inv @ T_current

        # 4. Apply Tc: T_head = Tc * T_current
        T_head = self.Tc @ T_current

        # 5. Write back root transform to p
        R_head = T_head[:3, :3]
        t_head = T_head[:3, 3]
        p[:3] = _logmap(R_head)
        p[3:6] = t_head

        # 6. Cyclic blending near cycle boundary
        if self.bvh.cyclic:
            max_time = self.bvh.max_time
            t_mod = math.fmod(t, max_time)
            t_mod_ratio = t_mod / max_time

            r = 0.95
            if t_mod_ratio > r:
                ratio = 1.0 / (r - 1.0) * t_mod_ratio - 1.0 / (r - 1.0)
                T01 = self.bvh.T1 @ _transform_inv(self.bvh.T0)
                delta = T01[1, 3]  # translation[1] = y
                delta *= ratio
                p[5] += delta  # p[5] = DART root trans z

            # 7. Cycle wrap: update Tc when t+dt wraps
            tdt_mod = math.fmod(t + dt, max_time)
            if tdt_mod - dt < 0.0:
                T01 = self.bvh.T1 @ _transform_inv(self.bvh.T0)
                # Extract axis-angle of T01 rotation, zero x and z
                p01 = _logmap(T01[:3, :3])
                p01[0] = 0.0
                p01[2] = 0.0
                T01[:3, :3] = _expmap_to_rotmat(p01)

                self.Tc = T01 @ self.Tc
                self.Tc[1, 3] = 0.0  # translation[1] = 0

        return p

    def get_target_positions_mujoco(self, t: float, dt: float) -> np.ndarray:
        """Get target positions in MuJoCo qpos format."""
        dart_q = self.get_target_positions_dart(t, dt)
        return dart_qpos_to_mujoco(dart_q, root_offset=DART_ROOT_OFFSET)

    def get_target_pos_and_vel_dart(
        self, t: float, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get target position and velocity in DART format.

        Port of Character::GetTargetPosAndVel.

        Returns:
            (p, v) where p is DART qpos and v is DART velocity = (p1-p)/dt.
        """
        p = self.get_target_positions_dart(t, dt)
        Tc_save = self.Tc.copy()
        p1 = self.get_target_positions_dart(t + dt, dt)
        self.Tc = Tc_save
        return p, (p1 - p) / dt

    def get_target_pos_and_vel_mujoco(
        self, t: float, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get target position and velocity in MuJoCo format.

        DART computes velocity as simple (p1 - p) / dt in exp-map space.
        We replicate this exactly and convert to MuJoCo qvel ordering.

        Returns:
            (mj_qpos, mj_qvel) in MuJoCo format.
        """
        # Get DART-space pos and velocity using DART's simple finite difference
        p_dart, v_dart = self.get_target_pos_and_vel_dart(t, dt)

        # Convert position to MuJoCo qpos
        mj_q = dart_qpos_to_mujoco(p_dart, root_offset=DART_ROOT_OFFSET)

        # Convert velocity to MuJoCo qvel ordering
        # v_dart is in DART qvel space: [angular(3), linear(3), ...non-root DOFs...]
        # MuJoCo qvel space: [linear(3), angular(3), ...non-root DOFs...]
        # Only root FreeJoint DOFs need reordering; non-root DOFs are the same.
        mj_vel = v_dart.copy()
        mj_vel[:3] = v_dart[3:6]   # MuJoCo linear ← DART translation
        mj_vel[3:6] = v_dart[:3]   # MuJoCo angular ← DART rotation

        return mj_q, mj_vel

    def get_spd_forces(
        self,
        engine: PhysicsEngine,
        p_desired_mj: np.ndarray,
    ) -> np.ndarray:
        """Compute SPD controller torques in MuJoCo qvel space.

        Port of Character::GetSPDForces. Uses engine dynamics quantities
        but performs qdqdt and positionDifferences in DART's exp-map space
        to exactly match DART's linear arithmetic.

        DART SPD formula:
            M_inv = (M + dt*Kv)^-1
            qdqdt = q + dq*dt           (LINEAR addition in exp-map space!)
            p_diff = -Kp * posDiff(qdqdt, p_desired)  (exp-map logMap)
            v_diff = -Kv * dq
            ddq = M_inv * (-C_G + p_diff + v_diff + C_f)
            tau = p_diff + v_diff - dt*Kv*ddq
            tau[0:6] = 0

        Args:
            engine: Physics engine (must have current state, call forward() first).
            p_desired_mj: Desired position in MuJoCo qpos format (nq,).

        Returns:
            Torques in MuJoCo qvel space (nv,). Root DOFs (0:6) are zeroed.
        """
        from rl_mass.joint_utils import (
            mujoco_qpos_to_dart, dart_position_differences,
            DART_ROOT_OFFSET,
        )

        nv = engine.nv
        dt = engine.timestep

        # Current state
        q_mj = engine.get_qpos()
        dq = engine.get_qvel()

        # Mass matrix (dense nv x nv)
        M = engine.get_mass_matrix()

        # M_inv = (M + dt * Kv)^-1
        M_mod = M + dt * np.diag(self.Kv)
        M_inv = np.linalg.inv(M_mod)

        # --- DART-space computation for qdqdt and p_diff ---
        # Convert MuJoCo qpos to DART qpos (exp-map)
        q_dart = mujoco_qpos_to_dart(q_mj, root_offset=DART_ROOT_OFFSET)
        p_des_dart = mujoco_qpos_to_dart(p_desired_mj, root_offset=DART_ROOT_OFFSET)

        # qdqdt = q + dq*dt  (DART does LINEAR addition in exp-map space)
        # CRITICAL: MuJoCo qvel root DOFs are [linear(3), angular(3)]
        # but DART qpos root DOFs are [angular(3), linear(3)].
        # Must reorder dq root before adding to q_dart.
        dq_dart_order = dq.copy()
        dq_dart_order[:3] = dq[3:6]  # angular velocity -> first 3 (DART rotation)
        dq_dart_order[3:6] = dq[:3]  # linear velocity -> last 3 (DART translation)
        qdqdt_dart = q_dart + dq_dart_order * dt

        # p_diff = -Kp * getPositionDifferences(qdqdt, p_desired)
        # DART: getPositionDifferences(q2, q1) = logMap(R1^T * R2) ~ q2 - q1
        # So: posDiff(qdqdt, p_des) = qdqdt - p_des
        #     p_diff = -Kp * (qdqdt - p_des)
        pos_diff = dart_position_differences(qdqdt_dart, p_des_dart)
        p_diff = -self.Kp * pos_diff
        # Convert p_diff root from DART order [angular, linear] back to
        # MuJoCo qvel order [linear, angular] for compatibility with M, C_G, v_diff
        p_diff_root_ang = p_diff[:3].copy()
        p_diff[:3] = p_diff[3:6]
        p_diff[3:6] = p_diff_root_ang

        # v_diff = -Kv * dq
        v_diff = -self.Kv * dq

        # Coriolis + gravity
        coriolis_gravity = engine.get_coriolis_gravity()

        # Constraint forces
        constraint_forces = engine.get_constraint_forces()

        # ddq = M_inv * (-C_G + p_diff + v_diff + C_f)
        ddq = M_inv @ (-coriolis_gravity + p_diff + v_diff + constraint_forces)

        # tau = p_diff + v_diff - dt * Kv * ddq
        tau = p_diff + v_diff - dt * self.Kv * ddq

        # Zero root forces (first 6 DOFs for FreeJoint)
        tau[:6] = 0.0

        return tau


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

def _test_character() -> None:
    """Test Character class: target positions and SPD."""
    from rl_mass.engine import MuJoCoEngine

    print("=" * 60)
    print("Character Class Test")
    print("=" * 60)

    engine = MuJoCoEngine("data/rl_mass/humanoid.xml", timestep=1.0 / 600)

    char = Character(engine, kp=300.0)
    char.load_bvh("external/MASS/data/motion/walk.bvh", cyclic=True)
    char.reset()

    print(f"  Model nq={engine.nq}, nv={engine.nv}")
    print(f"  BVH max_time={char.bvh.max_time:.4f}")
    print(f"  Kp={char.Kp[6]:.1f}, Kv={char.Kv[6]:.4f}")

    # Test GetTargetPositions (DART)
    dt_ctrl = 1.0 / 30.0
    p_dart = char.get_target_positions_dart(0.0, dt_ctrl)
    print(f"\n  DART target pos at t=0 (first 12): {p_dart[:12]}")

    # Test GetTargetPositions (MuJoCo)
    char.reset()  # reset Tc since previous call may have mutated it
    p_mj = char.get_target_positions_mujoco(0.0, dt_ctrl)
    print(f"  MuJoCo target pos at t=0 (first 12): {p_mj[:12]}")
    print(f"  MuJoCo target pos shape: {p_mj.shape} (expected {engine.nq})")
    assert p_mj.shape[0] == engine.nq

    # Test GetTargetPosAndVel
    char.reset()
    p_mj2, v_mj = char.get_target_pos_and_vel_mujoco(0.0, dt_ctrl)
    print(f"\n  Target velocity (first 12): {v_mj[:12]}")
    print(f"  Velocity shape: {v_mj.shape} (expected {engine.nv})")

    # Set initial state and test SPD
    char.reset()
    p_init, v_init = char.get_target_pos_and_vel_mujoco(0.0, dt_ctrl)
    engine.set_qpos(p_init)
    engine.set_qvel(v_init)
    engine.forward()

    # SPD with zero action (target = current frame)
    tau = char.get_spd_forces(engine, p_init)
    print(f"\n  SPD torques (first 12): {tau[:12]}")
    print(f"  SPD torques shape: {tau.shape}")
    print(f"  Root torques (should be 0): {tau[:6]}")
    assert np.allclose(tau[:6], 0.0), "Root torques should be zero"
    print(f"  Max non-root torque: {np.max(np.abs(tau[6:])):.4f}")

    # Test cyclic motion continuity
    print("\n  Testing cyclic motion continuity:")
    char.reset()
    max_time = char.bvh.max_time
    # Test positions near cycle boundary
    for t_frac in [0.0, 0.5, 0.9, 0.95, 0.99]:
        t = t_frac * max_time
        p = char.get_target_positions_mujoco(t, dt_ctrl)
        engine.set_qpos(p)
        engine.forward()
        pelvis_id = engine.body_id("Pelvis")
        py = engine.get_body_xpos(pelvis_id)[1]
        print(f"    t={t:.3f} ({t_frac:.2f}*max): pelvis_y={py:.4f}")

    # Test cycle wrapping
    print("\n  Testing cycle wrap:")
    char.reset()
    Tc_before = char.Tc.copy()
    _ = char.get_target_positions_dart(max_time - 0.001, dt_ctrl)
    Tc_after = char.Tc.copy()
    tc_changed = not np.allclose(Tc_before, Tc_after)
    print(f"    Tc changed after near-boundary call: {tc_changed}")
    if tc_changed:
        print(f"    Tc translation: {Tc_after[:3, 3]}")

    print("\n" + "=" * 60)
    print("Character Class Test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _test_character()
