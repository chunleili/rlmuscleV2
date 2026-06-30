"""MassEnv: MuJoCo port of MASS Environment.cpp.

This is the main simulation environment for muscle-actuated character control.
Ported from external/MASS/core/Environment.cpp.

Usage:
    from rl_mass.env import MassEnv
    env = MassEnv("data/rl_mass/config.json")
    state = env.reset(rsi=False)
    env.set_action(np.zeros(env.num_action))
    for _ in range(env.num_steps):
        env.step()
    reward = env.get_reward()
    state = env.get_state()
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rl_mass.character import Character
from rl_mass.engine import MuJoCoEngine
from rl_mass.joint_utils import (
    DART_ROOT_OFFSET,
    MASS_BVH_MAP,
    MASS_BODY_NAMES,
    MASS_END_EFFECTORS,
    MASS_JOINT_TYPES,
    dart_position_differences,
    dart_qpos_to_mujoco,
    mujoco_qpos_to_dart,
)
from rl_mass.muscle import Muscle, MuscleSystem, parse_muscle_xml


# ---------------------------------------------------------------------------
# MuscleTuple (matches Environment.h struct)
# ---------------------------------------------------------------------------

@dataclass
class MuscleTuple:
    """Data collected at a random simulation sub-step for muscle training."""
    JtA: np.ndarray  # Concatenated related JtA for all muscles
    L: np.ndarray  # Vectorized full JtA matrix (non-root rows)
    b: np.ndarray  # Passive forces (non-root)
    tau_des: np.ndarray  # Desired torques (non-root)
    phase: float = 0.0  # Gait-cycle phase [0,1) at collection (EMG prior lookup)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DART ground body COM y-position, from ground.xml: translation=(0, -0.49958, 0)
_GROUND_COM_Y = -0.49958


# ---------------------------------------------------------------------------
# MassEnv
# ---------------------------------------------------------------------------

class MassEnv:
    """MuJoCo port of the MASS Environment.

    Args:
        config_path: Path to config JSON file.
    """

    def __init__(self, config_path: str = "data/rl_mass/config.json"):
        with open(config_path) as f:
            cfg = json.load(f)

        # Hz settings
        self.control_hz: int = cfg.get("control_hz", 30)
        self.simulation_hz: int = cfg.get("simulation_hz", 600)
        self.use_muscle: bool = cfg.get("use_muscle", True)

        # Reward weights
        rp = cfg.get("reward_param", {})
        self.w_q: float = rp.get("w_q", 0.65)
        self.w_v: float = rp.get("w_v", 0.1)
        self.w_ee: float = rp.get("w_ee", 0.15)
        self.w_com: float = rp.get("w_com", 0.1)

        # Track the root's SO(3) rotation in r_q (and the pose-error
        # termination norm) while still ignoring root translation. Useful for
        # motions whose reference root translates non-trivially (e.g. a kick
        # with a preroll wind-up), where penalizing translation drift makes
        # tracking infeasible but rotation drift is still a meaningful signal.
        self.track_root_rotation: bool = cfg.get("track_root_rotation", False)

        # Track the root's translation in r_q (and the pose-error termination
        # norm). Useful for motions where the reference root translates and
        # the character must follow (e.g. squat: pelvis drops & shifts back —
        # without tracking translation the policy only bends knees in place
        # and falls over). DART FreeJoint stores translation diff as
        # R1^T * (t2 - t1), scaled in meters, so it can dominate r_q if the
        # reference root moves a lot — start with shorter references / lower
        # w_q when first enabling this.
        self.track_root_translation: bool = cfg.get("track_root_translation", False)

        # Terminate episodes when the root drops below a standing height.
        # Disable for tasks with no ground/gravity (e.g. swimming), where the
        # reference pose is not upright and the root height is meaningless.
        self.height_termination: bool = cfg.get("height_termination", True)

        # Terminate when the joint-pose tracking error exceeds this squared
        # norm (||p_diff||^2 over BVH-mapped joints). This is the divergence
        # cutoff for tasks without a height check (e.g. swimming): it plays
        # the same role "falling over" does for locomotion, ending hopeless
        # episodes before they feed garbage muscle tuples to the optimizer.
        # None disables the check.
        pet = cfg.get("pose_error_threshold", None)
        self.pose_error_threshold: Optional[float] = (
            float(pet) if pet is not None else None
        )

        # RSI sampling upper bound as a fraction of clip duration. Default 0.9
        # keeps initialization away from the clip boundary/wrap seam; set to
        # 1.0 to sample uniformly over the whole clip [0, T].
        self.rsi_max_ratio: float = float(cfg.get("rsi_max_ratio", 0.9))

        # SPD parameters
        kp = cfg.get("spd", {}).get("kp", 300.0)

        # Create physics engine
        skel_file = cfg["skeleton_file"]
        self.engine = MuJoCoEngine(skel_file, timestep=1.0 / self.simulation_hz)

        # Convenience aliases for backward compatibility (used by eval.py, train.py)
        self.model = self.engine.model
        self.data = self.engine.data

        # Character (BVH + SPD)
        self.character = Character(self.engine, kp=kp)
        bvh_file = cfg["bvh_file"]
        bvh_cyclic = cfg.get("bvh_cyclic", True)
        # "character1" (default) -> MotionBuilder/HumanIK Character1_* joint
        #   names (18-joint biped) via MASS_BVH_MAP.
        # "native" -> BVH already uses MASS body names (Pelvis, ArmL, ...).
        bvh_naming = cfg.get("bvh_naming", "character1")
        if bvh_naming == "native":
            from rl_mass.joint_utils import MASS_NATIVE_BVH_MAP
            bvh_map = MASS_NATIVE_BVH_MAP
        elif bvh_naming == "character1":
            bvh_map = None
        else:
            raise ValueError(f"bvh_naming must be 'character1' or 'native', got {bvh_naming!r}")
        self.character.load_bvh(bvh_file, cyclic=bvh_cyclic, bvh_map=bvh_map)

        # Root joint DOFs (6 for FreeJoint)
        self.root_joint_dof: int = 6
        self.num_active_dof: int = self.engine.nv - self.root_joint_dof  # 50

        # Action (in DART velocity space, will be added to non-root target positions)
        self.action = np.zeros(self.num_active_dof, dtype=np.float64)

        # Muscles
        self.muscles: list[Muscle] = []
        self.muscle_system: Optional[MuscleSystem] = None
        self.num_total_related_dofs: int = 0
        if self.use_muscle:
            # Parse muscles at rest pose
            self.engine.forward()
            muscle_file = cfg["muscle_file"]
            self.muscles = parse_muscle_xml(muscle_file, self.engine)
            for m in self.muscles:
                m.update(self.engine)
                self.num_total_related_dofs += m.num_related_dofs
            # Build vectorized muscle system
            self.muscle_system = MuscleSystem(self.muscles)

        # Muscle tuple storage
        self.current_muscle_tuple: Optional[MuscleTuple] = None
        self.activation_levels = np.zeros(len(self.muscles), dtype=np.float64)
        self.average_activation_levels = np.zeros(len(self.muscles), dtype=np.float64)
        self.muscle_tuples: list[MuscleTuple] = []
        # Lazily-built gather index for the related-DOF JtA tuple field.
        self._jta_gather_dof: Optional[np.ndarray] = None
        self._jta_gather_mus: Optional[np.ndarray] = None

        # Desired torques (full nv-space)
        self.desired_torque = np.zeros(self.engine.nv, dtype=np.float64)

        # Target positions and velocities (MuJoCo format)
        self.target_positions_mj = np.zeros(self.engine.nq, dtype=np.float64)
        self.target_velocities_mj = np.zeros(self.engine.nv, dtype=np.float64)

        # Simulation step counters
        self.sim_count: int = 0
        self.random_sample_index: int = 0

        # Body ID cache
        self._root_bid = self.engine.body_id("Pelvis")
        self._body_ids = [
            self.engine.body_id(name) for name in MASS_BODY_NAMES
        ]
        self._ee_body_ids = [
            self.engine.body_id(name) for name in MASS_END_EFFECTORS
        ]

        # BVH map: body_name -> joint info, for reward computation
        # Precompute DART DOF index for each BVH-mapped body
        self._bvh_reward_info = self._build_bvh_reward_info()

        # Initialize state
        self.reset(rsi=False)
        self.num_state: int = self.get_state().shape[0]

    def _build_bvh_reward_info(self) -> list[tuple[str, str, int, int]]:
        """Build info for reward computation: (body_name, joint_type, dart_dof_start, dart_dof_count).

        For each BVH-mapped body, record the DART DOF index range of its parent joint.
        Skip root (FreeJoint). For Revolute: 1 DOF. For Ball: 3 DOFs.
        """
        info = []
        # Compute DART DOF offset for each body/joint
        dart_dof_offset = 0
        dart_dof_map = {}  # body_name -> (dart_idx, dart_ndof, joint_type)
        for i, (body_name, jt) in enumerate(zip(MASS_BODY_NAMES, MASS_JOINT_TYPES)):
            ndof = {"Free": 6, "Ball": 3, "Revolute": 1}[jt]
            dart_dof_map[body_name] = (dart_dof_offset, ndof, jt)
            dart_dof_offset += ndof

        for body_name, bvh_name in MASS_BVH_MAP.items():
            dart_idx, dart_ndof, jt = dart_dof_map[body_name]
            if jt == "Free":
                # Include root orientation (exp_map DOFs 0..3) but skip root
                # translation. DART FreeJoint layout: [exp_map(3), trans(3)].
                # Without this, prone-target swim was tracked only in joint
                # angles while the body stayed upright (see env r_q docstring).
                info.append((body_name, jt, dart_idx, 3))
                continue
            info.append((body_name, jt, dart_idx, dart_ndof))
        return info

    @property
    def num_action(self) -> int:
        """Number of actions (non-root DOFs)."""
        return self.num_active_dof

    @property
    def num_steps(self) -> int:
        """Number of simulation steps per control step."""
        return self.simulation_hz // self.control_hz

    @property
    def num_muscles(self) -> int:
        """Number of muscles."""
        return len(self.muscles)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, rsi: bool = True, t: Optional[float] = None) -> np.ndarray:
        """Reset the environment.

        Args:
            rsi: If True, use random start initialization.
            t: If provided, use this time directly (overrides RSI).

        Returns:
            Initial state vector.
        """
        self.engine.reset()

        if t is not None:
            start_time = t
        elif rsi:
            max_time = self.character.bvh.max_time
            start_time = random.uniform(0.0, max_time * self.rsi_max_ratio)
        else:
            start_time = 0.0

        self.engine.set_time(start_time)
        self.character.reset()
        self.action[:] = 0.0

        # Get target positions and velocities
        dt_ctrl = 1.0 / self.control_hz
        mj_q, mj_v = self.character.get_target_pos_and_vel_mujoco(start_time, dt_ctrl)
        self.target_positions_mj = mj_q.copy()
        self.target_velocities_mj = mj_v.copy()

        # Set state
        self.engine.set_qpos(mj_q)
        self.engine.set_qvel(mj_v)
        self.engine.forward()

        # NOTE: Do NOT clear muscle_tuples here.
        # DART's Reset() does not clear mMuscleTuples — they accumulate
        # across episodes until ComputeMuscleTuples() collects them.

        return self.get_state()

    # ------------------------------------------------------------------
    # SetAction
    # ------------------------------------------------------------------

    def set_action(self, a: np.ndarray) -> None:
        """Set action from policy. Port of Environment::SetAction.

        Args:
            a: Action vector of shape (num_active_dof,). Scaled by 0.1.
        """
        a = np.asarray(a, dtype=np.float64).ravel()
        assert a.shape[0] == self.num_active_dof, (
            f"Action size mismatch: got {a.shape[0]}, expected {self.num_active_dof}"
        )
        self.action = a * 0.1

        t = self.data.time
        dt_ctrl = 1.0 / self.control_hz
        mj_q, mj_v = self.character.get_target_pos_and_vel_mujoco(t, dt_ctrl)
        self.target_positions_mj = mj_q.copy()
        self.target_velocities_mj = mj_v.copy()

        self.sim_count = 0
        self.random_sample_index = random.randint(0, self.num_steps - 1)
        self.average_activation_levels[:] = 0.0

    # ------------------------------------------------------------------
    # GetDesiredTorques
    # ------------------------------------------------------------------

    def get_desired_torques(self) -> np.ndarray:
        """Compute desired SPD torques with action offset. Port of Environment::GetDesiredTorques.

        DART: p_des.tail(ndof - rootDof) += mAction  (LINEAR addition in exp-map space)
        We replicate this exactly: convert to DART qpos, add action, convert back.

        Returns:
            Desired torques for non-root DOFs, shape (num_active_dof,).
        """
        from rl_mass.joint_utils import (
            mujoco_qpos_to_dart, dart_qpos_to_mujoco,
            DART_ROOT_OFFSET,
        )

        # Convert target positions to DART exp-map space
        p_des_dart = mujoco_qpos_to_dart(
            self.target_positions_mj, root_offset=DART_ROOT_OFFSET
        )

        # Add action to non-root DOFs (linear addition in exp-map space)
        # DART: p_des.tail(ndof - rootDof) += mAction
        # root DOFs = 6 (FreeJoint), so tail starts at index 6
        p_des_dart[self.root_joint_dof:] += self.action

        # Convert back to MuJoCo qpos
        p_des_mj = dart_qpos_to_mujoco(
            p_des_dart, root_offset=DART_ROOT_OFFSET
        )

        self.desired_torque = self.character.get_spd_forces(
            self.engine, p_des_mj
        )
        return self.desired_torque[self.root_joint_dof:]

    # ------------------------------------------------------------------
    # GetMuscleTorques
    # ------------------------------------------------------------------

    def get_muscle_torques(self) -> np.ndarray:
        """Compute concatenated related JtA for all muscles. Port of Environment::GetMuscleTorques.

        Returns:
            Concatenated JtA vector of shape (num_total_related_dofs,).
        """
        ms = self.muscle_system
        ms.update_all(self.engine)
        return ms.get_all_muscle_torques(self.engine)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Execute one simulation sub-step. Port of Environment::Step.

        For muscle mode: apply muscle forces, collect muscle tuples at random step.
        For non-muscle mode: apply SPD torques directly.
        """
        if self.use_muscle:
            ms = self.muscle_system
            # Clear external forces before applying muscles
            self.engine.clear_applied_forces()

            # Vectorized: update all anchor positions + l_mt + l_m
            ms.update_all(self.engine)

            # Vectorized: compute and apply all muscle forces
            ms.apply_forces_all(self.engine, self.activation_levels)

            # At random step, collect muscle tuple
            if self.sim_count == self.random_sample_index:
                self._collect_muscle_tuple()
        else:
            # Non-muscle mode: apply SPD torques directly
            self.get_desired_torques()
            self.engine.set_qfrc_applied(self.desired_torque)

        self.engine.step()
        self.sim_count += 1

    def _collect_muscle_tuple(self) -> None:
        """Collect muscle data at the random sample step.

        Uses MuscleSystem's cached anchor positions to avoid redundant updates.
        Jacobian computation (mj_jac) is still per-anchor but only called once
        per control step.
        """
        ms = self.muscle_system
        nv = self.engine.nv
        n_muscles = ms.n_muscles

        # Compute full JtA matrix and Jtp using MuscleSystem cached data
        JtA_full = np.zeros((nv, n_muscles), dtype=np.float64)
        Jtp = np.zeros(nv, dtype=np.float64)

        # --- Vectorized per-anchor force directions (all anchors at once) ---
        # Bit-identical to the old per-muscle loop: each anchor accumulates
        # its outgoing segment unit dir (forward pass) then its incoming
        # segment unit dir (backward pass = negated outgoing dir of the
        # previous segment). Segments are within-muscle only, so every anchor
        # appears at most once in seg_a and once in seg_b.
        diff = ms.anchor_world_pos[ms.seg_b] - ms.anchor_world_pos[ms.seg_a]
        seg_norm = np.linalg.norm(diff, axis=1)
        seg_dir = diff.copy()
        nz = seg_norm > 1e-12
        seg_dir[nz] = diff[nz] / seg_norm[nz, None]
        force_dir_all = np.zeros((ms.total_anchors, 3), dtype=np.float64)
        np.add.at(force_dir_all, ms.seg_a, seg_dir)   # forward: outgoing dir
        np.add.at(force_dir_all, ms.seg_b, -seg_dir)  # backward: incoming dir

        for m_i in range(n_muscles):
            start = ms.muscle_anchor_start[m_i]
            n_anc = ms.muscle_n_anchors[m_i]

            # Build Jacobian transpose
            Jt = np.zeros((nv, 3 * n_anc), dtype=np.float64)
            for a_i in range(n_anc):
                flat_idx = start + a_i
                point = ms.anchor_world_pos[flat_idx]
                bid = int(ms.anchor_primary_bid[flat_idx])
                jacp = self.engine.get_linear_jacobian(point, bid)
                Jt[:, a_i * 3 : (a_i + 1) * 3] = jacp.T

            force_dir = force_dir_all[start : start + n_anc]

            l_m_norm = ms.l_m[m_i] / ms.l_m0[m_i]
            f_a = ms.f0[m_i] * np.exp(-((l_m_norm - 1.0) ** 2) / 0.5)

            # Passive force
            if l_m_norm < 1.0:
                f_p = 0.0
            else:
                f_p = ms.f0[m_i] * (
                    np.exp(4.0 * (l_m_norm - 1.0) / 0.6) - 1.0
                ) / (np.exp(4.0) - 1.0)

            A = (force_dir * f_a).ravel()
            p = (force_dir * f_p).ravel()

            JtA_full[:, m_i] = Jt @ A
            Jtp += Jt @ p

        # JtA = concatenated related JtA. Identical to get_all_muscle_torques()
        # (same Jt, same A => same Jt@A): gather the related-DOF rows straight
        # out of JtA_full instead of recomputing every Jacobian.
        if self._jta_gather_dof is None:
            dof_list: list[int] = []
            mus_list: list[int] = []
            for m_i in range(n_muscles):
                for d in ms.related_dof_indices[m_i]:
                    dof_list.append(int(d))
                    mus_list.append(m_i)
            self._jta_gather_dof = np.array(dof_list, dtype=np.intp)
            self._jta_gather_mus = np.array(mus_list, dtype=np.intp)
        JtA = JtA_full[self._jta_gather_dof, self._jta_gather_mus]

        # L = non-root rows of JtA_full, vectorized row-by-row
        L_matrix = JtA_full[self.root_joint_dof:, :]  # (50, 284)
        L_vectorized = L_matrix.ravel()  # row-major is the same as iterating rows

        # b = passive forces (non-root)
        b = Jtp[self.root_joint_dof:]

        # tau_des = desired torques (non-root), use CACHED value from the
        # most recent get_desired_torques() call in the inner loop.
        tau_des = self.desired_torque[self.root_joint_dof:]

        # Gait-cycle phase [0,1) — same definition as get_state's phase, used by
        # the optional EMG activation prior to look up the per-muscle target.
        max_time = self.character.bvh.max_time
        phase = math.fmod(self.engine.get_time(), max_time) / max_time

        self.current_muscle_tuple = MuscleTuple(
            JtA=JtA.copy(),
            L=L_vectorized.copy(),
            b=b.copy(),
            tau_des=tau_des.copy(),
            phase=float(phase),
        )
        self.muscle_tuples.append(self.current_muscle_tuple)

    # ------------------------------------------------------------------
    # GetState
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Compute state vector. Port of Environment::GetState.

        State = [relative_body_positions (66), body_velocities (69), phase (1)]
        Total: 136

        Body positions: COM of body i (i=1..N-1) relative to root body frame.
        Body velocities: COM linear velocities of body 1..N-1, then root.

        Returns:
            State vector of shape (136,).
        """
        num_bodies = len(self._body_ids)  # 23
        root_bid = self._root_bid

        # Root body frame
        R_root = self.engine.get_body_xmat(root_bid)
        t_root = self.engine.get_body_xpos(root_bid)

        # Relative positions: body i COM in root body frame
        # getCOM(root) in DART = R_root^T @ (xipos[i] - xpos[root])
        p = np.zeros((num_bodies - 1) * 3, dtype=np.float64)
        for idx in range(1, num_bodies):
            bid = self._body_ids[idx]
            com_world = self.engine.get_body_com(bid)
            com_rel = R_root.T @ (com_world - t_root)
            p[3 * (idx - 1) : 3 * idx] = com_rel

        # Velocities: COM linear velocity of each body
        # DART getCOMLinearVelocity() = world-frame linear velocity of COM
        v = np.zeros(num_bodies * 3, dtype=np.float64)

        for idx in range(1, num_bodies):
            bid = self._body_ids[idx]
            vel = self.engine.get_body_com_velocity(bid)
            v[3 * (idx - 1) : 3 * idx] = vel

        # Root velocity at the end
        root_vel = self.engine.get_body_com_velocity(root_bid)
        v[-3:] = root_vel

        # Phase
        max_time = self.character.bvh.max_time
        phi = math.fmod(self.engine.get_time(), max_time) / max_time

        # Scale
        p *= 0.8
        v *= 0.2

        # Concatenate
        state = np.concatenate([p, v, [phi]])
        return state

    # ------------------------------------------------------------------
    # GetReward
    # ------------------------------------------------------------------

    def get_reward(self) -> float:
        """Compute tracking reward. Port of Environment::GetReward.

        r = r_ee * (w_q * r_q + w_v * r_v)
        where:
            r_q = exp(-2 * ||p_diff||^2)  position error (BVH-mapped joints only)
            r_v = exp(-0.1 * ||v_diff||^2)  velocity error (all DOFs)
            r_ee = exp(-40 * ||ee_diff||^2)  end-effector error

        Returns:
            Scalar reward in [0, 1].
        """
        # Current state in DART format
        cur_pos_dart = mujoco_qpos_to_dart(
            self.engine.get_qpos(), root_offset=DART_ROOT_OFFSET
        )
        cur_vel_dart = self.engine.get_qvel()

        # Target in DART format
        target_pos_dart = mujoco_qpos_to_dart(
            self.target_positions_mj, root_offset=DART_ROOT_OFFSET
        )
        target_vel_dart = self.target_velocities_mj.copy()

        # Position differences using DART's getPositionDifferences semantics:
        # getPositionDifferences(_q2=target, _q1=current) = logMap(R_cur^T * R_target)
        # This is the SO(3) geodesic difference, NOT simple subtraction.
        p_diff_all = dart_position_differences(target_pos_dart, cur_pos_dart)

        # Filter by BVH-mapped joints only
        p_diff = np.zeros(self.engine.nv, dtype=np.float64)
        v_diff = np.zeros(self.engine.nv, dtype=np.float64)

        for body_name, jt, dart_idx, dart_ndof in self._bvh_reward_info:
            p_diff[dart_idx : dart_idx + dart_ndof] = p_diff_all[dart_idx : dart_idx + dart_ndof]

        # Optionally include the root's SO(3) rotation diff (DART FreeJoint
        # layout: [exp_map(3), translation(3), ...]). Indices 3:6 are root
        # translation, gated by a separate flag.
        if self.track_root_rotation:
            p_diff[0:3] = p_diff_all[0:3]
        if self.track_root_translation:
            p_diff[3:6] = p_diff_all[3:6]

        # v_diff is NOT filtered in the original code:
        # v_diff stays zero, so r_v = exp(0) = 1.0.

        # End effector comparison
        ee_diff = np.zeros(len(self._ee_body_ids) * 3, dtype=np.float64)

        # Current end effector positions (world COM)
        for i, ee_bid in enumerate(self._ee_body_ids):
            ee_diff[i * 3 : (i + 1) * 3] = self.engine.get_body_com(ee_bid)

        # Current whole-body COM
        cur_com = self._get_skeleton_com()

        # Switch to target pose to compute target end effector positions
        saved_qpos = self.engine.get_qpos()
        saved_qvel = self.engine.get_qvel()
        self.engine.set_qpos(self.target_positions_mj)
        self.engine.kinematics()
        self.engine.com_pos()

        target_com = self._get_skeleton_com()
        com_diff = cur_com - target_com

        for i, ee_bid in enumerate(self._ee_body_ids):
            target_ee_pos = self.engine.get_body_com(ee_bid)
            ee_diff[i * 3 : (i + 1) * 3] -= (target_ee_pos + com_diff)

        # Restore current state
        self.engine.set_qpos(saved_qpos)
        self.engine.set_qvel(saved_qvel)
        self.engine.forward()

        # Compute reward components
        r_q = math.exp(-2.0 * np.dot(p_diff, p_diff))
        r_v = math.exp(-0.1 * np.dot(v_diff, v_diff))
        r_ee = math.exp(-40.0 * np.dot(ee_diff, ee_diff))
        r_com = math.exp(-10.0 * np.dot(com_diff, com_diff))

        # r_com is enabled when w_com > 0 (default 0.1, but originally unused —
        # set w_com=0 in a task's config to fall back to the pre-COM reward).
        r = r_ee * (self.w_q * r_q + self.w_v * r_v + self.w_com * r_com)
        return r

    def _get_skeleton_com(self) -> np.ndarray:
        """Compute the center of mass of the skeleton (all character bodies).

        This matches DART's skel->getCOM() which is the mass-weighted average
        of all body COMs.
        """
        total_mass = 0.0
        com = np.zeros(3, dtype=np.float64)
        for bid in self._body_ids:
            mass = self.engine.body_mass(bid)
            com += mass * self.engine.get_body_com(bid)
            total_mass += mass
        return com / total_mass

    # ------------------------------------------------------------------
    # IsEndOfEpisode
    # ------------------------------------------------------------------

    def _pose_error_sq(self) -> float:
        """Squared norm of the joint-pose tracking error (BVH-mapped joints).

        Same p_diff used by get_reward's r_q term: the SO(3) geodesic
        difference between the current pose and the BVH target, restricted
        to BVH-mapped (non-root) joints.
        """
        cur_pos_dart = mujoco_qpos_to_dart(
            self.engine.get_qpos(), root_offset=DART_ROOT_OFFSET
        )
        target_pos_dart = mujoco_qpos_to_dart(
            self.target_positions_mj, root_offset=DART_ROOT_OFFSET
        )
        p_diff_all = dart_position_differences(target_pos_dart, cur_pos_dart)

        err = 0.0
        for _, _, dart_idx, dart_ndof in self._bvh_reward_info:
            seg = p_diff_all[dart_idx : dart_idx + dart_ndof]
            err += float(np.dot(seg, seg))
        if self.track_root_rotation:
            seg = p_diff_all[0:3]
            err += float(np.dot(seg, seg))
        if self.track_root_translation:
            seg = p_diff_all[3:6]
            err += float(np.dot(seg, seg))
        return err

    def is_end_of_episode(self) -> bool:
        """Check termination conditions. Port of Environment::IsEndOfEpisode.

        Terminal if:
        - Root body height < 1.3 (relative to ground COM y), unless
          height_termination is disabled (e.g. swimming task)
        - Joint-pose tracking error ||p_diff||^2 > pose_error_threshold,
          when that threshold is configured (divergence cutoff for tasks
          with no height check, e.g. swimming)
        - NaN in positions or velocities
        - Simulation time > 10.0

        Returns:
            True if episode should terminate.
        """
        # Root body height relative to ground COM
        if self.height_termination:
            root_y = self.engine.get_body_xpos(self._root_bid)[1] - _GROUND_COM_Y
            if root_y < 1.3:
                return True

        # Joint-pose divergence cutoff
        if self.pose_error_threshold is not None:
            if self._pose_error_sq() > self.pose_error_threshold:
                return True

        qpos = self.engine.get_qpos()
        qvel = self.engine.get_qvel()
        if np.any(np.isnan(qpos)) or np.any(np.isnan(qvel)):
            return True

        if self.engine.get_time() > 10.0:
            return True

        return False

    # ------------------------------------------------------------------
    # Convenience: steps_at_once
    # ------------------------------------------------------------------

    def steps_at_once(self) -> None:
        """Execute all simulation sub-steps for one control step."""
        for _ in range(self.num_steps):
            self.step()

    # ------------------------------------------------------------------
    # Properties for external access
    # ------------------------------------------------------------------

    def get_activation_levels(self) -> np.ndarray:
        return self.activation_levels.copy()

    def set_activation_levels(self, a: np.ndarray) -> None:
        self.activation_levels = np.asarray(a, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Diagnostics (eval-only, used by rl_mass/eval.py)
    # ------------------------------------------------------------------

    def compute_diagnostics(self) -> dict:
        """Compute per-control-step muscle diagnostics at the current pose.

        Returns dict with non-root-DOF (50-D) tensors:
            R: (num_active_dof, num_muscles) pure moment-arm matrix
               R[j, m] = sum_a (J_a)^T_j . force_dir_{a,m}
            L: (num_active_dof, num_muscles) active-force-weighted matrix
               L[j, m] = R[j, m] * f_a(m)
            b: (num_active_dof,) passive contribution = sum_m R[:, m] * f_p(m)
            tau_des: (num_active_dof,) cached SPD desired torques
            tau_muscle: (num_active_dof,) applied muscle torque = L @ a + b
            a: (num_muscles,) current activations (last set by policy)
            l_m_norm: (num_muscles,) normalized fiber length l_m / l_m0
            f_a: (num_muscles,) active force scalar f0 * g_al(l_m_norm)
            f_p: (num_muscles,) passive force scalar f0 * g_pl(l_m_norm)
        """
        ms = self.muscle_system
        assert ms is not None, "compute_diagnostics requires use_muscle=True"
        nv = self.engine.nv
        n_muscles = ms.n_muscles

        ms.update_all(self.engine)

        R_full = np.zeros((nv, n_muscles), dtype=np.float64)
        L_full = np.zeros((nv, n_muscles), dtype=np.float64)
        Jtp_full = np.zeros(nv, dtype=np.float64)
        f_a_arr = np.zeros(n_muscles, dtype=np.float64)
        f_p_arr = np.zeros(n_muscles, dtype=np.float64)
        l_m_norm_arr = np.zeros(n_muscles, dtype=np.float64)

        for m_i in range(n_muscles):
            start = ms.muscle_anchor_start[m_i]
            n_anc = ms.muscle_n_anchors[m_i]
            positions = ms.anchor_world_pos[start : start + n_anc]

            force_dir = np.zeros((n_anc, 3), dtype=np.float64)
            for k in range(n_anc - 1):
                d = positions[k + 1] - positions[k]
                norm = np.linalg.norm(d)
                if norm > 1e-12:
                    d = d / norm
                force_dir[k] = force_dir[k] + d
            for k in range(1, n_anc):
                d = positions[k - 1] - positions[k]
                norm = np.linalg.norm(d)
                if norm > 1e-12:
                    d = d / norm
                force_dir[k] = force_dir[k] + d

            col_R = np.zeros(nv, dtype=np.float64)
            for a_i in range(n_anc):
                point = ms.anchor_world_pos[start + a_i]
                bid = int(ms.anchor_primary_bid[start + a_i])
                jacp = self.engine.get_linear_jacobian(point, bid)
                col_R += jacp.T @ force_dir[a_i]
            R_full[:, m_i] = col_R

            l_m_norm = ms.l_m[m_i] / ms.l_m0[m_i]
            f_a = ms.f0[m_i] * np.exp(-((l_m_norm - 1.0) ** 2) / 0.5)
            if l_m_norm < 1.0:
                f_p = 0.0
            else:
                f_p = ms.f0[m_i] * (
                    np.exp(4.0 * (l_m_norm - 1.0) / 0.6) - 1.0
                ) / (np.exp(4.0) - 1.0)

            L_full[:, m_i] = col_R * f_a
            Jtp_full += col_R * f_p
            f_a_arr[m_i] = f_a
            f_p_arr[m_i] = f_p
            l_m_norm_arr[m_i] = l_m_norm

        R = R_full[self.root_joint_dof :, :]
        L = L_full[self.root_joint_dof :, :]
        b = Jtp_full[self.root_joint_dof :]
        a = self.activation_levels.copy()
        tau_des = self.desired_torque[self.root_joint_dof :].copy()
        tau_muscle = L @ a + b
        return {
            "R": R,
            "L": L,
            "b": b,
            "a": a,
            "tau_des": tau_des,
            "tau_muscle": tau_muscle,
            "l_m_norm": l_m_norm_arr,
            "f_a": f_a_arr,
            "f_p": f_p_arr,
        }

    def get_muscle_metadata(self) -> dict:
        """Return static muscle metadata for plotting (names, region, f0).

        Region is derived from the set of body names touched by the muscle's
        anchors, mapped to one of {lower, trunk, l_arm, r_arm, head}.
        """
        from rl_mass.joint_utils import MASS_BODY_NAMES

        LEG = {"Pelvis", "FemurR", "TibiaR", "TalusR", "FootThumbR", "FootPinkyR",
               "FemurL", "TibiaL", "TalusL", "FootThumbL", "FootPinkyL"}
        TRUNK = {"Spine", "Torso"}
        HEAD = {"Neck", "Head"}
        L_ARM = {"ShoulderL", "ArmL", "ForeArmL", "HandL"}
        R_ARM = {"ShoulderR", "ArmR", "ForeArmR", "HandR"}
        REGION_MAP = [("lower", LEG), ("trunk", TRUNK),
                      ("l_arm", L_ARM), ("r_arm", R_ARM), ("head", HEAD)]

        bid_to_name = {}
        for name in MASS_BODY_NAMES:
            bid_to_name[self.engine.body_id(name)] = name

        names = []
        regions = []
        f0_arr = []
        for m in self.muscles:
            names.append(m.name)
            f0_arr.append(m.f0)
            touched = set()
            for anchor in m.anchors:
                for bid in anchor.body_ids:
                    nm = bid_to_name.get(int(bid))
                    if nm is not None:
                        touched.add(nm)
            best_region = "lower"
            best_overlap = -1
            for rname, rset in REGION_MAP:
                ov = len(touched & rset)
                if ov > best_overlap:
                    best_region = rname
                    best_overlap = ov
            regions.append(best_region)
        return {
            "names": names,
            "regions": regions,
            "f0": np.array(f0_arr, dtype=np.float64),
        }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

def _test_env() -> None:
    """Quick smoke test for MassEnv."""
    print("=" * 60)
    print("MassEnv Quick Test")
    print("=" * 60)

    env = MassEnv("data/rl_mass/config.json")
    print(f"  num_state:  {env.num_state}")
    print(f"  num_action: {env.num_action}")
    print(f"  num_steps:  {env.num_steps}")
    print(f"  num_muscles: {env.num_muscles}")
    print(f"  use_muscle: {env.use_muscle}")
    print(f"  control_hz: {env.control_hz}")
    print(f"  simulation_hz: {env.simulation_hz}")
    print(f"  num_total_related_dofs: {env.num_total_related_dofs}")

    # Test reset
    state0 = env.reset(rsi=False)
    print(f"\n  State at t=0 (first 5): {state0[:5]}")
    print(f"  State shape: {state0.shape}")
    print(f"  Phase (last): {state0[-1]:.6f}")

    # Test reward at initial state
    r0 = env.get_reward()
    print(f"  Reward at t=0: {r0:.6f}")

    # Test is_end_of_episode
    eoe0 = env.is_end_of_episode()
    print(f"  End of episode at t=0: {eoe0}")

    # Test step with zero action
    env.set_action(np.zeros(env.num_action))
    env.steps_at_once()
    state1 = env.get_state()
    r1 = env.get_reward()
    print(f"\n  After 1 control step (zero action):")
    print(f"  State (first 5): {state1[:5]}")
    print(f"  Reward: {r1:.6f}")
    print(f"  Time: {env.data.time:.6f}")
    print(f"  End of episode: {env.is_end_of_episode()}")

    # Test muscle torques
    if env.use_muscle:
        mt = env.get_muscle_torques()
        print(f"\n  Muscle torques shape: {mt.shape}")
        print(f"  Muscle torques (first 10): {mt[:10]}")

        dt = env.get_desired_torques()
        print(f"  Desired torques shape: {dt.shape}")
        print(f"  Desired torques (first 10): {dt[:10]}")

    print("\n" + "=" * 60)
    print("MassEnv Quick Test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _test_env()
