"""BimanualEnv: native-MuJoCo MASS env for the MyoBimanualArm case.

Fixed-base (no root), tendon-backed 1D muscles, reference motion from a
musclemimic .npz. Duck-types the MassEnv interface consumed by vec_env
and PPO. MuJoCo's own actuators are disabled so only rl_mass Hill forces
(via TendonMuscleSystem) drive the model.
"""
from __future__ import annotations

import json
import math
import random
from typing import Optional

import numpy as np

from rl_mass.engine import MuJoCoEngine
from rl_mass.env import MuscleTuple
from rl_mass.native_spd import native_spd_forces
from rl_mass.tendon_muscle import TendonMuscleSystem
from rl_mass.trajectory_ref import TrajectoryRef

_QVEL_STATE_SCALE = 0.1


class BimanualEnv:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            cfg = json.load(f)

        self.control_hz: int = cfg.get("control_hz", 30)
        self.simulation_hz: int = cfg.get("simulation_hz", 600)
        self.use_muscle: bool = cfg.get("use_muscle", True)
        rp = cfg.get("reward_param", {})
        self.w_q: float = rp.get("w_q", 0.85)
        self.w_v: float = rp.get("w_v", 0.15)
        self.pose_error_threshold: Optional[float] = (
            float(cfg["pose_error_threshold"])
            if cfg.get("pose_error_threshold") is not None else None
        )
        self.kp: float = cfg.get("spd", {}).get("kp", 100.0)
        self.kv: float = math.sqrt(2.0 * self.kp)
        self.muscle_force_scale: float = cfg.get("muscle_force_scale", 1.0)

        self.engine = MuJoCoEngine(cfg["skeleton_file"], timestep=1.0 / self.simulation_hz)
        # Disable MuJoCo's own actuator forces; we apply Hill forces ourselves.
        import mujoco
        self.engine.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
        self.model = self.engine.model
        self.data = self.engine.data

        self.nv = self.engine.nv
        # Constraint-dependent DOFs (scapulohumeral rhythm): 22 shoulder-girdle
        # joints are slaved to shoulder_elv via mjEQ_JOINT equality constraints.
        # Muscles cannot drive them — MuJoCo's constraint forces do — so they are
        # excluded from the muscle-tuple target and the action space, exactly as
        # rl_mass excludes the unactuated root DOFs. On the remaining independent
        # DOFs the muscle moment matrix is full rank (validated: LS residual 0).
        dep_dofs = set()
        for i in range(self.model.neq):
            if int(self.model.eq_type[i]) == int(mujoco.mjtEq.mjEQ_JOINT):
                jid = int(self.model.eq_obj1id[i])
                dep_dofs.add(int(self.model.jnt_dofadr[jid]))
        self.active_dof = np.array(
            [d for d in range(self.nv) if d not in dep_dofs], dtype=np.intp
        )
        self.num_action_: int = int(self.active_dof.size)  # independent DOFs
        self.action = np.zeros(self.num_action_, dtype=np.float64)

        self.ref = TrajectoryRef(cfg["trajectory_file"], nq=self.engine.nq)

        self.engine.forward()
        self.muscle_system: Optional[TendonMuscleSystem] = None
        self.num_total_related_dofs: int = 0
        if self.use_muscle:
            self.muscle_system = TendonMuscleSystem(
                self.engine, force_scale=self.muscle_force_scale
            )
            self.num_total_related_dofs = self.muscle_system.num_total_related_dofs

        self.n_muscles = int(self.model.nu)
        self.activation_levels = np.zeros(self.n_muscles, dtype=np.float64)
        self.muscle_tuples: list[MuscleTuple] = []
        self.current_muscle_tuple: Optional[MuscleTuple] = None

        self.desired_torque = np.zeros(self.nv, dtype=np.float64)
        self.target_q = np.zeros(self.engine.nq, dtype=np.float64)
        self.target_v = np.zeros(self.nv, dtype=np.float64)

        self.sim_count = 0
        self.random_sample_index = 0

        self.reset(rsi=False)
        self.num_state: int = self.get_state().shape[0]

    # --- interface attrs ---
    @property
    def num_action(self) -> int:
        return self.num_action_

    @property
    def num_muscles(self) -> int:
        return self.n_muscles

    @property
    def num_steps(self) -> int:
        return self.simulation_hz // self.control_hz

    # --- reset ---
    def reset(self, rsi: bool = True, t: Optional[float] = None) -> np.ndarray:
        self.engine.reset()
        self.engine.model.opt.disableflags |= self._actuation_disable_bit()
        if t is not None:
            start = t
        elif rsi:
            start = random.uniform(0.0, self.ref.max_time * 0.9)
        else:
            start = 0.0
        self.engine.set_time(start)
        self.action[:] = 0.0
        dt_ctrl = 1.0 / self.control_hz
        q, v = self.ref.pos_vel(start, dt_ctrl)
        self.target_q = q.copy()
        self.target_v = v.copy()
        self.engine.set_qpos(q)
        self.engine.set_qvel(v)
        self.engine.forward()
        return self.get_state()

    def _actuation_disable_bit(self) -> int:
        import mujoco
        return int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)

    # --- action ---
    def set_action(self, a: np.ndarray) -> None:
        a = np.asarray(a, dtype=np.float64).ravel()
        assert a.shape[0] == self.num_action_, (a.shape, self.num_action_)
        self.action = a * 0.1
        t = self.engine.get_time()
        dt_ctrl = 1.0 / self.control_hz
        q, v = self.ref.pos_vel(t, dt_ctrl)
        self.target_q = q.copy()
        self.target_v = v.copy()
        self.sim_count = 0
        self.random_sample_index = random.randint(0, self.num_steps - 1)

    def get_desired_torques(self) -> np.ndarray:
        q_des = self.target_q.copy()
        q_des[self.active_dof] += self.action  # offset independent-DOF targets
        tau = native_spd_forces(self.engine, q_des, self.kp, self.kv)
        self.desired_torque = tau  # full nv (used by the non-muscle qfrc path)
        return tau[self.active_dof]  # SPD target on independent DOFs only

    def get_muscle_torques(self) -> np.ndarray:
        return self.muscle_system.get_muscle_torques(self.engine)

    def set_activation_levels(self, a: np.ndarray) -> None:
        self.activation_levels = np.asarray(a, dtype=np.float64).ravel()

    def get_activation_levels(self) -> np.ndarray:
        return self.activation_levels.copy()

    # --- step ---
    def step(self) -> None:
        if self.use_muscle:
            self.muscle_system.update(self.engine)
            self.muscle_system.apply_forces(self.engine, self.activation_levels)
            if self.sim_count == self.random_sample_index:
                self._collect_muscle_tuple()
        else:
            self.get_desired_torques()
            self.engine.set_qfrc_applied(self.desired_torque)
        self.engine.step()
        self.sim_count += 1

    def steps_at_once(self) -> None:
        for _ in range(self.num_steps):
            self.step()

    def _collect_muscle_tuple(self) -> None:
        # tau_des on the independent DOFs (also caches the full self.desired_torque).
        tau_des = self.get_desired_torques()  # (num_action,)
        L, b, JtA = self.muscle_system.tuple_terms(self.engine, self.activation_levels)
        # L and b are full (nv, ...); slice to independent DOFs so the MuscleNN
        # target (tau_des) and the moment matrix L agree on the 54-DOF space.
        # JtA (per-muscle related-DOF moments) stays full — it is an input feature,
        # not a target, mirroring how rl_mass keeps root rows in JtA but not in L.
        self.current_muscle_tuple = MuscleTuple(
            JtA=JtA.copy(),
            L=L[self.active_dof].ravel().copy(),
            b=b[self.active_dof].copy(),
            tau_des=tau_des.copy(),
        )
        self.muscle_tuples.append(self.current_muscle_tuple)

    # --- state / reward / termination ---
    def get_state(self) -> np.ndarray:
        q = self.engine.get_qpos()
        v = self.engine.get_qvel() * _QVEL_STATE_SCALE
        phase = math.fmod(self.engine.get_time(), self.ref.max_time) / self.ref.max_time
        return np.concatenate([q, v, [phase]])

    def get_reward(self) -> float:
        q = self.engine.get_qpos()
        v = self.engine.get_qvel()
        q_diff = q - self.target_q
        v_diff = v - self.target_v
        r_q = math.exp(-2.0 * float(np.dot(q_diff, q_diff)))
        r_v = math.exp(-0.1 * float(np.dot(v_diff, v_diff)))
        return self.w_q * r_q + self.w_v * r_v

    def _pose_error_sq(self) -> float:
        q_diff = self.engine.get_qpos() - self.target_q
        return float(np.dot(q_diff, q_diff))

    def is_end_of_episode(self) -> bool:
        if self.pose_error_threshold is not None:
            if self._pose_error_sq() > self.pose_error_threshold:
                return True
        if np.any(np.isnan(self.engine.get_qpos())) or np.any(np.isnan(self.engine.get_qvel())):
            return True
        if self.engine.get_time() > 10.0:
            return True
        return False
