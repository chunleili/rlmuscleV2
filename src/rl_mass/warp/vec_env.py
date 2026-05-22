"""Shared-engine vec-env backed by `WarpEngine`."""

from __future__ import annotations

import json
import random
import math
from typing import Optional

import numpy as np
import torch

from rl_mass.env import MassEnv
from rl_mass.joint_utils import (
    DART_ROOT_OFFSET,
    dart_qpos_batch_to_mujoco,
    dart_position_differences,
    mujoco_qpos_to_dart,
)
from rl_mass.muscle import MuscleSystem
from rl_mass.vec_env import MassVecEnv
from rl_mass.warp.dart_diff import WarpDartDiff
from rl_mass.warp.engine import WarpEngine, WarpWorldView
from rl_mass.warp.muscle import WarpMuscleSystem


class WarpVecEnv(MassVecEnv):
    """Batched vec-env that shares one `WarpEngine` across all worlds.

    Phase 1 intentionally keeps the MASS control / reward / muscle logic in
    Python by reusing `MassEnv` per world, but all physics state lives inside
    one tiled `mujoco_warp` data structure and each simulation sub-step is
    executed once for all worlds together.
    """

    backend_name = "warp"

    def __init__(self, config_path: str, num_envs: int, seed: Optional[int] = None):
        del seed
        with open(config_path) as f:
            cfg = json.load(f)

        self.config_path = config_path
        self.num_envs = num_envs
        self._engine = WarpEngine(
            cfg["skeleton_file"],
            timestep=1.0 / cfg.get("simulation_hz", 600),
            num_worlds=num_envs,
        )
        self.envs = [
            MassEnv(
                config_path,
                engine=WarpWorldView(self._engine, i),
                vectorized_muscles=False,
            )
            for i in range(num_envs)
        ]

        e0 = self.envs[0]
        self.num_state = e0.num_state
        self.num_action = e0.num_action
        self.num_muscles = e0.num_muscles
        self.use_muscle = e0.use_muscle
        self.simulation_hz = e0.simulation_hz
        self.control_hz = e0.control_hz
        self.num_total_related_dofs = e0.num_total_related_dofs
        self.num_steps = e0.num_steps
        self._dt_ctrl = 1.0 / self.control_hz
        self._dart_diff = WarpDartDiff(
            num_worlds=self.num_envs,
            device=self._engine.device,
        )
        self._root_joint_dof = e0.root_joint_dof
        self._root_bid = e0._root_bid
        self._body_ids = np.asarray(e0._body_ids, dtype=np.intp)
        self._non_root_body_ids = self._body_ids[1:]
        self._ee_body_ids = np.asarray(e0._ee_body_ids, dtype=np.intp)
        self._body_masses = np.asarray(
            [self._engine.body_mass(int(bid)) for bid in self._body_ids],
            dtype=np.float64,
        )
        self._body_mass_sum = float(np.sum(self._body_masses))
        self._reward_dof_idx = np.concatenate(
            [
                np.arange(dart_idx, dart_idx + dart_ndof, dtype=np.intp)
                for _, _, dart_idx, dart_ndof in e0._bvh_reward_info
            ]
        )
        self._kp_t = torch.as_tensor(
            np.stack([env.character.Kp for env in self.envs], axis=0),
            device=self._engine.device,
            dtype=torch.float32,
        )
        self._kv_t = torch.as_tensor(
            np.stack([env.character.Kv for env in self.envs], axis=0),
            device=self._engine.device,
            dtype=torch.float32,
        )
        self._target_positions_dart = np.zeros(
            (self.num_envs, self._engine.nv),
            dtype=np.float64,
        )
        self._target_velocities_dart = np.zeros_like(self._target_positions_dart)
        self._target_positions_mj = np.zeros(
            (self.num_envs, self._engine.nq),
            dtype=np.float64,
        )
        self._target_velocities_mj = np.zeros(
            (self.num_envs, self._engine.nv),
            dtype=np.float64,
        )
        self._reward_cache = np.zeros((self.num_envs,), dtype=np.float64)
        self._pose_error_sq_cache = np.zeros((self.num_envs,), dtype=np.float64)
        self._eoe_cache = np.zeros((self.num_envs,), dtype=bool)
        self._metrics_dirty = True

        # Batched muscle system shared across all worlds (topology is identical
        # since model is shared). Built from one MassEnv's muscle list — this
        # also gives us a CPU MuscleSystem for any per-world fallback callers.
        if self.use_muscle:
            self._cpu_ms_template = MuscleSystem(e0.muscles)
            self._warp_ms = WarpMuscleSystem(self._cpu_ms_template, self._engine)
            self._activations_buf = np.zeros(
                (self.num_envs, self.num_muscles), dtype=np.float32
            )
        else:
            self._cpu_ms_template = None
            self._warp_ms = None
            self._activations_buf = None

    def get_states(self) -> np.ndarray:
        root_R = self._engine.data.xmat.numpy()[:, self._root_bid, :, :]
        root_t = self._engine.data.xpos.numpy()[:, self._root_bid, :]
        body_com = self._engine.data.xipos.numpy()[:, self._non_root_body_ids, :]
        rel = body_com - root_t[:, None, :]
        rel_root = np.einsum("wji,wbj->wbi", root_R, rel, optimize=True)

        body_vel = self._engine.data.cvel.numpy()[:, self._non_root_body_ids, 3:]
        root_vel = self._engine.data.cvel.numpy()[:, self._root_bid, 3:]
        max_time = self.envs[0].character.bvh.max_time
        phi = np.fmod(self._engine.get_time(), max_time) / max_time

        p = rel_root.reshape(self.num_envs, -1) * 0.8
        v = np.concatenate(
            [
                body_vel.reshape(self.num_envs, -1),
                root_vel,
            ],
            axis=1,
        ) * 0.2
        return np.concatenate([p, v, phi[:, None]], axis=1).astype(np.float64, copy=False)

    def set_actions(self, acts: np.ndarray) -> None:
        acts = np.asarray(acts, dtype=np.float64)
        times = self._engine.get_time().astype(np.float64, copy=False)
        target_dart, target_vel_dart = self.envs[0].character.get_target_pos_and_vel_dart_batch(
            times,
            self._dt_ctrl,
        )
        self._target_positions_dart[:] = target_dart
        self._target_velocities_dart[:] = target_vel_dart
        target_q = dart_qpos_batch_to_mujoco(
            target_dart,
            root_offset=DART_ROOT_OFFSET,
        )
        target_v = target_vel_dart.copy()
        target_v[:, :3] = target_vel_dart[:, 3:6]
        target_v[:, 3:6] = target_vel_dart[:, :3]
        self._target_positions_mj[:] = target_q
        self._target_velocities_mj[:] = target_v
        for i, (env, action) in enumerate(zip(self.envs, acts)):
            env.action = np.asarray(action, dtype=np.float64).ravel() * 0.1
            env.target_positions_mj = target_q[i].copy()
            env.target_velocities_mj = target_v[i].copy()
            env.sim_count = 0
            env.random_sample_index = random.randint(0, self.num_steps - 1)
            env.average_activation_levels[:] = 0.0
        self._metrics_dirty = True

    def get_muscle_torques(self) -> np.ndarray:
        if self._warp_ms is None:
            return np.array(
                [env.get_muscle_torques() for env in self.envs], dtype=np.float32
            )
        self._warp_ms.update_all()
        return self._warp_ms.get_all_muscle_torques().numpy().astype(np.float32)

    def get_desired_torques(self) -> np.ndarray:
        self._engine.forward_dynamics()
        p_des_dart = np.empty((self.num_envs, self._engine.nv), dtype=np.float32)
        qdqdt_dart = np.empty_like(p_des_dart)
        q_mj = self._engine.get_qpos()
        dq = self._engine.get_qvel()
        dt = self._engine.timestep

        for i, env in enumerate(self.envs):
            p_des_i = self._target_positions_dart[i].copy()
            p_des_i[env.root_joint_dof :] += env.action
            p_des_dart[i] = p_des_i.astype(np.float32, copy=False)

            q_dart_i = mujoco_qpos_to_dart(
                q_mj[i],
                root_offset=DART_ROOT_OFFSET,
            )
            dq_i = dq[i]
            dq_dart_i = dq_i.copy()
            dq_dart_i[:3] = dq_i[3:6]
            dq_dart_i[3:6] = dq_i[:3]
            qdqdt_dart[i] = (q_dart_i + dq_dart_i * dt).astype(np.float32, copy=False)

        import warp as wp

        with wp.ScopedDevice(self._engine.device):
            q1_wp = wp.array(p_des_dart, dtype=wp.float32)
            q2_wp = wp.array(qdqdt_dart, dtype=wp.float32)
            out_wp = wp.zeros_like(q1_wp)
            self._dart_diff.compute(q1_wp, q2_wp, out_wp)
            pos_diff_batch = wp.to_torch(out_wp).to(dtype=torch.float32)

        dq_t = torch.as_tensor(dq, device=pos_diff_batch.device, dtype=torch.float32)
        kp_t = self._kp_t
        kv_t = self._kv_t
        p_diff = -kp_t * pos_diff_batch
        p_diff_root_ang = p_diff[:, :3].clone()
        p_diff[:, :3] = p_diff[:, 3:6]
        p_diff[:, 3:6] = p_diff_root_ang

        v_diff = -kv_t * dq_t
        M = self._engine.mass_matrix_tensor()
        diag_idx = self._engine._torch_diag_idx
        M[:, diag_idx, diag_idx] += dt * kv_t
        rhs = (
            -self._engine.coriolis_gravity_tensor()
            + p_diff
            + v_diff
            + self._engine.constraint_forces_tensor()
        )
        ddq = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)
        tau = p_diff + v_diff - dt * kv_t * ddq

        out = np.empty((self.num_envs, self.num_action), dtype=np.float32)
        tau_cpu = tau.detach().cpu().numpy().astype(np.float64, copy=False)
        for i, env in enumerate(self.envs):
            tau_i = tau_cpu[i]
            tau_i[: env.root_joint_dof] = 0.0
            env.desired_torque = tau_i.copy()
            out[i] = tau_i[env.root_joint_dof :].astype(np.float32, copy=False)

        return out

    def set_activation_levels(self, acts: np.ndarray) -> None:
        if self._warp_ms is None:
            for env, activation in zip(self.envs, acts):
                env.set_activation_levels(activation)
            return
        a = np.asarray(acts, dtype=np.float32)
        self._activations_buf[:] = a
        self._warp_ms.set_activations(self._activations_buf)
        # Mirror into per-world MassEnv so non-batched fallback paths
        # (compute_diagnostics, muscle tuple sampler) see the same numbers.
        for env, activation in zip(self.envs, acts):
            env.set_activation_levels(activation)

    def _substep_batched(self) -> None:
        """One simulation substep, batched across all worlds via warp kernels.

        Muscle path: clear xfrc, batched update + apply muscle forces, then
        one batched physics step. Non-muscle path: per-world SPD qfrc (CPU
        round-trip) then batched physics step.
        """
        if self._warp_ms is not None:
            self._engine.clear_applied_forces()
            self._warp_ms.update_all()
            self._warp_ms.apply_forces_all()
            import os
            if os.environ.get("WARP_NAN_TRAP"):
                xfrc = self._engine.data.xfrc_applied.numpy()
                xmax = np.max(np.linalg.norm(xfrc[:,:,:3], axis=2))
                vmax = float(np.max(np.abs(self._engine.get_qvel())))
                lmn_arr = self._warp_ms.l_mt.numpy()
                lmn_max = float(np.max(lmn_arr))
                trip = (lmn_max > 1.05) or (xmax > 1.0e3) or (vmax > 50.0) or not np.all(np.isfinite(xfrc))
                if trip:
                    if lmn_max > 1.05:
                        bad_w_idx = int(np.unravel_index(np.argmax(lmn_arr), lmn_arr.shape)[0])
                        bad_m = int(np.unravel_index(np.argmax(lmn_arr), lmn_arr.shape)[1])
                    else:
                        bad_w_idx = int(np.argmax(np.max(np.linalg.norm(xfrc[:,:,:3], axis=2), axis=1)))
                        bad_m = -1
                    np.savez("/tmp/warp_div_xfrc.npz",
                             xfrc=xfrc,
                             q=self._engine.get_qpos(),
                             v=self._engine.get_qvel(),
                             acts=self._activations_buf,
                             anchor_pos=self._warp_ms.anchor_world_pos.numpy(),
                             l_mt=self._warp_ms.l_mt.numpy(),
                             l_m=self._warp_ms.l_m.numpy(),
                             muscle_force=self._warp_ms.muscle_force.numpy() if hasattr(self._warp_ms,'muscle_force') else np.zeros(1),
                             bad_world=bad_w_idx)
                    raise FloatingPointError(f"trip: max|F|={xmax:.3e} max|v|={vmax:.3e} max_lmn={lmn_max:.3f} world={bad_w_idx} m={bad_m} (dumped /tmp/warp_div_xfrc.npz)")
            for env in self.envs:
                if env.sim_count == env.random_sample_index:
                    env._collect_muscle_tuple()
                env.sim_count += 1
            self._engine.step()
            if os.environ.get("WARP_NAN_TRAP"):
                q = self._engine.get_qpos()
                v = self._engine.get_qvel()
                if not (np.all(np.isfinite(q)) and np.all(np.isfinite(v))):
                    bad_w = np.where(~np.all(np.isfinite(q), axis=1))[0]
                    np.savez("/tmp/warp_div_step.npz", q=q, v=v,
                             xfrc=self._engine.data.xfrc_applied.numpy(),
                             acts=self._activations_buf)
                    raise FloatingPointError(f"qpos/qvel non-finite after warp step bad_worlds={bad_w.tolist()[:8]}")
        else:
            for env in self.envs:
                env.get_desired_torques()
                env.engine.set_qfrc_applied(env.desired_torque)
                env.sim_count += 1
            self._engine.step()

    def steps(self, n: int) -> None:
        for _ in range(n):
            self._substep_batched()
        self._metrics_dirty = True

    def steps_at_once(self) -> None:
        for _ in range(self.num_steps):
            self._substep_batched()
        self._metrics_dirty = True

    def is_end_of_episode(self, idx: int) -> bool:
        self._refresh_metrics_if_needed()
        return bool(self._eoe_cache[idx])

    def get_reward(self, idx: int) -> float:
        self._refresh_metrics_if_needed()
        return float(self._reward_cache[idx])

    def reset(self, rsi: bool = True, idx: Optional[int] = None):
        if idx is None:
            self._engine.reset()
            max_time = self.envs[0].character.bvh.max_time
            if rsi:
                start_time = np.array(
                    [random.uniform(0.0, max_time * 0.9) for _ in range(self.num_envs)],
                    dtype=np.float32,
                )
            else:
                start_time = np.zeros((self.num_envs,), dtype=np.float32)
            self._engine.set_time(start_time)
            target_dart, target_vel_dart = self.envs[0].character.get_target_pos_and_vel_dart_batch(
                start_time,
                self._dt_ctrl,
            )
            self._target_positions_dart[:] = target_dart
            self._target_velocities_dart[:] = target_vel_dart
            target_q = dart_qpos_batch_to_mujoco(
                target_dart,
                root_offset=DART_ROOT_OFFSET,
            )
            target_v = target_vel_dart.copy()
            target_v[:, :3] = target_vel_dart[:, 3:6]
            target_v[:, 3:6] = target_vel_dart[:, :3]
            self._target_positions_mj[:] = target_q
            self._target_velocities_mj[:] = target_v
            self._engine.set_qpos(target_q)
            self._engine.set_qvel(target_v)
            self._engine.forward()
            for i, env in enumerate(self.envs):
                env.character.reset()
                env.action[:] = 0.0
                env.target_positions_mj = target_q[i].copy()
                env.target_velocities_mj = target_v[i].copy()
                env.sim_count = 0
                env.random_sample_index = 0
                env.average_activation_levels[:] = 0.0
                if env.activation_levels.size:
                    env.activation_levels[:] = 0.0
            self._metrics_dirty = True
            return None
        state = self.envs[idx].reset(rsi=rsi)
        self._target_positions_mj[idx] = self.envs[idx].target_positions_mj
        self._target_velocities_mj[idx] = self.envs[idx].target_velocities_mj
        self._target_positions_dart[idx] = mujoco_qpos_to_dart(
            self.envs[idx].target_positions_mj,
            root_offset=DART_ROOT_OFFSET,
        )
        target_vel_dart = self.envs[idx].target_velocities_mj.copy()
        target_vel_dart[:3] = self.envs[idx].target_velocities_mj[3:6]
        target_vel_dart[3:6] = self.envs[idx].target_velocities_mj[:3]
        self._target_velocities_dart[idx] = target_vel_dart
        self._metrics_dirty = True
        return state

    def gather_muscle_tuples(self) -> list:
        out = []
        for env in self.envs:
            out.extend(env.muscle_tuples)
            env.muscle_tuples.clear()
        return out

    def update_muscle_nn(self, state_dict: dict) -> None:
        del state_dict

    def run_control_step(self, actions: np.ndarray) -> list[tuple[np.ndarray, float, bool]]:
        raise NotImplementedError(
            "WarpVecEnv packed control path is not implemented yet; "
            "use the batch API with `--vec_env_mode warp --no_packed`."
        )

    def close(self) -> None:
        self._engine.close()

    def _refresh_metrics_if_needed(self) -> None:
        if not self._metrics_dirty:
            return

        qpos = self._engine.get_qpos().astype(np.float64, copy=False)
        qvel = self._engine.get_qvel().astype(np.float64, copy=False)
        times = self._engine.get_time().astype(np.float64, copy=False)
        cur_pos_dart = np.empty((self.num_envs, self._engine.nv), dtype=np.float64)
        for i in range(self.num_envs):
            cur_pos_dart[i] = mujoco_qpos_to_dart(
                qpos[i],
                root_offset=DART_ROOT_OFFSET,
            )

        p_diff_all = np.empty_like(cur_pos_dart)
        for i in range(self.num_envs):
            p_diff_all[i] = dart_position_differences(
                self._target_positions_dart[i],
                cur_pos_dart[i],
            )

        reward_seg = p_diff_all[:, self._reward_dof_idx]
        pose_error_sq = np.sum(reward_seg * reward_seg, axis=1)
        r_q = np.exp(-2.0 * pose_error_sq)
        r_v = np.ones((self.num_envs,), dtype=np.float64)

        cur_com = np.tensordot(
            self._engine.data.xipos.numpy()[:, self._body_ids, :],
            self._body_masses,
            axes=([1], [0]),
        ) / self._body_mass_sum
        cur_ee = self._engine.data.xipos.numpy()[:, self._ee_body_ids, :].copy()

        saved_qpos = qpos.copy()
        saved_qvel = qvel.copy()
        self._engine.set_qpos(self._target_positions_mj)
        self._engine.kinematics()
        self._engine.com_pos()
        target_xipos = self._engine.data.xipos.numpy()
        target_com = np.tensordot(
            target_xipos[:, self._body_ids, :],
            self._body_masses,
            axes=([1], [0]),
        ) / self._body_mass_sum
        target_ee = target_xipos[:, self._ee_body_ids, :].copy()
        self._engine.set_qpos(saved_qpos)
        self._engine.set_qvel(saved_qvel)
        self._engine.forward()

        com_diff = cur_com - target_com
        ee_diff = cur_ee - (target_ee + com_diff[:, None, :])
        ee_err_sq = np.sum(ee_diff * ee_diff, axis=(1, 2))
        r_ee = np.exp(-40.0 * ee_err_sq)

        reward = r_ee * (self.envs[0].w_q * r_q + self.envs[0].w_v * r_v)

        root_y = self._engine.data.xpos.numpy()[:, self._root_bid, 1] - (-0.49958)
        has_nan = np.any(np.isnan(qpos), axis=1) | np.any(np.isnan(qvel), axis=1)
        eoe = has_nan | (times > 10.0)
        for i, env in enumerate(self.envs):
            if env.height_termination and root_y[i] < 1.3:
                eoe[i] = True
            if (
                env.pose_error_threshold is not None
                and pose_error_sq[i] > env.pose_error_threshold
            ):
                eoe[i] = True

        self._reward_cache[:] = reward
        self._pose_error_sq_cache[:] = pose_error_sq
        self._eoe_cache[:] = eoe
        self._metrics_dirty = False
