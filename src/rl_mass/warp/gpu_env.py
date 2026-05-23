"""Batched GPU env backed by `WarpEngine`.

GPU path is intentionally NOT modeled as "a vector of N MassEnvs". All state
lives as `(N, ...)` tensors / arrays and every operation is one batched op
over all worlds. Per-index loops (especially per-env reset) are the throughput
killer on GPU — see commit `ee7df6f` for the cProfile evidence.

This module deliberately does NOT inherit from `MassVecEnv` — that base class
exists to abstract over CPU vec backends and its slow per-index fallbacks
(``get_end_of_episodes`` / ``get_rewards``) are exactly the pattern we are
trying to leave behind.
"""

from __future__ import annotations

import json
import random
import math
from typing import Optional

import numpy as np
import torch

from rl_mass.env import MassEnv, batch_init_mass_envs
from rl_mass.joint_utils import (
    DART_ROOT_OFFSET,
    dart_qpos_batch_to_mujoco,
    dart_position_differences_batched,
    mujoco_qpos_to_dart,
    mujoco_qpos_batch_to_dart,
)
from rl_mass.muscle import MuscleSystem
from rl_mass.warp.dart_diff import WarpDartDiff, WarpMujocoToDart
from rl_mass.warp.engine import WarpEngine, WarpWorldView
from rl_mass.warp.muscle import WarpMuscleSystem


class WarpGPUEnv:
    """Single batched env over N worlds, backed by one `WarpEngine`."""

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
        self.envs = batch_init_mass_envs(
            config_path,
            engine_for=lambda i: WarpWorldView(self._engine, i),
            num_envs=num_envs,
            vectorized_muscles=False,
        )

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
        self._mj_to_dart = WarpMujocoToDart(
            num_worlds=self.num_envs,
            root_offset=DART_ROOT_OFFSET,
            device=self._engine.device,
        )
        # Pre-allocated device buffers for the GPU SPD pipeline. Avoids
        # the per-iter wp.array() H->D allocation+upload that dominated
        # env_step pre-refactor.
        import warp as _wp
        _dv = self._engine.nv
        with _wp.ScopedDevice(self._engine.device):
            self._q_dart_buf = _wp.zeros((self.num_envs, _dv), dtype=_wp.float32)
            self._p_des_buf = _wp.zeros((self.num_envs, _dv), dtype=_wp.float32)
            self._qdqdt_buf = _wp.zeros((self.num_envs, _dv), dtype=_wp.float32)
            self._pos_diff_buf = _wp.zeros((self.num_envs, _dv), dtype=_wp.float32)
            self._target_dart_buf = _wp.zeros((self.num_envs, _dv), dtype=_wp.float32)
        # Torch views over the wp buffers (same memory). Use these for
        # the SPD math; writes through these update the underlying wp.array
        # so dart_diff sees them.
        self._q_dart_t = _wp.to_torch(self._q_dart_buf)
        self._p_des_t = _wp.to_torch(self._p_des_buf)
        self._qdqdt_t = _wp.to_torch(self._qdqdt_buf)
        self._pos_diff_t = _wp.to_torch(self._pos_diff_buf)
        self._target_dart_t = _wp.to_torch(self._target_dart_buf)
        # Cached SPD intermediates.
        self._actions_t: Optional[torch.Tensor] = None
        self._M_buf: Optional[torch.Tensor] = None
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

        # Lazy CPU sync of SPD tau + activations — populated by tensor-API paths.
        self._latest_tau_t: Optional[torch.Tensor] = None
        self._tau_synced = True
        self._latest_acts_t: Optional[torch.Tensor] = None
        self._acts_synced = True

        # Cached long tensors for batched gather in get_states_tensor.
        self._non_root_body_ids_t = torch.as_tensor(
            self._non_root_body_ids, device=self._engine.device, dtype=torch.long
        )
        self._max_time = float(self.envs[0].character.bvh.max_time)

        # Batched muscle system shared across all worlds (topology is identical
        # since model is shared). Built from one MassEnv's muscle list — this
        # also gives us a CPU MuscleSystem for any per-world fallback callers.
        if self.use_muscle:
            self._cpu_ms_template = MuscleSystem(e0.muscles)
            self._warp_ms = WarpMuscleSystem(self._cpu_ms_template, self._engine)
            self._activations_buf = np.zeros(
                (self.num_envs, self.num_muscles), dtype=np.float32
            )
            self._muscle_tuple_chunks: list[dict[str, torch.Tensor]] = []
            # CUDA graph for the inner muscle substep (clear+update+apply+step).
            # Built lazily on first non-trap substep; replayed via capture_launch.
            self._substep_graph = None
            self._substep_warmup_done = False
        else:
            self._cpu_ms_template = None
            self._warp_ms = None
            self._activations_buf = None
            self._muscle_tuple_chunks = []

    def get_states(self) -> np.ndarray:
        return self.get_states_tensor().detach().cpu().numpy().astype(
            np.float64, copy=False
        )

    def get_states_tensor(self) -> torch.Tensor:
        """Device-resident obs assembly. Returns (num_envs, num_state) float32.

        Replaces 4 separate `.numpy()` syncs (xmat/xpos/xipos/cvel×2) with a
        pure-GPU gather + einsum. Caller is free to keep the tensor on device
        for policy forward; the numpy wrapper falls back to a single sync.
        """
        import warp as wp

        xmat = wp.to_torch(self._engine.data.xmat)
        xpos = wp.to_torch(self._engine.data.xpos)
        xipos = wp.to_torch(self._engine.data.xipos)
        cvel = wp.to_torch(self._engine.data.cvel)
        time_t = wp.to_torch(self._engine.data.time)

        root_R = xmat[:, self._root_bid, :, :]
        root_t = xpos[:, self._root_bid, :]
        body_com = xipos.index_select(1, self._non_root_body_ids_t)
        rel = body_com - root_t.unsqueeze(1)
        # einsum("wji,wbj->wbi", R, rel) = (R^T @ rel^T)^T per world
        rel_root = torch.einsum("wji,wbj->wbi", root_R, rel)

        body_vel = cvel.index_select(1, self._non_root_body_ids_t)[..., 3:]
        root_vel = cvel[:, self._root_bid, 3:]
        phi = torch.fmod(time_t, self._max_time) / self._max_time

        p = rel_root.reshape(self.num_envs, -1) * 0.8
        v = torch.cat(
            [body_vel.reshape(self.num_envs, -1), root_vel],
            dim=1,
        ) * 0.2
        return torch.cat([p, v, phi.unsqueeze(1)], dim=1).to(dtype=torch.float32)

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

        # Batched precompute to drop per-env Python work:
        # - scale all actions in one numpy op
        # - draw all random_sample_indices in one call
        # - skip target_q[i].copy() — target_q is fresh per call and never
        #   mutated in-place by env code paths (always reassigned), so handing
        #   each env a view is safe.
        scaled = (acts * 0.1).astype(np.float64, copy=False)
        rsi = np.random.randint(0, self.num_steps, size=self.num_envs)
        for i, env in enumerate(self.envs):
            env.action = scaled[i].ravel()
            env.target_positions_mj = target_q[i]
            env.target_velocities_mj = target_v[i]
            env.sim_count = 0
            env.random_sample_index = int(rsi[i])
            env.average_activation_levels[:] = 0.0
        # Upload target_dart + actions to device once per control step. These
        # are consumed by _compute_tau_tensor across the 15 muscle iters;
        # pre-uploading lets the SPD loop stay on the GPU.
        import warp as _wp
        self._target_dart_buf.assign(target_dart.astype(np.float32, copy=False))
        self._actions_t = torch.as_tensor(
            scaled, device=self._engine.device, dtype=torch.float32,
        )
        self._metrics_dirty = True

    def get_muscle_torques(self) -> np.ndarray:
        if self._warp_ms is None:
            return np.array(
                [env.get_muscle_torques() for env in self.envs], dtype=np.float32
            )
        self._warp_ms.update_all()
        return self._warp_ms.get_all_muscle_torques().numpy().astype(np.float32)

    def get_muscle_torques_tensor(self) -> torch.Tensor:
        """Device-resident JtA — avoids host round-trip in the muscle loop."""
        if self._warp_ms is None:
            arr = self.get_muscle_torques()
            return torch.as_tensor(arr, device=self._engine.device, dtype=torch.float32)
        import warp as wp

        self._warp_ms.update_all()
        jta_wp = self._warp_ms.get_all_muscle_torques()
        return wp.to_torch(jta_wp).to(dtype=torch.float32)

    def _compute_tau_tensor(self) -> torch.Tensor:
        """Run batched SPD entirely on GPU. Returns full-DOF tau torch tensor.

        Pre-refactor this was the dominant rollout cost (53% of env_step at
        N=16384) because every iter did 4 GPU<->CPU round-trips:
        get_qpos/get_qvel sync, scipy mujoco_qpos_batch_to_dart, then
        wp.array(p_des)+wp.array(qdqdt) re-uploads. Now all data stays on
        device; the only host op is the engine.forward_dynamics() call.

        Side effect: leaves SPD intermediate state (kinematics+forward_dynamics)
        consistent for the upcoming substep step.
        """
        self._engine.forward_dynamics()
        dt = self._engine.timestep
        rjd = int(self._root_joint_dof)

        # MuJoCo qpos -> DART qpos via warp kernel; writes into self._q_dart_buf.
        self._mj_to_dart.compute(self._engine.data.qpos, self._q_dart_buf)
        # Device-resident views of mujoco qvel.
        import warp as wp
        qvel_t = wp.to_torch(self._engine.data.qvel).to(dtype=torch.float32)

        # dq_dart: mujoco qvel with the [0:3] (root rot) and [3:6] (root trans)
        # 3-DOF blocks swapped. Build on device with a torch concat.
        if qvel_t.shape[1] > 6:
            dq_dart_t = torch.cat(
                [qvel_t[:, 3:6], qvel_t[:, 0:3], qvel_t[:, 6:]], dim=1,
            )
        else:
            dq_dart_t = torch.cat([qvel_t[:, 3:6], qvel_t[:, 0:3]], dim=1)

        # p_des = target_dart + actions (after root DOFs). Write into the
        # p_des wp.array via its torch view so dart_diff sees the result.
        self._p_des_t.copy_(self._target_dart_t)
        self._p_des_t[:, rjd:].add_(self._actions_t)

        # qdqdt = q_dart + dq_dart * dt. Same buffer-aliasing trick.
        self._qdqdt_t.copy_(self._q_dart_t)
        self._qdqdt_t.add_(dq_dart_t, alpha=dt)

        # Batched DART position differences on device.
        self._dart_diff.compute(
            self._p_des_buf, self._qdqdt_buf, self._pos_diff_buf,
        )
        pos_diff_batch = self._pos_diff_t

        kp_t = self._kp_t
        kv_t = self._kv_t
        p_diff = -kp_t * pos_diff_batch
        # Swap root rot/trans 3-blocks (kp is in dart layout, mujoco needs
        # the swap before the linear solve).
        p_diff_root_ang = p_diff[:, :3].clone()
        p_diff[:, :3] = p_diff[:, 3:6]
        p_diff[:, 3:6] = p_diff_root_ang

        v_diff = -kv_t * qvel_t
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
        if rjd > 0:
            tau[:, :rjd] = 0.0
        self._latest_tau_t = tau
        self._tau_synced = False
        return tau

    def _sync_desired_torque_to_envs(self) -> None:
        """Copy latest tau tensor down to each MassEnv.desired_torque (CPU)."""
        if self._tau_synced or self._latest_tau_t is None:
            return
        tau_cpu = (
            self._latest_tau_t.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        for i, env in enumerate(self.envs):
            env.desired_torque = tau_cpu[i].copy()
        self._tau_synced = True

    def get_desired_torques(self) -> np.ndarray:
        tau = self._compute_tau_tensor()
        self._sync_desired_torque_to_envs()
        rjd = int(self._root_joint_dof)
        tau_cpu = tau.detach().cpu().numpy()
        return tau_cpu[:, rjd:].astype(np.float32, copy=False)

    def get_desired_torques_tensor(self) -> torch.Tensor:
        """Device-resident desired torques sized (nworld, num_action).

        Defers CPU sync of `env.desired_torque` until something needs it
        (e.g. muscle-tuple collection in `_substep_batched`).
        """
        tau = self._compute_tau_tensor()
        rjd = int(self._root_joint_dof)
        return tau[:, rjd:].contiguous()

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
        self._latest_acts_t = None
        self._acts_synced = True

    def set_activation_levels_tensor(self, acts) -> None:
        """Device-to-device activation upload from torch tensor.

        Skips per-env CPU mirror until `_sync_activation_levels_to_envs` is
        triggered (only needed when a muscle-tuple sample is about to be
        collected).
        """
        if self._warp_ms is None:
            arr = acts.detach().cpu().numpy().astype(np.float32, copy=False)
            for env, activation in zip(self.envs, arr):
                env.set_activation_levels(activation)
            return
        self._warp_ms.set_activations_tensor(acts)
        self._latest_acts_t = acts.detach()
        self._acts_synced = False

    def _sync_activation_levels_to_envs(self) -> None:
        if self._acts_synced or self._latest_acts_t is None:
            return
        a = self._latest_acts_t.cpu().numpy().astype(np.float32, copy=False)
        self._activations_buf[:] = a
        for env, activation in zip(self.envs, a):
            env.set_activation_levels(activation)
        self._acts_synced = True

    def _collect_muscle_tuples_batched(self) -> None:
        if self._warp_ms is None:
            return
        sample_idx = [
            i for i, env in enumerate(self.envs)
            if env.sim_count == env.random_sample_index
        ]
        if not sample_idx:
            return

        import warp as wp

        related_jta_wp, jta_per_muscle_wp, passive_torque_wp = (
            self._warp_ms.get_training_terms()
        )
        device = self._engine.device
        idx_t = torch.as_tensor(sample_idx, device=device, dtype=torch.long)
        rjd = int(self._root_joint_dof)

        related_jta = wp.to_torch(related_jta_wp).index_select(0, idx_t)
        jta_per_muscle = wp.to_torch(jta_per_muscle_wp).index_select(0, idx_t)
        passive_torque = wp.to_torch(passive_torque_wp).index_select(0, idx_t)
        tau = self._latest_tau_t
        if tau is None:
            tau = self._compute_tau_tensor()
        tau_des = tau.index_select(0, idx_t)[:, rjd:]

        # CPU trainer expects L flattened row-major from (num_action, n_muscles).
        # Keep tensors on device; OptimizeMuscleNN reshapes them directly.
        l = jta_per_muscle[:, :, rjd:].transpose(1, 2).contiguous()
        self._muscle_tuple_chunks.append(
            {
                "JtA": related_jta.detach(),
                "TauDes": tau_des.detach(),
                "L": l.reshape(l.shape[0], -1).detach(),
                "b": passive_torque[:, rjd:].detach(),
            }
        )

    def _run_substep_physics(self) -> None:
        """Pure-GPU substep kernels: safe to record into a CUDA graph.

        The D->H NaN-guard lives in `_engine.recover_nan_if_any` and must be
        called outside the captured graph.
        """
        self._engine.clear_applied_forces()
        self._warp_ms.update_all()
        self._warp_ms.apply_forces_all()
        self._engine.step_physics_only()

    def _substep_batched(self) -> None:
        """One simulation substep, batched across all worlds via warp kernels.

        Muscle path: pure GPU kernel sequence (CUDA-graph-captured after warmup).
        Non-muscle path: per-world SPD qfrc (CPU round-trip) then batched step.
        """
        import os
        if self._warp_ms is not None:
            trap = bool(os.environ.get("WARP_NAN_TRAP"))
            if trap:
                # Diagnostic path: bypass graph, run kernels eagerly, sample
                # device state between forces and step to catch NaN/blowup.
                self._engine.clear_applied_forces()
                self._warp_ms.update_all()
                self._warp_ms.apply_forces_all()
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
                self._engine.step()
                q = self._engine.get_qpos()
                v = self._engine.get_qvel()
                if not (np.all(np.isfinite(q)) and np.all(np.isfinite(v))):
                    bad_w = np.where(~np.all(np.isfinite(q), axis=1))[0]
                    np.savez("/tmp/warp_div_step.npz", q=q, v=v,
                             xfrc=self._engine.data.xfrc_applied.numpy(),
                             acts=self._activations_buf)
                    raise FloatingPointError(f"qpos/qvel non-finite after warp step bad_worlds={bad_w.tolist()[:8]}")
            else:
                # Fast path: lazy capture of substep physics into a CUDA graph
                # on call #2, replayed via wp.capture_launch on all subsequent
                # calls. First call runs kernels eagerly to initialize any
                # warp/mujoco_warp internal state used by capture.
                import warp as wp
                if not self._substep_warmup_done:
                    self._run_substep_physics()
                    self._substep_warmup_done = True
                elif self._substep_graph is None:
                    with wp.ScopedCapture() as cap:
                        self._run_substep_physics()
                    self._substep_graph = cap.graph
                    wp.capture_launch(self._substep_graph)
                else:
                    wp.capture_launch(self._substep_graph)
                # NaN guard lives outside the graph (D->H check + rollback).
                self._engine.recover_nan_if_any()
            need_sync = any(
                env.sim_count == env.random_sample_index for env in self.envs
            )
            if need_sync:
                self._collect_muscle_tuples_batched()
            for env in self.envs:
                env.sim_count += 1
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

    def get_end_of_episodes(self) -> np.ndarray:
        self._refresh_metrics_if_needed()
        return self._eoe_cache.copy()

    def get_rewards(self) -> np.ndarray:
        self._refresh_metrics_if_needed()
        return self._reward_cache.copy()

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

    def reset_subset(self, indices, rsi: bool = True) -> np.ndarray:
        """Batched reset for a subset of envs.

        cProfile evidence (N=2048, 1K transitions): the per-index path
        ``reset(idx=j)`` was 55% of rollout wall — each call does 8 full
        ``(N, ...)`` GPU<->CPU syncs + one full-N ``forward()``. At N=16384
        that pattern is the dominant per-transition cost and the reason
        throughput inverts (197 trans/s @ N=2048 -> 94 @ N=16384). This
        method collapses k per-index resets into a single scatter + one
        forward.

        Args:
            indices: 1-D array-like of env indices to reset.
            rsi: random start initialization.

        Returns:
            ``(k, num_state)`` ndarray of post-reset states, in the same
            order as ``indices``.
        """
        indices = np.asarray(indices, dtype=np.intp)
        k = int(indices.shape[0])
        if k == 0:
            return np.empty((0, self.num_state), dtype=np.float64)
        if k == self.num_envs:
            self.reset(rsi=rsi)
            return (
                self.get_states_tensor().detach().cpu().numpy()
                .astype(np.float64, copy=False)
            )

        if rsi:
            start_times = np.array(
                [random.uniform(0.0, self._max_time * 0.9) for _ in range(k)],
                dtype=np.float32,
            )
        else:
            start_times = np.zeros((k,), dtype=np.float32)
        target_dart, target_vel_dart = (
            self.envs[0].character.get_target_pos_and_vel_dart_batch(
                start_times, self._dt_ctrl,
            )
        )
        self._target_positions_dart[indices] = target_dart
        self._target_velocities_dart[indices] = target_vel_dart
        target_q = dart_qpos_batch_to_mujoco(
            target_dart, root_offset=DART_ROOT_OFFSET,
        )
        target_v = target_vel_dart.copy()
        target_v[:, :3] = target_vel_dart[:, 3:6]
        target_v[:, 3:6] = target_vel_dart[:, :3]
        self._target_positions_mj[indices] = target_q
        self._target_velocities_mj[indices] = target_v

        eng = self._engine
        qpos = eng.data.qpos.numpy()
        qpos[indices] = target_q.astype(np.float32)
        eng.data.qpos.assign(qpos)

        qvel = eng.data.qvel.numpy()
        qvel[indices] = target_v.astype(np.float32)
        eng.data.qvel.assign(qvel)

        times = eng.data.time.numpy()
        times[indices] = start_times
        eng.data.time.assign(times)

        qfrc = eng.data.qfrc_applied.numpy()
        qfrc[indices] = 0.0
        eng.data.qfrc_applied.assign(qfrc)

        xfrc = eng.data.xfrc_applied.numpy()
        xfrc[indices] = 0.0
        eng.data.xfrc_applied.assign(xfrc)

        eng.forward()

        for k_idx, i in enumerate(indices.tolist()):
            env = self.envs[i]
            env.character.reset()
            env.action[:] = 0.0
            env.target_positions_mj = target_q[k_idx].copy()
            env.target_velocities_mj = target_v[k_idx].copy()
            env.sim_count = 0
            env.random_sample_index = 0
            env.average_activation_levels[:] = 0.0
            if env.activation_levels.size:
                env.activation_levels[:] = 0.0

        self._metrics_dirty = True
        states_np = (
            self.get_states_tensor().detach().cpu().numpy()
            .astype(np.float64, copy=False)
        )
        return states_np[indices]

    def gather_muscle_tuples(self) -> list:
        if self._muscle_tuple_chunks:
            out = {
                key: torch.cat([chunk[key] for chunk in self._muscle_tuple_chunks], dim=0)
                for key in ("JtA", "TauDes", "L", "b")
            }
            self._muscle_tuple_chunks.clear()
            for env in self.envs:
                env.muscle_tuples.clear()

            # Filter rows with any non-finite value (NaN/Inf) before they
            # reach OptimizeMuscleNN. A single bad row poisons the muscle
            # network gradients and breaks training globally.
            JtA = out["JtA"]
            TauDes = out["TauDes"]
            L = out["L"]
            b = out["b"]
            n_total = JtA.shape[0]

            def _row_finite(t: torch.Tensor) -> torch.Tensor:
                if t.dim() == 1:
                    return torch.isfinite(t)
                return torch.isfinite(t.reshape(t.shape[0], -1)).all(dim=1)

            mask = (
                _row_finite(JtA)
                & _row_finite(TauDes)
                & _row_finite(L)
                & _row_finite(b)
            )
            n_valid = int(mask.sum().item())
            if n_valid < n_total:
                dropped = n_total - n_valid
                print(
                    f"[muscle] dropped {dropped}/{n_total} non-finite muscle tuples"
                )
                if n_valid == 0:
                    return {
                        "JtA": JtA[:0],
                        "TauDes": TauDes[:0],
                        "L": L[:0],
                        "b": b[:0],
                    }
                idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                out = {
                    "JtA": JtA.index_select(0, idx),
                    "TauDes": TauDes.index_select(0, idx),
                    "L": L.index_select(0, idx),
                    "b": b.index_select(0, idx),
                }
            return out
        out = []
        for env in self.envs:
            out.extend(env.muscle_tuples)
            env.muscle_tuples.clear()
        return out

    def update_muscle_nn(self, state_dict: dict) -> None:
        del state_dict

    def run_control_step(self, actions: np.ndarray) -> list[tuple[np.ndarray, float, bool]]:
        raise NotImplementedError(
            "WarpGPUEnv packed control path is not implemented yet; "
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

        # Vectorized: replaces two O(N) Python loops over the scalar
        # mujoco_qpos_to_dart + dart_position_differences helpers.
        cur_pos_dart = mujoco_qpos_batch_to_dart(qpos, root_offset=DART_ROOT_OFFSET)
        p_diff_all = dart_position_differences_batched(
            self._target_positions_dart, cur_pos_dart,
        )

        reward_seg = p_diff_all[:, self._reward_dof_idx]
        pose_error_sq = np.sum(reward_seg * reward_seg, axis=1)
        r_q = np.exp(-2.0 * pose_error_sq)
        r_v = np.ones((self.num_envs,), dtype=np.float64)

        # Single sync of xipos for both cur_com and cur_ee.
        cur_xipos = self._engine.data.xipos.numpy()
        cur_com = np.tensordot(
            cur_xipos[:, self._body_ids, :],
            self._body_masses,
            axes=([1], [0]),
        ) / self._body_mass_sum
        cur_ee = cur_xipos[:, self._ee_body_ids, :].copy()

        # Compute target-pose kinematics by temporarily swapping qpos.
        # set_qvel(saved) is intentionally omitted — set_qpos doesn't touch
        # qvel, so it was never disturbed. forward() is replaced with
        # kinematics() + com_pos(): we only need xpos/xmat/xipos restored
        # for the next get_states; the rest of forward() (inverse dynamics,
        # solver, narrowphase) is wasted work here.
        saved_qpos = qpos.copy()
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
        self._engine.kinematics()
        self._engine.com_pos()

        com_diff = cur_com - target_com
        ee_diff = cur_ee - (target_ee + com_diff[:, None, :])
        ee_err_sq = np.sum(ee_diff * ee_diff, axis=(1, 2))
        r_ee = np.exp(-40.0 * ee_err_sq)

        reward = r_ee * (self.envs[0].w_q * r_q + self.envs[0].w_v * r_v)

        root_y = self._engine.data.xpos.numpy()[:, self._root_bid, 1] - (-0.49958)
        has_nan = np.any(np.isnan(qpos), axis=1) | np.any(np.isnan(qvel), axis=1)
        eoe = has_nan | (times > 10.0)
        # Vectorized termination: all envs share the same config (single
        # config_path passed to WarpGPUEnv ctor), so the thresholds are
        # scalar — no per-env Python loop needed.
        e0 = self.envs[0]
        if e0.height_termination:
            eoe |= root_y < 1.3
        if e0.pose_error_threshold is not None:
            eoe |= pose_error_sq > e0.pose_error_threshold

        self._reward_cache[:] = reward
        self._pose_error_sq_cache[:] = pose_error_sq
        self._eoe_cache[:] = eoe
        self._metrics_dirty = False
