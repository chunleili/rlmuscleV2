"""Shared-engine vec-env backed by `WarpEngine`."""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from rl_mass.env import MassEnv
from rl_mass.muscle import MuscleSystem
from rl_mass.vec_env import MassVecEnv
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
        return np.array([env.get_state() for env in self.envs], dtype=np.float64)

    def set_actions(self, acts: np.ndarray) -> None:
        for env, action in zip(self.envs, acts):
            env.set_action(action)

    def get_muscle_torques(self) -> np.ndarray:
        if self._warp_ms is None:
            return np.array(
                [env.get_muscle_torques() for env in self.envs], dtype=np.float32
            )
        self._warp_ms.update_all()
        return self._warp_ms.get_all_muscle_torques().numpy().astype(np.float32)

    def get_desired_torques(self) -> np.ndarray:
        return np.array(
            [env.get_desired_torques() for env in self.envs],
            dtype=np.float32,
        )

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
            for env in self.envs:
                if env.sim_count == env.random_sample_index:
                    env._collect_muscle_tuple()
                env.sim_count += 1
            self._engine.step()
        else:
            for env in self.envs:
                env.get_desired_torques()
                env.engine.set_qfrc_applied(env.desired_torque)
                env.sim_count += 1
            self._engine.step()

    def steps(self, n: int) -> None:
        for _ in range(n):
            self._substep_batched()

    def steps_at_once(self) -> None:
        for _ in range(self.num_steps):
            self._substep_batched()

    def is_end_of_episode(self, idx: int) -> bool:
        return self.envs[idx].is_end_of_episode()

    def get_reward(self, idx: int) -> float:
        return self.envs[idx].get_reward()

    def reset(self, rsi: bool = True, idx: Optional[int] = None):
        if idx is None:
            for env in self.envs:
                env.reset(rsi=rsi)
            return None
        return self.envs[idx].reset(rsi=rsi)

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
