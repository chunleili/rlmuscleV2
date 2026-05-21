"""Shared-engine vec-env backed by `WarpEngine`."""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from rl_mass.env import MassEnv
from rl_mass.vec_env import MassVecEnv
from rl_mass.warp.engine import WarpEngine, WarpWorldView


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

    def get_states(self) -> np.ndarray:
        return np.array([env.get_state() for env in self.envs], dtype=np.float64)

    def set_actions(self, acts: np.ndarray) -> None:
        for env, action in zip(self.envs, acts):
            env.set_action(action)

    def get_muscle_torques(self) -> np.ndarray:
        return np.array(
            [env.get_muscle_torques() for env in self.envs],
            dtype=np.float32,
        )

    def get_desired_torques(self) -> np.ndarray:
        return np.array(
            [env.get_desired_torques() for env in self.envs],
            dtype=np.float32,
        )

    def set_activation_levels(self, acts: np.ndarray) -> None:
        for env, activation in zip(self.envs, acts):
            env.set_activation_levels(activation)

    def steps(self, n: int) -> None:
        for _ in range(n):
            for env in self.envs:
                env.step()

    def steps_at_once(self) -> None:
        for _ in range(self.envs[0].num_steps):
            for env in self.envs:
                env.step()

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
