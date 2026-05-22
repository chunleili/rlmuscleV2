import unittest

import numpy as np

from rl_mass.warp.engine import WarpEngine
from rl_mass.warp.vec_env import WarpVecEnv


class WarpVecEnvRuntimeTests(unittest.TestCase):
    def test_uses_shared_warp_engine_and_exposes_states(self):
        env = WarpVecEnv("data/rl_mass/configs/swim.json", num_envs=2, seed=123)
        try:
            self.assertIsInstance(env._engine, WarpEngine)
            self.assertEqual(env.num_envs, 2)
            states = env.get_states()
            self.assertEqual(states.shape, (2, env.num_state))
        finally:
            env.close()

    def test_batched_states_match_per_env_state(self):
        env = WarpVecEnv("data/rl_mass/configs/swim.json", num_envs=3, seed=123)
        try:
            env.reset(rsi=False)
            expected = np.array([subenv.get_state() for subenv in env.envs], dtype=np.float64)
            actual = env.get_states()
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        finally:
            env.close()

    def test_steps_advances_all_world_times_together(self):
        env = WarpVecEnv("data/rl_mass/configs/swim.json", num_envs=2, seed=123)
        try:
            env.set_actions(np.zeros((2, env.num_action), dtype=np.float32))
            t0 = env._engine.get_time().copy()
            env.steps(1)
            t1 = env._engine.get_time()
            np.testing.assert_allclose(
                t1 - t0,
                np.full((2,), env._engine.timestep, dtype=np.float32),
                atol=1e-6,
            )
        finally:
            env.close()

    def test_set_actions_updates_targets_from_shared_batch_sampler(self):
        env = WarpVecEnv("data/rl_mass/configs/swim.json", num_envs=3, seed=123)
        try:
            env.reset(rsi=False)
            env._engine.set_time(np.array([0.0, 0.05, 0.1], dtype=np.float32))
            acts = np.random.default_rng(7).normal(
                size=(3, env.num_action)
            ).astype(np.float32)

            expected_q = []
            expected_v = []
            for subenv, action in zip(env.envs, acts):
                subenv.set_action(action)
                expected_q.append(subenv.target_positions_mj.copy())
                expected_v.append(subenv.target_velocities_mj.copy())

            env.set_actions(acts.copy())

            for i, subenv in enumerate(env.envs):
                np.testing.assert_allclose(
                    subenv.target_positions_mj,
                    expected_q[i],
                    atol=1e-8,
                    rtol=1e-8,
                )
                np.testing.assert_allclose(
                    subenv.target_velocities_mj,
                    expected_v[i],
                    atol=1e-8,
                    rtol=1e-8,
                )
                np.testing.assert_allclose(
                    subenv.action,
                    acts[i] * 0.1,
                    atol=1e-8,
                    rtol=1e-8,
                )
        finally:
            env.close()


class WarpVecEnvBatchedMuscleTests(unittest.TestCase):
    def test_batched_desired_torques_match_per_env_fallback(self):
        env = WarpVecEnv("data/rl_mass/configs/swim.json", num_envs=3, seed=0)
        try:
            env.reset(rsi=False)
            acts = np.random.default_rng(1).normal(
                size=(3, env.num_action)
            ).astype(np.float32)
            env.set_actions(acts)

            expected = np.array(
                [subenv.get_desired_torques() for subenv in env.envs],
                dtype=np.float32,
            )
            actual = env.get_desired_torques()

            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
        finally:
            env.close()

    def test_full_control_step_with_batched_muscles(self):
        env = WarpVecEnv("data/rl_mass/configs/swim.json", num_envs=4, seed=0)
        try:
            env.reset(rsi=False)
            self.assertIsNotNone(env._warp_ms)
            acts = np.zeros((4, env.num_action), dtype=np.float32)
            env.set_actions(acts)

            mus = np.random.default_rng(0).uniform(
                0.0, 0.5, (4, env.num_muscles)
            ).astype(np.float32)
            env.set_activation_levels(mus)

            t0 = env._engine.get_time().copy()
            env.steps_at_once()
            t1 = env._engine.get_time()
            np.testing.assert_allclose(
                t1 - t0,
                np.full((4,), env.num_steps * env._engine.timestep, dtype=np.float32),
                atol=1e-4,
            )

            jta = env.get_muscle_torques()
            self.assertEqual(jta.shape, (4, env.num_total_related_dofs))
            self.assertTrue(np.all(np.isfinite(jta)))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
