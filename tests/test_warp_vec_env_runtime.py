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


if __name__ == "__main__":
    unittest.main()
