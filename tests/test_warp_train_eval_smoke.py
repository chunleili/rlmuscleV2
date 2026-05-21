import unittest

import numpy as np

from rl_mass.train import PPO
from rl_mass.warp.vec_env import WarpVecEnv


class WarpTrainEvalSmokeTests(unittest.TestCase):
    def test_warp_env_compute_diagnostics(self):
        env = WarpVecEnv("data/rl_mass/configs/swim.json", num_envs=1, seed=123)
        try:
            env.set_actions(np.zeros((1, env.num_action), dtype=np.float32))
            env.steps_at_once()
            diag = env.envs[0].compute_diagnostics()
            self.assertEqual(diag["R"].shape, (50, env.num_muscles))
            self.assertEqual(diag["L"].shape, (50, env.num_muscles))
            self.assertEqual(diag["b"].shape, (50,))
            self.assertEqual(diag["tau_des"].shape, (50,))
            self.assertEqual(diag["tau_muscle"].shape, (50,))
            self.assertEqual(diag["a"].shape, (env.num_muscles,))
        finally:
            env.close()

    def test_ppo_initializes_with_warp_backend(self):
        ppo = PPO(
            "data/rl_mass/configs/swim.json",
            num_envs=1,
            vec_env_mode="warp",
            seed=123,
            packed_muscle_loop=False,
            reward_csv_path=None,
            nn_dir="output/test-warp-nn",
        )
        try:
            states = ppo.env.get_states()
            self.assertEqual(states.shape, (1, ppo.num_state))
            self.assertTrue(ppo.use_muscle)
        finally:
            ppo.env.close()


if __name__ == "__main__":
    unittest.main()
