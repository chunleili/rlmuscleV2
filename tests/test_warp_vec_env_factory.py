import sys
import types
import unittest

from rl_mass.vec_env import make_vec_env


class _DummyWarpVecEnv:
    def __init__(self, config_path: str, num_envs: int, seed=None):
        self.config_path = config_path
        self.num_envs = num_envs
        self.seed = seed


class WarpVecEnvFactoryTests(unittest.TestCase):
    def test_make_vec_env_routes_warp_mode_even_for_single_env(self):
        fake_module = types.ModuleType("rl_mass.warp")
        fake_module.WarpVecEnv = _DummyWarpVecEnv
        old_module = sys.modules.get("rl_mass.warp")
        sys.modules["rl_mass.warp"] = fake_module
        try:
            env = make_vec_env("dummy.json", num_envs=1, mode="warp", seed=123)
        finally:
            if old_module is None:
                sys.modules.pop("rl_mass.warp", None)
            else:
                sys.modules["rl_mass.warp"] = old_module

        self.assertIsInstance(env, _DummyWarpVecEnv)
        self.assertEqual(env.config_path, "dummy.json")
        self.assertEqual(env.num_envs, 1)
        self.assertEqual(env.seed, 123)

    def test_unknown_mode_error_mentions_warp_choice(self):
        with self.assertRaises(ValueError) as ctx:
            make_vec_env("dummy.json", num_envs=2, mode="bogus")

        self.assertIn("warp", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
