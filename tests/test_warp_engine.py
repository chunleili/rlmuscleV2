import unittest

import numpy as np

from rl_mass.warp.engine import WarpEngine


class WarpEngineTests(unittest.TestCase):
    def test_constructs_and_exposes_batched_dimensions(self):
        engine = WarpEngine("data/rl_mass/humanoid_swim.xml", timestep=1.0 / 600.0, num_worlds=2)
        try:
            self.assertEqual(engine.num_worlds, 2)
            self.assertGreater(engine.nq, 0)
            self.assertGreater(engine.nv, 0)
            self.assertGreater(engine.nbody, 0)
            self.assertAlmostEqual(engine.timestep, 1.0 / 600.0)
        finally:
            engine.close()

    def test_qpos_qvel_roundtrip_and_step(self):
        engine = WarpEngine("data/rl_mass/humanoid_swim.xml", timestep=1.0 / 600.0, num_worlds=2)
        try:
            qpos = engine.get_qpos()
            qvel = engine.get_qvel()
            self.assertEqual(qpos.shape, (2, engine.nq))
            self.assertEqual(qvel.shape, (2, engine.nv))

            qpos_new = qpos.copy()
            qvel_new = qvel.copy()
            qpos_new[1, 0] += 0.125
            qvel_new[1, 0] = 0.5

            engine.set_qpos(qpos_new)
            engine.set_qvel(qvel_new)
            np.testing.assert_allclose(engine.get_qpos(), qpos_new, atol=1e-6)
            np.testing.assert_allclose(engine.get_qvel(), qvel_new, atol=1e-6)

            t0 = engine.get_time().copy()
            engine.step()
            t1 = engine.get_time()
            self.assertEqual(t0.shape, (2,))
            np.testing.assert_allclose(t1 - t0, np.full((2,), engine.timestep), atol=1e-6)
        finally:
            engine.close()

    def test_reset_restores_initial_state(self):
        engine = WarpEngine("data/rl_mass/humanoid_swim.xml", timestep=1.0 / 600.0, num_worlds=2)
        try:
            qpos0 = engine.get_qpos().copy()
            qvel0 = engine.get_qvel().copy()
            engine.set_qvel(np.ones_like(qvel0, dtype=np.float32))
            engine.step()
            engine.reset()
            np.testing.assert_allclose(engine.get_qpos(), qpos0, atol=1e-6)
            np.testing.assert_allclose(engine.get_qvel(), qvel0, atol=1e-6)
            np.testing.assert_allclose(engine.get_time(), np.zeros((2,), dtype=np.float32), atol=1e-6)
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
