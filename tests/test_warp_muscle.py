"""Regression: WarpMuscleSystem vs CPU MuscleSystem."""

import unittest

import numpy as np

from rl_mass.env import MassEnv
from rl_mass.muscle import MuscleSystem
from rl_mass.warp.engine import WarpEngine
from rl_mass.warp.muscle import WarpMuscleSystem


CONFIG = "data/rl_mass/configs/swim.json"
ATOL = 1e-3
RTOL = 1e-3


class WarpMuscleUpdateTests(unittest.TestCase):
    def _build(self, num_worlds: int = 1):
        cpu_env = MassEnv(CONFIG)
        cpu_ms = MuscleSystem(cpu_env.muscles)
        import json
        with open(CONFIG) as fh:
            skel = json.load(fh)["skeleton_file"]
        warp_eng = WarpEngine(
            skel,
            timestep=1.0 / cpu_env.simulation_hz,
            num_worlds=num_worlds,
        )
        warp_ms = WarpMuscleSystem(cpu_ms, warp_eng)

        q0 = cpu_env.engine.get_qpos().astype(np.float32)
        v0 = cpu_env.engine.get_qvel().astype(np.float32)
        warp_eng.set_qpos(q0)
        warp_eng.set_qvel(v0)
        warp_eng.forward()
        return cpu_env, cpu_ms, warp_eng, warp_ms

    def test_update_all_matches_cpu_rest(self):
        cpu_env, cpu_ms, warp_eng, warp_ms = self._build(num_worlds=1)
        try:
            cpu_ms.update_all(cpu_env.engine)
            warp_ms.update_all()

            wp_pos = warp_ms.anchor_world_pos.numpy()[0]
            wp_lmt = warp_ms.l_mt.numpy()[0]
            wp_lm = warp_ms.l_m.numpy()[0]

            np.testing.assert_allclose(wp_pos, cpu_ms.anchor_world_pos, atol=ATOL, rtol=RTOL)
            np.testing.assert_allclose(wp_lmt, cpu_ms.l_mt, atol=ATOL, rtol=RTOL)
            np.testing.assert_allclose(wp_lm, cpu_ms.l_m, atol=ATOL, rtol=RTOL)
        finally:
            warp_eng.close()

    def test_update_all_matches_cpu_after_step(self):
        cpu_env, cpu_ms, warp_eng, warp_ms = self._build(num_worlds=2)
        try:
            for _ in range(20):
                cpu_env.engine.step()
                warp_eng.step()
            cpu_ms.update_all(cpu_env.engine)
            warp_ms.update_all()

            wp_pos = warp_ms.anchor_world_pos.numpy()
            wp_lmt = warp_ms.l_mt.numpy()
            wp_lm = warp_ms.l_m.numpy()

            for w in range(2):
                np.testing.assert_allclose(
                    wp_pos[w], cpu_ms.anchor_world_pos, atol=5e-3, rtol=5e-3,
                    err_msg=f"world {w} anchor_world_pos mismatch",
                )
                np.testing.assert_allclose(
                    wp_lmt[w], cpu_ms.l_mt, atol=5e-3, rtol=5e-3,
                )
                np.testing.assert_allclose(
                    wp_lm[w], cpu_ms.l_m, atol=5e-3, rtol=5e-3,
                )
        finally:
            warp_eng.close()


class WarpMuscleApplyForcesTests(unittest.TestCase):
    def test_apply_forces_matches_cpu(self):
        cpu_env = MassEnv(CONFIG)
        cpu_ms = MuscleSystem(cpu_env.muscles)
        import json
        with open(CONFIG) as fh:
            skel = json.load(fh)["skeleton_file"]
        warp_eng = WarpEngine(skel, timestep=1.0 / cpu_env.simulation_hz, num_worlds=1)
        warp_ms = WarpMuscleSystem(cpu_ms, warp_eng)
        try:
            q0 = cpu_env.engine.get_qpos().astype(np.float32)
            v0 = cpu_env.engine.get_qvel().astype(np.float32)
            warp_eng.set_qpos(q0)
            warp_eng.set_qvel(v0)
            warp_eng.forward()

            rng = np.random.default_rng(42)
            acts = rng.uniform(0.1, 0.9, cpu_ms.n_muscles).astype(np.float64)

            cpu_env.engine.clear_applied_forces()
            cpu_ms.update_all(cpu_env.engine)
            cpu_ms.apply_forces_all(cpu_env.engine, acts)
            cpu_xfrc = cpu_env.engine.data.xfrc_applied.copy()

            warp_eng.clear_applied_forces()
            warp_ms.set_activations(acts.astype(np.float32)[None, :])
            warp_ms.update_all()
            warp_ms.apply_forces_all()
            warp_xfrc = warp_eng.data.xfrc_applied.numpy()[0]

            np.testing.assert_allclose(warp_xfrc, cpu_xfrc, atol=5e-3, rtol=5e-3)
        finally:
            warp_eng.close()


class WarpMuscleTorquesTests(unittest.TestCase):
    def test_get_all_muscle_torques_matches_cpu(self):
        cpu_env = MassEnv(CONFIG)
        cpu_ms = MuscleSystem(cpu_env.muscles)
        import json
        with open(CONFIG) as fh:
            skel = json.load(fh)["skeleton_file"]
        warp_eng = WarpEngine(skel, timestep=1.0 / cpu_env.simulation_hz, num_worlds=1)
        warp_ms = WarpMuscleSystem(cpu_ms, warp_eng)
        try:
            q0 = cpu_env.engine.get_qpos().astype(np.float32)
            v0 = cpu_env.engine.get_qvel().astype(np.float32)
            warp_eng.set_qpos(q0)
            warp_eng.set_qvel(v0)
            warp_eng.forward()

            cpu_ms.update_all(cpu_env.engine)
            cpu_jta = cpu_ms.get_all_muscle_torques(cpu_env.engine)

            warp_ms.set_activations(np.zeros((1, cpu_ms.n_muscles), dtype=np.float32))
            warp_ms.update_all()
            warp_jta_arr = warp_ms.get_all_muscle_torques()
            warp_jta = warp_jta_arr.numpy()[0]

            self.assertEqual(warp_jta.shape, cpu_jta.shape)
            np.testing.assert_allclose(warp_jta, cpu_jta, atol=5e-3, rtol=5e-3)
        finally:
            warp_eng.close()


if __name__ == "__main__":
    unittest.main()
