"""Numerical regression: WarpEngine vs MuJoCoEngine on a free-rollout.

fp32 GPU step won't bit-match fp64 CPU mj_step, so we use a loose tolerance
on a short horizon. Goal: catch ordering / unit / sign bugs, not chase
exactness.
"""

import unittest

import numpy as np

from rl_mass.env import MassEnv
from rl_mass.engine import MuJoCoEngine
from rl_mass.warp.engine import WarpEngine
from rl_mass.warp.gpu_env import WarpGPUEnv


MODEL = "data/rl_mass/humanoid_swim.xml"
TIMESTEP = 1.0 / 600.0
N_STEPS = 30
ATOL = 5e-3
RTOL = 1e-2


class WarpVsCpuTests(unittest.TestCase):
    def test_qpos_qvel_xpos_match_after_free_rollout(self):
        cpu = MuJoCoEngine(MODEL, timestep=TIMESTEP)
        warp = WarpEngine(MODEL, timestep=TIMESTEP, num_worlds=1)
        try:
            q0 = cpu.get_qpos()
            v0 = cpu.get_qvel()

            warp.set_qpos(q0.astype(np.float32))
            warp.set_qvel(v0.astype(np.float32))
            warp.set_time(0.0)
            warp.forward()
            cpu.forward()

            for _ in range(N_STEPS):
                cpu.step()
                warp.step()

            q_cpu = cpu.get_qpos()
            v_cpu = cpu.get_qvel()
            q_warp = warp.get_qpos()[0]
            v_warp = warp.get_qvel()[0]

            np.testing.assert_allclose(q_warp, q_cpu, atol=ATOL, rtol=RTOL)
            np.testing.assert_allclose(v_warp, v_cpu, atol=ATOL, rtol=RTOL)

            for bid in range(cpu.nbody):
                p_cpu = cpu.get_body_xpos(bid)
                p_warp = warp.get_body_xpos(bid)[0]
                np.testing.assert_allclose(
                    p_warp, p_cpu, atol=ATOL, rtol=RTOL,
                    err_msg=f"xpos mismatch at body {bid}",
                )
        finally:
            warp.close()

    def test_control_step_matches_cpu_with_zero_action(self):
        cpu = MassEnv("data/rl_mass/configs/swim.json")
        warp = WarpGPUEnv("data/rl_mass/configs/swim.json", num_envs=1, seed=0)
        try:
            cpu.reset(rsi=False)
            warp.reset(rsi=False)

            action = np.zeros(cpu.num_action, dtype=np.float64)
            activation = np.zeros(cpu.num_muscles, dtype=np.float64)

            cpu.set_action(action)
            cpu.set_activation_levels(activation)
            warp.set_actions(action[None, :].astype(np.float32))
            warp.set_activation_levels(activation[None, :].astype(np.float32))

            cpu.steps_at_once()
            warp.steps_at_once()

            q_cpu = cpu.engine.get_qpos()
            v_cpu = cpu.engine.get_qvel()
            q_warp = warp._engine.get_qpos()[0]
            v_warp = warp._engine.get_qvel()[0]

            np.testing.assert_allclose(q_warp, q_cpu, atol=1e-2, rtol=1e-2)
            np.testing.assert_allclose(v_warp, v_cpu, atol=1e-2, rtol=1e-2)
            self.assertAlmostEqual(
                warp.get_reward(0),
                cpu.get_reward(),
                delta=5e-3,
            )
            self.assertEqual(
                warp.is_end_of_episode(0),
                cpu.is_end_of_episode(),
            )
        finally:
            warp.close()

    def test_reward_and_termination_match_cpu_after_reset(self):
        cpu = MassEnv("data/rl_mass/configs/swim.json")
        warp = WarpGPUEnv("data/rl_mass/configs/swim.json", num_envs=1, seed=0)
        try:
            t = 0.35
            cpu.reset(rsi=False, t=t)
            warp.reset(rsi=False)
            warp._engine.set_time(np.array([t], dtype=np.float32))
            q = cpu.engine.get_qpos()
            v = cpu.engine.get_qvel()
            warp._engine.set_qpos(q.astype(np.float32))
            warp._engine.set_qvel(v.astype(np.float32))
            warp._engine.forward()
            warp.envs[0].target_positions_mj = cpu.target_positions_mj.copy()
            warp.envs[0].target_velocities_mj = cpu.target_velocities_mj.copy()
            warp._target_positions_mj[0] = cpu.target_positions_mj.copy()
            warp._target_velocities_mj[0] = cpu.target_velocities_mj.copy()
            warp._target_positions_dart[0] = cpu.character.get_target_positions_dart(t, 1.0 / cpu.control_hz)
            target_vel_dart = cpu.target_velocities_mj.copy()
            target_vel_dart[:3] = cpu.target_velocities_mj[3:6]
            target_vel_dart[3:6] = cpu.target_velocities_mj[:3]
            warp._target_velocities_dart[0] = target_vel_dart
            warp._metrics_dirty = True

            self.assertAlmostEqual(
                warp.get_reward(0),
                cpu.get_reward(),
                delta=5e-3,
            )
            self.assertEqual(
                warp.is_end_of_episode(0),
                cpu.is_end_of_episode(),
            )
        finally:
            warp.close()


if __name__ == "__main__":
    unittest.main()
