"""Numerical regression: WarpEngine vs MuJoCoEngine on a free-rollout.

fp32 GPU step won't bit-match fp64 CPU mj_step, so we use a loose tolerance
on a short horizon. Goal: catch ordering / unit / sign bugs, not chase
exactness.
"""

import unittest

import numpy as np

from rl_mass.engine import MuJoCoEngine
from rl_mass.warp.engine import WarpEngine


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


if __name__ == "__main__":
    unittest.main()
