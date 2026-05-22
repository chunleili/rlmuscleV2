import unittest

import numpy as np
import warp as wp

from rl_mass.joint_utils import DART_QPOS_SIZE, dart_position_differences
from rl_mass.warp.dart_diff import WarpDartDiff


class WarpDartDiffTests(unittest.TestCase):
    def test_matches_cpu_dart_position_differences(self):
        rng = np.random.default_rng(0)
        num_worlds = 8
        q1 = rng.normal(scale=0.2, size=(num_worlds, DART_QPOS_SIZE)).astype(np.float32)
        q2 = rng.normal(scale=0.2, size=(num_worlds, DART_QPOS_SIZE)).astype(np.float32)

        cpu = np.stack(
            [dart_position_differences(q2[i], q1[i]) for i in range(num_worlds)],
            axis=0,
        ).astype(np.float32)

        diff = WarpDartDiff(num_worlds=num_worlds)
        device = diff.device
        with wp.ScopedDevice(device):
            q1_wp = wp.array(q1, dtype=wp.float32)
            q2_wp = wp.array(q2, dtype=wp.float32)
            out_wp = wp.zeros_like(q1_wp)
            diff.compute(q1_wp, q2_wp, out_wp)
            gpu = out_wp.numpy()

        np.testing.assert_allclose(gpu, cpu, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
