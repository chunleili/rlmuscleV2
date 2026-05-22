import unittest

import numpy as np

from rl_mass.character import Character
from rl_mass.engine import MuJoCoEngine
from rl_mass.joint_utils import DART_ROOT_OFFSET, dart_qpos_to_mujoco


MODEL = "data/rl_mass/humanoid_swim.xml"
BVH = "data/rl_mass/motion/walk.bvh"
TIMESTEP = 1.0 / 600.0
DT_CTRL = 1.0 / 30.0


class CharacterMotionCacheTests(unittest.TestCase):
    def test_cached_target_sampling_matches_reference_sequence(self):
        engine = MuJoCoEngine(MODEL, timestep=TIMESTEP)
        try:
            char = Character(engine)
            char.load_bvh(BVH, cyclic=True)
            char.reset()

            Tc_ref = char.bvh.T0.copy()
            Tc_ref[1, 3] = 0.0
            max_time = char.bvh.max_time
            times = [
                0.0,
                0.25 * max_time,
                0.94 * max_time,
                0.97 * max_time,
                1.02 * max_time,
                1.35 * max_time,
            ]

            for t in times:
                p_ref, Tc_ref_next = char._get_target_positions_dart_uncached(
                    t, DT_CTRL, Tc_ref
                )
                p1_ref, _ = char._get_target_positions_dart_uncached(
                    t + DT_CTRL, DT_CTRL, Tc_ref_next
                )
                v_ref = (p1_ref - p_ref) / DT_CTRL
                mj_q_ref = dart_qpos_to_mujoco(p_ref, root_offset=DART_ROOT_OFFSET)
                mj_v_ref = v_ref.copy()
                mj_v_ref[:3] = v_ref[3:6]
                mj_v_ref[3:6] = v_ref[:3]

                mj_q, mj_v = char.get_target_pos_and_vel_mujoco(t, DT_CTRL)
                np.testing.assert_allclose(mj_q, mj_q_ref, atol=1e-8, rtol=1e-8)
                np.testing.assert_allclose(mj_v, mj_v_ref, atol=1e-8, rtol=1e-8)

                Tc_ref = Tc_ref_next
        finally:
            pass

    def test_batched_sampler_matches_scalar_sampler(self):
        engine = MuJoCoEngine(MODEL, timestep=TIMESTEP)
        try:
            char = Character(engine)
            char.load_bvh(BVH, cyclic=True)

            times = np.array(
                [
                    0.0,
                    0.1,
                    0.3,
                    0.7,
                    0.95 * char.bvh.max_time,
                ],
                dtype=np.float64,
            )
            q_batch, v_batch = char.get_target_pos_and_vel_mujoco_batch(
                times,
                DT_CTRL,
            )

            q_expected = []
            v_expected = []
            for t in times:
                char.reset()
                q_i, v_i = char.get_target_pos_and_vel_mujoco(float(t), DT_CTRL)
                q_expected.append(q_i)
                v_expected.append(v_i)

            np.testing.assert_allclose(
                q_batch,
                np.asarray(q_expected, dtype=np.float64),
                atol=1e-8,
                rtol=1e-8,
            )
            np.testing.assert_allclose(
                v_batch,
                np.asarray(v_expected, dtype=np.float64),
                atol=1e-8,
                rtol=1e-8,
            )
        finally:
            pass


if __name__ == "__main__":
    unittest.main()
