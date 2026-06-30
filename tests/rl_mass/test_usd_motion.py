"""UsdSkelMotion parity vs BVH.

The motion USD `motion.mass.walk.usd` is a faithful UsdSkel export of
`walk.bvh`, downsampled 120 -> 60 fps (USD frame i == BVH frame 2i). So
reading it through `UsdSkelMotion` must yield the same DART/MuJoCo target
poses the `BVH` path yields at the corresponding times.
"""
import math
from pathlib import Path

import numpy as np

from rl_mass.bvh import BVH
from rl_mass.character import Character
from rl_mass.engine import MuJoCoEngine
from rl_mass.usd_motion import UsdSkelMotion

ROOT = Path(__file__).resolve().parents[2]
BVH_FILE = ROOT / "data/rl_mass/motion/walk.bvh"
USD_FILE = ROOT / "data/rl_mass/motion/motion.mass.walk.usd"
XML_FILE = ROOT / "data/rl_mass/humanoid.xml"


def test_parse_shape():
    usd = UsdSkelMotion()
    usd.parse(str(USD_FILE), cyclic=True)
    assert usd.num_total_frames == 66, usd.num_total_frames
    assert abs(usd.time_step - 1.0 / 60.0) < 1e-9, usd.time_step
    assert abs(usd.max_time - 1.1) < 1e-6, usd.max_time
    print(f"  frames={usd.num_total_frames} dt={usd.time_step:.5f} max_time={usd.max_time:.4f}")


def test_T0_matches_bvh():
    bvh = BVH()
    bvh.parse(str(BVH_FILE), cyclic=True)
    usd = UsdSkelMotion()
    usd.parse(str(USD_FILE), cyclic=True)
    # Frame 0 is shared between BVH and USD -> root transform must match.
    assert np.allclose(usd.T0, bvh.T0, atol=1e-3), np.abs(usd.T0 - bvh.T0).max()
    print(f"  T0 max|diff|={np.abs(usd.T0 - bvh.T0).max():.2e}")


def test_dart_parity_per_frame():
    bvh = BVH()
    bvh.parse(str(BVH_FILE), cyclic=True)
    usd = UsdSkelMotion()
    usd.parse(str(USD_FILE), cyclic=True)
    # USD frame i (t = i/60) == BVH frame 2i (BVH t=i/60 -> floor(t/dt_bvh)=2i).
    # Skip the last sample (fractional resample endpoint).
    worst = 0.0
    for i in range(usd.num_total_frames - 2):
        t = i / 60.0
        pu = usd.get_motion_dart(t)
        pb = bvh.get_motion_dart(t)
        assert pu.shape == pb.shape == (56,), (pu.shape, pb.shape)
        d = float(np.abs(pu - pb).max())
        worst = max(worst, d)
        assert d < 5e-3, (i, d)
    print(f"  per-frame DART parity worst max|diff|={worst:.2e} over {usd.num_total_frames-2} frames")


def test_character_load_usd_targets():
    eng = MuJoCoEngine(str(XML_FILE), timestep=1.0 / 600)
    ch_b = Character(eng)
    ch_b.load_bvh(str(BVH_FILE), cyclic=True)
    ch_b.reset()
    ch_u = Character(eng)
    ch_u.load_usd(str(USD_FILE), cyclic=True)
    ch_u.reset()
    worst = 0.0
    for i in range(30):
        t = i / 30.0
        qb = ch_b.get_target_positions_mujoco(t, 1.0 / 30.0)
        qu = ch_u.get_target_positions_mujoco(t, 1.0 / 30.0)
        assert qu.shape == (71,), qu.shape
        # Character cyclic wrap (Tc) advances per call; at t aligned to shared
        # frames the targets agree closely.
        d = float(np.abs(qb - qu).max())
        worst = max(worst, d)
    print(f"  Character target qpos worst max|diff| over 30 ctrl steps = {worst:.2e}")
    assert worst < 5e-2, worst


def main() -> int:
    for fn in (
        test_parse_shape,
        test_T0_matches_bvh,
        test_dart_parity_per_frame,
        test_character_load_usd_targets,
    ):
        print(f"[{fn.__name__}]")
        fn()
        print("  PASS")
    print("ALL USD-MOTION PARITY TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
