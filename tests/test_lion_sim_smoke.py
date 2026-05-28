"""End-to-end smoke test for OneWaySim against real lion data."""
import json
from pathlib import Path

import numpy as np
import pytest
import warp as wp

wp.init()

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "model" / "lion"
CONFIG = ROOT / "data" / "scene" / "lion_hpbd.json"

skip_no_data = pytest.mark.skipif(
    not (DATA / "restpos.geo").exists(),
    reason="lion .geo blobs not present",
)


@skip_no_data
def test_lion_init_and_one_frame_no_nan():
    from muscle.coupleOneWay2.oneWay import OneWaySim

    cfg = json.load(open(CONFIG))
    cfg["geo_dir"] = str(DATA)
    sim = OneWaySim(cfg)

    p_rest = sim.positions_np().copy()
    assert sim.NV == p_rest.shape[0]
    assert sim.NCONS > 0

    sim.step_frame(sim.start_frame)
    p_after = sim.positions_np()
    assert not np.isnan(p_after).any(), "NaN in positions after one frame"
    assert not np.isinf(p_after).any(), "Inf in positions after one frame"

    # Drift relative to mesh diagonal stays small under one frame of soft ARAP.
    diag = float(np.linalg.norm(p_rest.max(0) - p_rest.min(0)))
    max_disp = float(np.linalg.norm(p_after - p_rest, axis=1).max())
    rel = max_disp / diag
    # Frame 1's target_pos may equal rest pos (when start_frame=1), giving ~0
    # drift; either way it should be well under 10% of the diagonal.
    assert rel < 0.1, f"unexpectedly large drift on frame 1: {rel:.3%} of diag"


@skip_no_data
def test_lion_five_frames_run():
    from muscle.coupleOneWay2.oneWay import OneWaySim

    cfg = json.load(open(CONFIG))
    cfg["geo_dir"] = str(DATA)
    cfg["end_frame"] = 5
    sim = OneWaySim(cfg)

    for frame in range(sim.start_frame, sim.end_frame + 1):
        sim.step_frame(frame)
        p = sim.positions_np()
        assert not np.isnan(p).any(), f"NaN at frame {frame}"
