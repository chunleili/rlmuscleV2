"""EMGLoader: parse Moreira 2021 sEMG envelope files into GTSeries.

Tests:
  - Real P8_V4_R data → correct names, shape, values.
  - Missing-muscle raises ValueError.
  - Empty directory raises ValueError.
"""
import os
import tempfile

import numpy as np

from rl_mass.finger.gt import EMGLoader


EMG_DIR = "data/rl_mass/emg_moreira/P8_V4_R"


def test_real_data():
    gt = EMGLoader(EMG_DIR).load()
    assert gt.names == ["VL", "BF", "TA", "GAL"], f"names mismatch: {gt.names}"
    assert gt.activation.shape == (101, 4), f"shape {gt.activation.shape}"
    assert float(gt.activation.min()) >= 0.0, "negative activations"
    assert float(gt.time[0]) == 0.0
    assert abs(float(gt.time[-1]) - 100.0) < 1e-6, f"time[-1]={gt.time[-1]}"
    assert len(gt.time) == 101


def test_subset_muscles():
    gt = EMGLoader(EMG_DIR, muscles=["VL", "TA"]).load()
    assert gt.names == ["VL", "TA"]
    assert gt.activation.shape == (101, 2)


def test_missing_muscle_raises():
    try:
        EMGLoader(EMG_DIR, muscles=["VL", "NONEXISTENT"]).load()
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "NONEXISTENT" in str(e), str(e)


def test_empty_dir_raises():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            EMGLoader(tmp).load()
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "no T*.txt" in str(e).lower() or "T*.txt" in str(e), str(e)


def main():
    test_real_data()
    test_subset_muscles()
    test_missing_muscle_raises()
    test_empty_dir_raises()
    print("OK")


if __name__ == "__main__":
    main()
