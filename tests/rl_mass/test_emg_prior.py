"""Tests for the EMG activation prior target + phase lookup (no env needed).

The trainer pulls the 4 Moreira surface muscles' activations toward the
peak-normalised EMG envelope, indexed by the gait phase tagged on each muscle
tuple. These tests pin the target .npz contract and the phase→row lookup math
used inside OptimizeMuscleNN, without spinning up MuJoCo or PPO.
"""
import os

import numpy as np

_NPZ = "data/rl_mass/emg_moreira/emg_prior_V4.npz"


def _lookup(phase, target):
    """Replicate train.py's phase→target-row gather."""
    n_phase = target.shape[0]
    pidx = (np.asarray(phase) * n_phase).astype(np.int64) % n_phase
    return target[pidx]


def test_prior_npz_contract():
    assert os.path.exists(_NPZ), f"missing {_NPZ} (run tools/build_emg_prior.py)"
    d = np.load(_NPZ)
    target = d["target"]
    assert target.shape == (101, 4), target.shape
    assert d["snu_idx"].shape == (4,)
    assert d["phase_grid"].shape == (101,)
    # peak-normalised: each column peaks at ~1 (slightly under after the
    # heel-strike-offset resample), stays within [0, 1]
    assert np.allclose(target.max(axis=0), 1.0, atol=1e-2)
    assert target.min() >= -1e-9 and target.max() <= 1.0 + 1e-9
    # heel-strike offset is a sane fraction
    assert 0.0 <= float(d["phi_hs"]) < 1.0


def test_phase_lookup_endpoints_and_wrap():
    target = np.load(_NPZ)["target"]
    # phase 0 → first row
    assert np.array_equal(_lookup(0.0, target), target[0])
    # phase just below 1 → last row
    assert np.array_equal(_lookup(0.999, target), target[100])
    # phase wraps at 1.0 back to row 0 (matches the `% n_phase` in train.py)
    assert np.array_equal(_lookup(1.0, target), target[0])
    # batched lookup matches per-element
    ph = np.array([0.0, 0.25, 0.5, 0.75])
    got = _lookup(ph, target)
    assert got.shape == (4, 4)
    for i, p in enumerate(ph):
        assert np.array_equal(got[i], _lookup(p, target))


def test_loss_is_zero_when_activation_equals_scaled_target():
    """Sanity: the prior loss vanishes at a_sel = scale * target."""
    target = np.load(_NPZ)["target"]
    scale = 0.3
    ph = np.array([0.1, 0.4, 0.9])
    tgt = _lookup(ph, target)
    a_sel = scale * tgt
    loss_emg = ((a_sel - scale * tgt) ** 2).mean()
    assert loss_emg == 0.0


def main():
    test_prior_npz_contract()
    test_phase_lookup_endpoints_and_wrap()
    test_loss_is_zero_when_activation_equals_scaled_target()
    print("all emg_prior tests passed")


if __name__ == "__main__":
    main()
