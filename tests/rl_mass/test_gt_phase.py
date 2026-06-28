"""Tests for rl_mass.finger.gt.phase: anterior_projection, detect_heel_strikes,
fold_cycles.

All tests are synthetic — no env or torch dependency.
"""
import numpy as np

from rl_mass.finger.gt import anterior_projection, detect_heel_strikes, fold_cycles


# ──────────────────────────────────────────────────────────────────────────────
# anterior_projection
# ──────────────────────────────────────────────────────────────────────────────

def test_anterior_projection_basic():
    """A foot moving +x relative to a forward-moving (+x) pelvis should give
    positive, increasing projection values."""
    T = 10
    # Pelvis moves steadily in +x; foot starts ahead and moves faster
    pelvis = np.column_stack([np.linspace(0, 1, T), np.zeros(T), np.zeros(T)])
    foot   = np.column_stack([np.linspace(0.5, 2.0, T), np.zeros(T), np.zeros(T)])
    proj   = anterior_projection(foot, pelvis, up_axis=1)
    assert proj.shape == (T,)
    # All values positive (foot is always ahead)
    assert np.all(proj > 0), f"expected positive proj, got {proj}"
    # Monotonically increasing
    assert np.all(np.diff(proj) >= 0), f"expected non-decreasing, got {proj}"


def test_anterior_projection_up_axis_zeroed():
    """Vertical displacement of pelvis should not affect walking direction."""
    T = 6
    # Pelvis moves in +x AND bounces in y (up_axis=1)
    pelvis = np.column_stack([
        np.linspace(0, 1, T),
        np.sin(np.linspace(0, np.pi, T)),  # vertical bounce
        np.zeros(T),
    ])
    # Foot is always 0.5 ahead in x, same y as pelvis
    foot = pelvis + np.array([0.5, 0, 0])
    proj = anterior_projection(foot, pelvis, up_axis=1)
    # Relative foot position in x is always 0.5; projection ~ 0.5
    assert np.allclose(proj, 0.5, atol=1e-6), f"proj={proj}"


# ──────────────────────────────────────────────────────────────────────────────
# detect_heel_strikes
# ──────────────────────────────────────────────────────────────────────────────

def test_detect_heel_strikes_triangle():
    """Triangle wave: peaks at known indices."""
    # Build a signal with peaks at t=10, 30, 50, 70 (period=20)
    T = 80
    t = np.arange(T, dtype=float)
    # Sawtooth-then-fall: rise for 10 samples then fall for 10
    proj = 1.0 - np.abs((t % 20) - 10) / 10.0   # peaks at 10,30,50,70
    strikes = detect_heel_strikes(proj)
    # Should find peaks at 10, 30, 50, 70 (boundaries at 0 and 79 excluded)
    assert set(strikes) == {10, 30, 50, 70}, f"got strikes {strikes}"


def test_detect_heel_strikes_single_peak():
    proj = np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0], dtype=float)
    strikes = detect_heel_strikes(proj)
    assert strikes == [2], f"got {strikes}"


def test_detect_heel_strikes_no_peaks():
    proj = np.zeros(10)
    strikes = detect_heel_strikes(proj)
    # Flat signal: no strict local max
    assert strikes == [], f"got {strikes}"


# ──────────────────────────────────────────────────────────────────────────────
# fold_cycles
# ──────────────────────────────────────────────────────────────────────────────

def test_fold_cycles_shape():
    """With two cycles of length 50, output should be (101, 2)."""
    T = 150
    signal = np.column_stack([np.linspace(0, 1, T), np.linspace(1, 2, T)])
    strikes = [10, 60, 110]        # two cycles of length 50
    out = fold_cycles(signal, strikes, n_points=101)
    assert out.shape == (101, 2), f"shape {out.shape}"


def test_fold_cycles_ramp_mean():
    """A linear ramp from 0→1 over every cycle should fold to ~[0,1] linspace."""
    T = 200
    # Build a signal that ramps 0→1 over each 50-sample cycle
    cycle_len = 50
    ramp = np.tile(np.linspace(0, 1, cycle_len), T // cycle_len)[:T]
    signal = ramp[:, np.newaxis]
    strikes = list(range(0, T, cycle_len))    # [0, 50, 100, 150]
    out = fold_cycles(signal, strikes, n_points=101)
    expected = np.linspace(0, 1, 101)
    assert np.allclose(out[:, 0], expected, atol=0.02), (
        f"max err {np.max(np.abs(out[:, 0] - expected)):.4f}"
    )


def test_fold_cycles_skips_short():
    """Pairs shorter than min_len should be ignored."""
    T = 120
    signal = np.ones((T, 1))
    strikes = [10, 15, 60, 110]   # (10→15)=5 short, (15→60)=45 ok, (60→110)=50 ok
    out = fold_cycles(signal, strikes, n_points=101, min_len=10)
    # With valid input=1 everywhere, mean should be 1
    assert np.allclose(out, 1.0), f"mean not 1: {out}"


def test_fold_cycles_no_valid_pairs():
    """All pairs shorter than min_len → zeros."""
    signal = np.ones((20, 2))
    strikes = [0, 5, 10]    # both pairs length 5 < min_len=10
    out = fold_cycles(signal, strikes, n_points=101, min_len=10)
    assert np.all(out == 0), f"expected zeros, got {out}"


def main():
    test_anterior_projection_basic()
    test_anterior_projection_up_axis_zeroed()
    test_detect_heel_strikes_triangle()
    test_detect_heel_strikes_single_peak()
    test_detect_heel_strikes_no_peaks()
    test_fold_cycles_shape()
    test_fold_cycles_ramp_mean()
    test_fold_cycles_skips_short()
    test_fold_cycles_no_valid_pairs()
    print("OK")


if __name__ == "__main__":
    main()
