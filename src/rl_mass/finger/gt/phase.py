"""Gait-phase utilities for heel-strike anchoring (Zeni 2008).

All functions are pure NumPy — no env or torch dependency. Used by the
walk-vs-EMG validation tool to fold a policy rollout onto the 0–100% gait
cycle grid via right-heel-strike detection.
"""
from __future__ import annotations

import numpy as np


def anterior_projection(
    foot_xyz: np.ndarray,
    pelvis_xyz: np.ndarray,
    up_axis: int = 1,
) -> np.ndarray:
    """Project foot position (relative to pelvis) onto the walking direction.

    Walking direction is the *horizontal* net pelvis displacement over the
    entire rollout (the ``up_axis`` component is zeroed out).  The projection
    gives the anterior distance of the foot ahead of the pelvis — it peaks at
    right heel strike (Zeni 2008).

    Args:
        foot_xyz:   (T, 3) world-frame foot marker positions (e.g. TalusR).
        pelvis_xyz: (T, 3) world-frame pelvis positions.
        up_axis:    axis index for vertical (1 = Y-up MuJoCo default).

    Returns:
        (T,) projection values.
    """
    foot_xyz = np.asarray(foot_xyz, dtype=float)
    pelvis_xyz = np.asarray(pelvis_xyz, dtype=float)

    # Net pelvis displacement, horizontal only
    disp = pelvis_xyz[-1] - pelvis_xyz[0]
    disp[up_axis] = 0.0
    norm = np.linalg.norm(disp)
    fwd = disp / (norm + 1e-9)

    return ((foot_xyz - pelvis_xyz) * fwd).sum(axis=1)


def detect_heel_strikes(proj: np.ndarray) -> list[int]:
    """Indices of right heel strikes in an anterior-projection signal.

    Uses the Zeni (2008) criterion: local maxima of the anterior foot
    projection relative to pelvis.  The neighbour test requires:
        proj[i] >= proj[i-1]  AND  proj[i] >  proj[i+1]
        proj[i] >  proj[i-2]  AND  proj[i] >= proj[i+2]
    (handles plateaus on the leading edge, breaks ties on the trailing edge).

    Args:
        proj: (T,) signal from :func:`anterior_projection`.

    Returns:
        Sorted list of integer indices.
    """
    proj = np.asarray(proj, dtype=float)
    n = len(proj)
    return [
        i for i in range(2, n - 2)
        if (
            proj[i] >= proj[i - 1]
            and proj[i] >  proj[i + 1]
            and proj[i] >  proj[i - 2]
            and proj[i] >= proj[i + 2]
        )
    ]


def fold_cycles(
    signal: np.ndarray,
    strikes: list[int],
    n_points: int = 101,
    min_len: int = 10,
) -> np.ndarray:
    """Fold a multi-column signal onto a fixed gait-cycle grid.

    For each consecutive pair of heel-strike indices ``(a, b)`` with
    ``b - a >= min_len``, the slice ``signal[a:b]`` is linearly resampled to
    ``n_points`` samples spanning 0→100% of the gait cycle.  The result is
    averaged across all valid cycles.

    Args:
        signal:   (T, ncol) array — e.g. per-muscle activation columns.
        strikes:  sorted heel-strike indices from :func:`detect_heel_strikes`.
        n_points: output grid length (default 101 → 0%, 1%, …, 100%).
        min_len:  minimum cycle duration in samples; shorter pairs are skipped.

    Returns:
        (n_points, ncol) mean gait-cycle activation.  Returns zeros if no
        valid cycles are found.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    ncol = signal.shape[1]

    grid = np.linspace(0, 100, n_points)
    cycles: list[np.ndarray] = []

    for a, b in zip(strikes[:-1], strikes[1:]):
        if b - a < min_len:
            continue
        seg = signal[a:b]                                  # (seg_len, ncol)
        x = np.linspace(0, 100, len(seg))
        resampled = np.column_stack(
            [np.interp(grid, x, seg[:, j]) for j in range(ncol)]
        )
        cycles.append(resampled)

    if not cycles:
        return np.zeros((n_points, ncol))
    return np.mean(cycles, axis=0)
