"""Align a learned-activation series to a GT series: column name-map + time
resample. No knowledge of the GT source — pure array/time operations."""
from __future__ import annotations

from typing import Optional

import numpy as np


class MuscleNameMap:
    def __init__(self, mapping: Optional[dict] = None):
        self.mapping = mapping or {}

    def reorder(self, src_names, dst_names, values):
        """Return values reordered so columns match dst_names.

        src_names label the columns of `values`; each is first translated
        through the mapping, then matched to dst_names.
        """
        translated = [self.mapping.get(n, n) for n in src_names]
        pos = {n: i for i, n in enumerate(translated)}
        idx = [pos[d] for d in dst_names]  # raises KeyError if a dst muscle is absent
        return values[:, idx]


def resample_to_common_grid(t_a, a, t_b, b):
    """Linearly interpolate (t_a, a) and (t_b, b) onto a shared time grid.

    Grid spans the overlap [max(start), min(end)] sampled at the finer of the
    two input rates. a, b are (T, M) with the SAME M and column order.
    """
    t0 = max(float(t_a[0]), float(t_b[0]))
    t1 = min(float(t_a[-1]), float(t_b[-1]))
    da = (t_a[-1] - t_a[0]) / max(len(t_a) - 1, 1)
    db = (t_b[-1] - t_b[0]) / max(len(t_b) - 1, 1)
    dt = min(da, db)
    if dt > 0:
        n = int(np.floor((t1 - t0) / dt)) + 1
        tc = t0 + dt * np.arange(n)
    else:
        # Degenerate (single-sample) input: a single overlap point at t0.
        tc = np.array([t0])

    def interp(t, v):
        return np.column_stack([np.interp(tc, t, v[:, j]) for j in range(v.shape[1])])

    return tc, interp(t_a, a), interp(t_b, b)
