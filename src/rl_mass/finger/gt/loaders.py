"""Ground-truth activation loaders. The comparison core depends only on the
GTSeries shape; each GT source (OpenSim now, EMG later) is one GTLoader."""
from __future__ import annotations

import glob as _glob
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Sequence

import numpy as np


class GTSeries(NamedTuple):
    names: list[str]
    time: np.ndarray        # (T,)
    activation: np.ndarray  # (T, M)


class GTLoader(ABC):
    @abstractmethod
    def load(self) -> GTSeries:
        ...


def _read_storage(path):
    """Parse an OpenSim Storage (.sto/.mot). Returns (col_names, matrix(T,ncols))."""
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    for i, ln in enumerate(lines):
        if ln.strip().lower() == "endheader":
            break
    else:
        raise ValueError(f"no 'endheader' marker in {path}")
    header = lines[i + 1].split("\t")
    rows = [r for r in lines[i + 2:] if r.strip()]
    mat = np.array([[float(x) for x in r.split("\t")] for r in rows], dtype=np.float64)
    return header, mat


class OpenSimStoLoader(GTLoader):
    def __init__(self, sto_path: str, muscle_columns: Optional[list[str]] = None):
        self.sto_path = sto_path
        self.muscle_columns = muscle_columns

    def load(self) -> GTSeries:
        header, mat = _read_storage(self.sto_path)
        time = mat[:, header.index("time")]
        available = [c for c in header if c != "time"]
        cols = self.muscle_columns or available
        missing = [c for c in cols if c not in header]
        if missing:
            raise ValueError(
                f"GT columns not found: {missing}. Available: {available}")
        idx = [header.index(c) for c in cols]
        return GTSeries(names=list(cols), time=time, activation=mat[:, idx])


class EMGLoader(GTLoader):
    """Load Moreira-2021-style sEMG envelope files into a GTSeries.

    Each ``T*.txt`` file under *emg_dir* contains a tab-separated header row
    followed by ~1001 rows (one full gait cycle, 0→100%).  The header has the
    form ``St1_VL St2_VL St1_BF St2_BF St1_TA St2_TA St1_GAL St2_GAL``
    (stride × muscle).  Every column whose suffix matches a requested muscle
    name is treated as one independent stride envelope.

    Returns:
        GTSeries with:
          - ``names``: list of the requested muscle labels (e.g. ``["VL","BF","TA","GAL"]``)
          - ``time``:  ``np.linspace(0, 100, n_points)``  (% gait cycle)
          - ``activation``: ``(n_points, len(muscles))`` mean envelope in raw μV.
            Do **not** peak-normalise here — callers that need shape comparison
            can normalise themselves.

    Raises:
        ValueError: if *emg_dir* contains no ``T*.txt`` files, or if any
            requested muscle name is absent from all file headers.
    """

    def __init__(
        self,
        emg_dir: str,
        muscles: Sequence[str] = ("VL", "BF", "TA", "GAL"),
        n_points: int = 101,
    ):
        self.emg_dir = emg_dir
        self.muscles = list(muscles)
        self.n_points = n_points

    def load(self) -> GTSeries:
        files = sorted(_glob.glob(f"{self.emg_dir}/T*.txt"))
        if not files:
            raise ValueError(
                f"EMGLoader: no T*.txt files found in '{self.emg_dir}'"
            )

        grid = np.linspace(0, 100, self.n_points)

        # Accumulate stride envelopes per requested muscle
        acc: dict[str, list[np.ndarray]] = {m: [] for m in self.muscles}
        found: set[str] = set()

        for fpath in files:
            with open(fpath) as fh:
                hdr = fh.readline().split()
            data = np.loadtxt(fpath, skiprows=1)

            for j, col in enumerate(hdr):
                parts = col.split("_", 1)
                if len(parts) != 2:
                    continue
                _stride_tag, muscle = parts
                if muscle not in acc:
                    continue
                found.add(muscle)
                # Resample this stride to n_points over 0–100%
                x = np.linspace(0, 100, len(data))
                acc[muscle].append(np.interp(grid, x, data[:, j]))

        missing = [m for m in self.muscles if m not in found]
        if missing:
            raise ValueError(
                f"EMGLoader: muscle(s) {missing} not found in headers of "
                f"'{self.emg_dir}/T*.txt'. "
                f"Available suffixes: {sorted(found)}"
            )

        # Mean over all strides (n_trials × n_strides_per_trial)
        activation = np.column_stack(
            [np.mean(acc[m], axis=0) for m in self.muscles]
        )  # (n_points, M)

        return GTSeries(
            names=list(self.muscles),
            time=grid,
            activation=activation,
        )
