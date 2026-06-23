"""Ground-truth activation loaders. The comparison core depends only on the
GTSeries shape; each GT source (OpenSim now, EMG later) is one GTLoader."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

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
