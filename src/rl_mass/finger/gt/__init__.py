"""GT-source-agnostic activation validation: loaders, alignment, metrics."""
from rl_mass.finger.gt.loaders import (
    GTLoader, GTSeries, OpenSimStoLoader, EMGLoader, EmgCohortLoader,
)
from rl_mass.finger.gt.align import MuscleNameMap, resample_to_common_grid
from rl_mass.finger.gt.metrics import (
    co_contraction_index, pearson_r, r2, rmse, timing_lag,
)
from rl_mass.finger.gt.phase import (
    anterior_projection, detect_heel_strikes, fold_cycles,
)

__all__ = [
    "GTLoader", "GTSeries", "OpenSimStoLoader", "EMGLoader", "EmgCohortLoader",
    "MuscleNameMap", "resample_to_common_grid",
    "pearson_r", "rmse", "r2", "timing_lag", "co_contraction_index",
    "anterior_projection", "detect_heel_strikes", "fold_cycles",
]
