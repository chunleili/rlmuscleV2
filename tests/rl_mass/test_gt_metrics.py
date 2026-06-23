"""Metrics on known inputs."""
import numpy as np
from rl_mass.finger.gt.metrics import pearson_r, rmse, r2, timing_lag, co_contraction_index

def main():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    assert abs(pearson_r(x, 2 * x + 1) - 1.0) < 1e-9
    assert abs(pearson_r(x, -x) + 1.0) < 1e-9
    assert abs(rmse(x, x)) < 1e-12
    assert abs(rmse(np.zeros(3), np.ones(3)) - 1.0) < 1e-12
    assert abs(r2(x, x) - 1.0) < 1e-9
    assert r2(x, np.full_like(x, x.mean())) < 1e-9 + 0.0  # predicting the mean -> R2 ~ 0

    # timing: y is x delayed by 2 samples, dt=0.1 -> lag ~ +0.2 s
    base = np.sin(np.linspace(0, 6.28, 50))
    y = np.roll(base, 2); y[:2] = 0.0
    lag = timing_lag(base, y, dt=0.1)
    assert abs(lag - 0.2) < 0.05, lag

    # co-contraction: identical groups -> 1.0 ; disjoint on/off -> 0.0
    a = np.array([0.2, 0.4, 0.6])
    assert abs(co_contraction_index(a, a) - 1.0) < 1e-9
    on = np.array([1.0, 0.0]); off = np.array([0.0, 1.0])
    assert abs(co_contraction_index(on, off)) < 1e-9
    print("OK")

if __name__ == "__main__":
    main()
