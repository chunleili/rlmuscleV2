"""MuscleNameMap reorders columns; resample lands both series on one grid."""
import numpy as np
from rl_mass.finger.gt import MuscleNameMap, resample_to_common_grid

def main():
    nm = MuscleNameMap()  # identity
    vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (T=2, 3)
    out = nm.reorder(["A", "B", "C"], ["C", "A"], vals)
    assert out.shape == (2, 2)
    assert np.allclose(out[:, 0], [3.0, 6.0]) and np.allclose(out[:, 1], [1.0, 4.0])

    nm2 = MuscleNameMap({"rl_bic": "BIClong"})
    out2 = nm2.reorder(["rl_bic"], ["BIClong"], np.array([[0.5], [0.7]]))
    assert np.allclose(out2[:, 0], [0.5, 0.7])

    # resample: a at 0..1 step .5 ; b at 0..1 step .25 -> common grid = finer (.25)
    ta = np.array([0.0, 0.5, 1.0]); a = np.array([[0.0], [1.0], [2.0]])
    tb = np.array([0.0, 0.25, 0.5, 0.75, 1.0]); b = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    tc, ac, bc = resample_to_common_grid(ta, a, tb, b)
    assert tc[0] == 0.0 and abs(tc[-1] - 1.0) < 1e-12
    assert ac.shape[0] == bc.shape[0] == tc.shape[0]
    # a is linear 0->2 over [0,1]; at t=0.25 expect 0.5
    j = np.argmin(np.abs(tc - 0.25))
    assert abs(ac[j, 0] - 0.5) < 1e-9, ac[j, 0]

    # single-sample input must not crash (dt==0 guard)
    t1s = np.array([0.5]); s1 = np.array([[1.0]])
    t2s = np.array([0.5]); s2 = np.array([[2.0]])
    tcs, a1s, a2s = resample_to_common_grid(t1s, s1, t2s, s2)
    assert tcs.shape == (1,) and a1s.shape == (1, 1) and a2s.shape == (1, 1)
    assert abs(a1s[0, 0] - 1.0) < 1e-9 and abs(a2s[0, 0] - 2.0) < 1e-9
    print("OK")

if __name__ == "__main__":
    main()
