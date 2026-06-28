"""EmgCohortLoader: aggregate per-subject sEMG into a mean ± across-subject SD.

Tests:
  - Synthetic 2-subject cohort → exact cohort mean and SD.
  - Real V4_R cohort → 13 subjects, (101,4) mean + non-trivial SD band.
  - Root with no subject sub-dirs raises ValueError.
"""
import os
import tempfile

import numpy as np

from rl_mass.finger.gt import EmgCohortLoader


_HEADER = "St1_VL\tSt2_VL\tSt1_BF\tSt2_BF\tSt1_TA\tSt2_TA\tSt1_GAL\tSt2_GAL"


def _write_subject(root, name, value, n_rows=5):
    """One subject dir with a single T1.txt whose every column == value."""
    d = os.path.join(root, name)
    os.makedirs(d)
    rows = "\n".join("\t".join([f"{value}"] * 8) for _ in range(n_rows))
    with open(os.path.join(d, "T1.txt"), "w") as f:
        f.write(_HEADER + "\n" + rows + "\n")


def test_synthetic_mean_and_std():
    with tempfile.TemporaryDirectory() as tmp:
        _write_subject(tmp, "Participant1", 1.0)
        _write_subject(tmp, "Participant2", 3.0)
        gt = EmgCohortLoader(tmp).load()

        assert gt.names == ["VL", "BF", "TA", "GAL"], gt.names
        assert gt.activation.shape == (101, 4), gt.activation.shape
        assert gt.std is not None, "cohort loader must populate std"
        assert gt.std.shape == (101, 4), gt.std.shape
        # subject means are 1.0 and 3.0 -> cohort mean 2.0, population SD 1.0
        assert np.allclose(gt.activation, 2.0), gt.activation.mean()
        assert np.allclose(gt.std, 1.0), gt.std.mean()


def test_real_cohort():
    gt = EmgCohortLoader("data/rl_mass/emg_moreira/V4_R").load()
    assert gt.activation.shape == (101, 4), gt.activation.shape
    assert gt.std is not None and gt.std.shape == (101, 4)
    assert float(gt.std.max()) > 0.0, "across-subject SD should be non-zero"
    assert float(gt.activation.min()) >= 0.0


def test_no_subjects_raises():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            EmgCohortLoader(tmp).load()
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "subject" in str(e).lower(), str(e)


def main():
    test_synthetic_mean_and_std()
    test_real_cohort()
    test_no_subjects_raises()
    print("OK")


if __name__ == "__main__":
    main()
