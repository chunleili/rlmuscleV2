"""validate_vs_gt end-to-end on Arm26 with random-init nets (smoke).

Uses untrained nets (no checkpoint dependency) by pointing at a config whose
checkpoint is the empty string -> harness builds fresh nets. Asserts the three
output files exist and metrics dict has the right shape."""
import os, sys, json, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from tools.validate_vs_gt import compare

TRAIN_CFG = "data/rl_mass_arm26/config.json"
STO = ("/home/chunleli/Dev/OpenSimExample/Models/Arm26/OutputReference/"
       "StaticOptimization/PhysiologicalResults/arm26_StaticOptimization_activation.sto")

def main():
    d = tempfile.mkdtemp()
    cfg = os.path.join(d, "val.json")
    out = os.path.join(d, "out")
    json.dump({
        "train_config": TRAIN_CFG,
        "checkpoint": "",          # empty -> random-init nets (smoke)
        "gt_sto": STO,
        "out_dir": out,
        "max_steps": 30,
    }, open(cfg, "w"))

    res = compare(cfg)
    assert set(res["per_muscle"]) == {"TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"}
    for m, d_ in res["per_muscle"].items():
        assert set(d_) == {"pearson", "rmse", "r2", "timing_lag"}, (m, d_)
    agg = res["aggregate"]
    assert {"mean_pearson", "mean_rmse", "mean_r2", "cci_rl", "cci_gt"} <= set(agg)
    assert os.path.exists(os.path.join(out, "metrics.json"))
    assert os.path.exists(os.path.join(out, "compare.png"))
    assert os.path.exists(os.path.join(out, "compare.csv"))
    print("OK")

if __name__ == "__main__":
    main()
