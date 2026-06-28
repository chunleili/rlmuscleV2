"""Smoke test: validate_walk_vs_emg.compare() with max_steps=150.

Loads the real walk checkpoint + MassEnv + Moreira EMG data.
Checks that outputs exist and have the right structure.
"""
import json
import os
import sys
import tempfile

# Repo root on sys.path so `from tools.X` resolves when run with uv run python.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tools.validate_walk_vs_emg import compare


WALK_CONFIG = "data/rl_mass/configs/walk.json"
CHECKPOINT  = "data/rl_mass/checkpoints/walk_max"
EMG_DIR     = "data/rl_mass/emg_moreira/P8_V4_R"
MUSCLE_MAP  = {
    "VL":  "R_Vastus_Lateralis",
    "BF":  "R_Bicep_Femoris_Longus",
    "TA":  "R_Tibialis_Anterior",
    "GAL": "R_Gastrocnemius_Lateral_Head",
}


def main():
    with tempfile.TemporaryDirectory() as tmp_out:
        cfg_path = os.path.join(tmp_out, "val_cfg.json")
        out_dir  = os.path.join(tmp_out, "out")
        cfg = {
            "walk_config": WALK_CONFIG,
            "checkpoint":  CHECKPOINT,
            "emg_dir":     EMG_DIR,
            "muscle_map":  MUSCLE_MAP,
            "out_dir":     out_dir,
            "max_steps":   150,
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        res = compare(cfg_path)

        # ── Output files ───────────────────────────────────────────────────────
        assert os.path.exists(os.path.join(out_dir, "metrics.json")), \
            "metrics.json missing"
        assert os.path.exists(os.path.join(out_dir, "compare.png")), \
            "compare.png missing"

        # ── metrics.json structure ─────────────────────────────────────────────
        with open(os.path.join(out_dir, "metrics.json")) as f:
            saved = json.load(f)

        assert set(saved["per_muscle"]) == set(MUSCLE_MAP), (
            f"per_muscle keys {set(saved['per_muscle'])} != {set(MUSCLE_MAP)}"
        )
        for m, d in saved["per_muscle"].items():
            assert "pearson"    in d, f"{m}: missing 'pearson'"
            assert "timing_lag" in d, f"{m}: missing 'timing_lag'"

        assert "mean_pearson" in saved["aggregate"], "aggregate missing mean_pearson"

        # ── Return dict mirrors saved metrics ─────────────────────────────────
        assert res["per_muscle"] == saved["per_muscle"]
        assert res["aggregate"]  == saved["aggregate"]

    print("OK")


if __name__ == "__main__":
    main()
