"""Compare rl_mass learned muscle activations against a ground-truth source.

Rolls out a (trained) policy on the reference motion, loads GT activations,
aligns by muscle name + time, and writes per-muscle + aggregate metrics and an
overlay plot. The comparison core is GT-source-agnostic; only the GTLoader is
source-specific (OpenSim Static Optimization here).

Config JSON keys:
    train_config : path to the BimanualEnv training config (defines model+motion)
    checkpoint   : checkpoint prefix (e.g. output/arm26/nn/max); "" -> random nets
    gt_sto       : OpenSim Static-Optimization activation .sto
    out_dir      : directory for metrics.json / compare.png / compare.csv
    muscle_name_map : optional {rl_name: gt_name}
    max_steps    : optional rollout cap (default 10000)

Usage:
    uv run python tools/validate_vs_gt.py -c data/rl_mass_arm26/validate.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import torch

# Ensure repo root is on sys.path so `tools._rollout` resolves whether this
# file is imported by a test (which sets sys.path itself) or run directly
# as `uv run python tools/validate_vs_gt.py`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_mass.env_factory import make_env
from rl_mass.model import MuscleNN, SimulationNN
from rl_mass.finger.gt import (
    MuscleNameMap, OpenSimStoLoader, co_contraction_index,
    pearson_r, r2, resample_to_common_grid, rmse, timing_lag,
)
from tools._rollout import rollout


def _muscle_names(env):
    import mujoco
    return [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(env.num_muscles)]


def _control_times(n_frames, control_hz):
    return np.arange(n_frames) / float(control_hz)


def compare(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    env = make_env(cfg["train_config"])

    model = SimulationNN(env.num_state, env.num_action)
    mm = MuscleNN(env.num_total_related_dofs, env.num_action, env.num_muscles)
    if cfg.get("checkpoint"):
        model.load(cfg["checkpoint"] + ".pt")
        mm.load(cfg["checkpoint"] + "_muscle.pt")
    if torch.cuda.is_available():
        model.cuda(); mm.cuda()
    model.eval(); mm.eval()

    _, _, acts, _, _ = rollout(
        env, model, mm, max_steps=int(cfg.get("max_steps", 10000)), deterministic=True)
    rl_names = _muscle_names(env)
    t_rl = _control_times(acts.shape[0], env.control_hz)

    gt = OpenSimStoLoader(cfg["gt_sto"]).load()
    name_map = MuscleNameMap(cfg.get("muscle_name_map"))
    # GT columns reordered to the rl muscle order (rl name -> GT name via map).
    gt_dst = [name_map.mapping.get(n, n) for n in rl_names]
    gt_cols = name_map.reorder(gt.names, gt_dst, gt.activation)  # (T_gt, nu) in rl order

    tc, a_rl, a_gt = resample_to_common_grid(t_rl, acts, gt.time, gt_cols)
    dt = float(tc[1] - tc[0]) if len(tc) > 1 else 1.0

    per = {}
    for j, name in enumerate(rl_names):
        per[name] = {
            "pearson": pearson_r(a_gt[:, j], a_rl[:, j]),
            "rmse": rmse(a_gt[:, j], a_rl[:, j]),
            "r2": r2(a_gt[:, j], a_rl[:, j]),
            "timing_lag": timing_lag(a_gt[:, j], a_rl[:, j], dt),
        }

    with open(cfg["train_config"]) as f:
        groups = json.load(f).get("muscle_groups", {})
    flex = groups.get("flexor", []); ext = groups.get("extensor", [])
    idx = {n: i for i, n in enumerate(rl_names)}

    def _grp_sum(mat, names):
        cols = [idx[n] for n in names if n in idx]
        return mat[:, cols].sum(axis=1) if cols else np.zeros(mat.shape[0])

    cci_rl = co_contraction_index(_grp_sum(a_rl, flex), _grp_sum(a_rl, ext)) if flex and ext else 0.0
    cci_gt = co_contraction_index(_grp_sum(a_gt, flex), _grp_sum(a_gt, ext)) if flex and ext else 0.0

    agg = {
        "mean_pearson": float(np.mean([per[n]["pearson"] for n in rl_names])),
        "mean_rmse": float(np.mean([per[n]["rmse"] for n in rl_names])),
        "mean_r2": float(np.mean([per[n]["r2"] for n in rl_names])),
        "cci_rl": cci_rl,
        "cci_gt": cci_gt,
    }

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"per_muscle": per, "aggregate": agg}, f, indent=2)

    # overlay plot
    n = len(rl_names)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.6 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for j, name in enumerate(rl_names):
        axes[j].plot(tc, a_gt[:, j], label="GT (SO)", color="k", lw=1.2)
        axes[j].plot(tc, a_rl[:, j], label="rl_mass", color="tab:red", lw=1.0, alpha=0.8)
        axes[j].set_ylabel(name, fontsize=8)
        axes[j].set_ylim(-0.02, 1.02)
        if j == 0:
            axes[j].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"activation vs GT  mean_r={agg['mean_pearson']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare.png"), dpi=120)
    plt.close(fig)

    # aligned CSV
    header = ["time"] + [f"{n}_gt" for n in rl_names] + [f"{n}_rl" for n in rl_names]
    rows = np.column_stack([tc, a_gt, a_rl])
    np.savetxt(os.path.join(out_dir, "compare.csv"), rows,
               delimiter=",", header=",".join(header), comments="")

    return {"per_muscle": per, "aggregate": agg, "out_dir": out_dir}


def main():
    p = argparse.ArgumentParser(description="Validate rl_mass activations vs GT")
    p.add_argument("-c", "--config", required=True)
    a = p.parse_args()
    res = compare(a.config)
    print(json.dumps(res["aggregate"], indent=2))


if __name__ == "__main__":
    main()
