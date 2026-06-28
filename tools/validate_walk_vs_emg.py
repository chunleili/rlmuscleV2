"""Compare rl_mass humanoid walk activations vs real adult sEMG (Moreira 2021).

Rolls out the walk policy, phase-anchors on right heel strike (Zeni 2008),
folds activations onto the 0–100% gait cycle, and compares to the Moreira
EMG envelopes for 4 right-leg muscles.

Config JSON keys:
    walk_config  : path to the MassEnv walk config JSON
    checkpoint   : checkpoint prefix without extension
                   (e.g. "data/rl_mass/checkpoints/walk_max")
    emg_dir      : directory containing T*.txt EMG envelope files
    muscle_map   : {EMG_label: SNU_muscle_name} mapping
    out_dir      : directory for outputs (metrics.json, compare.png)
    max_steps    : optional rollout cap (default 420)

Usage:
    uv run python tools/validate_walk_vs_emg.py --config data/rl_mass/configs/walk_vs_emg_validate.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure repo root on sys.path whether run directly or imported by a test.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import Tensor

from rl_mass.env import MassEnv
from rl_mass.model import MuscleNN, SimulationNN
from rl_mass.finger.gt import (
    EMGLoader,
    anterior_projection,
    detect_heel_strikes,
    fold_cycles,
    pearson_r,
    timing_lag,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _device_of(net):
    try:
        return next(net.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _peak_norm(y):
    m = float(np.max(y))
    return y / m if m > 1e-9 else y


# ──────────────────────────────────────────────────────────────────────────────
# Main compare function
# ──────────────────────────────────────────────────────────────────────────────

def compare(config_path: str) -> dict:
    """Run the walk-vs-EMG validation.

    Args:
        config_path: path to a JSON config file (see module docstring).

    Returns:
        dict with keys ``per_muscle``, ``aggregate``, ``out_dir``.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    walk_config  = cfg["walk_config"]
    chk_prefix   = cfg["checkpoint"]           # e.g. ".../walk_max"
    emg_dir      = cfg["emg_dir"]
    muscle_map   = cfg["muscle_map"]           # {EMG_label: SNU_name}
    out_dir      = cfg["out_dir"]
    max_steps    = int(cfg.get("max_steps", 420))
    n_points     = 101

    # ── Environment ────────────────────────────────────────────────────────────
    env = MassEnv(walk_config)
    import mujoco

    # ── Networks ───────────────────────────────────────────────────────────────
    model = SimulationNN(env.num_state, env.num_action)
    mm    = MuscleNN(env.num_total_related_dofs, env.num_action, env.num_muscles)
    model.load(chk_prefix + ".pt")
    mm.load(chk_prefix + "_muscle.pt")
    model.eval(); mm.eval()

    # Each net may be on a different device (MuscleNN self-cudas in __init__).
    sdev = _device_of(model)
    mdev = _device_of(mm)

    # ── Body ids ───────────────────────────────────────────────────────────────
    m_model = env.model
    pid = mujoco.mj_name2id(m_model, mujoco.mjtObj.mjOBJ_BODY, "Pelvis")
    fid = mujoco.mj_name2id(m_model, mujoco.mjtObj.mjOBJ_BODY, "TalusR")
    if pid < 0:
        raise RuntimeError("Body 'Pelvis' not found in model")
    if fid < 0:
        raise RuntimeError("Body 'TalusR' not found in model")

    # ── Muscle index lookup ────────────────────────────────────────────────────
    meta  = env.get_muscle_metadata()
    names = list(meta["names"])                            # SNU 284-muscle list
    try:
        midx = {emg_k: names.index(snu_v) for emg_k, snu_v in muscle_map.items()}
    except ValueError as e:
        raise ValueError(f"Muscle not found in env metadata: {e}") from e

    # ── Rollout ────────────────────────────────────────────────────────────────
    state = env.reset(rsi=False)
    mujoco.mj_forward(m_model, env.data)
    nspc = env.simulation_hz // env.control_hz

    A: list[np.ndarray]    = []
    pelv: list[np.ndarray] = []
    foot: list[np.ndarray] = []

    for _ in range(max_steps):
        s  = Tensor(state.reshape(1, -1).astype(np.float32)).to(sdev)
        a_dist, _ = model(s)
        action = a_dist.loc.cpu().detach().numpy().squeeze()
        env.set_action(action)

        mt = Tensor(env.get_muscle_torques().reshape(1, -1).astype(np.float32)).to(mdev)
        for _ in range(nspc // 2):
            dt_t = Tensor(env.get_desired_torques().reshape(1, -1).astype(np.float32)).to(mdev)
            act  = mm(mt, dt_t).cpu().detach().numpy().squeeze()
            env.set_activation_levels(act)
            env.step()
            env.step()

        A.append(env.get_activation_levels().copy())
        pelv.append(env.data.xpos[pid].copy())
        foot.append(env.data.xpos[fid].copy())

        if env.is_end_of_episode():
            break
        state = env.get_state()

    A    = np.array(A)     # (T, num_muscles)
    pelv = np.array(pelv)  # (T, 3)
    foot = np.array(foot)  # (T, 3)

    # ── Heel-strike phase anchoring ────────────────────────────────────────────
    proj   = anterior_projection(foot, pelv, up_axis=1)
    strikes = detect_heel_strikes(proj)

    if len(strikes) < 2:
        raise RuntimeError(
            f"Only {len(strikes)} heel strike(s) detected — "
            "rollout too short or character fell early."
        )

    # Extract activation columns for the mapped muscles (in EMG label order)
    emg_labels = list(muscle_map.keys())
    act_cols   = A[:, [midx[k] for k in emg_labels]]      # (T, 4)
    rl_cycles  = fold_cycles(act_cols, strikes, n_points=n_points)  # (101, 4)

    # ── EMG ground truth ───────────────────────────────────────────────────────
    gt = EMGLoader(emg_dir, muscles=emg_labels, n_points=n_points).load()
    # gt.activation : (101, 4), raw μV

    # ── Metrics ────────────────────────────────────────────────────────────────
    # timing_lag dt = one gait-cycle percent step
    dt_pct = 100.0 / (n_points - 1)

    per_muscle: dict[str, dict] = {}
    for j, emg_k in enumerate(emg_labels):
        gt_env = gt.activation[:, j]
        rl_env = rl_cycles[:, j]
        per_muscle[emg_k] = {
            "pearson":    pearson_r(gt_env, rl_env),
            "timing_lag": timing_lag(gt_env, rl_env, dt_pct),
        }

    aggregate = {
        "mean_pearson": float(np.mean([v["pearson"] for v in per_muscle.values()])),
    }

    # ── Outputs ────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"per_muscle": per_muscle, "aggregate": aggregate}, f, indent=2)

    # Overlay plot — peak-normalized for shape comparison
    labels = {
        "VL":  "Vastus Lateralis",
        "BF":  "Biceps Femoris (hamstring)",
        "TA":  "Tibialis Anterior",
        "GAL": "Gastrocnemius Lat",
    }
    grid = np.linspace(0, 100, n_points)
    n_musc = len(emg_labels)
    fig, axes = plt.subplots(n_musc, 1, figsize=(8, 2.2 * n_musc), sharex=True)
    axes = np.atleast_1d(axes)
    for j, emg_k in enumerate(emg_labels):
        gt_env = _peak_norm(gt.activation[:, j])
        rl_env = _peak_norm(rl_cycles[:, j])
        axes[j].plot(grid, gt_env, "k", lw=1.5,
                     label="real EMG (Moreira 2021, P8 male 1.72 m, speed V4)")
        axes[j].fill_between(grid, 0, gt_env, color="0.7", alpha=0.4)
        axes[j].plot(grid, rl_env, color="tab:red", lw=1.5, label="rl_mass walk")
        axes[j].set_ylabel(labels.get(emg_k, emg_k), fontsize=8)
        axes[j].set_ylim(0, 1.12)
        if j == 0:
            axes[j].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("% gait cycle  (0 = right heel strike, Zeni-anchored)")
    fig.suptitle(
        f"rl_mass walk vs sEMG (Moreira 2021) — peak-normalised  "
        f"mean_r={aggregate['mean_pearson']:.3f}\n"
        f"confound: gait cadence NOT speed-matched (rl_mass walk.bvh vs Moreira V4); "
        f"shape/timing only, magnitude not comparable",
        fontsize=9,
    )
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "compare.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    return {"per_muscle": per_muscle, "aggregate": aggregate, "out_dir": out_dir}


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Validate walk vs Moreira 2021 sEMG")
    p.add_argument("--config", "-c", required=True,
                   help="Path to validate_walk_vs_emg JSON config")
    a = p.parse_args()
    res = compare(a.config)
    print(json.dumps({"aggregate": res["aggregate"], "per_muscle": res["per_muscle"]},
                     indent=2))


if __name__ == "__main__":
    main()
