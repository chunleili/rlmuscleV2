"""Compare rl_mass humanoid walk *kinematics* vs real adult gait (Moreira 2021).

This is the **fairness gate** for the EMG comparison: activations can only be
compared muscle-to-muscle if the underlying motion is the same gait. If the
rl_mass joint trajectories (hip/knee/ankle sagittal flexion) do not match the
Moreira cohort, then any activation mismatch is a *motion* confound, not a
muscle-redundancy result.

Rolls out the walk policy, phase-anchors on right heel strike (Zeni 2008,
reusing the EMG tool's anchoring), folds the right-leg sagittal joint angles
onto the 0-100% gait cycle, and overlays them on the Moreira cohort band.

Moreira angle channels (deg): St{1,2}_{Hip,Knee,Ankle}_X are the sagittal
plane (verified by range: Knee_X ~0->65, Hip_X ~-15->+17, Ankle_X ~3->20).

Config JSON keys:
    walk_config   : path to the MassEnv walk config JSON
    checkpoint    : checkpoint prefix without extension
    angles_cohort_dir : Moreira raw zip OR a dir; one Participant* per subject
    out_dir       : directory for outputs (metrics.json, compare.png, compare.csv)
    max_steps     : optional rollout cap (default 420)

Usage:
    uv run python tools/validate_walk_kinematics.py -c data/rl_mass/configs/walk_kinematics_validate.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

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
    anterior_projection,
    detect_heel_strikes,
    fold_cycles,
    pearson_r,
)

# Right-leg joints we compare, in display order. Each maps the SNU joint name
# to the Moreira sagittal channel prefix.
JOINTS = [
    ("Hip",   "j_FemurR", "Hip"),
    ("Knee",  "j_TibiaR", "Knee"),
    ("Ankle", "j_TalusR", "Ankle"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Angle helpers
# ──────────────────────────────────────────────────────────────────────────────

def _quat_flexion_deg(q: np.ndarray) -> float:
    """Sagittal flexion (rotation about model X) from a MuJoCo ball-joint quat.

    MuJoCo stores quats as (w, x, y, z). Flexion in the sagittal plane is the
    roll about the body X axis (the knee hinge axis is also ``1 0 0``), so we
    extract the X-Euler component, which is the dominant rotation in gait.
    """
    w, x, y, z = q
    return float(np.degrees(np.arctan2(2.0 * (w * x + y * z),
                                       1.0 - 2.0 * (x * x + y * y))))


def _device_of(net):
    try:
        return next(net.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Moreira cohort kinematics loader
# ──────────────────────────────────────────────────────────────────────────────

def _resample(col: np.ndarray, n: int) -> np.ndarray:
    x = np.linspace(0, 100, len(col))
    return np.interp(np.linspace(0, 100, n), x, col)


def load_angles_cohort(src: str, n_points: int = 101):
    """Load Moreira sagittal joint angles → (mean, std) each (n_points, 3).

    ``src`` may be a directory of ``Participant*/T*.txt`` files, or a path to
    the raw ``moreira_ascii.zip`` (in which case V4/R Angles are read directly).
    Per subject: average both strides (St1, St2) across all trials. Then stack
    subjects → across-subject mean and std. Column order = Hip, Knee, Ankle.
    """
    chans = [j[2] for j in JOINTS]               # ["Hip","Knee","Ankle"]

    def parse(text: str) -> np.ndarray:
        lines = text.strip().split("\n")
        hdr = lines[0].split("\t")
        data = np.array([[float(v) for v in ln.split("\t")] for ln in lines[1:]])
        # average St1 and St2 sagittal-X for each joint
        out = []
        for ch in chans:
            cols = [i for i, h in enumerate(hdr) if h in (f"St1_{ch}_X", f"St2_{ch}_X")]
            out.append(_resample(data[:, cols].mean(axis=1), n_points))
        return np.column_stack(out)               # (n_points, 3)

    per_subject: list[np.ndarray] = []
    if zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as zf:
            names = zf.namelist()
            subs: dict[str, list[str]] = {}
            for nm in names:
                # .../<Participant>/Processed_Data/V4/R/Angles/T*.txt
                if "/Processed_Data/V4/R/Angles/" in nm and nm.endswith(".txt"):
                    pid = nm.split("/Processed_Data/")[0].split("/")[-1]
                    subs.setdefault(pid, []).append(nm)
            for pid in sorted(subs):
                trials = [parse(zf.read(nm).decode()) for nm in sorted(subs[pid])]
                if trials:
                    per_subject.append(np.mean(trials, axis=0))
    else:
        for pid in sorted(os.listdir(src)):
            pdir = os.path.join(src, pid)
            if not os.path.isdir(pdir):
                continue
            trials = [parse(open(os.path.join(pdir, f)).read())
                      for f in sorted(os.listdir(pdir)) if f.endswith(".txt")]
            if trials:
                per_subject.append(np.mean(trials, axis=0))

    if not per_subject:
        raise ValueError(f"No Moreira angle subjects found in {src}")
    stack = np.stack(per_subject)                 # (S, n_points, 3)
    return stack.mean(axis=0), stack.std(axis=0), len(per_subject)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def compare(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)

    walk_config  = cfg["walk_config"]
    chk_prefix   = cfg["checkpoint"]
    cohort_src   = cfg["angles_cohort_dir"]
    out_dir      = cfg["out_dir"]
    max_steps    = int(cfg.get("max_steps", 420))
    n_points     = 101

    env = MassEnv(walk_config)
    import mujoco
    m_model = env.model

    model = SimulationNN(env.num_state, env.num_action)
    mm    = MuscleNN(env.num_total_related_dofs, env.num_action, env.num_muscles)
    model.load(chk_prefix + ".pt")
    mm.load(chk_prefix + "_muscle.pt")
    model.eval(); mm.eval()
    sdev = _device_of(model); mdev = _device_of(mm)

    pid = mujoco.mj_name2id(m_model, mujoco.mjtObj.mjOBJ_BODY, "Pelvis")
    fid = mujoco.mj_name2id(m_model, mujoco.mjtObj.mjOBJ_BODY, "TalusR")

    # qpos addresses for the three right-leg joints
    qadr = {}
    for _, jname, _ in JOINTS:
        jid = mujoco.mj_name2id(m_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise RuntimeError(f"Joint {jname} not found")
        qadr[jname] = (m_model.jnt_qposadr[jid], int(m_model.jnt_type[jid]))

    state = env.reset(rsi=False)
    mujoco.mj_forward(m_model, env.data)
    nspc = env.simulation_hz // env.control_hz

    def _angles_from_qpos(qp):
        """Extract [hip, knee, ankle] sagittal flexion (deg) from a qpos vector."""
        row = []
        for _, jname, _ in JOINTS:
            adr, jtype = qadr[jname]
            if jtype == mujoco.mjtJoint.mjJNT_HINGE:
                row.append(float(np.degrees(qp[adr])))
            else:
                row.append(_quat_flexion_deg(qp[adr:adr + 4]))
        return row

    ang: list[list[float]] = []   # actual policy rollout
    ref: list[list[float]] = []   # BVH reference target the policy chases
    pelv: list[np.ndarray] = []
    foot: list[np.ndarray] = []

    for _ in range(max_steps):
        s = Tensor(state.reshape(1, -1).astype(np.float32)).to(sdev)
        a_dist, _ = model(s)
        action = a_dist.loc.cpu().detach().numpy().squeeze()
        env.set_action(action)
        mt = Tensor(env.get_muscle_torques().reshape(1, -1).astype(np.float32)).to(mdev)
        for _ in range(nspc // 2):
            dt = Tensor(env.get_desired_torques().reshape(1, -1).astype(np.float32)).to(mdev)
            act = mm(mt, dt).cpu().detach().numpy().squeeze()
            env.set_activation_levels(act)
            env.step(); env.step()

        ang.append(_angles_from_qpos(env.data.qpos))
        ref.append(_angles_from_qpos(env.target_positions_mj))
        pelv.append(env.data.xpos[pid].copy())
        foot.append(env.data.xpos[fid].copy())

        if env.is_end_of_episode():
            break
        state = env.get_state()

    ang  = np.array(ang)    # (T, 3)
    ref  = np.array(ref)    # (T, 3) BVH reference target
    pelv = np.array(pelv)
    foot = np.array(foot)

    strikes = detect_heel_strikes(anterior_projection(foot, pelv, up_axis=1))
    if len(strikes) < 2:
        raise RuntimeError(f"Only {len(strikes)} heel strike(s) detected")
    rl_cycles  = fold_cycles(ang, strikes, n_points=n_points)    # (101, 3)
    ref_cycles = fold_cycles(ref, strikes, n_points=n_points)    # (101, 3)

    gt_mean, gt_std, n_subj = load_angles_cohort(cohort_src, n_points)

    # ── Metrics ──────────────────────────────────────────────────────────────
    grid = np.linspace(0, 100, n_points)
    per_joint = {}
    for j, (disp, _, _) in enumerate(JOINTS):
        gt = gt_mean[:, j]; rl = rl_cycles[:, j]; rf = ref_cycles[:, j]
        r      = pearson_r(gt, rl)    # GT vs actual policy rollout
        r_ref  = pearson_r(gt, rf)    # GT vs BVH reference  → reference quality
        r_track = pearson_r(rf, rl)   # reference vs rollout → tracking quality
        # Sign-align: zero/sign conventions differ between SNU and Moreira;
        # gait *shape* match is what matters, so report |r| with sign flag and
        # peak-flexion timing/value to judge whether it is the same gait.
        per_joint[disp] = {
            "pearson":        r,
            "abs_pearson":    abs(r),
            "sign_flipped":   bool(r < 0),
            "ref_vs_gt_absr":   abs(r_ref),    # is the BVH reference itself real gait?
            "rollout_vs_ref_absr": abs(r_track),  # does the policy track the reference?
            "gt_range_deg":   float(gt.max() - gt.min()),
            "ref_range_deg":  float(rf.max() - rf.min()),
            "rl_range_deg":   float(rl.max() - rl.min()),
            "gt_peak_pct":    float(grid[int(np.argmax(gt))]),
            "rl_peak_pct":    float(grid[int(np.argmax(rl if r >= 0 else -rl))]),
        }
    aggregate = {
        "n_subjects": n_subj,
        "mean_abs_pearson": float(np.mean([v["abs_pearson"] for v in per_joint.values()])),
        "mean_pearson":     float(np.mean([v["pearson"] for v in per_joint.values()])),
    }

    # ── Outputs ──────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"per_joint": per_joint, "aggregate": aggregate}, f, indent=2)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for j, (disp, _, _) in enumerate(JOINTS):
        gt = gt_mean[:, j]; sd = gt_std[:, j]; rl = rl_cycles[:, j]; rf = ref_cycles[:, j]
        ax = axes[j]
        ax.fill_between(grid, gt - sd, gt + sd, color="0.6", alpha=0.35,
                        label=f"Moreira 2021 (n={n_subj}, V4) mean±std" if j == 0 else None)
        ax.plot(grid, gt, "k", lw=1.2)
        ax.plot(grid, rf, color="tab:blue", lw=1.2, ls="--",
                label="BVH reference" if j == 0 else None)
        ax.plot(grid, rl, color="tab:red", lw=1.5,
                label="rl_mass rollout" if j == 0 else None)
        pj = per_joint[disp]
        ax.set_ylabel(f"{disp} flex (deg)\nref|r|={pj['ref_vs_gt_absr']:.2f} "
                      f"track|r|={pj['rollout_vs_ref_absr']:.2f}",
                      fontsize=8)
        if j == 0:
            ax.legend(fontsize=8, loc="upper right", frameon=False)
    axes[-1].set_xlabel("% gait cycle (0 = right heel strike, Zeni-anchored)")
    fig.suptitle(
        f"Walk kinematics fairness gate — rl_mass vs Moreira 2021\n"
        f"mean |r| = {aggregate['mean_abs_pearson']:.2f}  "
        f"(if low → activation mismatch is a MOTION confound, not redundancy)",
        fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare.png"), dpi=120)
    plt.close(fig)

    # CSV
    cols = ["time_pct"]; data_cols = [grid]
    for j, (disp, _, _) in enumerate(JOINTS):
        cols += [f"{disp}_gt_mean", f"{disp}_gt_std", f"{disp}_ref", f"{disp}_rl"]
        data_cols += [gt_mean[:, j], gt_std[:, j], ref_cycles[:, j], rl_cycles[:, j]]
    np.savetxt(os.path.join(out_dir, "compare.csv"), np.column_stack(data_cols),
               delimiter=",", header=",".join(cols), comments="")

    return {"per_joint": per_joint, "aggregate": aggregate, "out_dir": out_dir}


def main():
    p = argparse.ArgumentParser(description="Validate walk kinematics vs Moreira 2021")
    p.add_argument("--config", "-c", required=True)
    a = p.parse_args()
    res = compare(a.config)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
