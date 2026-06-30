"""Precompute the EMG activation-prior target for walk muscle training.

Produces an .npz the trainer loads: for each of the 4 Moreira surface muscles
(mapped to SNU indices), a peak-normalised activation target as a function of
the BVH gait-cycle phase phi in [0,1).

Two coordinate systems must be reconciled:
  * Moreira EMG is indexed by gait cycle %, 0 = right heel strike.
  * The trainer tags each muscle tuple with phi = fmod(time, max_time)/max_time,
    0 = BVH frame 0 (NOT heel strike).
We calibrate the offset phi_hs (the phase at which right heel strike occurs in
the BVH reference) by playing the reference target poses over one cycle, running
forward kinematics, and detecting the right heel strike (Zeni 2008). Then
    gait%(phi) = ((phi - phi_hs) mod 1) * 100
and the target curve is resampled onto a phase grid.

Usage:
    uv run python tools/build_emg_prior.py \
        --walk-config data/rl_mass/configs/walk_moreira.json \
        --emg-cohort data/rl_mass/emg_moreira/V4_R \
        --out data/rl_mass/emg_moreira/emg_prior_V4.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from rl_mass.env import MassEnv
from rl_mass.finger.gt import (
    EmgCohortLoader,
    anterior_projection,
    detect_heel_strikes,
)

# EMG label -> SNU muscle name(s). A surface electrode detects a region, so a
# label may map to a GROUP of synergists; the model signal is then the
# f0(PCSA)-weighted mean activation of the group. GAL (lateral gastroc
# electrode) maps to the whole triceps-surae group because the model's
# min-activation solver zeros the weak lateral head and carries push-off on the
# stronger soleus / medial gastroc (see emg-prior diagnostics).
MUSCLE_MAP = {
    # VL electrode → quadriceps group
    "VL":  ["R_Vastus_Lateralis", "R_Vastus_Lateralis1",
            "R_Vastus_Medialis", "R_Vastus_Medialis1", "R_Vastus_Medialis2",
            "R_Vastus_Intermedius", "R_Vastus_Intermedius1",
            "R_Rectus_Femoris", "R_Rectus_Femoris1"],
    # BF electrode → hamstring group
    "BF":  ["R_Bicep_Femoris_Longus", "R_Bicep_Femoris_Short",
            "R_Bicep_Femoris_Short1", "R_Semitendinosus",
            "R_Semimembranosus", "R_Semimembranosus1"],
    # TA electrode → dorsiflexor group
    "TA":  ["R_Tibialis_Anterior", "R_Extensor_Digitorum_Longus",
            "R_Extensor_Digitorum_Longus1", "R_Extensor_Digitorum_Longus2",
            "R_Extensor_Digitorum_Longus3", "R_Extensor_Digitorum1",
            "R_Extensor_Hallucis_Longus", "R_Peroneus_Tertius",
            "R_Peroneus_Tertius1"],
    # GAL electrode → triceps-surae / plantarflexor group
    "GAL": ["R_Gastrocnemius_Lateral_Head", "R_Gastrocnemius_Medial_Head",
            "R_Soleus", "R_Soleus1"],
}


def calibrate_phi_hs(env, n=240) -> float:
    """Phase phi in [0,1) at which right heel strike occurs in the BVH reference."""
    import mujoco
    m = env.model
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Pelvis")
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "TalusR")
    max_time = env.character.bvh.max_time
    dt = max_time / n
    foot, pelv, phis = [], [], []
    for k in range(n):
        t = k * dt
        mj_q, _ = env.character.get_target_pos_and_vel_mujoco(t, dt)
        env.data.qpos[:] = mj_q
        mujoco.mj_forward(m, env.data)
        foot.append(env.data.xpos[fid].copy())
        pelv.append(env.data.xpos[pid].copy())
        phis.append((t % max_time) / max_time)
    proj = anterior_projection(np.array(foot), np.array(pelv), up_axis=1)
    strikes = detect_heel_strikes(proj)
    if not strikes:
        raise RuntimeError("No heel strike found in reference playback")
    return float(phis[strikes[0]])


def main():
    p = argparse.ArgumentParser(description="Build EMG activation prior target")
    p.add_argument("--walk-config", default="data/rl_mass/configs/walk_moreira.json")
    p.add_argument("--emg-cohort", default="data/rl_mass/emg_moreira/V4_R")
    p.add_argument("--out", default="data/rl_mass/emg_moreira/emg_prior_V4.npz")
    p.add_argument("--n-phase", type=int, default=101)
    a = p.parse_args()

    env = MassEnv(a.walk_config)
    names = list(env.get_muscle_metadata()["names"])
    f0 = np.asarray(env.muscle_system.f0, dtype=np.float64)
    labels = list(MUSCLE_MAP.keys())

    # Build a padded (n_labels, G) group index + f0-weight matrix. Singletons
    # use one slot with weight 1; groups use f0/PCSA-weighted means. Unused
    # slots get index 0 and weight 0 so they contribute nothing.
    groups = [[MUSCLE_MAP[k]] if isinstance(MUSCLE_MAP[k], str) else MUSCLE_MAP[k]
              for k in labels]
    G = max(len(g) for g in groups)
    group_idx = np.zeros((len(labels), G), dtype=np.int64)
    group_w   = np.zeros((len(labels), G), dtype=np.float64)
    for r, g in enumerate(groups):
        idxs = [names.index(n) for n in g]
        w = np.array([f0[i] for i in idxs], dtype=np.float64)
        w = w / w.sum()
        group_idx[r, :len(idxs)] = idxs
        group_w[r, :len(idxs)] = w
    # Back-compat scalar index = the primary (first) muscle of each channel.
    snu_idx = group_idx[:, 0].copy()

    # Moreira cohort mean EMG, peak-normalised per muscle (shape prior).
    gt = EmgCohortLoader(a.emg_cohort, muscles=labels, n_points=101).load()
    emg = gt.activation.astype(np.float64)              # (101, 4) %MVC
    emg_norm = emg / np.clip(emg.max(axis=0), 1e-9, None)  # peak -> 1

    phi_hs = calibrate_phi_hs(env)

    # Resample onto the BVH phase grid via the heel-strike offset.
    phase_grid = np.linspace(0, 1, a.n_phase, endpoint=False)
    gait_pct = ((phase_grid - phi_hs) % 1.0) * 100.0
    src_pct = np.linspace(0, 100, emg_norm.shape[0])
    target = np.column_stack([np.interp(gait_pct, src_pct, emg_norm[:, j])
                              for j in range(len(labels))])  # (n_phase, 4)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out,
             snu_idx=snu_idx,
             group_idx=group_idx,
             group_w=group_w.astype(np.float32),
             phase_grid=phase_grid,
             target=target,
             phi_hs=np.float64(phi_hs),
             labels=np.array(labels))
    print(json.dumps({
        "out": a.out, "phi_hs": phi_hs,
        "groups": {labels[r]: [names[i] for i, w in zip(group_idx[r], group_w[r]) if w > 0]
                   for r in range(len(labels))},
        "group_w": {labels[r]: [round(float(w), 3) for w in group_w[r] if w > 0]
                    for r in range(len(labels))},
        "target_peaks_at_pct": [float(src_pct[int(np.argmax(emg_norm[:, j]))])
                                for j in range(len(labels))],
    }, indent=2))


if __name__ == "__main__":
    main()
