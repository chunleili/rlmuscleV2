"""Convergence-curve test: ARAP C-residual history at a FIXED boundary condition.

The per-frame benches measure only the final residual. The real multigrid claim
is about CONVERGENCE RATE: at a fixed problem (frozen bone target = one frame's
BC), how fast does each scheme drive the ARAP constraint residual C = ‖σ(F)−1‖
toward zero per iteration? A good coarse correction removes the low-frequency
error the fine smoother is slow on, so its curve should drop faster/lower.

Protocol (same fixed problem for every mode):
  1. reset the muscle to REST (pos=pos0, vel=0),
  2. kinematic-step ONCE to a strained frame F0 (hard-pins the tendon to F0; the
     freed belly starts at rest → a big LOW-FREQUENCY error to resolve — exactly
     what MG should accelerate),
  3. run M substeps holding that F0 target, recording after each substep:
       - ARAP C-residual RMS (shape distortion, the MG metric),
       - belly-attach track error (the compliant-BC residual),
       - inverted-tet count.

Compares off / projection / hpbd_sync (+ optional hpbd_sync_attach). Writes a
per-substep CSV and a log-scale PNG. Only CLI arg is --config; options live in
the JSON: conv_frame (F0), conv_substeps (M), diag_variants.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import warp as wp

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from muscle.mg.mg import MultiGridSim, load_config, _rest_B_p3ref
from muscle.mg.kernel import arap_residual_accum_kernel


def arap_sumsq(sim, fine_B, sumbuf):
    """Sum over tets of ‖σ(F)−1‖² (raw, un-normalized). RMS = sqrt(sumsq/NT)."""
    sumbuf.zero_()
    wp.launch(arap_residual_accum_kernel, dim=sim.NT,
              inputs=[sim.tet_indices, sim.pos, fine_B, sumbuf], device=sim._device)
    return float(sumbuf.numpy()[0])


def attach_residual(sim):
    """belly-attach constraint violation over the freed belly verts (pin_active==0):
    returns (mean ‖p−tgt‖ [track error], sum ‖p−tgt‖²)."""
    active = sim.pin_active.numpy() if sim.pin_active is not None else None
    idx = (np.arange(len(sim._pin_pts_np)) if active is None
           else np.where(active == 0)[0])
    if idx.size == 0:
        return 0.0, 0.0
    tgt = sim.target_pos.numpy()[idx] + sim.rest_offset.numpy()[idx]
    p = sim.pos.numpy()[sim._pin_pts_np[idx]]
    dist = np.linalg.norm(p - tgt, axis=1)
    return float(dist.mean()), float((dist ** 2).sum())


def _stiffness(cfg, name, default):
    for c in cfg.get("constraints", []):
        if c.get("name") == name:
            return float(c.get("stiffness", default))
    return default


def sample(sim, fine_B, sumbuf, k_arap, k_attach):
    """One measurement: (arap RMS, track mm-in-m, inv, TOTAL constraint energy).
    Total energy = ½·k_arap·Σ‖σ−1‖² + ½·k_attach·Σ‖p−tgt‖² — the actual XPBD
    objective, un-gameable: leaving the belly at rest blows up the attach term,
    teleporting the cage blows up the ARAP term. Only satisfying BOTH lowers it."""
    ssq = arap_sumsq(sim, fine_B, sumbuf)
    track, asq = attach_residual(sim)
    rms = np.sqrt(ssq / max(sim.NT, 1))
    energy = 0.5 * k_arap * ssq + 0.5 * k_attach * asq
    inv = sim.count_inverted()
    return rms, track, inv, energy


def run_variant(cfg, label, overrides, F0, M, device, k_arap, k_attach):
    cfg_m = dict(cfg)
    cfg_m.update(overrides)
    sim = MultiGridSim(cfg_m, device=device)
    fine_B = wp.from_numpy(_rest_B_p3ref(sim.pos0_np.astype(np.float64), sim.tet_np),
                           dtype=wp.mat33, device=device)
    sumbuf = wp.zeros(1, dtype=wp.float32, device=device)

    sim.preload_targets([F0])
    # fresh REST start, same for every mode
    sim.pos = wp.from_numpy(sim.pos0_np.astype(np.float32), dtype=wp.vec3, device=device)
    sim.pprev = wp.from_numpy(sim.pos0_np.astype(np.float32), dtype=wp.vec3, device=device)
    sim.vel.zero_()
    sim._kinematic_step(F0)             # freeze BC at F0 (hard-pin tendon, set targets)

    # s0 = the COMMON origin: rest state measured against the F0 BC (ARAP≈0, but the
    # belly-attach term is huge -> total energy is large and identical for all modes).
    rms, track, inv, energy = sample(sim, fine_B, sumbuf, k_arap, k_attach)
    rows = [(0, rms, track, inv, energy)]
    for k in range(M):
        sim.substep()                  # one relaxation substep against the frozen F0 BC
        rms, track, inv, energy = sample(sim, fine_B, sumbuf, k_arap, k_attach)
        rows.append((k + 1, rms, track, inv, energy))
        if np.isnan(sim.pos.numpy()).any():
            print(f"  [{label:16s}] NaN at substep {k+1}")
            break
    print(f"  [{label:16s}] energy {rows[0][4]:.3e} -> {rows[-1][4]:.3e}; "
          f"arapRMS {rows[-1][1]:.3e}; track {rows[-1][2]*1000:.2f}mm; inv {rows[-1][3]}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    wp.init()
    device = cfg.get("demo", {}).get("device", "cuda:0")
    F0 = int(cfg.get("conv_frame", 120))
    M = int(cfg.get("conv_substeps", 120))
    out_dir = cfg.get("bench_out", "output/hpbd_conv")
    os.makedirs(out_dir, exist_ok=True)

    k_arap = _stiffness(cfg, "arap", 1e6)
    k_attach = _stiffness(cfg, "belly_attach", 1e6)
    HP = {"use_multigrid": True, "mg_mode": "hpbd", "mg_coarse_sync_state": True}
    all_variants = {
        "off":              {"use_multigrid": False},
        "projection":       {"use_multigrid": True, "mg_mode": "projection"},
        "hpbd_sync":        {**HP, "mg_coarse_arap": True},
        "hpbd_sync_attach": {**HP, "mg_coarse_arap": False},
    }
    names = cfg.get("diag_variants", ["off", "projection", "hpbd_sync"])

    print(f"Convergence curve: frame F0={F0}, {M} substeps held, modes={names}; "
          f"k_arap={k_arap:g} k_attach={k_attach:g}")
    results = {}
    for name in names:
        print(f"\n=== {name} ===")
        results[name] = run_variant(cfg, name, all_variants[name], F0, M, device,
                                    k_arap, k_attach)

    for name in names:
        with open(os.path.join(out_dir, f"conv_{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["substep", "arap_res", "track_m", "inv", "total_energy"])
            w.writerows(results[name])

    # 3 panels vs substep (all start at s0 = common rest origin), log-y:
    #   (1) TOTAL constraint energy — the fair, un-gameable convergence metric
    #   (2) belly track error       — the LOW-FREQUENCY error MG must accelerate
    #   (3) ARAP shape residual     — shown for context; off is low here only
    #                                 because it leaves the BC unsatisfied (panel 2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cols = {"total": 4, "track": 2, "arap": 1}
    titles = ["TOTAL constraint energy (½kΣC²) — fair metric",
              "belly track error (mm) — low-freq error MG targets",
              "ARAP shape residual RMS — off low = BC unsatisfied"]
    ylabels = ["total constraint energy", "belly track error (mm)", "ARAP C-residual RMS"]
    scales = [1.0, 1000.0, 1.0]
    for ax, key, title, yl, sc in zip(axes, ["total", "track", "arap"], titles, ylabels, scales):
        for name in names:
            r = np.array(results[name], dtype=float)
            y = np.maximum(r[:, cols[key]] * sc, 1e-12)
            ax.semilogy(r[:, 0], y, label=name, lw=2, marker=".", ms=4)
        ax.set(xlabel="substep held at F0 (0 = rest, common origin)", ylabel=yl, title=title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(f"Convergence at frozen BC F0={F0} (rest->far target, pure low-freq problem)",
                 fontsize=13)
    fig.tight_layout()
    png = os.path.join(out_dir, f"conv_curve_F{F0}.png")
    fig.savefig(png, dpi=110)
    print(f"\nwrote {png} + per-mode CSV -> {out_dir}")


if __name__ == "__main__":
    main()
