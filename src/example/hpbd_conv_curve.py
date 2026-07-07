"""Convergence-curve test: ARAP C-residual history under three boundary regimes.

The multigrid claim is about CONVERGENCE RATE: at a fixed problem, how fast does
each scheme drive the ARAP constraint residual C = ‖σ(F)−1‖ toward zero per
substep? A correct coarse correction captures the low-frequency error the fine
Jacobi smoother is slow on, so its curve should drop faster/lower. This harness
plots ONLY that C-residual (no track panel).

Three regimes (one panel each), to compare MG under no BC vs the two BC types:

  squash  (无 BC): compress the whole mesh about its centroid (anisotropic scale)
          and let it recover by PURE ARAP — no pin, no attach. A uniform affine
          squash is the LOWEST-frequency error for the local Jacobi smoother
          (displacement field ∝ x), so it is the cleanest elasticity-MG test.

  pin     (Dirichlet BC): hard-pin the bone-nearest fraction of verts to the F0
          bone pose; the rest of the belly relaxes by ARAP from that pinned
          boundary. attach is neutralized so the pin is the only drive.

  attach  (Robin BC): every muscle vert soft-attached (belly_attach) to the F0
          bone pose, driven from rest. A compliant spring (force ∝ k(target−x))
          is a Robin / elastic-support BC — NOT Neumann (which prescribes a
          displacement-independent traction). Limits: k→∞ = Dirichlet, k→0 =
          Neumann. This is the humanV2 muscle's real BC — the regime where
          hpbd_sync was shown to be BIASED.

projection MG only appears in the attach panel: it restricts and teleports the
full bone target, which is meaningful only for all-vert attach. In squash it
would cheat (target≈rest → instant solve); in pin it would over-drive (push all
verts to bone, not just the pinned subset). squash/pin therefore compare off vs
hpbd_sync (the state-based coarse correction that needs no target).

Protocol per (regime × variant): reset to REST, apply the regime's drive (squash
the init, or kinematic-step to F0), run M substeps, record ARAP C-residual RMS.

Only CLI arg is --config; options live in the JSON: conv_frame (F0),
conv_substeps (M), conv_regimes, conv_squash_scale, conv_pin_frac. The base
belly_attach entry's enable flag is overridden per regime by the harness (on for
attach, off for squash/pin), so its base value does not matter.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from muscle.mg.mg import MultiGridSim, load_config, _rest_B_p3ref
from muscle.mg.kernel import arap_residual_accum_kernel


def arap_rms(sim, fine_B, sumbuf):
    """RMS over tets of ‖σ(F)−1‖ — the ARAP shape residual (the MG metric)."""
    sumbuf.zero_()
    wp.launch(arap_residual_accum_kernel, dim=sim.NT,
              inputs=[sim.tet_indices, sim.pos, fine_B, sumbuf], device=sim._device)
    return float(np.sqrt(sumbuf.numpy()[0] / max(sim.NT, 1)))


# Solver knobs per variant (merged into the base cfg).
VARIANTS = {
    "off":        {"use_multigrid": False},
    "projection": {"use_multigrid": True, "mg_mode": "projection"},
    "hpbd_sync":  {"use_multigrid": True, "mg_mode": "hpbd",
                   "mg_coarse_sync_state": True, "mg_coarse_arap": True},
}

# regime -> (drive, list of variants to run). drive selects the BC / initial
# condition; variants are the solver settings compared within that panel.
REGIMES = {
    "squash": ("squash", ["off", "hpbd_sync"]),
    "pin":    ("pin",    ["off", "hpbd_sync"]),
    "attach": ("attach", ["off", "hpbd_sync"]),
    # round-trip: drive the attach target rest -> F0 bone -> rest (triangle alpha).
    # Ends at rest, so the FINAL residual-vs-rest is meaningful (recovery quality).
    "cycle":  ("cycle",  ["off", "hpbd_sync"]),
}


def build_cfg(base, solver_over, drive):
    """Deep-merge base + solver override + regime BC override.

    The Robin belly-attach is a genuine on/off per regime: attach ENABLES it (it
    IS the Robin/elastic-support BC); squash/pin DISABLE it entirely — squash recovers freely,
    pin is driven only by its hard pins. squash/pin also kill the coarse soft-attach
    (beta=0). The base belly_attach entry supplies the stiffness the attach regime
    uses; its enable flag is overridden here per regime."""
    cfg = copy.deepcopy(base)
    cfg.update(solver_over)
    for c in cfg.get("constraints", []):
        if c.get("name") == "belly_attach":
            c["enable"] = drive in ("attach", "cycle")
    if drive in ("squash", "pin"):
        cfg["mg_coarse_attach_beta"] = 0.0
    return cfg


def bone_nearest_pin_active(sim, frac):
    """pin_active mask (NCONS int32): 1 for the `frac` fraction of muscle verts
    closest to the bone surface (the true attachment sites), 0 otherwise."""
    start = int(getattr(sim, "start_frame", 1))
    bpos = np.asarray(sim._bone_source.positions(start), dtype=np.float64)
    gap, _ = cKDTree(bpos).query(sim.pos0_np.astype(np.float64), k=1)
    n_pin = max(1, int(round(frac * gap.shape[0])))
    thresh = np.partition(gap, n_pin - 1)[n_pin - 1]
    active = (gap <= thresh).astype(np.int32)
    return active


def squash_positions(pos0, scale):
    """Anisotropic compression about the centroid: p' = c + (p−c)·scale."""
    c = pos0.mean(axis=0, keepdims=True)
    return (c + (pos0 - c) * np.asarray(scale, dtype=pos0.dtype)).astype(np.float32)


def run_variant(base, name, drive, F0, M, device, squash_scale, pin_frac):
    cfg = build_cfg(base, VARIANTS[name], drive)
    sim = MultiGridSim(cfg, device=device)
    fine_B = wp.from_numpy(_rest_B_p3ref(sim.pos0_np.astype(np.float64), sim.tet_np),
                           dtype=wp.mat33, device=device)
    sumbuf = wp.zeros(1, dtype=wp.float32, device=device)
    sim.preload_targets([F0])

    tgt_cycle = None
    if drive == "squash":
        # no BC: start compressed, relax freely (no kinematic drive)
        sq = squash_positions(sim.pos0_np.astype(np.float32), squash_scale)
        sim.pos = wp.from_numpy(sq, dtype=wp.vec3, device=device)
        sim.pprev = wp.from_numpy(sq, dtype=wp.vec3, device=device)
        sim.vel.zero_()
    elif drive == "cycle":
        # round-trip: start at REST, drive the attach target rest -> F0 bone -> rest
        # over M substeps (triangle alpha). Ends at rest so the final residual-vs-rest
        # measures how cleanly the solver recovered (no hysteresis / stuck fold).
        rest_np = sim.pos0_np.astype(np.float32)
        sim.pos = wp.from_numpy(rest_np, dtype=wp.vec3, device=device)
        sim.pprev = wp.from_numpy(rest_np, dtype=wp.vec3, device=device)
        sim.vel.zero_()
        sim.load_frame_targets(F0)                    # target_pos = F0 bone goal
        tgt_F0 = sim.target_pos.numpy().copy()
        tgt_rest = rest_np.copy()                     # alpha=0 goal = rest (mesh rest_offset=0)
        tgt_cycle = (tgt_rest, tgt_F0)
    else:
        # pin/attach: start at rest, freeze the bone BC at F0
        sim.pos = wp.from_numpy(sim.pos0_np.astype(np.float32), dtype=wp.vec3, device=device)
        sim.pprev = wp.from_numpy(sim.pos0_np.astype(np.float32), dtype=wp.vec3, device=device)
        sim.vel.zero_()
        if drive == "pin":
            # Dirichlet: hard-pin only the bone-nearest subset to the F0 pose
            active = bone_nearest_pin_active(sim, pin_frac)
            sim.pin_active = wp.array(active, dtype=wp.int32, device=device)
            print(f"    [pin] {int(active.sum())}/{active.size} verts hard-pinned "
                  f"(bone-nearest {pin_frac:.0%})")
        sim._kinematic_step(F0)

    rows = [(0, arap_rms(sim, fine_B, sumbuf), sim.count_inverted())]
    for k in range(M):
        if tgt_cycle is not None:
            # triangle alpha over the M substeps: 0 -> 1 (at M/2) -> 0 (at M)
            frac = (k + 1) / M
            alpha = 1.0 - abs(2.0 * frac - 1.0)
            tgt = ((1.0 - alpha) * tgt_cycle[0] + alpha * tgt_cycle[1]).astype(np.float32)
            sim.target_pos.assign(tgt)
            sim._update_belly_attach_targets()        # fine attach follows; coarse tracks in _coarse_solve
        sim.substep()
        r = arap_rms(sim, fine_B, sumbuf)
        rows.append((k + 1, r, sim.count_inverted()))
        if np.isnan(sim.pos.numpy()).any():
            print(f"  [{drive:6s} {name:11s}] NaN at substep {k+1}")
            break
    print(f"  [{drive:6s} {name:11s}] arapRMS {rows[0][1]:.3e} -> {rows[-1][1]:.3e}; "
          f"inv {rows[-1][2]}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    base = load_config(args.config)
    wp.init()
    device = base.get("demo", {}).get("device", "cuda:0")
    F0 = int(base.get("conv_frame", 120))
    M = int(base.get("conv_substeps", 120))
    squash_scale = base.get("conv_squash_scale", [1.1, 0.6, 1.1])
    pin_frac = float(base.get("conv_pin_frac", 0.15))
    regimes = base.get("conv_regimes", ["squash", "pin", "attach"])
    out_dir = base.get("bench_out", "output/hpbd_conv")
    os.makedirs(out_dir, exist_ok=True)

    titles = {"squash": "no BC - squash recovery (free ARAP)",
              "pin": f"Dirichlet BC - pin (bone-nearest {pin_frac:.0%})",
              "attach": "Robin BC - attach (belly_attach)",
              "cycle": "round-trip rest->F0->rest (final residual = recovery)"}
    print(f"C-residual convergence: F0={F0}, {M} substeps; regimes={regimes}; "
          f"squash_scale={squash_scale}; pin_frac={pin_frac:g}")

    results = {}  # (regime, name) -> rows
    for regime in regimes:
        drive, variants = REGIMES[regime]
        print(f"\n=== regime: {regime} ({drive}) ===")
        for name in variants:
            results[(regime, name)] = run_variant(
                base, name, drive, F0, M, device, squash_scale, pin_frac)

    for (regime, name), rows in results.items():
        with open(os.path.join(out_dir, f"conv_{regime}_{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["substep", "arap_res_rms", "inv"])
            w.writerows(rows)

    # One panel per regime; ARAP C-residual RMS vs substep (log-y). Shared y-lims
    # so the three regimes are directly comparable.
    colors = {"off": "C0", "projection": "C1", "hpbd_sync": "C2"}
    fig, axes = plt.subplots(1, len(regimes), figsize=(6.2 * len(regimes), 5.2),
                             squeeze=False)
    ymin, ymax = np.inf, -np.inf
    for rows in results.values():
        y = np.array(rows, dtype=float)[:, 1]
        pos = y[y > 0]
        if pos.size:
            ymin, ymax = min(ymin, pos.min()), max(ymax, y.max())
    for ax, regime in zip(axes[0], regimes):
        _, variants = REGIMES[regime]
        for name in variants:
            r = np.array(results[(regime, name)], dtype=float)
            ax.semilogy(r[:, 0], np.maximum(r[:, 1], 1e-12), label=name,
                        color=colors.get(name), lw=2, marker=".", ms=4)
        ax.set(xlabel="substep held at F0 (0 = start)",
               ylabel="ARAP C-residual RMS  ‖σ(F)−1‖",
               title=titles.get(regime, regime))
        ax.set_ylim(ymin * 0.7, ymax * 1.4)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(f"ARAP C-residual convergence, F0={F0}: no BC vs Dirichlet vs Robin",
                 fontsize=13)
    fig.tight_layout()
    png = os.path.join(out_dir, f"conv_cresidual_F{F0}.png")
    fig.savefig(png, dpi=120)
    print(f"\nwrote {png} + per-(regime,variant) CSV -> {out_dir}")


if __name__ == "__main__":
    main()
