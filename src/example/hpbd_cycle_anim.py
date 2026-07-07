"""Round-trip (cycle) animation dump for the mg HPBD muscle solver.

Runs the SAME round-trip drive the convergence bench validates — start at REST,
ease the belly-attach target rest -> F0 bone pose -> rest over M substeps
(triangle alpha) — but instead of logging the ARAP residual it BAKES the muscle
surface as a point-animated USD mesh so the deform-and-recover can be eyeballed.
Because the motion ends back at rest, a clean recovery = the mesh returns to its
rest shape with no stuck fold; that is exactly the quality hpbd_sync + the
FAS-consistent coarse_B + coarse-attach-off recipe wins on.

Pure sim -> mesh dump, NO plotting (plot lives in hpbd_conv_curve.py). Only CLI
arg is --config; everything else comes from the JSON:
  conv_frame (F0), conv_substeps (M), demo.device, cycle_anim_out (optional; else
  derived from bench_out). The solver is forced to the winning hpbd_sync variant
  (state-sync restrict + coarse ARAP) regardless of the file's mg_mode so the
  dump always shows the validated scheme.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import warp as wp

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from muscle.mg.mg import MultiGridSim, load_config
from muscle.util.mesh_io import vectorized_surface_tris

# A/B variants (mirror hpbd_conv_curve.VARIANTS): single-level baseline vs the
# validated state-sync HPBD. Both baked over the SAME round-trip so the recovery
# can be compared frame-for-frame in usdview.
VARIANTS = {
    "off":  {"use_multigrid": False},
    "hpbd": {"use_multigrid": True, "mg_mode": "hpbd",
             "mg_coarse_sync_state": True, "mg_coarse_arap": True},
}


def muscle_surface(sim):
    """Flat (3*ntri,) int32 muscle surface tris indexing sim.pos."""
    if getattr(sim, "surface_tris_np", None) is not None:
        return np.asarray(sim.surface_tris_np, dtype=np.int32).reshape(-1)
    return vectorized_surface_tris(
        np.asarray(sim.tet_idx.numpy(), dtype=np.int64)).reshape(-1).astype(np.int32)


def run_variant(base, name, over, frames, device, out):
    """Build the sim for one variant, sweep the real bone frames, bake USD.

    frames = the round-trip frame schedule (start -> F0 -> start). Driving by REAL
    bone frames (load_frame_targets each substep) makes the belly-attach target
    ride the true bone ARC — unlike a straight-line world-space position lerp,
    which chords through a limb's rotation and collapses the distal targets toward
    the rotation center mid-cycle (spaghettifies thin forearm/hand muscles)."""
    from pxr import Usd, UsdGeom, Vt

    cfg = copy.deepcopy(base)
    cfg.update(over)
    for c in cfg.get("constraints", []):          # attach IS the round-trip drive
        if c.get("name") == "belly_attach":
            c["enable"] = True

    sim = MultiGridSim(cfg, device=device)
    sim.preload_targets(sorted(set(frames)))
    tris = muscle_surface(sim)
    ntri = tris.shape[0] // 3
    nsub = len(frames)
    print(f"[{name}] NV={sim.NV} NT={sim.NT} surface_tris={ntri}; "
          f"cycle F{frames[0]}->F{max(frames)}->F{frames[-1]}, {nsub} substeps -> {out}")

    # rest init (muscle rest == start-frame bone pose, target reconstructs it exactly)
    rest_np = sim.pos0_np.astype(np.float32)
    sim.pos = wp.from_numpy(rest_np, dtype=wp.vec3, device=device)
    sim.pprev = wp.from_numpy(rest_np, dtype=wp.vec3, device=device)
    sim.vel.zero_()

    stage = Usd.Stage.CreateNew(out)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(nsub)
    stage.SetTimeCodesPerSecond(1.0 / max(float(base.get("delta_t", 0.04166)), 1e-6))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    mesh = UsdGeom.Mesh.Define(stage, "/muscle_cycle")
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
        np.full(ntri, 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris))
    pts_attr = mesh.CreatePointsAttr()

    def bake(t):
        pts_attr.Set(Vt.Vec3fArray.FromNumpy(
            sim.pos.numpy().astype(np.float32)), Usd.TimeCode(t))

    bake(0)                                        # rest
    for k, f in enumerate(frames):
        sim.load_frame_targets(f)                  # bone -> frame f, target rides real arc
        sim.substep()
        bake(k + 1)
        if np.isnan(sim.pos.numpy()).any():
            print(f"  [{name}] NaN at substep {k + 1} (frame {f})")
            break
        if (k + 1) % 40 == 0 or k + 1 == nsub:
            print(f"  [{name}] substep {k + 1}/{nsub}  frame={f}  "
                  f"inv={sim.count_inverted()}")

    stage.GetRootLayer().Save()
    print(f"[{name}] wrote {out}  ({nsub + 1} frames)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    base = load_config(args.config)
    wp.init()
    device = base.get("demo", {}).get("device", "cuda:0")
    start = int(base.get("start_frame", 1))
    F0 = int(base.get("conv_frame", 120))
    out_dir = os.path.dirname(base.get("cycle_anim_out") or "") or \
        base.get("bench_out", "output/hpbd_cycle")
    os.makedirs(out_dir, exist_ok=True)

    # one bone frame per substep, up then back: start -> F0 -> start (true arc)
    frames = list(range(start, F0 + 1)) + list(range(F0 - 1, start - 1, -1))

    for name, over in VARIANTS.items():
        run_variant(base, name, over, frames, device,
                    os.path.join(out_dir, f"muscle_cycle_{name}.usd"))


if __name__ == "__main__":
    main()
