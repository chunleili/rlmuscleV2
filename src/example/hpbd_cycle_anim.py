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

# Force the validated hpbd_sync variant (mirror hpbd_conv_curve.VARIANTS["hpbd_sync"]).
HPBD_SYNC = {"use_multigrid": True, "mg_mode": "hpbd",
             "mg_coarse_sync_state": True, "mg_coarse_arap": True}


def muscle_surface(sim):
    """Flat (3*ntri,) int32 muscle surface tris indexing sim.pos."""
    if getattr(sim, "surface_tris_np", None) is not None:
        return np.asarray(sim.surface_tris_np, dtype=np.int32).reshape(-1)
    return vectorized_surface_tris(
        np.asarray(sim.tet_idx.numpy(), dtype=np.int64)).reshape(-1).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    base = load_config(args.config)
    wp.init()
    device = base.get("demo", {}).get("device", "cuda:0")
    F0 = int(base.get("conv_frame", 120))
    M = int(base.get("conv_substeps", 120))
    out = base.get("cycle_anim_out") or os.path.join(
        base.get("bench_out", "output/hpbd_cycle"), "muscle_cycle_hpbd.usd")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    cfg = copy.deepcopy(base)
    cfg.update(HPBD_SYNC)
    for c in cfg.get("constraints", []):          # attach IS the round-trip drive
        if c.get("name") == "belly_attach":
            c["enable"] = True

    sim = MultiGridSim(cfg, device=device)
    sim.preload_targets([F0])
    tris = muscle_surface(sim)
    ntri = tris.shape[0] // 3
    print(f"NV={sim.NV} NT={sim.NT} surface_tris={ntri}; cycle rest->F{F0}->rest, "
          f"{M} substeps -> {out}")

    # rest init + F0 bone target (the alpha=1 goal); alpha=0 goal = rest.
    rest_np = sim.pos0_np.astype(np.float32)
    sim.pos = wp.from_numpy(rest_np, dtype=wp.vec3, device=device)
    sim.pprev = wp.from_numpy(rest_np, dtype=wp.vec3, device=device)
    sim.vel.zero_()
    sim.load_frame_targets(F0)
    tgt_F0 = sim.target_pos.numpy().copy()
    tgt_rest = rest_np.copy()

    from pxr import Usd, UsdGeom, Vt, Sdf

    stage = Usd.Stage.CreateNew(out)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(M)
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
    for k in range(M):
        frac = (k + 1) / M
        alpha = 1.0 - abs(2.0 * frac - 1.0)        # triangle: 0 -> 1 (mid) -> 0
        tgt = ((1.0 - alpha) * tgt_rest + alpha * tgt_F0).astype(np.float32)
        sim.target_pos.assign(tgt)
        sim._update_belly_attach_targets()
        sim.substep()
        bake(k + 1)
        if np.isnan(sim.pos.numpy()).any():
            print(f"  NaN at substep {k + 1}")
            break
        if (k + 1) % 20 == 0 or k + 1 == M:
            print(f"  substep {k + 1}/{M}  alpha={alpha:.3f}  inv={sim.count_inverted()}")

    stage.GetRootLayer().Save()
    print(f"\nwrote {out}  ({M + 1} frames)")


if __name__ == "__main__":
    main()
