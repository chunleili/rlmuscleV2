"""humanV2 fiber-contraction probe: act=1 vs act=0 with a STATIC bone.

Reproduces the real humanV2 solver (MultiGridSim: arap+volume+attach+pin, the
config's constraint set) but ADDS the fibernorm fiber constraint that humanV2.json
omits, then isolates the fiber effect by holding the skeleton at frame 1:

  P0 = settle with activation 0          (passive baseline)
  P1 = settle with activation 1 on biceps (active)
  disp = P1 - P0  -> purely the fiber response (bone static, same pins/targets)

Compares solver modes (--mode jacobi|cgs) to test whether GPU Jacobi corrupts the
contraction. Writes a colored USD (displayColor = per-vertex displacement heatmap,
passive shown beside active) so the deformation is visible in usdview.

    PYTHONPATH=src python tools/fiber_probe_humanV2.py --mode jacobi --muscle bicepsBrachialis
"""
import argparse
import numpy as np
import warp as wp

from muscle.mg.mg import load_config
from example.humanV2 import build_humanV2_sim
from muscle.util.mesh_io import vectorized_surface_tris


def jet(x):
    """Simple blue->green->red heatmap for x in [0,1] -> (N,3) float."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], 1).astype(np.float32)


def settle(sim, frame, n_steps):
    for _ in range(n_steps):
        sim.step_frame(frame)
    wp.synchronize()
    return sim.positions_np().copy()


def fiber_stretch_per_tet(sim, pos, tet_sel):
    """Current along-fiber stretch |F w0| per selected muscle tet (1.0 = rest)."""
    tets = sim.tet_np[tet_sel]
    P = pos[tets]                      # (M,4,3)
    # Ds = [p0-p3, p1-p3, p2-p3]
    Ds = np.stack([P[:, 0] - P[:, 3], P[:, 1] - P[:, 3], P[:, 2] - P[:, 3]], axis=2)
    f0 = sim.materialW_np[tets].mean(1)               # (M,3) rest fiber dir at tet
    # rest_matrix Dminv: reconstruct from rest pos
    R = sim.pos0_np[tets]
    Dm = np.stack([R[:, 0] - R[:, 3], R[:, 1] - R[:, 3], R[:, 2] - R[:, 3]], axis=2)
    Dminv = np.linalg.inv(Dm)
    F = Ds @ Dminv                                    # (M,3,3)
    Ff = np.einsum('mij,mj->mi', F, f0)
    return np.linalg.norm(Ff, axis=1)


def write_colored_usd(path, surf_flat, P0, P1, used, out_dir):
    from pxr import Usd, UsdGeom, Vt, Sdf, Gf
    # compact to used verts
    remap = np.full(P0.shape[0], -1, np.int64)
    remap[used] = np.arange(used.shape[0])
    local = remap[surf_flat].astype(np.int32)
    p0, p1 = P0[used], P1[used]
    disp = np.linalg.norm(p1 - p0, axis=1)
    dmax = max(disp.max(), 1e-9)
    col = jet(disp / dmax)

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    counts = Vt.IntArray([3] * (local.shape[0] // 3))
    idx = Vt.IntArray.FromNumpy(local)

    # bbox width to offset the passive copy in +x
    w = float(p0[:, 0].max() - p0[:, 0].min()) * 1.3 + 0.05

    def mesh(name, pts, colors, tx):
        m = UsdGeom.Mesh.Define(stage, name)
        m.CreateFaceVertexCountsAttr(counts)
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(
            np.ascontiguousarray(pts + np.array([tx, 0, 0], np.float32), np.float32)))
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(colors, np.float32)))
        return m

    mesh("/active_heat", p1, col, 0.0)                       # active, disp heatmap
    mesh("/passive", p0, np.tile([0.6, 0.6, 0.6], (p0.shape[0], 1)), w)  # gray baseline
    stage.GetRootLayer().Save()
    print(f"[usd] wrote {path}  (active heat | passive gray, max_disp={dmax*1000:.2f}mm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/humanV2/config/humanV2.json")
    ap.add_argument("--mode", default="jacobi", choices=["jacobi", "cgs"])
    ap.add_argument("--muscle", default="bicepsBrachialis")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--settle", type=int, default=60)
    ap.add_argument("--act", type=float, default=1.0)
    ap.add_argument("--fiber-stiffness", type=float, default=1e8)
    ap.add_argument("--contraction-ratio", type=float, default=0.4)
    ap.add_argument("--fiber-model", default="aniso_arap",
                    choices=["fibernorm", "aniso_arap"],
                    help="fibernorm = linear TETFIBERNORM (branch2 + target); "
                         "aniso_arap = HOBAK quadratic anisotropic ARAP (branch1, "
                         "TETARAPANISO: SI4 inversion flag + detF/s>1.5 guard)")
    ap.add_argument("--fiber-scale", type=float, default=None,
                    help="override sim.fiber_stiffness_scale (default mg.py hardcoded 1.0)")
    ap.add_argument("--collision", action="store_true", help="keep SDF bone collision on")
    ap.add_argument("--belly-stiffness", type=float, default=None,
                    help="override belly_attach stiffness (humanV2.json=1e6); "
                         "lower it to free the belly and test the all-vert-clamp hypothesis")
    ap.add_argument("--fine-iters", type=int, default=None,
                    help="override fine_iterations (default 20); test Jacobi convergence")
    ap.add_argument("--arap-stiffness", type=float, default=None,
                    help="override arap (tetarapsvd) stiffness (default 1e6); test ARAP lock")
    ap.add_argument("--multigrid", action="store_true",
                    help="enable use_multigrid (coarse cage V-cycle accelerator, off in humanV2.json)")
    ap.add_argument("--pin-region", default=None,
                    help="override muscleToBonePin region (tendonmask|pinmask|full)")
    ap.add_argument("--pin-threshold", type=float, default=None,
                    help="hard-pin threshold for the chosen region (e.g. 0.5)")
    ap.add_argument("--quasi-static", action="store_true",
                    help="drop inertia (no velocity carry) -> static equilibrium, no overshoot/drift")
    ap.add_argument("--out", default="output/humanV2_fiber_probe.usd")
    args = ap.parse_args()

    wp.init()

    cfg = load_config(args.config)
    cfg["start_frame"] = 1
    cfg["end_frame"] = 1                       # hold a single static frame
    cfg["use_fascia"] = False
    if args.belly_stiffness is not None:
        for c in cfg.get("constraints", []):
            if c.get("name") == "belly_attach":
                c["stiffness"] = float(args.belly_stiffness)
    if args.arap_stiffness is not None:
        for c in cfg.get("constraints", []):
            if c.get("name") == "arap":
                c["stiffness"] = float(args.arap_stiffness)
    if args.fine_iters is not None:
        cfg["fine_iterations"] = int(args.fine_iters)
    if args.multigrid:
        cfg["use_multigrid"] = True
    if args.quasi_static:
        cfg["quasi_static"] = True
    if args.pin_region is not None:
        for c in cfg.get("constraints", []):
            if c.get("name") == "muscleToBonePin":
                c["region"] = args.pin_region
                if args.pin_threshold is not None:
                    if args.pin_region == "tendonmask":
                        c["tendon_pin_threshold"] = float(args.pin_threshold)
                    else:
                        c["pin_threshold"] = float(args.pin_threshold)
    cfg["use_jacobi"] = (args.mode == "jacobi")
    cfg["use_colored_gs"] = (args.mode == "cgs")
    # Inject fiber constraint humanV2.json omits.
    cfg.setdefault("constraints", []).append(dict(
        type="fiber", name="fiber", enable=True, fiber_model=args.fiber_model,
        stiffness=args.fiber_stiffness, dampingratio=0.1,
        contraction_ratio=args.contraction_ratio))
    demo = cfg.setdefault("demo", {})
    demo["device"] = args.device
    demo.setdefault("collision", {})["enabled"] = bool(args.collision)
    demo["activate"] = None                    # we drive activation directly

    sim = build_humanV2_sim(cfg)
    if args.fiber_scale is not None:
        sim.fiber_stiffness_scale = float(args.fiber_scale)
    print(f"[probe] mode={args.mode} use_jacobi={sim.use_jacobi} "
          f"use_colored_gs={getattr(sim,'use_colored_gs',None)} use_fiber={sim.use_fiber} "
          f"fiber_model={sim.fiber_model} k_fib={sim.fiber_stiffness:g} "
          f"fiber_scale={sim.fiber_stiffness_scale:g} ratio={sim.contraction_ratio}")

    mask = sim.tet_mask_for_muscles(args.muscle)        # per muscle-tet bool
    n_sel = int(mask.sum())
    print(f"[probe] muscle '{args.muscle}': {n_sel}/{mask.shape[0]} muscle tets selected")
    if n_sel == 0:
        raise SystemExit("no tets matched; check --muscle")
    sel_idx = np.where(mask)[0]

    # passive baseline
    sim.set_fiber_activation(np.zeros(mask.shape[0], np.float32))
    P0 = settle(sim, 1, args.settle)
    s0 = fiber_stretch_per_tet(sim, P0, sel_idx)

    # active
    sim.set_fiber_activation(mask.astype(np.float32) * float(args.act))
    P1 = settle(sim, 1, args.settle)
    s1 = fiber_stretch_per_tet(sim, P1, sel_idx)

    # metrics on the selected muscle's vertices
    sel_verts = np.unique(sim.tet_np[sel_idx])
    disp = np.linalg.norm((P1 - P0)[sel_verts], axis=1)
    print(f"[RESULT] mode={args.mode} muscle={args.muscle}")
    print(f"   biceps verts={sel_verts.shape[0]}  mean_disp={disp.mean()*1000:.3f}mm  "
          f"max_disp={disp.max()*1000:.3f}mm")
    print(f"   fiber_stretch passive: mean={s0.mean():.4f}  active: mean={s1.mean():.4f}  "
          f"shrink={(s0.mean()-s1.mean())/s0.mean()*100:.3f}%")
    print(f"   (target_stretch at act={args.act}: {1.0 - args.act*args.contraction_ratio:.3f} "
          f"for belly tendonmask=0)")
    verdict = "CONTRACTED" if disp.max() * 1000 > 1.0 else "NO-DEFORM"
    print(f"[VERDICT] mode={args.mode} -> {verdict} (max_disp={disp.max()*1000:.2f}mm, "
          f"fiber shrink={(s0.mean()-s1.mean())/s0.mean()*100:.2f}%)")

    # colored USD on the biceps surface only (compact to its tets)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)
    biceps_surf = vectorized_surface_tris(sim.tet_np[sel_idx]).reshape(-1).astype(np.int32)
    write_colored_usd(args.out, biceps_surf, P0, P1, sel_verts, args.out)
    np.savez(args.out.replace(".usd", ".npz"), P0=P0, P1=P1,
             sel_verts=sel_verts, sel_idx=sel_idx, s0=s0, s1=s1)


if __name__ == "__main__":
    main()
