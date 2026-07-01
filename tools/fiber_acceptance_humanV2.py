"""humanV2 fiber contraction ACCEPTANCE render — time-sampled USD.

Applies the fix (tendon-only hard pin + freed belly + fiber constraint), holds
the skeleton static at frame 1, ramps biceps activation 0->1 across N output
frames, and writes ONE time-sampled USD:

  * /muscle : full muscle surface, points per frame, displayColor = per-vertex
              displacement heatmap vs the relaxed (act=0) shape (mm)
  * /bones  : static skeleton (grey) for spatial reference

Open in usdview and scrub the timeline: the biceps brachii belly should visibly
bulge / shorten and light up red as activation ramps in.

    PYTHONPATH=src python tools/fiber_acceptance_humanV2.py \
        --muscle bicepsBrachialis --frames 60 --fine-iters 80 \
        --out output/humanV2_fiber_accept.usd
"""
import argparse
import os
import numpy as np
import warp as wp

from muscle.mg.mg import load_config
from example.humanV2 import build_humanV2_sim
from muscle.util.bone_source import make_bone_source


def jet(x):
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], 1).astype(np.float32)


def compact(surf_flat, n_verts):
    used = np.unique(surf_flat)
    remap = np.full(n_verts, -1, np.int64)
    remap[used] = np.arange(used.shape[0])
    local = remap[surf_flat].astype(np.int32)
    return used, local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/humanV2/config/humanV2.json")
    ap.add_argument("--muscle", default="bicepsBrachialis", help="muscle to ACTIVATE")
    ap.add_argument("--view-muscle", default=None,
                    help="muscle set for the PNG metric/panels (default = --muscle); "
                         "e.g. activate L only but view both arms to inspect asymmetry")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=48, help="activation ramp output frames")
    ap.add_argument("--substeps-per-frame", type=int, default=3,
                    help="step_frame calls per output frame (converge to that activation)")
    ap.add_argument("--settle-rest", type=int, default=25, help="passive settle before ramp")
    ap.add_argument("--fine-iters", type=int, default=80)
    ap.add_argument("--fiber-stiffness", type=float, default=1e8)
    ap.add_argument("--contraction-ratio", type=float, default=0.6)
    ap.add_argument("--pin-region", default=None,
                    help="muscleToBonePin region override; default None keeps humanV2.json "
                         "(pinmask -> 0 hard pins). tendon hard-pins over-constrain the belly.")
    ap.add_argument("--pin-threshold", type=float, default=0.5)
    ap.add_argument("--quasi-static", action="store_true",
                    help="drop inertia -> static equilibrium per frame (no overshoot/drift)")
    ap.add_argument("--belly-stiffness", type=float, default=0.0,
                    help="belly_attach stiffness; 0 frees the belly (static-bone acceptance)")
    ap.add_argument("--vmax-mm", type=float, default=None,
                    help="fixed color scale max (mm); default = auto from the sequence")
    ap.add_argument("--out", default="output/humanV2_fiber_accept.usd")
    args = ap.parse_args()

    wp.init()
    cfg = load_config(args.config)
    cfg["start_frame"] = 1
    cfg["end_frame"] = 1
    cfg["use_fascia"] = False
    cfg["use_multigrid"] = False
    cfg["use_jacobi"] = True
    if args.quasi_static:
        cfg["quasi_static"] = True
    cfg["fine_iterations"] = int(args.fine_iters)
    for c in cfg.get("constraints", []):
        if c.get("name") == "belly_attach":
            c["stiffness"] = float(args.belly_stiffness)
        if c.get("name") == "muscleToBonePin" and args.pin_region is not None:
            c["region"] = args.pin_region
            if args.pin_region == "tendonmask":
                c["tendon_pin_threshold"] = float(args.pin_threshold)
            else:
                c["pin_threshold"] = float(args.pin_threshold)
    cfg.setdefault("constraints", []).append(dict(
        type="fiber", name="fiber", enable=True, fiber_model="fibernorm",
        stiffness=args.fiber_stiffness, dampingratio=0.1,
        contraction_ratio=args.contraction_ratio))
    demo = cfg.setdefault("demo", {})
    demo["device"] = args.device
    demo.setdefault("collision", {})["enabled"] = False
    demo["activate"] = None

    sim = build_humanV2_sim(cfg)
    print(f"[accept] use_jacobi={sim.use_jacobi} use_fiber={sim.use_fiber} "
          f"fine_iters={sim.fine_iters} k_fib={sim.fiber_stiffness:g} "
          f"ratio={sim.contraction_ratio} belly_k={args.belly_stiffness:g}")

    # ACTIVATION mask (which muscle gets driven) vs VIEW mask (which verts the PNG
    # metric spans). Default view = the activated muscle; pass --view-muscle to see
    # a larger set (e.g. activate L only, view both arms to inspect asymmetry).
    mask = sim.tet_mask_for_muscles(args.muscle)
    view_name = args.view_muscle or args.muscle
    view_mask = sim.tet_mask_for_muscles(view_name)
    sel_verts = np.unique(sim.tet_np[np.where(view_mask)[0]])
    print(f"[accept] activate '{args.muscle}': {int(mask.sum())} tets  |  "
          f"view '{view_name}': {int(view_mask.sum())} tets, {sel_verts.shape[0]} verts")

    # passive rest reference
    sim.set_fiber_activation(np.zeros(mask.shape[0], np.float32))
    for _ in range(args.settle_rest):
        sim.step_frame(1)
    wp.synchronize()
    P_rest = sim.positions_np().copy()

    # ramp, record every frame
    N = args.frames
    frames_pos = [P_rest.copy()]
    acts = [0.0]
    for f in range(1, N + 1):
        a = f / N
        sim.set_fiber_activation(mask.astype(np.float32) * a)
        for _ in range(args.substeps_per_frame):
            sim.step_frame(1)
        wp.synchronize()
        frames_pos.append(sim.positions_np().copy())
        acts.append(a)
        if f % 10 == 0 or f == N:
            d = np.linalg.norm((frames_pos[-1] - P_rest)[sel_verts], axis=1)
            print(f"  frame {f:3d}  act={a:.2f}  biceps max_disp={d.max()*1000:.2f}mm")

    # color scale
    all_d = np.concatenate([np.linalg.norm((P - P_rest)[sel_verts], axis=1)
                            for P in frames_pos])
    vmax = float(args.vmax_mm) / 1000.0 if args.vmax_mm else max(all_d.max(), 1e-6)
    print(f"[accept] color vmax = {vmax*1000:.2f}mm")

    # write time-sampled USD
    from pxr import Usd, UsdGeom, Vt, Sdf
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)
    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(N)

    m_used, m_local = compact(sim.surf_flat, sim.NV)
    mmesh = UsdGeom.Mesh.Define(stage, "/muscle")
    mmesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (m_local.shape[0] // 3)))
    mmesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(m_local))
    pts_attr = mmesh.CreatePointsAttr()
    col_pv = UsdGeom.PrimvarsAPI(mmesh).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
    for t, P in enumerate(frames_pos):
        pts_attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(P[m_used], np.float32)), t)
        disp = np.linalg.norm((P - P_rest)[m_used], axis=1)
        col_pv.Set(Vt.Vec3fArray.FromNumpy(jet(disp / vmax)), t)

    # static bones grey reference
    bs = make_bone_source(cfg, 1)
    bverts = bs.positions(1)
    btris = bs.tris.reshape(-1).astype(np.int32)
    b_used, b_local = compact(btris, bverts.shape[0])
    bmesh = UsdGeom.Mesh.Define(stage, "/bones")
    bmesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (b_local.shape[0] // 3)))
    bmesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(b_local))
    bmesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(
        np.ascontiguousarray(bverts[b_used], np.float32)))
    UsdGeom.PrimvarsAPI(bmesh).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant
    ).Set(Vt.Vec3fArray.FromNumpy(np.array([[0.8, 0.78, 0.72]], np.float32)))

    stage.GetRootLayer().Save()
    d_final = np.linalg.norm((frames_pos[-1] - P_rest)[sel_verts], axis=1)
    print(f"[accept] wrote {args.out}  frames=0..{N}  "
          f"final biceps max_disp={d_final.max()*1000:.2f}mm mean={d_final.mean()*1000:.2f}mm")

    # quick 3-panel PNG (rest / mid / final) so acceptance needs no usdview
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        idxs = [0, N // 2, N]
        labels = ["rest (act=0)", f"mid (act={acts[N//2]:.2f})", f"final (act={acts[N]:.2f})"]
        fig = plt.figure(figsize=(18, 6))
        pv0 = frames_pos[0][sel_verts]
        c = pv0.mean(0)
        r = float(np.abs(pv0 - c).max())
        for k, ti in enumerate(idxs):
            P = frames_pos[ti]
            d = np.linalg.norm((P - P_rest)[sel_verts], axis=1) * 1000.0
            ax = fig.add_subplot(1, 3, k + 1, projection="3d")
            pv = P[sel_verts]
            sc = ax.scatter(pv[:, 0], pv[:, 1], pv[:, 2], c=d, cmap="jet",
                            vmin=0, vmax=vmax * 1000, s=3, depthshade=False)
            ax.set_title(f"{labels[k]}\nmax={d.max():.2f}mm mean={d.mean():.2f}mm", fontsize=11)
            ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev=12, azim=-70)
        fig.colorbar(sc, ax=fig.axes, shrink=0.6, pad=0.02).set_label("displacement vs rest (mm)")
        fig.suptitle(f"humanV2 {args.muscle}: fiber activation ramp (fixed config, static bone)",
                     fontsize=13)
        png = args.out.rsplit(".", 1)[0] + ".png"
        fig.savefig(png, dpi=120, bbox_inches="tight")
        print(f"[accept] wrote {png}")
    except Exception as e:
        print(f"[accept] PNG skipped: {e}")


if __name__ == "__main__":
    main()
