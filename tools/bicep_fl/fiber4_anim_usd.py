"""Animated USD of the bicep contracting as activation ramps 0 -> 1.

Reads the per-activation equilibrium meshes dumped by fiber4.py (fiber4_anim.npz)
and packs them into ONE timeSampled USD: each fiber model is a /fiber/<model>
Mesh laid out along X, its points + turbo displayColor keyframed per activation
level (frame f = activation acts[f]). Colored by per-vertex displacement from the
REST reference (turbo), so the bulge/shortening grows as activation rises.

nofiber is omitted on purpose: with the fiber constraint deleted (and no attach,
only the proximal tendon pin) activation does nothing, so nofiber == rest for
every a. rest is shown once as the static grey reference; only the three fiber
models animate.

Output goes to <repo>/output/bicep/fiber4_anim.usd.
    .venv/bin/python tools/bicep_fl/fiber4_anim_usd.py [--out ...] [--fps 6]
"""
import argparse
import os

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pxr import Usd, UsdGeom, Vt, Sdf

HERE = os.path.dirname(__file__)
# generated artifacts live in <repo>/output/bicep, not next to the script
OUT = os.path.join(HERE, "..", "..", "output", "bicep")
os.makedirs(OUT, exist_ok=True)
# rest = static reference; the rest animate. nofiber excluded (== rest).
ANIM = ["plain", "aniso_arap", "millard"]
LAYOUT = ["rest"] + ANIM

# jet colormap (blue->cyan->green->yellow->red), matching output/emil colorbar
# and tools/bicep_fl/bicep_colorbar.py. Mesh is in metres; VMAX is in mm.
STOPS = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], np.float32)
JET = LinearSegmentedColormap.from_list("vsjet", list(map(tuple, STOPS)), N=256)
VMAX_MM = 50.0                       # displacement (mm) mapped to full jet range


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(OUT, "fiber4_anim.npz"))
    ap.add_argument("--out", default=os.path.join(OUT, "fiber4_anim.usd"))
    ap.add_argument("--fps", type=float, default=20.0)
    # linearly interpolate --sub in-between frames per activation step, so the
    # contraction plays back slow and smooth (no resim). --sub 1 = raw equilibria.
    # default: (11-1)*10+1 = 101 frames @ 20fps = 5.0s
    ap.add_argument("--sub", type=int, default=10)
    ap.add_argument("--vmax", type=float, default=VMAX_MM,
                    help="displacement (mm) at the top of the jet scale")
    args = ap.parse_args()

    z = np.load(args.npz)
    faces = z["faces"].astype(np.int32)
    rest = z["rest"].astype(np.float32)
    pos = {m: z[f"pos_{m}"].astype(np.float32) for m in ANIM}   # (K, NV, 3)
    K = pos[ANIM[0]].shape[0]
    acts = z["acts"].astype(np.float32) if "acts" in z else np.linspace(0, 1, K)

    # upsample the K equilibria to F = (K-1)*sub + 1 interpolated frames
    sub = max(1, args.sub)
    src = np.linspace(0.0, K - 1.0, (K - 1) * sub + 1) if (sub > 1 and K > 1) \
        else np.arange(K, dtype=float)
    lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, K - 1)
    w = src - lo
    for m in ANIM:
        P = pos[m]
        pos[m] = ((1 - w[:, None, None]) * P[lo] + w[:, None, None] * P[hi]).astype(np.float32)
    # per-frame activation (same interp param), keyframed as a USD attribute
    frame_act = ((1 - w) * acts[lo] + w * acts[hi]).astype(np.float32)
    F = pos[ANIM[0]].shape[0]

    # colour scale: displacement from rest (mm), mapped to jet over 0..VMAX_MM.
    # Fixed vmax (shared with the standalone colorbar) so frames/models compare.
    ap_vmax = args.vmax
    disp = {m: 1000.0 * np.linalg.norm(pos[m] - rest[None], axis=2) for m in ANIM}  # (F, NV) mm

    span = rest.max(0) - rest.min(0)
    step = span[0] * 1.35                       # lay models out along X

    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetStartTimeCode(1)
    stage.SetEndTimeCode(F)
    stage.SetTimeCodesPerSecond(args.fps)
    root = UsdGeom.Xform.Define(stage, "/fiber")
    stage.SetDefaultPrim(root.GetPrim())

    # global activation ramp, keyframed on the root as a constant primvar so it
    # shows in the attribute panel for the whole rig (0 -> 1 over the animation).
    actRoot = UsdGeom.PrimvarsAPI(root.GetPrim()).CreatePrimvar(
        "activation", Sdf.ValueTypeNames.Float, UsdGeom.Tokens.constant)
    for f in range(F):
        actRoot.Set(float(frame_act[f]), Usd.TimeCode(f + 1))

    counts = Vt.IntArray([3] * len(faces))
    idx = Vt.IntArray(faces.reshape(-1).tolist())

    def jetcol(d):                       # d in mm -> jet rgb clipped at vmax
        return np.ascontiguousarray(
            JET(np.clip(d / ap_vmax, 0, 1))[:, :3].astype(np.float32))

    for i, m in enumerate(LAYOUT):
        mesh = UsdGeom.Mesh.Define(stage, f"/fiber/{m}")
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(idx)
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        ptsA = mesh.CreatePointsAttr()
        actA = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "activation", Sdf.ValueTypeNames.Float, UsdGeom.Tokens.constant)

        if m == "rest":
            # static grey reference: no time samples (holds for the whole range)
            P = rest.copy(); P[:, 0] += i * step
            ptsA.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(P)))
            grey = np.full((len(rest), 3), 0.6, np.float32)
            pv.Set(Vt.Vec3fArray.FromNumpy(grey))
            actA.Set(0.0)                                  # rest = unactivated
            continue

        # animated fiber model: keyframe points + colour + activation per frame
        for f in range(F):
            P = pos[m][f].copy(); P[:, 0] += i * step
            t = Usd.TimeCode(f + 1)
            actA.Set(float(frame_act[f]), t)
            ptsA.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(P)), t)
            pv.GetAttr().Set(Vt.Vec3fArray.FromNumpy(jetcol(disp[m][f])), t)

    stage.GetRootLayer().Save()
    print(f"wrote {args.out}  ({len(LAYOUT)} prims: rest + {ANIM}, "
          f"{F} frames @ {args.fps}fps = {(F-1)/args.fps:.2f}s "
          f"(sub={sub}), jet 0..{ap_vmax:.0f}mm)")


if __name__ == "__main__":
    main()
