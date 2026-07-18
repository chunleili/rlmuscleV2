"""Pack the four deformed biceps (a=1) + rest into one USD for usdview.

Each fiber model becomes a /fiber/<model> Mesh prim, offset along X and colored
by per-vertex displacement from the no-fiber baseline (turbo heatmap), matching
fiber4_mesh.png. Open in usdview and orbit to compare contract-and-bulge.

Output goes to <repo>/output/bicep/fiber4_compare.usd.
    .venv/bin/python tools/bicep_fl/plys_to_usd.py [--out ...]
"""
import argparse
import os
import numpy as np
import meshio
from matplotlib.colors import LinearSegmentedColormap
from pxr import Usd, UsdGeom, Vt, Gf, Sdf

HERE = os.path.dirname(__file__)
# generated artifacts live in <repo>/output/bicep, not next to the script
OUT = os.path.join(HERE, "..", "..", "output", "bicep")
os.makedirs(OUT, exist_ok=True)
MODELS = ["nofiber", "plain", "aniso_arap", "millard", "rest"]

# jet colormap + fixed scale, shared with fiber4_anim_usd.py and bicep_colorbar.py
STOPS = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], np.float32)
JET = LinearSegmentedColormap.from_list("vsjet", list(map(tuple, STOPS)), N=256)
VMAX_MM = 50.0                       # displacement (mm) mapped to full jet range


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "fiber4_compare.usd"))
    ap.add_argument("--vmax", type=float, default=VMAX_MM,
                    help="displacement (mm) at the top of the jet scale")
    args = ap.parse_args()

    meshes = {m: meshio.read(os.path.join(OUT, f"fiber4_{m}.ply")) for m in MODELS}
    faces = meshes["nofiber"].cells[0].data.astype(np.int32)
    base = meshes["nofiber"].points
    # displacement from the no-fiber baseline, in mm (mesh is in metres)
    disp = {m: 1000.0 * np.linalg.norm(meshes[m].points - base, axis=1) for m in MODELS}

    span = base.max(0) - base.min(0)
    step = span[0] * 1.35        # lay models out along X

    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = UsdGeom.Xform.Define(stage, "/fiber")

    counts = Vt.IntArray([3] * len(faces))
    idx = Vt.IntArray(faces.reshape(-1).tolist())

    for i, m in enumerate(MODELS):
        P = np.ascontiguousarray(meshes[m].points.astype(np.float32))
        P[:, 0] += i * step
        mesh = UsdGeom.Mesh.Define(stage, f"/fiber/{m}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P))
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(idx)
        cols = np.ascontiguousarray(JET(np.clip(disp[m] / args.vmax, 0, 1))[:, :3].astype(np.float32))
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(cols))

    stage.GetRootLayer().Save()
    print("wrote", args.out, f"({len(MODELS)} prims, jet 0..{args.vmax:.0f}mm)")


if __name__ == "__main__":
    main()
