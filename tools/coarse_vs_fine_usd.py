"""Build a side-by-side USD comparing the fine and coarse muscle meshes.

  /world/fine_left   = fine muscle surfaces, translated -0.6 m on X
  /world/coarse_right = coarse muscle surfaces, translated +0.0 m on X
  /world/overlay      = both at original position (semi-transparent compare)

Usage:
    PYTHONPATH=src .venv/bin/python tools/coarse_vs_fine_usd.py \
        --coarse data/Human/coarse_per_muscle.npz \
        --fine data/Human/Human.usd \
        --out data/Human/coarse_vs_fine.usd
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from muscle.mesh_io import build_surface_tris, load_mesh  # noqa: E402


def add_mesh(stage, path, pos, tris, color=(0.7, 0.7, 0.7), opacity=1.0,
             translate=(0.0, 0.0, 0.0)):
    from pxr import UsdGeom, Vt, Gf
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    mesh_prim = stage.DefinePrim(path + "/Mesh", "Mesh")
    mesh = UsdGeom.Mesh(mesh_prim)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pos.astype(np.float32)))
    mesh.GetFaceVertexIndicesAttr().Set(
        Vt.IntArray.FromNumpy(tris.flatten().astype(np.int32)))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * tris.shape[0]))
    mesh.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    if opacity < 1.0:
        mesh.CreateDisplayOpacityAttr().Set(Vt.FloatArray([float(opacity)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coarse", default="data/Human/coarse_per_muscle.npz")
    p.add_argument("--fine", default="data/Human/Human.usd")
    p.add_argument("--out", default="data/Human/coarse_vs_fine.usd")
    args = p.parse_args()

    cdata = np.load(args.coarse, allow_pickle=True)
    cpos = np.asarray(cdata["positions"], dtype=np.float32)
    ctets = np.asarray(cdata["tet_indices"], dtype=np.int32)
    ctris = build_surface_tris(ctets, cpos)
    print(f"coarse: {cpos.shape[0]}v / {ctets.shape[0]}t / {ctris.shape[0]} surface tris")

    fpos, ftets, _, _, _ = load_mesh(args.fine, prim_root="/Human/muscle")
    ftris = build_surface_tris(ftets, fpos)
    print(f"fine:   {fpos.shape[0]}v / {ftets.shape[0]}t / {ftris.shape[0]} surface tris")

    from pxr import Usd, UsdGeom, Sdf
    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/world")

    # Side-by-side: fine on left (x-0.6), coarse on right (x+0.0)
    add_mesh(stage, "/world/fine_left", fpos, ftris,
             color=(0.85, 0.45, 0.40), translate=(-0.6, 0.0, 0.0))
    add_mesh(stage, "/world/coarse_right", cpos, ctris,
             color=(0.40, 0.70, 0.85), translate=(0.0, 0.0, 0.0))

    # Overlay: both at original location, fine semi-transparent
    add_mesh(stage, "/world/overlay_fine", fpos, ftris,
             color=(0.85, 0.45, 0.40), opacity=0.25,
             translate=(0.6, 0.0, 0.0))
    add_mesh(stage, "/world/overlay_coarse", cpos, ctris,
             color=(0.40, 0.70, 0.85), opacity=1.0,
             translate=(0.6, 0.0, 0.0))

    stage.GetRootLayer().Save()
    print(f"Wrote {args.out}")
    print("Open with: usdview", args.out)


if __name__ == "__main__":
    main()
