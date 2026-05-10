"""Export the per-muscle coarse mesh to a viewable USD/OBJ.

Writes both:
  - <out>.usd  with a UsdGeom.TetMesh prim (visible in usdview)
  - <out>_surface.obj  with the boundary triangles only (for any 3D viewer)

Usage:
    PYTHONPATH=src .venv/bin/python tools/coarse_to_usd.py \
        --in data/Human/coarse_per_muscle.npz \
        --out data/Human/coarse_per_muscle
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from muscle.mesh_io import build_surface_tris  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="data/Human/coarse_per_muscle.npz")
    p.add_argument("--out", default="data/Human/coarse_per_muscle")
    args = p.parse_args()

    data = np.load(args.inp, allow_pickle=True)
    pos = np.asarray(data["positions"], dtype=np.float32)
    tets = np.asarray(data["tet_indices"], dtype=np.int32)
    print(f"Loaded {pos.shape[0]} verts, {tets.shape[0]} tets from {args.inp}")

    surface_tris = build_surface_tris(tets, pos)
    print(f"Surface boundary: {surface_tris.shape[0]} triangles")

    obj_path = Path(args.out + "_surface.obj")
    with open(obj_path, "w") as f:
        for v in pos:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in surface_tris:
            # OBJ uses 1-based indices.
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
    print(f"Wrote {obj_path}")

    from pxr import Usd, UsdGeom, Vt
    usd_path = args.out + ".usd"
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    root = UsdGeom.Xform.Define(stage, "/coarse")
    tm_prim = stage.DefinePrim("/coarse/coarse_Tet", "TetMesh")
    tm = UsdGeom.TetMesh(tm_prim)
    tm.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pos.astype(np.float32)))
    tm.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray.FromNumpy(tets.astype(np.int32)))
    # Also write a Mesh prim with the surface tris so usdview shows something
    # (TetMesh has limited renderer support).
    surf_prim = stage.DefinePrim("/coarse/coarse_surface", "Mesh")
    surf = UsdGeom.Mesh(surf_prim)
    surf.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pos.astype(np.float32)))
    surf.GetFaceVertexIndicesAttr().Set(
        Vt.IntArray.FromNumpy(surface_tris.flatten().astype(np.int32)))
    surf.GetFaceVertexCountsAttr().Set(
        Vt.IntArray([3] * surface_tris.shape[0]))
    stage.GetRootLayer().Save()
    print(f"Wrote {usd_path}  (open with usdview)")


if __name__ == "__main__":
    main()
