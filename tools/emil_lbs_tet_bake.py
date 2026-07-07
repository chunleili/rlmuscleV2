"""Bake a SMOOTH LBS target for the Emil solver TET (74k verts) — the driving
target for target_mode="lbs".

The default target_mode="mesh" rigidly binds each muscle vert to ONE bone-surface
triangle, so verts straddling a joint follow different bones and shear the tets
between them (elbow crease inversions). This bakes a smooth alternative: each tet
vert rides the LBS-skinned muscle SURFACE via DISPLACEMENT TRANSFER — it follows
the nearest SKM (skinned surface) vertex's linear-blend-skinning displacement.
LBS blends bones smoothly, so the target stays continuous across joints; the ARAP
solver then adds volume back on top of it.

The tet carries no skin weights, so we transfer displacement (not weights):
  tet_target[f] = tet_rest_world + (SKM_skinned[f] - SKM_rest)[nearest SKM vert]
using the SKM LBS surface already baked by tools.emil_lbs_bake (emil_lbs.npz,
world metres — same space as the solver). Output emil_lbs_tet.npz has (F, NV, 3)
world points in the Emil.usd TET vertex order (== the solver's tet order).

Run: PYTHONPATH=src .venv/bin/python -m tools.emil_lbs_tet_bake
"""
import numpy as np
from scipy.spatial import cKDTree
from pxr import Usd, UsdGeom

ASSET = "data/Emil/model/Emil.usd"
TET = "/char/muscle/Tet_Emil_upperBodyMuscles"
SKM = "/char/muscle/SKM_Emil_upperBodyMuscles"
LBS_NPZ = "output/emil/emil_lbs.npz"          # SKM skinned world per frame (emil_lbs_bake)
OUT = "output/emil/emil_lbs_tet.npz"


def world_points(stage, path):
    prim = stage.GetPrimAtPath(path)
    p = np.asarray(UsdGeom.PointBased(prim).GetPointsAttr().Get(), np.float32)
    M = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).reshape(4, 4)
    return (np.c_[p, np.ones(len(p), np.float32)] @ M)[:, :3]


stage = Usd.Stage.Open(ASSET)
tet_rest = world_points(stage, TET)           # (NV,3) world metres
skm_rest = world_points(stage, SKM)           # (M,3)  world metres
NV = len(tet_rest)

z = np.load(LBS_NPZ)
skm_lbs, frames = z["points"], z["frames"]    # (F,M,3) skinned world, frame ids
assert skm_lbs.shape[1] == len(skm_rest), f"SKM count mismatch {skm_lbs.shape[1]} vs {len(skm_rest)}"

# nearest SKM vertex per tet vertex (world rest space, same for tet & SKM)
idx = cKDTree(skm_rest).query(tet_rest, workers=-1)[1]

# displacement transfer: tet follows nearest SKM vert's LBS displacement from rest
disp = skm_lbs - skm_rest[None]               # (F,M,3)
tet_target = tet_rest[None] + disp[:, idx]    # (F,NV,3) world metres

np.savez_compressed(OUT, points=tet_target.astype(np.float32),
                    frames=np.asarray(frames, np.int32))
print(f"WROTE {OUT}  frames={len(frames)} NV={NV}")
print(f"  f{frames[0]} bbox {tet_target[0].min(0).round(3)}..{tet_target[0].max(0).round(3)}")
print(f"  f{frames[-1]} bbox {tet_target[-1].min(0).round(3)}..{tet_target[-1].max(0).round(3)}")
print(f"  max per-vert travel f0->f-1: {np.linalg.norm(tet_target[-1]-tet_target[0],axis=1).max()*1e3:.0f}mm")
