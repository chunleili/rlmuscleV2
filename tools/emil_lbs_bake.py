"""Bake Emil's built-in LBS muscle (UsdSkel skinned surface) to a world-space
mesh sequence — the artist-rig baseline the physics solver is compared against.

Emil.usd ships two muscle representations driven by the SAME skeleton animation
(``/char/muscle/ANIM_Emil_liftWeights``):
  * ``SKM_Emil_upperBodyMuscles`` — a skinned SURFACE mesh (linear blend skinning)
  * ``Tet_Emil_upperBodyMuscles`` — the tet the mg solver simulates

This bakes the LBS surface: per frame, resolve the skeleton's skinning transforms,
apply ``ComputeSkinnedPoints`` (mesh-local), then push through the prim's static
local-to-world so the points land in the SAME world metres as the solver output
(``output/emil/emil.anim.usd`` ``/char/sim/muscle``). Writes an npz of
(frames, NV, 3) world points + face triangles for the compare step.

Run: PYTHONPATH=src .venv/bin/python -m tools.emil_lbs_bake
"""
import numpy as np
from pxr import Usd, UsdGeom, UsdSkel, Vt

ASSET = "data/Emil/model/Emil.usd"
SKM = "/char/muscle/SKM_Emil_upperBodyMuscles"
SKEL = "/char/muscle/Skel"
ROOT = "/char/muscle"
OUT = "output/emil/emil_lbs.npz"
F0, F1 = 1, 200                                  # match solver output frame range

stage = Usd.Stage.Open(ASSET)
cache = UsdSkel.Cache()
cache.Populate(UsdSkel.Root(stage.GetPrimAtPath(ROOT)), Usd.PrimDefaultPredicate)
skm_prim = stage.GetPrimAtPath(SKM)
skin_q = cache.GetSkinningQuery(skm_prim)
skel_q = cache.GetSkelQuery(UsdSkel.Skeleton(stage.GetPrimAtPath(SKEL)))
assert skin_q and skel_q, "skinning/skel query failed to resolve"

rest = np.asarray(UsdGeom.PointBased(skm_prim).GetPointsAttr().Get(), np.float32)
NV = len(rest)

# prim-local -> world is static (only the skeleton animates); compute once.
M = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(skm_prim)).reshape(4, 4)

# surface triangles (SKM is already all-tris: faceVertexCounts all == 3)
mesh = UsdGeom.Mesh(skm_prim)
counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), np.int32)
idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int32)
assert (counts == 3).all(), f"SKM has non-tri faces: {np.unique(counts)}"
tris = idx.reshape(-1, 3)

frames = list(range(F0, F1 + 1))
world = np.empty((len(frames), NV, 3), np.float32)
for i, f in enumerate(frames):
    xforms = skel_q.ComputeSkinningTransforms(f)
    pts = Vt.Vec3fArray.FromNumpy(rest.copy())
    assert skin_q.ComputeSkinnedPoints(xforms, pts, f), f"skinning failed @ {f}"
    a = np.asarray(pts, np.float32)
    world[i] = (np.c_[a, np.ones(len(a), np.float32)] @ M)[:, :3]

np.savez_compressed(OUT, points=world, tris=tris, frames=np.array(frames, np.int32))
print(f"WROTE {OUT}  frames={len(frames)} NV={NV} tris={len(tris)}")
print(f"  f{F0} bbox {world[0].min(0).round(3)}..{world[0].max(0).round(3)}")
print(f"  f{F1} bbox {world[-1].min(0).round(3)}..{world[-1].max(0).round(3)}")
