"""Emil solver-vs-LBS muscle comparison — jet heatmap of surface deviation,
same scheme as tools/humanV2_ablation_dist.py.

Two muscle deformations driven by the SAME Emil skeleton animation:
  * SOLVER  — mg tet-muscle physics surface, output/emil/emil.anim.usd /char/sim/muscle
  * LBS     — Emil's built-in skinned surface, baked by tools.emil_lbs_bake -> npz

Topologies differ (solver 66k verts vs LBS 90k), so per-vertex correspondence is
impossible; instead every solver vertex is colored by its nearest-point distance
to the LBS surface (jet, 0..p99 mm). Hot = where the physics muscle departs from
the linear-blend-skinning shape (volume-preserving bulge vs LBS collapse).

Writes:
  * output/emil/emil_cmp_dist.usdc — /emil_cmp/solver (jet dist heatmap, animated)
                                      + /emil_cmp/lbs (grey reference surface)
  * output/emil/colorbar_emil_dist.png — jet mm legend
  * output/emil/emil_volume.csv — per-frame muscle volume (solver vs LBS), for a
    separate plotting script (never plot in a sim/postproc-dump script).

Run: PYTHONPATH=src .venv/bin/python -m tools.emil_lbs_compare
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.spatial import cKDTree
from pxr import Usd, UsdGeom, UsdSkel, Sdf, Vt

SOLVER_USD = "output/emil/emil.anim.usd"
SOLVER_PRIM = "/char/sim/muscle"
BONE_ROOT = "/char/bone"                       # UsdSkel SkelRoot in the solver USD
BONE_MESH = "/char/bone/SKM_skeleton_upper_body"
BONE_SKEL = "/char/bone/Skel"
LBS_NPZ = "output/emil/emil_lbs.npz"
OUT = "output/emil/emil_cmp_dist.usdc"
CBAR = "output/emil/colorbar_emil_dist.png"
VOLCSV = "output/emil/emil_volume.csv"

STOPS = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], np.float32)  # jet
SX = np.linspace(0, 1, len(STOPS))


def jet(t):
    t = np.clip(t, 0, 1)
    return np.stack([np.interp(t, SX, STOPS[:, i]) for i in range(3)], 1).astype(np.float32)


def tri_volume(pts, tris):
    """Closed-surface signed volume via divergence theorem; abs handles winding."""
    v0, v1, v2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    return abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


# ---- solver surface (world == local, net xform identity) ----
s = Usd.Stage.Open(SOLVER_USD)
sm = UsdGeom.Mesh(s.GetPrimAtPath(SOLVER_PRIM))
t0, t1 = int(s.GetStartTimeCode()), int(s.GetEndTimeCode())
frames = list(range(t0, t1 + 1))
pa = sm.GetPointsAttr()
sfc = np.asarray(sm.GetFaceVertexCountsAttr().Get(frames[0]), np.int32)
sfi = np.asarray(sm.GetFaceVertexIndicesAttr().Get(frames[0]), np.int32)
assert (sfc == 3).all(), "solver surface not all-tris"
solver_tris = sfi.reshape(-1, 3)
solver_pts = np.stack([np.asarray(pa.Get(f), np.float32) for f in frames])
NVs = solver_pts.shape[1]
print(f"solver: frames={len(frames)} NV={NVs} tris={len(solver_tris)}")

# ---- LBS baked surface ----
z = np.load(LBS_NPZ)
lbs_pts, lbs_tris, lbs_frames = z["points"], z["tris"], z["frames"]
assert list(lbs_frames) == frames, f"frame mismatch: {lbs_frames[0]}..{lbs_frames[-1]} vs {frames[0]}..{frames[-1]}"
print(f"lbs:    frames={len(lbs_frames)} NV={lbs_pts.shape[1]} tris={len(lbs_tris)}")

# ---- per-frame nearest-point distance (solver vert -> LBS surface verts) ----
dist = np.empty((len(frames), NVs), np.float32)
vol = np.empty((len(frames), 2), np.float64)          # [solver, lbs]
for i in range(len(frames)):
    d, _ = cKDTree(lbs_pts[i]).query(solver_pts[i], workers=-1)
    dist[i] = d
    vol[i, 0] = tri_volume(solver_pts[i], solver_tris)
    vol[i, 1] = tri_volume(lbs_pts[i], lbs_tris)
VMAX = float(np.percentile(dist, 99))
print(f"dist mm: max={dist.max()*1e3:.2f} mean={dist.mean()*1e3:.3f} "
      f"p50={np.percentile(dist,50)*1e3:.2f} p99(VMAX)={VMAX*1e3:.2f}")

# ---- volume CSV (dump only; plot in a separate script) ----
with open(VOLCSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame", "vol_solver_m3", "vol_lbs_m3", "ratio_solver_over_lbs"])
    for i, f in enumerate(frames):
        w.writerow([f, f"{vol[i,0]:.6e}", f"{vol[i,1]:.6e}", f"{vol[i,0]/vol[i,1]:.4f}"])
print(f"WROTE {VOLCSV}  solver vol f{frames[0]}->f{frames[-1]}: "
      f"{vol[0,0]*1e3:.2f}L->{vol[-1,0]*1e3:.2f}L  lbs: {vol[0,1]*1e3:.2f}L->{vol[-1,1]*1e3:.2f}L")

# ---- bake the skeleton (UsdSkel-skinned in the solver USD -> per-frame points so
#      ANY viewer shows bone + muscle moving together; gives the attach context) ----
_cache = UsdSkel.Cache()
_cache.Populate(UsdSkel.Root(s.GetPrimAtPath(BONE_ROOT)), Usd.PrimDefaultPredicate)
_bp = s.GetPrimAtPath(BONE_MESH)
_sq = _cache.GetSkinningQuery(_bp)
_skq = _cache.GetSkelQuery(UsdSkel.Skeleton(s.GetPrimAtPath(BONE_SKEL)))
_brest = np.asarray(UsdGeom.PointBased(_bp).GetPointsAttr().Get(), np.float32)
_bM = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(_bp)).reshape(4, 4)
_bmesh = UsdGeom.Mesh(_bp)
bone_fc = np.asarray(_bmesh.GetFaceVertexCountsAttr().Get(), np.int32)
bone_fi = np.asarray(_bmesh.GetFaceVertexIndicesAttr().Get(), np.int32)
bone_pts = np.empty((len(frames), len(_brest), 3), np.float32)
for i, f in enumerate(frames):
    _pts = Vt.Vec3fArray.FromNumpy(_brest.copy())
    _sq.ComputeSkinnedPoints(_skq.ComputeSkinningTransforms(f), _pts, f)
    a = np.asarray(_pts, np.float32)
    bone_pts[i] = (np.c_[a, np.ones(len(a), np.float32)] @ _bM)[:, :3]
print(f"bone: baked {len(_brest)} verts x {len(frames)} frames")

# ---- heatmap stage ----
stage = Usd.Stage.CreateNew(OUT)
stage.SetStartTimeCode(t0); stage.SetEndTimeCode(t1)
stage.SetTimeCodesPerSecond(s.GetTimeCodesPerSecond())
UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(s))
root = UsdGeom.Xform.Define(stage, "/emil_cmp")
stage.SetDefaultPrim(root.GetPrim())

# solver mesh, jet-colored by distance to LBS
heat = UsdGeom.Mesh.Define(stage, "/emil_cmp/solver")
heat.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(sfc))
heat.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(sfi))
hp = heat.GetPointsAttr()
pv = UsdGeom.PrimvarsAPI(heat.GetPrim()).CreatePrimvar(
    "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
for i, f in enumerate(frames):
    hp.Set(Vt.Vec3fArray.FromNumpy(solver_pts[i]), Usd.TimeCode(f))
    cols = jet(dist[i] / VMAX if VMAX > 0 else dist[i])
    pv.GetAttr().Set(Vt.Vec3fArray.FromNumpy(cols), Usd.TimeCode(f))

# LBS reference surface, constant grey (toggle visibility to overlay)
ref = UsdGeom.Mesh.Define(stage, "/emil_cmp/lbs")
ref.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(lbs_tris), 3, np.int32)))
ref.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(lbs_tris.reshape(-1)))
rp = ref.GetPointsAttr()
for i, f in enumerate(frames):
    rp.Set(Vt.Vec3fArray.FromNumpy(lbs_pts[i]), Usd.TimeCode(f))
UsdGeom.PrimvarsAPI(ref.GetPrim()).CreatePrimvar(
    "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant
).GetAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([[0.6, 0.6, 0.6]], np.float32)))
UsdGeom.Imageable(ref.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)  # off by default

# skeleton, baked per-frame points, constant dark-grey (attach context)
bone = UsdGeom.Mesh.Define(stage, "/emil_cmp/bone")
bone.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(bone_fc))
bone.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(bone_fi))
bpp = bone.GetPointsAttr()
for i, f in enumerate(frames):
    bpp.Set(Vt.Vec3fArray.FromNumpy(bone_pts[i]), Usd.TimeCode(f))
UsdGeom.PrimvarsAPI(bone.GetPrim()).CreatePrimvar(
    "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant
).GetAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([[0.35, 0.35, 0.4]], np.float32)))

stage.GetRootLayer().Save()
print(f"WROTE {OUT}  (/emil_cmp/solver jet heatmap + /emil_cmp/lbs grey ref + /emil_cmp/bone)")

# ---- colorbar ----
vmm = round(VMAX * 1e3, 1)
cmap = LinearSegmentedColormap.from_list("vsjet", list(map(tuple, STOPS)), N=256)
sm2 = ScalarMappable(norm=Normalize(0, vmm), cmap=cmap)
plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
step = max(1, round(vmm / 6))
fig = plt.figure(figsize=(4.8, 1.3))
cax = fig.add_axes([0.06, 0.55, 0.88, 0.22])
cb = fig.colorbar(sm2, cax=cax, orientation="horizontal", extend="max",
                  ticks=list(np.arange(0, vmm + 1e-6, step)))
cb.set_label("Solver muscle distance to LBS surface — mm")
fig.savefig(CBAR, dpi=300, bbox_inches="tight", transparent=True)
print(f"WROTE {CBAR}  (VMAX={vmm}mm)")
