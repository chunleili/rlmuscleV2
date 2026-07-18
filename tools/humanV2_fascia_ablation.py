"""Fascia ablation — muscle "散开" (no fascia) vs "聚拢" (squeeze fascia on).

Two runs driven by the SAME skeleton animation, IDENTICAL muscle topology:
  * OFF  — output/humanV2/fascia_off.usd  /root/muscle  (use_fascia=false)
  * ON   — output/humanV2/fascia_on.usd   /root/muscle  (fascia squeeze, the
           shell shrinks + glue drags the muscle belly inward => gather)

Pure post-processing (no simulation here). Emits, into output/humanV2/fascia:
  * fascia_ablation.usdc — /fascia_ab/off (grey, the scattered state) laid out
    beside /fascia_ab/on (jet heatmap = per-vertex |on-off| displacement, i.e.
    how far fascia pulled each muscle vertex) + /fascia_ab/shell (the fascia
    shell from the ON run, grey context). off|on side-by-side = 散开 vs 聚拢.
  * colorbar_fascia_dist.png — jet mm legend for the ON heatmap.
  * fascia_spread.csv — per-frame GATHER metric (off vs on), plotted separately.

GATHER METRIC (self-defined, so rest_scale can be tuned against a number):
  lateral cross-sectional spread of the muscle cloud. Per frame f, take the
  muscle-group centroid c and principal (long) axis u = top PCA eigenvector of
  the OFF cloud (same axis for both runs so the comparison is apples-to-apples),
  then for every muscle vertex measure its PERPENDICULAR distance to that axis
  line, r_perp = |(x-c) - ((x-c)·u)u|. spread = mean r_perp (also p95). Fascia
  gather pulls the cloud toward its long axis => spread_on < spread_off while the
  along-axis length (and per-tet volume) is preserved; rms3d is reported too as a
  direction-blind control. Headline gather% = (off-on)/off of mean r_perp.

Run: PYTHONPATH=src .venv/bin/python -m tools.humanV2_fascia_ablation
"""
import csv
import numpy as np
from pxr import Usd, UsdGeom, Sdf, Vt

OFF_USD = "output/humanV2/fascia_off.usd"
ON_USD = "output/humanV2/fascia_on.usd"
MUSCLE = "/root/muscle"
SHELL = "/root/fascia"
OUT = "output/humanV2/fascia/fascia_ablation.usdc"
CBAR = "output/humanV2/fascia/colorbar_fascia_dist.png"
SPREADCSV = "output/humanV2/fascia/fascia_spread.csv"

STOPS = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], np.float32)  # jet
SX = np.linspace(0, 1, len(STOPS))


def jet(t):
    t = np.clip(t, 0, 1)
    return np.stack([np.interp(t, SX, STOPS[:, i]) for i in range(3)], 1).astype(np.float32)


def load_points(usd, prim, frames):
    s = Usd.Stage.Open(usd)
    pa = UsdGeom.PointBased(s.GetPrimAtPath(prim)).GetPointsAttr()
    return s, np.stack([np.asarray(pa.Get(f), np.float32) for f in frames])


def perp_spread(P, axis):
    """mean & p95 perpendicular distance of cloud P (N,3) to its centroid's
    long axis; plus rms 3d radius (direction-blind control)."""
    c = P.mean(0)
    d = P - c
    along = d @ axis
    perp = np.linalg.norm(d - np.outer(along, axis), axis=1)
    r3d = np.linalg.norm(d, axis=1)
    return perp.mean(), np.percentile(perp, 95), np.sqrt((r3d ** 2).mean())


import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ---- frame range from the OFF stage ----
_s = Usd.Stage.Open(OFF_USD)
t0, t1 = int(_s.GetStartTimeCode()), int(_s.GetEndTimeCode())
fps = _s.GetTimeCodesPerSecond()
up = UsdGeom.GetStageUpAxis(_s)
frames = list(range(t0, t1 + 1))

s_off, off = load_points(OFF_USD, MUSCLE, frames)
s_on, on = load_points(ON_USD, MUSCLE, frames)
assert off.shape == on.shape, f"topology mismatch {off.shape} vs {on.shape}"
NV = off.shape[1]
print(f"frames={len(frames)}  muscle NV={NV}  up={up}")

# topology (constant) from the OFF muscle
m_off = UsdGeom.Mesh(s_off.GetPrimAtPath(MUSCLE))
fc = np.asarray(m_off.GetFaceVertexCountsAttr().Get(frames[0]), np.int32)
fi = np.asarray(m_off.GetFaceVertexIndicesAttr().Get(frames[0]), np.int32)

# ---- per-frame displacement (heatmap) + gather metric ----
dist = np.linalg.norm(on - off, axis=2).astype(np.float32)   # (F, NV) metres
VMAX = float(np.percentile(dist, 99))
print(f"displacement mm: max={dist.max()*1e3:.2f} mean={dist.mean()*1e3:.3f} "
      f"p50={np.percentile(dist,50)*1e3:.2f} p99(VMAX)={VMAX*1e3:.2f}")

rows = []
for i, f in enumerate(frames):
    c = off[i].mean(0)
    d0 = off[i] - c
    axis = np.linalg.svd(d0 - d0.mean(0), full_matrices=False)[2][0]  # top PCA dir of OFF
    pm_o, p95_o, r_o = perp_spread(off[i], axis)
    pm_n, p95_n, r_n = perp_spread(on[i], axis)
    rows.append((f, pm_o, pm_n, p95_o, p95_n, r_o, r_n))

with open(SPREADCSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame", "perp_mean_off_m", "perp_mean_on_m",
                "perp_p95_off_m", "perp_p95_on_m", "rms3d_off_m", "rms3d_on_m"])
    for r in rows:
        w.writerow([r[0]] + [f"{v:.6e}" for v in r[1:]])
lastf = rows[-1]
gather = (lastf[1] - lastf[2]) / lastf[1] * 100.0
print(f"WROTE {SPREADCSV}")
print(f"  perp-spread mean f{frames[-1]}: off={lastf[1]*1e3:.2f}mm on={lastf[2]*1e3:.2f}mm "
      f"=> GATHER {gather:+.2f}%  (rms3d off={lastf[5]*1e3:.1f} on={lastf[6]*1e3:.1f}mm)")

# ---- combined heatmap stage: off (grey) beside on (jet |Δ|) + shell ----
span = off[0].max(0) - off[0].min(0)
step = span[0] * 1.3                          # lay off | on along X

stage = Usd.Stage.CreateNew(OUT)
stage.SetStartTimeCode(t0); stage.SetEndTimeCode(t1)
stage.SetTimeCodesPerSecond(fps)
UsdGeom.SetStageUpAxis(stage, up)
root = UsdGeom.Xform.Define(stage, "/fascia_ab")
stage.SetDefaultPrim(root.GetPrim())


def add_mesh(path, counts, indices):
    m = UsdGeom.Mesh.Define(stage, path)
    m.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(counts))
    m.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(indices))
    return m


# OFF muscle — grey reference, the scattered (散开) state
off_m = add_mesh("/fascia_ab/off", fc, fi)
op = off_m.GetPointsAttr()
for i, f in enumerate(frames):
    op.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(off[i])), Usd.TimeCode(f))
UsdGeom.PrimvarsAPI(off_m.GetPrim()).CreatePrimvar(
    "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant
).GetAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([[0.6, 0.6, 0.6]], np.float32)))

# ON muscle — jet heatmap of |on-off|, the gathered (聚拢) state, offset in X
on_m = add_mesh("/fascia_ab/on", fc, fi)
np_ = on_m.GetPointsAttr()
pv = UsdGeom.PrimvarsAPI(on_m.GetPrim()).CreatePrimvar(
    "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
for i, f in enumerate(frames):
    P = on[i].copy(); P[:, 0] += step
    np_.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(P)), Usd.TimeCode(f))
    cols = jet(dist[i] / VMAX if VMAX > 0 else dist[i])
    pv.GetAttr().Set(Vt.Vec3fArray.FromNumpy(cols), Usd.TimeCode(f))

# fascia shell from the ON run — grey context, sits with the ON muscle
sm = UsdGeom.Mesh(s_on.GetPrimAtPath(SHELL))
if sm.GetPrim().IsValid():
    sfc = np.asarray(sm.GetFaceVertexCountsAttr().Get(frames[0]), np.int32)
    sfi = np.asarray(sm.GetFaceVertexIndicesAttr().Get(frames[0]), np.int32)
    spa = sm.GetPointsAttr()
    shell = add_mesh("/fascia_ab/shell", sfc, sfi)
    shp = shell.GetPointsAttr()
    for i, f in enumerate(frames):
        P = np.asarray(spa.Get(f), np.float32).copy(); P[:, 0] += step
        shp.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(P)), Usd.TimeCode(f))
    UsdGeom.PrimvarsAPI(shell.GetPrim()).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant
    ).GetAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([[0.85, 0.8, 0.55]], np.float32)))
    UsdGeom.Imageable(shell.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

stage.GetRootLayer().Save()
print(f"WROTE {OUT}  (/fascia_ab/off grey | /fascia_ab/on jet heatmap + shell)")

# ---- colorbar (jet, mm) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

vmm = round(VMAX * 1e3, 1)
cmap = LinearSegmentedColormap.from_list("vsjet", list(map(tuple, STOPS)), N=256)
sm2 = ScalarMappable(norm=Normalize(0, vmm), cmap=cmap)
plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
stepc = max(1, round(vmm / 6))
fig = plt.figure(figsize=(1.1, 4.8))
cax = fig.add_axes([0.05, 0.05, 0.34, 0.9])
fig.colorbar(sm2, cax=cax, orientation="vertical", extend="max",
             ticks=list(np.arange(0, vmm + 1e-6, stepc)))
fig.savefig(CBAR, dpi=300, bbox_inches="tight", transparent=True)
print(f"WROTE {CBAR}  (VMAX={vmm}mm)")
