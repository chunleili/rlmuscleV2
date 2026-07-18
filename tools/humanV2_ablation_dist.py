"""humanV2 ablation displacement heatmap — per-vertex distance to a baseline run,
jet-colored on the muscle mesh (same scheme as output/otto/ablation).

Reads two already-exported humanV2 USDs (identical muscle topology), computes the
per-frame per-vertex distance ‖variant − baseline‖ by direct correspondence, and
writes ONE combined stage carrying an `ablation` variantSet (baseline + each
variant); the variant meshes pull points/topology by REFERENCE (no data copy) and
this script only authors the `displayColor` distance heatmap. Also writes the
shared mm colorbar (jet, scale = p99 over all variant distances).

Run: PYTHONPATH=src .venv/bin/python -m tools.humanV2_ablation_dist
(baseline / variants / muscle prim are the constants below — edit to retarget.)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from pxr import Usd, UsdGeom, Sdf, Vt

DIR = "output/humanV2"
OUT = f"{DIR}/ablation_dist.usdc"
MUSCLE = "/root/muscle"
BASELINE = "humanV2"                       # off (no multigrid) = the baseline
VARIANTS = {"off": "humanV2", "gmg": "humanV2_gmg"}   # label -> usd stem

STOPS = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], np.float32)  # jet
SX = np.linspace(0, 1, len(STOPS))


def jet(t):
    t = np.clip(t, 0, 1)
    return np.stack([np.interp(t, SX, STOPS[:, i]) for i in range(3)], 1).astype(np.float32)


# ---- open runs, grab muscle points ----
stages, pts = {}, {}
for label, stem in VARIANTS.items():
    s = Usd.Stage.Open(f"{DIR}/{stem}.usd")
    stages[label] = s
    pts[label] = UsdGeom.PointBased(s.GetPrimAtPath(MUSCLE)).GetPointsAttr()

base_label = next(k for k, v in VARIANTS.items() if v == BASELINE)
s0 = stages[base_label]
t0, t1 = int(s0.GetStartTimeCode()), int(s0.GetEndTimeCode())
frames = list(range(t0, t1 + 1))
N = len(pts[base_label].Get(t0))
base_pts = np.stack([np.asarray(pts[base_label].Get(f), np.float32) for f in frames])
print(f"frames={len(frames)}  N={N}  baseline={base_label}")

# ---- per-variant per-frame distance to baseline ----
dist = {}
for label in VARIANTS:
    if label == base_label:
        dist[label] = np.zeros((len(frames), N), np.float32)
        continue
    d = np.stack([np.linalg.norm(np.asarray(pts[label].Get(f), np.float32) - base_pts[i], axis=1)
                  for i, f in enumerate(frames)]).astype(np.float32)
    dist[label] = d
    print(f"  {label:6s} dist  max={d.max()*1e3:.2f}mm  mean={d.mean()*1e3:.3f}mm  "
          f"p50={np.percentile(d,50)*1e3:.2f}mm  p99={np.percentile(d,99)*1e3:.2f}mm")

alld = np.concatenate([d.reshape(-1) for k, d in dist.items() if k != base_label])
VMAX = float(np.percentile(alld, 99))
np.save(f"{DIR}/dist_vmax_m.npy", np.array(VMAX))
print(f"shared VMAX (p99) = {VMAX*1e3:.3f} mm  (max {alld.max()*1e3:.2f}mm)")

# ---- combined stage: ablation variantSet, dist displayColor per variant ----
stage = Usd.Stage.CreateNew(OUT)
stage.SetStartTimeCode(t0); stage.SetEndTimeCode(t1)
stage.SetTimeCodesPerSecond(s0.GetTimeCodesPerSecond())
UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(s0))
root = UsdGeom.Xform.Define(stage, "/ablation")
stage.SetDefaultPrim(root.GetPrim())
muscle = UsdGeom.Mesh.Define(stage, "/ablation/muscle")
prim = muscle.GetPrim()
vset = prim.GetVariantSets().AddVariantSet("ablation")
for label, stem in VARIANTS.items():
    vset.AddVariant(label)
    vset.SetVariantSelection(label)
    with vset.GetVariantEditContext():
        prim.GetReferences().AddReference(f"./{stem}.usd", MUSCLE)   # points+topology free
        pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        for i, f in enumerate(frames):
            cols = jet(dist[label][i] / VMAX if VMAX > 0 else dist[label][i])
            pv.GetAttr().Set(Vt.Vec3fArray.FromNumpy(cols), Usd.TimeCode(f))
vset.SetVariantSelection("gmg")            # default: show the variant heatmap
stage.GetRootLayer().Save()
print(f"WROTE {OUT}  (variantSet ablation = {list(VARIANTS)}, dist heatmap vs {base_label})")

# ---- colorbar (jet, mm) ----
vmm = round(VMAX * 1e3, 1)
cmap = LinearSegmentedColormap.from_list("vsjet", list(map(tuple, STOPS)), N=256)
sm = ScalarMappable(norm=Normalize(0, vmm), cmap=cmap)
plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
step = max(1, round(vmm / 6))
fig = plt.figure(figsize=(4.8, 1.3))
cax = fig.add_axes([0.06, 0.55, 0.88, 0.22])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max",
                  ticks=list(np.arange(0, vmm + 1e-6, step)))
cb.set_label("Distance to baseline (off) — mm")
fig.savefig(f"{DIR}/colorbar_dist.png", dpi=300, bbox_inches="tight", transparent=True)
print(f"wrote {DIR}/colorbar_dist.png  (VMAX={vmm}mm)")
