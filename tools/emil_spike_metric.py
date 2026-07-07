"""Quantify + visualize surface spikiness of the Emil solver muscle: pin vs nopin.

Spike = local surface roughness = ‖p − mean(1-ring neighbors)‖ (discrete Laplacian
magnitude). A hard-pinned vertex yanked onto a distorted bone target sticks out
past its neighbors → large roughness. Both runs share the SAME tet surface
topology, so roughness is directly comparable per vertex.

Writes:
  * output/emil/emil_spike.usdc — variantSet 'run' = {pin, nopin}; each the solver
    surface jet-colored by roughness (shared VMAX = p99 over both, mm).
  * output/emil/emil_spike.csv — per-frame roughness mean/p99/max for both runs
    (plot separately).

Run: PYTHONPATH=src .venv/bin/python -m tools.emil_spike_metric
"""
import csv
import numpy as np
from pxr import Usd, UsdGeom, Sdf, Vt

# pin=full-pin diagnostic (emil_full.json), nopin=attach-only (emil_nopin.json),
# tendon=canonical tendon-only pin (emil.json).
RUNS = {"pin": "output/emil/emil_full.anim.usd", "nopin": "output/emil/emil_nopin.anim.usd",
        "tendon": "output/emil/emil.anim.usd"}
PRIM = "/char/sim/muscle"
OUT = "output/emil/emil_spike.usdc"
CSV = "output/emil/emil_spike.csv"

STOPS = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], np.float32)
SX = np.linspace(0, 1, len(STOPS))


def jet(t):
    t = np.clip(t, 0, 1)
    return np.stack([np.interp(t, SX, STOPS[:, i]) for i in range(3)], 1).astype(np.float32)


def load(usd):
    s = Usd.Stage.Open(usd)
    m = UsdGeom.Mesh(s.GetPrimAtPath(PRIM))
    t0, t1 = int(s.GetStartTimeCode()), int(s.GetEndTimeCode())
    frames = list(range(t0, t1 + 1))
    fc = np.asarray(m.GetFaceVertexCountsAttr().Get(frames[0]), np.int32)
    fi = np.asarray(m.GetFaceVertexIndicesAttr().Get(frames[0]), np.int32)
    pa = m.GetPointsAttr()
    pts = np.stack([np.asarray(pa.Get(f), np.float32) for f in frames])
    return frames, fc, fi.reshape(-1, 3), pts


def onering(NV, tris):
    """CSR-ish 1-ring neighbor sums via bincount; returns (sum_fn, deg)."""
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])                       # undirected
    e = np.unique(e, axis=0)
    src, dst = e[:, 0], e[:, 1]
    deg = np.bincount(src, minlength=NV).astype(np.float32)
    deg[deg == 0] = 1.0
    return src, dst, deg


def roughness(pts, src, dst, deg):
    """‖p − mean(neighbors)‖ per vertex, per frame."""
    F, NV, _ = pts.shape
    out = np.empty((F, NV), np.float32)
    for i in range(F):
        nb = np.zeros((NV, 3), np.float32)
        np.add.at(nb, src, pts[i, dst])
        out[i] = np.linalg.norm(pts[i] - nb / deg[:, None], axis=1)
    return out


data, rough = {}, {}
ref_frames = None
for name, usd in RUNS.items():
    frames, fc, tris, pts = load(usd)
    ref_frames = ref_frames or frames
    assert frames == ref_frames, "frame range mismatch"
    src, dst, deg = onering(pts.shape[1], tris)
    r = roughness(pts, src, dst, deg)
    data[name] = (fc, tris, pts)
    rough[name] = r
    print(f"{name:6s} NV={pts.shape[1]} tris={len(tris)}  roughness mm: "
          f"mean={r.mean()*1e3:.3f} p99={np.percentile(r,99)*1e3:.2f} max={r.max()*1e3:.2f}")

VMAX = float(max(np.percentile(r, 99) for r in rough.values()))
print(f"shared VMAX (p99) = {VMAX*1e3:.2f} mm")

# ---- CSV ----
with open(CSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame"] + [f"{n}_{s}_mm" for n in RUNS for s in ("mean", "p99", "max")])
    for i, f in enumerate(ref_frames):
        row = [f]
        for n in RUNS:
            r = rough[n][i]
            row += [f"{r.mean()*1e3:.4f}", f"{np.percentile(r,99)*1e3:.3f}", f"{r.max()*1e3:.3f}"]
        w.writerow(row)
print(f"WROTE {CSV}")

# ---- variant heatmap stage ----
stage = Usd.Stage.CreateNew(OUT)
stage.SetStartTimeCode(ref_frames[0]); stage.SetEndTimeCode(ref_frames[-1])
s0 = Usd.Stage.Open(next(iter(RUNS.values())))
stage.SetTimeCodesPerSecond(s0.GetTimeCodesPerSecond())
UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(s0))
root = UsdGeom.Xform.Define(stage, "/emil_spike")
stage.SetDefaultPrim(root.GetPrim())
mesh = UsdGeom.Mesh.Define(stage, "/emil_spike/muscle")
prim = mesh.GetPrim()
vset = prim.GetVariantSets().AddVariantSet("run")
for name in RUNS:
    fc, tris, pts = data[name]
    vset.AddVariant(name)
    vset.SetVariantSelection(name)
    with vset.GetVariantEditContext():
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(fc))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(tris.reshape(-1)))
        pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        for i, f in enumerate(ref_frames):
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts[i]), Usd.TimeCode(f))
            cols = jet(rough[name][i] / VMAX if VMAX > 0 else rough[name][i])
            pv.GetAttr().Set(Vt.Vec3fArray.FromNumpy(cols), Usd.TimeCode(f))
vset.SetVariantSelection("pin")
stage.GetRootLayer().Save()
print(f"WROTE {OUT}  (variantSet run = {list(RUNS)}, jet roughness VMAX={VMAX*1e3:.2f}mm)")
