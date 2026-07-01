"""humanV2 fiber-activation demo — contract one muscle with a static skeleton.

The baseline ``humanV2.json`` ships with NO fiber constraint (skeletal follow
only). This example injects a HOBAK **anisotropic-ARAP** (``aniso_arap``) fiber
constraint, hard-pins only the tendon TIPS (freeing the belly so it can bulge),
holds the skeleton static at frame 1, and ramps the chosen muscle's activation
**0 → 1** in equal levels — each settled to equilibrium. It writes a single
time-sampled USD whose ``displayColor`` is a per-vertex displacement heatmap
(blue → red), so scrubbing the timeline shows the fiber contraction build up. A
static grey skeleton (``/bones``) is included for reference.

``aniso_arap`` (quadratic HOBAK energy ``0.5*(s - SI4*fiberscale)^2``) is used
instead of the linear ``fibernorm``: the latter normalizes its XPBD compliance by
the tet rest VOLUME (``alpha = 1/(k*V0*dt^2)``, V0 ~ 1e-9), which makes the
effective stiffness ~1e9x too small so the constraint never reaches its target
(~0.2% shrink, bulge only visible via inertial overshoot). ``aniso_arap`` has no
volume term (``alpha = 1/(k*dt^2)``), reaches the target, and gives a strong
TARGET-DRIVEN static contraction (~25mm bulge vs ~2.6mm), insensitive to k.

    uv run -m example.humanV2_fiber                       # right bicep (default)
    uv run -m example.humanV2_fiber --muscle L_bicepsBrachialis
    # -> output/humanV2_fiber.usd

Every other knob is a json-derived default; only ``--config``, ``--muscle`` and
``--out`` are exposed (mirroring the other examples).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import warp as wp

from muscle.mg.mg import load_config
from muscle.util.bone_source import make_bone_source
from example.humanV2 import build_humanV2_sim

# Fixed fiber-demo parameters (tuned for a stable, visible, drift-free contraction
# on the 468k-tet humanV2 mesh under the GPU Jacobi solver).
FIBER_MODEL = "aniso_arap"     # HOBAK quadratic anisotropic ARAP (target-driven)
FIBER_STIFFNESS = 1.0e8        # aniso_arap stiffness (k-insensitive: reaches target)
CONTRACTION_RATIO = 0.6        # target along-fiber shortening at activation 1
FINE_ITERS = 100              # Jacobi iters/substep (large mesh needs many)
PIN_THRESHOLD = 0.9           # tendonmask>thr -> hard pin (tips only; frees belly)
BELLY_STIFFNESS = 0.0         # free the belly (0 = no soft pin back to rest)
LEVELS = 5                    # activation levels 0 -> 1
SETTLE_PER_LEVEL = 40         # step_frame calls per level (settle to equilibrium)
SETTLE_REST = 30             # passive settle before the ramp


def _jet(x):
    """Blue -> green -> red heatmap for x in [0, 1] -> (N, 3) float32."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], 1).astype(np.float32)


def _compact(surf_flat, n_verts):
    """Trim a flat tri-index list to the verts it references; remap to local."""
    used = np.unique(surf_flat)
    remap = np.full(n_verts, -1, np.int64)
    remap[used] = np.arange(used.shape[0])
    return used, remap[surf_flat].astype(np.int32)


def build_fiber_config(config_path, muscle):
    """Patch the baseline humanV2 config for the fiber-activation demo: static
    single frame, no fascia/collision, tendon-tip pin (freed belly), and an
    injected aniso_arap (HOBAK) fiber constraint. Returns the cfg dict."""
    cfg = load_config(config_path)
    cfg["start_frame"] = 1
    cfg["end_frame"] = 1
    cfg["use_fascia"] = False
    cfg["use_multigrid"] = False
    cfg["use_jacobi"] = True
    cfg["fine_iterations"] = FINE_ITERS
    for c in cfg.get("constraints", []):
        if c.get("name") == "belly_attach":
            c["stiffness"] = BELLY_STIFFNESS
        if c.get("name") == "muscleToBonePin":
            c["region"] = "tendonmask"
            c["tendon_pin_threshold"] = PIN_THRESHOLD
    cfg.setdefault("constraints", []).append(dict(
        type="fiber", name="fiber", enable=True, fiber_model=FIBER_MODEL,
        stiffness=FIBER_STIFFNESS, dampingratio=0.1,
        contraction_ratio=CONTRACTION_RATIO))
    demo = cfg.setdefault("demo", {})
    demo.setdefault("collision", {})["enabled"] = False
    demo["activate"] = None                       # activation is driven here
    return cfg


def run(config_path, muscle, out):
    wp.init()
    cfg = build_fiber_config(config_path, muscle)
    sim = build_humanV2_sim(cfg)
    print(f"[fiber-demo] activate '{muscle}'  {sim.fiber_model} k={sim.fiber_stiffness:g} "
          f"ratio={sim.contraction_ratio}  fine_iters={sim.fine_iters}  "
          f"tip-pin>{PIN_THRESHOLD} belly_k={BELLY_STIFFNESS:g}")

    mask = sim.tet_mask_for_muscles(muscle)
    if int(mask.sum()) == 0:
        raise SystemExit(f"no tets matched muscle '{muscle}'")
    view_verts = np.unique(sim.tet_np[np.where(
        sim.tet_mask_for_muscles("bicepsBrachialis"))[0]])

    # passive rest reference
    sim.set_fiber_activation(np.zeros(mask.shape[0], np.float32))
    for _ in range(SETTLE_REST):
        sim.step_frame(1)
    wp.synchronize()
    P_rest = sim.positions_np().copy()

    # ramp activation 0 -> 1, settling each level to equilibrium
    frames_pos = [P_rest.copy()]
    for lv in range(1, LEVELS + 1):
        a = lv / LEVELS
        sim.set_fiber_activation(mask.astype(np.float32) * a)
        for _ in range(SETTLE_PER_LEVEL):
            sim.step_frame(1)
        wp.synchronize()
        frames_pos.append(sim.positions_np().copy())
        d = np.linalg.norm((frames_pos[-1] - P_rest)[view_verts], axis=1)
        print(f"  level {lv}/{LEVELS}  act={a:.2f}  max_disp={d.max() * 1000:.2f}mm")

    all_d = np.concatenate([np.linalg.norm((P - P_rest)[view_verts], axis=1)
                            for P in frames_pos])
    vmax = max(float(all_d.max()), 1e-6)

    _write_usd(out, sim, cfg, frames_pos, P_rest, vmax)
    d_final = np.linalg.norm((frames_pos[-1] - P_rest)[view_verts], axis=1)
    print(f"[fiber-demo] wrote {out}  levels=0..{LEVELS}  "
          f"final max_disp={d_final.max() * 1000:.2f}mm")


def _write_usd(out, sim, cfg, frames_pos, P_rest, vmax):
    from pxr import Usd, UsdGeom, Vt, Sdf
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if os.path.exists(out):
        os.remove(out)
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(frames_pos) - 1)

    m_used, m_local = _compact(sim.surf_flat, sim.NV)
    mmesh = UsdGeom.Mesh.Define(stage, "/muscle")
    mmesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (m_local.shape[0] // 3)))
    mmesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(m_local))
    pts_attr = mmesh.CreatePointsAttr()
    col_pv = UsdGeom.PrimvarsAPI(mmesh).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
    for t, P in enumerate(frames_pos):
        pts_attr.Set(Vt.Vec3fArray.FromNumpy(
            np.ascontiguousarray(P[m_used], np.float32)), t)
        disp = np.linalg.norm((P - P_rest)[m_used], axis=1)
        col_pv.Set(Vt.Vec3fArray.FromNumpy(_jet(disp / vmax)), t)

    # static grey skeleton reference
    bs = make_bone_source(cfg, 1)
    bverts = bs.positions(1)
    b_used, b_local = _compact(bs.tris.reshape(-1).astype(np.int32), bverts.shape[0])
    bmesh = UsdGeom.Mesh.Define(stage, "/bones")
    bmesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (b_local.shape[0] // 3)))
    bmesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(b_local))
    bmesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(
        np.ascontiguousarray(bverts[b_used], np.float32)))
    UsdGeom.PrimvarsAPI(bmesh).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant
    ).Set(Vt.Vec3fArray.FromNumpy(np.array([[0.8, 0.78, 0.72]], np.float32)))
    stage.GetRootLayer().Save()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="data/humanV2/config/humanV2.json",
                    help="baseline humanV2 scene config")
    ap.add_argument("--muscle", default="R_bicepsBrachialis",
                    help="muscle to activate (substring match; default right bicep)")
    ap.add_argument("--out", default="output/humanV2_fiber.usd")
    args = ap.parse_args(argv)
    run(args.config, args.muscle, args.out)


if __name__ == "__main__":
    main()
