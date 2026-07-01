"""Bicep fiber-contraction probe across solver modes (CPU GS / GPU Jacobi / GPU colored-GS).

Isolates the fiber constraint's effect: settle the muscle passively (activation=0),
record P0; then hold full activation (=1), settle, record P1. The deformation
P1-P0 is purely the fiber response (bone static, no gravity in bicep.json).

Reports a contraction signature (mean/max vertex displacement, length-along-fiber
shrink, girth growth, volume error) so the three solver modes can be compared
numerically. Writes an npz with P0/P1/disp for optional colored export.

    PYTHONPATH=src python tools/fiber_probe_bicep.py --device cuda:0 --mode jacobi
    PYTHONPATH=src python tools/fiber_probe_bicep.py --device cuda:0 --mode cgs
    PYTHONPATH=src python tools/fiber_probe_bicep.py --device cpu    --mode gs
"""
import argparse
import numpy as np
import warp as wp

from muscle.util.util import load_config
from muscle.musclesim import MuscleSim
from muscle.engine import fill_float_kernel


def settle(sim, act_value, n_steps):
    """Hold a constant activation for n_steps full steps, return final positions."""
    wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
              inputs=[sim.activation, float(act_value)])
    for _ in range(n_steps):
        wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
                  inputs=[sim.activation, float(act_value)])
        sim.step()
    wp.synchronize()
    return sim.pos.numpy().copy()


def contraction_signature(P0, P1, fiber_axis):
    """Numeric fingerprint of the fiber deformation P0->P1."""
    disp = P1 - P0
    dmag = np.linalg.norm(disp, axis=1)
    # length along principal fiber axis, girth perpendicular to it
    def extent_along(P, axis):
        proj = P @ axis
        return proj.max() - proj.min()
    def girth(P, axis):
        proj = P @ axis
        perp = P - np.outer(proj, axis)
        r = np.linalg.norm(perp - perp.mean(0), axis=1)
        return float(np.percentile(r, 95))
    len0, len1 = extent_along(P0, fiber_axis), extent_along(P1, fiber_axis)
    g0, g1 = girth(P0, fiber_axis), girth(P1, fiber_axis)
    return {
        "mean_disp_mm": float(dmag.mean() * 1000),
        "max_disp_mm": float(dmag.max() * 1000),
        "len_along_fiber_0": float(len0),
        "len_along_fiber_1": float(len1),
        "len_shrink_pct": float((len0 - len1) / len0 * 100),
        "girth_p95_0": g0,
        "girth_p95_1": g1,
        "girth_grow_pct": float((g1 - g0) / g0 * 100),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mode", default="jacobi", choices=["jacobi", "cgs", "gs"])
    ap.add_argument("--config", default="data/muscle/config/bicep.json")
    ap.add_argument("--settle", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wp.init()
    wp.set_device(args.device)

    cfg = load_config(args.config)
    cfg.arch = args.device
    cfg.gui = False
    cfg.render_mode = None
    # Solver mode selection
    if args.mode == "jacobi":
        cfg.use_jacobi = True
        cfg.use_colored_gs = False
    elif args.mode == "cgs":
        cfg.use_jacobi = False
        cfg.use_colored_gs = True
    else:  # gs (sequential; only valid/safe on cpu)
        cfg.use_jacobi = False
        cfg.use_colored_gs = False

    sim = MuscleSim(cfg)
    print(f"[probe] mode={args.mode} device={args.device} "
          f"use_jacobi={sim.use_jacobi} use_colored_gs={sim.use_colored_gs} "
          f"NV={sim.n_verts} fiber_scale={sim.fiber_stiffness_scale} "
          f"contraction_ratio={sim.contraction_ratio}")

    # principal fiber axis from the rest fiber field (mean direction), fallback to PCA
    if getattr(sim, "v_fiber_np", None) is not None:
        fa = np.asarray(sim.v_fiber_np, np.float64)
        fa = fa[np.linalg.norm(fa, axis=1) > 1e-6]
        fiber_axis = fa.mean(0)
        fiber_axis /= (np.linalg.norm(fiber_axis) + 1e-12)
    else:
        fiber_axis = np.array([0.0, 1.0, 0.0])

    P0 = settle(sim, 0.0, args.settle)
    ve0 = sim.calc_vol_error()
    P1 = settle(sim, 1.0, args.settle)
    ve1 = sim.calc_vol_error()

    sig = contraction_signature(P0, P1, fiber_axis)
    print(f"[probe] fiber_axis={fiber_axis.round(3).tolist()}")
    print(f"[probe] vol_err passive={ve0:.4f} active={ve1:.4f}")
    print("[RESULT] " + "  ".join(f"{k}={v:.4f}" for k, v in sig.items()))
    # Headline: did the muscle visibly deform under activation?
    verdict = "CONTRACTED" if sig["max_disp_mm"] > 1.0 else "NO-DEFORM"
    print(f"[VERDICT] mode={args.mode} device={args.device} -> {verdict} "
          f"(max_disp={sig['max_disp_mm']:.2f}mm, len_shrink={sig['len_shrink_pct']:.2f}%, "
          f"girth_grow={sig['girth_grow_pct']:.2f}%)")

    if args.out:
        np.savez(args.out, P0=P0, P1=P1, disp=P1 - P0, fiber_axis=fiber_axis,
                 tet=sim.tet_np if hasattr(sim, "tet_np") else np.zeros((0, 4)))
        print(f"[probe] saved {args.out}")


if __name__ == "__main__":
    main()
