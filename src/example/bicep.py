"""Warp-based muscle simulation with OpenGL or USD rendering.

Uses Warp MuscleSim for PBD computation and WarpRenderer for visualization.

Usage (OpenGL interactive):
    PYTHONPATH=src python -m example.bicep

Usage (headless auto-test, no rendering):
    PYTHONPATH=src python -m example.bicep --auto --render none

Usage (auto-test with USD output):
    PYTHONPATH=src python -m example.bicep --auto --render usd

Usage (USD output):
    PYTHONPATH=src python -m example.bicep --render usd
"""

import argparse
import logging

import warp as wp

from muscle.util.util import setup_logging
from muscle.util.util import load_config
from muscle.musclesim import MuscleSim
from muscle.engine import activation_ramp, fill_float_kernel

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Warp muscle simulation example.")
    parser.add_argument("--auto", action="store_true", help="Headless auto-test mode.")
    parser.add_argument("--render", type=str, default=None, choices=["human", "usd", "none"],
                        help="Render mode: 'human' (OpenGL), 'usd' (USD file), 'none'. "
                             "Default: 'human' for interactive, 'usd' for --auto.")
    parser.add_argument("--device", type=str, default="cpu", help="Warp device (cpu or cuda:0).")
    args = parser.parse_args()

    setup_logging()

    wp.init()
    wp.set_device(args.device)

    # Resolve render mode: default is 'usd' for --auto, 'human' for interactive
    if args.render is None:
        render_mode = "usd" if args.auto else "human"
    elif args.render == "none":
        render_mode = None
    else:
        render_mode = args.render

    cfg = load_config("data/muscle/config/bicep.json")
    cfg.gui = render_mode is not None
    cfg.render_mode = render_mode

    # OpenGL interactive: run until window closed (effectively unlimited)
    if render_mode == "human" and not args.auto:
        cfg.nsteps = 1_000_000

    sim = MuscleSim(cfg)
    log.info(f"n_verts={sim.n_verts}, use_jacobi={sim.use_jacobi}, render_mode={render_mode}")

    # USD mode is non-interactive, always use auto test path
    if args.auto or render_mode == "usd":
        _run_auto_test(sim, cfg)
    else:
        sim.run()


def _run_auto_test(sim, cfg):
    """Headless test: ramp activation over nsteps, render each frame, print diagnostics."""
    log.info(f"Running {cfg.nsteps} steps (auto, render={cfg.render_mode})...")
    for step in range(1, cfg.nsteps + 1):
        act = activation_ramp(step / cfg.nsteps)
        wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
                  inputs=[sim.activation, act])
        sim.step()
        sim.step_cnt = step
        sim.render()

        if step % 50 == 0 or step == 1:
            pos_np = sim.pos.numpy()
            centroid = pos_np.mean(axis=0)
            vol_err = sim.calc_vol_error()
            log.info(f"step={step:4d}  act={act:.2f}  "
                     f"centroid=({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})  "
                     f"vol_err={vol_err:.4f}")

    if sim.renderer is not None:
        sim.renderer.save()
    log.info("Auto test done.")


if __name__ == "__main__":
    main()
