"""One-way bone-driven muscle (no feedback to bones).

Loads a single muscle from Human.usd (TetMesh) and the Otto Ragdoll bones,
drives the bones from a per-joint motion JSON (otto_motion.json) produced
by Houdini KineFX FBIK, and pushes new bone_pos into the Warp attach-target
buffer each frame.

Usage:
    PYTHONPATH=src .venv/bin/python -m muscle.coupleOneWay.example_drive \
        --auto --steps 1000 --device cpu
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import warp as wp

from muscle.config import load_config
from muscle.coupleOneWay.bone_anim import BoneAnimation
from muscle.log import setup_logging
from muscle.muscle_warp import MuscleSim, fill_float_kernel

log = logging.getLogger(__name__)

DEFAULT_CONFIG = "data/muscle/config/bicep_drive.json"


def main():
    p = argparse.ArgumentParser(description="One-way bone-driven muscle.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--auto", action="store_true", help="Headless run.")
    p.add_argument("--steps", type=int, default=None, help="Override cfg.nsteps.")
    p.add_argument("--render", choices=["human", "usd", "none"], default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--activation", type=float, default=0.0,
                   help="Constant activation level (0.0 = passive, bone-driven only).")
    args = p.parse_args()

    setup_logging()
    wp.init()
    wp.set_device(args.device)

    if args.render is None:
        render_mode = "usd" if args.auto else "human"
    elif args.render == "none":
        render_mode = None
    else:
        render_mode = args.render

    cfg = load_config(args.config)
    cfg.gui = render_mode is not None
    cfg.render_mode = render_mode
    if args.steps is not None:
        cfg.nsteps = args.steps
    if render_mode == "human" and not args.auto:
        cfg.nsteps = 1_000_000

    sim = MuscleSim(cfg)
    log.info(f"n_verts={sim.n_verts}, n_bone_pts={sim.bone_pos.shape[0]}, "
             f"render={render_mode}")

    anim = BoneAnimation(
        human_usd=cfg.bone_geo_path,
        motion_json=cfg.motion_json_path,
        bone_root=getattr(cfg, "bone_root", "/Human/Ragdoll/Bones"),
    )

    # Sanity: anim layout must match what muscle_common loaded.
    if anim.bone_pos_rest.shape != sim.bone_pos.shape:
        raise RuntimeError(
            f"BoneAnimation layout mismatch: anim={anim.bone_pos_rest.shape} "
            f"vs sim.bone_pos={sim.bone_pos.shape}. Check bone_root + traversal.")
    drift = float(np.linalg.norm(anim.bone_pos_rest - sim.bone_pos, axis=1).max())
    log.info(f"rest layout match OK; max-rest-drift = {drift:.6f}")

    log.info(f"motion: {anim.n_frames} frames @ {anim.fps} fps "
             f"(span {anim.time_seconds_range[1]:.2f}s)")

    def frame_at(step: int) -> int:
        # one sim-step = cfg.dt seconds; loop motion when sim runs longer
        return int(round(step * cfg.dt * anim.fps)) % anim.n_frames

    if args.auto or render_mode == "usd":
        _run_auto(sim, anim, cfg, args, frame_at)
    else:
        _run_interactive(sim, anim, cfg, args, frame_at)


def _drive_one_step(sim, anim, frame_idx: int):
    bone_pos_world = anim.sample_at_frame(frame_idx).astype(np.float32)
    sim.bone_pos_field.assign(bone_pos_world)
    sim.update_attach_targets()


def _run_auto(sim, anim, cfg, args, frame_at):
    log.info(f"Running {cfg.nsteps} steps (auto, render={cfg.render_mode}, "
             f"activation={args.activation})...")
    wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
              inputs=[sim.activation, float(args.activation)])
    for step in range(1, cfg.nsteps + 1):
        f_idx = frame_at(step)
        _drive_one_step(sim, anim, f_idx)
        sim.step()
        sim.step_cnt = step
        sim.render()

        if step % 50 == 0 or step == 1:
            pos_np = sim.pos.numpy()
            centroid = pos_np.mean(axis=0)
            log.info(f"step={step:4d}  frame={f_idx:3d}  "
                     f"centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})  "
                     f"vol_err={sim.calc_vol_error():.4f}")

    if sim.renderer is not None:
        sim.renderer.save()
    log.info("Done.")


def _run_interactive(sim, anim, cfg, args, frame_at):
    wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
              inputs=[sim.activation, float(args.activation)])
    step = 0
    while step < cfg.nsteps:
        f_idx = frame_at(step)
        _drive_one_step(sim, anim, f_idx)
        sim.step()
        sim.step_cnt = step
        sim.render()
        step += 1
        if sim.renderer is not None and getattr(sim.renderer, "should_close",
                                                lambda: False)():
            break
    if sim.renderer is not None:
        sim.renderer.save()


if __name__ == "__main__":
    main()
