"""One-way bone-driven muscle (no feedback to bones).

Loads a single muscle from Human.usd (TetMesh) and the Otto Ragdoll bones,
drives the bones from a per-joint motion JSON (otto_motion.json) produced
by Houdini KineFX FBIK, and pushes new bone_pos into the Warp attach-target
buffer each frame.

Usage:
    PYTHONPATH=src .venv/bin/python -m muscle.coupleOneWay.example \
        --auto --steps 1000 --device cpu
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import warp as wp

from muscle.util.util import load_config
from muscle.coupleOneWay.bone_anim import BoneAnimation
from muscle.coupleOneWay.lbs_skin import LBSDriver
from muscle.util.util import setup_logging
from muscle.muscle_warp import MuscleSim, activation_ramp, fill_float_kernel

log = logging.getLogger(__name__)

DEFAULT_CONFIG = "data/muscle/config/bicep_drive.json"


def main():
    p = argparse.ArgumentParser(description="One-way bone-driven muscle.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--auto", action="store_true", help="Headless run.")
    p.add_argument("--steps", type=int, default=None, help="Override cfg.nsteps.")
    p.add_argument("--frames", type=int, default=None,
                   help="Number of *motion* frames to cover. Overrides --steps "
                        "and cfg.nsteps. nsteps becomes round(frames / "
                        "(bone_speed_scale * dt * motion_fps)). Default: cfg.nframes "
                        "if set, else fall back to nsteps.")
    p.add_argument("--render", choices=["human", "usd", "none"], default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--activation", type=float, default=None,
                   help="Constant activation level. If unset, use activation_ramp(t) "
                        "over the run (0->0.5->1.0->0.7->0.3->0).")
    p.add_argument("--output", default=None,
                   help="Output USD path or basename. If a bare name is given, "
                        "writes to output/<name>.usd. If unset, defaults to "
                        "output/<cfg.name>.usd.")
    p.add_argument("--lbs-blend", type=float, default=None,
                   help="0..1 weight of LBS-skinned target over inertial pos "
                        "each substep, before constraint solve. Overrides "
                        "cfg.lbs_blend. 0 disables LBS entirely.")
    p.add_argument("--lbs-only", action="store_true",
                   help="Pure LBS baseline: skip integrate / constraints / "
                        "pintotarget entirely. Each step writes pos = "
                        "pos0 @ T_rel[bone] then renders. Implies lbs_blend=1.")
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

    if render_mode == "usd":
        if args.output:
            out = args.output
            # Bare name (no dir, no .usd) → output/<name>.usd
            if os.path.dirname(out) == "" and not out.endswith(".usd"):
                out = os.path.join("output", out + ".usd")
            elif not out.endswith(".usd"):
                out = out + ".usd"
        else:
            out = os.path.join("output", f"{cfg.name}.usd")
        cfg.output_path = out
        log.info(f"USD output: {out}")

    sim = MuscleSim(cfg)
    log.info(f"n_verts={sim.n_verts}, n_bone_pts={sim.bone_pos.shape[0]}, "
             f"render={render_mode}")

    anim = BoneAnimation(
        human_usd=cfg.bone_geo_path,
        motion_json=cfg.motion_json_path,
        bone_root=getattr(cfg, "bone_root", "/Human/Ragdoll/Bones"),
    )

    lbs_blend = (args.lbs_blend if args.lbs_blend is not None
                 else float(getattr(cfg, "lbs_blend", 0.0)))
    if args.lbs_only:
        lbs_blend = 1.0
    lbs = None
    if lbs_blend > 0.0 or args.lbs_only:
        cache_path = os.path.join("data", "Human", f"lbs_weights_{cfg.name}.json")
        lbs = LBSDriver(sim.pos0_np, anim, cache_path=cache_path)
        log.info(f"LBS enabled: blend={lbs_blend} "
                 f"(only={args.lbs_only}, cache={cache_path})")

    # Sanity: anim layout must match what MuscleSim loaded.
    if anim.bone_pos_rest.shape != sim.bone_pos.shape:
        raise RuntimeError(
            f"BoneAnimation layout mismatch: anim={anim.bone_pos_rest.shape} "
            f"vs sim.bone_pos={sim.bone_pos.shape}. Check bone_root + traversal.")
    drift = float(np.linalg.norm(anim.bone_pos_rest - sim.bone_pos, axis=1).max())
    log.info(f"rest layout match OK; max-rest-drift = {drift:.6f}")

    log.info(f"motion: {anim.n_frames} frames @ {anim.fps} fps "
             f"(span {anim.time_seconds_range[1]:.2f}s)")

    speed = float(getattr(cfg, "bone_speed_scale", 1.0))
    motion_per_step = max(speed * cfg.dt * anim.fps, 1e-9)

    # Frame-based run length: --frames > cfg.nframes > existing nsteps.
    # nsteps = round(nframes / motion_per_step) so output covers `nframes`
    # motion frames regardless of bone_speed_scale.
    nframes = args.frames if args.frames is not None else getattr(cfg, "nframes", None)
    if nframes is not None:
        cfg.nsteps = max(1, int(round(nframes / motion_per_step)))
        log.info(f"nframes={nframes} → nsteps={cfg.nsteps} (motion_per_step={motion_per_step:.4f})")

    def frame_at(step: int) -> int:
        # Bone advances `motion_per_step` motion-frames per sim-step. Loop
        # motion when sim runs longer than the original animation.
        return int(round(step * motion_per_step)) % anim.n_frames

    # Render stride: emit one USD frame per *motion* frame, so the output
    # timeline matches the original animation regardless of bone_speed_scale
    # (retime affects the solver only, not output cadence).
    render_stride = max(1, int(round(1.0 / motion_per_step)))
    log.info(f"render_stride={render_stride} (1 USD frame per {render_stride} sim-steps; "
             f"~{cfg.nsteps // render_stride} output frames)")

    if args.lbs_only:
        if lbs is None:
            raise RuntimeError("--lbs-only requires LBSDriver init (got None).")
        _run_lbs_only(sim, anim, cfg, args, frame_at, render_stride, lbs)
    elif args.auto or render_mode == "usd":
        _run_auto(sim, anim, cfg, args, frame_at, render_stride, lbs, lbs_blend)
    else:
        _run_interactive(sim, anim, cfg, args, frame_at, render_stride, lbs, lbs_blend)


def _drive_one_step(sim, anim, frame_idx: int):
    bone_pos_world = anim.sample_at_frame(frame_idx).astype(np.float32)
    sim.bone_pos_field.assign(bone_pos_world)
    sim.update_attach_targets()


def _make_substep_hooks(sim, anim, cfg, step_index: int, lbs, lbs_blend: float):
    """Build (pre_substep_hook, post_integrate_hook).

    ``pre`` lerps the bone target from frame_at(step-1) to frame_at(step) and
    refreshes attach targets. ``post`` (only if ``lbs`` is provided) skins the
    muscle pos toward an LBS prediction from the same fractional frame so the
    constraint solve corrects from a bone-driven shape rather than from
    inertia drift. cfg.bone_speed_scale (default 1.0) scales playback speed.
    """
    speed = float(getattr(cfg, "bone_speed_scale", 1.0))
    f_a = (step_index - 1) * cfg.dt * anim.fps * speed
    f_b = step_index * cfg.dt * anim.fps * speed
    state = {"f": f_b}  # mutated by pre, read by post

    def pre(sub_i, n_sub):
        t = (sub_i + 1) / n_sub
        f = f_a + t * (f_b - f_a)
        state["f"] = f
        bone_pos_world = anim.sample_at_fractional_frame(f).astype(np.float32)
        sim.bone_pos_field.assign(bone_pos_world)
        sim.update_attach_targets()

    if lbs is None or lbs_blend <= 0.0:
        return pre, None

    def post(sub_i, n_sub):
        joint_world = anim.joint_world_at_fractional_frame(state["f"])
        flat = lbs.compute_T_rel_flat(joint_world)
        sim.set_lbs_T_rel(flat)
        sim.apply_lbs(lbs.vertex_bone_field(sim), blend=lbs_blend)

    return pre, post


def _step_with_bone_hooks(sim, pre, post):
    """One frame with bone-anim pre-substep + (optional) LBS post-integrate.

    This is the LBS-predictor → constraint-corrector flow: pre() advances the
    bone target each substep, post() (if given) overwrites pos with an LBS
    skin so the solver corrects from a bone-driven shape rather than inertia.
    """
    debug_per_sub = bool(getattr(sim.cfg, "debug_per_substep", False))
    n_sub = sim.cfg.num_substeps
    for sub_i in range(n_sub):
        pre(sub_i, n_sub)
        sim.clear_forces()
        sim.accumulate_active_fiber_force()
        sim.integrate()
        if post is not None:
            post(sub_i, n_sub)
        if hasattr(sim, "apply_pintotarget"):
            sim.apply_pintotarget()
        sim.clear()
        sim.solve_constraints()
        if sim.use_jacobi:
            sim.apply_dP()
        sim.post_smooth()
        sim.freeze_near_inverted()
        sim.repair_inverted_tets()
        if hasattr(sim, "apply_pintotarget"):
            sim.apply_pintotarget()
        sim.update_velocities()
        if debug_per_sub:
            sim._debug_check_substep(sub_i, n_sub)


def _run_auto(sim, anim, cfg, args, frame_at, render_stride=1, lbs=None, lbs_blend=0.0):
    mode = "ramp" if args.activation is None else f"const={args.activation}"
    log.info(f"Running {cfg.nsteps} steps (auto, render={cfg.render_mode}, "
             f"activation={mode}, per-substep bone interp=ON, "
             f"lbs_blend={lbs_blend if lbs is not None else 0.0})...")
    for step in range(1, cfg.nsteps + 1):
        f_idx = frame_at(step)
        acti = (args.activation if args.activation is not None
                else activation_ramp(step / max(cfg.nsteps, 1)))
        wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
                  inputs=[sim.activation, float(acti)])
        sim.step_cnt = step
        pre, post = _make_substep_hooks(sim, anim, cfg, step, lbs, lbs_blend)
        _step_with_bone_hooks(sim, pre, post)
        if step % render_stride == 0:
            # Stamp USD at motion-frame time (step/stride * dt) so playback
            # matches original motion fps regardless of bone_speed_scale.
            saved = sim.step_cnt
            sim.step_cnt = step // render_stride
            sim.render()
            sim.step_cnt = saved

        if step % 50 == 0 or step == 1:
            pos_np = sim.pos.numpy()
            centroid = pos_np.mean(axis=0)
            log.info(f"step={step:4d}  frame={f_idx:3d}  acti={acti:.2f}  "
                     f"centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})  "
                     f"vol_err={sim.calc_vol_error():.4f}")

    if sim.renderer is not None:
        sim.renderer.save()
    log.info("Done.")


def _run_lbs_only(sim, anim, cfg, args, frame_at, render_stride, lbs):
    """Pure-LBS baseline: pos = pos0 @ T_rel[bone] each step. No solver at all."""
    speed = float(getattr(cfg, "bone_speed_scale", 1.0))
    motion_per_step = max(speed * cfg.dt * anim.fps, 1e-9)
    log.info(f"Running {cfg.nsteps} steps (LBS-ONLY: no integrate, no constraints)...")
    vertex_bone_field = lbs.vertex_bone_field(sim)
    for step in range(1, cfg.nsteps + 1):
        f = step * motion_per_step
        joint_world = anim.joint_world_at_fractional_frame(f)
        flat = lbs.compute_T_rel_flat(joint_world)
        sim.set_lbs_T_rel(flat)
        sim.apply_lbs(vertex_bone_field, blend=1.0)
        # Also update the bone mesh so the renderer sees moving bones
        # (the regular pipeline does this in pre_substep_hook).
        bone_pos_world = anim.sample_at_fractional_frame(f).astype(np.float32)
        sim.bone_pos_field.assign(bone_pos_world)
        if step % render_stride == 0:
            saved = sim.step_cnt
            sim.step_cnt = step // render_stride
            sim.render()
            sim.step_cnt = saved
        if step % 50 == 0 or step == 1:
            pos_np = sim.pos.numpy()
            centroid = pos_np.mean(axis=0)
            log.info(f"step={step:4d}  frame={frame_at(step):3d}  "
                     f"centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
    if sim.renderer is not None:
        sim.renderer.save()
    log.info("Done (LBS-only).")


def _run_interactive(sim, anim, cfg, args, frame_at, render_stride=1, lbs=None, lbs_blend=0.0):
    step = 0
    while step < cfg.nsteps:
        acti = (args.activation if args.activation is not None
                else activation_ramp(step / max(cfg.nsteps, 1)))
        wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
                  inputs=[sim.activation, float(acti)])
        sim.step_cnt = step + 1
        pre, post = _make_substep_hooks(sim, anim, cfg, step + 1, lbs, lbs_blend)
        _step_with_bone_hooks(sim, pre, post)
        if (step + 1) % render_stride == 0:
            saved = sim.step_cnt
            sim.step_cnt = (step + 1) // render_stride
            sim.render()
            sim.step_cnt = saved
        step += 1
        if sim.renderer is not None and getattr(sim.renderer, "should_close",
                                                lambda: False)():
            break
    if sim.renderer is not None:
        sim.renderer.save()


if __name__ == "__main__":
    main()
