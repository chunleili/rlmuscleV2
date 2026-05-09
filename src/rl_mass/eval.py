"""Evaluation & rendering for trained MASS-MuJoCo policy.

Loads a trained SimulationNN + MuscleNN, runs deterministic rollouts,
prints statistics, and optionally renders video.

Usage:
    # Stats only
    PYTHONPATH=. uv run python rl_mass/eval.py -d data/rl_mass/config.json -m nn/max

    # With video
    PYTHONPATH=. MUJOCO_GL=egl uv run python rl_mass/eval.py \
        -d data/rl_mass/config.json -m nn/max --video

    # Specify output path
    PYTHONPATH=. MUJOCO_GL=egl uv run python rl_mass/eval.py \
        -d data/rl_mass/config.json -m nn/max --video -o output/eval.mp4
"""

from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np

from rl_mass.env import MassEnv
from rl_mass.model import MuscleNN, SimulationNN, Tensor, use_cuda


def rollout(
    env: MassEnv,
    model: SimulationNN,
    muscle_model: MuscleNN | None,
    max_steps: int = 10000,
    deterministic: bool = True,
    capture_diag: bool = False,
) -> tuple[list[float], int, dict | None, list[np.ndarray]]:
    """Run one episode.

    Args:
        env: MassEnv instance.
        model: Trained SimulationNN.
        muscle_model: Trained MuscleNN (None if not using muscles).
        max_steps: Max control steps before forced termination.
        deterministic: If True, use mean action; if False, sample.
        capture_diag: If True, record per-control-step diagnostics
            (activations / R / L / b / tau_des / tau_muscle / l_m_norm) for
            downstream plotting.

    Returns:
        (rewards, total_control_steps, diag_dict_or_None,
         body_traj_dict, qpos_trajectory).
        diag_dict is populated only when capture_diag=True (stacked NPZ-ready
        per-step muscle arrays + metadata).
        body_traj_dict has 'xpos' (T+1, nbody, 3), 'xquat' (T+1, nbody, 4),
        'names' list[str]; world-space body transforms per control step.
    """
    state = env.reset(rsi=False)
    mujoco.mj_forward(env.model, env.data)
    rewards = []
    qpos_traj = [env.data.qpos.copy()]
    xpos_traj = [env.data.xpos.copy()]
    xquat_traj = [env.data.xquat.copy()]
    num_sim_per_control = env.simulation_hz // env.control_hz

    diag_steps: list[dict] = []

    for step in range(max_steps):
        # Policy forward
        import torch

        s_tensor = Tensor(state.reshape(1, -1).astype(np.float32))
        a_dist, v = model(s_tensor)

        if deterministic:
            action = a_dist.loc.cpu().detach().numpy().squeeze()
        else:
            action = a_dist.sample().cpu().detach().numpy().squeeze()

        env.set_action(action)

        # Step with muscles
        if env.use_muscle and muscle_model is not None:
            mt = Tensor(env.get_muscle_torques().reshape(1, -1).astype(np.float32))
            for i in range(num_sim_per_control // 2):
                dt = Tensor(
                    env.get_desired_torques().reshape(1, -1).astype(np.float32)
                )
                activations = muscle_model(mt, dt).cpu().detach().numpy().squeeze()
                env.set_activation_levels(activations)
                env.step()
                env.step()
        else:
            env.steps_at_once()

        qpos_traj.append(env.data.qpos.copy())
        xpos_traj.append(env.data.xpos.copy())
        xquat_traj.append(env.data.xquat.copy())

        if env.is_end_of_episode():
            break

        # Diagnostics snapshot uses the activations / desired_torque
        # produced by the LAST inner sub-step of this control step.
        if capture_diag and env.use_muscle:
            diag = env.compute_diagnostics()
            diag["t"] = float(env.engine.get_time())
            diag_steps.append(diag)

        reward = env.get_reward()
        rewards.append(reward)
        state = env.get_state()

    diag_out = None
    if capture_diag and len(diag_steps) > 0:
        # Stack per-step diagnostics → arrays of shape (T, ...).
        keys = ["R", "L", "b", "a", "tau_des", "tau_muscle",
                "l_m_norm", "f_a", "f_p"]
        diag_out = {k: np.stack([d[k] for d in diag_steps]).astype(np.float32)
                    for k in keys}
        diag_out["t"] = np.array([d["t"] for d in diag_steps], dtype=np.float32)
        diag_out["reward"] = np.array(rewards[: len(diag_steps)], dtype=np.float32)
        meta = env.get_muscle_metadata()
        diag_out["muscle_names"] = np.array(meta["names"])
        diag_out["muscle_regions"] = np.array(meta["regions"])
        diag_out["muscle_f0"] = meta["f0"].astype(np.float32)

    body_names = [env.model.body(i).name for i in range(env.model.nbody)]
    body_traj = {
        "xpos": np.stack(xpos_traj).astype(np.float32),
        "xquat": np.stack(xquat_traj).astype(np.float32),
        "names": body_names,
    }
    return rewards, len(rewards), diag_out, body_traj, qpos_traj


def render_video(
    env: MassEnv,
    qpos_traj: list[np.ndarray],
    output_path: str,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    cam_distance: float = 3.0,
    cam_azimuth: float = 180.0,
    cam_elevation: float = -8.0,
    track_body: str = "Pelvis",
) -> None:
    """Render qpos trajectory to MP4 with a Y-up body-tracking camera.

    MASS humanoid is authored with Y-up (gravity = [0, -9.8, 0]) but the
    default mujoco.Renderer assumes Z-up, which renders the character lying
    on its side. We override scene.camera[i].pos/forward/up after
    update_scene to force a Y-up camera orbit around the tracked body.

    Args:
        cam_distance: Camera distance to tracked body (smaller = closer).
        cam_azimuth: Yaw around Y (world-up) axis, degrees.
        cam_elevation: Tilt above horizontal, degrees (negative = looking down).
        track_body: Body name to track (camera lookat follows its xipos).
    """
    import imageio
    import math

    # Boost scene lighting: default headlight ambient=0.1 and diffuse=0.4 leave
    # the character very dark with only one directional fill light. Raise both
    # for a well-lit eval render.
    env.model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    env.model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    env.model.vis.headlight.specular[:] = [0.2, 0.2, 0.2]

    renderer = mujoco.Renderer(env.model, height=height, width=width)

    # mjvCamera is only used to get scene.camera sized; we manually compute
    # pos/forward/up for a Y-up orbit below, so azimuth/elevation here are ignored.
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = cam_distance

    track_bid = env.engine.body_id(track_body)
    az = math.radians(cam_azimuth)
    el = math.radians(cam_elevation)

    # Camera offset from target, in world (Y-up) frame:
    #   +Y is up; azimuth rotates around Y axis; elevation tilts above horizontal.
    offset = np.array([
        cam_distance * math.cos(el) * math.sin(az),
        cam_distance * math.sin(el),
        cam_distance * math.cos(el) * math.cos(az),
    ])

    # Camera is fixed: lookat is captured from the FIRST frame's tracked body
    # and reused for every subsequent frame (no tracking).
    env.data.qpos[:] = qpos_traj[0]
    mujoco.mj_forward(env.model, env.data)
    fixed_lookat = env.engine.get_body_xpos(track_bid).copy()
    fixed_cam_pos = fixed_lookat + offset
    forward = fixed_lookat - fixed_cam_pos
    forward /= np.linalg.norm(forward) + 1e-9
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-9
    fixed_up = np.cross(right, forward)

    frames = []
    for qpos in qpos_traj:
        env.data.qpos[:] = qpos
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data, camera=cam)

        for gcam in renderer.scene.camera:
            gcam.pos[:] = fixed_cam_pos
            gcam.forward[:] = forward
            gcam.up[:] = fixed_up

        frame = renderer.render()
        frames.append(frame.copy())

    renderer.close()

    # Write video
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Video saved to {output_path} ({len(frames)} frames, {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MASS-MuJoCo policy")
    parser.add_argument("-d", "--meta", required=True, help="Config JSON file")
    parser.add_argument(
        "-m", "--model", required=True, help="Model path prefix (e.g. nn/max)"
    )
    parser.add_argument("--video", action="store_true", help="Render video")
    parser.add_argument(
        "-o", "--output", default=None, help="Video output path (default: output/eval.mp4)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000, help="Max control steps per episode"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=1, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--stochastic", action="store_true", help="Use stochastic (sampled) actions"
    )
    parser.add_argument("--cam-distance", type=float, default=3.0)
    parser.add_argument("--cam-azimuth", type=float, default=120.0)
    parser.add_argument("--cam-elevation", type=float, default=-12.0)
    parser.add_argument("--track-body", type=str, default="Pelvis")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--diag", type=str, default=None,
        help="Path to write per-step muscle diagnostics NPZ (last episode)",
    )
    parser.add_argument(
        "--save-traj", type=str, default=None,
        help="Save body world transforms + qpos trajectory of last episode to NPZ",
    )
    args = parser.parse_args()

    # Create environment
    env = MassEnv(args.meta)
    print(
        f"num states: {env.num_state}, num actions: {env.num_action}, "
        f"num muscles: {env.num_muscles}"
    )

    # Load models
    model = SimulationNN(env.num_state, env.num_action)
    muscle_model = None
    if env.use_muscle:
        muscle_model = MuscleNN(
            env.num_total_related_dofs, env.num_action, env.num_muscles
        )

    model.load(args.model + ".pt")
    if use_cuda:
        model.cuda()
    model.eval()

    if muscle_model is not None:
        muscle_model.load(args.model + "_muscle.pt")
        if use_cuda:
            muscle_model.cuda()
        muscle_model.eval()

    deterministic = not args.stochastic

    # Run episodes
    all_rewards = []
    all_steps = []
    last_qpos_traj = None
    last_diag = None
    last_body_traj = None

    for ep in range(args.num_episodes):
        is_last = (ep == args.num_episodes - 1)
        rewards, steps, diag, body_traj, qpos_traj = rollout(
            env, model, muscle_model,
            max_steps=args.max_steps,
            deterministic=deterministic,
            capture_diag=(args.diag is not None and is_last),
        )
        if diag is not None:
            last_diag = diag
        ep_return = sum(rewards)
        avg_reward = ep_return / max(steps, 1)
        all_rewards.append(ep_return)
        all_steps.append(steps)
        last_qpos_traj = qpos_traj
        last_body_traj = body_traj
        print(
            f"Episode {ep + 1}: return={ep_return:.3f}, "
            f"avg_reward={avg_reward:.4f}, steps={steps}"
        )

    # Summary
    print("\n--- Summary ---")
    print(f"Episodes: {args.num_episodes}")
    print(f"Avg return: {np.mean(all_rewards):.3f} +/- {np.std(all_rewards):.3f}")
    print(f"Avg steps: {np.mean(all_steps):.1f} +/- {np.std(all_steps):.1f}")

    # Save diagnostics (last episode)
    if args.diag is not None and last_diag is not None:
        os.makedirs(os.path.dirname(args.diag) or ".", exist_ok=True)
        np.savez_compressed(args.diag, **last_diag)
        print(f"Diagnostics saved to {args.diag} "
              f"(T={last_diag['a'].shape[0]}, Nm={last_diag['a'].shape[1]})")

    # Save trajectory NPZ (last episode)
    if args.save_traj and last_body_traj is not None:
        os.makedirs(os.path.dirname(args.save_traj) or ".", exist_ok=True)
        np.savez_compressed(
            args.save_traj,
            qpos=np.stack(last_qpos_traj).astype(np.float32),
            body_xpos=last_body_traj["xpos"],
            body_xquat=last_body_traj["xquat"],
            body_names=np.array(last_body_traj["names"]),
            control_hz=np.int32(env.control_hz),
        )
        print(
            f"Trajectory saved to {args.save_traj} "
            f"(T+1={last_body_traj['xpos'].shape[0]}, "
            f"nbody={last_body_traj['xpos'].shape[1]})"
        )

    # Render video of last episode
    if args.video and last_qpos_traj is not None:
        output_path = args.output or "output/eval.mp4"
        render_video(
            env, last_qpos_traj, output_path,
            width=args.width, height=args.height, fps=env.control_hz,
            cam_distance=args.cam_distance,
            cam_azimuth=args.cam_azimuth,
            cam_elevation=args.cam_elevation,
            track_body=args.track_body,
        )


if __name__ == "__main__":
    main()
