# tools/_rollout.py
"""Deterministic policy rollout shared by the USD bake and the GT harness.

Mirrors eval.py's inner muscle loop: per control step, the policy proposes an
action; then for `num_sim_per_control // 2` sub-iterations the muscle network
maps (muscle torques, desired torques) -> activations, which drive two sim
steps. Records per-control-step geom transforms, activations, qpos, rewards.
"""
from __future__ import annotations

import numpy as np

import torch
from torch import Tensor

_USE_CUDA = torch.cuda.is_available()


def rollout(env, model, muscle_model, max_steps, deterministic):
    import mujoco

    # Infer device from each model's current placement. Production callers
    # (e.g. the bake tool) move models to CUDA before calling rollout; in
    # tests, fresh nets start on CPU (SimulationNN does NOT self-cuda) or
    # CUDA (MuscleNN self-cudas in __init__ when use_cuda). Either way each
    # net's input tensors are sent to that net's own device, so they match.
    def _device_of(m):
        try:
            return next(m.parameters()).device
        except StopIteration:
            return torch.device("cuda" if _USE_CUDA else "cpu")

    _device = _device_of(model)
    _mm_device = _device_of(muscle_model) if muscle_model is not None else _device

    state = env.reset(rsi=False)
    mujoco.mj_forward(env.model, env.data)
    num_sim_per_control = env.simulation_hz // env.control_hz

    geom_xpos = [env.data.geom_xpos.copy()]
    geom_xmat = [env.data.geom_xmat.copy().reshape(-1, 3, 3)]
    activations = [np.zeros(env.num_muscles, dtype=np.float32)]
    qpos = [env.data.qpos.copy()]
    rewards = []

    for _ in range(max_steps):
        s = Tensor(state.reshape(1, -1).astype(np.float32)).to(_device)
        a_dist, _ = model(s)
        action = (a_dist.loc if deterministic else a_dist.sample())
        action = action.cpu().detach().numpy().squeeze()
        env.set_action(action)

        if env.use_muscle and muscle_model is not None:
            mt = Tensor(env.get_muscle_torques().reshape(1, -1).astype(np.float32)).to(_mm_device)
            for _ in range(num_sim_per_control // 2):
                dt = Tensor(env.get_desired_torques().reshape(1, -1).astype(np.float32)).to(_mm_device)
                act = muscle_model(mt, dt).cpu().detach().numpy().squeeze()
                env.set_activation_levels(act)
                env.step()
                env.step()
        else:
            env.steps_at_once()

        geom_xpos.append(env.data.geom_xpos.copy())
        geom_xmat.append(env.data.geom_xmat.copy().reshape(-1, 3, 3))
        activations.append(env.get_activation_levels().astype(np.float32))
        qpos.append(env.data.qpos.copy())

        if env.is_end_of_episode():
            break
        rewards.append(env.get_reward())
        state = env.get_state()

    return (
        np.stack(geom_xpos).astype(np.float32),
        np.stack(geom_xmat).astype(np.float32),
        np.stack(activations).astype(np.float32),
        np.stack(qpos).astype(np.float32),
        rewards,
    )
