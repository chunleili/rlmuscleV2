"""Locate where warp rollout diverges from CPU.

Run identical muscle+SPD closed-loop on CPU MassEnv and WarpVecEnv(num_envs=1),
report per-control-step max |Δqpos|, |Δqvel|, |Δxfrc|, |Δl_mt|. Stop at first
control step where divergence exceeds threshold or NaN appears.
"""

import json
import sys

import numpy as np

CONFIG = "data/rl_mass/configs/swim.json"
N_CTRL = 200
DIVERGE = 1.0


def main():
    from rl_mass.env import MassEnv
    from rl_mass.warp.vec_env import WarpVecEnv

    cpu_env = MassEnv(CONFIG)
    with open(CONFIG) as f:
        skel = json.load(f)["skeleton_file"]

    warp_env = WarpVecEnv(CONFIG, num_envs=1, seed=0)
    warp_env.reset(rsi=False)
    cpu_env.reset(rsi=False)

    # Match initial state.
    q0 = cpu_env.engine.get_qpos().astype(np.float32)
    v0 = cpu_env.engine.get_qvel().astype(np.float32)
    warp_env._engine.set_qpos(q0)
    warp_env._engine.set_qvel(v0)
    warp_env._engine.forward()

    rng = np.random.default_rng(7)
    act = rng.uniform(-0.05, 0.05, cpu_env.num_action).astype(np.float64)
    mus = rng.uniform(0.0, 0.2, cpu_env.num_muscles).astype(np.float64)

    cpu_env.set_action(act)
    cpu_env.set_activation_levels(mus)

    warp_env.set_actions(act[None, :].astype(np.float32))
    warp_env.set_activation_levels(mus[None, :].astype(np.float32))

    for ctrl in range(N_CTRL):
        cpu_env.steps_at_once()
        warp_env.steps_at_once()

        q_cpu = cpu_env.engine.get_qpos()
        v_cpu = cpu_env.engine.get_qvel()
        q_warp = warp_env._engine.get_qpos()[0]
        v_warp = warp_env._engine.get_qvel()[0]

        dq = np.max(np.abs(q_warp.astype(np.float64) - q_cpu))
        dv = np.max(np.abs(v_warp.astype(np.float64) - v_cpu))
        qnan = not np.all(np.isfinite(q_warp))
        print(f"ctrl {ctrl:2d}: |Δq|={dq:.4e}  |Δv|={dv:.4e}  warp_qnorm={np.linalg.norm(q_warp):.3e}  cpu_qnorm={np.linalg.norm(q_cpu):.3e}  nan={qnan}")

        if qnan or dq > DIVERGE or dv > DIVERGE:
            print("DIVERGE")
            # dump first 10 components
            print("  q_cpu[:10]", q_cpu[:10])
            print("  q_warp[:10]", q_warp[:10])
            break

        cpu_env.set_action(act)
        cpu_env.set_activation_levels(mus)
        warp_env.set_actions(act[None, :].astype(np.float32))
        warp_env.set_activation_levels(mus[None, :].astype(np.float32))

    warp_env.close()


if __name__ == "__main__":
    main()
