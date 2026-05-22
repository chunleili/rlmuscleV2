"""cProfile a few control steps of the warp rollout at num_envs=1024.

Print top-N hot functions by tottime, then by cumtime. Goal: see whether
Python per-env loops, mujoco_warp launches, or nefc-overflow fallbacks
dominate. Runs ~30s total.
"""
import argparse
import cProfile
import io
import json
import pstats
import time

import numpy as np

import rl_mass.warp  # noqa: F401 — registers warp backend
from rl_mass.warp.vec_env import WarpVecEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/rl_mass/configs/walk.json")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--ctrl-steps", type=int, default=128,
                    help="control steps to profile (after warmup); 128 = 1 PPO iter")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--no-profile", action="store_true",
                    help="skip cProfile (faster, wall-clock only)")
    args = ap.parse_args()

    print(f"config={args.config} num_envs={args.num_envs} ctrl_steps={args.ctrl_steps}")

    t0 = time.time()
    env = WarpVecEnv(args.config, num_envs=args.num_envs)
    env.reset(rsi=True)
    num_action = env.envs[0].num_action
    n_substeps = env.num_steps
    print(f"init={time.time()-t0:.2f}s num_action={num_action} substeps_per_ctrl={n_substeps}")

    rng = np.random.default_rng(0)

    def random_actions():
        return rng.standard_normal((args.num_envs, num_action)).astype(np.float32) * 0.1

    # warmup
    for _ in range(args.warmup):
        env.set_actions(random_actions())
        env.steps_at_once()
    print(f"warmup done at t={time.time()-t0:.2f}s")

    # profile (cProfile is optional — skip for fastest wall-clock)
    pr = cProfile.Profile() if not args.no_profile else None
    t_start = time.time()
    if pr is not None:
        pr.enable()
    for _ in range(args.ctrl_steps):
        env.set_actions(random_actions())
        env.steps_at_once()
    out = env.gather_muscle_tuples()
    if pr is not None:
        pr.disable()
    elapsed = time.time() - t_start

    if isinstance(out, dict):
        shapes = {k: tuple(v.shape) for k, v in out.items()}
    else:
        shapes = f"list len={len(out)}"
    print(f"\n=== {args.ctrl_steps} ctrl steps wall={elapsed:.3f}s "
          f"= {elapsed/args.ctrl_steps*1000:.1f} ms/step ===")
    print(f"gather_muscle_tuples returned: {shapes}")
    if args.ctrl_steps != 128:
        full_iter_ctrl = 128
        est_iter = elapsed / args.ctrl_steps * full_iter_ctrl
        print(f"--> extrapolated full iter (128 ctrl): {est_iter:.1f}s")
    print()

    if pr is not None:
        for sort_key in ("tottime", "cumtime"):
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
            ps.print_stats(args.top)
            print(f"--- top {args.top} by {sort_key} ---")
            print(s.getvalue())


if __name__ == "__main__":
    main()
