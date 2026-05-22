#!/usr/bin/env python3
"""Benchmark rl_mass warp backend core loops and rollout collection.

Examples:
    PYTHONPATH=src .venv/bin/python tools/profile_warp.py --config data/rl_mass/configs/swim.json
    PYTHONPATH=src .venv/bin/python tools/profile_warp.py --num-envs 1 4 8 16 --buffer-size 256 --json-out output/warp_profile.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Metric:
    name: str
    seconds: float
    gpu_util_avg: float | None
    gpu_util_max: float | None
    gpu_mem_mb_max: float | None


class GpuSampler:
    def __init__(self, device_index: int = 0, interval_s: float = 0.2):
        self.device_index = device_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[float, float]] = []
        self.available = self._probe()

    def _probe(self) -> bool:
        try:
            subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_index}",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={self.device_index}",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                util_s, mem_s = [x.strip() for x in out.split(",")]
                self.samples.append((float(util_s), float(mem_s)))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        if self.available:
            self._stop.clear()
            self.samples.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._thread = None

    def stats(self) -> tuple[float | None, float | None, float | None]:
        if not self.samples:
            return None, None, None
        util = [u for u, _ in self.samples]
        mem = [m for _, m in self.samples]
        return float(np.mean(util)), float(np.max(util)), float(np.max(mem))


def time_block(name: str, fn, gpu_index: int) -> Metric:
    sampler = GpuSampler(device_index=gpu_index)
    t0 = time.perf_counter()
    with sampler:
        fn()
    dt = time.perf_counter() - t0
    util_avg, util_max, mem_max = sampler.stats()
    return Metric(
        name=name,
        seconds=dt,
        gpu_util_avg=util_avg,
        gpu_util_max=util_max,
        gpu_mem_mb_max=mem_max,
    )


def bench_env(config: str, num_envs: int, rounds: int, gpu_index: int) -> dict:
    from rl_mass.warp.vec_env import WarpVecEnv

    env = WarpVecEnv(config, num_envs=num_envs, seed=0)
    try:
        env.reset(rsi=False)
        rng = np.random.default_rng(123)
        actions = rng.normal(size=(num_envs, env.num_action)).astype(np.float32) * 0.05
        activations = rng.uniform(0.0, 0.3, size=(num_envs, env.num_muscles)).astype(np.float32)
        env.set_actions(actions)
        env.set_activation_levels(activations)

        metrics: list[Metric] = []
        metrics.append(time_block("get_states", lambda: env.get_states(), gpu_index))
        metrics.append(time_block("get_muscle_torques", lambda: env.get_muscle_torques(), gpu_index))
        metrics.append(time_block("get_desired_torques", lambda: env.get_desired_torques(), gpu_index))
        metrics.append(time_block("steps_2", lambda: [env.steps(2) for _ in range(rounds)], gpu_index))
        metrics.append(
            time_block(
                "steps_at_once",
                lambda: [env.steps_at_once() for _ in range(max(1, rounds // 2))],
                gpu_index,
            )
        )
        return {
            "num_envs": num_envs,
            "metrics": [asdict(m) for m in metrics],
        }
    finally:
        env.close()


def bench_rollout(
    config: str,
    num_envs: int,
    buffer_size: int,
    gpu_index: int,
) -> dict:
    from rl_mass.train import PPO

    ppo = PPO(
        config,
        num_envs=num_envs,
        vec_env_mode="warp",
        seed=0,
        packed_muscle_loop=False,
        reward_csv_path=None,
        nn_dir="output/profile-warp-nn",
    )
    try:
        ppo.buffer_size = buffer_size
        metric = time_block("generate_transitions", lambda: ppo.GenerateTransitions(), gpu_index)
        completed_episode_tuples = sum(
            len(ep.data) for ep in getattr(ppo, "total_episodes", [])
        )
        return {
            "num_envs": num_envs,
            "buffer_size": buffer_size,
            "completed_episode_tuples": completed_episode_tuples,
            "num_episode": len(getattr(ppo, "total_episodes", [])),
            "metric": asdict(metric),
        }
    finally:
        ppo.env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/rl_mass/configs/swim.json")
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=256)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    results = {
        "config": args.config,
        "buffer_size": args.buffer_size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env_bench": [],
        "rollout_bench": [],
    }

    for num_envs in args.num_envs:
        env_row = bench_env(args.config, num_envs, args.rounds, args.gpu_index)
        results["env_bench"].append(env_row)
        print(f"[env] num_envs={num_envs}")
        for metric in env_row["metrics"]:
            print(
                f"  {metric['name']:<18} {metric['seconds']:.3f}s"
                f"  gpu_avg={metric['gpu_util_avg']}  gpu_max={metric['gpu_util_max']}"
                f"  mem_max={metric['gpu_mem_mb_max']}"
            )

        if not args.skip_rollout:
            rollout_row = bench_rollout(args.config, num_envs, args.buffer_size, args.gpu_index)
            results["rollout_bench"].append(rollout_row)
            metric = rollout_row["metric"]
            print(
                f"[rollout] num_envs={num_envs} buffer={args.buffer_size}"
                f"  {metric['seconds']:.3f}s"
                f"  gpu_avg={metric['gpu_util_avg']}  gpu_max={metric['gpu_util_max']}"
                f"  mem_max={metric['gpu_mem_mb_max']}"
            )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
