"""Plot training reward curve from train_rewards.csv.

Usage:
    PYTHONPATH=. uv run python rl_mass/plot_rewards.py \
        --csv output/rl_mass/train_rewards.csv \
        -o docs/imgs/rl_mass/train_rewards.png
"""

from __future__ import annotations

import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser(description="Plot MASS training reward curve")
    parser.add_argument("--csv", default="output/rl_mass/train_rewards.csv")
    parser.add_argument("-o", "--output", default="output/rl_mass/train_rewards.png")
    parser.add_argument("--title", default="RL-MASS training (P2 packed subproc, num_envs=16)")
    args = parser.parse_args()

    iters, avg_returns, max_returns, elapsed = [], [], [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iter"]))
            avg_returns.append(float(row["avg_return"]))
            max_returns.append(float(row["max_return"]))
            elapsed.append(int(row["elapsed_s"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(iters, avg_returns, "-o", markersize=3, linewidth=1.2, label="Avg return / episode")
    ax1.plot(iters, max_returns, "--", linewidth=1.2, color="tab:orange", label="Max avg return (best)")
    ax1.set_xlabel("PPO iteration")
    ax1.set_ylabel("Return")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Secondary axis: cumulative wall-clock minutes
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    # Map iter ticks to minutes by interpolation
    if iters:
        major_iters = ax1.get_xticks()
        minute_labels = []
        for it in major_iters:
            if 0 <= it <= iters[-1] and iters:
                # linear interp
                idx = min(range(len(iters)), key=lambda i: abs(iters[i] - it))
                minute_labels.append(f"{elapsed[idx] // 60}m")
            else:
                minute_labels.append("")
        ax2.set_xticks(major_iters)
        ax2.set_xticklabels(minute_labels)
        ax2.set_xlabel("wall-clock time (min)")

    plt.title(args.title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=120)
    print(f"Saved {args.output}")
    print(f"  iters: {iters[0]}..{iters[-1]}  (N={len(iters)})")
    print(f"  avg_return: first={avg_returns[0]:.3f}  last={avg_returns[-1]:.3f}  max={max(avg_returns):.3f}")
    print(f"  wall clock: {elapsed[-1]}s ({elapsed[-1]/60:.1f} min)")


if __name__ == "__main__":
    main()
