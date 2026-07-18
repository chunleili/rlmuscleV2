"""Combined 2x2 reward plot for the 4 concurrent training tasks."""
from __future__ import annotations
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = [
    ("swim-ellipsoid", "output/swim-ellipsoid/runs/20260522_152506-nid011212-n64/rewards.csv"),
    ("kick-preroll",   "output/kick-preroll/runs/20260522_152506-nid011212-n64/rewards.csv"),
    ("walk",           "output/walk/runs/20260522_174007-nid011212-n64-gpu3/rewards.csv"),
    ("squat", "output/overhead-squat/runs/20260522_184139-nid011212-n64-heightfix-posecutoff/rewards.csv"),
]


def load(path):
    iters, avg_r, max_r, elapsed = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iter"]))
            avg_r.append(float(row["avg_return"]))
            max_r.append(float(row["max_return"]))
            elapsed.append(int(row["elapsed_s"]))
    return iters, avg_r, max_r, elapsed


# Per-task panel
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
for ax, (name, csv_path) in zip(axes.flat, TASKS):
    iters, avg_r, max_r, elapsed = load(csv_path)
    ax.plot(iters, avg_r, "-", lw=0.8, alpha=0.6, label="avg_return")
    ax.plot(iters, max_r, "--", lw=1.2, color="tab:orange", label="max_return")
    ax.set_title(f"{name}   (iter {iters[-1]}, {elapsed[-1]/3600:.1f}h, max={max(max_r):.2f})")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
fig.suptitle("rlmuscleV2 — 4 concurrent trainings on nid011212", fontsize=13)
fig.tight_layout()
out = "output/all_tasks_rewards.png"
fig.savefig(out, dpi=120)
print(f"saved {out}")

# Combined avg_return overlay (rescaled so all curves are visible)
fig2, ax = plt.subplots(figsize=(11, 5))
colors = ["tab:blue", "tab:green", "tab:red", "tab:purple"]
for (name, csv_path), c in zip(TASKS, colors):
    iters, avg_r, max_r, _ = load(csv_path)
    ax.plot(iters, max_r, "-", lw=1.4, color=c, label=f"{name} (max={max(max_r):.1f})")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("max_return")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")
ax.set_title("max_return over PPO iterations — 4 tasks")
fig2.tight_layout()
out2 = "output/all_tasks_max_return.png"
fig2.savefig(out2, dpi=120)
print(f"saved {out2}")
