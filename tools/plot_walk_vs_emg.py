"""Plot a walk-vs-EMG comparison from a compare.csv produced by
tools/validate_walk_vs_emg.py.

The CSV has one row per gait-cycle % and, per muscle, either:
  cohort mode  : <M>_human_mean, <M>_human_std, <M>_rl
  single-subj  : <M>_human, <M>_rl
Human columns are %MVC, rl is neural activation — each column is peak-normalised
here for shape/timing comparison.

Usage:
    uv run python tools/plot_walk_vs_emg.py \
        -i output/walk_vs_emg_cohort/compare.csv \
        -o output/walk_vs_emg_cohort/replot.png
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Friendly names. Each surface-EMG channel is compared to the f0-weighted
# activation of the corresponding synergist group (electrode reads a region;
# the redundant solver distributes within the group), so the model side is
# labelled as the group.
LABELS = {
    "VL": "Vastus Lateralis\n(vs quadriceps)",
    "BF": "Biceps Femoris\n(vs hamstrings)",
    "TA": "Tibialis Anterior\n(vs dorsiflexors)",
    "GAL": "Gastrocnemius Lat.\n(vs triceps surae)",
}


def _norm(y):
    m = float(np.max(y))
    return y / m if m > 1e-9 else y


def plot(csv_path: str, out_path: str):
    # Read header + data
    with open(csv_path) as f:
        header = f.readline().lstrip("#").strip().split(",")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    col = {name: data[:, i] for i, name in enumerate(header)}
    t = col["time_pct"]

    # Discover muscles from the column names (order preserved)
    muscles = []
    for name in header:
        if name.endswith("_rl"):
            muscles.append(name[:-3])
    cohort = any(f"{m}_human_mean" in col for m in muscles)

    fig, axes = plt.subplots(len(muscles), 1, figsize=(8, 2.2 * len(muscles)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for i, (ax, m) in enumerate(zip(axes, muscles)):
        rl = _norm(col[f"{m}_rl"])
        if cohort:
            mean = col[f"{m}_human_mean"]
            peak = float(np.max(mean)) or 1.0
            mean_n = mean / peak
            sd_n = col[f"{m}_human_std"] / peak
            ax.fill_between(t, mean_n - sd_n, mean_n + sd_n, color="0.6",
                            alpha=0.35, label="EMG std (Moreira 2021, 13 subj)")
            ax.plot(t, mean_n, "k", lw=1.2, label="EMG mean (Moreira 2021)")
        else:
            human = _norm(col[f"{m}_human"])
            ax.plot(t, human, "k", lw=1.5, label="EMG mean (Moreira 2021)")
            ax.fill_between(t, 0, human, color="0.7", alpha=0.4)
        ax.plot(t, rl, color="tab:red", lw=1.5, label="Ours")
        ax.set_ylabel(LABELS.get(m, m), fontsize=9)
        ax.set_ylim(0, 1.12)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right", ncol=1, frameon=False)
    axes[-1].set_xlabel("% gait cycle (0 = right heel strike)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({len(muscles)} muscles, cohort={cohort})")


def main():
    p = argparse.ArgumentParser(description="Re-plot walk-vs-EMG from compare.csv")
    p.add_argument("-i", "--input", required=True, help="compare.csv path")
    p.add_argument("-o", "--output", default="replot.png", help="output png")
    a = p.parse_args()
    plot(a.input, a.output)


if __name__ == "__main__":
    main()
