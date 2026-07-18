"""Before/after walk-vs-EMG figure: tracking-only baseline vs EMG-prior.

Two columns (Baseline | With EMG prior), four muscle rows. Each panel overlays
the model curve ("Ours", red) on the Moreira human mean (black) and +/-1 std
band (grey), peak-normalised. The only difference between columns is the EMG
prior (same corrected reference, same f0-weighted synergist-group comparison).

Usage:
    uv run python tools/plot_walk_vs_emg_ba.py \
        --before output/walk_vs_emg_baseline_grp/compare.csv \
        --after  output/walk_vs_emg_emgprior_final/compare.csv \
        -o output/walk_vs_emg_emgprior_final/before_after.pdf
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "VL":  "Vastus Lateralis\n(vs quadriceps)",
    "BF":  "Biceps Femoris\n(vs hamstrings)",
    "TA":  "Tibialis Anterior\n(vs dorsiflexors)",
    "GAL": "Gastrocnemius Lat.\n(vs triceps surae)",
}
MUSCLES = ["VL", "BF", "TA", "GAL"]


def _load(path):
    with open(path) as f:
        header = f.readline().lstrip("#").strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return {n: data[:, i] for i, n in enumerate(header)}


def _norm(y):
    m = float(np.max(y))
    return y / m if m > 1e-9 else y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()

    cols = {"without EMG prior": _load(a.before),
            "With EMG prior": _load(a.after)}
    fig, axes = plt.subplots(len(MUSCLES), 2, figsize=(8, 8), sharex=True)
    t = cols["With EMG prior"]["time_pct"]

    for ci, (title, col) in enumerate(cols.items()):
        for ri, m in enumerate(MUSCLES):
            ax = axes[ri, ci]
            mean = col[f"{m}_human_mean"]; peak = float(np.max(mean)) or 1.0
            mean_n = mean / peak; sd_n = col[f"{m}_human_std"] / peak
            rl = _norm(col[f"{m}_rl"])
            ax.fill_between(t, mean_n - sd_n, mean_n + sd_n, color="0.6", alpha=0.35,
                            label="EMG std (Moreira 2021, 13 subj)" if (ri == 0 and ci == 0) else None)
            ax.plot(t, mean_n, "k", lw=1.1,
                    label="EMG mean (Moreira 2021)" if (ri == 0 and ci == 0) else None)
            ax.plot(t, rl, color="tab:red", lw=1.4,
                    label="Ours" if (ri == 0 and ci == 0) else None)
            ax.set_ylim(0, 1.12)
            if ci == 0:
                ax.set_ylabel(LABELS[m], fontsize=8)
            else:
                ax.set_yticklabels([])
            if ri == 0:
                ax.set_title(title, fontsize=10)
    axes[0, 0].legend(fontsize=7, loc="upper right", frameon=False)
    for ci in range(2):
        axes[-1, ci].set_xlabel("% gait cycle (0 = right heel strike)")
    fig.tight_layout()
    fig.savefig(a.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
