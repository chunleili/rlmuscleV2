"""Re-render output/bicep/fiber4_act_len.png from the saved CSV (no resim).

Reads output/bicep/fiber4_act_len.csv (columns: activation, {model}_axial,
{model}_radial) and redraws the 2-panel activation-vs-length/girth figure with
the same styles/labels as fiber4.py. Use this to tweak legend text without
re-running the CUDA simulation.

    .venv/bin/python tools/bicep_fl/replot_act_len.py
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "output", "bicep")

MODELS = ["nofiber", "plain", "aniso_arap", "millard"]
LABELS = {"nofiber": "no fiber", "plain": "plain $C=|Fd|$",
          "aniso_arap": "anisotropic ARAP", "millard": "Hill--Millard energy"}
STYLES = {"nofiber": "k:", "plain": "C1--", "aniso_arap": "C0-", "millard": "C2-."}


def main():
    acts, axial, radial = [], {m: [] for m in MODELS}, {m: [] for m in MODELS}
    with open(os.path.join(OUT, "fiber4_act_len.csv")) as fh:
        for row in csv.DictReader(fh):
            acts.append(float(row["activation"]))
            for m in MODELS:
                axial[m].append(float(row[f"{m}_axial"]))
                radial[m].append(float(row[f"{m}_radial"]))
    acts = np.asarray(acts)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 3.6))
    for m in MODELS:
        lab = LABELS[m].replace("--", "-")
        axL.plot(acts, 100 * (np.asarray(axial[m]) - 1), STYLES[m], lw=2, label=lab)
        axR.plot(acts, 100 * (np.asarray(radial[m]) - 1), STYLES[m], lw=2, label=lab)
    axL.set_xlabel("activation $a$"); axL.set_ylabel("axial length change (\\%)")
    axL.set_title("Muscle shortening"); axL.grid(alpha=0.3); axL.legend(fontsize=8)
    axR.set_xlabel("activation $a$"); axR.set_ylabel("radial girth change (\\%)")
    axR.set_title("Muscle bulging"); axR.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT, "fiber4_act_len.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
