"""Paper figure: full-body muscle round-trip convergence, off vs hpbd MG.

Pure post-processing — reads the CSVs already dumped by hpbd_conv_curve.py
(output/hpbd_conv/conv_cycle_realframe_{off,hpbd}.csv) and renders a 2-panel
figure:
  left  — ARAP constraint residual C
  right — inverted-tet count
over the round-trip drive (rest -> far bone pose -> rest). No simulation here.

Two x-axes are emitted (same curves, different abscissa):
  * substep   — one point per XPBD substep (muscle-mg-converge.png)
  * iteration — substep x FINE_ITERS fine XPBD iters/substep, i.e. cumulative
                fine solver iterations (muscle-mg-converge-iter.png). Both
                variants use the same fine budget so the rescale is a fair,
                shape-preserving relabel of the substep axis.
"""
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_DIR = Path("output/hpbd_conv")            # sim-dumped source CSVs
OUT_DIR = Path("docs/images/mg-conv")         # figure + archived data live here
FINE_ITERS = 20                               # fine_iterations per substep (humanV2_hpbd_conv.json)

VARIANTS = [("conv_cycle_realframe_off.csv", "without MG", "C0"),
            ("conv_cycle_realframe_hpbd.csv", "with MG", "C3")]


def load(name):
    rows = []
    with open(CSV_DIR / name) as f:
        r = csv.reader(f)
        next(r)
        for s, res, inv in r:
            rows.append((int(s), float(res), int(inv)))
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2]


def render(xscale, xlabel, out):
    """One 2-panel figure with x = substep * xscale."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for name, label, c in VARIANTS:
        s, res, inv = load(name)
        x = s * xscale
        axL.plot(x, res, color=c, lw=2, label=label)
        axR.plot(x, inv, color=c, lw=2, label=label)
    axL.set(xlabel=xlabel, ylabel=r"constraint residual $C$", title="Residual")
    axR.set(xlabel=xlabel, ylabel="inverted tetrahedra", title="Inversions")
    for ax in (axL, axR):
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


OUT_DIR.mkdir(parents=True, exist_ok=True)
render(1, "substep", OUT_DIR / "muscle-mg-converge.png")
render(FINE_ITERS, "iteration", OUT_DIR / "muscle-mg-converge-iter.png")

# archive the source CSVs next to the figures
for name, _, _ in VARIANTS:
    shutil.copy(CSV_DIR / name, OUT_DIR / name)
print(f"archived {[n for n, _, _ in VARIANTS]} to {OUT_DIR}")
