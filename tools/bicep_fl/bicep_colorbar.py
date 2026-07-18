"""Standalone jet colorbar for the biceps displacement USDs.

Vertical jet (blue->cyan->green->yellow->red) bar with an arrow cap at the top (over-range),
numeric tick labels in mm, transparent background, no descriptive text. The
scale (0..VMAX mm) is the SAME one baked into fiber4_anim_usd.py and
plys_to_usd.py, so this bar reads the deformed-biceps colours directly.

    .venv/bin/python tools/bicep_fl/bicep_colorbar.py [--vmax 50] [--step 10]
Output: output/bicep/colorbar_bicep_dist.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "output", "bicep")
os.makedirs(OUT, exist_ok=True)

STOPS = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], np.float32)  # jet
VMAX_MM = 50.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmax", type=float, default=VMAX_MM,
                    help="displacement (mm) at the top of the jet scale")
    ap.add_argument("--step", type=float, default=10.0, help="tick spacing (mm)")
    ap.add_argument("--out", default=os.path.join(OUT, "colorbar_bicep_dist.png"))
    args = ap.parse_args()

    cmap = LinearSegmentedColormap.from_list("vsjet", list(map(tuple, STOPS)), N=256)
    sm = ScalarMappable(norm=Normalize(0, args.vmax), cmap=cmap)
    ticks = list(np.arange(0, args.vmax + 1e-6, args.step))

    fig = plt.figure(figsize=(1.1, 4.8))
    cax = fig.add_axes([0.08, 0.05, 0.32, 0.90])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical", extend="max", ticks=ticks)
    cb.ax.tick_params(labelsize=18, length=7, width=1.4)
    # tick labels as plain integers (mm), no unit text — emil style
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    fig.savefig(args.out, dpi=300, bbox_inches="tight", transparent=True)
    print(f"wrote {args.out}  (jet 0..{args.vmax:.0f}mm, step {args.step:g})")


if __name__ == "__main__":
    main()
