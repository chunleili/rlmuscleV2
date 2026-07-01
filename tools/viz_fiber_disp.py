"""Headless (no-VNC) visualization of humanV2 biceps fiber deformation.

Loads probe npz outputs (P0 passive, P1 active, sel_verts) and renders a
multi-panel 3D scatter of the biceps vertices colored by per-vertex
displacement magnitude (mm) under each condition, so the contraction (or its
absence) is visible as an image.

    python tools/viz_fiber_disp.py out.png a.npz "label a" b.npz "label b" ...
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out = sys.argv[1]
    items = sys.argv[2:]
    pairs = [(items[i], items[i + 1]) for i in range(0, len(items), 2)]

    # shared color scale across panels
    dmaxes = []
    data = []
    for path, label in pairs:
        d = np.load(path)
        sv = d["sel_verts"]
        p0, p1 = d["P0"][sv], d["P1"][sv]
        disp = np.linalg.norm(p1 - p0, axis=1) * 1000.0   # mm
        data.append((p1, disp, label, float(disp.max()), float(disp.mean())))
        dmaxes.append(float(disp.max()))
    vmax = max(dmaxes)

    n = len(data)
    fig = plt.figure(figsize=(6 * n, 6))
    for k, (p1, disp, label, dmx, dmn) in enumerate(data):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        sc = ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2], c=disp, cmap="jet",
                        vmin=0, vmax=vmax, s=3, depthshade=False)
        ax.set_title(f"{label}\nmax={dmx:.2f}mm  mean={dmn:.2f}mm", fontsize=10)
        ax.set_box_aspect((1, 1, 1))
        # equalize axes around the biceps
        c = p1.mean(0)
        r = float(np.abs(p1 - c).max())
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=12, azim=-70)
    cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.6, pad=0.02)
    cbar.set_label("vertex displacement act1 - act0 (mm)")
    fig.suptitle("humanV2 biceps brachii: fiber activation deformation "
                 "(static bone, GPU Jacobi)", fontsize=13)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}  (vmax={vmax:.2f}mm)")


if __name__ == "__main__":
    main()
