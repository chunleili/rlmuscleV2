"""Plot Emil muscle volume over time — solver (physics) vs LBS (skinning).
Reads the CSV dumped by tools.emil_lbs_compare (no simulation here).

Run: .venv/bin/python -m tools.emil_volume_plot
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "output/emil/emil_volume.csv"
OUT = "output/emil/emil_volume.png"

f, vs, vl = [], [], []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        f.append(int(r["frame"]))
        vs.append(float(r["vol_solver_m3"]) * 1e3)   # L
        vl.append(float(r["vol_lbs_m3"]) * 1e3)
f, vs, vl = np.array(f), np.array(vs), np.array(vl)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(f, vs, label="solver (physics)", lw=1.8)
ax[0].plot(f, vl, label="LBS (skinning)", lw=1.8, ls="--")
ax[0].set(xlabel="frame", ylabel="muscle volume (L)", title="Muscle volume")
ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(f, (vs / vs[0] - 1) * 100, label="solver", lw=1.8)
ax[1].plot(f, (vl / vl[0] - 1) * 100, label="LBS", lw=1.8, ls="--")
ax[1].axhline(0, color="k", lw=0.6)
ax[1].set(xlabel="frame", ylabel="volume change vs rest (%)", title="Volume preservation")
ax[1].legend(); ax[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"WROTE {OUT}  solver {vs[0]:.2f}->{vs[-1]:.2f}L ({(vs[-1]/vs[0]-1)*100:+.2f}%)  "
      f"LBS {vl[0]:.2f}->{vl[-1]:.2f}L ({(vl[-1]/vl[0]-1)*100:+.2f}%)")
