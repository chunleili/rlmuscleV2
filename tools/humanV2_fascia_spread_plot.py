"""Plot the fascia gather metric — muscle lateral spread, no-fascia vs fascia.
Reads the CSV dumped by tools.humanV2_fascia_ablation (no simulation here).

left  — mean perpendicular spread of the muscle cloud to its long axis (mm),
        no-fascia vs fascia; shaded band = up to the p95 spread.
right — gather = (off - on) / off of the mean spread, per frame (%). Positive
        = fascia pulls the muscle cloud inward (聚拢).

Run: .venv/bin/python -m tools.humanV2_fascia_spread_plot
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "output/humanV2/fascia/fascia_spread.csv"
OUT = "output/humanV2/fascia/fascia_spread.png"

f, pmo, pmn, p95o, p95n = [], [], [], [], []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        f.append(int(r["frame"]))
        pmo.append(float(r["perp_mean_off_m"]) * 1e3)   # mm
        pmn.append(float(r["perp_mean_on_m"]) * 1e3)
        p95o.append(float(r["perp_p95_off_m"]) * 1e3)
        p95n.append(float(r["perp_p95_on_m"]) * 1e3)
f = np.array(f); pmo = np.array(pmo); pmn = np.array(pmn)
p95o = np.array(p95o); p95n = np.array(p95n)
gather = (pmo - pmn) / pmo * 100.0

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
# shade the gather gap between the two mean-spread curves (fascia sits inside)
ax[0].fill_between(f, pmn, pmo, color="C2", alpha=0.18, label="gather")
ax[0].plot(f, pmo, label="No fascia", lw=1.8, color="C0")
ax[0].plot(f, pmn, label="Fascia (ours)", lw=1.8, color="C3")
ax[0].set(xlabel="frame", ylabel="muscle lateral spread (mm)", title="Muscle spread")
ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(f, gather, lw=1.8, color="C2")
ax[1].axhline(0, color="k", lw=0.6)
ax[1].fill_between(f, 0, gather, where=gather > 0, color="C2", alpha=0.15)
ax[1].set(xlabel="frame", ylabel="gather vs no-fascia (%)", title="Fascia gather")
ax[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"WROTE {OUT}  spread f{f[-1]}: no-fascia {pmo[-1]:.1f}mm -> fascia {pmn[-1]:.1f}mm "
      f"(gather {gather[-1]:+.2f}%, peak {gather.max():+.2f}%)")
