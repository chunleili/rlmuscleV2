"""Constitutive comparison of the fiber constraints: stored energy and along-
fiber force versus the along-fiber stretch

    lambda = |F d| = sqrt(I4),   I4 = d^T S d = d^T F^T F d = |F d|^2

shown at three activation levels (a = 0, 0.5, 1).  Contraction ratio r = 0.4.
Plotted against stretch (not I4) so the anisotropic-ARAP energy appears as the
symmetric parabola it is; I4 = lambda^2 would warp it.

Per model at activation a (belly tissue, m = 0):
  plain C=|Fd|      : Psi = a * 1/2 lam^2,           f = a * lam           (target 0)
  aniso ARAP (ours) : Psi = 1/2 (lam - s*(a))^2,     f = lam - s*(a),  s*(a)=1-a r
  Hill-Millard      : Psi = a * int f_L,             f = a * f_L(lam)      (bell, peak lam=1)

The aniso ARAP well is passive at a=0 (symmetric about the rest length lam=1)
and slides toward shorter lengths as activation rises; the Millard/plain forces
scale in amplitude but keep their shape.

Run:
    .venv/bin/python tools/bicep_fl/energy_force_I4.py
"""
import os
import sys
import csv
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from muscle.engine import MillardCurves  # noqa: E402

HERE = os.path.dirname(__file__)
R = 0.4                                    # contraction ratio
XLIM = (0.4, 1.6)                          # along-fiber stretch, symmetric about 1
A_LEVELS = [0.0, 0.5, 1.0]

lam = np.linspace(0.4, 1.6, 500)           # along-fiber stretch
fL = MillardCurves().fl.eval(lam)
Emill_unit = np.concatenate([[0.0], np.cumsum(0.5 * (fL[1:] + fL[:-1]) * np.diff(lam))])  # int f_L


def curves(a):
    """(label, style, energy, force) for each model at activation a."""
    sstar = 1.0 - a * R
    return [
        ("plain $C=|Fd|$",     "C1--", a * 0.5 * lam ** 2,        a * lam),
        ("aniso ARAP (ours)",  "C0-",  0.5 * (lam - sstar) ** 2,  lam - sstar),
        ("Hill--Millard",      "C2-.", a * Emill_unit,            a * fL),
    ]


def sstar(a):
    return 1.0 - a * R


# --- CSV (all activations) ---
with open(os.path.join(HERE, "energy_force_I4.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    hdr = ["a", "lambda", "I4"]
    for lab, *_ in curves(1.0):
        key = lab.split()[0].strip("$")
        hdr += [f"E_{key}", f"F_{key}"]
    w.writerow(hdr)
    for a in A_LEVELS:
        for i in range(len(lam)):
            row = [a, f"{lam[i]:.4f}", f"{lam[i]**2:.4f}"]
            for _, _, E, F in curves(a):
                row += [f"{E[i]:.5f}", f"{F[i]:.5f}"]
            w.writerow(row)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

nrow = len(A_LEVELS)
fig, axes = plt.subplots(nrow, 2, figsize=(9, 3.0 * nrow), sharex=True)
E_YLIM = (-0.03, 1.05)
F_YLIM = (-0.75, 1.55)

for r, a in enumerate(A_LEVELS):
    axE, axF = axes[r]
    for lab, st, E, F in curves(a):
        axE.plot(lam, E, st, lw=2, label=lab.replace("--", "-"))
        axF.plot(lam, F, st, lw=2)
    for ax in (axE, axF):
        ax.axvline(1.0, color="gray", ls=":", lw=1.2)              # rest / optimal length
        ax.axvline(sstar(a), color="C0", ls=":", lw=1.2, alpha=0.6)  # aniso target s*(a)
        ax.grid(alpha=0.3)
        ax.set_xlim(*XLIM)
    axF.axhline(0.0, color="k", lw=0.6)
    axE.set_ylim(*E_YLIM); axF.set_ylim(*F_YLIM)
    axE.set_ylabel(f"$a={a:g}$\n\nstored energy $\\Psi$")
    axF.set_ylabel("PK1 stress  $\\partial\\Psi/\\partial\\mathbf{F}$")
    if r == 0:
        axE.set_title("Energy vs stretch")
        axF.set_title("PK1 stress vs stretch")
        axE.legend(fontsize=8, loc="upper center")
axes[-1][0].set_xlabel(r"along-fiber stretch  $\lambda_{\mathrm{f}}=|\mathbf{F}\mathbf{d}|=\sqrt{I_4}$")
axes[-1][1].set_xlabel(r"along-fiber stretch  $\lambda_{\mathrm{f}}=|\mathbf{F}\mathbf{d}|=\sqrt{I_4}$")

fig.tight_layout()
out = os.path.join(HERE, "energy_force_I4.png")
fig.savefig(out, dpi=150)
print("wrote", out)
