"""Bicep force-length comparison: Millard fiber-constraint vs plain F@d, vs OpenSim.

Goal: show the `fibermillard` constraint reproduces the OpenSim Millard 2012
force-length law, while the plain MASS "F@d" fiber model (force = f * direction,
with Gaussian g_al / exponential g_pl) does NOT -- a fundamentally different
active force-length profile.

Three force-length laws, all as a function of normalized fiber length
lambda = l_m / l_opt, at full activation a = 1:

  * OpenSim   : default Millard2012 ActiveForceLengthCurve / FiberForceLengthCurve
                (read from opensim_ref.csv, produced by opensim_ref.py)
  * Millard   : V2 `fibermillard` constraint curves -- MillardCurves().fl / .fpe
                (the energy-constitutive model used by solve_tetfibermillard_kernel)
  * F@d (MASS): V2 `fiber` line-muscle force law -- g_al / g_pl from rl_mass.muscle
                (force projected along fiber direction; src/rl_mass/muscle.py)

Run in the project uv env:
    uv run python tools/bicep_fl/compare.py
"""
import csv
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

from muscle.muscle_warp import MillardCurves           # noqa: E402
from rl_mass.muscle import g_al, g_pl                  # noqa: E402

HERE = os.path.dirname(__file__)
REF_CSV = os.path.join(HERE, "opensim_ref.csv")
OUT_PNG = os.path.join(HERE, "bicep_fl_compare.png")
OUT_CSV = os.path.join(HERE, "bicep_fl_compare.csv")


def load_opensim_ref():
    lam, fa, fp = [], [], []
    with open(REF_CSV) as f:
        for r in csv.DictReader(f):
            lam.append(float(r["lambda"]))
            fa.append(float(r["f_active"]))
            fp.append(float(r["f_passive"]))
    return np.array(lam), np.array(fa), np.array(fp)


def main():
    lam, os_act, os_pas = load_opensim_ref()

    # --- Millard (V2 fibermillard constraint curves) ---
    mc = MillardCurves()
    mil_act = mc.fl.eval(lam)
    mil_pas = mc.fpe.eval(lam)

    # --- F@d plain fiber (MASS line muscle) ---
    fad_act = np.array([g_al(x) for x in lam])
    fad_pas = np.array([g_pl(x) for x in lam])

    # --- error metrics vs OpenSim reference (active limb where it matters) ---
    def stats(name, y):
        d = y - os_act
        rmse = float(np.sqrt(np.mean(d ** 2)))
        mae = float(np.max(np.abs(d)))
        print(f"  {name:14s} active vs OpenSim:  RMSE={rmse:.4f}  max|err|={mae:.4f}")
        return rmse, mae

    print("Force-length comparison (active, normalized, a=1):")
    stats("Millard(V2)", mil_act)
    stats("F@d (MASS)", fad_act)

    # peak-location / shape summary
    print("\nShape summary:")
    print(f"  OpenSim   peak={os_act.max():.3f} at lambda={lam[os_act.argmax()]:.3f}; "
          f"f_act(0.44)={np.interp(0.4441, lam, os_act):.3f}, f_act(1.81)={np.interp(1.8123, lam, os_act):.3f}")
    print(f"  Millard   peak={mil_act.max():.3f} at lambda={lam[mil_act.argmax()]:.3f}; "
          f"f_act(0.44)={np.interp(0.4441, lam, mil_act):.3f}, f_act(1.81)={np.interp(1.8123, lam, mil_act):.3f}")
    print(f"  F@d       peak={fad_act.max():.3f} at lambda={lam[fad_act.argmax()]:.3f}; "
          f"f_act(0.44)={np.interp(0.4441, lam, fad_act):.3f}, f_act(1.81)={np.interp(1.8123, lam, fad_act):.3f}")

    # --- dump combined CSV ---
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda",
                    "os_active", "os_passive",
                    "millard_active", "millard_passive",
                    "fad_active", "fad_passive"])
        for i in range(len(lam)):
            w.writerow([lam[i], os_act[i], os_pas[i],
                        mil_act[i], mil_pas[i], fad_act[i], fad_pas[i]])
    print(f"\nwrote {OUT_CSV}")

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
    })
    from matplotlib.ticker import MaxNLocator

    fig, ax0 = plt.subplots(1, 1, figsize=(8, 6.5))

    ax0.plot(lam, os_act, "k-", lw=3, alpha=0.4, label="OpenSim")
    ax0.plot(lam, mil_act, "C0--", lw=2, label="Ours")
    ax0.plot(lam, fad_act, "C3-", lw=2, label="Plain Fiber")
    ax0.axvline(1.0, color="gray", ls=":", lw=1)
    ax0.set_title("Active force-length  (a = 1)")
    ax0.set_xlabel("normalized fiber length")
    ax0.set_ylabel(r"normalized active force  $f_L$")
    ax0.legend()

    # --- passive plot (disabled; backup) ---
    # ax[1].plot(lam, os_pas, "k-", lw=3, alpha=0.4, label="OpenSim")
    # ax[1].plot(lam, mil_pas, "C0--", lw=2, label="Ours")
    # ax[1].plot(lam, fad_pas, "C3-", lw=2, label="Plain Fiber")
    # ax[1].axvline(1.0, color="gray", ls=":", lw=1)
    # ax[1].set_title("Passive force-length")
    # ax[1].set_xlabel("normalized fiber length")
    # ax[1].set_ylabel(r"normalized passive force  $f_{PE}$")
    # ax[1].set_ylim(-0.05, 1.6)
    # ax[1].legend()

    ax0.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=5))

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
