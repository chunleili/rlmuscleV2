"""Dump OpenSim Millard2012 bicep force-length curves to CSV (reference).

Run inside the `opensim` conda env (OpenSim 4.6):
    conda run -n opensim python tools/bicep_fl/opensim_ref.py

Emits tools/bicep_fl/opensim_ref.csv with columns:
    lambda, f_active, f_passive
where lambda = normalized fiber length (l_m / l_opt), and forces are
normalized (units of F_max). These are OpenSim's own default Millard 2012
ActiveForceLengthCurve / FiberForceLengthCurve evaluated on a grid -- the
ground-truth reference for the bicep active/passive force-length law.
"""
import csv
import os

import numpy as np
import opensim as osim

OUT = os.path.join(os.path.dirname(__file__), "opensim_ref.csv")


def calc(curve, x):
    """Evaluate an OpenSim curve at scalar x via SimTK::Vector(1)."""
    v = osim.Vector(1, float(x))
    return curve.calcValue(v)


def main():
    # OpenSim Millard 2012 defaults (same params the V2 port claims to match).
    fal = osim.ActiveForceLengthCurve()   # active force-length
    fpe = osim.FiberForceLengthCurve()    # passive fiber force-length

    lam = np.linspace(0.30, 1.90, 321)
    rows = []
    for x in lam:
        rows.append((x, calc(fal, x), calc(fpe, x)))

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "f_active", "f_passive"])
        w.writerows(rows)
    print(f"opensim {osim.__version__}: wrote {len(rows)} rows -> {OUT}")
    # quick sanity print
    for x in (0.4441, 0.73, 1.0, 1.3, 1.8123):
        print(f"  lambda={x:.4f}  f_act={calc(fal, x):.4f}  f_pas={calc(fpe, x):.4f}")


if __name__ == "__main__":
    main()
