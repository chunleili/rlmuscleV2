"""Visual mesh comparison: bicep deformed under `fiber` (plain F@d) vs `fibermillard`.

Same bicep tet mesh, same attach constraints, same activation ramp -- only the
active fiber constraint type differs. Runs both, exports the deformed surface
to PLY, and renders a side-by-side displacement comparison.

Run (project uv env):
    uv run python tools/bicep_fl/mesh_compare.py [--nsteps 600]

Outputs (in tools/bicep_fl/):
    mesh_fiber.ply          deformed surface, plain F@d fiber constraint
    mesh_fibermillard.ply   deformed surface, Millard fiber constraint
    mesh_rest.ply           undeformed reference
    mesh_compare.png        3-panel render colored by displacement from rest
"""
import argparse
import copy
import os

import numpy as np
import warp as wp

from muscle.util.util import load_config, setup_logging
from muscle.muscle_warp import MuscleSim, activation_ramp, fill_float_kernel
from muscle.util.mesh_io import save_ply

HERE = os.path.dirname(__file__)
CFG = "data/muscle/config/bicep.json"

# fibermillard active constraint (from bicep_fibermillard_coupled.json)
MILLARD_CON = {
    "type": "fibermillard",
    "name": "fiber_millard_active",
    "stiffness": 10000.0,
    "sigma0": 300000.0,
    "contraction_factor": 0.4,
    "dampingratio": 0.1,
}

# Stability: each substep is one XPBD jacobi sweep (step() has no inner solver
# loop), so more substeps = more jacobi iterations = converged, non-distorted
# mesh. No post-smooth / tet-repair (those artificially modify geometry) -- just
# converge harder. Applied to BOTH runs so only the constraint type differs.
NUM_SUBSTEPS = 50
TENDON_PIN_THRESHOLD = 0.9  # freeze verts with tendonmask above this


def apply_stab(cfg):
    cfg.num_substeps = NUM_SUBSTEPS


def freeze_tendon(sim):
    """Pin the tendon (tendonmask > threshold) kinematic.

    The bicep mesh anchors only ~13 tip verts via the muscleEnd attach, but ~171
    verts are tendon (tendonmask~1). The free tendon flops when the belly pulls on
    it -- and Millard pulls harder than plain F@d, so its tendon floppped most
    (that was the 'distortion', not a stiffness problem). Freezing the whole
    tendon gives a clean isometric contraction: ends fixed, belly bulges.
    """
    tm = sim.v_tendonmask_np
    if tm is None:
        return 0
    st = sim.stopped.numpy()
    pin = tm > TENDON_PIN_THRESHOLD
    st[pin] = 1
    sim.stopped.assign(st)
    return int(pin.sum())


def run(cfg, nsteps, act=1.0):
    """Hold activation at a constant `act` for all steps and settle.

    Step input (full a=1 from the start) maximizes the contraction difference.
    NOTE: do not use activation_ramp() -- it returns to 0 at t=1 (it is a
    0->0.5->1->0.7->0.3->0 excitation profile), sampling the relaxed muscle.
    rest = a=0 (mesh_rest.ply), active = a=1.
    """
    sim = MuscleSim(cfg)
    n_pin = freeze_tendon(sim)
    print(f"  froze {n_pin} tendon verts (tendonmask>{TENDON_PIN_THRESHOLD})")
    for step in range(1, nsteps + 1):
        wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
                  inputs=[sim.activation, float(act)])
        sim.step()
    pos = sim.pos.numpy().copy()
    faces = np.asarray(sim.surface_tris)
    return pos, faces, sim.pos0_np.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, default=600)
    args = ap.parse_args()

    setup_logging()
    wp.init()
    wp.set_device("cpu")

    # --- plain F@d fiber ---
    cfg_f = load_config(CFG)
    cfg_f.gui = False
    cfg_f.render_mode = None
    cfg_f.nsteps = args.nsteps
    apply_stab(cfg_f)
    pos_f, faces, rest = run(cfg_f, args.nsteps)

    # --- Millard fiber constraint (swap constraint #1) ---
    cfg_m = load_config(CFG)
    cfg_m.gui = False
    cfg_m.render_mode = None
    cfg_m.nsteps = args.nsteps
    apply_stab(cfg_m)
    cfg_m.constraints = copy.deepcopy(cfg_m.constraints)
    idx = next(i for i, c in enumerate(cfg_m.constraints) if c.get("type") == "fiber")
    cfg_m.constraints[idx] = copy.deepcopy(MILLARD_CON)
    pos_m, _, _ = run(cfg_m, args.nsteps)

    # --- export PLY ---
    save_ply(os.path.join(HERE, "mesh_rest.ply"), rest, faces)
    save_ply(os.path.join(HERE, "mesh_fiber.ply"), pos_f, faces)
    save_ply(os.path.join(HERE, "mesh_fibermillard.ply"), pos_m, faces)

    # --- displacement metrics ---
    d_f = np.linalg.norm(pos_f - rest, axis=1)
    d_m = np.linalg.norm(pos_m - rest, axis=1)
    d_fm = np.linalg.norm(pos_f - pos_m, axis=1)  # model-to-model difference
    bbox = np.linalg.norm(rest.max(0) - rest.min(0))
    print(f"mesh bbox diag = {bbox:.4f}")
    print(f"plain F@d   displacement from rest:  mean={d_f.mean():.4f}  max={d_f.max():.4f}")
    print(f"Millard     displacement from rest:  mean={d_m.mean():.4f}  max={d_m.max():.4f}")
    print(f"F@d vs Millard vertex difference:    mean={d_fm.mean():.4f}  max={d_fm.max():.4f}  "
          f"({100*d_fm.mean()/bbox:.2f}% of bbox diag)")

    # --- render 3-panel ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def panel(ax, pos, scalar, title, vmax):
        tris = pos[faces]
        fc = plt.cm.viridis(np.clip(scalar[faces].mean(1) / vmax, 0, 1))
        col = Poly3DCollection(tris, facecolors=fc, edgecolors="k", linewidths=0.05, alpha=1.0)
        ax.add_collection3d(col)
        mn, mx = pos.min(0), pos.max(0)
        c = (mn + mx) / 2
        r = (mx - mn).max() / 2
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=-70)

    vmax = max(d_f.max(), d_m.max())
    fig = plt.figure(figsize=(15, 5))
    a0 = fig.add_subplot(131, projection="3d")
    panel(a0, pos_f, d_f, f"plain F@d fiber\nmax disp {d_f.max():.3f}", vmax)
    a1 = fig.add_subplot(132, projection="3d")
    panel(a1, pos_m, d_m, f"Millard fiber-constraint\nmax disp {d_m.max():.3f}", vmax)
    a2 = fig.add_subplot(133, projection="3d")
    panel(a2, pos_m, d_fm, f"F@d vs Millard diff\nmax {d_fm.max():.3f} ({100*d_fm.max()/bbox:.1f}% bbox)", d_fm.max() or 1.0)
    fig.suptitle("Bicep deformation: plain F@d vs Millard fiber constraint (color = displacement)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "mesh_compare.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print(f"wrote mesh_fiber.ply, mesh_fibermillard.ply, mesh_rest.ply in {HERE}")


if __name__ == "__main__":
    main()
