"""4-way fiber-constraint comparison on the bicep: no fiber / plain F@d /
anisotropic ARAP (production) / Hill--Millard energy.

Same bicep tet mesh, same attaches + tendon pin, same activation. Only the
active fiber constraint differs. Two outputs:

  fiber4_mesh.png   4-panel deformed surface, colored by displacement from the
                    NO-FIBER baseline (isolates the fiber-induced deformation).
  fiber4_act_len.png/.csv
                    mean normalized belly fiber stretch (length) vs activation,
                    one curve per fiber model.

Run (project venv):
    .venv/bin/python tools/bicep_fl/fiber4.py [--nsteps 400] [--device cuda:0]
"""
import argparse
import copy
import os

import numpy as np
import warp as wp

from muscle.util.util import load_config, setup_logging
from muscle.musclesim import MuscleSim
from muscle.engine import fill_float_kernel
from muscle.util.mesh_io import save_ply

HERE = os.path.dirname(__file__)
# generated artifacts go to <repo>/output/bicep, not next to the script
OUT = os.path.join(HERE, "..", "..", "output", "bicep")
os.makedirs(OUT, exist_ok=True)
CFG = "data/muscle/config/bicep.json"

NUM_SUBSTEPS = 50
TENDON_PIN_THRESHOLD = 0.9
SMOOTH_RELAX_STEPS = 10

# Constraint dicts that replace the stock "fiber" (plain F@d) entry.
ANISO_CON = {"type": "aniso_arap", "name": "fiber", "stiffness": 1000.0,
             "dampingratio": 0.1}
MILLARD_CON = {"type": "fibermillard", "name": "fiber", "stiffness": 10000.0,
               "sigma0": 300000.0, "contraction_factor": 0.4, "dampingratio": 0.1}

MODELS = ["nofiber", "plain", "aniso_arap", "millard"]
LABELS = {"nofiber": "no fiber", "plain": "plain $C=|Fd|$",
          "aniso_arap": "anisotropic ARAP", "millard": "Hill--Millard energy"}


def make_cfg(nsteps, model):
    cfg = load_config(CFG)
    cfg.gui = False
    cfg.render_mode = None
    cfg.nsteps = nsteps
    cfg.num_substeps = NUM_SUBSTEPS
    cfg.constraints = copy.deepcopy(cfg.constraints)
    # Strip bone attaches: hold the muscle ONLY by the proximal kinematic pin so
    # all four models see an identical isotonic setup and the free distal end can
    # shorten. Keeps volume + ARAP + (swapped) fiber.
    cfg.constraints = [c for c in cfg.constraints if c.get("type") != "attach"]
    idx = next(i for i, c in enumerate(cfg.constraints) if c.get("type") == "fiber")
    if model == "nofiber":
        del cfg.constraints[idx]
    elif model == "plain":
        pass  # keep stock fiber
    elif model == "aniso_arap":
        cfg.constraints[idx] = copy.deepcopy(ANISO_CON)
    elif model == "millard":
        cfg.constraints[idx] = copy.deepcopy(MILLARD_CON)
    return cfg


def freeze_tendon(sim):
    """Pin ONLY the proximal tendon end so the muscle can shorten isotonically.

    Freezing both tendon ends makes the contraction isometric: the along-fiber
    length is geometrically clamped, so a target-driven constraint (aniso ARAP)
    cannot reach its target and looks inert. Pinning one end frees the distal
    tendon to be pulled in, giving a clean activation->length response.
    """
    tm = sim.v_tendonmask_np
    if tm is None:
        return 0
    tendon = tm > TENDON_PIN_THRESHOLD
    # principal fiber axis (mean rest fiber direction) -> project verts on it
    d = sim.v_fiber_np[tendon].mean(0)
    d = d / (np.linalg.norm(d) + 1e-12)
    proj = sim.pos0_np @ d
    # proximal = tendon verts on the lower-projection side of the tendon spread
    thr = np.median(proj[tendon])
    pin = tendon & (proj <= thr)
    st = sim.stopped.numpy()
    st[pin] = 1
    sim.stopped.assign(st)
    return int(pin.sum())


def settle(sim, nsteps, act):
    for _ in range(nsteps):
        wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
                  inputs=[sim.activation, float(act)])
        sim.step()
    for _ in range(SMOOTH_RELAX_STEPS):
        wp.launch(fill_float_kernel, dim=sim.activation.shape[0],
                  inputs=[sim.activation, 0.0])
        sim.step()
    wp.synchronize()
    return sim.pos.numpy().copy()


def _belly_axis(sim):
    """Belly vertex mask + unit principal fiber axis (mean rest fiber dir)."""
    belly = sim.v_tendonmask_np < 0.5
    d = sim.v_fiber_np[belly].mean(0)
    return belly, d / (np.linalg.norm(d) + 1e-12)


def belly_geom(sim, pos):
    """(axial extent, mean radial extent) of the belly, in world units.

    axial  = end-to-end span of the belly along the fiber axis (muscle LENGTH);
    radial = mean distance of belly verts from that axis (BULGE / girth).
    Muscle contraction shortens the axial extent and thickens the radial one.
    """
    belly, d = _belly_axis(sim)
    P = pos[belly]
    ax = P @ d
    axial = float(ax.max() - ax.min())
    radial = float(np.linalg.norm(P - np.outer(ax, d), axis=1).mean())
    return axial, radial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--acts", type=int, default=11)
    args = ap.parse_args()

    setup_logging()
    wp.init()
    try:
        wp.set_device(args.device)
    except Exception:
        wp.set_device("cpu")
    print(f"device = {wp.get_device()}")

    acts = np.linspace(0.0, 1.0, args.acts)
    axial = {}        # model -> normalized axial length(a)  (1.0 = rest)
    radial = {}       # model -> normalized radial girth(a)  (1.0 = rest)
    mesh_a1 = {}      # model -> deformed pos at a=1
    mesh_all = {}     # model -> (nacts, NV, 3) equilibrium mesh per activation (anim)
    faces = rest = None
    rest_ax = rest_rad = None

    for model in MODELS:
        axs, rads, frames = [], [], []
        for a in acts:
            sim = MuscleSim(make_cfg(args.nsteps, model))
            freeze_tendon(sim)
            pos = settle(sim, args.nsteps, float(a))
            ax, rad = belly_geom(sim, pos)
            axs.append(ax); rads.append(rad)
            frames.append(pos)
            if faces is None:
                faces = np.asarray(sim.surface_tris)
                rest = sim.pos0_np.copy()
            if rest_ax is None:
                rest_ax, rest_rad = belly_geom(sim, sim.pos0_np)
            if abs(a - 1.0) < 1e-9:
                mesh_a1[model] = pos
        mesh_all[model] = np.asarray(frames, np.float32)
        axial[model] = np.array(axs) / rest_ax
        radial[model] = np.array(rads) / rest_rad
        print(f"{model:12s} axial(a=1)={axial[model][-1]:.4f} ({100*(axial[model][-1]-1):+.1f}%)  "
              f"radial(a=1)={radial[model][-1]:.4f} ({100*(radial[model][-1]-1):+.1f}%)")

    # ---- save meshes ----
    save_ply(os.path.join(OUT, "fiber4_rest.ply"), rest, faces)
    for m in MODELS:
        save_ply(os.path.join(OUT, f"fiber4_{m}.ply"), mesh_a1[m], faces)

    # ---- per-activation equilibrium meshes for the 0->1 animation ----
    # (nofiber == rest for all a: no fiber constraint => activation does nothing.
    # Kept in the npz for completeness; the USD builder shows rest as the static
    # reference and animates only the fiber models.)
    npz = os.path.join(OUT, "fiber4_anim.npz")
    np.savez_compressed(npz, faces=faces, acts=acts, rest=rest.astype(np.float32),
                        **{f"pos_{m}": mesh_all[m] for m in MODELS})
    print(f"wrote {npz}  (acts={len(acts)}, NV={rest.shape[0]})")

    # ---- activation-length CSV ----
    csv = os.path.join(OUT, "fiber4_act_len.csv")
    with open(csv, "w") as f:
        f.write("activation," + ",".join(f"{m}_axial" for m in MODELS)
                + "," + ",".join(f"{m}_radial" for m in MODELS) + "\n")
        for i, a in enumerate(acts):
            f.write(f"{a:.3f},"
                    + ",".join(f"{axial[m][i]:.5f}" for m in MODELS) + ","
                    + ",".join(f"{radial[m][i]:.5f}" for m in MODELS) + "\n")
    print("wrote", csv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # ---- activation vs length (axial) and bulge (radial) ----
    styles = {"nofiber": "k:", "plain": "C1--", "aniso_arap": "C0-", "millard": "C2-."}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 3.6))
    for m in MODELS:
        lab = LABELS[m].replace("--", "-")
        axL.plot(acts, 100 * (axial[m] - 1), styles[m], lw=2, label=lab)
        axR.plot(acts, 100 * (radial[m] - 1), styles[m], lw=2, label=lab)
    axL.set_xlabel("activation $a$"); axL.set_ylabel("axial length change (\\%)")
    axL.set_title("Muscle shortening"); axL.grid(alpha=0.3); axL.legend(fontsize=8)
    axR.set_xlabel("activation $a$"); axR.set_ylabel("radial girth change (\\%)")
    axR.set_title("Muscle bulging"); axR.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fiber4_act_len.png"), dpi=150)
    print("wrote fiber4_act_len.png")

    # ---- 4-panel mesh, colored by displacement from no-fiber baseline ----
    base = mesh_a1["nofiber"]
    disp = {m: np.linalg.norm(mesh_a1[m] - base, axis=1) for m in MODELS}
    # clip color scale to a robust percentile so a single free-tendon-tip spike
    # does not wash out the belly bulge gradient.
    alld = np.concatenate([disp[m] for m in MODELS])
    vmax = float(np.percentile(alld[alld > 0], 92)) or 1.0

    mn, mx = base.min(0), base.max(0)
    c = (mn + mx) / 2
    span = (mx - mn)
    pad = 1.04

    def panel(ax, pos, scalar, title):
        fc = plt.cm.turbo(np.clip(scalar[faces].mean(1) / vmax, 0, 1))
        ax.add_collection3d(Poly3DCollection(pos[faces], facecolors=fc,
                            edgecolors="k", linewidths=0.03, alpha=1.0))
        for lim, ci, si in ((ax.set_xlim, c[0], span[0]), (ax.set_ylim, c[1], span[1]),
                            (ax.set_zlim, c[2], span[2])):
            lim(ci - pad*si/2, ci + pad*si/2)
        ax.set_box_aspect(tuple(span / span.max()))   # box hugs the (flat) mesh
        ax.set_title(title, fontsize=10, pad=-10)
        ax.set_axis_off()
        ax.view_init(elev=10, azim=-88)

    fig = plt.figure(figsize=(11, 5.2))
    for i, m in enumerate(MODELS):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        panel(ax, mesh_a1[m], disp[m],
              f"{LABELS[m].replace('--','-')}   (max disp {disp[m].max():.3f})")
    fig.suptitle("Bicep contraction ($a{=}1$): displacement from the no-fiber baseline",
                 fontsize=11, y=0.99)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.93, wspace=0.0, hspace=0.0)
    out = os.path.join(OUT, "fiber4_mesh.png")
    fig.savefig(out, dpi=150)
    try:  # autocrop uniform whitespace border
        from PIL import Image, ImageChops
        im = Image.open(out).convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bbox = ImageChops.difference(im, bg).getbbox()
        if bbox:
            m = 12
            bbox = (max(0, bbox[0]-m), max(0, bbox[1]-m),
                    min(im.width, bbox[2]+m), min(im.height, bbox[3]+m))
            im.crop(bbox).save(out)
    except Exception as e:
        print("autocrop skipped:", e)
    print("wrote fiber4_mesh.png")


if __name__ == "__main__":
    main()
