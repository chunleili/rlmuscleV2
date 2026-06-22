"""Faithful Gaia S01 U-drop: MassSpring material + boundary + DCD self-collision.

Unlike example_dcd_u.py (which drove the U with MuscleSim's volume+ARAP
material -> soft fold -> deep self-crossings -> DCD explosion), this driver
uses the Gaia S01 MassSpring material (engine.MassSpringSim — stiff DISTANCE
springs + hard det(F)=1 volume + inverted-tet untangle on the shared
XpbdEngine) in the exact Gaia substep order:

  per substep (Gaia runStepCPU, iterations=1, collisionDetectionSubSteps=1):
    integrate -> spring -> volume -> boundary -> inversion -> narrowPhase -> updateVel

Initial conditions match Models.json / Parameters.json:
  translation (0,0.2,0), initialVelocity (0,-5,0), gravity (0,-9.8,0),
  floor y=0, worldBounds x,z in [-2,2], boundary friction s=0.5 d=0.4,
  dt=1/60, numSubsteps=20.

Compare output PLYs against ~/Dev/Gaia_UDCF/Simulator/testresults_U/A*.ply.

Run::

    PYTHONPATH=src .venv/bin/python -m example.dcd_drop \
        --geo data/collisionTest/U.geo --steps 60
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import warp as wp

from muscle.massspringsim import MassSpringSim
from muscle.engine import WarpRenderer


def write_ply(path, pos_np, tris):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n"
                f"element vertex {pos_np.shape[0]}\n"
                "property float x\nproperty float y\nproperty float z\n"
                f"element face {tris.shape[0]}\n"
                "property list uchar int vertex_indices\nend_header\n")
        np.savetxt(f, pos_np, fmt="%.6g")
        np.savetxt(f, np.hstack([np.full((tris.shape[0], 1), 3, np.int32),
                                 tris]), fmt="%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", default="data/collisionTest/U.geo")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--num_substeps", type=int, default=20)
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--density", type=float, default=3.0)
    ap.add_argument("--gravity", type=float, default=-9.8)
    ap.add_argument("--init_vel_y", type=float, default=-5.0)
    ap.add_argument("--translate_y", type=float, default=0.2)
    ap.add_argument("--spring_compliance", type=float, default=1e-4)
    ap.add_argument("--spring_damping", type=float, default=0.01)
    ap.add_argument("--mu_s", type=float, default=0.5)
    ap.add_argument("--mu_d", type=float, default=0.4)
    ap.add_argument("--thickness", type=float, default=1e-3)
    ap.add_argument("--no_dcd", action="store_true",
                    help="Material+boundary only (no self-collision)")
    ap.add_argument("--no_inversion", action="store_true")
    ap.add_argument("--init_state", default=None,
                    help="Gaia RecoveryStates/A*.json — start from this "
                         "frame's position+velocity (skips translate/init_vel)")
    ap.add_argument("--out_usd", default="output/udcf_u_drop.usd",
                    help="default mesh output (USD sequence)")
    ap.add_argument("--out_ply", default=None,
                    help="optional PLY-per-frame export dir (A<step>.ply)")
    ap.add_argument("--out_every", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--ref_dir", default=None,
                    help="Gaia testresults_U dir for per-frame L2 compare")
    args = ap.parse_args()

    wp.init()

    # --init_state: load recovery pose BEFORE construction so it becomes the
    # MassSpringSim rest state (zero pre-strain), matching pre-refactor semantics.
    init_verts = init_vel = None
    if args.init_state:
        from muscle.util.geo_io import GeoFile as _GeoFile
        n_v_geo = np.asarray(_GeoFile(args.geo).positions, np.float32).shape[0]
        st = json.load(open(args.init_state))["meshesState"][0]
        init_verts = np.asarray(st["position"], np.float32)[:n_v_geo]
        init_vel = np.asarray(st["velocity"], np.float32)[:n_v_geo]

    sim = MassSpringSim.u_drop(
        args.geo, dt=args.dt, n_substeps=args.num_substeps, density=args.density,
        gravity=args.gravity, init_vel_y=args.init_vel_y, translate_y=args.translate_y,
        spring_compliance=args.spring_compliance, spring_damping=args.spring_damping,
        mu_s=args.mu_s, mu_d=args.mu_d, thickness=args.thickness,
        no_dcd=args.no_dcd, no_inversion=args.no_inversion,
        init_verts=init_verts, init_vel=init_vel, device=args.device)
    surf = sim.surf

    surf_flat = surf.reshape(-1).astype(np.int32)
    # Static ground quad at y=0 for visual context (the box-boundary floor).
    floor_v = np.array([[-2, 0, -2], [2, 0, -2], [2, 0, 2], [-2, 0, 2]],
                       np.float32)
    floor_t = np.array([0, 1, 2, 0, 2, 3], np.int32)

    renderer = None
    if args.out_usd:
        os.makedirs(os.path.dirname(args.out_usd) or ".", exist_ok=True)
        renderer = WarpRenderer(mode="usd", stage_path=args.out_usd)

    if args.out_ply:
        os.makedirs(args.out_ply, exist_ok=True)
        write_ply(f"{args.out_ply}/A{0:08d}.ply", sim.pos.numpy(), surf)

    def render_frame(step, pos_np):
        if renderer is None:
            return
        inner = renderer.renderer
        inner.begin_frame(step * args.dt)
        inner.render_mesh(name="u", points=pos_np, indices=surf_flat,
                          colors=(0.8, 0.3, 0.2))
        inner.render_mesh(name="floor", points=floor_v, indices=floor_t,
                          colors=(0.55, 0.55, 0.55))
        inner.end_frame()

    render_frame(0, sim.pos.numpy())
    ref = args.ref_dir

    for step in range(1, args.steps + 1):
        sim.step()
        p = sim.pos.numpy()
        if step % args.out_every == 0 or step == args.steps:
            render_frame(step, p)
            if args.out_ply:
                write_ply(f"{args.out_ply}/A{step:08d}.ply", p, surf)
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            line = (f"step={step:4d} min_y={p[:, 1].min():+.4f} "
                    f"max_y={p[:, 1].max():+.4f} "
                    f"cen_y={p[:, 1].mean():+.4f}")
            if np.isnan(p).any():
                line += "  NaN!"
            if ref:
                rp = f"{ref}/A{step:08d}.ply"
                if os.path.exists(rp):
                    gv = _load_ply_verts(rp)
                    m = min(gv.shape[0], p.shape[0])
                    d = np.linalg.norm(p[:m] - gv[:m], axis=1)
                    line += f"  vs_gaia_L2_max={d.max():.4f} mean={d.mean():.4f}"
            print(line)

    if renderer is not None:
        renderer.save()
    out = args.out_usd or args.out_ply
    print(f"done -> {out}")


def _load_ply_verts(path):
    with open(path) as f:
        lines = f.read().splitlines()
    nv = 0
    for i, l in enumerate(lines):
        if l.startswith("element vertex"):
            nv = int(l.split()[-1])
        if l.strip() == "end_header":
            hs = i + 1
            break
    return np.array([[float(x) for x in lines[hs + j].split()[:3]]
                     for j in range(nv)], np.float32)


if __name__ == "__main__":
    main()
