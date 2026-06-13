"""Faithful Gaia S01 U-drop: MassSpring material + boundary + DCD self-collision.

Unlike example_dcd_u.py (which drove the U with MuscleSim's volume+ARAP
material -> soft fold -> deep self-crossings -> DCD explosion), this driver
uses the actual Gaia S01 material (udcf_massspring.MassSpringTet) and the
exact Gaia substep order, bypassing MuscleSim entirely:

  per substep (Gaia runStepCPU, iterations=1, collisionDetectionSubSteps=1):
    integrate -> spring -> volume -> boundary -> inversion -> narrowPhase -> updateVel

Initial conditions match Models.json / Parameters.json:
  translation (0,0.2,0), initialVelocity (0,-5,0), gravity (0,-9.8,0),
  floor y=0, worldBounds x,z in [-2,2], boundary friction s=0.5 d=0.4,
  dt=1/60, numSubsteps=20.

Compare output PLYs against ~/Dev/Gaia_UDCF/Simulator/testresults_U/A*.ply.

Run::

    PYTHONPATH=src .venv/bin/python -m muscle.collision.udcf.example.u_drop \
        --geo data/collisionTest/U.geo --steps 60
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import warp as wp

from muscle.collision.udcf.kernel import box_boundary_kernel
from muscle.collision.udcf.solver import UdcfDcdCollider, UdcfParams
from muscle.collision.udcf.massspring import MassSpringTet, MassSpringParams
from muscle.muscle_warp import WarpRenderer
from muscle.util.geo_io import GeoFile
from muscle.util.mesh_io import build_surface_tris


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
    g = GeoFile(args.geo)
    verts = np.asarray(g.positions, np.float32)
    tets = np.asarray(g.tet_indices, np.int32)
    n_v = verts.shape[0]

    init_vel = None
    if args.init_state:
        st = json.load(open(args.init_state))["meshesState"][0]
        verts = np.asarray(st["position"], np.float32)[:n_v]
        init_vel = np.asarray(st["velocity"], np.float32)[:n_v]
    else:
        verts = verts.copy()
        verts[:, 1] += args.translate_y

    dt_sub = args.dt / args.num_substeps
    msp = MassSpringParams(spring_compliance=args.spring_compliance,
                           spring_damping=args.spring_damping,
                           density=args.density, gravity=args.gravity)
    ms = MassSpringTet(verts, tets, dt_sub, params=msp, device=args.device)

    if args.init_state:
        ms.vel.assign(init_vel)
    elif args.init_vel_y != 0.0:
        v0 = np.zeros((n_v, 3), np.float32)
        v0[:, 1] = args.init_vel_y
        ms.vel.assign(v0)

    surf = build_surface_tris(tets, verts).astype(np.int32)
    n_surf = surf.shape[0]
    max_pairs = max(65536, n_surf * (n_surf - 1) // 2)

    dummy_v = np.array([[1e4, 1e4, 1e4], [1e4 + 1, 1e4, 1e4],
                        [1e4, 1e4, 1e4 + 1]], np.float32)
    dummy_t = np.array([[0, 1, 2]], np.int32)

    dcd = None
    if not args.no_dcd:
        dcd = UdcfDcdCollider(
            ms.pos, ms.inv_mass, surface_tris=surf,
            static_verts=dummy_v, static_tris=dummy_t,
            params=UdcfParams(allow_self=True, max_pairs=max_pairs,
                              thickness=args.thickness, jacobi_avg=False,
                              gauss_seidel=True, max_push=0.0),
            device=args.device)

    n_mov = n_v
    bounds_lo = wp.vec3(-2.0, 0.0, -2.0)
    bounds_hi = wp.vec3(2.0, 5.0e4, 2.0)

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
        write_ply(f"{args.out_ply}/A{0:08d}.ply", ms.pos.numpy(), surf)

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

    render_frame(0, ms.pos.numpy())
    ref = args.ref_dir

    def substep():
        ms.integrate()
        ms.solve_material()            # spring -> volume
        wp.launch(box_boundary_kernel, dim=n_mov,
                  inputs=[ms.pos, ms.pprev, bounds_lo, bounds_hi,
                          wp.float32(args.mu_s), wp.float32(args.mu_d)],
                  device=args.device)
        if not args.no_inversion:
            ms.solve_inversion()
        if dcd is not None:
            dcd.solve(0)               # narrowPhase once (iterations=1)
        ms.update_velocities()

    for step in range(1, args.steps + 1):
        for _ in range(args.num_substeps):
            substep()
        p = ms.pos.numpy()
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
