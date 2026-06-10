"""Placeholder validation: fuse a rigid SDF collider into the humanV2 GMG sim.

humanV2 has no bone mesh — the skeleton is implicit (per-vertex pin targets).
Before wiring real bone geometry, this proves the SDF collision hook works
inside the quasi-static GMG ``OneWaySim``: a test sphere is placed in front of
the torso and, each substep after the ARAP relaxation, free (non-pinned) muscle
vertices that penetrate the sphere are pushed back out along the SDF gradient
(``solve_sdf_contact_pos_kernel`` — position only, no velocity).

Run::

    PYTHONPATH=src .venv/bin/python -m muscle.collision.example_humanV2_collision \\
        --config data/humanV2/config/humanV2_gmg.json \\
        --out_dir result/humanV2_collision --end_frame 20
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import warp as wp

from muscle.coupleOneWayGmg.solver import OneWaySim, load_config
from muscle.coupleOneWayGmg.example import _vectorized_surface_tris
from muscle.collision.sdf_warp import solve_sdf_contact_pos_kernel
from muscle.collision.example_sdf_drop import _build_uv_sphere
from muscle.util.mesh_io import save_ply


class SphereCollider:
    """A static sphere SDF pushing free muscle verts out each substep.

    The collision ``inv_mass`` is the sim's ``inv_mass`` with pinned (bone-
    attached) verts zeroed, so the push only moves free belly tissue and never
    fights the pin-to-target overwrite.
    """

    def __init__(self, sim: OneWaySim, center, radius, contact_offset=0.0,
                 skip_pinned=False):
        self.sim = sim
        self.device = sim._device
        self.center = wp.vec3(*[float(c) for c in center])
        self.radius = float(radius)
        self.contact_offset = float(contact_offset)

        # In humanV2 GMG EVERY vert is pin-driven (pts == all NV), so masking
        # pinned verts would disable collision entirely. We instead let the
        # collision run as the LAST op each substep (after ARAP), so it wins
        # over that substep's pin overwrite; pin re-asserts next substep and
        # collision re-applies, converging to a pushed-out final state.
        inv = sim.inv_mass.numpy().copy()
        if skip_pinned:
            inv[sim.pts.numpy()] = 0.0
        self.coll_inv_mass = wp.array(inv.astype(np.float32),
                                      dtype=wp.float32, device=self.device)
        self.n_free = int((inv > 0).sum())

    def __call__(self):
        wp.launch(
            solve_sdf_contact_pos_kernel, dim=self.sim.NV,
            inputs=[self.sim.pos, self.coll_inv_mass,
                    wp.int32(1), wp.float32(0.0),
                    self.center, wp.float32(self.radius),
                    wp.float32(self.contact_offset)],
            device=self.device)

    def penetration_count(self, pos_np, eps=1e-3):
        """Return (deep, onsurface): verts genuinely inside the sphere
        (``d < r - eps``) and verts resting on its surface (the dent)."""
        c = np.array([self.center[0], self.center[1], self.center[2]])
        d = np.linalg.norm(pos_np - c[None, :], axis=1)
        rr = self.radius + self.contact_offset
        deep = int((d < rr - eps).sum())
        onsurf = int(((d >= rr - eps) & (d < rr + eps)).sum())
        return deep, onsurf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/humanV2/config/humanV2_gmg.json")
    ap.add_argument("--out_dir", default="result/humanV2_collision")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--end_frame", type=int, default=20)
    ap.add_argument("--sphere", type=float, nargs=4, metavar=("X", "Y", "Z", "R"),
                    default=[0.0, 1.0, 0.4, 0.4],
                    help="collider sphere center xyz + radius")
    ap.add_argument("--no_collision", action="store_true",
                    help="baseline run without the collider")
    ap.add_argument("--usd", default=None,
                    help="also write a USD (muscle + collider sphere) to this path")
    args = ap.parse_args()

    wp.init()
    cfg = load_config(args.config)
    cfg["end_frame"] = args.end_frame

    sim = OneWaySim(cfg, device=args.device)
    cx, cy, cz, r = args.sphere
    collider = None
    if not args.no_collision:
        collider = SphereCollider(sim, (cx, cy, cz), r)
        sim.collision_solvers = [collider]
        print(f"collider: sphere c=({cx},{cy},{cz}) r={r}  free_verts={collider.n_free}/{sim.NV}")

    print(f"NV={sim.NV}  NT={sim.NT}  frames={sim.start_frame}..{sim.end_frame}")

    out_dir = Path(args.out_dir)
    mesh_out = out_dir / "mesh"
    mesh_out.mkdir(parents=True, exist_ok=True)
    surface_faces = _vectorized_surface_tris(sim.tet_idx.numpy())

    # Collider sphere mesh (for PLY dump + USD viz).
    sphere_v = sphere_t = None
    if collider is not None:
        sphere_v, sphere_t = _build_uv_sphere((cx, cy, cz), r)
        save_ply(str(out_dir / "collider_sphere.ply"), sphere_v, sphere_t)

    renderer = None
    if args.usd:
        from muscle.muscle_warp import WarpRenderer
        Path(args.usd).parent.mkdir(parents=True, exist_ok=True)
        renderer = WarpRenderer(mode="usd", stage_path=args.usd)
        surf_flat = surface_faces.reshape(-1).astype(np.int32)
        sphere_flat = sphere_t.reshape(-1).astype(np.int32) if sphere_t is not None else None
        print(f"[render] writing USD to {args.usd}")

    t0 = time.perf_counter()
    for frame in range(sim.start_frame, sim.end_frame + 1):
        sim.step_frame(frame)
        pos_np = sim.positions_np()
        save_ply(str(mesh_out / f"{frame:04d}.ply"), pos_np, surface_faces)
        if renderer is not None:
            inner = renderer.renderer
            inner.begin_frame(frame * sim.dt)
            inner.render_mesh(name="muscle", points=pos_np,
                              indices=surf_flat, colors=(0.8, 0.3, 0.25))
            if sphere_flat is not None:
                inner.render_mesh(name="collider", points=sphere_v,
                                  indices=sphere_flat, colors=(0.55, 0.55, 0.6))
            inner.end_frame()
        if frame % 5 == 0 or frame == sim.end_frame:
            elapsed = time.perf_counter() - t0
            if collider:
                deep, onsurf = collider.penetration_count(pos_np)
                print(f"  frame {frame:>3d}  {elapsed:6.1f}s  "
                      f"deep_penetration={deep}  dented_onsurface={onsurf}")
            else:
                print(f"  frame {frame:>3d}  {elapsed:6.1f}s")
    if renderer is not None:
        renderer.save()
        print(f"USD saved -> {args.usd}")
    print(f"done -> {mesh_out}")


if __name__ == "__main__":
    main()
