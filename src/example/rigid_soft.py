"""Dynamic rigid sphere falling onto a soft tet box — a y-up replica of
Newton's ``example_rigid_soft_contact``.

A soft tet box rests on the ground. A rigid sphere (mass + gravity) drops
off-centre onto it, squishes it, and is pushed back up by the soft body — the
soft-vertex/sphere contact splits each penetration correction by inverse mass
(Newton's third law in position-based form, see
``muscle.collision.sdf.kernel.solve_sphere_soft_contact_kernel``). The soft body
recovers its shape via volume + ARAP.

Two SDFs act on the soft body each substep:
  * ground plane  (static, infinite mass — one-way push)
  * rigid sphere  (dynamic — two-way, sphere recoils)

The sphere is integrated as a translational rigid body (point mass; rotation is
irrelevant for a sphere on near-vertical contact).

Run::

    PYTHONPATH=src .venv/bin/python -m example.rigid_soft \\
        --config data/collisionTest/rigid_soft.json
"""
from __future__ import annotations

import argparse
import logging
import math
import os

import numpy as np
import warp as wp

from muscle.collision.sdf.kernel import (
    solve_sdf_contact_kernel,           # soft vs static ground plane
    solve_sphere_soft_contact_kernel,   # soft vs dynamic sphere (two-way)
)
from example.sdf_drop import (
    _build_plane_mesh_xz, _build_uv_sphere,
)
from muscle.util.util import load_config, setup_logging
from muscle.engine import MuscleSim, WarpRenderer

log = logging.getLogger(__name__)

DEFAULT_CONFIG = "data/collisionTest/rigid_soft.json"


# ----------------------------------------------------------------------
# Soft tet box grid (Freudenthal 6-tet decomposition per hex cell)
# ----------------------------------------------------------------------

def _build_tet_box(dim_x, dim_y, dim_z, cell):
    """Tet box with ``dim_*`` cells of size ``cell``. Centred in X-Z, bottom
    face on ``y = 0``. Returns ``(verts (N,3), tets (M,4))`` with every tet
    positively oriented."""
    nx, ny, nz = int(dim_x), int(dim_y), int(dim_z)
    Lx, Lz = nx * cell, nz * cell
    xs = np.arange(nx + 1) * cell - 0.5 * Lx
    ys = np.arange(ny + 1) * cell
    zs = np.arange(nz + 1) * cell - 0.5 * Lz
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    def vid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    # 6 tets sharing the main diagonal v0-v6 of each hex cell.
    HEX = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
           (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [
                    vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k), vid(i, j + 1, k),
                    vid(i, j, k + 1), vid(i + 1, j, k + 1), vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1),
                ]
                for a, b, cc, d in HEX:
                    tets.append([c[a], c[b], c[cc], c[d]])
    tets = np.asarray(tets, dtype=np.int32)

    # Ensure positive orientation (det[b-a, c-a, d-a] > 0); swap last two if not.
    p = verts[tets]
    det = np.linalg.det(np.stack([p[:, 1] - p[:, 0],
                                  p[:, 2] - p[:, 0],
                                  p[:, 3] - p[:, 0]], axis=-1))
    flip = det < 0.0
    tets[flip] = tets[flip][:, [0, 1, 3, 2]]
    return verts, tets


class RigidSoftDemo:
    """Owns the dynamic sphere state + the soft-body contact buffers, and drives
    the two-SDF contact solve each substep.

    Per substep (``collision_solver`` is called once per substep, after the
    soft body's ARAP/volume solve and before ``update_velocities``):
      1. integrate the sphere under gravity (predict)
      2. push soft verts out of the static ground plane (one-way)
      3. push soft verts out of the sphere, accumulating the sphere recoil
         (two-way); apply the recoil to the sphere centre + velocity
      4. resolve the sphere against the ground
    """

    def __init__(self, sim: MuscleSim, *, ground_y: float,
                 sphere_radius: float, sphere_density: float,
                 sphere_center0, contact_offset: float,
                 restitution_n: float, restitution_t: float,
                 restitution_ground: float, gravity: float, log_every: int):
        self.sim = sim
        self.device = str(sim.pos.device)
        self.dt = float(sim.dt)            # substep dt
        self.gravity = float(gravity)      # signed (negative = down)
        self.ground_y = float(ground_y)
        self.contact_offset = float(contact_offset)
        self.restitution_n = float(restitution_n)
        self.restitution_t = float(restitution_t)
        self.restitution_ground = float(restitution_ground)
        self.log_every = max(1, int(log_every))
        self._frame_counter = 0

        self.sphere_r = float(sphere_radius)
        mass = sphere_density * (4.0 / 3.0) * math.pi * self.sphere_r ** 3
        self.sphere_mass = float(mass)
        self.inv_mass_sphere = float(1.0 / mass)
        self.sphere_center = np.asarray(sphere_center0, dtype=np.float64).copy()
        self.sphere_vel = np.zeros(3, dtype=np.float64)
        self._sphere_c_wp = wp.vec3(*self.sphere_center.tolist())

        n_v = int(sim.pos.shape[0])
        self.n_v = n_v
        m = sim.mass.numpy()
        inv = np.where(m > 0.0, 1.0 / np.where(m > 0.0, m, 1.0), 0.0)
        if hasattr(sim, "stopped"):
            inv = np.where(sim.stopped.numpy() == 0, inv, 0.0)
        self.inv_mass = wp.array(inv.astype(np.float32), dtype=wp.float32,
                                 device=self.device)

        self.is_collide = wp.zeros(n_v, dtype=wp.int32, device=self.device)
        self.contact_normal = wp.zeros(n_v, dtype=wp.vec3, device=self.device)
        self._sphere_dC = wp.zeros(1, dtype=wp.vec3, device=self.device)

        log.info(f"[rigid_soft] sphere r={self.sphere_r} mass={self.sphere_mass:.3f} "
                 f"center0={self.sphere_center.tolist()}")
        log.info(f"[rigid_soft] ground_y={self.ground_y} verts={n_v} dt_sub={self.dt:.5f}")

    def collision_solver(self):
        sim = self.sim
        dt = self.dt

        # Clear contact flags once per substep. Both SDF kernels below only ever
        # SET is_collide/contact_normal on contact (never clear), so the sphere
        # pass can't wipe the ground pass's flags — otherwise ground-contacting
        # bottom verts would lose velocity reflection and the box would jitter.
        self.is_collide.zero_()
        self.contact_normal.zero_()

        # 1. Integrate sphere under gravity (predict).
        self.sphere_vel[1] += self.gravity * dt
        self.sphere_center += self.sphere_vel * dt
        self._sphere_c_wp = wp.vec3(*self.sphere_center.tolist())

        # 2. Soft vs static ground plane (one-way push).
        wp.launch(
            solve_sdf_contact_kernel, dim=self.n_v,
            inputs=[sim.pos, sim.pprev, self.inv_mass,
                    wp.int32(0), wp.float32(self.ground_y),
                    wp.vec3(0.0, 0.0, 0.0), wp.float32(0.0),
                    wp.float32(self.contact_offset),
                    self.is_collide, self.contact_normal],
            device=self.device)

        # 3. Soft vs dynamic sphere (two-way). Accumulate sphere recoil.
        self._sphere_dC.zero_()
        wp.launch(
            solve_sphere_soft_contact_kernel, dim=self.n_v,
            inputs=[sim.pos, sim.pprev, self.inv_mass,
                    self._sphere_c_wp, wp.float32(self.sphere_r),
                    wp.float32(self.inv_mass_sphere),
                    wp.float32(self.contact_offset),
                    self.is_collide, self.contact_normal, self._sphere_dC],
            device=self.device)
        dC = self._sphere_dC.numpy()[0].astype(np.float64)
        self.sphere_center += dC
        self.sphere_vel += dC / dt          # contact recoil -> sphere velocity

        # 4. Sphere vs ground (don't let it tunnel if the box is squished thin).
        low = self.ground_y + self.sphere_r
        if self.sphere_center[1] < low:
            self.sphere_center[1] = low
            if self.sphere_vel[1] < 0.0:
                self.sphere_vel[1] = -self.restitution_ground * self.sphere_vel[1]
        self._sphere_c_wp = wp.vec3(*self.sphere_center.tolist())

    def log_diag(self, step: int):
        self._frame_counter += 1
        if self._frame_counter % self.log_every != 0:
            return
        n = int(self.is_collide.numpy().sum())
        log.info(f"[rigid_soft] step={step:4d} sphere_y={self.sphere_center[1]:.4f} "
                 f"sphere_vy={self.sphere_vel[1]:+.4f} contacts={n}")


def _build_sim(cfg, soft_cfg) -> tuple[MuscleSim, np.ndarray]:
    verts, tets = _build_tet_box(soft_cfg["dim_x"], soft_cfg["dim_y"],
                                 soft_cfg["dim_z"], float(soft_cfg["cell"]))
    fiber = np.zeros((tets.shape[0], 3), dtype=np.float32)
    fiber[:, 0] = 1.0
    gravity_y = -abs(float(getattr(cfg, "gravity_z", getattr(cfg, "gravity", 9.8))))
    sim = MuscleSim.from_procedural(
        verts, tets, fiber,
        constraint_configs=list(cfg.constraints),
        dts=float(cfg.dt), num_substeps=int(cfg.num_substeps),
        device=cfg.arch, density=float(cfg.density),
        veldamping=float(cfg.veldamping), gravity=gravity_y)
    # SVD inversion repair: a tet directly under the sphere contact would
    # otherwise flatten/invert under the heavy point load. Clamping F's
    # singular values to sigma_min each substep keeps tets valid and lets the
    # sphere rest in a shallow dimple instead of crushing through.
    sim.cfg.repair_alpha = float(getattr(cfg, "repair_alpha", 1.0))
    sim.cfg.repair_iters = int(getattr(cfg, "repair_iters", 2))
    sim.cfg.repair_sigma_min = float(getattr(cfg, "repair_sigma_min", 0.1))
    log.info(f"sim: n_verts={sim.n_verts} n_tets={sim.tet_np.shape[0]} "
             f"box_top_y={verts[:,1].max():.3f} "
             f"repair(alpha={sim.cfg.repair_alpha},iters={sim.cfg.repair_iters})")
    return sim, verts


def main():
    p = argparse.ArgumentParser(description="Rigid sphere vs soft box (Newton replica).")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    if args.steps is not None:
        cfg.nsteps = args.steps
    if args.output is not None:
        cfg.output_path = args.output

    wp.init()
    wp.set_device(cfg.arch)

    soft_cfg = cfg.soft_grid
    ground_cfg = cfg.ground
    sphere_cfg = cfg.sphere
    coll_cfg = getattr(cfg, "collision", {}) or {}

    sim, box_verts = _build_sim(cfg, soft_cfg)
    box_top = float(box_verts[:, 1].max())

    off = sphere_cfg.get("init_offset", [0.2, 1.2, 0.0])
    sphere_center0 = [float(off[0]),
                      box_top + float(off[1]),
                      float(off[2])]
    gravity_y = -abs(float(getattr(cfg, "gravity_z", 9.8)))

    demo = RigidSoftDemo(
        sim, ground_y=float(ground_cfg.get("y", 0.0)),
        sphere_radius=float(sphere_cfg["radius"]),
        sphere_density=float(sphere_cfg["density"]),
        sphere_center0=sphere_center0,
        contact_offset=float(coll_cfg.get("contact_offset", 0.0)),
        restitution_n=float(coll_cfg.get("restitution_n", 0.0)),
        restitution_t=float(coll_cfg.get("restitution_t", 0.9)),
        restitution_ground=float(sphere_cfg.get("restitution_ground", 0.0)),
        gravity=gravity_y,
        log_every=int(coll_cfg.get("log_every", 20)))
    sim.constraint_solvers = [demo.collision_solver]
    sim.is_collide = demo.is_collide
    sim.contact_normal = demo.contact_normal
    sim.restitution_n = demo.restitution_n
    sim.restitution_t = demo.restitution_t

    # Render meshes: ground plane, unit sphere (translated each frame), soft surf.
    ground_v, ground_t = _build_plane_mesh_xz(
        y=float(ground_cfg.get("y", 0.0)),
        half_extent=float(ground_cfg.get("half_extent", 1.5)),
        subdiv=int(ground_cfg.get("subdiv", 8)))
    sphere_v0, sphere_t = _build_uv_sphere([0.0, 0.0, 0.0], demo.sphere_r)

    renderer = None
    if cfg.render_mode == "usd":
        os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
        renderer = WarpRenderer(mode="usd", stage_path=cfg.output_path)
        log.info(f"[render] writing USD to {cfg.output_path}")

    soft_t_flat = sim.surface_tris_flat
    ground_t_flat = ground_t.reshape(-1).astype(np.int32)
    sphere_t_flat = sphere_t.reshape(-1).astype(np.int32)
    render_stride = max(1, int(round((1.0 / cfg.render_fps) / cfg.dt)))

    for step in range(1, cfg.nsteps + 1):
        sim.step_cnt = step
        sim.step()
        demo.log_diag(step)

        if step % render_stride == 0 or step == 1:
            soft_pos = sim.pos.numpy()
            sphere_pos = sphere_v0 + demo.sphere_center.astype(np.float32)[None, :]
            if renderer is not None:
                inner = renderer.renderer
                inner.begin_frame(step * cfg.dt)
                inner.render_mesh(name="soft", points=soft_pos,
                                  indices=soft_t_flat, colors=(0.3, 0.5, 0.85))
                inner.render_mesh(name="sphere", points=sphere_pos,
                                  indices=sphere_t_flat, colors=(0.95, 0.43, 0.18))
                inner.render_mesh(name="ground", points=ground_v,
                                  indices=ground_t_flat, colors=(0.55, 0.55, 0.55))
                inner.end_frame()

    if renderer is not None:
        renderer.save()
        log.info(f"Done — USD saved to {cfg.output_path}")


if __name__ == "__main__":
    main()
