"""Soft tet body falling onto a rigid analytic SDF (plane or sphere).

Simplest classic PBD soft-vs-rigid contact, and the base for muscle-bone
coupling. Each muscle vertex queries the rigid SDF every substep and is pushed
out along the gradient (``muscle.collision.sdf.kernel``); normal restitution +
tangential friction come from ``MuscleSim.update_velocities``'s contact branch.

Contrast with ``example_u_collision`` (triangle-triangle UDCF, with broadphase +
self-collision): here there is no broadphase, no pair list, no winding — one
``wp.launch`` per substep over all muscle verts.

Run::

    PYTHONPATH=src .venv/bin/python -m example.sdf_drop \\
        --config data/collisionTest/sdf_drop.json --shape plane
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import warp as wp

from muscle.collision.sdf.kernel import solve_sdf_contact_kernel
from muscle.util.util import load_config, setup_logging
from muscle.util.geo_io import GeoFile
from muscle.util.mesh_io import build_surface_tris
from muscle.musclesim import MuscleSim
from muscle.engine import WarpRenderer

log = logging.getLogger(__name__)

DEFAULT_CONFIG = "data/collisionTest/sdf_drop.json"

SHAPE_PLANE = 0
SHAPE_SPHERE = 1


def _build_plane_mesh_xz(y, half_extent, subdiv, center_xz=(0.0, 0.0)):
    """Subdivided horizontal plane in the X-Z plane at ``y`` (render only)."""
    n = int(max(1, subdiv))
    cx, cz = float(center_xz[0]), float(center_xz[1])
    xs_1d = cx + np.linspace(-half_extent, +half_extent, n + 1)
    zs_1d = cz + np.linspace(-half_extent, +half_extent, n + 1)
    xs, zs = np.meshgrid(xs_1d, zs_1d, indexing="xy")
    verts = np.stack([xs.flatten(),
                      np.full(xs.size, y),
                      zs.flatten()], axis=1).astype(np.float32)
    tris = []
    for j in range(n):
        for i in range(n):
            v00 = j * (n + 1) + i
            v10 = v00 + 1
            v01 = (j + 1) * (n + 1) + i
            v11 = v01 + 1
            tris.append([v00, v11, v10])
            tris.append([v00, v01, v11])
    return verts, np.asarray(tris, dtype=np.int32)


def _load_u_mesh(geo_path, drop_height):
    """Load the U tet mesh, centre it in X-Z, and shift so its lowest vertex
    sits at ``y = drop_height`` above the contact surface at ``y = 0``."""
    g = GeoFile(geo_path)
    verts = np.asarray(g.positions, dtype=np.float32)
    tets = np.asarray(g.tet_indices, dtype=np.int32)
    if tets.shape[1] != 4:
        raise ValueError(f"{geo_path}: expected tets (4 verts), got {tets.shape}")
    cx = 0.5 * (verts[:, 0].min() + verts[:, 0].max())
    cz = 0.5 * (verts[:, 2].min() + verts[:, 2].max())
    verts[:, 0] -= cx
    verts[:, 2] -= cz
    verts[:, 1] += (drop_height - float(verts[:, 1].min()))
    surface_tris = build_surface_tris(tets, verts)
    log.info(f"[u] loaded {len(verts)} verts, {len(tets)} tets, "
             f"{len(surface_tris)} surface tris from {geo_path}")
    return verts, tets, surface_tris


def _build_uv_sphere(center, radius, n_lat=16, n_lon=24):
    """Triangulated UV sphere, for rendering the rigid SDF only (the sim never
    touches these verts)."""
    cx, cy, cz = (float(c) for c in center)
    verts = []
    for i in range(n_lat + 1):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / n_lon
            verts.append([
                cx + radius * np.sin(theta) * np.cos(phi),
                cy + radius * np.cos(theta),
                cz + radius * np.sin(theta) * np.sin(phi),
            ])
    verts = np.asarray(verts, dtype=np.float32)
    tris = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + (j + 1) % n_lon
            tris.append([a, c, d])
            tris.append([a, d, b])
    return verts, np.asarray(tris, dtype=np.int32)


class SdfCollisionDemo:
    """Owns the SDF shape parameters + the per-vertex contact buffers, and
    drives the one-kernel contact solve each substep.

    Registered on the sim:
      * ``collision_solver`` -> ``sim.constraint_solvers`` (per substep)
      * ``is_collide`` / ``contact_normal`` -> consumed by
        ``sim.update_velocities`` for normal restitution + tangential friction.
    """

    def __init__(self, sim: MuscleSim, shape_type: int, *,
                 plane_y: float, sphere_center, sphere_radius: float,
                 contact_offset: float, restitution_n: float,
                 restitution_t: float, log_every: int):
        self.sim = sim
        self.shape_type = int(shape_type)
        self.plane_y = float(plane_y)
        self.sphere_c = wp.vec3(*[float(c) for c in sphere_center])
        self.sphere_r = float(sphere_radius)
        self.contact_offset = float(contact_offset)
        self.restitution_n = float(restitution_n)
        self.restitution_t = float(restitution_t)
        self.log_every = max(1, int(log_every))
        self._frame_counter = 0

        device = str(sim.pos.device)
        self.device = device
        n_v = int(sim.pos.shape[0])
        self.n_v = n_v

        # inv_mass for the muscle verts (pinned verts -> 0, never pushed).
        mass = sim.mass.numpy()
        inv = np.where(mass > 0.0, 1.0 / np.where(mass > 0.0, mass, 1.0), 0.0)
        if hasattr(sim, "stopped"):
            inv = np.where(sim.stopped.numpy() == 0, inv, 0.0)
        self.inv_mass = wp.array(inv.astype(np.float32), dtype=wp.float32,
                                 device=device)

        self.is_collide = wp.zeros(n_v, dtype=wp.int32, device=device)
        self.contact_normal = wp.zeros(n_v, dtype=wp.vec3, device=device)

        log.info(f"[sdf] shape={'sphere' if self.shape_type==SHAPE_SPHERE else 'plane'} "
                 f"plane_y={self.plane_y} sphere_c={list(self.sphere_c)} "
                 f"sphere_r={self.sphere_r} offset={self.contact_offset}")
        log.info(f"[sdf] restitution_n={self.restitution_n} "
                 f"restitution_t={self.restitution_t} verts={n_v}")

    def collision_solver(self):
        """Per-substep hook: one kernel launch — push every penetrating vertex
        out of the rigid SDF and flag it for the velocity solve."""
        # Kernel only sets is_collide/contact_normal on contact (never clears),
        # so clear here each substep.
        self.is_collide.zero_()
        self.contact_normal.zero_()
        wp.launch(
            solve_sdf_contact_kernel, dim=self.n_v,
            inputs=[
                self.sim.pos, self.sim.pprev, self.inv_mass,
                wp.int32(self.shape_type),
                wp.float32(self.plane_y),
                self.sphere_c, wp.float32(self.sphere_r),
                wp.float32(self.contact_offset),
                self.is_collide, self.contact_normal,
            ],
            device=self.device,
        )

    def log_diag(self, step: int):
        self._frame_counter += 1
        if self._frame_counter % self.log_every != 0:
            return
        n_collide = int(self.is_collide.numpy().sum())
        log.info(f"[sdf] step={step:4d} colliding_verts={n_collide:4d}")


def _resolve_shape(coll_cfg: dict, override: str | None) -> int:
    name = (override or coll_cfg.get("shape", "plane")).lower()
    if name == "sphere":
        return SHAPE_SPHERE
    if name == "plane":
        return SHAPE_PLANE
    raise ValueError(f"shape must be 'plane' or 'sphere', got {name!r}")


def main():
    p = argparse.ArgumentParser(description="Soft body vs analytic-SDF drop demo.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--shape", choices=["plane", "sphere"], default=None,
                   help="override collision.shape in the config")
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

    coll_cfg = getattr(cfg, "collision", {}) or {}
    shape_type = _resolve_shape(coll_cfg, args.shape)
    plane_cfg = coll_cfg.get("plane", {})
    sphere_cfg = coll_cfg.get("sphere", {})

    sim = _build_sim(cfg, float(coll_cfg.get("drop_height", 0.5)))

    demo = SdfCollisionDemo(
        sim, shape_type,
        plane_y=float(plane_cfg.get("y", 0.0)),
        sphere_center=sphere_cfg.get("center", [0.0, -0.6, 0.0]),
        sphere_radius=float(sphere_cfg.get("radius", 0.6)),
        contact_offset=float(coll_cfg.get("contact_offset", 0.0)),
        restitution_n=float(coll_cfg.get("restitution_n", 0.0)),
        restitution_t=float(coll_cfg.get("restitution_t", 0.8)),
        log_every=int(coll_cfg.get("log_every", 10)),
    )
    sim.constraint_solvers = [demo.collision_solver]
    sim.is_collide = demo.is_collide
    sim.contact_normal = demo.contact_normal
    sim.restitution_n = demo.restitution_n
    sim.restitution_t = demo.restitution_t

    # Rigid body render mesh.
    if shape_type == SHAPE_SPHERE:
        rigid_v, rigid_t = _build_uv_sphere(
            sphere_cfg.get("center", [0.0, -0.6, 0.0]),
            float(sphere_cfg.get("radius", 0.6)))
    else:
        rigid_v, rigid_t = _build_plane_mesh_xz(
            y=float(plane_cfg.get("y", 0.0)),
            half_extent=float(plane_cfg.get("half_extent", 1.5)),
            subdiv=int(plane_cfg.get("subdiv", 8)))

    renderer = None
    if cfg.render_mode == "usd":
        os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
        renderer = WarpRenderer(mode="usd", stage_path=cfg.output_path)
        log.info(f"[render] writing USD to {cfg.output_path}")

    muscle_t_flat = sim.surface_tris_flat
    rigid_t_flat = rigid_t.reshape(-1).astype(np.int32)
    render_stride = max(1, int(round((1.0 / cfg.render_fps) / cfg.dt)))

    for step in range(1, cfg.nsteps + 1):
        sim.step_cnt = step
        sim.step()
        demo.log_diag(step)

        if step % render_stride == 0 or step == 1:
            mu_pos = sim.pos.numpy()
            log.info(f"step={step:4d} t={step*cfg.dt:6.3f}s "
                     f"min_y={mu_pos[:,1].min():.4f}")
            if renderer is not None:
                inner = renderer.renderer
                inner.begin_frame(step * cfg.dt)
                inner.render_mesh(name="muscle", points=mu_pos,
                                  indices=muscle_t_flat, colors=(0.8, 0.3, 0.2))
                inner.render_mesh(name="rigid", points=rigid_v,
                                  indices=rigid_t_flat, colors=(0.55, 0.55, 0.55))
                inner.end_frame()

    if renderer is not None:
        renderer.save()
        log.info(f"Done — USD saved to {cfg.output_path}")


def _build_sim(cfg, drop_height: float) -> MuscleSim:
    """Load U.geo, shift above the contact surface, build a MuscleSim with just
    volume + ARAP (no fiber, no activation)."""
    verts, tets, _ = _load_u_mesh(cfg.geo_path, drop_height=drop_height)
    fiber_zero = np.zeros((tets.shape[0], 3), dtype=np.float32)
    fiber_zero[:, 0] = 1.0  # unused
    gravity_y = -abs(float(getattr(cfg, "gravity_z",
                                   getattr(cfg, "gravity", 9.8))))
    sim = MuscleSim.from_procedural(
        verts, tets, fiber_zero,
        constraint_configs=list(cfg.constraints),
        dts=float(cfg.dt),
        num_substeps=int(cfg.num_substeps),
        device=cfg.arch,
        density=float(cfg.density),
        veldamping=float(cfg.veldamping),
        gravity=gravity_y,
    )
    log.info(f"sim: n_verts={sim.n_verts} n_tets={sim.tet_np.shape[0]} "
             f"n_surface_tris={sim.surface_tris.shape[0]}")
    return sim


if __name__ == "__main__":
    main()
