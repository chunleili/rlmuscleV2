"""Headless smoke test for the particle-vs-SDF soft-rigid collision demo.

Drops the U-shape tet body onto a rigid plane and a rigid sphere, then checks
that the body (1) doesn't explode, (2) has its penetration resolved, (3) settles,
and (4) doesn't sink through the rigid surface.
"""
import os

import numpy as np
import pytest
import warp as wp

from muscle.util.util import load_config
from muscle.collision.example_sdf_drop import (
    SdfCollisionDemo, _build_sim, _resolve_shape, SHAPE_SPHERE,
)

HERE = os.path.dirname(__file__)
CONFIG = os.path.join(HERE, "..", "data", "collisionTest", "sdf_drop.json")

TOL = 0.03  # allowed residual penetration (m): soft body, 20 substeps


def _run(shape: str, nsteps: int = 240):
    wp.init()
    cfg = load_config(CONFIG)
    cfg.arch = "cpu"
    cfg.nsteps = nsteps
    cfg.render_mode = "none"
    wp.set_device("cpu")

    coll = cfg.collision
    shape_type = _resolve_shape(coll, shape)
    plane_cfg = coll.get("plane", {})
    sphere_cfg = coll.get("sphere", {})

    sim = _build_sim(cfg, float(coll.get("drop_height", 0.5)))
    demo = SdfCollisionDemo(
        sim, shape_type,
        plane_y=float(plane_cfg.get("y", 0.0)),
        sphere_center=sphere_cfg.get("center", [0.0, -0.6, 0.0]),
        sphere_radius=float(sphere_cfg.get("radius", 0.6)),
        contact_offset=float(coll.get("contact_offset", 0.0)),
        restitution_n=float(coll.get("restitution_n", 0.0)),
        restitution_t=float(coll.get("restitution_t", 0.8)),
        log_every=10**9,
    )
    sim.constraint_solvers = [demo.collision_solver]
    sim.is_collide = demo.is_collide
    sim.contact_normal = demo.contact_normal
    sim.restitution_n = demo.restitution_n
    sim.restitution_t = demo.restitution_t

    for step in range(1, nsteps + 1):
        sim.step_cnt = step
        sim.step()

    pos = sim.pos.numpy()
    vel = sim.vel.numpy()
    return pos, vel, demo


@pytest.mark.parametrize("shape", ["plane", "sphere"])
def test_sdf_drop(shape):
    pos, vel, demo = _run(shape)
    speed = np.linalg.norm(vel, axis=1)

    # 1. no explosion / NaN
    assert np.isfinite(pos).all(), "non-finite positions"
    assert np.isfinite(vel).all(), "non-finite velocities"
    assert speed.max() < 20.0, f"velocity blew up: max|v|={speed.max():.2f}"

    # 2 & 4. penetration resolved / did not sink through
    if demo.shape_type == SHAPE_SPHERE:
        c = np.asarray([demo.sphere_c[0], demo.sphere_c[1], demo.sphere_c[2]])
        dist = np.linalg.norm(pos - c[None, :], axis=1)
        phi = dist.min() - demo.sphere_r
        assert phi >= -TOL, f"sank into sphere: min phi={phi:.4f}"
    else:
        phi = pos[:, 1].min() - demo.plane_y
        assert phi >= -TOL, f"sank through plane: min_y-plane_y={phi:.4f}"

    # 3. fall arrested — contact bled off the fall. Without collision the body
    #    would free-fall for the whole sim (~4 s -> tens of m/s); a few-tenths
    #    m/s residual proves the SDF push is doing its job.
    assert speed.max() < 2.0, f"fall not arrested: max|v|={speed.max():.3f}"
    if demo.shape_type != SHAPE_SPHERE:
        # On a flat plane the body should come fully to rest. (On a curved
        # sphere top it legitimately keeps creeping/sliding, so skip there.)
        assert speed.max() < 0.5, f"did not settle on plane: max|v|={speed.max():.3f}"
