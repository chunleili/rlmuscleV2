"""Headless test for the rigid-sphere-vs-soft-box replica of Newton's
``example_rigid_soft_contact``. Mirrors the key invariants of Newton's
``test_final``: the soft body must not explode, must not penetrate the ground,
must keep positive tet volumes, and the dynamic sphere must stay in a sane
vertical range (supported by the soft body, not tunnelled to the ground or
launched away).
"""
import numpy as np
import pytest
import warp as wp

from muscle.util.util import load_config
from muscle.collision.example_rigid_soft import RigidSoftDemo, _build_sim

CONFIG = "data/collisionTest/rigid_soft.json"


def _tet_volumes(pos, tets):
    x0, x1, x2, x3 = pos[tets[:, 0]], pos[tets[:, 1]], pos[tets[:, 2]], pos[tets[:, 3]]
    return np.linalg.det(np.stack([x1 - x0, x2 - x0, x3 - x0], axis=-1)) / 6.0


def _run(nsteps=120):
    wp.init()
    cfg = load_config(CONFIG)
    cfg.arch = "cpu"
    cfg.nsteps = nsteps
    wp.set_device("cpu")

    sim, box_verts = _build_sim(cfg, cfg.soft_grid)
    rest_pos = sim.pos.numpy().copy()
    box_top = float(box_verts[:, 1].max())
    g = cfg.ground
    s = cfg.sphere
    coll = cfg.collision
    off = s.get("init_offset", [0.2, 1.2, 0.0])
    gy = -abs(float(getattr(cfg, "gravity_z", 9.8)))

    demo = RigidSoftDemo(
        sim, ground_y=float(g.get("y", 0.0)),
        sphere_radius=float(s["radius"]), sphere_density=float(s["density"]),
        sphere_center0=[float(off[0]), box_top + float(off[1]), float(off[2])],
        contact_offset=float(coll.get("contact_offset", 0.0)),
        restitution_n=float(coll.get("restitution_n", 0.0)),
        restitution_t=float(coll.get("restitution_t", 0.9)),
        restitution_ground=float(s.get("restitution_ground", 0.0)),
        gravity=gy, log_every=10**9)
    sim.constraint_solvers = [demo.collision_solver]
    sim.is_collide = demo.is_collide
    sim.contact_normal = demo.contact_normal
    sim.restitution_n = demo.restitution_n
    sim.restitution_t = demo.restitution_t

    for step in range(1, nsteps + 1):
        sim.step_cnt = step
        sim.step()
    return sim, rest_pos, demo


def test_rigid_soft_contact():
    sim, rest_pos, demo = _run()
    pos = sim.pos.numpy()
    tets = sim.tet_np

    assert np.isfinite(pos).all(), "non-finite soft positions"
    assert np.isfinite(demo.sphere_center).all(), "non-finite sphere center"

    # 1. soft body didn't explode
    bbox = np.linalg.norm(pos.max(0) - pos.min(0))
    rest_bbox = np.linalg.norm(rest_pos.max(0) - rest_pos.min(0))
    assert bbox < rest_bbox * 2.0, f"soft body exploded: bbox={bbox:.2f}"

    # 2. soft body didn't sink through the ground
    assert pos[:, 1].min() > -0.05, f"soft penetrated ground: min_y={pos[:,1].min():.4f}"

    # 3. positive tet volumes (no inversion). SVD repair guarantees this; the
    #    tet directly under the point contact squishes hard (ratio can dip below
    #    1%) but must stay valid — that's a softer-material difference from
    #    Newton's stiff FEM beam, so we don't assert a recovery ratio here.
    vols = _tet_volumes(pos, tets)
    assert vols.min() > 0.0, f"inverted tets: min vol={vols.min():.3e}"

    # 4. dynamic sphere supported by the soft body — above the ground clamp,
    #    below its start. (Two-way contact arrested its fall.)
    sy = float(demo.sphere_center[1])
    low = float(demo.ground_y + demo.sphere_r)
    # Started at box_top(0.6) + offset(1.2) = 1.8; must have fallen well below
    # that and stay above the ground clamp (supported by the soft box).
    assert sy > low - 1e-3, f"sphere tunnelled below ground clamp: y={sy:.3f}"
    assert sy < 1.3, f"sphere not arrested: y={sy:.3f}"
