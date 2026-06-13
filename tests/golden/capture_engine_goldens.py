# tests/golden/capture_engine_goldens.py
"""Capture per-frame pos/vel goldens from the CURRENT (pre-refactor) sims.

Run ONCE on pre-refactor HEAD:
    PYTHONPATH=src python tests/golden/capture_engine_goldens.py

Writes output/golden/<sim>.npz with arrays pos[F,N,3] (+ vel where dynamic).
These are the bit-wise reference for tests/test_engine_parity.py.
"""
import json
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "golden"
DEVICE = "cuda:0"
N_FRAMES = 5


def _save(name, pos_list, vel_list=None):
    OUT.mkdir(parents=True, exist_ok=True)
    d = {"pos": np.stack(pos_list).astype(np.float32)}
    if vel_list is not None:
        d["vel"] = np.stack(vel_list).astype(np.float32)
    np.savez(OUT / f"{name}.npz", **d)
    print(f"  wrote {OUT / f'{name}.npz'}  pos{d['pos'].shape}")


def capture_oneway():
    from muscle.coupleOneWayGmg.solver import OneWaySim
    cfg = json.load(open(ROOT / "data/lion/config/lion_nomg.json"))
    cfg["geo_dir"] = str(ROOT / "data/lion/model")
    cfg["end_frame"] = N_FRAMES
    sim = OneWaySim(cfg, device=DEVICE)
    pos = []
    for f in range(sim.start_frame, sim.start_frame + N_FRAMES):
        sim.step_frame(f)
        pos.append(sim.positions_np().copy())
    _save("oneway_lion", pos)


def _capture_oneway(name, cfg_rel):
    """Generic OneWaySim capture: absolutize paths, run N frames from
    start_frame. Covers multigrid / fascia phases when the config enables them."""
    from muscle.coupleOneWayGmg.solver import OneWaySim
    cfg = json.load(open(ROOT / cfg_rel))
    for k in ("geo_dir", "cons_dir", "rest_path", "fascia_path"):
        if k in cfg:
            cfg[k] = str(ROOT / cfg[k])
    cfg["end_frame"] = int(cfg.get("start_frame", 1)) + N_FRAMES
    sim = OneWaySim(cfg, device=DEVICE)
    s0 = sim.start_frame
    pos = []
    for f in range(s0, s0 + N_FRAMES):
        sim.step_frame(f)
        pos.append(sim.positions_np().copy())
    _save(name, pos)


def capture_oneway_humanv2():
    # use_multigrid=True -> exercises the multigrid V-cycle pre_phase.
    _capture_oneway("oneway_humanv2_gmg", "data/humanV2/config/humanV2_gmg.json")


def capture_oneway_fascia():
    # use_multigrid=True + fascia -> exercises multigrid + anchor + glue phases.
    _capture_oneway("oneway_fascia", "data/humanV2/config/humanV2_fascia.json")


def capture_musclesim():
    from muscle.engine import MuscleSim
    from muscle.util.util import load_config
    cfg = load_config(str(ROOT / "data/muscle/config/bicep.json"))
    cfg.gui = False
    cfg.render_mode = None
    sim = MuscleSim(cfg)
    pos = []
    for _ in range(N_FRAMES):
        sim.step()
        pos.append(sim.pos.numpy().copy())
    _save("musclesim_bicep", pos)


def capture_massspring():
    import numpy as _np
    from muscle.collision.udcf.massspring import MassSpringTet, MassSpringParams
    from muscle.collision.udcf.solver import UdcfDcdCollider, UdcfParams
    from muscle.collision.udcf.kernel import box_boundary_kernel
    from muscle.util.geo_io import GeoFile
    from muscle.util.mesh_io import build_surface_tris
    g = GeoFile(str(ROOT / "data/collisionTest/U.geo"))
    verts = _np.asarray(g.positions, _np.float32).copy(); verts[:, 1] += 0.2
    tets = _np.asarray(g.tet_indices, _np.int32)
    dt = 1.0 / 60.0; nsub = 20
    ms = MassSpringTet(verts, tets, dt / nsub,
                       params=MassSpringParams(), device=DEVICE)
    v0 = _np.zeros((verts.shape[0], 3), _np.float32); v0[:, 1] = -5.0
    ms.vel.assign(v0)
    surf = build_surface_tris(tets, verts).astype(_np.int32)
    mp = max(65536, surf.shape[0] * (surf.shape[0] - 1) // 2)
    dv = _np.array([[1e4, 1e4, 1e4], [1e4 + 1, 1e4, 1e4], [1e4, 1e4, 1e4 + 1]], _np.float32)
    dcd = UdcfDcdCollider(ms.pos, ms.inv_mass, surface_tris=surf,
                          static_verts=dv, static_tris=_np.array([[0, 1, 2]], _np.int32),
                          params=UdcfParams(allow_self=True, max_pairs=mp, thickness=1e-3,
                                            jacobi_avg=False, gauss_seidel=True, max_push=0.0),
                          device=DEVICE)
    lo = wp.vec3(-2.0, 0.0, -2.0); hi = wp.vec3(2.0, 5.0e4, 2.0)
    pos, vel = [], []
    for _f in range(N_FRAMES):
        for _ in range(nsub):
            ms.integrate(); ms.solve_material()
            wp.launch(box_boundary_kernel, dim=verts.shape[0],
                      inputs=[ms.pos, ms.pprev, lo, hi, wp.float32(0.5), wp.float32(0.4)],
                      device=DEVICE)
            ms.solve_inversion(); dcd.solve(0); ms.update_velocities()
        pos.append(ms.pos.numpy().copy()); vel.append(ms.vel.numpy().copy())
    _save("massspring_u", pos, vel)


if __name__ == "__main__":
    wp.init()
    for name, fn in [("oneway_lion", capture_oneway),
                     ("oneway_humanv2_gmg", capture_oneway_humanv2),
                     ("oneway_fascia", capture_oneway_fascia),
                     ("musclesim_bicep", capture_musclesim),
                     ("massspring_u", capture_massspring)]:
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  SKIP {name}: {e}")
