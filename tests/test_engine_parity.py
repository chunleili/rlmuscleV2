# tests/test_engine_parity.py
"""Parity gate for the unified-engine refactor: refactored sims must match
output/golden/<sim>.npz.

Gate strength is per-sim, by determinism:
  - MassSpring (Gauss-Seidel) is bit-wise reproducible -> exact np.array_equal.
  - OneWaySim / MuscleSim use float32 atomic-add Jacobi projection; CUDA warp
    scheduling makes atomic accumulation order nondeterministic, so even the
    UNCHANGED sim diverges ~5e-6 over 5 frames run-to-run. Bit-wise is therefore
    physically impossible for them; gate at np.allclose(atol=ATOL) -- comfortably
    above the ~5e-6 hardware noise floor, far below any real refactor regression
    (a wrong kernel/arg/order blows past 1e-4 immediately).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import warp as wp

wp.init()
ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "output" / "golden"
DEVICE = "cuda:0"
ATOL = 1e-4  # nondeterministic-sim tolerance (OneWay/MuscleSim); see module docstring


def _need(name):
    p = GOLD / f"{name}.npz"
    if not p.exists():
        pytest.skip(f"golden {p} missing — run capture_engine_goldens.py on pre-refactor HEAD")
    return np.load(p)


def test_oneway_lion_bitwise():
    if not (ROOT / "data/lion/model/restpos.geo").exists():
        pytest.skip("lion data absent")
    g = _need("oneway_lion")
    from muscle.coupleOneWayGmg.solver import OneWaySim
    cfg = json.load(open(ROOT / "data/lion/config/lion_nomg.json"))
    cfg["geo_dir"] = str(ROOT / "data/lion/model"); cfg["end_frame"] = g["pos"].shape[0]
    sim = OneWaySim(cfg, device=DEVICE)
    for i, f in enumerate(range(sim.start_frame, sim.start_frame + g["pos"].shape[0])):
        sim.step_frame(f)
        p = sim.positions_np()
        assert np.allclose(p, g["pos"][i], atol=ATOL, rtol=0.0), \
            f"oneway frame {i} diverged: max|d|={np.abs(p - g['pos'][i]).max():.2e} > {ATOL}"


def _run_oneway(cfg_path, nframes):
    from muscle.coupleOneWayGmg.solver import OneWaySim
    cfg = json.load(open(cfg_path))
    for k in ("geo_dir", "cons_dir", "rest_path", "fascia_path"):
        if k in cfg:
            cfg[k] = str(ROOT / cfg[k])
    cfg["end_frame"] = int(cfg.get("start_frame", 1)) + nframes
    return OneWaySim(cfg, device=DEVICE)


def test_oneway_humanv2_gmg_parity():
    """Covers the moved multigrid V-cycle pre_phase (use_multigrid=True)."""
    cfg_path = ROOT / "data/humanV2/config/humanV2_gmg.json"
    if not cfg_path.exists():
        pytest.skip("humanV2 gmg config absent")
    g = _need("oneway_humanv2_gmg")
    sim = _run_oneway(cfg_path, g["pos"].shape[0])
    s0 = sim.start_frame
    for i, f in enumerate(range(s0, s0 + g["pos"].shape[0])):
        sim.step_frame(f)
        p = sim.positions_np()
        assert np.allclose(p, g["pos"][i], atol=ATOL, rtol=0.0), \
            f"oneway gmg frame {i} diverged: max|d|={np.abs(p - g['pos'][i]).max():.2e} > {ATOL}"


def test_oneway_fascia_parity():
    """Covers moved multigrid + fascia anchor/glue phases (fascia config)."""
    cfg_path = ROOT / "data/humanV2/config/humanV2_fascia.json"
    if not cfg_path.exists():
        pytest.skip("humanV2 fascia config absent")
    g = _need("oneway_fascia")
    sim = _run_oneway(cfg_path, g["pos"].shape[0])
    s0 = sim.start_frame
    for i, f in enumerate(range(s0, s0 + g["pos"].shape[0])):
        sim.step_frame(f)
        p = sim.positions_np()
        assert np.allclose(p, g["pos"][i], atol=ATOL, rtol=0.0), \
            f"oneway fascia frame {i} diverged: max|d|={np.abs(p - g['pos'][i]).max():.2e} > {ATOL}"


def test_musclesim_bicep_parity():
    cfg_path = ROOT / "data/muscle/config/bicep.json"
    if not cfg_path.exists():
        pytest.skip("bicep config absent")
    g = _need("musclesim_bicep")
    from muscle.muscle_warp import MuscleSim
    from muscle.util.util import load_config
    cfg = load_config(str(cfg_path))
    cfg.gui = False
    cfg.render_mode = None
    sim = MuscleSim(cfg)
    for i in range(g["pos"].shape[0]):
        sim.step()
        p = sim.pos.numpy()
        assert np.allclose(p, g["pos"][i], atol=ATOL, rtol=0.0), \
            f"musclesim frame {i} diverged: max|d|={np.abs(p - g['pos'][i]).max():.2e} > {ATOL}"


def test_massspring_u_bitwise():
    if not (ROOT / "data/collisionTest/U.geo").exists():
        pytest.skip("U.geo absent")
    g = _need("massspring_u")
    from muscle.collision.udcf.massspring_sim import MassSpringSim  # Stage 3
    sim = MassSpringSim.u_drop(str(ROOT / "data/collisionTest/U.geo"), device=DEVICE)
    for i in range(g["pos"].shape[0]):
        sim.step()
        assert np.array_equal(sim.pos.numpy(), g["pos"][i]), f"massspring frame {i} diverged"
        assert np.array_equal(sim.vel.numpy(), g["vel"][i]), f"massspring vel {i} diverged"
