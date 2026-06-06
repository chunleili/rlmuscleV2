"""Synthetic merged-mesh fascia smoke test (no real assets, runs on CPU)."""
import numpy as np
import warp as wp

wp.init()


def test_fascia_glue_pulls_muscle_inward():
    """Two-tet system: a shrinking fascia tet glued to a muscle vertex pulls
    that muscle vertex toward the fascia."""
    from muscle.coupleOneWayGmg import solver as S

    muscle_pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    fascia_pos = muscle_pos + np.array([1.5, 0, 0], dtype=np.float32)
    state = S.build_merged_state(
        muscle_pos=muscle_pos,
        muscle_tet=np.array([[0, 1, 2, 3]], np.int32),
        fascia_pos=fascia_pos,
        fascia_tet=np.array([[0, 1, 2, 3]], np.int32),
        fascia_rest_scale=0.5,
        muscle_stiffness=1.0e9,
        fascia_stiffness=5000.0,
        glue_fascia=np.array([0], np.int32),       # fascia LOCAL vert 0
        glue_muscle=np.array([1], np.int32),       # muscle vert 1 (the +x corner)
        glue_restlen=np.array([0.1], np.float32),
        glue_stiffness=np.array([1000.0], np.float32),
        anchor_fascia=np.zeros(0, np.int32),
        anchor_muscle=np.zeros(0, np.int32),
        total_mass=1.0, device=None,
    )
    p0 = state.pos.numpy().copy()
    S.run_substeps(state, nsubsteps=20, fine_iters=3, glue_iters=4,
                   dt=0.004166, omega=0.2)
    p1 = state.pos.numpy()
    assert not np.isnan(p1).any()
    # muscle vert 1 (index 1) pulled toward fascia vert 0 (index 4) → +x increases
    assert p1[1][0] > p0[1][0]


def test_merged_state_handles_unreferenced_and_degenerate(monkeypatch):
    """An extra fascia point with no incident tet, plus a degenerate tet,
    must not produce inf/NaN in inv_mass, B, or after stepping."""
    import numpy as np
    import warp as wp
    from muscle.coupleOneWayGmg import solver as S

    muscle_pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    # fascia: 4 verts forming a tet + a 5th unreferenced point far away
    fascia_pos = np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1],
                           [9, 9, 9]], dtype=np.float32)
    state = S.build_merged_state(
        muscle_pos=muscle_pos, muscle_tet=np.array([[0, 1, 2, 3]], np.int32),
        fascia_pos=fascia_pos, fascia_tet=np.array([[0, 1, 2, 3]], np.int32),
        fascia_rest_scale=0.9, muscle_stiffness=1e9, fascia_stiffness=5000.0,
        glue_fascia=np.array([0], np.int32), glue_muscle=np.array([1], np.int32),
        glue_restlen=np.array([0.1], np.float32), glue_stiffness=np.array([1000.0], np.float32),
        anchor_fascia=np.zeros(0, np.int32), anchor_muscle=np.zeros(0, np.int32),
        total_mass=1.0, device=None)
    im = state.inv_mass.numpy()
    assert np.isfinite(im).all()
    assert im[-1] == 0.0  # unreferenced 5th fascia point -> pinned (inv_mass 0)
    assert np.isfinite(state.B.numpy()).all()
    S.run_substeps(state, nsubsteps=10, fine_iters=2, glue_iters=2, dt=0.004166, omega=0.2)
    assert np.isfinite(state.pos.numpy()).all()


def test_substep_runs_glue_when_fascia_present(monkeypatch):
    """OneWaySim.substep applies glue iters when a merged state exists."""
    import numpy as np
    import warp as wp
    from muscle.coupleOneWayGmg import solver as S

    calls = {"glue": 0}
    real_solve_glue = S.solve_glue

    def spy(state, dt, omega, iters, device=None):
        calls["glue"] += 1
        return real_solve_glue(state, dt, omega, iters, device)

    monkeypatch.setattr(S, "solve_glue", spy)

    sim = S.OneWaySim.__new__(S.OneWaySim)
    sim.use_multigrid = False
    sim.fine_iters = 2
    sim.fascia_glue_iters = 3
    sim.dt = 0.004166
    sim.omega = 0.2
    sim._device = None
    sim.merged = S.build_merged_state(
        muscle_pos=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32),
        muscle_tet=np.array([[0, 1, 2, 3]], np.int32),
        fascia_pos=np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1]], np.float32),
        fascia_tet=np.array([[0, 1, 2, 3]], np.int32),
        fascia_rest_scale=0.9, muscle_stiffness=1e9, fascia_stiffness=5000.0,
        glue_fascia=np.array([0], np.int32), glue_muscle=np.array([1], np.int32),
        glue_restlen=np.array([0.1], np.float32), glue_stiffness=np.array([1000.0], np.float32),
        anchor_fascia=np.zeros(0, np.int32), anchor_muscle=np.zeros(0, np.int32),
        total_mass=1.0, device=None)
    sim.pos = sim.merged.pos
    sim.pos_mid = sim.merged.pos_mid
    sim.tet_idx = sim.merged.tet_idx
    sim.B = sim.merged.B
    sim.alpha = sim.merged.alpha
    sim.L = sim.merged.L
    sim.inv_mass = sim.merged.inv_mass
    sim.NT = sim.merged.NT

    sim.substep()
    assert calls["glue"] == 1
    assert not np.isnan(sim.pos.numpy()).any()


import json
from pathlib import Path
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_HV2 = _ROOT / "data" / "humanV2" / "model"
_FASCIA_CFG = _ROOT / "data" / "humanV2" / "config" / "humanV2_fascia.json"

skip_no_hv2 = pytest.mark.skipif(
    not (_HV2 / "muscle" / "restpos.geo").exists()
    or not (_HV2 / "fascia" / "fascia.geo").exists()
    or not _FASCIA_CFG.exists(),
    reason="humanV2 model not extracted / fascia.geo not copied / config missing",
)


@skip_no_hv2
def test_humanV2_fascia_one_frame_no_nan():
    from muscle.coupleOneWayGmg.solver import OneWaySim, load_config
    cfg = load_config(str(_FASCIA_CFG))
    cfg["end_frame"] = cfg["start_frame"]
    # Run on CPU if no GPU; keep it to a single frame.
    sim = OneWaySim(cfg)
    assert sim.NV > 157369          # muscle (157369) + fascia merged
    assert sim.merged.n_glue > 0
    rest = sim.positions_np().copy()
    sim.step_frame(sim.start_frame)
    after = sim.positions_np()
    assert not np.isnan(after).any() and not np.isinf(after).any()
    # fascia verts (index >= nvm) actually moved (contraction/glue active)
    fascia_disp = np.linalg.norm(after[sim.nvm:] - rest[sim.nvm:], axis=1).mean()
    assert fascia_disp > 1e-5
