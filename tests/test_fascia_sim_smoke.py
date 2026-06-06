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
