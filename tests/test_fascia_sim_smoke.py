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
