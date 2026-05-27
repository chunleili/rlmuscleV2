"""Single-tet unit tests for the lion ARAP-HPBD Warp kernels."""
import numpy as np
import warp as wp

wp.init()


def _make_tet_arrays(pos_np):
    pos = wp.array(pos_np.astype(np.float32), dtype=wp.vec3)
    tets = wp.array(np.array([[0, 1, 2, 3]], dtype=np.int32), dtype=wp.vec4i)
    return pos, tets


def test_precompute_regular_tet():
    """Reference tet (0,0,0),(1,0,0),(0,1,0),(0,0,1) → Dm=I, V0=1/6, alpha=6/mu."""
    from muscle.coupleOneWay.lion_sim import precompute_rest_kernel

    pos_np = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    pos, tets = _make_tet_arrays(pos_np)
    B = wp.zeros(1, dtype=wp.mat33)
    V0 = wp.zeros(1, dtype=wp.float32)
    alpha = wp.zeros(1, dtype=wp.float32)
    mu = 1.0e9

    wp.launch(
        precompute_rest_kernel,
        dim=1,
        inputs=[pos, tets, B, V0, alpha, float(mu)],
    )
    wp.synchronize()

    B_np = B.numpy()[0]
    np.testing.assert_allclose(B_np, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(V0.numpy()[0], 1.0 / 6.0, atol=1e-7)
    np.testing.assert_allclose(alpha.numpy()[0], 6.0 / mu, rtol=1e-6)
