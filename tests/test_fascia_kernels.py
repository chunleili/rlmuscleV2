"""Unit tests for fascia Warp kernels."""
import numpy as np
import warp as wp

wp.init()

_REG_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)


def _tet_volume(p, idx=(0, 1, 2, 3)):
    a, b, c, d = (p[i] for i in idx)
    return abs(np.dot(np.cross(b - a, c - a), d - a)) / 6.0


def test_scaled_precompute_shrinks_rest():
    """B = inv(s*Dm); rest_volume uses UNSCALED Dm; alpha = 1/(stiff*V0)."""
    from muscle.coupleOneWayGmg.solver import precompute_rest_scaled_kernel

    pos = wp.array(_REG_TET, dtype=wp.vec3)
    tets = wp.array(np.array([[0, 1, 2, 3]], dtype=np.int32), dtype=wp.vec4i)
    scale = wp.array(np.array([0.5], dtype=np.float32), dtype=wp.float32)
    stiff = wp.array(np.array([5000.0], dtype=np.float32), dtype=wp.float32)
    B = wp.zeros(1, dtype=wp.mat33)
    V0 = wp.zeros(1, dtype=wp.float32)
    alpha = wp.zeros(1, dtype=wp.float32)

    wp.launch(precompute_rest_scaled_kernel, dim=1,
              inputs=[pos, tets, scale, stiff, B, V0, alpha])
    wp.synchronize()

    # Dm = I (regular tet) → B = inv(0.5*I) = 2*I
    np.testing.assert_allclose(B.numpy()[0], 2.0 * np.eye(3), atol=1e-5)
    np.testing.assert_allclose(V0.numpy()[0], 1.0 / 6.0, atol=1e-7)   # unscaled
    np.testing.assert_allclose(alpha.numpy()[0], 1.0 / (5000.0 * (1.0 / 6.0)), rtol=1e-5)


def test_free_tet_contracts_to_scale():
    """A free tet driven by ARAP toward a 0.5-scaled rest converges to 1/8 volume."""
    from muscle.coupleOneWayGmg.solver import (
        precompute_rest_scaled_kernel, arap_svd_kernel)

    pos_np = _REG_TET.copy()
    pos = wp.array(pos_np, dtype=wp.vec3)
    pos_mid = wp.clone(pos)
    tets = wp.array(np.array([[0, 1, 2, 3]], dtype=np.int32), dtype=wp.vec4i)
    scale = wp.array(np.array([0.5], dtype=np.float32), dtype=wp.float32)
    stiff = wp.array(np.array([1.0e9], dtype=np.float32), dtype=wp.float32)
    inv_mass = wp.array(np.ones(4, dtype=np.float32), dtype=wp.float32)
    B = wp.zeros(1, dtype=wp.mat33)
    V0 = wp.zeros(1, dtype=wp.float32)
    alpha = wp.zeros(1, dtype=wp.float32)
    L = wp.zeros(1, dtype=wp.float32)

    wp.launch(precompute_rest_scaled_kernel, dim=1,
              inputs=[pos, tets, scale, stiff, B, V0, alpha])
    dt = 0.004166
    for _ in range(2000):
        wp.copy(pos_mid, pos)
        wp.launch(arap_svd_kernel, dim=1,
                  inputs=[pos_mid, tets, B, inv_mass, alpha, L, dt, 0.2, pos])
    wp.synchronize()

    v = _tet_volume(pos.numpy())
    np.testing.assert_allclose(v, (0.5 ** 3) * (1.0 / 6.0), rtol=0.05)
