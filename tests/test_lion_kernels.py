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


def test_arap_identity_no_displacement():
    """F=I → constraint magnitude below threshold → no displacement, L unchanged."""
    from muscle.coupleOneWay.lion_sim import precompute_rest_kernel, arap_svd_kernel

    pos_np = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    pos, tets = _make_tet_arrays(pos_np)
    pos_mid = wp.clone(pos)
    B = wp.zeros(1, dtype=wp.mat33)
    V0 = wp.zeros(1, dtype=wp.float32)
    alpha = wp.zeros(1, dtype=wp.float32)
    inv_mass = wp.array(np.ones(4, dtype=np.float32), dtype=wp.float32)
    L = wp.zeros(1, dtype=wp.float32)
    mu = 1.0e9
    dt = 0.004166
    omega = 0.1

    wp.launch(precompute_rest_kernel, dim=1, inputs=[pos, tets, B, V0, alpha, float(mu)])
    wp.launch(
        arap_svd_kernel,
        dim=1,
        inputs=[pos_mid, tets, B, inv_mass, alpha, L, float(dt), float(omega), pos],
    )
    wp.synchronize()

    np.testing.assert_allclose(pos.numpy(), pos_np, atol=1e-6)
    assert abs(float(L.numpy()[0])) < 1e-12


def test_arap_uniform_stretch_contracts_edges():
    """Stretched tet (×1.1) → C>0 → one ARAP iter shrinks edges toward rest."""
    from muscle.coupleOneWay.lion_sim import precompute_rest_kernel, arap_svd_kernel

    rest_np = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    stretched_np = rest_np * 1.1
    pos = wp.array(stretched_np, dtype=wp.vec3)
    tets = wp.array(np.array([[0, 1, 2, 3]], dtype=np.int32), dtype=wp.vec4i)
    pos_mid = wp.clone(pos)
    B = wp.zeros(1, dtype=wp.mat33)
    V0 = wp.zeros(1, dtype=wp.float32)
    alpha = wp.zeros(1, dtype=wp.float32)
    inv_mass = wp.array(np.ones(4, dtype=np.float32), dtype=wp.float32)
    L = wp.zeros(1, dtype=wp.float32)
    mu = 1.0e9
    dt = 0.004166
    omega = 0.1

    # Precompute B on the REST tet.
    rest_pos = wp.array(rest_np, dtype=wp.vec3)
    wp.launch(precompute_rest_kernel, dim=1,
              inputs=[rest_pos, tets, B, V0, alpha, float(mu)])

    wp.launch(
        arap_svd_kernel,
        dim=1,
        inputs=[pos_mid, tets, B, inv_mass, alpha, L, float(dt), float(omega), pos],
    )
    wp.synchronize()

    new = pos.numpy()
    edge_before = 1.1
    edge_after = float(np.linalg.norm(new[1] - new[0]))
    assert edge_after < edge_before, (
        f"edge should contract under one ARAP iter: before={edge_before}, after={edge_after}"
    )
    # And L should have moved (became negative, since we're stretched past rest).
    assert float(L.numpy()[0]) < 0.0
