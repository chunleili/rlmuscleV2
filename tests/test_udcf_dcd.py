"""Golden replay tests: our Warp UDCF DCD kernels vs the instrumented
Gaia_UDCF C++ reference (tests/golden/udcf_pairs.npz).

Parity is asserted record-level. A tiny mismatch budget (0.1%) absorbs the
one known reference bug we deliberately do not replicate (stale Q[i][j]
reuse across pairs in narrowPhase) plus fp32 candidate ties.
"""
from pathlib import Path

import numpy as np
import pytest
import warp as wp

wp.init()

GOLDEN = Path(__file__).parent / "golden" / "udcf_pairs.npz"

skip_no_golden = pytest.mark.skipif(not GOLDEN.exists(),
                                    reason="golden npz not generated")

ATOL = 2e-5          # absolute tolerance on depths / barys / normals
PASS_RATE = 0.999    # allowed mismatch budget
TIE_EPS = 5e-8       # C1-vs-C2 tie window: below this the branch pick is
                     # fp noise (CPU-Eigen vs GPU-Warp last-bit divergence)


def _load():
    return dict(np.load(GOLDEN))


def _run_narrow(g):
    """Feed every golden pair (T0, T1) through narrow_kernel as one batch.

    Each record gets its own 6 verts; record i uses verts [6i..6i+6) and
    tris (2i, 2i+1), so contacts map back to records via vert_id // 6.
    """
    from muscle.collision.udcf_dcd import narrow_kernel

    n = len(g["stage"])
    x_np = np.stack([g["q0"], g["q1"]], axis=1).reshape(-1, 3)
    tris_np = np.arange(n * 6, dtype=np.int32).reshape(n * 2, 3)
    pair_a = np.arange(0, 2 * n, 2, dtype=np.int32)
    pair_b = np.arange(1, 2 * n, 2, dtype=np.int32)

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    x = wp.array(x_np.astype(np.float32), dtype=wp.vec3, device=dev)
    tris = wp.array2d(tris_np, dtype=wp.int32, device=dev)
    pa = wp.array(pair_a, dtype=wp.int32, device=dev)
    pb = wp.array(pair_b, dtype=wp.int32, device=dev)
    n_pairs = wp.array([n], dtype=wp.int32, device=dev)

    max_c = n
    n_contacts = wp.zeros(1, dtype=wp.int32, device=dev)
    c_cur = wp.zeros((max_c, 3), dtype=wp.int32, device=dev)
    c_closest = wp.zeros((max_c, 3), dtype=wp.int32, device=dev)
    c_C = wp.zeros(max_c, dtype=wp.float32, device=dev)
    c_C1 = wp.zeros(max_c, dtype=wp.float32, device=dev)
    c_C2 = wp.zeros(max_c, dtype=wp.float32, device=dev)
    c_n = wp.zeros(max_c, dtype=wp.vec3, device=dev)
    c_b1 = wp.zeros(max_c, dtype=wp.vec3, device=dev)
    c_b2 = wp.zeros(max_c, dtype=wp.vec3, device=dev)

    wp.launch(narrow_kernel, dim=n,
              inputs=[pa, pb, n_pairs, tris, x,
                      wp.float32(1.0e-6), wp.int32(max_c),
                      n_contacts, c_cur, c_closest,
                      c_C, c_C1, c_C2, c_n, c_b1, c_b2],
              device=dev)
    wp.synchronize()
    nc = int(n_contacts.numpy()[0])
    return dict(nc=nc, cur=c_cur.numpy()[:nc], closest=c_closest.numpy()[:nc],
                C=c_C.numpy()[:nc], C1=c_C1.numpy()[:nc], C2=c_C2.numpy()[:nc],
                n=c_n.numpy()[:nc], b1=c_b1.numpy()[:nc], b2=c_b2.numpy()[:nc])


@skip_no_golden
def test_narrow_golden_replay():
    g = _load()
    out = _run_narrow(g)
    n = len(g["stage"])
    applied = (g["stage"] >= 2)

    # Map each emitted contact back to its golden record.
    got = np.zeros(n, dtype=bool)
    rec_of_contact = out["cur"][:, 0] // 6
    mismatches = []
    for ci, ri in enumerate(rec_of_contact):
        got[ri] = True
        st = 2 if (out["cur"][ci, 0] % 6) < 3 else 3   # cur==T0 → stage2
        depths_ok = (abs(out["C1"][ci] - g["C1"][ri]) <= ATOL
                     and abs(out["C2"][ci] - g["C2"][ri]) <= ATOL)
        if st == g["stage"][ri]:
            ok = (depths_ok
                  and np.allclose(out["n"][ci], g["n"][ri], atol=ATOL)
                  and np.allclose(out["b1"][ci], g["bary_cur"][ri], atol=1e-3)
                  and np.allclose(out["b2"][ci], g["bary_closest"][ri], atol=1e-3))
        else:
            # Branch flip is acceptable ONLY on a C1≈C2 tie (≤ TIE_EPS, on
            # either side's arithmetic): the reference's pick is last-bit fp
            # noise there, and the golden n/bary belong to the other branch
            # so only the depths are comparable.
            ok = depths_ok and (
                abs(g["C1"][ri] - g["C2"][ri]) <= TIE_EPS
                or abs(out["C1"][ci] - out["C2"][ci]) <= TIE_EPS)
        if not ok:
            mismatches.append(ri)

    false_neg = np.flatnonzero(applied & ~got)        # golden applied, we rejected
    false_pos = np.flatnonzero(~applied & got)        # golden rejected, we applied
    bad = len(mismatches) + len(false_neg) + len(false_pos)
    rate = 1.0 - bad / n
    assert rate >= PASS_RATE, (
        f"parity {rate:.4%} < {PASS_RATE:.1%}: value-mismatch={len(mismatches)} "
        f"false-neg={len(false_neg)} false-pos={len(false_pos)} of {n}; "
        f"first bad records: {(mismatches + list(false_neg) + list(false_pos))[:10]}")


def _run_respond(x_np, prev_np, im_np, cur_ids, closest_ids, C, n, b1, b2,
                 mult, mu_s, mu_d):
    """One respond_kernel launch over a batch of single contacts."""
    from muscle.collision.udcf_dcd import respond_kernel, apply_kernel

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    nv = x_np.shape[0]
    nc = len(C)
    x = wp.array(x_np.astype(np.float32), dtype=wp.vec3, device=dev)
    xprev = wp.array(prev_np.astype(np.float32), dtype=wp.vec3, device=dev)
    im = wp.array(im_np.astype(np.float32), dtype=wp.float32, device=dev)
    n_contacts = wp.array([nc], dtype=wp.int32, device=dev)
    c_cur = wp.array2d(cur_ids.astype(np.int32), dtype=wp.int32, device=dev)
    c_closest = wp.array2d(closest_ids.astype(np.int32), dtype=wp.int32, device=dev)
    c_C = wp.array(C.astype(np.float32), dtype=wp.float32, device=dev)
    c_n = wp.array(n.astype(np.float32), dtype=wp.vec3, device=dev)
    c_b1 = wp.array(b1.astype(np.float32), dtype=wp.vec3, device=dev)
    c_b2 = wp.array(b2.astype(np.float32), dtype=wp.vec3, device=dev)
    mu_s_a = wp.array(mu_s.astype(np.float32), dtype=wp.float32, device=dev)
    mu_d_a = wp.array(mu_d.astype(np.float32), dtype=wp.float32, device=dev)
    dP = wp.zeros(nv, dtype=wp.vec3, device=dev)
    dPw = wp.zeros(nv, dtype=wp.float32, device=dev)

    wp.launch(respond_kernel, dim=nc,
              inputs=[n_contacts, c_cur, c_closest, c_C, c_n, c_b1, c_b2,
                      x, xprev, im, wp.float32(mult), mu_s_a, mu_d_a,
                      dP, dPw],
              device=dev)
    wp.launch(apply_kernel, dim=nv, inputs=[x, dP, dPw, wp.int32(1)],
              device=dev)
    wp.synchronize()
    return x.numpy()


@skip_no_golden
def test_respond_golden_replay():
    g = _load()
    sel = np.flatnonzero(g["stage"] >= 2)
    nrec = len(sel)
    # 6 isolated verts per record: [T0 verts, T1 verts]
    x_np = np.stack([g["q0"][sel], g["q1"][sel]], axis=1).reshape(nrec, 6, 3)
    prev_np = np.stack([g["prev0"][sel], g["prev1"][sel]], axis=1).reshape(nrec, 6, 3)
    base = (np.arange(nrec) * 6)[:, None]
    t0_ids = base + np.array([[0, 1, 2]])
    t1_ids = base + np.array([[3, 4, 5]])
    st = g["stage"][sel]
    cur_ids = np.where((st == 2)[:, None], t0_ids, t1_ids)
    closest_ids = np.where((st == 2)[:, None], t1_ids, t0_ids)
    fixed_t0 = g["fixed0"][sel]
    fixed_t1 = g["fixed1"][sel]
    im_np = np.ones(nrec * 6, dtype=np.float32)
    im_np[(t0_ids[fixed_t0 == 1]).reshape(-1)] = 0.0
    im_np[(t1_ids[fixed_t1 == 1]).reshape(-1)] = 0.0
    mult = float(g["mult"][sel][0])
    assert np.allclose(g["mult"][sel], mult), "mixed multipliers in golden set"

    x_out = _run_respond(
        x_np.reshape(-1, 3), prev_np.reshape(-1, 3), im_np,
        cur_ids, closest_ids,
        g["C"][sel] if "C" in g else np.minimum(g["C1"][sel], g["C2"][sel]),
        g["n"][sel], g["bary_cur"][sel], g["bary_closest"][sel],
        mult, g["mu_s"][sel], g["mu_d"][sel],
    ).reshape(nrec, 6, 3)

    post = np.stack([g["post0"][sel], g["post1"][sel]], axis=1).reshape(nrec, 6, 3)
    err = np.abs(x_out - post).max(axis=(1, 2))
    bad = np.flatnonzero(err > ATOL)
    rate = 1.0 - len(bad) / nrec
    assert rate >= PASS_RATE, (
        f"respond parity {rate:.4%}: {len(bad)}/{nrec} records exceed "
        f"{ATOL} (worst {err.max():.2e} at record {int(err.argmax())})")


def test_respond_solid_side_full_push():
    """One side fixed -> movable face takes FULL C*mult along n; fixed face
    does not move (reference :2966-2967 skips the 0.5 split, :3009/:3018
    skip the solid side; friction on fixed verts is nulled by re-pinning,
    which our inv_mass guard reproduces)."""
    n = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    C = np.array([0.01], dtype=np.float32)
    mult = 1.01
    # cur = movable tri at y=0; closest = fixed tri slightly above
    x_np = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],   # cur (movable)
        [0.0, 0.005, 0.0], [1.0, 0.005, 0.0], [0.0, 0.005, 1.0],  # closest (fixed)
    ], dtype=np.float32)
    prev_np = x_np.copy()                      # zero velocity -> zero friction
    im_np = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
    cur_ids = np.array([[0, 1, 2]])
    closest_ids = np.array([[3, 4, 5]])
    b = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float32)
    mu = np.array([0.5], dtype=np.float32)

    x_out = _run_respond(x_np, prev_np, im_np, cur_ids, closest_ids,
                         C, n, b, b, mult, mu, mu * 0.8)
    push = C[0] * mult
    assert np.allclose(x_out[0:3, 1], push, atol=1e-6), x_out[0:3]
    assert np.allclose(x_out[3:6], x_np[3:6], atol=1e-7), "fixed face moved"


def test_respond_both_movable_half_split():
    """Both movable -> C*mult*0.5 each, opposite directions."""
    n = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    C = np.array([0.01], dtype=np.float32)
    mult = 1.01
    x_np = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
        [0.0, 0.005, 0.0], [1.0, 0.005, 0.0], [0.0, 0.005, 1.0],
    ], dtype=np.float32)
    prev_np = x_np.copy()
    im_np = np.ones(6, dtype=np.float32)
    b = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float32)
    mu = np.array([0.5], dtype=np.float32)

    x_out = _run_respond(x_np, prev_np, im_np, np.array([[0, 1, 2]]),
                         np.array([[3, 4, 5]]), C, n, b, b, mult, mu, mu * 0.8)
    half = C[0] * mult * 0.5
    assert np.allclose(x_out[0:3, 1] - x_np[0:3, 1], +half, atol=1e-6)
    assert np.allclose(x_out[3:6, 1] - x_np[3:6, 1], -half, atol=1e-6)


def test_emit_pairs_groups():
    """Group rules: cross-group always; movable-self only when allowed;
    static-static never; shared-vert self pairs filtered."""
    from muscle.collision.udcf_dcd import emit_pairs_groups_kernel
    from muscle.collision.udcf_warp import centroid_radius_kernel

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    # 4 tris stacked at the origin: 0,1 movable (group 0; tri 1 shares vert 1
    # with tri 0), 2,3 static (group 1)
    x_np = np.array([
        [0, 0, 0], [1, 0, 0], [0, 0, 1],     # tri0 (movable)
        [0.1, 0.01, 0.1], [1, 0.01, 1],      # tri1 = [1, 3, 4] shares vert 1
        [0, 0.02, 0], [1, 0.02, 0], [0, 0.02, 1],   # tri2 (static)
        [0, 0.03, 0], [1, 0.03, 0], [0, 0.03, 1],   # tri3 (static)
    ], dtype=np.float32)
    tris_np = np.array([[0, 1, 2], [1, 3, 4], [5, 6, 7], [8, 9, 10]],
                       dtype=np.int32)
    group_np = np.array([0, 0, 1, 1], dtype=np.int32)

    x = wp.array(x_np, dtype=wp.vec3, device=dev)
    tris = wp.array2d(tris_np, dtype=wp.int32, device=dev)
    group = wp.array(group_np, dtype=wp.int32, device=dev)
    cent = wp.zeros(4, dtype=wp.vec3, device=dev)
    radii = wp.zeros(4, dtype=wp.float32, device=dev)
    wp.launch(centroid_radius_kernel, dim=4, inputs=[tris, x, cent, radii],
              device=dev)
    grid = wp.HashGrid(8, 8, 8, device=dev)
    grid.build(cent, 4.0)

    def emit(allow_self):
        pa = wp.zeros(64, dtype=wp.int32, device=dev)
        pb = wp.zeros(64, dtype=wp.int32, device=dev)
        n = wp.zeros(1, dtype=wp.int32, device=dev)
        wp.launch(emit_pairs_groups_kernel, dim=4,
                  inputs=[grid.id, cent, radii, tris, group,
                          wp.int32(allow_self), wp.float32(4.0),
                          wp.float32(0.1), wp.int32(64), pa, pb, n],
                  device=dev)
        wp.synchronize()
        nn = int(n.numpy()[0])
        return set(map(tuple, np.stack([pa.numpy()[:nn], pb.numpy()[:nn]], 1)))

    pairs = emit(allow_self=0)
    # cross pairs only: (0,2),(0,3),(1,2),(1,3); never (2,3); no self
    assert pairs == {(0, 2), (0, 3), (1, 2), (1, 3)}, pairs
    pairs = emit(allow_self=1)
    # self allowed, but (0,1) shares vert 1 → adjacency-filtered
    assert pairs == {(0, 2), (0, 3), (1, 2), (1, 3)}, pairs


def _run_collider_resolves_penetration(**param_overrides):
    """A movable triangle pushed into a static one separates over a few
    solve() iterations along the static face's normal."""
    from muscle.collision.udcf_dcd import UdcfDcdCollider, UdcfParams

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    # movable tri tilted through the static one (guarantees both C1>0,C2>0)
    mov_np = np.array([[-0.5, -0.02, -0.3], [0.5, 0.04, -0.3], [0.0, -0.02, 0.6]],
                      dtype=np.float32)
    static_v = np.array([[-1, 0, -1], [1, 0, -1], [0, 0, 1.5]], dtype=np.float32)
    static_t = np.array([[0, 1, 2]], dtype=np.int32)

    pos = wp.array(mov_np, dtype=wp.vec3, device=dev)
    inv_mass = wp.array(np.ones(3, dtype=np.float32), dtype=wp.float32,
                        device=dev)
    coll = UdcfDcdCollider(pos, inv_mass,
                           surface_tris=np.array([[0, 1, 2]], dtype=np.int32),
                           static_verts=static_v, static_tris=static_t,
                           params=UdcfParams(max_pairs=64, max_contacts=64,
                                             **param_overrides),
                           device=dev)
    depth0 = -float(pos.numpy()[:, 1].min())
    for it in range(8):
        coll.solve(it if it == 0 else 1)   # broadphase once, then iterate
    wp.synchronize()
    stats = coll.stats()
    assert stats["n_pairs"] >= 1
    assert stats["n_contacts"] >= 1
    depth1 = -float(pos.numpy()[:, 1].min())
    assert depth1 < depth0 * 0.5, (depth0, depth1)


def test_collider_resolves_penetration():
    _run_collider_resolves_penetration()


def test_collider_resolves_penetration_gauss_seidel():
    """Same scenario through the fused serial GS narrow+respond path."""
    _run_collider_resolves_penetration(gauss_seidel=True)


def test_collider_resolves_penetration_max_push():
    """Depth clamp still separates (over more iterations, bounded steps)."""
    _run_collider_resolves_penetration(max_push=0.005)


def test_respond_solid_side_friction_slipping():
    """Fixed-side contact with tangential motion → friction clamps to kinetic.

    One contact: cur = movable tri (inv_mass 1), closest = fixed tri (inv_mass 0).
    Normal n=(0,1,0), C=0.01, mult=1.01, barys b1=b2=(1/3,1/3,1/3),
    mu_s=0.5, mu_d=0.4.  Movable verts have uniform tangential velocity vx=0.02
    (prev = x - [vx, 0, 0]).  Fixed verts are stationary (prev = x).

    Derivation from respond_kernel (line-by-line):
      pen  = 0.01
      C    = pen * mult = 0.0101         (no 0.5 split: solid_clo=1)
      push_cur  = C * n = (0, 0.0101, 0)
      push_clo  = 0                      (solid_clo → skipped)
      v_cur per vert = (x + push_cur) - prev = (vx, C, 0) = (0.02, 0.0101, 0)
      v_clo per vert = 0
      g    = (0,1,0)
      vcur = b2-weighted avg = (0.02, 0.0101, 0)
      vclo = 0
      perp_cur = vcur - g*dot(vcur,g) = (0.02, 0, 0)
      perp_clo = 0
      diff = perp_cur - perp_clo = (0.02, 0, 0)
      dnorm = 0.02;  static_norm = pen*mu_s = 0.005  → slipping (0.02 > 0.005)
      diff *= pen*mu_d/dnorm = 0.2  →  diff = (0.004, 0, 0)
      diffV = diff*0.5 = (0.002, 0, 0)
      movable push = push_cur - diffV = (-0.002, 0.0101, 0)
      fixed  push = 0  (inv_mass == 0 guard)
    """
    n_arr = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    C = np.array([0.01], dtype=np.float32)
    mult = 1.01
    vx = 0.02
    Cm = float(C[0]) * mult   # = 0.0101

    x_np = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],   # cur (movable)
        [0.0, 0.005, 0.0], [1.0, 0.005, 0.0], [0.0, 0.005, 1.0],  # closest (fixed)
    ], dtype=np.float32)
    # movable verts have tangential velocity vx in x; fixed verts are stationary
    prev_np = x_np.copy()
    prev_np[0:3, 0] -= vx          # prev = x - [vx, 0, 0]  → v = [vx, 0, 0]

    im_np = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
    cur_ids = np.array([[0, 1, 2]])
    closest_ids = np.array([[3, 4, 5]])
    b = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float32)
    mu_s = np.array([0.5], dtype=np.float32)
    mu_d = np.array([0.4], dtype=np.float32)

    x_out = _run_respond(x_np, prev_np, im_np, cur_ids, closest_ids,
                         C, n_arr, b, b, mult, mu_s, mu_d)

    # movable verts: pushed by (-0.002, Cm, 0) relative to x
    expected_mov_delta = np.array([-0.002, Cm, 0.0], dtype=np.float32)
    assert np.allclose(x_out[0:3] - x_np[0:3], expected_mov_delta, atol=1e-6), \
        f"movable delta wrong: got {x_out[0:3] - x_np[0:3]}, want {expected_mov_delta}"
    # fixed verts: unmoved
    assert np.allclose(x_out[3:6], x_np[3:6], atol=1e-7), \
        f"fixed face moved: {x_out[3:6] - x_np[3:6]}"


import json
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
_LION = _ROOT / "data" / "lion" / "model"
_LION_CFG = _ROOT / "data" / "lion" / "config" / "lion_nomg.json"

skip_no_lion = pytest.mark.skipif(not (_LION / "restpos.geo").exists(),
                                  reason="lion .geo blobs not present")


@skip_no_lion
def test_solver_iter_hook_called_per_fine_iteration():
    from muscle.coupleOneWayGmg.solver import OneWaySim

    cfg = json.load(open(_LION_CFG))
    cfg["geo_dir"] = str(_LION)
    sim = OneWaySim(cfg)

    calls = []
    sim.collision_solvers_iter = [lambda it: calls.append(it)]
    end_calls = []
    sim.collision_solvers = [lambda: end_calls.append(1)]

    sim.step_frame(sim.start_frame)
    expected = sim.fine_iters * sim.nsubsteps
    assert len(calls) == expected, (len(calls), expected)
    # each substep counts 0..fine_iters-1
    assert calls[:sim.fine_iters] == list(range(sim.fine_iters))
    assert len(end_calls) == sim.nsubsteps   # old hook untouched
