"""Triangle-triangle collision in Warp: broadphase + UDCF narrow + XPBD apply.

Exposed as flat kernels — the host owns every buffer and ``wp.launch``-es each
stage directly, no wrapper classes. The pipeline is, in order:

  1. ``centroid_radius_kernel``       per-tri bounding sphere
  2. ``max_radius_reduce_kernel``     (optional) auto cell size
  3. ``wp.HashGrid.build``            host call
  4. ``emit_pairs_kernel``            candidate pair list, with adjacency
                                       + per-tri-group self-collision filter
  5. ``collide_and_solve_kernel``     fused UDCF narrow phase (Kikuchi &
                                       Kanai EG2025) + XPBD position
                                       correction. C / gradC / bary / dom
                                       all live as kernel locals and are
                                       consumed in-place — no per-pair
                                       output buffer.

UDCF's paper-canonical output per pair is ``(C, gradC)`` (see upstream
C++ ``udcf_algorithm``); ``bary``/``dominant`` are the XPBD extensions we
need for distributing the correction across the dominant triangle's 3
verts, but they never escape the kernel.

Coordinate / sign conventions used inside ``udcf_narrow``::

    C        : penetration depth (paper Eq.6 f-value), 0 if disjoint
    gradC    : vec3, normal of the OPPOSING triangle (paper's -∇f)
    bary     : vec3, barycentric on dominant triangle (sums to 1)
    dominant : 0 = T1 corrected, 1 = T2 corrected, -1 = no collision

All arithmetic is fp32.
"""
from __future__ import annotations

import warp as wp


EPS_TOUCH = 1.0e-7  # paper threshold for ignoring shallow contacts


# ======================================================================
# Broadphase: per-tri bounding sphere, auto cell size, candidate pairs
# ======================================================================

@wp.kernel
def centroid_radius_kernel(
    tris: wp.array2d(dtype=wp.int32),  # (M, 3)
    verts: wp.array(dtype=wp.vec3),    # (N,)
    centroids: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    v0 = verts[tris[i, 0]]
    v1 = verts[tris[i, 1]]
    v2 = verts[tris[i, 2]]
    c = (v0 + v1 + v2) / 3.0
    r0 = wp.length(v0 - c)
    r1 = wp.length(v1 - c)
    r2 = wp.length(v2 - c)
    centroids[i] = c
    radii[i] = wp.max(wp.max(r0, r1), r2)


@wp.kernel
def max_radius_reduce_kernel(
    radii: wp.array(dtype=wp.float32),
    out_max: wp.array(dtype=wp.float32),
):
    """Single-thread max-reduction. Fine for M up to ~10^5; replace with a
    parallel reduction if it shows up in profiles.
    """
    n = radii.shape[0]
    m = float(0.0)
    for i in range(n):
        if radii[i] > m:
            m = radii[i]
    out_max[0] = m


@wp.kernel
def emit_pairs_kernel(
    grid: wp.uint64,
    centroids: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32),
    tris: wp.array2d(dtype=wp.int32),
    tri_group: wp.array(dtype=wp.int32),
    allow_self_group0: wp.int32,
    query_radius: wp.float32,
    thickness: wp.float32,
    max_pairs: wp.int32,
    pair_a: wp.array(dtype=wp.int32),
    pair_b: wp.array(dtype=wp.int32),
    n_pairs: wp.array(dtype=wp.int32),
):
    """Emit unique candidate pairs ``(i, j)`` with ``i < j`` after:

    1. Sphere-overlap test ``|c_a - c_b| <= r_a + r_b + thickness``
    2. Shared-vertex adjacency filter (skip pairs with any shared
       global vertex index — works across meshes when index spaces
       are globally concatenated, since per-mesh ranges are disjoint)
    3. Self-collision filter — when ``allow_self_group0 == 0``, skip
       any pair whose two tris both carry ``tri_group == 0``

    ``n_pairs`` is the device atomic counter; saturates at ``max_pairs``.
    """
    i = wp.tid()
    ci = centroids[i]
    ri = radii[i]
    ai0 = tris[i, 0]
    ai1 = tris[i, 1]
    ai2 = tris[i, 2]
    gi = tri_group[i]

    q = wp.hash_grid_query(grid, ci, query_radius)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if j <= i:
            continue
        cj = centroids[j]
        rj = radii[j]
        if wp.length(ci - cj) > ri + rj + thickness:
            continue
        bj0 = tris[j, 0]
        bj1 = tris[j, 1]
        bj2 = tris[j, 2]
        if (ai0 == bj0 or ai0 == bj1 or ai0 == bj2 or
            ai1 == bj0 or ai1 == bj1 or ai1 == bj2 or
            ai2 == bj0 or ai2 == bj1 or ai2 == bj2):
            continue
        gj = tri_group[j]
        if gi == 0 and gj == 0 and allow_self_group0 == 0:
            continue

        slot = wp.atomic_add(n_pairs, 0, 1)
        if slot < max_pairs:
            pair_a[slot] = i
            pair_b[slot] = j


# ======================================================================
# UDCF narrow phase
# ======================================================================

@wp.struct
class UdcfPairResult:
    intersecting: int
    C: float
    gradC: wp.vec3
    bary: wp.vec3
    dominant: int


@wp.func
def _inside_tri(t0: wp.vec3, t1: wp.vec3, t2: wp.vec3, p: wp.vec3) -> int:
    ab = t1 - t0
    bp = p - t1
    bc = t2 - t1
    cp = p - t2
    ca = t0 - t2
    ap = p - t0
    c1 = wp.cross(ab, bp)
    c2 = wp.cross(bc, cp)
    c3 = wp.cross(ca, ap)
    if wp.dot(c1, c2) > 0.0 and wp.dot(c1, c3) > 0.0:
        return 1
    return 0


@wp.func
def _bary_tri(t0: wp.vec3, t1: wp.vec3, t2: wp.vec3, p: wp.vec3) -> wp.vec3:
    n_full = wp.cross(t1 - t0, t2 - t0)
    nrm2 = wp.dot(n_full, n_full)
    if nrm2 == 0.0:
        return wp.vec3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    nrm = wp.sqrt(nrm2)
    n_hat = n_full / nrm
    sub0 = wp.cross(t1 - p, t2 - p)
    sub1 = wp.cross(t2 - p, t0 - p)
    sub2 = wp.cross(t0 - p, t1 - p)
    w0 = wp.dot(sub0, n_hat) / nrm
    w1 = wp.dot(sub1, n_hat) / nrm
    w2 = wp.dot(sub2, n_hat) / nrm
    return wp.vec3(w0, w1, w2)


@wp.func
def _line_isect_2d(p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, p4: wp.vec3) -> float:
    """Co-planar intersection parameter ``tp`` along (p1->p2), or -1.0 if
    invalid. Returns tp in [0,1] iff both edges' parameters are in [0,1]."""
    d1 = p2 - p1
    d2 = p4 - p3
    n = wp.cross(d1, d2)
    nrm = wp.length(n)
    if nrm == 0.0:
        return -1.0
    denom = wp.dot(n, n)
    if denom == 0.0:
        return -1.0
    v = p3 - p1
    t1p = wp.dot(wp.cross(v, d2), n) / denom
    t2p = wp.dot(wp.cross(v, d1), n) / denom
    if t2p >= 0.0 and t2p <= 1.0 and t1p >= 0.0 and t1p <= 1.0:
        return t1p
    return -1.0


@wp.func
def _find_isect_pt(N: wp.vec3, p0: wp.vec3, p: wp.vec3, line_dir: wp.vec3):
    """Intersect ray (p, +line_dir) with plane (N, p0). Returns
    (valid: int, pt: vec3); valid=0 if parallel."""
    denom = wp.dot(N, line_dir)
    if denom == 0.0:
        return 0, wp.vec3(0.0, 0.0, 0.0)
    t = wp.dot(N, p0 - p) / denom
    return 1, p + t * line_dir


@wp.func
def _gen_t(D: wp.vec3,
           v0: wp.vec3, v1: wp.vec3, v2: wp.vec3,
           d0: float, d1: float, d2: float):
    """Möller's projection interval [t_lo, t_hi] of a triangle along D.
    Returns (valid: int, t_lo, t_hi); valid=0 if all verts same side."""
    if d0 <= 0.0 and d1 <= 0.0 and d2 <= 0.0:
        return 0, 0.0, 0.0
    if d0 >= 0.0 and d1 >= 0.0 and d2 >= 0.0:
        return 0, 0.0, 0.0

    # Reorder so v1 is the lone-side vertex (opposite-sign distance).
    sv0 = v0
    sv1 = v1
    sv2 = v2
    sd0 = d0
    sd1 = d1
    sd2 = d2
    if (d0 <= 0.0 and d1 >= 0.0 and d2 >= 0.0) or (d0 >= 0.0 and d1 <= 0.0 and d2 <= 0.0):
        sv0 = v1
        sv1 = v0
        sv2 = v2
        sd0 = d1
        sd1 = d0
        sd2 = d2
    elif (d0 >= 0.0 and d1 <= 0.0 and d2 >= 0.0) or (d0 <= 0.0 and d1 >= 0.0 and d2 <= 0.0):
        sv0 = v0
        sv1 = v1
        sv2 = v2
        sd0 = d0
        sd1 = d1
        sd2 = d2
    elif (d0 >= 0.0 and d1 >= 0.0 and d2 <= 0.0) or (d0 <= 0.0 and d1 <= 0.0 and d2 >= 0.0):
        sv0 = v0
        sv1 = v2
        sv2 = v1
        sd0 = d0
        sd1 = d2
        sd2 = d1

    p0 = wp.dot(D, sv0)
    p1 = wp.dot(D, sv1)
    p2 = wp.dot(D, sv2)
    tA = p0 + (p1 - p0) * wp.abs(sd0 / (sd0 - sd1))
    tB = p2 + (p1 - p2) * wp.abs(sd2 / (sd2 - sd1))
    if tA > tB:
        return 1, tB, tA
    return 1, tA, tB


@wp.func
def udcf_narrow(
    a0: wp.vec3, a1: wp.vec3, a2: wp.vec3,
    b0: wp.vec3, b1: wp.vec3, b2: wp.vec3,
) -> UdcfPairResult:
    """Per-pair UDCF narrow-phase test (T1=a, T2=b)."""
    res = UdcfPairResult()
    res.intersecting = 0
    res.C = 0.0
    res.gradC = wp.vec3(0.0, 0.0, 0.0)
    res.bary = wp.vec3(0.0, 0.0, 0.0)
    res.dominant = -1

    N1raw = wp.cross(a1 - a0, a2 - a0)
    n1l = wp.length(N1raw)
    if n1l == 0.0:
        return res
    N1 = N1raw / n1l

    N2raw = wp.cross(b1 - b0, b2 - b0)
    n2l = wp.length(N2raw)
    if n2l == 0.0:
        return res
    N2 = N2raw / n2l

    d1c = -wp.dot(N1, a0)
    d2c = -wp.dot(N2, b0)

    da0 = wp.dot(N2, a0) + d2c
    da1 = wp.dot(N2, a1) + d2c
    da2 = wp.dot(N2, a2) + d2c

    db0 = wp.dot(N1, b0) + d1c
    db1 = wp.dot(N1, b1) + d1c
    db2 = wp.dot(N1, b2) + d1c

    Draw = wp.cross(N1, N2)
    Dl = wp.length(Draw)
    if Dl == 0.0:
        return res
    D = Draw / Dl

    iv1_ok, iv1_lo, iv1_hi = _gen_t(D, a0, a1, a2, da0, da1, da2)
    if iv1_ok == 0:
        return res
    iv2_ok, iv2_lo, iv2_hi = _gen_t(D, b0, b1, b2, db0, db1, db2)
    if iv2_ok == 0:
        return res
    if not (iv1_hi >= iv2_lo and iv2_hi >= iv1_lo):
        return res

    P0_0 = a0 - wp.dot(N2, a0 - b0) * N2
    P0_1 = a1 - wp.dot(N2, a1 - b0) * N2
    P0_2 = a2 - wp.dot(N2, a2 - b0) * N2

    P1_0 = b0 - wp.dot(N1, b0 - a0) * N1
    P1_1 = b1 - wp.dot(N1, b1 - a0) * N1
    P1_2 = b2 - wp.dot(N1, b2 - a0) * N1

    Q0_0_ok, Q0_0 = _find_isect_pt(N1, a0, b0, N2)
    Q0_1_ok, Q0_1 = _find_isect_pt(N1, a0, b1, N2)
    Q0_2_ok, Q0_2 = _find_isect_pt(N1, a0, b2, N2)

    Q1_0_ok, Q1_0 = _find_isect_pt(N2, b0, a0, N1)
    Q1_1_ok, Q1_1 = _find_isect_pt(N2, b0, a1, N1)
    Q1_2_ok, Q1_2 = _find_isect_pt(N2, b0, a2, N1)

    best_d_t1 = float(0.0)
    best_b_t1 = wp.vec3(0.0, 0.0, 0.0)
    best_d_t2 = float(0.0)
    best_b_t2 = wp.vec3(0.0, 0.0, 0.0)

    # ===== T1 candidate set =====
    if _inside_tri(b0, b1, b2, P0_0) == 1:
        depth = -da0
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = wp.vec3(1.0, 0.0, 0.0)
    if _inside_tri(b0, b1, b2, P0_1) == 1:
        depth = -da1
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = wp.vec3(0.0, 1.0, 0.0)
    if _inside_tri(b0, b1, b2, P0_2) == 1:
        depth = -da2
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = wp.vec3(0.0, 0.0, 1.0)

    if Q0_0_ok == 1 and _inside_tri(a0, a1, a2, Q0_0) == 1:
        depth = -(wp.dot(N2, Q0_0) + d2c)
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = _bary_tri(a0, a1, a2, Q0_0)
    if Q0_1_ok == 1 and _inside_tri(a0, a1, a2, Q0_1) == 1:
        depth = -(wp.dot(N2, Q0_1) + d2c)
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = _bary_tri(a0, a1, a2, Q0_1)
    if Q0_2_ok == 1 and _inside_tri(a0, a1, a2, Q0_2) == 1:
        depth = -(wp.dot(N2, Q0_2) + d2c)
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = _bary_tri(a0, a1, a2, Q0_2)

    # Edge-edge for T1: 3 P-edges × 3 T2 edges, unrolled.
    tp = _line_isect_2d(P0_0, P0_1, b0, b1)
    if tp >= 0.0:
        lifted = a0 + tp * (a1 - a0)
        depth = -(wp.dot(N2, lifted) + d2c)
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = wp.vec3(1.0 - tp, tp, 0.0)
    tp = _line_isect_2d(P0_0, P0_1, b1, b2)
    if tp >= 0.0:
        lifted = a0 + tp * (a1 - a0)
        depth = -(wp.dot(N2, lifted) + d2c)
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = wp.vec3(1.0 - tp, tp, 0.0)
    tp = _line_isect_2d(P0_0, P0_1, b2, b0)
    if tp >= 0.0:
        lifted = a0 + tp * (a1 - a0)
        depth = -(wp.dot(N2, lifted) + d2c)
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = wp.vec3(1.0 - tp, tp, 0.0)
    tp = _line_isect_2d(P0_1, P0_2, b0, b1)
    if tp >= 0.0:
        lifted = a1 + tp * (a2 - a1)
        depth = -(wp.dot(N2, lifted) + d2c)
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = wp.vec3(0.0, 1.0 - tp, tp)
    tp = _line_isect_2d(P0_1, P0_2, b1, b2)
    if tp >= 0.0:
        lifted = a1 + tp * (a2 - a1)
        depth = -(wp.dot(N2, lifted) + d2c)
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = wp.vec3(0.0, 1.0 - tp, tp)
    tp = _line_isect_2d(P0_1, P0_2, b2, b0)
    if tp >= 0.0:
        lifted = a1 + tp * (a2 - a1)
        depth = -(wp.dot(N2, lifted) + d2c)
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = wp.vec3(0.0, 1.0 - tp, tp)
    tp = _line_isect_2d(P0_2, P0_0, b0, b1)
    if tp >= 0.0:
        lifted = a2 + tp * (a0 - a2)
        depth = -(wp.dot(N2, lifted) + d2c)
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = wp.vec3(tp, 0.0, 1.0 - tp)
    tp = _line_isect_2d(P0_2, P0_0, b1, b2)
    if tp >= 0.0:
        lifted = a2 + tp * (a0 - a2)
        depth = -(wp.dot(N2, lifted) + d2c)
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = wp.vec3(tp, 0.0, 1.0 - tp)
    tp = _line_isect_2d(P0_2, P0_0, b2, b0)
    if tp >= 0.0:
        lifted = a2 + tp * (a0 - a2)
        depth = -(wp.dot(N2, lifted) + d2c)
        if depth > best_d_t1:
            best_d_t1 = depth
            best_b_t1 = wp.vec3(tp, 0.0, 1.0 - tp)

    # ===== T2 candidate set =====
    if _inside_tri(a0, a1, a2, P1_0) == 1:
        depth = -db0
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = wp.vec3(1.0, 0.0, 0.0)
    if _inside_tri(a0, a1, a2, P1_1) == 1:
        depth = -db1
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = wp.vec3(0.0, 1.0, 0.0)
    if _inside_tri(a0, a1, a2, P1_2) == 1:
        depth = -db2
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = wp.vec3(0.0, 0.0, 1.0)

    if Q1_0_ok == 1 and _inside_tri(b0, b1, b2, Q1_0) == 1:
        depth = -(wp.dot(N1, Q1_0) + d1c)
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = _bary_tri(b0, b1, b2, Q1_0)
    if Q1_1_ok == 1 and _inside_tri(b0, b1, b2, Q1_1) == 1:
        depth = -(wp.dot(N1, Q1_1) + d1c)
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = _bary_tri(b0, b1, b2, Q1_1)
    if Q1_2_ok == 1 and _inside_tri(b0, b1, b2, Q1_2) == 1:
        depth = -(wp.dot(N1, Q1_2) + d1c)
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = _bary_tri(b0, b1, b2, Q1_2)

    tp = _line_isect_2d(P1_0, P1_1, a0, a1)
    if tp >= 0.0:
        lifted = b0 + tp * (b1 - b0)
        depth = -(wp.dot(N1, lifted) + d1c)
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = wp.vec3(1.0 - tp, tp, 0.0)
    tp = _line_isect_2d(P1_0, P1_1, a1, a2)
    if tp >= 0.0:
        lifted = b0 + tp * (b1 - b0)
        depth = -(wp.dot(N1, lifted) + d1c)
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = wp.vec3(1.0 - tp, tp, 0.0)
    tp = _line_isect_2d(P1_0, P1_1, a2, a0)
    if tp >= 0.0:
        lifted = b0 + tp * (b1 - b0)
        depth = -(wp.dot(N1, lifted) + d1c)
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = wp.vec3(1.0 - tp, tp, 0.0)
    tp = _line_isect_2d(P1_1, P1_2, a0, a1)
    if tp >= 0.0:
        lifted = b1 + tp * (b2 - b1)
        depth = -(wp.dot(N1, lifted) + d1c)
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = wp.vec3(0.0, 1.0 - tp, tp)
    tp = _line_isect_2d(P1_1, P1_2, a1, a2)
    if tp >= 0.0:
        lifted = b1 + tp * (b2 - b1)
        depth = -(wp.dot(N1, lifted) + d1c)
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = wp.vec3(0.0, 1.0 - tp, tp)
    tp = _line_isect_2d(P1_1, P1_2, a2, a0)
    if tp >= 0.0:
        lifted = b1 + tp * (b2 - b1)
        depth = -(wp.dot(N1, lifted) + d1c)
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = wp.vec3(0.0, 1.0 - tp, tp)
    tp = _line_isect_2d(P1_2, P1_0, a0, a1)
    if tp >= 0.0:
        lifted = b2 + tp * (b0 - b2)
        depth = -(wp.dot(N1, lifted) + d1c)
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = wp.vec3(tp, 0.0, 1.0 - tp)
    tp = _line_isect_2d(P1_2, P1_0, a1, a2)
    if tp >= 0.0:
        lifted = b2 + tp * (b0 - b2)
        depth = -(wp.dot(N1, lifted) + d1c)
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = wp.vec3(tp, 0.0, 1.0 - tp)
    tp = _line_isect_2d(P1_2, P1_0, a2, a0)
    if tp >= 0.0:
        lifted = b2 + tp * (b0 - b2)
        depth = -(wp.dot(N1, lifted) + d1c)
        if depth > best_d_t2:
            best_d_t2 = depth
            best_b_t2 = wp.vec3(tp, 0.0, 1.0 - tp)

    # Pick dominant (smaller positive depth). Both depths init to 0 and are
    # only raised by candidate hits — so depth==0 means "no candidate fired on
    # this side". The paper's strict `t1<t2 AND t2<t1` formulation silently
    # drops the asymmetric-depth case (one side > 0, other side == 0), which
    # is the common deep-penetration scenario in self-collision. Fix: also
    # fire the side that found a positive depth even when the other side
    # didn't.
    v_t1 = best_d_t1 > EPS_TOUCH
    v_t2 = best_d_t2 > EPS_TOUCH
    pick_t1 = v_t1 and ((not v_t2) or best_d_t1 <= best_d_t2)
    pick_t2 = v_t2 and (not pick_t1)
    if pick_t1:
        res.intersecting = 1
        res.C = best_d_t1
        res.gradC = N2
        res.bary = best_b_t1
        res.dominant = 0
        return res
    if pick_t2:
        res.intersecting = 1
        res.C = best_d_t2
        res.gradC = N1
        res.bary = best_b_t2
        res.dominant = 1
        return res
    return res


# ======================================================================
# Fused narrow phase + XPBD position correction
# ======================================================================

@wp.kernel
def collide_and_solve_kernel(
    pair_a: wp.array(dtype=wp.int32),
    pair_b: wp.array(dtype=wp.int32),
    n_pairs: wp.array(dtype=wp.int32),
    tris: wp.array2d(dtype=wp.int32),  # (M, 3) global vertex indices
    verts: wp.array(dtype=wp.vec3),    # (N,) world positions
    inv_mass: wp.array(dtype=wp.float32),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    contact_normal: wp.array(dtype=wp.vec3),
    is_collide: wp.array(dtype=wp.int32),
    n_collide: wp.array(dtype=wp.int32),   # 1-elem diagnostic
    max_C: wp.array(dtype=wp.float32),     # 1-elem diagnostic
):
    """Fused UDCF narrow phase + XPBD position correction.

    Per candidate pair from the broadphase:
      1. ``udcf_narrow`` → ``(C, gradC, bary, dominant)`` as kernel locals
         (paper output convention; bary/dominant never crosses a buffer
         boundary).
      2. If intersecting, distribute the rigid-contact correction across
         the dominant triangle's 3 verts:
            Δp_i = w_i · bary_i · gradC · (C / Σ_j w_j bary_j²)
         and accumulate ``gradC`` (= outward normal of opposing surface)
         into ``contact_normal`` for the host-side post-velocity
         projection.

    Diagnostics: ``n_collide`` counts colliding pairs, ``max_C`` tracks
    the worst penetration depth. Both 1-elem atomics — host zeros them
    before launch.
    """
    i = wp.tid()
    if i >= n_pairs[0]:
        return
    a = pair_a[i]
    b = pair_b[i]
    a0 = verts[tris[a, 0]]
    a1 = verts[tris[a, 1]]
    a2 = verts[tris[a, 2]]
    b0 = verts[tris[b, 0]]
    b1 = verts[tris[b, 1]]
    b2 = verts[tris[b, 2]]

    r = udcf_narrow(a0, a1, a2, b0, b1, b2)
    if r.intersecting == 0:
        return

    if r.dominant == 0:
        v0 = tris[a, 0]
        v1 = tris[a, 1]
        v2 = tris[a, 2]
        o0 = tris[b, 0]
        o1 = tris[b, 1]
        o2 = tris[b, 2]
    else:
        v0 = tris[b, 0]
        v1 = tris[b, 1]
        v2 = tris[b, 2]
        o0 = tris[a, 0]
        o1 = tris[a, 1]
        o2 = tris[a, 2]

    bary = r.bary
    gradC = r.gradC
    C = r.C

    w0 = inv_mass[v0]
    w1 = inv_mass[v1]
    w2 = inv_mass[v2]
    ow0 = inv_mass[o0]
    ow1 = inv_mass[o1]
    ow2 = inv_mass[o2]

    # 2-body XPBD: dom side uses paper's bary; opposing side gets a uniform
    # 1/3 split (we don't compute opposing bary). Plane (inv_mass=0) silently
    # drops out → muscle↔plane behaves exactly as the dom-only paper version.
    # muscle↔muscle gets symmetric position push, which the bary-scan diag
    # confirms is required ("陷进去出不来" otherwise).
    denom_dom = w0 * bary[0] * bary[0] + w1 * bary[1] * bary[1] + w2 * bary[2] * bary[2]
    denom_opp = (ow0 + ow1 + ow2) * (1.0 / 9.0)
    denom = denom_dom + denom_opp
    if denom <= 1.0e-12:
        return
    lam = C / denom

    dp0 = (w0 * bary[0] * lam) * gradC
    dp1 = (w1 * bary[1] * lam) * gradC
    dp2 = (w2 * bary[2] * lam) * gradC

    if w0 > 0.0:
        wp.atomic_add(dP, v0, dp0)
        wp.atomic_add(dPw, v0, 1.0)
        wp.atomic_add(contact_normal, v0, gradC)
        is_collide[v0] = 1
    if w1 > 0.0:
        wp.atomic_add(dP, v1, dp1)
        wp.atomic_add(dPw, v1, 1.0)
        wp.atomic_add(contact_normal, v1, gradC)
        is_collide[v1] = 1
    if w2 > 0.0:
        wp.atomic_add(dP, v2, dp2)
        wp.atomic_add(dPw, v2, 1.0)
        wp.atomic_add(contact_normal, v2, gradC)
        is_collide[v2] = 1

    neg_gradC = -gradC
    coef_opp = lam * (1.0 / 3.0)
    if ow0 > 0.0:
        wp.atomic_add(dP, o0, (ow0 * coef_opp) * neg_gradC)
        wp.atomic_add(dPw, o0, 1.0)
        wp.atomic_add(contact_normal, o0, neg_gradC)
        is_collide[o0] = 1
    if ow1 > 0.0:
        wp.atomic_add(dP, o1, (ow1 * coef_opp) * neg_gradC)
        wp.atomic_add(dPw, o1, 1.0)
        wp.atomic_add(contact_normal, o1, neg_gradC)
        is_collide[o1] = 1
    if ow2 > 0.0:
        wp.atomic_add(dP, o2, (ow2 * coef_opp) * neg_gradC)
        wp.atomic_add(dPw, o2, 1.0)
        wp.atomic_add(contact_normal, o2, neg_gradC)
        is_collide[o2] = 1

    wp.atomic_add(n_collide, 0, 1)
    wp.atomic_max(max_C, 0, C)


@wp.kernel
def collide_and_solve_gs_kernel(
    pair_a: wp.array(dtype=wp.int32),
    pair_b: wp.array(dtype=wp.int32),
    n_pairs: wp.array(dtype=wp.int32),
    tris: wp.array2d(dtype=wp.int32),
    verts: wp.array(dtype=wp.vec3),            # (n_total,) world (mirror)
    sim_pos: wp.array(dtype=wp.vec3),          # (n_mu,) host sim pos
    sim_pprev: wp.array(dtype=wp.vec3),        # (n_mu,) host sim pprev
    inv_mass: wp.array(dtype=wp.float32),
    n_mu: wp.int32,
    contact_normal: wp.array(dtype=wp.vec3),
    is_collide: wp.array(dtype=wp.int32),
    n_collide: wp.array(dtype=wp.int32),
    max_C: wp.array(dtype=wp.float32),
):
    """Gauss-Seidel narrow phase + position push.

    Launched with dim=1 — a single thread runs the per-pair loop
    sequentially. Each pair's push is written directly to ``verts``
    (so the next pair's narrow phase sees it) and to ``sim_pos`` +
    ``sim_pprev`` for muscle verts (so the host sim state stays in
    sync). No dP/dPw buffer; no Jacobi averaging — same vert in
    multiple contact pairs gets every push in full, in sequence.
    """
    tid = wp.tid()
    if tid != 0:
        return
    n = n_pairs[0]
    for i in range(n):
        a = pair_a[i]
        b = pair_b[i]
        a0 = verts[tris[a, 0]]
        a1 = verts[tris[a, 1]]
        a2 = verts[tris[a, 2]]
        b0 = verts[tris[b, 0]]
        b1 = verts[tris[b, 1]]
        b2 = verts[tris[b, 2]]
        r = udcf_narrow(a0, a1, a2, b0, b1, b2)
        if r.intersecting == 1:
            v0 = int(0)
            v1 = int(0)
            v2 = int(0)
            o0 = int(0)
            o1 = int(0)
            o2 = int(0)
            if r.dominant == 0:
                v0 = tris[a, 0]
                v1 = tris[a, 1]
                v2 = tris[a, 2]
                o0 = tris[b, 0]
                o1 = tris[b, 1]
                o2 = tris[b, 2]
            else:
                v0 = tris[b, 0]
                v1 = tris[b, 1]
                v2 = tris[b, 2]
                o0 = tris[a, 0]
                o1 = tris[a, 1]
                o2 = tris[a, 2]
            bary = r.bary
            gradC = r.gradC
            C = r.C
            w0 = inv_mass[v0]
            w1 = inv_mass[v1]
            w2 = inv_mass[v2]
            ow0 = inv_mass[o0]
            ow1 = inv_mass[o1]
            ow2 = inv_mass[o2]
            denom_dom = w0 * bary[0] * bary[0] + w1 * bary[1] * bary[1] + w2 * bary[2] * bary[2]
            denom_opp = (ow0 + ow1 + ow2) * (1.0 / 9.0)
            denom = denom_dom + denom_opp
            if denom > 1.0e-12:
                lam = C / denom
                dp0 = (w0 * bary[0] * lam) * gradC
                dp1 = (w1 * bary[1] * lam) * gradC
                dp2 = (w2 * bary[2] * lam) * gradC
                if w0 > 0.0:
                    verts[v0] = verts[v0] + dp0
                    if v0 < n_mu:
                        sim_pos[v0] = sim_pos[v0] + dp0
                        sim_pprev[v0] = sim_pprev[v0] + dp0
                        contact_normal[v0] = contact_normal[v0] + gradC
                        is_collide[v0] = 1
                if w1 > 0.0:
                    verts[v1] = verts[v1] + dp1
                    if v1 < n_mu:
                        sim_pos[v1] = sim_pos[v1] + dp1
                        sim_pprev[v1] = sim_pprev[v1] + dp1
                        contact_normal[v1] = contact_normal[v1] + gradC
                        is_collide[v1] = 1
                if w2 > 0.0:
                    verts[v2] = verts[v2] + dp2
                    if v2 < n_mu:
                        sim_pos[v2] = sim_pos[v2] + dp2
                        sim_pprev[v2] = sim_pprev[v2] + dp2
                        contact_normal[v2] = contact_normal[v2] + gradC
                        is_collide[v2] = 1
                neg_gradC = -gradC
                coef_opp = lam * (1.0 / 3.0)
                if ow0 > 0.0:
                    d0 = (ow0 * coef_opp) * neg_gradC
                    verts[o0] = verts[o0] + d0
                    if o0 < n_mu:
                        sim_pos[o0] = sim_pos[o0] + d0
                        sim_pprev[o0] = sim_pprev[o0] + d0
                        contact_normal[o0] = contact_normal[o0] + neg_gradC
                        is_collide[o0] = 1
                if ow1 > 0.0:
                    d1 = (ow1 * coef_opp) * neg_gradC
                    verts[o1] = verts[o1] + d1
                    if o1 < n_mu:
                        sim_pos[o1] = sim_pos[o1] + d1
                        sim_pprev[o1] = sim_pprev[o1] + d1
                        contact_normal[o1] = contact_normal[o1] + neg_gradC
                        is_collide[o1] = 1
                if ow2 > 0.0:
                    d2 = (ow2 * coef_opp) * neg_gradC
                    verts[o2] = verts[o2] + d2
                    if o2 < n_mu:
                        sim_pos[o2] = sim_pos[o2] + d2
                        sim_pprev[o2] = sim_pprev[o2] + d2
                        contact_normal[o2] = contact_normal[o2] + neg_gradC
                        is_collide[o2] = 1
                n_collide[0] = n_collide[0] + 1
                if C > max_C[0]:
                    max_C[0] = C
