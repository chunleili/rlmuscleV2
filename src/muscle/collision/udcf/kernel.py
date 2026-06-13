"""Faithful Warp port of Gaia_UDCF's triangle-triangle DCD.

Reference: ~/Dev/Gaia_UDCF/Simulator/Modules/PBD/PBDPhysics.cpp
  narrowPhase()                          :1264-1762
  applyNarrowCollisionConstraintsCPU()   :2894-3116
  runStepCPU() loop structure            :622-719

Pipeline (host orchestrated by UdcfDcdCollider, one call per solver
fine-iteration):

  broadphase (hashgrid, per substep)  ->  narrow_kernel (per candidate pair)
  ->  respond_kernel (per contact)     ->  apply_kernel (Jacobi dP/count)

Replicated reference quirks (numerical parity with the golden dump beats
taste — see docs/superpowers/specs/2026-06-11-dcd-collision-design.md):
  * candidate depths use offset=0; only the final accept test uses 1e-6
  * smaller-positive of (C1, C2) wins; tie goes to the C2 branch
  * response pushes all 3 verts of each movable face EQUALLY (no mass or
    bary weighting); both movable -> half each; one fixed -> full to movable
  * friction interpolates closest-face velocities with the cur face's
    barys and vice versa (reference :3032/:3035)
Deviations (documented, accepted):
  * Jacobi accumulation (dP / count) instead of serial Gauss-Seidel
  * invalid plane-line intersections are skipped instead of reusing the
    previous pair's stale Q point (reference bug)
  * parallel-plane pairs (|N1xN2| < 1e-8) are skipped
  * _bary returns (1/3,1/3,1/3) on denom==0 (reference would yield inf/NaN)
  * dnorm > 0 guard in friction avoids 0/0 when mu_s==0 and diff==0
  * reference's iType1==TriType → N2=-N2 cloth-flip is dropped; outward-
    winding convention is required instead (see winding note in __init__)

All arithmetic fp32.
"""
from __future__ import annotations

import warp as wp

THRESH = wp.constant(1.0e-5)   # gen_t same-side threshold (reference :1366)


# ======================================================================
# Broadphase helpers (centroid + bounding radius per tri, global max
# radius) — used by UdcfDcdCollider to size the hashgrid cell.
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


# ======================================================================
# Box boundary constraint — faithful Warp port of Gaia_UDCF's
# ``PBDTetMeshMassSpring::solveBoxBoundaryConstraintCPU``
# (Simulator/Modules/PBD/PBDTetMeshMassSpring.cpp:370).
#
# Gaia handles the ground plane (``usePlaneGround``) through this boundary
# solve, NOT through tri-tri DCD: a one-sided plane gives C2 <= 0 in the
# narrow phase so the DCD never fires floor contacts.  Any scene that uses
# UdcfDcdCollider for self/mutual collision should pair it with this kernel
# for world bounds.  Per vertex/dim: clamp to [lower, upper] (penetration
# measured pre-clamp), then friction on the PBD displacement (pos - pprev)
# perpendicular to the contact normal; static stick if |tangential| <=
# pen*mu_s, else dynamic scale to min(pen*mu_d, |tangential|).  ``pprev`` is
# read-only — the normal clamp removes normal velocity, the elastic material
# produces rebound.
# ======================================================================

@wp.kernel
def box_boundary_kernel(pos: wp.array(dtype=wp.vec3),
                        pprev: wp.array(dtype=wp.vec3),
                        lower: wp.vec3,
                        upper: wp.vec3,
                        mu_s: wp.float32,
                        mu_d: wp.float32):
    i = wp.tid()
    for dim in range(3):
        p = pos[i]
        pen = float(0.0)
        if p[dim] < lower[dim]:
            pen = lower[dim] - p[dim]
            pos[i] = wp.vec3(
                wp.where(dim == 0, lower[0], p[0]),
                wp.where(dim == 1, lower[1], p[1]),
                wp.where(dim == 2, lower[2], p[2]))
        elif p[dim] > upper[dim]:
            pen = p[dim] - upper[dim]
            pos[i] = wp.vec3(
                wp.where(dim == 0, upper[0], p[0]),
                wp.where(dim == 1, upper[1], p[1]),
                wp.where(dim == 2, upper[2], p[2]))
        if pen > 0.0:
            # Tangential displacement this substep (post normal clamp).
            vel = pos[i] - pprev[i]
            diff = wp.vec3(
                wp.where(dim == 0, 0.0, vel[0]),
                wp.where(dim == 1, 0.0, vel[1]),
                wp.where(dim == 2, 0.0, vel[2]))
            dnorm = wp.length(diff)
            if dnorm > 0.0:
                dynamic_norm = pen * mu_d
                if dynamic_norm > dnorm:
                    dynamic_norm = dnorm
                static_norm = pen * mu_s
                if dnorm > static_norm:
                    diff = diff * (dynamic_norm / dnorm)
                pos[i] = pos[i] - diff


# ======================================================================
# Geometry helpers — line-for-line ports of narrowPhase()'s `internal`
# ======================================================================

@wp.func
def _pick(a: wp.vec3, b: wp.vec3, c: wp.vec3, i: int) -> wp.vec3:
    if i == 0:
        return a
    if i == 1:
        return b
    return c


@wp.func
def _inside_tri(p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, p: wp.vec3) -> int:
    # reference inside_triangle_on_same_plane (:1302)
    c1 = wp.cross(p2 - p1, p - p2)
    c2 = wp.cross(p3 - p2, p - p3)
    c3 = wp.cross(p1 - p3, p - p1)
    if wp.dot(c1, c2) > 0.0 and wp.dot(c1, c3) > 0.0:
        return 1
    return 0


@wp.func
def _bary(A: wp.vec3, B: wp.vec3, C: wp.vec3, P: wp.vec3) -> wp.vec3:
    # reference computeBarycentric (:1395) — dot-product form, NOT the
    # area form used by udcf_warp._bary_tri
    v0 = B - A
    v1 = C - A
    v2 = P - A
    d00 = wp.dot(v0, v0)
    d01 = wp.dot(v0, v1)
    d11 = wp.dot(v1, v1)
    d20 = wp.dot(v2, v0)
    d21 = wp.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if denom == 0.0:
        return wp.vec3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return wp.vec3(1.0 - v - w, v, w)


@wp.func
def _line_isect(p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, p4: wp.vec3,
                v1: wp.vec3, v2: wp.vec3) -> int:
    # reference line_intersection_on_same_plane (:1270): edges (p1,p2) and
    # (p3,p4) are co-planar projections; the hit is evaluated on the
    # ORIGINAL edge (v1,v2). Returns valid flag; point returned via out-param
    # pattern (separate call for valid-then-use).
    d1 = p2 - p1
    d2 = p4 - p3
    nrm = wp.cross(d1, d2)
    if wp.length(nrm) < 1.0e-6:
        return 0
    denom = wp.dot(nrm, nrm)
    if denom < 1.0e-6:
        return 0
    v = p3 - p1
    t1 = wp.dot(wp.cross(v, d2), nrm) / denom
    t2 = wp.dot(wp.cross(v, d1), nrm) / denom
    if t2 >= 0.0 and t2 <= 1.0 and t1 >= 0.0 and t1 <= 1.0:
        return 1
    return 0


@wp.func
def _line_isect_pt(p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, p4: wp.vec3,
                   v1: wp.vec3, v2: wp.vec3) -> wp.vec3:
    # Return the actual hit point (only call after _line_isect returned 1).
    d1 = p2 - p1
    d2 = p4 - p3
    nrm = wp.cross(d1, d2)
    denom = wp.dot(nrm, nrm)
    v = p3 - p1
    t1 = wp.dot(wp.cross(v, d2), nrm) / denom
    return v1 + t1 * (v2 - v1)


@wp.func
def _find_isect_pt(N: wp.vec3, p0: wp.vec3, p: wp.vec3, line_dir: wp.vec3) -> int:
    # reference find_intersection_point (:1293) — returns valid flag
    denom = wp.dot(N, line_dir)
    if denom == 0.0:
        return 0
    return 1


@wp.func
def _find_isect_pt_val(N: wp.vec3, p0: wp.vec3, p: wp.vec3, line_dir: wp.vec3) -> wp.vec3:
    # Return the actual intersection point (only call after _find_isect_pt returned 1).
    denom = wp.dot(N, line_dir)
    t = wp.dot(N, p0 - p) / denom
    return p + t * line_dir


@wp.func
def _gen_t(N: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3,
           d: float, D: wp.vec3) -> int:
    # reference gen_t (:1361): signed distances vs plane (N, d), same-side
    # reject with THRESH, reorder so sv1 is the lone-side vertex, then the
    # Moller interval [t_lo, t_hi] along D. Returns valid flag.
    d0 = wp.dot(N, v0) + d
    d1 = wp.dot(N, v1) + d
    d2 = wp.dot(N, v2) + d
    if d0 <= THRESH and d1 <= THRESH and d2 <= THRESH:
        return 0
    if d0 >= -THRESH and d1 >= -THRESH and d2 >= -THRESH:
        return 0
    return 1


@wp.func
def _gen_t_interval(N: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3,
                    d: float, D: wp.vec3) -> wp.vec2:
    # Compute and return (t_lo, t_hi) sorted interval (only call after
    # _gen_t returned 1).
    d0 = wp.dot(N, v0) + d
    d1 = wp.dot(N, v1) + d
    d2 = wp.dot(N, v2) + d

    sv0 = v0
    sv1 = v1
    sv2 = v2
    sd0 = d0
    sd1 = d1
    sd2 = d2
    # reference swap (:1322): two cascaded if/else-if blocks
    if (d0 <= 0.0 and d1 >= 0.0 and d2 >= 0.0) or \
       (d0 >= 0.0 and d1 <= 0.0 and d2 <= 0.0):
        sv0 = v1
        sv1 = v0
        sd0 = d1
        sd1 = d0
    elif (d0 >= 0.0 and d1 >= 0.0 and d2 <= 0.0) or \
         (d0 <= 0.0 and d1 <= 0.0 and d2 >= 0.0):
        sv1 = v2
        sv2 = v1
        sd1 = d2
        sd2 = d1

    p0 = wp.dot(D, sv0)
    p1 = wp.dot(D, sv1)
    p2 = wp.dot(D, sv2)
    tA = p0 + (p1 - p0) * wp.abs(sd0 / (sd0 - sd1))
    tB = p2 + (p1 - p2) * wp.abs(sd2 / (sd2 - sd1))
    if tA > tB:
        return wp.vec2(tB, tA)
    return wp.vec2(tA, tB)


# ======================================================================
# Narrow phase kernel — port of narrowPhase()'s per-pair body
# ======================================================================

@wp.kernel
def narrow_kernel(
    pair_a: wp.array(dtype=wp.int32),
    pair_b: wp.array(dtype=wp.int32),
    n_pairs: wp.array(dtype=wp.int32),
    tris: wp.array2d(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    offset_eps: wp.float32,            # reference final accept offset (1e-6)
    max_contacts: wp.int32,
    n_contacts: wp.array(dtype=wp.int32),
    c_cur: wp.array2d(dtype=wp.int32),       # (max_contacts, 3) vert ids, +C*n side
    c_closest: wp.array2d(dtype=wp.int32),   # (max_contacts, 3) vert ids, -C*n side
    c_C: wp.array(dtype=wp.float32),         # chosen depth (pre-multiplier)
    c_C1: wp.array(dtype=wp.float32),        # diagnostics / golden replay
    c_C2: wp.array(dtype=wp.float32),
    c_n: wp.array(dtype=wp.vec3),            # gradient = opposing plane normal
    c_b1: wp.array(dtype=wp.vec3),           # bary as apply's barycentric1
    c_b2: wp.array(dtype=wp.vec3),           # bary as apply's barycentric2
):
    k = wp.tid()
    if k >= n_pairs[0]:
        return
    ta = pair_a[k]
    tb = pair_b[k]
    q00 = x[tris[ta, 0]]
    q01 = x[tris[ta, 1]]
    q02 = x[tris[ta, 2]]
    q10 = x[tris[tb, 0]]
    q11 = x[tris[tb, 1]]
    q12 = x[tris[tb, 2]]

    n1f = wp.cross(q01 - q00, q02 - q00)
    n2f = wp.cross(q11 - q10, q12 - q10)
    l1 = wp.length(n1f)
    l2 = wp.length(n2f)
    if l1 == 0.0 or l2 == 0.0:           # degenerate triangle
        return
    N1 = n1f / l1
    N2 = n2f / l2
    d1 = -wp.dot(N1, q00)
    d2 = -wp.dot(N2, q10)

    cp = wp.cross(N1, N2)
    lcp = wp.length(cp)
    if lcp < 1.0e-8:                     # parallel planes (deviation: skip)
        return
    D = cp / lcp

    ok1 = _gen_t(N2, q00, q01, q02, d2, D)
    ok2 = _gen_t(N1, q10, q11, q12, d1, D)
    if ok1 == 0 or ok2 == 0:
        return
    iv1 = _gen_t_interval(N2, q00, q01, q02, d2, D)
    iv2 = _gen_t_interval(N1, q10, q11, q12, d1, D)
    t1 = iv1[0]
    t2 = iv1[1]
    t3 = iv2[0]
    t4 = iv2[1]
    if not (t2 >= t3 and t4 >= t1):
        return

    C1 = float(0.0)
    C2 = float(0.0)
    b1 = wp.vec3(0.0, 0.0, 0.0)
    b2 = wp.vec3(0.0, 0.0, 0.0)
    b3 = wp.vec3(0.0, 0.0, 0.0)
    b4 = wp.vec3(0.0, 0.0, 0.0)

    for j in range(3):
        q0j = _pick(q00, q01, q02, j)
        q1j = _pick(q10, q11, q12, j)
        # vertex-face: T0 vert projected onto T1's plane (reference P[0][j])
        P0j = q0j - wp.dot(N2, q0j - q10) * N2
        if _inside_tri(q10, q11, q12, P0j) == 1:
            cand = wp.dot(N2, q0j) + d2
            if C1 < -cand:
                C1 = -cand
                b1 = _bary(q00, q01, q02, q0j)
                b2 = _bary(q10, q11, q12, q0j - cand * N2)
        # Q point: T1 vert shot along N2 onto T0's plane (reference Q[0][j])
        okq = _find_isect_pt(N1, q00, q1j, N2)
        if okq == 1:
            Q0j = _find_isect_pt_val(N1, q00, q1j, N2)
            if _inside_tri(q00, q01, q02, Q0j) == 1:
                cand = wp.dot(N2, Q0j) + d2
                if C1 < -cand:
                    C1 = -cand
                    b1 = _bary(q00, q01, q02, Q0j)
                    b2 = _bary(q10, q11, q12, Q0j - cand * N2)
        # symmetric direction (reference P[1][j] / Q[1][j])
        P1j = q1j - wp.dot(N1, q1j - q00) * N1
        if _inside_tri(q00, q01, q02, P1j) == 1:
            cand = wp.dot(N1, q1j) + d1
            if C2 < -cand:
                C2 = -cand
                b3 = _bary(q10, q11, q12, q1j)
                b4 = _bary(q00, q01, q02, q1j - cand * N1)
        okq = _find_isect_pt(N2, q10, q0j, N1)
        if okq == 1:
            Q1j = _find_isect_pt_val(N2, q10, q0j, N1)
            if _inside_tri(q10, q11, q12, Q1j) == 1:
                cand = wp.dot(N1, Q1j) + d1
                if C2 < -cand:
                    C2 = -cand
                    b3 = _bary(q10, q11, q12, Q1j)
                    b4 = _bary(q00, q01, q02, Q1j - cand * N1)
        # edge-edge (reference inner k loop)
        jn = (j + 1) % 3
        q0jn = _pick(q00, q01, q02, jn)
        q1jn = _pick(q10, q11, q12, jn)
        P0jn = q0jn - wp.dot(N2, q0jn - q10) * N2
        P1jn = q1jn - wp.dot(N1, q1jn - q00) * N1
        for kk in range(3):
            kn = (kk + 1) % 3
            q1k = _pick(q10, q11, q12, kk)
            q1kn = _pick(q10, q11, q12, kn)
            oke = _line_isect(P0j, P0jn, q1k, q1kn, q0j, q0jn)
            if oke == 1:
                out = _line_isect_pt(P0j, P0jn, q1k, q1kn, q0j, q0jn)
                cand = wp.dot(N2, out) + d2
                if C1 < -cand:
                    C1 = -cand
                    b1 = _bary(q00, q01, q02, out)
                    b2 = _bary(q10, q11, q12, out - cand * N2)
            q0k = _pick(q00, q01, q02, kk)
            q0kn = _pick(q00, q01, q02, kn)
            oke = _line_isect(P1j, P1jn, q0k, q0kn, q1j, q1jn)
            if oke == 1:
                out = _line_isect_pt(P1j, P1jn, q0k, q0kn, q1j, q1jn)
                cand = wp.dot(N1, out) + d1
                if C2 < -cand:
                    C2 = -cand
                    b3 = _bary(q10, q11, q12, out)
                    b4 = _bary(q00, q01, q02, out - cand * N1)

    # reference :1698-1745: both must exceed offset; smaller wins (tie -> C2)
    if C1 <= offset_eps or C2 <= offset_eps:
        return
    slot = wp.atomic_add(n_contacts, 0, 1)
    if slot >= max_contacts:
        return
    if C1 < C2:
        for j in range(3):
            c_cur[slot, j] = tris[ta, j]
            c_closest[slot, j] = tris[tb, j]
        c_C[slot] = C1
        c_n[slot] = N2
        c_b1[slot] = b1
        c_b2[slot] = b2
    else:
        for j in range(3):
            c_cur[slot, j] = tris[tb, j]
            c_closest[slot, j] = tris[ta, j]
        c_C[slot] = C2
        c_n[slot] = N1
        c_b1[slot] = b3
        c_b2[slot] = b4
    c_C1[slot] = C1
    c_C2[slot] = C2


# ======================================================================
# Fused serial Gauss-Seidel narrow+respond — faithful to the reference
# control flow: narrowPhase() iterates candidate pairs SERIALLY, and
# applyNarrowCollisionConstraintsCPU() writes positions IMMEDIATELY, so
# pair k's detection runs on positions already corrected by pairs 0..k-1.
# This interleaving is load-bearing: under fold/press loads, successive
# pairs covering the same crossing each re-measure a SMALLER penetration
# and the total push converges, where parallel-Jacobi accumulation either
# under-corrects (count-average) or explodes (sum).
# Launched with dim=1 (single thread). Serial on purpose.
# ======================================================================

@wp.kernel
def narrow_respond_gs_kernel(
    pair_a: wp.array(dtype=wp.int32),
    pair_b: wp.array(dtype=wp.int32),
    n_pairs: wp.array(dtype=wp.int32),
    max_pairs: wp.int32,
    tris: wp.array2d(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),              # read AND written in-loop
    x_prev: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    offset_eps: wp.float32,
    multiplier: wp.float32,
    mu_s: wp.float32,
    mu_d: wp.float32,
    max_push: wp.float32,
    n_contacts: wp.array(dtype=wp.int32),
    dbg: wp.array(dtype=wp.float32),         # [0]=max pen, [1]=max |diff|
):
    npairs = n_pairs[0]
    if npairs > max_pairs:
        npairs = max_pairs
    for k in range(npairs):
        ta = pair_a[k]
        tb = pair_b[k]
        q00 = x[tris[ta, 0]]
        q01 = x[tris[ta, 1]]
        q02 = x[tris[ta, 2]]
        q10 = x[tris[tb, 0]]
        q11 = x[tris[tb, 1]]
        q12 = x[tris[tb, 2]]

        n1f = wp.cross(q01 - q00, q02 - q00)
        n2f = wp.cross(q11 - q10, q12 - q10)
        l1 = wp.length(n1f)
        l2 = wp.length(n2f)
        if l1 == 0.0 or l2 == 0.0:
            continue
        N1 = n1f / l1
        N2 = n2f / l2
        d1 = -wp.dot(N1, q00)
        d2 = -wp.dot(N2, q10)

        cp = wp.cross(N1, N2)
        lcp = wp.length(cp)
        if lcp < 1.0e-8:
            continue
        D = cp / lcp

        ok1 = _gen_t(N2, q00, q01, q02, d2, D)
        ok2 = _gen_t(N1, q10, q11, q12, d1, D)
        if ok1 == 0 or ok2 == 0:
            continue
        iv1 = _gen_t_interval(N2, q00, q01, q02, d2, D)
        iv2 = _gen_t_interval(N1, q10, q11, q12, d1, D)
        if not (iv1[1] >= iv2[0] and iv2[1] >= iv1[0]):
            continue

        C1 = float(0.0)
        C2 = float(0.0)
        b1 = wp.vec3(0.0, 0.0, 0.0)
        b2 = wp.vec3(0.0, 0.0, 0.0)
        b3 = wp.vec3(0.0, 0.0, 0.0)
        b4 = wp.vec3(0.0, 0.0, 0.0)

        for j in range(3):
            q0j = _pick(q00, q01, q02, j)
            q1j = _pick(q10, q11, q12, j)
            P0j = q0j - wp.dot(N2, q0j - q10) * N2
            if _inside_tri(q10, q11, q12, P0j) == 1:
                cand = wp.dot(N2, q0j) + d2
                if C1 < -cand:
                    C1 = -cand
                    b1 = _bary(q00, q01, q02, q0j)
                    b2 = _bary(q10, q11, q12, q0j - cand * N2)
            okq = _find_isect_pt(N1, q00, q1j, N2)
            if okq == 1:
                Q0j = _find_isect_pt_val(N1, q00, q1j, N2)
                if _inside_tri(q00, q01, q02, Q0j) == 1:
                    cand = wp.dot(N2, Q0j) + d2
                    if C1 < -cand:
                        C1 = -cand
                        b1 = _bary(q00, q01, q02, Q0j)
                        b2 = _bary(q10, q11, q12, Q0j - cand * N2)
            P1j = q1j - wp.dot(N1, q1j - q00) * N1
            if _inside_tri(q00, q01, q02, P1j) == 1:
                cand = wp.dot(N1, q1j) + d1
                if C2 < -cand:
                    C2 = -cand
                    b3 = _bary(q10, q11, q12, q1j)
                    b4 = _bary(q00, q01, q02, q1j - cand * N1)
            okq = _find_isect_pt(N2, q10, q0j, N1)
            if okq == 1:
                Q1j = _find_isect_pt_val(N2, q10, q0j, N1)
                if _inside_tri(q10, q11, q12, Q1j) == 1:
                    cand = wp.dot(N1, Q1j) + d1
                    if C2 < -cand:
                        C2 = -cand
                        b3 = _bary(q10, q11, q12, Q1j)
                        b4 = _bary(q00, q01, q02, Q1j - cand * N1)
            jn = (j + 1) % 3
            q0jn = _pick(q00, q01, q02, jn)
            q1jn = _pick(q10, q11, q12, jn)
            P0jn = q0jn - wp.dot(N2, q0jn - q10) * N2
            P1jn = q1jn - wp.dot(N1, q1jn - q00) * N1
            for kk in range(3):
                kn = (kk + 1) % 3
                q1k = _pick(q10, q11, q12, kk)
                q1kn = _pick(q10, q11, q12, kn)
                oke = _line_isect(P0j, P0jn, q1k, q1kn, q0j, q0jn)
                if oke == 1:
                    out = _line_isect_pt(P0j, P0jn, q1k, q1kn, q0j, q0jn)
                    cand = wp.dot(N2, out) + d2
                    if C1 < -cand:
                        C1 = -cand
                        b1 = _bary(q00, q01, q02, out)
                        b2 = _bary(q10, q11, q12, out - cand * N2)
                q0k = _pick(q00, q01, q02, kk)
                q0kn = _pick(q00, q01, q02, kn)
                oke = _line_isect(P1j, P1jn, q0k, q0kn, q1j, q1jn)
                if oke == 1:
                    out = _line_isect_pt(P1j, P1jn, q0k, q0kn, q1j, q1jn)
                    cand = wp.dot(N1, out) + d1
                    if C2 < -cand:
                        C2 = -cand
                        b3 = _bary(q10, q11, q12, out)
                        b4 = _bary(q00, q01, q02, out - cand * N1)

        if C1 <= offset_eps or C2 <= offset_eps:
            continue

        # branch select (smaller positive wins, tie -> C2) then respond
        # with IMMEDIATE writes — port of applyNarrowCollisionConstraintsCPU
        cu0 = tris[ta, 0]
        cu1 = tris[ta, 1]
        cu2 = tris[ta, 2]
        cl0 = tris[tb, 0]
        cl1 = tris[tb, 1]
        cl2 = tris[tb, 2]
        pen = C2
        nrm = N1
        rb1 = b3
        rb2 = b4
        if C1 < C2:
            cu0 = tris[ta, 0]
            cu1 = tris[ta, 1]
            cu2 = tris[ta, 2]
            cl0 = tris[tb, 0]
            cl1 = tris[tb, 1]
            cl2 = tris[tb, 2]
            pen = C1
            nrm = N2
            rb1 = b1
            rb2 = b2
        else:
            cu0 = tris[tb, 0]
            cu1 = tris[tb, 1]
            cu2 = tris[tb, 2]
            cl0 = tris[ta, 0]
            cl1 = tris[ta, 1]
            cl2 = tris[ta, 2]
        wp.atomic_add(n_contacts, 0, 1)

        if max_push > 0.0 and pen > max_push:
            pen = max_push
        C = pen * multiplier
        if C == 0.0:
            continue
        solid_cur = 0
        solid_clo = 0
        if (inv_mass[cu0] == 0.0 or inv_mass[cu1] == 0.0
                or inv_mass[cu2] == 0.0):
            solid_cur = 1
        if (inv_mass[cl0] == 0.0 or inv_mass[cl1] == 0.0
                or inv_mass[cl2] == 0.0):
            solid_clo = 1
        if solid_cur == 0 and solid_clo == 0:
            C = C * 0.5

        push_clo = wp.vec3(0.0, 0.0, 0.0)
        if solid_clo == 0:
            push_clo = -C * nrm
        push_cur = wp.vec3(0.0, 0.0, 0.0)
        if solid_cur == 0:
            push_cur = C * nrm
        v0 = (x[cl0] + push_clo) - x_prev[cl0]
        v1 = (x[cl1] + push_clo) - x_prev[cl1]
        v2 = (x[cl2] + push_clo) - x_prev[cl2]
        v3 = (x[cu0] + push_cur) - x_prev[cu0]
        v4 = (x[cu1] + push_cur) - x_prev[cu1]
        v5 = (x[cu2] + push_cur) - x_prev[cu2]

        g = nrm / wp.length(nrm)
        vclo = v0 * rb1[0] + v1 * rb1[1] + v2 * rb1[2]
        perp_clo = vclo - g * wp.dot(vclo, g)
        vcur = v3 * rb2[0] + v4 * rb2[1] + v5 * rb2[2]
        perp_cur = vcur - g * wp.dot(vcur, g)
        diff = perp_cur - perp_clo
        dnorm = wp.length(diff)
        static_norm = pen * mu_s
        if dnorm > static_norm and dnorm > 0.0:
            diff = diff * (pen * mu_d / dnorm)
        diffV = diff * 0.5
        diffT = diff - diffV
        wp.atomic_max(dbg, 0, pen)
        wp.atomic_max(dbg, 1, wp.length(diff))

        if inv_mass[cu0] > 0.0:
            x[cu0] = x[cu0] + push_cur - diffV
        if inv_mass[cu1] > 0.0:
            x[cu1] = x[cu1] + push_cur - diffV
        if inv_mass[cu2] > 0.0:
            x[cu2] = x[cu2] + push_cur - diffV
        if inv_mass[cl0] > 0.0:
            x[cl0] = x[cl0] + push_clo + diffT
        if inv_mass[cl1] > 0.0:
            x[cl1] = x[cl1] + push_clo + diffT
        if inv_mass[cl2] > 0.0:
            x[cl2] = x[cl2] + push_clo + diffT


# ======================================================================
# Response — port of applyNarrowCollisionConstraintsCPU (:2894-3116)
# ======================================================================

@wp.kernel
def respond_kernel(
    n_contacts: wp.array(dtype=wp.int32),
    c_cur: wp.array2d(dtype=wp.int32),
    c_closest: wp.array2d(dtype=wp.int32),
    c_C: wp.array(dtype=wp.float32),         # pre-multiplier depth
    c_n: wp.array(dtype=wp.vec3),
    c_b1: wp.array(dtype=wp.vec3),           # apply's barycentric1
    c_b2: wp.array(dtype=wp.vec3),           # apply's barycentric2
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    multiplier: wp.float32,
    mu_s: wp.array(dtype=wp.float32),        # per-contact (golden replay
    mu_d: wp.array(dtype=wp.float32),        # carries per-mesh values)
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
):
    k = wp.tid()
    if k >= n_contacts[0]:
        return
    pen = c_C[k]                       # reference penetrationDepth arg
    C = pen * multiplier               # :2901
    if C == 0.0:
        return
    nrm = c_n[k]

    cu0 = c_cur[k, 0]
    cu1 = c_cur[k, 1]
    cu2 = c_cur[k, 2]
    cl0 = c_closest[k, 0]
    cl1 = c_closest[k, 1]
    cl2 = c_closest[k, 2]

    solid_cur = 0
    solid_clo = 0
    if inv_mass[cu0] == 0.0 or inv_mass[cu1] == 0.0 or inv_mass[cu2] == 0.0:
        solid_cur = 1
    if inv_mass[cl0] == 0.0 or inv_mass[cl1] == 0.0 or inv_mass[cl2] == 0.0:
        solid_clo = 1
    if solid_cur == 0 and solid_clo == 0:    # :2966
        C = C * 0.5

    # normal push (local; written to x only via dP) + post-push velocities
    push_clo = wp.vec3(0.0, 0.0, 0.0)
    if solid_clo == 0:
        push_clo = -C * nrm
    push_cur = wp.vec3(0.0, 0.0, 0.0)
    if solid_cur == 0:
        push_cur = C * nrm
    v0 = (x[cl0] + push_clo) - x_prev[cl0]   # :3005-3023 velocities[0..5]
    v1 = (x[cl1] + push_clo) - x_prev[cl1]
    v2 = (x[cl2] + push_clo) - x_prev[cl2]
    v3 = (x[cu0] + push_cur) - x_prev[cu0]
    v4 = (x[cu1] + push_cur) - x_prev[cu1]
    v5 = (x[cu2] + push_cur) - x_prev[cu2]

    g = nrm / wp.length(nrm)                 # :3029-3030
    b1 = c_b1[k]
    b2 = c_b2[k]
    # reference quirk (:3032/:3035): closest velocities x barycentric1,
    # cur velocities x barycentric2 — replicated for golden parity
    vclo = v0 * b1[0] + v1 * b1[1] + v2 * b1[2]
    perp_clo = vclo - g * wp.dot(vclo, g)
    vcur = v3 * b2[0] + v4 * b2[1] + v5 * b2[2]
    perp_cur = vcur - g * wp.dot(vcur, g)
    diff = perp_cur - perp_clo               # :3038
    dnorm = wp.length(diff)
    static_norm = pen * mu_s[k]              # :3058
    if dnorm > static_norm and dnorm > 0.0:  # :3060 slipping -> pen*mud
        diff = diff * (pen * mu_d[k] / dnorm)
    diffV = diff * 0.5                       # :3072 cur faces get -diffV
    diffT = diff - diffV                     # :3077 closest faces get +diffT

    # accumulate; inv_mass==0 verts skipped — equivalent to the reference
    # moving fixed verts then re-pinning them in applyConstraints()
    if inv_mass[cu0] > 0.0:
        wp.atomic_add(dP, cu0, push_cur - diffV)
        wp.atomic_add(dPw, cu0, 1.0)
    if inv_mass[cu1] > 0.0:
        wp.atomic_add(dP, cu1, push_cur - diffV)
        wp.atomic_add(dPw, cu1, 1.0)
    if inv_mass[cu2] > 0.0:
        wp.atomic_add(dP, cu2, push_cur - diffV)
        wp.atomic_add(dPw, cu2, 1.0)
    if inv_mass[cl0] > 0.0:
        wp.atomic_add(dP, cl0, push_clo + diffT)
        wp.atomic_add(dPw, cl0, 1.0)
    if inv_mass[cl1] > 0.0:
        wp.atomic_add(dP, cl1, push_clo + diffT)
        wp.atomic_add(dPw, cl1, 1.0)
    if inv_mass[cl2] > 0.0:
        wp.atomic_add(dP, cl2, push_clo + diffT)
        wp.atomic_add(dPw, cl2, 1.0)


@wp.kernel
def apply_kernel(
    pos: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
    use_avg: wp.int32,
):
    i = wp.tid()
    w = dPw[i]
    if w > 0.0:
        if use_avg != 0:
            pos[i] = pos[i] + dP[i] / w
        else:
            pos[i] = pos[i] + dP[i]


@wp.kernel
def _clamp_contact_depth_kernel(
    n_contacts: wp.array(dtype=wp.int32),
    c_C: wp.array(dtype=wp.float32),
    max_push: wp.float32,
):
    k = wp.tid()
    if k >= n_contacts[0]:
        return
    if c_C[k] > max_push:
        c_C[k] = max_push


@wp.kernel
def _peak_update_kernel(
    n_contacts: wp.array(dtype=wp.int32),
    peak: wp.array(dtype=wp.int32),
):
    wp.atomic_max(peak, 0, n_contacts[0])


# ======================================================================
# Broadphase: candidate pairs with movable/static group rules
# ======================================================================

@wp.kernel
def emit_pairs_groups_kernel(
    grid: wp.uint64,
    centroids: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32),
    tris: wp.array2d(dtype=wp.int32),
    tri_group: wp.array(dtype=wp.int32),     # 0 = movable, 1 = static
    allow_self: wp.int32,                    # movable-vs-movable pairs
    query_radius: wp.float32,
    thickness: wp.float32,
    max_pairs: wp.int32,
    pair_a: wp.array(dtype=wp.int32),
    pair_b: wp.array(dtype=wp.int32),
    n_pairs: wp.array(dtype=wp.int32),
):
    """Unique candidate pairs (i < j). Rules:
      static-static          -> never (neither side can move)
      movable-self (0 vs 0)  -> only when allow_self, with shared-vertex
                               adjacency filter (vert spaces of distinct
                               groups are disjoint, so cross pairs never
                               share verts and skip the 9-way compare)
      cross (0 vs 1)         -> always
    Storage is bounded at max_pairs (slots beyond that are silently dropped),
    but the raw counter is NOT clamped — stats() reads the unbounded value to
    detect overflow."""
    i = wp.tid()
    ci = centroids[i]
    ri = radii[i]
    gi = tri_group[i]
    q = wp.hash_grid_query(grid, ci, query_radius)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if j <= i:
            continue
        gj = tri_group[j]
        if gi == 1 and gj == 1:
            continue
        if gi == 0 and gj == 0:
            if allow_self == 0:
                continue
            ai0 = tris[i, 0]
            ai1 = tris[i, 1]
            ai2 = tris[i, 2]
            bj0 = tris[j, 0]
            bj1 = tris[j, 1]
            bj2 = tris[j, 2]
            if (ai0 == bj0 or ai0 == bj1 or ai0 == bj2 or
                    ai1 == bj0 or ai1 == bj1 or ai1 == bj2 or
                    ai2 == bj0 or ai2 == bj1 or ai2 == bj2):
                continue
        if wp.length(ci - centroids[j]) > ri + radii[j] + thickness:
            continue
        slot = wp.atomic_add(n_pairs, 0, 1)
        if slot < max_pairs:
            pair_a[slot] = i
            pair_b[slot] = j

