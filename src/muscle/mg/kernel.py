from __future__ import annotations

import warp as wp


@wp.kernel
def compute_barycentric_kernel(
    query_pos: wp.array(dtype=wp.vec3),
    cage_pos: wp.array(dtype=wp.vec3),
    cage_tet: wp.array(dtype=wp.vec4i),
    out_cage_idx: wp.array(dtype=wp.int32),
    out_bary: wp.array(dtype=wp.vec3),
):
    """For each query point, find a cage tet containing it (brute force).

    Mirrors `compute_barycentric_kernel` in
    realtimeMuscle/engine/compute_R_acc.py — falls back to nearest-vertex tet
    if no container is found. Brute force is fine: this runs once at init.
    """
    i = wp.tid()
    p = query_pos[i]
    n_cage = cage_tet.shape[0]

    found_idx = int(-1)
    found_bary = wp.vec3(0.0, 0.0, 0.0)
    min_dis = float(1.0e30)
    nearest_tet = int(0)
    nearest_bary = wp.vec3(0.0, 0.0, 0.0)

    for t in range(n_cage):
        idx = cage_tet[t]
        p0 = cage_pos[idx[0]]
        p1 = cage_pos[idx[1]]
        p2 = cage_pos[idx[2]]
        p3 = cage_pos[idx[3]]
        Am = wp.mat33(
            p1[0] - p0[0], p2[0] - p0[0], p3[0] - p0[0],
            p1[1] - p0[1], p2[1] - p0[1], p3[1] - p0[1],
            p1[2] - p0[2], p2[2] - p0[2], p3[2] - p0[2],
        )
        bv = p - p0
        x = wp.inverse(Am) * bv
        contained = (x[0] >= 0.0 and x[1] >= 0.0 and x[2] >= 0.0
                     and (x[0] + x[1] + x[2] <= 1.0))
        if contained and found_idx == -1:
            found_idx = t
            found_bary = x

        d0 = wp.length(p0 - p)
        d1 = wp.length(p1 - p)
        d2 = wp.length(p2 - p)
        d3 = wp.length(p3 - p)
        dm = wp.min(wp.min(d0, d1), wp.min(d2, d3))
        if dm < min_dis:
            min_dis = dm
            nearest_tet = t
            nearest_bary = x

    if found_idx >= 0:
        out_cage_idx[i] = found_idx
        out_bary[i] = found_bary
    else:
        out_cage_idx[i] = nearest_tet
        out_bary[i] = nearest_bary

        
@wp.kernel
def scatter_target_kernel(
    pts: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3),
    rest_offset: wp.array(dtype=wp.vec3),
    dense_target: wp.array(dtype=wp.vec3),
):
    """Scatter the per-constraint bone target into a dense per-vertex array.

    ``dense_target[pts[i]] = target_pos[i] + rest_offset[i]``. For humanV2
    pts covers every muscle vertex, so dense_target is fully written; it
    feeds restrict_direct_kernel to build the coarse full-pin target.
    """
    i = wp.tid()
    dense_target[pts[i]] = target_pos[i] + rest_offset[i]


@wp.kernel
def restrict_galerkin_scatter_kernel(
    cage_idx: wp.array(dtype=wp.int32),       # fine vert -> coarse tet (prolong P)
    uvw: wp.array(dtype=wp.vec3),
    coarse_tet: wp.array(dtype=wp.vec4i),
    dense_target: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),               # out: b = P^T T (atomic scatter)
):
    """b = P^T T_dense: scatter each fine vert's bone target into its coarse tet's
    4 corners weighted by the prolong barycentric weights. First half of the
    Galerkin restrict Cdef = (P^T P)^-1 (P^T T)."""
    i = wp.tid()
    tet = coarse_tet[cage_idx[i]]
    uv = uvw[i]
    a = 1.0 - uv[0] - uv[1] - uv[2]
    T = dense_target[i]
    wp.atomic_add(b, tet[0], a * T)
    wp.atomic_add(b, tet[1], uv[0] * T)
    wp.atomic_add(b, tet[2], uv[1] * T)
    wp.atomic_add(b, tet[3], uv[2] * T)


@wp.kernel
def block_solve_kernel(
    minv: wp.array(dtype=wp.float32),         # per-muscle pinv(P^T P), row-major blocks
    base: wp.array(dtype=wp.int32),           # per coarse vert: first vert of its block
    dim: wp.array(dtype=wp.int32),            # per coarse vert: block size
    matoff: wp.array(dtype=wp.int32),         # per coarse vert: block offset in minv
    b: wp.array(dtype=wp.vec3),
    coarse_target: wp.array(dtype=wp.vec3),   # out: Cdef = Minv b
):
    """Cdef[c] = sum_j Minv_block[c, j] * b[base+j]. Minv is block-diagonal per
    muscle (a fine vert only weights its own muscle's cage), so each coarse vert
    reads its block's row."""
    c = wp.tid()
    bs = base[c]
    d = dim[c]
    row = matoff[c] + (c - bs) * d
    acc = wp.vec3(0.0, 0.0, 0.0)
    for j in range(d):
        acc = acc + minv[row + j] * b[bs + j]
    coarse_target[c] = acc



    

@wp.kernel
def restrict_direct_kernel(
    fine_pos: wp.array(dtype=wp.vec3),
    fine_tet: wp.array(dtype=wp.vec4i),
    cage_idx_c2f: wp.array(dtype=wp.int32),
    uvw_c2f: wp.array(dtype=wp.vec3),
    coarse_pos: wp.array(dtype=wp.vec3),
):
    """coarse.pos[i] = blend(fine.pos at containing fine tet).

    Direct restriction. Stable V-cycle: coarse mirrors fine each substep,
    relaxes elasticity, prolongs back, fine re-pins and refines. Replaces
    the dp variant from arap_hpbd_nogui.py which is unstable under
    quasi-static (fine_old_pos never updated → cumulative coarse strain).
    """
    i = wp.tid()
    t = cage_idx_c2f[i]
    idx = fine_tet[t]
    uvw = uvw_c2f[i]
    u = uvw[0]
    v = uvw[1]
    w = uvw[2]
    a = 1.0 - u - v - w
    coarse_pos[i] = (a * fine_pos[idx[0]] + u * fine_pos[idx[1]]
                     + v * fine_pos[idx[2]] + w * fine_pos[idx[3]])


@wp.kernel
def prolong_kernel(
    coarse_pos: wp.array(dtype=wp.vec3),
    coarse_tet: wp.array(dtype=wp.vec4i),
    cage_idx: wp.array(dtype=wp.int32),
    uvw: wp.array(dtype=wp.vec3),
    fine_pos: wp.array(dtype=wp.vec3),
):
    """fine.pos[i] = blend(coarse.pos at containing coarse tet).

    Direct (v1) prolongation — mirrors update_fine_mesh_mfree_kernel in
    realtimeMuscle/engine/arap_hpbd_nogui.py:253.
    """
    i = wp.tid()
    t = cage_idx[i]
    idx = coarse_tet[t]
    b = uvw[i]
    u = b[0]
    v = b[1]
    w = b[2]
    a = 1.0 - u - v - w
    fine_pos[i] = (a * coarse_pos[idx[0]] + u * coarse_pos[idx[1]]
                   + v * coarse_pos[idx[2]] + w * coarse_pos[idx[3]])


# ─── Kinematic pin-to-target (direct pos overwrite, NOT XPBD solve) ─────────
# These set pos[] directly each substep; they are not constraint-solve kernels
# (no dP/dPw accumulation) so they live here with the MG transfer kernels
# rather than in constraint/solve_kernels.py.

@wp.kernel
def pintotarget_overwrite_kernel(
    pts: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3),
    rest_offset: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
):
    """Hard-overwrite ``pos[pts[i]] = target_pos[i] + rest_offset[i]``."""
    i = wp.tid()
    pos[pts[i]] = target_pos[i] + rest_offset[i]


@wp.kernel
def pintotarget_overwrite_masked_kernel(
    pts: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3),
    rest_offset: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=wp.int32),     # 1 = apply pin, 0 = skip (freed vert)
    pos: wp.array(dtype=wp.vec3),
):
    """Like pintotarget_overwrite_kernel but skips verts whose pin is freed
    (active==0) so the fascia two-way glue can pull them inward."""
    i = wp.tid()
    if active[i] == 0:
        return
    pos[pts[i]] = target_pos[i] + rest_offset[i]


@wp.kernel
def pintotarget_soft_kernel(
    pts: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3),
    rest_offset: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=wp.int32),     # 0 = freed vert (gets soft pull), 1 = skip
    stiffness: float,                     # in [0,1]: fraction toward bone target / substep
    pos: wp.array(dtype=wp.vec3),
):
    """Soft pin for FREED verts: blend pos a fraction toward the bone target so
    they track gross body motion while still yielding to the fascia squeeze."""
    i = wp.tid()
    if active[i] != 0:
        return
    pid = pts[i]
    tgt = target_pos[i] + rest_offset[i]
    pos[pid] = pos[pid] + stiffness * (tgt - pos[pid])