from __future__ import annotations

import warp as wp

from muscle.constraint.solve_kernels import ssvd_fn


@wp.kernel
def prolong_delta_kernel(
    coarse_pos: wp.array(dtype=wp.vec3),      # coarse pos AFTER this level's solve
    coarse_prev: wp.array(dtype=wp.vec3),     # coarse pos saved BEFORE the solve (q_j)
    coarse_tet: wp.array(dtype=wp.vec4i),
    cage_idx: wp.array(dtype=wp.int32),
    uvw: wp.array(dtype=wp.vec3),
    fine_pos: wp.array(dtype=wp.vec3),        # in/out: fine.pos[i] += prolonged delta
):
    """TRUE HPBD prolongation — Müller 2008 "Hierarchical Position Based Dynamics"
    §7, eq (7):  p_i <- p_i + Σ_{j∈P(i)} w_ij (p_j − q_j).

    Propagate only the coarse level's projection DISPLACEMENT (coarse_pos −
    coarse_prev), barycentric-weighted onto the fine children, NOT the absolute
    coarse position (that is `prolong_kernel`). §8: "only the corrections computed
    on a level are propagated down ... this guarantees that all the small detail in
    the high resolution levels is preserved." Because the increment → 0 as the
    coarse reaches its own rest, it can never bias the fine equilibrium (unlike the
    absolute restrict/prolong FAS scheme). Low modes drive high modes."""
    i = wp.tid()
    t = cage_idx[i]
    idx = coarse_tet[t]
    b = uvw[i]
    u = b[0]
    v = b[1]
    w = b[2]
    a = 1.0 - u - v - w
    d0 = coarse_pos[idx[0]] - coarse_prev[idx[0]]
    d1 = coarse_pos[idx[1]] - coarse_prev[idx[1]]
    d2 = coarse_pos[idx[2]] - coarse_prev[idx[2]]
    d3 = coarse_pos[idx[3]] - coarse_prev[idx[3]]
    fine_pos[i] = fine_pos[i] + (a * d0 + u * d1 + v * d2 + w * d3)


@wp.kernel
def restrict_c2f_kernel(
    c2f_tet: wp.array(dtype=wp.vec4i),    # per coarse vert: its enclosing/nearest fine tet
    c2f_w: wp.array(dtype=wp.vec4),       # per coarse vert: barycentric weights in that tet
    fine_pos: wp.array(dtype=wp.vec3),
    coarse_pos: wp.array(dtype=wp.vec3),  # out: coarse_pos[c] = Σ_k w[c,k] fine_pos[tet_k]
):
    """BUNNY-STYLE (coarse-in-fine) direct position restriction, the realtimeMuscle
    scheme: each coarse vert is embedded in a FINE tet, so its position is read
    straight off the fine mesh by barycentric interpolation. Contrast the Galerkin
    restrict (restrict_galerkin_scatter_kernel + block_solve_kernel) which uses the
    fine-in-coarse operator P and its pseudo-inverse. Well-conditioned only when the
    coarse vert sits inside a fine tet (weights in [0,1]); enclosing cages extrapolate
    (|w|>1) and amplify fine motion."""
    c = wp.tid()
    t = c2f_tet[c]
    w = c2f_w[c]
    coarse_pos[c] = (w[0] * fine_pos[t[0]] + w[1] * fine_pos[t[1]]
                     + w[2] * fine_pos[t[2]] + w[3] * fine_pos[t[3]])


@wp.kernel
def dense_soft_attach_kernel(
    target: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=wp.int32),       # 1 = node has a valid restricted target
    beta: wp.float32,
    pos: wp.array(dtype=wp.vec3),
):
    """Coarse-level BC: soft-pull every REFERENCED coarse node toward its
    (restricted) bone target, ``pos += beta·(target − pos)``. beta=1 snaps, <1
    partial. Interleaved with the coarse ARAP sweeps so the coarse scaffold tracks
    the bone pose while ARAP keeps its shape; its per-substep motion is
    delta-prolonged to the fine.

    ``mask`` skips coarse nodes with a zero P column (no fine vert scatters onto
    them): the Galerkin block-solve leaves those targets at the ORIGIN, so
    attaching them would drag the cage to (0,0,0). Unmasked, they follow the ARAP
    sweep via their referenced neighbours — which is the correct free behaviour."""
    i = wp.tid()
    if mask[i] == 0:
        return
    pos[i] = pos[i] + beta * (target[i] - pos[i])


@wp.kernel
def arap_residual_accum_kernel(
    tet: wp.array(dtype=wp.vec4i),
    pos: wp.array(dtype=wp.vec3),
    rest_matrix: wp.array(dtype=wp.mat33),    # B = inv(Dm), Dm cols (p0-p3,p1-p3,p2-p3)
    out_sumsq: wp.array(dtype=wp.float32),    # length 1: accumulates Σ C^2
):
    """Accumulate Σ C² where C = ‖σ(F) − 1‖ per tet — the same ARAP-SVD constraint
    value `solve_tetarap_svd_kernel` drives to zero. Host divides by NT and sqrts
    for the residual RMS (matches tools/cage_diagnostic.arap_residual). Reference
    vert is pt3 (V2 convention), oriented SVD via ssvd_fn so tet winding is
    irrelevant to the metric."""
    c = wp.tid()
    pts = tet[c]
    p0 = pos[pts[0]]
    p1 = pos[pts[1]]
    p2 = pos[pts[2]]
    p3 = pos[pts[3]]
    c0 = p0 - p3
    c1 = p1 - p3
    c2 = p2 - p3
    Ds = wp.mat33(
        c0[0], c1[0], c2[0],
        c0[1], c1[1], c2[1],
        c0[2], c1[2], c2[2])
    F = Ds * rest_matrix[c]
    svd = ssvd_fn(F)
    s = svd.sigma
    csq = ((s[0] - 1.0) * (s[0] - 1.0)
           + (s[1] - 1.0) * (s[1] - 1.0)
           + (s[2] - 1.0) * (s[2] - 1.0))
    wp.atomic_add(out_sumsq, 0, csq)


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
    feeds the Galerkin restrict to build the coarse full-pin target.
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
def pintotarget_overwrite_masked_kernel(
    pts: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3),
    rest_offset: wp.array(dtype=wp.vec3),
    active: wp.array(dtype=wp.int32),     # 1 = apply pin, 0 = skip (freed vert)
    pos: wp.array(dtype=wp.vec3),
):
    """Hard-overwrite ``pos[pts[i]] = target_pos[i] + rest_offset[i]`` for pinned
    tendon verts; skip freed verts (active==0) so the fascia two-way glue +
    belly-attach can pull them inward."""
    i = wp.tid()
    if active[i] == 0:
        return
    pos[pts[i]] = target_pos[i] + rest_offset[i]


@wp.kernel
def presubframe_pintotarget_kernel(
    pts: wp.array(dtype=wp.int32),
    target_pos: wp.array(dtype=wp.vec3),
    rest_offset: wp.array(dtype=wp.vec3),
    beta: wp.float32,                     # 1 = snap to target, <1 = partial follow
    pos: wp.array(dtype=wp.vec3),
):
    """Kinematic skinning PRESTEP: blend EVERY muscle vert toward its bone target
    BEFORE the XPBD iteration. The bone target rides the skeleton (= the muscle's
    'skinned' position, verified ~rest-gap from bone every frame), so presetting it
    here removes the slow XPBD belly-attach convergence that otherwise leaves the
    belly lagging the bone by hundreds of mm under fast motion (Jacobi propagates
    one ring/iter; 200 iters can't catch a fast walk). XPBD then applies fiber
    bulge / collision / ARAP as LOCAL secondary deformation off this skinned base.

    Runs over ALL verts (not just freed belly): hard-pinned verts are already at
    target from apply_pintotarget, so blending them toward the same target is a
    no-op, and their kinematic lock (wi=0) is unaffected — this keeps the prestep
    correct even when there is NO pin mask (every vert is belly)."""
    i = wp.tid()
    v = pts[i]
    tgt = target_pos[i] + rest_offset[i]
    pos[v] = pos[v] + beta * (tgt - pos[v])


# ── Bone-surface target (target_mode="mesh", use_sliding gates re-projection) ─
# Each muscle vert binds to its nearest point on the bone SURFACE mesh. The
# binding (face + barycentric uv) may re-project ("slide") tangentially as the
# muscle moves, bounded to slide_range from the original anchor. The FULL rest
# offset (rest - closest_point) is stored in the bound face's LOCAL FRAME — not
# just the normal gap — so the target carries the vert rigidly with the face
# (tangential offset + normal gap both preserved, and both follow the bone's
# rotation). At the start frame this reconstructs rest_pos EXACTLY (identity),
# so the hard pin moves nothing when the bone has not moved.

@wp.func
def _face_basis(mesh: wp.uint64, f: wp.int32):
    """Orthonormal frame of bone face ``f`` as a mat33 whose COLUMNS are
    (t1, t2, n): t1 = first edge dir, n = face normal, t2 = n × t1. Built from
    the face's 3 corners so it rotates rigidly with the deforming bone.

    NB the corners come from ``mesh_eval_position`` at bary (0,0)/(1,0)/(0,1) —
    NOT ``mesh_get_index``/``mesh_get_point``: warp reorders the index buffer for
    its BVH, so ``f*3+k`` indexes a DIFFERENT triangle than the face id ``f`` used
    by ``mesh_eval_position`` / ``mesh_query_point``. Using the eval corners keeps
    the frame consistent with cp (else the basis is the wrong triangle's →
    rank-deficient reconstruction, frame-1 pin yanks verts, spurious inversions)."""
    p0 = wp.mesh_eval_position(mesh, f, 0.0, 0.0)
    p1 = wp.mesh_eval_position(mesh, f, 1.0, 0.0)
    p2 = wp.mesh_eval_position(mesh, f, 0.0, 1.0)
    n = wp.normalize(wp.cross(p1 - p0, p2 - p0))
    t1 = wp.normalize(p1 - p0)
    t2 = wp.cross(n, t1)
    return wp.mat33(t1[0], t2[0], n[0],
                    t1[1], t2[1], n[1],
                    t1[2], t2[2], n[2])


@wp.kernel
def bone_bind_kernel(
    mesh: wp.uint64,
    rest_pos: wp.array(dtype=wp.vec3),
    max_dist: wp.float32,
    out_face: wp.array(dtype=wp.int32),
    out_uv: wp.array(dtype=wp.vec2),
    out_local: wp.array(dtype=wp.vec3),
):
    """Bind: closest point on the bone mesh to each muscle rest vert -> face,
    barycentric (u,v), and the FULL rest offset (rest - cp) expressed in the
    face local frame (R^T·d). Reconstructs to rest_pos exactly on the bind
    bone (R·local + cp = d + cp = rest_pos)."""
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, rest_pos[i], max_dist)
    if q.result:
        out_face[i] = q.face
        out_uv[i] = wp.vec2(q.u, q.v)
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        R = _face_basis(mesh, q.face)
        out_local[i] = wp.transpose(R) * (rest_pos[i] - cp)
    else:
        out_face[i] = 0
        out_uv[i] = wp.vec2(0.0, 0.0)
        out_local[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def bone_slide_target_kernel(
    mesh: wp.uint64,                          # current frame's bone mesh
    muscle_pos: wp.array(dtype=wp.vec3),      # current muscle positions
    anchor_face: wp.array(dtype=wp.int32),    # leash origin (fixed bind) face
    anchor_uv: wp.array(dtype=wp.vec2),       # leash origin barycentric
    bind_face: wp.array(dtype=wp.int32),      # current binding face (in/out)
    bind_uv: wp.array(dtype=wp.vec2),         # current binding bary (in/out)
    local_offset: wp.array(dtype=wp.vec3),    # rest offset in face frame (in/out)
    slide_range: wp.float32,                  # max tangential drift from anchor
    reproject: wp.int32,                      # 1 = re-find closest (slide), else follow
    max_dist: wp.float32,
    target_out: wp.array(dtype=wp.vec3),      # out: per-vert bone target
):
    i = wp.tid()
    # Leash origin on the current (deformed) bone.
    anchor_p = wp.mesh_eval_position(mesh, anchor_face[i], anchor_uv[i][0], anchor_uv[i][1])
    if reproject == 1:
        q = wp.mesh_query_point_sign_winding_number(mesh, muscle_pos[i], max_dist)
        if q.result:
            cand = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
            # Accept the slide only if it stays within slide_range of the anchor;
            # otherwise keep the old binding (caps cumulative tangential drift).
            if wp.length(cand - anchor_p) <= slide_range:
                bind_face[i] = q.face
                bind_uv[i] = wp.vec2(q.u, q.v)
                # Re-anchor the offset to the new face at the current muscle pos
                # (slide = re-bind with zero target error).
                Rc = _face_basis(mesh, q.face)
                local_offset[i] = wp.transpose(Rc) * (muscle_pos[i] - cand)
    f = bind_face[i]
    cp = wp.mesh_eval_position(mesh, f, bind_uv[i][0], bind_uv[i][1])
    R = _face_basis(mesh, f)
    target_out[i] = cp + R * local_offset[i]


@wp.kernel
def solve_m2m_glue_kernel(
    pair_a: wp.array(dtype=wp.int32),        # muscle vert A (global)
    pair_b: wp.array(dtype=wp.int32),        # muscle vert B (global, other muscle)
    restlen: wp.array(dtype=wp.float32),     # rest gap to hold
    compliance: wp.array(dtype=wp.float32),  # 1/stiffness
    mass: wp.array(dtype=wp.float32),
    stopped: wp.array(dtype=wp.int32),       # pinned vert = kinematic (wi=0)
    L: wp.array(dtype=wp.float32),
    dt: float,
    omega: float,
    pos: wp.array(dtype=wp.vec3),
):
    """Two-way XPBD distance spring between a pair of muscle verts on DIFFERENT
    muscles, holding their rest gap so adjacent muscles slide together instead of
    separating. Both endpoints live in the same muscle ``pos`` array; a pinned
    (``stopped``) endpoint contributes wi=0 so the pair pulls only the free side
    (mirrors the fascia glue)."""
    c = wp.tid()
    a = pair_a[c]
    b = pair_b[c]
    n = pos[a] - pos[b]
    d = wp.length(n)
    if d < 1.0e-9:
        return
    dirn = n / d
    C = d - restlen[c]
    wa = float(0.0)
    if stopped[a] == 0:
        ma = mass[a]
        wa = wp.where(ma > 0.0, 1.0 / ma, 0.0)
    wb = float(0.0)
    if stopped[b] == 0:
        mb = mass[b]
        wb = wp.where(mb > 0.0, 1.0 / mb, 0.0)
    wsum = wa + wb
    alpha_tilde = compliance[c] / (dt * dt)
    denom = wsum + alpha_tilde
    if denom < 1.0e-12:
        return
    dlambda = -(C + alpha_tilde * L[c]) / denom
    L[c] = L[c] + dlambda
    f = omega * dlambda
    wp.atomic_add(pos, a, (f * wa) * dirn)
    wp.atomic_add(pos, b, (-f * wb) * dirn)