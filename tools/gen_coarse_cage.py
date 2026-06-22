"""Generate a per-muscle voxel embedding cage for the humanV2 multigrid solver.

The original ``restpos_lowres.geo`` is a single surface-fitted decimation: it is
NOT an enclosing cage. Measured against ``restpos.geo``, 61.8% of fine muscle
verts fall OUTSIDE every coarse tet, so prolongation extrapolates them (barycentric
weights up to +-32x) and 21.5% draw from a tet belonging to a DIFFERENT muscle.
Under deformation that |weight|~32 amplification is exactly the gmg "尖刺" (spikes).

This builds the cage the multigrid actually needs: per muscle, voxelize the
muscle's fine verts and split every occupied voxel into 6 tets. Every fine vert
lies in its own voxel by construction, so it is contained in a same-muscle coarse
tet with barycentric weights in [0,1] -- no extrapolation, no cross-muscle bleed.

Both transfer operators are baked here (rest pose, once) so the solver just loads them:
  * f->c (prolong P):  fine vert -> containing coarse tet + barycentric (weights in [0,1])
  * c->f (restrict):   Galerkin least-squares, Cdef = (P^T P)^-1 (P^T T) -- the
                       smoothest cage deformation reproducing the bone target. Measured
                       ~6x less injected ARAP strain than a nearest-vert gather (which
                       jitters cage corners), and in-solver it converges 3-6x faster
                       than non-MG. We bake the per-muscle dense block Minv=pinv(P^T P).

    PYTHONPATH=src uv run python tools/gen_coarse_cage.py \
        --rest data/humanV2/model/muscle/restpos.geo \
        --out  data/humanV2/model/muscle/coarse_cage.npz \
        --voxel 0.045
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import warp as wp

wp.init()
from muscle.util.geo_io import GeoFile
from muscle.coupleOneWayGmg.solver import compute_barycentric_kernel

DEV = "cuda:0"

# 6-tet split of a unit cube. Local corner b in 0..7 has offset bits (x,y,z) =
# (b&1, (b>>1)&1, (b>>2)&1); these 6 tets tile the cube with no gaps/overlaps.
CUBE = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7),
        (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]


def _oriented_surface(P, tets_local):
    """Boundary triangles of a tet mesh, oriented outward (normal away from the
    opposite/interior vertex). Returns (surf_verts_global_to_P_idx, V, F_local)."""
    f = np.concatenate([tets_local[:, [1, 2, 3]], tets_local[:, [0, 2, 3]],
                        tets_local[:, [0, 1, 3]], tets_local[:, [0, 1, 2]]])
    opp = np.concatenate([tets_local[:, 0], tets_local[:, 1],
                          tets_local[:, 2], tets_local[:, 3]])
    key = np.sort(f, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bnd = cnt[inv] == 1
    f, opp = f[bnd], opp[bnd]
    n = np.cross(P[f[:, 1]] - P[f[:, 0]], P[f[:, 2]] - P[f[:, 0]])
    inward = np.einsum('ij,ij->i', n, P[opp] - P[f[:, 0]]) > 0
    f[inward] = f[inward][:, [0, 2, 1]]
    sv = np.unique(f)
    rem = -np.ones(P.shape[0], np.int64); rem[sv] = np.arange(sv.size)
    return P[sv], rem[f].astype(np.int32)


def _muscle_cage_inflate(P, tets_local, inflate, edge_r, eps):
    """Anatomy-conforming coarse cage: take the muscle's surface, push it OUT
    along vertex normals by `inflate`*bbox-diag, then coarsen+tetrahedralize with
    fTetWild. The result is shaped like the muscle (so coarse ARAP/volume relaxes
    toward the right shape, not a voxel block) yet ENCLOSES the fine verts."""
    import wildmeshing as wm
    V, F = _oriented_surface(P, tets_local)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    diag = np.linalg.norm(V.max(0) - V.min(0))
    Vi = V + vn * (inflate * diag)
    t = wm.Tetrahedralizer(edge_length_r=edge_r, epsilon=eps, coarsen=True)
    t.set_log_level(6)
    fd = os.open(os.devnull, os.O_WRONLY); old = os.dup(1); os.dup2(fd, 1)
    try:
        t.set_mesh(Vi, F)
        t.tetrahedralize()
        out = t.get_tet_mesh()
    finally:
        os.dup2(old, 1); os.close(fd); os.close(old)
    return np.asarray(out[0]), np.asarray(out[1]).astype(np.int32)


def _muscle_cage(P: np.ndarray, h: float):
    """DEPRECATED (kept for reference / fallback). Blocky voxel cage for one
    muscle's fine verts P. Returns (coarse_V, coarse_T).

    Superseded by ``_muscle_cage_inflate`` (--method inflate, the default): the
    voxel cage is a box that encloses 100% of the fine verts but its rest shape
    is a block, not the muscle. That is fine for the projection-only path
    (Galerkin restrict), where it measured marginally better than inflate, but it
    cannot host a coarse ARAP/volume relaxation — relaxing pulls the cage toward
    its block rest shape and corrupts the prolonged start (measured ~100x worse
    fine residual). The inflate cage is muscle-shaped, so it stays usable if a
    future fiber-aware coarse solve wants to relax, not just project."""
    mn = P.min(0)
    ijk = np.floor((P - mn) / h).astype(np.int64)
    occ = np.unique(ijk, axis=0)
    cmap: dict = {}
    cverts: list = []

    def cid(c):
        t = (int(c[0]), int(c[1]), int(c[2]))
        j = cmap.get(t)
        if j is None:
            j = len(cverts)
            cmap[t] = j
            cverts.append(mn + np.array(c, dtype=np.float64) * h)
        return j

    tets = []
    for v in occ:
        i, j, k = int(v[0]), int(v[1]), int(v[2])
        cs = [cid((i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)))
              for b in range(8)]
        for a, b, c, d in CUBE:
            tets.append((cs[a], cs[b], cs[c], cs[d]))
    return np.asarray(cverts, dtype=np.float64), np.asarray(tets, dtype=np.int32)


def _barycentric(query: np.ndarray, cage_pos: np.ndarray, cage_tet: np.ndarray):
    """For each query point, the containing cage tet (nearest-tet fallback) + bary."""
    n = query.shape[0]
    q = wp.array(query.astype(np.float32), dtype=wp.vec3, device=DEV)
    cp = wp.array(cage_pos.astype(np.float32), dtype=wp.vec3, device=DEV)
    ct = wp.array(cage_tet.astype(np.int32), dtype=wp.vec4i, device=DEV)
    idx = wp.zeros(n, dtype=wp.int32, device=DEV)
    bary = wp.zeros(n, dtype=wp.vec3, device=DEV)
    wp.launch(compute_barycentric_kernel, dim=n,
              inputs=[q, cp, ct, idx, bary], device=DEV)
    wp.synchronize()
    return idx.numpy(), bary.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rest", default="data/humanV2/model/muscle/restpos.geo")
    ap.add_argument("--out", default="data/humanV2/model/muscle/coarse_cage.npz")
    ap.add_argument("--method", choices=["voxel", "inflate"], default="inflate",
                    help="inflate (default) = anatomy-conforming bounding cage "
                         "(muscle-shaped, projection + future relax). "
                         "voxel = DEPRECATED blocky cage, projection-only fallback")
    ap.add_argument("--voxel", type=float, default=0.045,
                    help="[voxel, DEPRECATED] cage voxel edge length (metres)")
    ap.add_argument("--inflate", type=float, default=0.02,
                    help="[inflate] outward push as fraction of muscle bbox diag")
    ap.add_argument("--edge_r", type=float, default=0.15,
                    help="[inflate] fTetWild ideal edge length / bbox diag")
    ap.add_argument("--eps", type=float, default=0.01,
                    help="[inflate] fTetWild envelope / bbox diag")
    a = ap.parse_args()

    g = GeoFile(a.rest)
    fp = g.positions.astype(np.float64)
    ft = g.tet_indices.astype(np.int64)
    mid = g.prim_attr("muscle_id")
    muscles = np.unique(mid)
    NV = fp.shape[0]
    print(f"fine: NV={NV} NT={ft.shape[0]} muscles={len(muscles)} "
          f"method={a.method} ({'voxel=%.3f' % a.voxel if a.method=='voxel' else 'inflate=%.3f edge_r=%.2f' % (a.inflate, a.edge_r)})")

    # Global coarse mesh + baked transfer maps.
    coarse_pos = []                       # list of (nv_m,3) per muscle
    coarse_tet = []                       # list of (nt_m,4) per muscle (global ids)
    coarse_muscle = []                    # per coarse tet -> fine muscle index
    cage_idx = np.full(NV, -1, np.int32)  # f->c: fine vert -> global coarse tet
    uvw = np.zeros((NV, 3), np.float32)
    # Galerkin least-squares restrict: per muscle Minv = pinv(P^T P) (cageV x cageV
    # dense block). Restrict Cdef = Minv (P^T T) is the projection of the bone
    # target onto the cage's deformation subspace -> the smoothest cage that
    # reproduces it (gather restrict instead jitters corners -> high strain).
    minv_flat = []                        # concat of per-muscle Minv.ravel()
    blk_base = []                         # per coarse vert: first coarse vert of its block
    blk_dim = []                          # per coarse vert: block size (cageV_m)
    blk_matoff = []                       # per coarse vert: offset into minv_flat
    coarse_tendon = []                    # per coarse vert: tendon fraction [0,1]
    tendonmask = (g.point_attr("tendonmask").astype(np.float64)
                  if g.has_point_attr("tendonmask")
                  else np.zeros(NV, np.float64))

    v_off = 0
    t_off = 0
    mat_off = 0
    t0 = time.time()
    for m in muscles:
        tmask = mid == m
        m_tets = ft[tmask]                         # (nt_m,4) global vert ids
        m_verts = np.unique(m_tets)                # global vert ids in muscle
        P = fp[m_verts]
        remap = -np.ones(NV, np.int64); remap[m_verts] = np.arange(m_verts.size)

        if a.method == "inflate":
            Vc, Tc_local = _muscle_cage_inflate(P, remap[m_tets], a.inflate,
                                                a.edge_r, a.eps)
        else:
            Vc, Tc_local = _muscle_cage(P, a.voxel)
        nc = Vc.shape[0]

        # f->c: each fine vert of the muscle -> containing coarse tet (in [0,1]).
        idx_fc, bary_fc = _barycentric(P, Vc, Tc_local)
        cage_idx[m_verts] = (t_off + idx_fc).astype(np.int32)
        uvw[m_verts] = bary_fc.astype(np.float32)

        # Prolong matrix P_m (fineV_m x nc) from the same weights, then the
        # Galerkin normal-matrix inverse. pinv tolerates cage corners that no
        # fine vert references (zero columns) -> their Cdef is irrelevant (prolong
        # never reads them).
        corners = Tc_local[idx_fc]                 # (nv_m, 4) local cage verts
        w4 = np.column_stack([1 - bary_fc.sum(1), bary_fc])   # (nv_m, 4)
        Pm = np.zeros((P.shape[0], nc))
        rows = np.repeat(np.arange(P.shape[0]), 4)
        np.add.at(Pm, (rows, corners.reshape(-1)), w4.reshape(-1))
        Minv = np.linalg.pinv(Pm.T @ Pm, rcond=1e-8)
        minv_flat.append(Minv.ravel().astype(np.float32))
        blk_base.append(np.full(nc, v_off, np.int32))
        blk_dim.append(np.full(nc, nc, np.int32))
        blk_matoff.append(np.full(nc, mat_off, np.int32))
        mat_off += nc * nc

        # Per coarse vert: tendon fraction of its nearest fine vert. Drives the
        # coarse attach stiffness (stiff at tendons = bone anchor, soft at belly)
        # so the coarse ARAP/volume smoother solves the SAME pinned BVP as fine.
        near = ((Vc[:, None, :] - P[None, :, :]) ** 2).sum(2).argmin(1)
        coarse_tendon.append(tendonmask[m_verts[near]].astype(np.float32))

        coarse_pos.append(Vc)
        coarse_tet.append(Tc_local + v_off)
        coarse_muscle.append(np.full(Tc_local.shape[0], m, np.int32))
        v_off += nc
        t_off += Tc_local.shape[0]

    coarse_pos = np.concatenate(coarse_pos).astype(np.float32)
    coarse_tet = np.concatenate(coarse_tet).astype(np.int32)
    coarse_muscle = np.concatenate(coarse_muscle)
    coarse_tendon = np.concatenate(coarse_tendon)
    minv_flat = np.concatenate(minv_flat)
    blk_base = np.concatenate(blk_base)
    blk_dim = np.concatenate(blk_dim)
    blk_matoff = np.concatenate(blk_matoff)

    # ---- verification (rest pose) ----
    W_fc = np.column_stack([1 - uvw.sum(1), uvw])
    contained = ((W_fc >= -1e-4).all(1) & (W_fc <= 1 + 1e-4).all(1))
    print(f"coarse: NV={coarse_pos.shape[0]} NT={coarse_tet.shape[0]}  "
          f"({time.time()-t0:.1f}s)")
    print(f"  f->c containment (fine verts inside coarse): "
          f"{contained.mean()*100:.2f}%  w[{W_fc.min():.3f},{W_fc.max():.3f}]")
    print(f"  Galerkin Minv blocks: {len(np.unique(blk_matoff))} muscles, "
          f"{minv_flat.size} floats ({minv_flat.nbytes/1e6:.1f} MB), "
          f"max block dim={int(blk_dim.max())}")
    assert (cage_idx >= 0).all(), "some fine vert got no coarse tet"

    np.savez(a.out,
             coarse_pos=coarse_pos, coarse_tet=coarse_tet,
             coarse_muscle=coarse_muscle, coarse_tendon=coarse_tendon,
             cage_idx=cage_idx, uvw=uvw,
             minv_flat=minv_flat, blk_base=blk_base,
             blk_dim=blk_dim, blk_matoff=blk_matoff,
             mg_fine_nv=np.int64(NV), voxel=np.float64(a.voxel))
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
