"""Measure humanV2 muscle-id culling potential before building it.

Per frame, builds a per-muscle AABB (from that muscle's vertices) and counts how
many of the 199*198/2 muscle pairs actually overlap — plus the fraction of
surface triangles belonging to "active" muscles (>=1 overlapping partner).  The
broadphase emit cost scales with the triangles fed into the grid, so the active
fraction is the expected speedup ceiling of muscle-id culling.

    PYTHONPATH=src .venv/bin/python tools/muscle_overlap_probe.py --end_frame 20
"""
from __future__ import annotations
import argparse

import numpy as np
import warp as wp

wp.init()
from example.humanV2 import build_humanV2_sim_from_args, AnimatedBoneMeshCollider
from muscle.util.mesh_io import vectorized_surface_tris as _vectorized_surface_tris
from muscle.util.geo_io import GeoFile

DEV = "cuda:0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/humanV2/config/humanV2_gmg.json")
    ap.add_argument("--bone_dir", default="data/humanV2/model/bone")
    ap.add_argument("--rest", default="data/humanV2/model/muscle/restpos.geo")
    ap.add_argument("--end_frame", type=int, default=20)
    a = ap.parse_args()
    a.device = DEV; a.max_dist = 0.2
    a.no_collision = False; a.static_bone = False; a.no_export = True; a.geo_out = None

    g = GeoFile(a.rest)
    mid = g.prim_attr("muscle_id")                 # per-tet muscle index
    names = g.prim_attr_strings("muscle_id")
    M = len(names)
    print(f"muscles: {M}   tets: {mid.size}")

    sim, cfg, surf, fas = build_humanV2_sim_from_args(a)
    tets = sim.tet_idx.numpy()
    assert tets.shape[0] == mid.size, f"tet mismatch {tets.shape[0]} vs {mid.size}"

    # Per-muscle vertex index lists (union of its tets' verts).
    muscle_verts = [np.unique(tets[mid == m].ravel()) for m in range(M)]

    # Per-surface-triangle muscle id: a surf tri's 3 verts all belong to one
    # muscle (tets don't span muscles), so take the muscle of vert 0 via a
    # vertex->muscle map (any incident tet; muscles are vertex-disjoint here).
    vmuscle = np.full(sim.NV, -1, np.int32)
    for m in range(M):
        vmuscle[muscle_verts[m]] = m
    surf_tris = surf.reshape(-1, 3)
    tri_muscle = vmuscle[surf_tris[:, 0]]
    tri_counts = np.bincount(tri_muscle[tri_muscle >= 0], minlength=M)
    n_tris = surf_tris.shape[0]

    bone = AnimatedBoneMeshCollider(sim, a.bone_dir, max_dist=0.2, contact_offset=1e-5)
    bone.preload(range(sim.start_frame, sim.end_frame + 1))
    sim.collision_solvers = [bone]

    npairs_total = M * (M - 1) // 2
    print(f"surf tris: {n_tris}   muscle pairs: {npairs_total}\n")
    print(f"{'frame':>5}{'overlap pairs':>15}{'pair %':>9}"
          f"{'active muscles':>16}{'active tri %':>14}")
    for frame in range(sim.start_frame, sim.end_frame + 1):
        bone.set_frame(frame)
        sim.step_frame(frame)
        pos = sim.pos.numpy()
        lo = np.stack([pos[v].min(0) for v in muscle_verts])   # (M,3)
        hi = np.stack([pos[v].max(0) for v in muscle_verts])
        # AABB overlap matrix (broadcast): i,j overlap iff lo_i<=hi_j & lo_j<=hi_i all axes
        ov = (lo[:, None, :] <= hi[None, :, :]).all(-1) & (lo[None, :, :] <= hi[:, None, :]).all(-1)
        np.fill_diagonal(ov, False)
        n_ov = int(np.triu(ov, 1).sum())
        active = ov.any(1)                                     # muscle in >=1 pair
        active_tris = int(tri_counts[active].sum())
        if frame % 5 == 0 or frame == sim.start_frame or frame == sim.end_frame:
            print(f"{frame:>5}{n_ov:>15}{100*n_ov/npairs_total:>8.1f}%"
                  f"{int(active.sum()):>16}{100*active_tris/n_tris:>13.1f}%")


if __name__ == "__main__":
    main()
