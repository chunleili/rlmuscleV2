"""Build a per-muscle coarse tet mesh for GMG.

For each fine TetMesh prim under /Human/muscle, extract its boundary triangles,
decimate the surface via pymeshlab, then re-tetrahedralize via tetgen. Concat
all per-muscle coarse pieces into one mesh and save to .npz.

Usage:
    PYTHONPATH=src .venv/bin/python tools/build_coarse_per_muscle.py \
        --fine data/Human/Human.usd \
        --prim-root /Human/muscle \
        --out data/Human/coarse_per_muscle.npz \
        --target-faces 30 \
        --tet-quality 1.4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from muscle.mesh_io import build_surface_tris  # noqa: E402


def per_prim_decimate_and_tet(prim_pos, prim_tets, target_faces, tet_q,
                              tet_max_vol, name):
    """Surface-extract → pymeshlab decimate → tetgen tetrahedralize."""
    import pymeshlab
    import tetgen

    if len(prim_tets) == 0 or len(prim_pos) < 4:
        return None, None

    tris = build_surface_tris(prim_tets, prim_pos)
    if len(tris) < 4:
        return None, None

    # Drop unused verts; remap.
    used = np.unique(tris.flatten())
    remap = -np.ones(prim_pos.shape[0], dtype=np.int32)
    remap[used] = np.arange(len(used))
    surf_pos = prim_pos[used].astype(np.float64)
    surf_tris = remap[tris].astype(np.int32)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(surf_pos, surf_tris))
    n_faces = len(surf_tris)
    if n_faces > target_faces:
        try:
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=int(target_faces),
                preservenormal=True,
                preserveboundary=True,
                optimalplacement=True,
            )
        except Exception as e:
            print(f"  [{name}] decimation failed: {e}")
            return None, None
    # Repair after decimation.
    try:
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_repair_non_manifold_edges()
    except Exception:
        pass

    deci = ms.current_mesh()
    deci_pos = np.asarray(deci.vertex_matrix(), dtype=np.float64)
    deci_tris = np.asarray(deci.face_matrix(), dtype=np.int32)

    if len(deci_pos) < 4 or len(deci_tris) < 4:
        return None, None

    try:
        tg = tetgen.TetGen(deci_pos, deci_tris)
        # `Y` preserves boundary; `q` quality; `O` mesh optimization;
        # `a` max tet volume.
        switches = f"pYq{tet_q}O5"
        if tet_max_vol > 0:
            switches += f"a{tet_max_vol}"
        tg.tetrahedralize(switches=switches)
    except Exception as e:
        print(f"  [{name}] tetgen failed: {e}")
        return None, None

    coarse_pos = np.asarray(tg.node, dtype=np.float32)
    coarse_tets = np.asarray(tg.elem, dtype=np.int32)
    if len(coarse_pos) < 4 or len(coarse_tets) < 1:
        return None, None
    return coarse_pos, coarse_tets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fine", default="data/Human/Human.usd")
    p.add_argument("--prim-root", default="/Human/muscle")
    p.add_argument("--out", default="data/Human/coarse_per_muscle.npz")
    p.add_argument("--target-faces", type=int, default=24,
                   help="Target face count per muscle after decimation.")
    p.add_argument("--tet-quality", type=float, default=1.6,
                   help="tetgen min radius-edge ratio (lower = tighter quality).")
    p.add_argument("--tet-max-vol", type=float, default=0.0,
                   help="tetgen max tet volume (0=disabled, scales with mesh).")
    p.add_argument("--limit", type=int, default=0, help="Debug: process only N prims.")
    args = p.parse_args()

    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(args.fine)
    root = stage.GetPrimAtPath(args.prim_root)
    if not root.IsValid():
        raise SystemExit(f"prim root not found: {args.prim_root}")
    rest_t = stage.GetStartTimeCode()
    if rest_t == Usd.TimeCode.Default().GetValue():
        rest_t = 1.0
    xcache = UsdGeom.XformCache(rest_t)

    all_pos = []
    all_tets = []
    all_mid = []
    offset = 0
    t_total = time.perf_counter()
    n_done = 0
    n_skip = 0

    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.TetMesh):
            continue
        name = prim.GetName()
        tm = UsdGeom.TetMesh(prim)
        pts_local = np.asarray(tm.GetPointsAttr().Get(), dtype=np.float32)
        tets_local = np.asarray(tm.GetTetVertexIndicesAttr().Get(),
                                dtype=np.int32).reshape(-1, 4)
        wm = xcache.GetLocalToWorldTransform(prim)
        R = np.array([[wm[i][j] for j in range(3)] for i in range(3)], dtype=np.float32)
        T = np.array([wm[3][j] for j in range(3)], dtype=np.float32)
        prim_pos = (pts_local @ R + T).astype(np.float32)

        bbox_diag = np.linalg.norm(prim_pos.max(axis=0) - prim_pos.min(axis=0))
        max_vol = (bbox_diag / 6.0) ** 3 if args.tet_max_vol == 0 else args.tet_max_vol

        t0 = time.perf_counter()
        result = per_prim_decimate_and_tet(
            prim_pos, tets_local, args.target_faces,
            args.tet_quality, max_vol, name)
        if result[0] is None:
            n_skip += 1
            print(f"  [{name}] SKIPPED (degenerate)")
            continue
        cpos, ctets = result
        all_pos.append(cpos)
        all_tets.append((ctets + offset).astype(np.int32))
        all_mid.extend([str(name)] * len(ctets))
        offset += len(cpos)
        n_done += 1
        print(f"  [{name}] fine={len(prim_pos):5d}v/{len(tets_local):5d}t  "
              f"-> coarse={len(cpos):3d}v/{len(ctets):3d}t  ({time.perf_counter()-t0:.2f}s)")

        if args.limit and n_done >= args.limit:
            break

    if not all_pos:
        raise SystemExit("No coarse pieces built.")

    coarse_pos = np.vstack(all_pos)
    coarse_tets = np.vstack(all_tets)
    print(f"\nTotal: {n_done} muscles processed, {n_skip} skipped.")
    print(f"Coarse mesh: {coarse_pos.shape[0]} verts, {coarse_tets.shape[0]} tets "
          f"({time.perf_counter()-t_total:.1f}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             positions=coarse_pos.astype(np.float32),
             tet_indices=coarse_tets.astype(np.int32),
             muscle_id=np.array(all_mid, dtype=object))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
