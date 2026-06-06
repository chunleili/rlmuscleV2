"""End-to-end driver for the ARAP-HPBD one-way pin-to-target sim.

Usage:
    python -m muscle.coupleOneWayGmg.example \
        --config data/lion/config/lion_nomg.json \
        --out_dir result/lion

    python -m muscle.coupleOneWayGmg.example \
        --config data/minotaur/config/minotaur.json \
        --out_dir result/minotaur

    python -m muscle.coupleOneWayGmg.example \
        --config data/minotaur_leg/config/minotaur_leg.json \
        --out_dir result/minotaur_leg
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import warp as wp

from muscle.coupleOneWayGmg.solver import OneWaySim, load_config
from muscle.util.mesh_io import save_ply


def _vectorized_surface_tris(tets: np.ndarray) -> np.ndarray:
    """Extract boundary tris from a tet mesh — fast vectorized version.

    A face is on the surface iff it appears in exactly one tet. Faces are
    canonicalized by sorting their 3 vertex ids; the original (oriented)
    triangle is recovered by looking up which tet+face slot produced it.
    """
    NT = tets.shape[0]
    # Each tet has 4 faces; use the standard outward winding convention.
    face_slots = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int32)
    tris = tets[:, face_slots].reshape(-1, 3)        # (4*NT, 3) oriented
    keys = np.sort(tris, axis=1)                      # canonical key

    # Find unique canonical faces and how many tets share each.
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    boundary_mask = counts[inv] == 1
    return tris[boundary_mask].astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/lion/config/lion_nomg.json")
    ap.add_argument("--out_dir", default="result/lion")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--end_frame", type=int, default=None,
                    help="override end_frame from config")
    ap.add_argument("--no_export", action="store_true",
                    help="don't write PLY files (timing-only run)")
    ap.add_argument("--split_fascia", action="store_true",
                    help="export muscle and fascia as separate PLY streams")
    args = ap.parse_args()

    wp.init()
    cfg = load_config(args.config)
    if args.end_frame is not None:
        cfg["end_frame"] = args.end_frame

    sim = OneWaySim(cfg, device=args.device)
    print(f"NV={sim.NV}  NT={sim.NT}  NCONS={sim.NCONS}  "
          f"frames={sim.start_frame}..{sim.end_frame}  "
          f"nsubsteps={sim.nsubsteps}  fine_iters={sim.fine_iters}")

    out_dir = Path(args.out_dir)
    mesh_out = out_dir / "mesh"
    if not args.no_export:
        mesh_out.mkdir(parents=True, exist_ok=True)

    print("building surface triangulation...")
    t_surf = time.perf_counter()
    surface_faces = _vectorized_surface_tris(sim.tet_idx.numpy())
    print(f"  {surface_faces.shape[0]} surface tris  ({time.perf_counter()-t_surf:.2f}s)")

    nvm = getattr(sim, "nvm", sim.NV)
    split = args.split_fascia and nvm < sim.NV
    if split:
        tets = sim.tet_idx.numpy()
        muscle_tets = tets[(tets < nvm).all(axis=1)]
        fascia_tets = tets[(tets >= nvm).all(axis=1)]
        muscle_faces = _vectorized_surface_tris(muscle_tets)
        fascia_faces = _vectorized_surface_tris(fascia_tets)
        print(f"  split: {muscle_faces.shape[0]} muscle tris, "
              f"{fascia_faces.shape[0]} fascia tris")

    n_frames = sim.end_frame - sim.start_frame + 1
    t0 = time.perf_counter()
    for frame in range(sim.start_frame, sim.end_frame + 1):
        sim.step_frame(frame)
        if not args.no_export:
            if split:
                save_ply(str(mesh_out / f"muscle_{frame:04d}.ply"), sim.positions_np(), muscle_faces)
                save_ply(str(mesh_out / f"fascia_{frame:04d}.ply"), sim.positions_np(), fascia_faces)
            else:
                save_ply(str(mesh_out / f"{frame:04d}.ply"), sim.positions_np(), surface_faces)
        if frame % 10 == 0 or frame == sim.end_frame:
            elapsed = time.perf_counter() - t0
            done = frame - sim.start_frame + 1
            fps = done / elapsed if elapsed > 0 else 0.0
            print(f"  frame {frame:>3d}  {elapsed:6.1f}s  {fps:5.1f} fps avg")

    total = time.perf_counter() - t0
    print(f"done: {n_frames} frames in {total:.1f}s "
          f"({n_frames / total:.1f} fps avg)")


if __name__ == "__main__":
    main()
