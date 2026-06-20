"""End-to-end driver for the ARAP-HPBD one-way pin-to-target sim.

Usage:
    python -m example.oneway \
        --config data/lion/config/lion_nomg.json \
        --out output/lion_oneway.usd

    python -m example.oneway \
        --config data/minotaur/config/minotaur.json \
        --out output/minotaur_oneway.usd

    python -m example.oneway \
        --config data/minotaur_leg/config/minotaur_leg.json \
        --out output/minotaur_leg_oneway.usd
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp

from muscle.coupleOneWayGmg.solver import OneWaySim, load_config
from muscle.util.mesh_io import MeshExporter, vectorized_surface_tris


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/lion/config/lion_nomg.json")
    ap.add_argument("--out", default="output/lion_oneway.usd",
                    help="output USD file path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--end_frame", type=int, default=None,
                    help="override end_frame from config")
    ap.add_argument("--no_export", action="store_true",
                    help="don't write USD (timing-only run)")
    ap.add_argument("--split_fascia", action="store_true",
                    help="export muscle and fascia as separate USD prims")
    args = ap.parse_args()

    wp.init()
    cfg = load_config(args.config)
    if args.end_frame is not None:
        cfg["end_frame"] = args.end_frame

    sim = OneWaySim(cfg, device=args.device)
    print(f"NV={sim.NV}  NT={sim.NT}  NCONS={sim.NCONS}  "
          f"frames={sim.start_frame}..{sim.end_frame}  "
          f"nsubsteps={sim.nsubsteps}  fine_iters={sim.fine_iters}")

    nvm = getattr(sim, "nvm", sim.NV)
    split = args.split_fascia and nvm < sim.NV
    fps = int(cfg.get("fps", 24))
    # MeshExporter appends ".usd"; strip it from the requested path stem.
    stem = args.out[:-4] if args.out.endswith(".usd") else args.out

    exporter = None
    if not args.no_export:
        tets = sim.tet_idx.numpy()
        if split:
            muscle_tets = tets[(tets < nvm).all(axis=1)]
            fascia_tets = tets[(tets >= nvm).all(axis=1)]
            exporter = MeshExporter(path=stem, format="usd",
                                    tet_indices=muscle_tets, fps=fps,
                                    prim_path="/muscle")
            exporter.add_mesh("fascia", vectorized_surface_tris(fascia_tets))
            print(f"  split: {muscle_tets.shape[0]} muscle tets, "
                  f"{fascia_tets.shape[0]} fascia tets")
        else:
            exporter = MeshExporter(path=stem, format="usd",
                                    tet_indices=tets, fps=fps,
                                    prim_path="/muscle")

    n_frames = sim.end_frame - sim.start_frame + 1
    t0 = time.perf_counter()
    for frame in range(sim.start_frame, sim.end_frame + 1):
        sim.step_frame(frame)
        if exporter is not None:
            pos = sim.positions_np()
            exporter.save_frame(pos, frame)
            if split:
                exporter.save_mesh_frame("fascia", pos, frame)
        if frame % 10 == 0 or frame == sim.end_frame:
            elapsed = time.perf_counter() - t0
            done = frame - sim.start_frame + 1
            fps = done / elapsed if elapsed > 0 else 0.0
            print(f"  frame {frame:>3d}  {elapsed:6.1f}s  {fps:5.1f} fps avg")

    if exporter is not None:
        exporter.finalize()

    total = time.perf_counter() - t0
    print(f"done: {n_frames} frames in {total:.1f}s "
          f"({n_frames / total:.1f} fps avg)")


if __name__ == "__main__":
    main()
