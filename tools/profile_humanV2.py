"""Profile humanV2 collision per-frame time distribution.

Decomposes each frame into: sim core (multigrid + ARAP + volume), SDF bone
collider, and DCD self (split into broadphase vs narrow+respond+apply).  All
sections are wp.synchronize-bracketed so the GPU times are real, not queue
submits.

    PYTHONPATH=src .venv/bin/python tools/profile_humanV2.py --end_frame 20 [--self]
"""
from __future__ import annotations
import argparse
import time

import numpy as np
import warp as wp

wp.init()
from muscle.collision.sdf.example.humanV2 import build_humanV2_sim, AnimatedBoneMeshCollider
from muscle.collision.udcf.solver import UdcfDcdCollider, UdcfParams
from muscle.collision.udcf.kernel import (centroid_radius_kernel, max_radius_reduce_kernel,
                                          broadphase_find_pairs_kernel)

DEV = "cuda:0"


def make_timed_broadphase(dcd, acc, timed):
    """Replica of UdcfDcdCollider._broadphase with each step bracketed."""
    d = dcd; p = d.params

    def bp():
        timed(acc, "bp_centroid", lambda: wp.launch(
            centroid_radius_kernel, dim=d.n_tris,
            inputs=[d.tris, d.x, d.centroids, d.radii], device=d.device))
        timed(acc, "bp_maxreduce", lambda: wp.launch(
            max_radius_reduce_kernel, dim=1,
            inputs=[d.radii, d._max_radius], device=d.device))
        cell = max(2.0 * float(d._max_radius.numpy()[0]) + p.thickness, 1.0e-4)
        timed(acc, "bp_gridbuild", lambda: d.grid.build(d.centroids, cell))
        d.n_pairs.zero_()
        timed(acc, "bp_emit", lambda: wp.launch(
            broadphase_find_pairs_kernel, dim=d.n_tris,
            inputs=[d.grid.id, d.centroids, d.radii, d.tris, d.tri_group,
                    d.normals, wp.float32(p.normal_cull), wp.int32(int(p.allow_self)),
                    wp.float32(cell), wp.float32(p.thickness), wp.int32(p.max_pairs),
                    d.pair_a, d.pair_b, d.n_pairs], device=d.device))
    return bp


class _Acc:
    def __init__(self): self.t = {}; self.n = 0
    def add(self, k, dt): self.t[k] = self.t.get(k, 0.0) + dt
    def __getitem__(self, k): return self.t.get(k, 0.0)
    def reset(self): self.t.clear(); self.n = 0   # in place, keeps closures valid


def timed(acc, key, fn):
    wp.synchronize(); t0 = time.perf_counter()
    r = fn()
    wp.synchronize(); acc.add(key, time.perf_counter() - t0)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/humanV2/config/humanV2_gmg.json")
    ap.add_argument("--bone_dir", default="data/humanV2/model/bone")
    ap.add_argument("--end_frame", type=int, default=20)
    ap.add_argument("--self", dest="self_on", action="store_true",
                    help="enable DCD self-collision in the profile")
    ap.add_argument("--warmup", type=int, default=2)
    a = ap.parse_args()
    a.device = DEV; a.max_dist = 0.2
    a.no_collision = False; a.static_bone = False; a.no_export = True; a.geo_out = None

    sim, cfg, surf, fas = build_humanV2_sim(a)
    bone = AnimatedBoneMeshCollider(sim, a.bone_dir, max_dist=0.2, contact_offset=1e-5)
    bone.preload(range(sim.start_frame, sim.end_frame + 1))

    acc = _Acc()

    # Wrap the SDF bone collider so its push time is attributed separately.
    def bone_timed():
        timed(acc, "bone", lambda: bone.__call__())
    sim.collision_solvers = [bone_timed]

    dcd = None
    if a.self_on:
        dcd = UdcfDcdCollider(sim.pos, sim.inv_mass, surface_tris=surf.reshape(-1, 3),
                              params=UdcfParams(allow_self=True, thickness=1e-3,
                                                max_pairs=1 << 22), device=DEV)
        timed_bp = make_timed_broadphase(dcd, acc, timed)
        fresh = {"v": True}

        def solve_timed(it=0):
            # broadphase once per frame (the demo default); time it (broken into
            # sub-steps inside timed_bp) apart from narrow+respond+apply.
            do_bp = fresh["v"]
            fresh["v"] = False
            orig = dcd._broadphase
            dcd._broadphase = timed_bp if do_bp else (lambda: None)
            timed(acc, "self_solve", lambda: UdcfDcdCollider.solve(dcd, 0 if do_bp else 1))
            dcd._broadphase = orig
        sim.collision_solvers = [solve_timed, bone_timed]

    frames = range(sim.start_frame, sim.end_frame + 1)
    for fi, frame in enumerate(frames):
        if fi == a.warmup:                 # discard warmup, keep timing live
            acc.reset()
        if dcd is not None:
            fresh["v"] = True              # allow one broadphase this frame
        bone.set_frame(frame)
        wp.synchronize(); t0 = time.perf_counter()
        sim.step_frame(frame)
        wp.synchronize(); acc.add("frame_total", time.perf_counter() - t0)
        acc.n += 1

    n = max(acc.n, 1)
    bone_t = acc["bone"]; self_t = acc["self_solve"]
    bp = (acc["bp_centroid"] + acc["bp_maxreduce"] + acc["bp_gridbuild"] + acc["bp_emit"])
    total = acc["frame_total"]
    core = total - bone_t - self_t
    narrow = self_t - bp

    def row(name, t):
        print(f"  {name:<30}{t/n*1e3:8.1f} ms/f   {100*t/total:5.1f}%")
    print(f"\nhumanV2 profile  ({n} frames, self={'ON' if a.self_on else 'OFF'})")
    print(f"  {'section':<30}{'per-frame':>11}{'':>9}share")
    row("frame TOTAL", total)
    row("  sim core (mg+ARAP+vol)", core)
    row("  SDF bone collider", bone_t)
    if a.self_on:
        row("  DCD self TOTAL", self_t)
        row("    broadphase TOTAL", bp)
        row("      bp centroid+radius", acc["bp_centroid"])
        row("      bp max-radius reduce", acc["bp_maxreduce"])
        row("      bp grid.build", acc["bp_gridbuild"])
        row("      bp emit pairs", acc["bp_emit"])
        row("    narrow+resp+apply", narrow)
        print(f"\n  self n_pairs={dcd.stats()['n_pairs']}  n_contacts={dcd.stats()['n_contacts']}")


if __name__ == "__main__":
    main()
