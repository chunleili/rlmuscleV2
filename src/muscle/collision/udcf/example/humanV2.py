"""humanV2 collision: SDF for muscle-vs-bone + UDCF DCD for muscle self.

Split by what each method is good at (established empirically — see the commit
that added this file):

  * muscle-vs-bone is a VOLUMETRIC overlap of the muscle against a penetrating
    animated target.  The SDF point-projection collider
    (``muscle.collision.sdf.example.humanV2.AnimatedBoneMeshCollider``) resolves
    it cleanly — it hard-projects each submerged vertex to the bone surface.  A
    surface tri-tri method (DCD) only ever sees the thin crossing rim, so it
    cannot clear the volume no matter how it is tuned (jacobi/GS, interleaved/
    last, multiplier, clamp all tried — count drops but depth grows).

  * muscle-vs-muscle self-collision IS a surface-crossing problem with no SDF
    available (deformable vs deformable), which is exactly the UDCF DCD
    (:class:`UdcfDcdCollider`, ``allow_self=True``) domain.

Wiring:
  * SDF bone collider on ``collision_solvers`` (post_phase) — the substep's
    FINAL word, hard-projecting muscle out of the moving skeleton.
  * DCD self collider on ``collision_solvers_iter`` (per-iteration) — interleaves
    self-contact with ARAP/volume; static body is a dummy far-away triangle so
    ONLY movable-vs-movable (muscle self) pairs are emitted.

The SDF collider also doubles as the penetration METER (its ``inside_depth``
drives the USD ``is_collide`` / ``pen_depth`` primvars), shared via
``run_humanV2_loop``.

Run::

    PYTHONPATH=src .venv/bin/python -m muscle.collision.udcf.example.humanV2 \\
        --config data/humanV2/config/humanV2_gmg.json \\
        --bone_dir data/humanV2/model/bone --end_frame 20 \\
        --usd output/humanV2_sdf_bone_dcd_self.usd
"""
from __future__ import annotations

import argparse

import numpy as np
import warp as wp

from muscle.collision.sdf.example.humanV2 import (
    AnimatedBoneMeshCollider,
    add_humanV2_args,
    build_humanV2_sim,
    run_humanV2_loop,
)
from muscle.collision.udcf.solver import UdcfDcdCollider, UdcfParams

# Far-away degenerate static mesh: UdcfDcdCollider requires a static body, but
# in self-only mode we want NO muscle-vs-static pairs — placing it at 1e4 keeps
# it out of every broadphase cell.
_DUMMY_STATIC_V = np.array([[1e4, 1e4, 1e4], [1e4 + 1, 1e4, 1e4],
                            [1e4, 1e4, 1e4 + 1]], np.float32)
_DUMMY_STATIC_T = np.array([[0, 1, 2]], np.int32)


def main(argv=None):
    ap = add_humanV2_args(argparse.ArgumentParser())
    ap.add_argument("--usd", default="output/humanV2_sdf_bone_dcd_self.usd")
    ap.add_argument("--no_bone", action="store_true",
                    help="disable SDF muscle-vs-bone resolution")
    ap.add_argument("--self_collision", action="store_true",
                    help="enable DCD muscle self-collision (OFF by default)")
    ap.add_argument("--thickness", type=float, default=1e-3,
                    help="DCD broadphase sphere slack (m)")
    ap.add_argument("--self_multiplier", type=float, default=1.01,
                    help="DCD collisionConstraintMultiplier for self-contact")
    ap.add_argument("--mu_s", type=float, default=0.5)
    ap.add_argument("--mu_d", type=float, default=0.4)
    ap.add_argument("--self_max_pairs", type=int, default=1 << 22,
                    help="self broadphase pair cap (humanV2 muscle emits ~2.5M)")
    ap.add_argument("--self_max_contacts", type=int, default=1 << 18)
    ap.add_argument("--self_gauss_seidel", action="store_true",
                    help="serial GS self-resolve (stable under fold, single-"
                         "thread/slow); default jacobi at this vertex count")
    ap.add_argument("--self_per_substep", action="store_true",
                    help="run DCD self once per SUBSTEP (post_phase) instead of "
                         "per fine-iteration — fewer response launches, looser "
                         "convergence")
    ap.add_argument("--self_normal_cull", type=float, default=1.0,
                    help="skip self pairs with n_i.n_j > this (1.0 disables, "
                         "0.0 culls co-oriented faces)")
    ap.add_argument("--self_bvh", action="store_true",
                    help="use the wp.Bvh broadphase backend for self-collision "
                         "instead of the default hashgrid (generally slower on "
                         "this mesh — rebuild cost; see UdcfParams.use_bvh)")
    ap.add_argument("--self_rebuild_substeps", action="store_true",
                    help="rebuild the self broadphase EVERY call (~10x slower). "
                         "Default rebuilds once per FRAME and reuses pairs across "
                         "substeps — muscle deforms little within a frame, so the "
                         "10 per-frame rebuilds of the 2.5M-pair grid are wasted.")
    args = ap.parse_args(argv)
    # --no_collision (shared flag) disables BOTH resolvers.
    if args.no_collision:
        args.no_bone = True
        args.self_collision = False

    wp.init()
    sim, cfg, surf_flat, fascia_flat = build_humanV2_sim(args)

    # --- muscle-vs-bone: SDF collider (proven resolver + penetration meter) ---
    bone = AnimatedBoneMeshCollider(sim, args.bone_dir, max_dist=args.max_dist,
                                    contact_offset=args.contact_offset)
    if not args.static_bone:
        bone.preload(range(sim.start_frame, sim.end_frame + 1))
    # --- muscle self-collision: UDCF DCD (allow_self, dummy static) -----------
    dcd = None
    if args.self_collision:
        dcd = UdcfDcdCollider(
            sim.pos, sim.inv_mass,
            surface_tris=surf_flat.reshape(-1, 3),
            static_verts=_DUMMY_STATIC_V, static_tris=_DUMMY_STATIC_T,
            params=UdcfParams(allow_self=True,
                              thickness=args.thickness,
                              multiplier=args.self_multiplier,
                              friction_static=args.mu_s,
                              friction_dynamic=args.mu_d,
                              max_pairs=args.self_max_pairs,
                              max_contacts=args.self_max_contacts,
                              gauss_seidel=args.self_gauss_seidel,
                              jacobi_avg=not args.self_gauss_seidel,
                              use_bvh=args.self_bvh,
                              normal_cull=args.self_normal_cull),
            device=args.device)

    # Self broadphase rebuild cadence.  dcd.solve(i_iter): i_iter==0 rebuilds the
    # broadphase (+ snapshots x_prev), i_iter>0 reuses the existing pairs.  By
    # default we rebuild ONCE per frame (first self call) and reuse across the
    # frame's substeps — a ~10x win, since the 2.5M-pair grid is near-constant
    # within a frame.  --self_rebuild_substeps forces a rebuild every call.
    self_fn = dcd.solve if dcd is not None else None
    reset_self = None
    if dcd is not None and not args.self_rebuild_substeps:
        state = {"fresh": True}

        def self_fn(it=0, _st=state, _d=dcd):
            _d.solve(0 if _st["fresh"] else 1)
            _st["fresh"] = False

        def reset_self(frame, _st=state):
            _st["fresh"] = True

    # post_phase runs once per substep; bone (SDF) must be the FINAL word so it
    # appends last.  Per-substep self goes ahead of it; per-fine-iter self uses
    # the interleaved hook instead.
    post = []
    if self_fn is not None and args.self_per_substep:
        post.append(self_fn)
    elif self_fn is not None:
        sim.collision_solvers_iter = [self_fn]
    if not args.no_bone:
        post.append(bone)
    if post:
        sim.collision_solvers = post

    print(f"humanV2 collision: bone={'SDF' if not args.no_bone else 'OFF'}  "
          f"self={'DCD' if args.self_collision else 'OFF'}  "
          f"({'STATIC@f1' if args.static_bone else 'ANIMATED'} skeleton, "
          f"{bone.n_bone_verts} bone verts)")

    run_humanV2_loop(sim, bone, args, cfg, surf_flat, fascia_flat,
                     pre_step=reset_self)
    if dcd is not None:
        print(f"DCD self final: {dcd.stats()}")


if __name__ == "__main__":
    main()
