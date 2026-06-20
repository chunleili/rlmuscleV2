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
from muscle.collision.udcf.solver import (
    ConservativeBoundCcd, UdcfDcdCollider, UdcfParams)

def main(argv=None):
    ap = add_humanV2_args(argparse.ArgumentParser())
    ap.add_argument("--usd", default="output/humanV2_sdf_bone_dcd_self.usd")
    ap.add_argument("--no_bone", action="store_true",
                    help="disable SDF muscle-vs-bone resolution")
    ap.add_argument("--no_gmg", action="store_true",
                    help="disable multigrid V-cycle (use_multigrid=false); pairs "
                         "with --tendon_pin_threshold for partial pinning")
    ap.add_argument("--tendon_pin_threshold", type=float, default=None,
                    help="PARTIAL pin: hard-pin only verts with tendonmask above "
                         "this; soft-attach the belly. None = full pin")
    ap.add_argument("--pin_soft_stiffness", type=float, default=None,
                    help="soft-pin fraction/substep for freed belly verts "
                         "(solver default 0.5)")
    ap.add_argument("--self_collision", action="store_true",
                    help="enable DCD muscle self-collision (OFF by default)")
    ap.add_argument("--thickness", type=float, default=1e-3,
                    help="DCD broadphase sphere slack (m)")
    ap.add_argument("--self_multiplier", type=float, default=0.3,
                    help="DCD collisionConstraintMultiplier for self-contact. "
                         "0.3 (soft) + a few --self_iters converges; the hard "
                         "1.01 single-push diverges in the pinned muscle context "
                         "(see the multiplier x iters sweep).")
    ap.add_argument("--self_iters", type=int, default=2,
                    help="DCD push iterations per substep (broadphase reused "
                         "across them). 2 balances depth vs cost; >4 re-diverges.")
    ap.add_argument("--mu_s", type=float, default=0.5)
    ap.add_argument("--mu_d", type=float, default=0.4)
    ap.add_argument("--self_max_pairs", type=int, default=1 << 22,
                    help="self broadphase pair cap (humanV2 muscle emits ~2.5M)")
    ap.add_argument("--self_max_contacts", type=int, default=1 << 18)
    ap.add_argument("--self_gauss_seidel", action="store_true",
                    help="serial GS self-resolve (stable under fold, single-"
                         "thread/slow); default jacobi at this vertex count")
    ap.add_argument("--self_ccd", action="store_true",
                    help="add conservative-bound CCD to self-collision: clamp "
                         "per-vertex displacement to ccd_relaxation x clearance "
                         "so verts can't tunnel through thin features")
    ap.add_argument("--ccd_relaxation", type=float, default=0.42,
                    help="CCD displacement bound = this x nearest-clearance")
    ap.add_argument("--ccd_query_radius", type=float, default=0.05,
                    help="CCD clearance search radius (m)")
    ap.add_argument("--self_normal_cull", type=float, default=1.0,
                    help="skip self pairs with n_i.n_j > this (1.0 disables, "
                         "0.0 culls co-oriented faces)")
    ap.add_argument("--self_hashgrid", action="store_true",
                    help="use the uniform-hashgrid broadphase instead of the "
                         "default wp.Bvh backend (BVH is ~4.5x faster here; see "
                         "UdcfParams.use_bvh)")
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
    # --- muscle self-collision: UDCF DCD (allow_self, no static body) ---------
    dcd = None
    if args.self_collision:
        ccd = ConservativeBoundCcd(relaxation=args.ccd_relaxation,
                                   query_radius=args.ccd_query_radius,
                                   device=args.device) if args.self_ccd else None
        dcd = UdcfDcdCollider(
            sim.pos, sim.inv_mass,
            surface_tris=surf_flat.reshape(-1, 3),
            ccd=ccd,
            params=UdcfParams(allow_self=True,
                              thickness=args.thickness,
                              multiplier=args.self_multiplier,
                              friction_static=args.mu_s,
                              friction_dynamic=args.mu_d,
                              max_pairs=args.self_max_pairs,
                              max_contacts=args.self_max_contacts,
                              gauss_seidel=args.self_gauss_seidel,
                              jacobi_avg=not args.self_gauss_seidel,
                              use_bvh=not args.self_hashgrid,
                              normal_cull=args.self_normal_cull),
            device=args.device)

    # Both colliders run in post_phase (the substep's final ops), NOT the
    # interleaved iter hook — interleaving lets the remaining ARAP iterations push
    # corrected verts back and spawn secondary self-crossings (max penetration
    # climbs).  Order is self DCD first, SDF bone LAST: self-before-bone keeps the
    # self relief (p99 -25%) while letting bone get the final word so muscle is
    # not pushed back into the skeleton (is_collide stays ~0 vs ~70 if bone ran
    # before self).  Per substep self does --self_iters soft pushes; dcd.solve(0)
    # rebuilds the broadphase (+ snapshots x_prev), solve(1) reuses the pairs.
    # Broadphase rebuilds once per frame (first substep) and is reused across the
    # rest (the 2.5M-pair grid is near-constant within a frame);
    # --self_rebuild_substeps rebuilds every substep.
    reset_self = None
    post = []
    if dcd is not None:
        state = {"fresh": True}
        K = max(1, args.self_iters)

        def self_fn(_st=state, _d=dcd, _K=K, _every=args.self_rebuild_substeps):
            rebuild = _st["fresh"] or _every
            _st["fresh"] = False
            _d.solve(0 if rebuild else 1)        # push 0 (+ broadphase if due)
            for _ in range(_K - 1):
                _d.solve(1)                       # extra pushes reuse the pairs

        def reset_self(frame, _st=state):
            _st["fresh"] = True

        post.append(self_fn)                     # self DCD first ...
    if not args.no_bone:
        post.append(bone)                        # ... SDF bone LAST (final word)
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
