"""humanV2 collision: SDF for muscle-vs-bone + closest-point for muscle-self.

Drop-in alternative to the UDCF DCD self-collision demo
(``muscle.collision.udcf.example.humanV2``).  Instead of surface tri-tri DCD it
resolves muscle self-collision as INTER-MUSCLE only: each of the 199 muscles is a
closed body, and a vertex inside another muscle is pushed to that muscle's
closest surface point (``InterMuscleCollider``) — the SDF-bone per-vertex push
generalised to a deformable closest-point target.  Intra-muscle self-folding is
ignored (rare, internal), which removes the connectivity/k-ring filtering the
tri-tri DCD needed and could not converge.

Wiring (same ordering rationale as the DCD demo): inter-muscle push FIRST, SDF
bone LAST (final word, so muscle is not shoved back into the skeleton).  The SDF
bone collider doubles as the penetration meter for ``is_collide`` / ``pen_depth``.

Run::

    PYTHONPATH=src .venv/bin/python -m muscle.collision.closest_point.example.humanV2_intermuscle \\
        --config data/humanV2/config/humanV2_gmg.json \\
        --bone_dir data/humanV2/model/bone --end_frame 20 --self_collision \\
        --usd output/humanV2_sdf_bone_intermuscle.usd
"""
from __future__ import annotations

import argparse

import warp as wp

from muscle.collision.sdf.example.humanV2 import (
    AnimatedBoneMeshCollider,
    add_humanV2_args,
    build_humanV2_sim,
    run_humanV2_loop,
)
from muscle.collision.closest_point.intermuscle import InterMuscleCollider


def main(argv=None):
    ap = add_humanV2_args(argparse.ArgumentParser())
    ap.add_argument("--usd", default="output/humanV2_sdf_bone_intermuscle.usd")
    ap.add_argument("--no_bone", action="store_true",
                    help="disable SDF muscle-vs-bone resolution")
    ap.add_argument("--self_collision", action="store_true",
                    help="enable inter-muscle self-collision (OFF by default)")
    ap.add_argument("--self_max_dist", type=float, default=0.02,
                    help="inter-muscle closest-point search radius (m); a vertex "
                         "deeper than this past another muscle's surface is missed")
    ap.add_argument("--self_offset", type=float, default=1e-4,
                    help="keep pushed verts this far outside the other muscle (m)")
    ap.add_argument("--self_iters", type=int, default=1,
                    help="jacobi push passes per substep. 1 is best: the ~10 "
                         "per-substep refits already iterate; K>1 reuses the work "
                         "list WITHOUT refitting between passes, so the stale "
                         "target surfaces make extra passes overshoot/diverge.")
    args = ap.parse_args(argv)
    if args.no_collision:
        args.no_bone = True
        args.self_collision = False

    wp.init()
    sim, cfg, surf_flat, fascia_flat = build_humanV2_sim(args)

    bone = AnimatedBoneMeshCollider(sim, args.bone_dir, max_dist=args.max_dist,
                                    contact_offset=args.contact_offset)
    if not args.static_bone:
        bone.preload(range(sim.start_frame, sim.end_frame + 1))

    inter = None
    if args.self_collision:
        inter = InterMuscleCollider(sim, cfg["rest_path"],
                                    max_dist=args.self_max_dist,
                                    contact_offset=args.self_offset,
                                    iters=args.self_iters, device=args.device)
        print(f"inter-muscle collider: {inter.n_muscle} muscles, "
              f"max_dist={args.self_max_dist} offset={args.self_offset}")

    post = []
    if inter is not None:
        post.append(inter)            # inter-muscle push first ...
    if not args.no_bone:
        post.append(bone)             # ... SDF bone LAST (final word)
    if post:
        sim.collision_solvers = post

    print(f"humanV2 collision: bone={'SDF' if not args.no_bone else 'OFF'}  "
          f"self={'INTER-MUSCLE' if args.self_collision else 'OFF'}  "
          f"({'STATIC@f1' if args.static_bone else 'ANIMATED'} skeleton)")

    pre_step = (lambda frame: inter.refresh()) if inter is not None else None
    run_humanV2_loop(sim, bone, args, cfg, surf_flat, fascia_flat,
                     pre_step=pre_step)
    if inter is not None:
        print(f"inter-muscle final inside_count={inter.inside_count()}  "
              f"pairs={inter.last_pairs}")


if __name__ == "__main__":
    main()
