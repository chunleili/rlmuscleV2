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
    ap.add_argument("--no_self", action="store_true",
                    help="disable DCD muscle self-collision")
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
    args = ap.parse_args(argv)
    # --no_collision (shared flag) disables BOTH resolvers.
    if args.no_collision:
        args.no_bone = args.no_self = True

    wp.init()
    sim, cfg, surf_flat, fascia_flat = build_humanV2_sim(args)

    # --- muscle-vs-bone: SDF collider (proven resolver + penetration meter) ---
    bone = AnimatedBoneMeshCollider(sim, args.bone_dir, max_dist=args.max_dist,
                                    contact_offset=args.contact_offset)
    if not args.static_bone:
        bone.preload(range(sim.start_frame, sim.end_frame + 1))
    if not args.no_bone:
        sim.collision_solvers = [bone]          # post_phase: final word

    # --- muscle self-collision: UDCF DCD (allow_self, dummy static) -----------
    dcd = None
    if not args.no_self:
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
                              jacobi_avg=not args.self_gauss_seidel),
            device=args.device)
        sim.collision_solvers_iter = [dcd.solve]  # per-iteration: interleaved

    print(f"humanV2 collision: bone={'SDF' if not args.no_bone else 'OFF'}  "
          f"self={'DCD' if not args.no_self else 'OFF'}  "
          f"({'STATIC@f1' if args.static_bone else 'ANIMATED'} skeleton, "
          f"{bone.n_bone_verts} bone verts)")

    run_humanV2_loop(sim, bone, args, cfg, surf_flat, fascia_flat)
    if dcd is not None:
        print(f"DCD self final: {dcd.stats()}")


if __name__ == "__main__":
    main()
