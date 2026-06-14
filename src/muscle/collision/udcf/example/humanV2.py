"""humanV2 muscle-vs-moving-skeleton via the Gaia UDCF DCD pipeline.

Same scene as ``muscle.collision.sdf.example.humanV2`` (the pin-driven humanV2
GMG muscle solver against the 200-frame animated SNU skeleton), but the contact
is resolved by the UDCF DCD narrow/respond pipeline (:class:`UdcfDcdCollider`)
instead of the single-push SDF mesh-contact kernel.

Reuse vs the SDF demo:
  * ``build_humanV2_sim`` / ``run_humanV2_loop`` / ``add_humanV2_args`` — the
    sim build, preload, fascia split, USD/geo export, and the per-frame loop are
    shared verbatim.
  * ``AnimatedBoneMeshCollider`` is reused unchanged as the penetration
    MEASUREMENT instrument (its ``wp.Mesh`` + ``inside_depth`` drive the
    ``is_collide`` / ``pen_depth`` primvars), so the USD output is directly
    A/B-comparable with the SDF run.

Only the contact RESPONSE differs: the muscle surface triangles are the movable
set and the per-frame skeleton mesh is the DCD static body.  The collider runs
on ``sim.collision_solvers_iter`` — the per-iteration hook that interleaves
DCD narrowPhase with ARAP/volume the way the Gaia reference PBD loop does —
rather than the SDF demo's once-per-substep ``collision_solvers`` post pass.

Run::

    PYTHONPATH=src .venv/bin/python -m muscle.collision.udcf.example.humanV2 \\
        --config data/humanV2/config/humanV2_gmg.json \\
        --bone_dir data/humanV2/model/bone --end_frame 20 \\
        --usd output/humanV2_dcd_coll.usd
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


def main(argv=None):
    ap = add_humanV2_args(argparse.ArgumentParser())
    ap.add_argument("--usd", default="output/humanV2_dcd_coll.usd")
    ap.add_argument("--self_collision", action="store_true",
                    help="also resolve muscle self-collision (movable-vs-movable)")
    ap.add_argument("--thickness", type=float, default=1e-3,
                    help="broadphase sphere slack (m)")
    ap.add_argument("--multiplier", type=float, default=1.01,
                    help="collisionConstraintMultiplier")
    ap.add_argument("--mu_s", type=float, default=0.5)
    ap.add_argument("--mu_d", type=float, default=0.4)
    ap.add_argument("--max_pairs", type=int, default=1 << 21,
                    help="broadphase pair cap (humanV2 skeleton emits ~1.1M)")
    ap.add_argument("--max_contacts", type=int, default=1 << 18)
    args = ap.parse_args(argv)

    wp.init()
    sim, cfg, surf_flat, fascia_flat = build_humanV2_sim(args)

    # Reuse the SDF collider purely as the bone-mesh holder + penetration meter:
    # it owns the per-frame wp.Mesh (set_frame / refit) and inside_depth, but is
    # NOT registered as a solver — the DCD collider does the actual response.
    meter = AnimatedBoneMeshCollider(sim, args.bone_dir, max_dist=args.max_dist)
    if not args.static_bone:
        meter.preload(range(sim.start_frame, sim.end_frame + 1))

    dcd = None
    if not args.no_collision:
        # Movable = muscle surface tris; static = the skeleton mesh (already
        # outward-wound by AnimatedBoneMeshCollider's [0,2,1] swap, exposed as
        # tris_flat).  DCD offsets the static tri indices by n_mov internally.
        bone_tris = meter.tris_flat.reshape(-1, 3)
        dcd = UdcfDcdCollider(
            sim.pos, sim.inv_mass,
            surface_tris=surf_flat.reshape(-1, 3),
            static_verts=np.asarray(meter._cur_verts, np.float32),
            static_tris=bone_tris,
            params=UdcfParams(allow_self=args.self_collision,
                              thickness=args.thickness,
                              multiplier=args.multiplier,
                              friction_static=args.mu_s,
                              friction_dynamic=args.mu_d,
                              max_pairs=args.max_pairs,
                              max_contacts=args.max_contacts),
            device=args.device)
        sim.collision_solvers_iter = [dcd.solve]

    print(f"bone collider (UDCF DCD): {meter.n_bone_verts} verts, "
          f"{meter.n_bone_tris} tris  "
          f"({'STATIC@f1' if args.static_bone else 'ANIMATED'}, "
          f"push={'OFF' if args.no_collision else 'ON'}, "
          f"self={'ON' if args.self_collision else 'OFF'})")

    def push_bones(frame):
        # meter.set_frame already loaded this frame's bone verts into _cur_verts;
        # mirror them into the DCD static region (old pose -> x_prev for friction).
        if dcd is not None:
            dcd.set_static_verts(np.asarray(meter._cur_verts, np.float32))

    run_humanV2_loop(sim, meter, args, cfg, surf_flat, fascia_flat,
                     pre_step=push_bones)
    if dcd is not None:
        print(f"DCD final: {dcd.stats()}")


if __name__ == "__main__":
    main()
