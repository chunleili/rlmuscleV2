"""Real bone collider for humanV2: muscle vs the SNU Ragdoll skeleton mesh.

Replaces the placeholder test sphere (``example_humanV2_collision``) with the
actual bone geometry from ``data/Human/model/Human.usd`` (whose rest bounds
align 1:1 with the humanV2 muscle). Each substep, free muscle vertices that
have sunk into a bone are pushed back out to the bone surface via a Warp
triangle-mesh SDF query (``solve_mesh_contact_pos_kernel``).

v1 uses the bones at their REST pose (static collider). The muscle is animated
by the per-frame pin targets, so near the rest frames this is a faithful
muscle-vs-bone contact; driving the bones with the matching skeleton animation
is the next step (needs the humanV2 motion that baked the cons targets — the
73-frame otto_motion does not span the GMG's 200 frames).

Run::

    PYTHONPATH=src .venv/bin/python -m muscle.collision.example_humanV2_bone \\
        --config data/humanV2/config/humanV2_gmg.json \\
        --out_dir result/humanV2_bone --end_frame 20 --usd output/humanV2_bone.usd
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import warp as wp

from muscle.coupleOneWayGmg.solver import OneWaySim, load_config
from muscle.coupleOneWayGmg.example import _vectorized_surface_tris
from muscle.collision.sdf_warp import solve_mesh_contact_pos_kernel
from muscle.util.mesh_io import load_bone_mesh, save_ply


@wp.kernel
def _count_inside_kernel(pos: wp.array(dtype=wp.vec3),
                         mesh: wp.uint64, max_dist: wp.float32,
                         count: wp.array(dtype=wp.int32)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_normal(mesh, pos[i], max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        if wp.length(pos[i] - cp) * q.sign < 0.0:
            wp.atomic_add(count, 0, 1)


class BoneMeshCollider:
    """Static SNU-skeleton triangle mesh as a rigid collider for the muscle."""

    def __init__(self, sim: OneWaySim, human_usd: str,
                 bone_root: str = "/Human/Ragdoll/Bones",
                 max_dist: float = 0.2, contact_offset: float = 0.0):
        self.sim = sim
        self.device = sim._device
        self.max_dist = float(max_dist)
        self.contact_offset = float(contact_offset)

        _, bverts, bidx, _ = load_bone_mesh(Path(human_usd), bone_root=bone_root)
        self.n_bone_verts = int(bverts.shape[0])
        self.n_bone_tris = int(bidx.shape[0] // 3)
        # The SNU bone meshes are wound inward relative to what Warp's signed
        # mesh query expects — without flipping, mesh_query_point_sign_normal
        # reports nearly every muscle vert as "inside" (157235/157369 at rest
        # vs the true 134). Swap each triangle's last two indices to flip the
        # winding / outward normal.
        bidx = bidx.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1)
        self.mesh = wp.Mesh(
            points=wp.array(bverts.astype(np.float32), dtype=wp.vec3, device=self.device),
            indices=wp.array(bidx.astype(np.int32), dtype=wp.int32, device=self.device),
        )
        # humanV2 is fully pin-driven (pts == all NV); use full inv_mass so the
        # push can act, running as the substep's last op to win over the pin.
        self.coll_inv_mass = wp.clone(sim.inv_mass)
        self._count = wp.zeros(1, dtype=wp.int32, device=self.device)

    def __call__(self):
        wp.launch(
            solve_mesh_contact_pos_kernel, dim=self.sim.NV,
            inputs=[self.sim.pos, self.coll_inv_mass, self.mesh.id,
                    wp.float32(self.max_dist), wp.float32(self.contact_offset)],
            device=self.device)

    def inside_count(self) -> int:
        self._count.zero_()
        wp.launch(_count_inside_kernel, dim=self.sim.NV,
                  inputs=[self.sim.pos, self.mesh.id,
                          wp.float32(self.max_dist), self._count],
                  device=self.device)
        return int(self._count.numpy()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/humanV2/config/humanV2_gmg.json")
    ap.add_argument("--human_usd", default="data/Human/model/Human.usd")
    ap.add_argument("--bone_root", default="/Human/Ragdoll/Bones")
    ap.add_argument("--out_dir", default="result/humanV2_bone")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--end_frame", type=int, default=20)
    ap.add_argument("--max_dist", type=float, default=0.2)
    ap.add_argument("--no_collision", action="store_true")
    ap.add_argument("--usd", default=None)
    args = ap.parse_args()

    wp.init()
    cfg = load_config(args.config)
    cfg["end_frame"] = args.end_frame
    sim = OneWaySim(cfg, device=args.device)

    collider = None
    if not args.no_collision:
        collider = BoneMeshCollider(sim, args.human_usd, args.bone_root,
                                    max_dist=args.max_dist)
        sim.collision_solvers = [collider]
        print(f"bone collider: {collider.n_bone_verts} verts, "
              f"{collider.n_bone_tris} tris")
    print(f"NV={sim.NV}  NT={sim.NT}  frames={sim.start_frame}..{sim.end_frame}")

    out_dir = Path(args.out_dir)
    mesh_out = out_dir / "mesh"
    mesh_out.mkdir(parents=True, exist_ok=True)
    surface_faces = _vectorized_surface_tris(sim.tet_idx.numpy())
    surf_flat = surface_faces.reshape(-1).astype(np.int32)

    # Dump bone mesh once for viz.
    if collider is not None:
        bpts = collider.mesh.points.numpy()
        bidx = collider.mesh.indices.numpy().reshape(-1, 3)
        save_ply(str(out_dir / "bones.ply"), bpts, bidx)

    renderer = None
    if args.usd:
        from muscle.muscle_warp import WarpRenderer
        Path(args.usd).parent.mkdir(parents=True, exist_ok=True)
        renderer = WarpRenderer(mode="usd", stage_path=args.usd)
        bone_flat = (collider.mesh.indices.numpy().astype(np.int32)
                     if collider else None)
        bone_pts = collider.mesh.points.numpy() if collider else None
        print(f"[render] writing USD to {args.usd}")

    t0 = time.perf_counter()
    for frame in range(sim.start_frame, sim.end_frame + 1):
        sim.step_frame(frame)
        pos_np = sim.positions_np()
        save_ply(str(mesh_out / f"{frame:04d}.ply"), pos_np, surface_faces)
        if renderer is not None:
            inner = renderer.renderer
            inner.begin_frame(frame * sim.dt)
            inner.render_mesh(name="muscle", points=pos_np,
                              indices=surf_flat, colors=(0.8, 0.3, 0.25))
            if bone_pts is not None:
                inner.render_mesh(name="bones", points=bone_pts,
                                  indices=bone_flat, colors=(0.9, 0.88, 0.8))
            inner.end_frame()
        if frame % 5 == 0 or frame == sim.end_frame:
            elapsed = time.perf_counter() - t0
            ins = collider.inside_count() if collider else 0
            print(f"  frame {frame:>3d}  {elapsed:6.1f}s  verts_inside_bone={ins}")
    if renderer is not None:
        renderer.save()
        print(f"USD saved -> {args.usd}")
    print(f"done -> {mesh_out}")


if __name__ == "__main__":
    main()
