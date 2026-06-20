import warp as wp; wp.init()
import numpy as np
import argparse
from example.humanV2 import build_humanV2_sim_from_args, AnimatedBoneMeshCollider


@wp.kernel
def dbg(pos: wp.array(dtype=wp.vec3), mesh: wp.uint64, md: wp.float32,
        vid: wp.int32, out: wp.array(dtype=wp.float32)):
    if wp.tid() != 0:
        return
    x = pos[vid]
    q = wp.mesh_query_point_sign_normal(mesh, x, md)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        delta = x - cp
        dl = wp.length(delta)
        out[0] = 1.0
        out[1] = dl
        out[2] = q.sign
        out[3] = dl * q.sign
        out[4] = float(q.face)
    else:
        out[0] = 0.0


def runto(fiber, upto, vid):
    a = argparse.Namespace(
        config='data/humanV2/config/humanV2.json', device='cuda:0',
        end_frame=upto, fiber=fiber, fiber_stiffness=1e9, contraction_ratio=0.8,
        activate=None, bone_dir='data/humanV2/model/bone', max_dist=0.2,
        contact_offset=1e-5, static_bone=False, no_collision=False)
    sim, cfg, sf, ff = build_humanV2_sim_from_args(a)
    col = AnimatedBoneMeshCollider(sim, a.bone_dir, max_dist=a.max_dist,
                                   contact_offset=a.contact_offset)
    col.preload(range(1, upto + 1))
    sim.collision_solvers = [col]
    for f in range(1, upto + 1):
        col.set_frame(f)
        sim.step_frame(f)
    out = wp.zeros(5, dtype=wp.float32, device='cuda:0')
    wp.launch(dbg, dim=1, inputs=[sim.pos, col.mesh.id, wp.float32(0.2),
                                  wp.int32(vid), out], device='cuda:0')
    wp.synchronize()
    o = out.numpy()
    print('PROBE %-9s v%d f%d: hit=%.0f delta=%.1fmm sign=%.0f signed=%.1fmm face=%.0f'
          % (fiber, vid, upto, o[0], o[1]*1000, o[2], o[3]*1000, o[4]))


if __name__ == '__main__':
    runto('none', 19, 8636)
    runto('fibernorm', 19, 8636)
