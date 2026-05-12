"""Rotation-copy retarget: source UsdSkel motion -> Human.usd ragdoll.

Approach (mirrors Houdini KineFX FBIK semantics)
------------------------------------------------
1. For each MAPPED Human bone, copy the source joint's world rotation with
   a per-bone alignment correction:
       R_target_W(t) = R_src_W(t) @ R_align[bi]
   R_align[bi] is the Rodrigues minimum rotation that maps target's REST
   bone-direction (rest_pos[child] - rest_pos[bone]) to source's REST
   bone-direction. This compensates for the T-pose-vs-A-pose mesh
   orientation gap (~68 deg on arms) even though both rigs have identity
   rest rotation matrices.
2. For UNMAPPED bones (L_fibula, L_patella, twists, fingers), propagate
   parent's target rotation forward keeping their rest local rotation --
   they rotate rigidly with their parent.
3. C_pelvis world position from source Pelvis (free root translation).
4. IK with rotation target per body + pelvis position. The system is
   exactly determined (276 constraints vs 276 DOF), LBFGS converges to
   the unique joint_q that realises the target rotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from pxr import Gf, Sdf, Usd, UsdGeom

from source_loader import load_source_motion


# Some entries in joint_mapping.json reference "logical" bones (e.g.
# "L_lowerarm") that aren't separate bodies in Human.usd's PhysX ragdoll
# (which splits the forearm into L_radius + L_ulna). Both physical bones
# should follow the logical bone's source rotation.
BODY_ALIAS = {
    "L_radius": "L_lowerarm",
    "L_ulna":   "L_lowerarm",
    "R_radius": "R_lowerarm",
    "R_ulna":   "R_lowerarm",
}


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def mat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s,
                         0.25 * s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([0.25 * s,
                         (R[0, 1] + R[1, 0]) / s,
                         (R[0, 2] + R[2, 0]) / s,
                         (R[2, 1] - R[1, 2]) / s])
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([(R[0, 1] + R[1, 0]) / s,
                         0.25 * s,
                         (R[1, 2] + R[2, 1]) / s,
                         (R[0, 2] - R[2, 0]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array([(R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s,
                     0.25 * s,
                     (R[1, 0] - R[0, 1]) / s])


def usd_mat_R(M: np.ndarray) -> np.ndarray:
    """USD row-major 4x4 -> column-major rotation 3x3 (R s.t. v_world = R @ v_local)."""
    return np.asarray(M, dtype=np.float64).T[:3, :3].copy()


def min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest 3x3 rotation that maps unit vector a to unit vector b (Rodrigues)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    c = float(a @ b)
    if c > 1.0 - 1e-8:
        return np.eye(3)
    if c < -1.0 + 1e-8:
        # 180-degree rotation: pick any axis perpendicular to a
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.eye(3) + 2.0 * K @ K
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def main() -> int:
    parser = argparse.ArgumentParser()
    ROOT = Path(__file__).resolve().parents[2]
    parser.add_argument("--source", type=Path,
                        default=ROOT.parent / "RLMuscle/output/walk_snu_ysy_v2.usd")
    parser.add_argument("--human-usd", type=Path,
                        default=ROOT / "data/Human/model/Human.usd")
    parser.add_argument("--joint-mapping", type=Path,
                        default=ROOT / "data/Human/config/joint_mapping.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "output/Human/walk_snu_ysy_v2_newton.usd")
    parser.add_argument("--rot-weight", type=float, default=1.0)
    parser.add_argument("--root-pos-weight", type=float, default=1.0)
    parser.add_argument("--joint-limit-weight", type=float, default=0.1)
    parser.add_argument("--ik-iters", type=int, default=500)
    parser.add_argument("--h0-scale", type=float, default=0.02)
    parser.add_argument("--max-frames", type=int, default=-1)
    args = parser.parse_args()

    # --- 1. Load Human ragdoll (Y-up) ---------------------------------------
    print(f"[retarget] loading Human: {args.human_usd}")
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
    info = builder.add_usd(str(args.human_usd), floating=True, verbose=False)
    body_paths = info["path_body_map"]
    bone_to_body = {p.split("/")[-1]: idx for p, idx in body_paths.items()}
    body_to_bone = {idx: name for name, idx in bone_to_body.items()}
    model = builder.finalize()
    B = model.body_count
    print(f"  bodies={B}  joints={model.joint_count}  q-coords={model.joint_coord_count}")

    rest_body_q = np.array(builder.body_q, dtype=np.float64)  # (B,7) px,py,pz,qx,qy,qz,qw

    # --- 2. Body tree (parent body for each body) ---------------------------
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    body_parent = np.full(B, -1, dtype=np.int64)
    for ji in range(model.joint_count):
        body_parent[int(jc[ji])] = int(jp[ji])  # -1 if connected to world

    # Topological order: parents before children
    topo: list[int] = []
    seen = np.zeros(B, dtype=bool)
    def visit(bi: int) -> None:
        if seen[bi]:
            return
        p = int(body_parent[bi])
        if p >= 0 and not seen[p]:
            visit(p)
        topo.append(bi)
        seen[bi] = True
    for bi in range(B):
        visit(bi)

    # --- 3. Rest world + rest local rotations -------------------------------
    rest_R_W = np.stack([quat_xyzw_to_mat(rest_body_q[bi, 3:7]) for bi in range(B)])
    rest_R_local = np.zeros_like(rest_R_W)
    for bi in range(B):
        p = int(body_parent[bi])
        rest_R_local[bi] = rest_R_W[bi] if p < 0 else rest_R_W[p].T @ rest_R_W[bi]

    # --- 4. Source motion ---------------------------------------------------
    print(f"[retarget] loading source: {args.source}")
    motion = load_source_motion(args.source)
    src_idx_of = {n: i for i, n in enumerate(motion.joint_names)}
    print(f"  fps={motion.fps}  frames=[{motion.frame_start},{motion.frame_end}]  "
          f"joints={motion.n_joints}")

    # --- 5. body -> source joint index (handles BODY_ALIAS one-to-many) ----
    mapping = json.loads(args.joint_mapping.read_text())["mappings"]
    body_to_src: dict[int, int] = {}
    for bi in range(B):
        bn = body_to_bone[bi]
        key = BODY_ALIAS.get(bn, bn)
        entry = mapping.get(key, {})
        if entry.get("reference") != "name":
            continue
        sn = entry["value"]
        if sn not in src_idx_of:
            continue
        body_to_src[bi] = src_idx_of[sn]
    mapped_names = sorted(body_to_bone[bi] for bi in body_to_src)
    print(f"  mapped: {len(body_to_src)}/{B} bodies")
    print(f"    e.g. {mapped_names[:10]}{' ...' if len(mapped_names)>10 else ''}")

    # --- 5b. Per-body R_align: target-rest bone direction vs source-rest --
    # Both rigs have identity rest rotations, but mesh "down-the-bone" axes
    # differ (T-pose source has arms in +X, A-pose target has arms +X-Y).
    # World-rotation copy alone leaves the arms 68 deg off; R_align fixes it.
    target_children: dict[int, list[int]] = {bi: [] for bi in range(B)}
    for bi in range(B):
        p = int(body_parent[bi])
        if p >= 0:
            target_children[p].append(bi)

    R_align = np.tile(np.eye(3)[None, :, :], (B, 1, 1))
    src_rest_pos = np.array([motion.rest_world[i][3, :3] for i in range(motion.n_joints)])
    print("  R_align per mapped body (>5 deg only):")
    for bi, sj in body_to_src.items():
        # Find a mapped child in target tree (skip aliases to parallel bones)
        mapped_child = None
        for cb in target_children[bi]:
            if cb in body_to_src and body_to_src[cb] != sj:  # avoid self-aliased pair
                mapped_child = cb
                break
        if mapped_child is None:
            continue
        sc = body_to_src[mapped_child]
        v_t = rest_body_q[mapped_child, :3] - rest_body_q[bi, :3]
        v_s = src_rest_pos[sc] - src_rest_pos[sj]
        if np.linalg.norm(v_t) < 1e-4 or np.linalg.norm(v_s) < 1e-4:
            continue
        R_align[bi] = min_rotation(v_t, v_s)
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R_align[bi]) - 1.0) / 2.0, -1.0, 1.0))))
        if ang > 5.0:
            print(f"    {body_to_bone[bi]:14s} {ang:5.1f}deg")
    # Aliased parallel bones (L_ulna, L_patella when mapped) inherit from sibling
    for bi, sj in body_to_src.items():
        if np.allclose(R_align[bi], np.eye(3)):
            # No own alignment; try sibling with same source mapping
            p = int(body_parent[bi])
            if p < 0:
                continue
            for sb in target_children[p]:
                if sb != bi and sb in body_to_src and body_to_src[sb] == sj \
                        and not np.allclose(R_align[sb], np.eye(3)):
                    R_align[bi] = R_align[sb]
                    break
            else:
                # No sibling alignment; inherit parent's
                if not np.allclose(R_align[p], np.eye(3)):
                    R_align[bi] = R_align[p]

    pelvis_bi = bone_to_body.get("C_pelvis")
    pelvis_si = src_idx_of.get("Pelvis")
    if pelvis_bi is None or pelvis_si is None:
        raise RuntimeError("C_pelvis / Pelvis missing on one side")

    # --- 6. Frame range -----------------------------------------------------
    f_first = motion.frame_start
    f_last = motion.frame_end
    if args.max_frames > 0:
        f_last = min(f_last, f_first + args.max_frames - 1)
    n_frames = f_last - f_first + 1
    print(f"[retarget] frames {f_first}..{f_last} ({n_frames})")

    # --- 7. Per-frame target world rotations + root position ----------------
    target_R = np.zeros((n_frames, B, 3, 3), dtype=np.float64)
    pelvis_pos = np.zeros((n_frames, 3), dtype=np.float32)
    for fi, f in enumerate(range(f_first, f_last + 1)):
        src_W = motion.world_xforms(float(f))
        src_R = [usd_mat_R(src_W[i]) for i in range(motion.n_joints)]
        pelvis_pos[fi] = src_W[pelvis_si][3, :3].astype(np.float32)
        for bi in topo:
            if bi in body_to_src:
                si = body_to_src[bi]
                # Rotation copy with bone-direction alignment correction:
                # R_align maps target's rest bone direction to source's, so that
                # copying source's world rotation puts the target bone's mesh
                # in the same world orientation as source's mesh.
                target_R[fi, bi] = src_R[si] @ R_align[bi]
            else:
                p = int(body_parent[bi])
                if p < 0:
                    target_R[fi, bi] = rest_R_W[bi]
                else:
                    target_R[fi, bi] = target_R[fi, p] @ rest_R_local[bi]

    # Convert to quaternions for IK objectives. (B, n_frames, 4)
    target_quats = np.zeros((B, n_frames, 4), dtype=np.float32)
    for bi in range(B):
        for fi in range(n_frames):
            target_quats[bi, fi] = mat_to_quat_xyzw(target_R[fi, bi]).astype(np.float32)

    # --- 8. IK objectives: rotation per body + pelvis position + joint limit
    objectives: list = []
    for bi in range(B):
        objectives.append(ik.IKObjectiveRotation(
            link_index=bi,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array(target_quats[bi], dtype=wp.vec4),
            weight=args.rot_weight,
        ))
    objectives.append(ik.IKObjectivePosition(
        link_index=pelvis_bi,
        link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array(pelvis_pos, dtype=wp.vec3),
        weight=args.root_pos_weight,
    ))
    objectives.append(ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower,
        joint_limit_upper=model.joint_limit_upper,
        weight=args.joint_limit_weight,
    ))
    print(f"[retarget] objectives: {B} rot + 1 pelvis-pos + 1 joint-limit")

    # --- 9. Init joint_q ----------------------------------------------------
    init_q = model.joint_q.numpy().copy()
    joint_q_np = np.tile(init_q[None, :], (n_frames, 1)).astype(np.float32)
    joint_q_np[:, 0:3] = pelvis_pos                       # FREE root translation
    joint_q_np[:, 3:7] = target_quats[pelvis_bi]          # FREE root orientation
    joint_q = wp.array(joint_q_np.flatten(), dtype=wp.float32)
    joint_q_view = joint_q.reshape((n_frames, model.joint_coord_count))

    # --- 10. Solve ----------------------------------------------------------
    solver = ik.IKSolver(
        model=model,
        n_problems=n_frames,
        objectives=objectives,
        optimizer="lbfgs",
        jacobian_mode=ik.IKJacobianType.AUTODIFF,
        h0_scale=args.h0_scale,
        line_search_alphas=[1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5, 1.0],
    )
    print(f"[retarget] solving LBFGS, {args.ik_iters} iters x {n_frames} problems...")
    solver.step(joint_q_view, joint_q_view, iterations=args.ik_iters)
    wp.synchronize()
    final_q = joint_q.numpy().reshape((n_frames, model.joint_coord_count))
    bad_mask = np.isnan(final_q).any(axis=1)
    n_bad = int(bad_mask.sum())
    if n_bad:
        bad_idxs = np.where(bad_mask)[0]
        print(f"[retarget]   WARNING: {n_bad}/{n_frames} frames have NaN q "
              f"(idxs: {bad_idxs[:20].tolist()}{'...' if n_bad>20 else ''})")
        # Repair NaN frames by copying nearest valid neighbour (prev preferred)
        last_good = None
        for fi in range(n_frames):
            if bad_mask[fi]:
                if last_good is not None:
                    final_q[fi] = final_q[last_good]
            else:
                last_good = fi
        # backwards pass for any leading NaNs
        next_good = None
        for fi in range(n_frames - 1, -1, -1):
            if bad_mask[fi] and np.isnan(final_q[fi]).any():
                if next_good is not None:
                    final_q[fi] = final_q[next_good]
            else:
                next_good = fi
        print(f"[retarget]   repaired NaN frames with nearest-valid neighbour")

    # --- 11. FK -> body world poses ----------------------------------------
    state = model.state()
    body_q = np.zeros((n_frames, B, 7), dtype=np.float64)
    qd_zero = wp.zeros(model.joint_dof_count, dtype=wp.float32)
    for fi in range(n_frames):
        newton.eval_fk(model, wp.array(final_q[fi], dtype=wp.float32), qd_zero, state)
        body_q[fi] = state.body_q.numpy()

    # --- 12. Sanity: rotation residual + chain connectivity -----------------
    for fi_ck in (0, n_frames // 2, n_frames - 1):
        worst_b, worst_deg = None, 0.0
        for bi in range(B):
            actual_R = quat_xyzw_to_mat(body_q[fi_ck, bi, 3:7])
            err_R = actual_R.T @ target_R[fi_ck, bi]
            ang = np.degrees(np.arccos(np.clip((np.trace(err_R) - 1.0) / 2.0, -1.0, 1.0)))
            if ang > worst_deg:
                worst_deg, worst_b = float(ang), body_to_bone[bi]
        print(f"[retarget]   frame {fi_ck:>3}: max rot err = {worst_deg:5.2f} deg ({worst_b})")

    for parent_b, child_b in [("L_calf", "L_fibula"), ("L_fibula", "L_foot"),
                              ("R_calf", "R_fibula"), ("R_fibula", "R_foot"),
                              ("L_thigh", "L_calf"), ("R_thigh", "R_calf"),
                              ("L_radius", "L_hand"), ("L_upperarm", "L_radius")]:
        if parent_b not in bone_to_body or child_b not in bone_to_body:
            continue
        pi, ci = bone_to_body[parent_b], bone_to_body[child_b]
        rest_d = float(np.linalg.norm(rest_body_q[pi, :3] - rest_body_q[ci, :3]))
        for fi_ck in (0, n_frames // 2):
            d = float(np.linalg.norm(body_q[fi_ck, pi, :3] - body_q[fi_ck, ci, :3]))
            print(f"[retarget]   f={fi_ck:>3}  {parent_b}<->{child_b} center dist "
                  f"{d*1000:6.1f}mm (rest {rest_d*1000:6.1f}mm)")

    # --- 13. Bake to USD ----------------------------------------------------
    print(f"[retarget] baking USD: {args.out}")
    src = Usd.Stage.Open(str(args.human_usd))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    flat = src.Flatten()
    out = Usd.Stage.Open(Sdf.Layer.CreateNew(str(args.out)))
    out.GetRootLayer().TransferContent(flat)
    out.SetTimeCodesPerSecond(float(motion.fps))
    out.SetStartTimeCode(float(f_first))
    out.SetEndTimeCode(float(f_last))

    op_by_body: dict[int, UsdGeom.XformOp] = {}
    for body_path, bi in body_paths.items():
        prim = out.GetPrimAtPath(body_path)
        if not prim:
            continue
        xf = UsdGeom.Xformable(prim)
        op = next((o for o in xf.GetOrderedXformOps()
                   if o.GetOpType() == UsdGeom.XformOp.TypeTransform), None)
        if op is None:
            op = xf.AddTransformOp()
        op.GetAttr().Clear()
        op_by_body[bi] = op

    for fi, f in enumerate(range(f_first, f_last + 1)):
        t = float(f)
        for bi, op in op_by_body.items():
            px, py, pz, qx, qy, qz, qw = body_q[fi, bi]
            R = quat_xyzw_to_mat(np.array([qx, qy, qz, qw]))
            M = np.eye(4, dtype=np.float64)
            M[:3, :3] = R.T  # USD row-vector convention
            M[3, :3] = [px, py, pz]
            op.Set(Gf.Matrix4d(*M.flatten().tolist()), t)
        if fi % 30 == 0 or fi == n_frames - 1:
            print(f"  frame {int(f):>4} ({fi+1}/{n_frames})")

    out.GetRootLayer().Save()
    sz = args.out.stat().st_size
    print(f"[retarget] wrote {args.out} ({sz/1024/1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
