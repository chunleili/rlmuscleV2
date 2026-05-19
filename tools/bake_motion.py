"""Bake rl_mass MASS trajectory onto snu_ysy.usd Joint_tree (UsdSkel SkelAnimation).

This **rewrite** drops the previous Bone-Xform/mesh-pivot retarget logic. The MASS
humanoid skeleton (data/rl_mass/humanoid.xml) is bit-identical to the
``/Human/Joint_tree/Skeleton`` UsdSkel hierarchy in the snu_ysy(3+) asset (rest
positions match within 1mm). No FK retarget is required.

Pipeline:
    walk_traj.npz  ──body_xpos/body_xquat──▶  UsdSkel SkelAnimation
                                              ├─ rotations (per joint, per frame, local quat)
                                              ├─ translations (Pelvis world; others rest_local)
                                              └─ scales (identity)
                                              bound on /Human/Joint_tree/Skeleton
                                              -> bone_mesh skins automatically.

    Also exports a clean BVH (OFFSETs from Joint_tree rest, MOTION as ZXY local
    euler) for downstream tools that prefer BVH over USD.

Asset requirement:
    The snu_ysy USD MUST contain ``/Human/Joint_tree/Skeleton`` (the
    identity-rest joint hierarchy). The original snu_ysy.usd that ships only
    with /Human/Bone/* mesh-pivot Xforms is NOT supported -- those are mesh
    authoring frames, not joints, and using them as joints introduces a 21cm
    Y-offset error in the leg chain. Use snu_ysy_v3.usd (or the equivalent
    re-authored asset) which exposes Joint_tree.

Usage:
    PYTHONPATH=. uv run python examples/rl_mass_anim/bake_motion.py \\
        --traj output/anim/walk_traj.npz \\
        --asset data/snu/snu_ysy_v3.usd \\
        -o output/anim/walk_snu_ysy
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt
from scipy.spatial.transform import Rotation as Rt


SKELETON_PATH = "/Human/Joint_tree/Skeleton"


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _matrix_translation(M: Gf.Matrix4d) -> np.ndarray:
    t = M.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=np.float64)


def _matrix_rotation(M: Gf.Matrix4d) -> np.ndarray:
    # USD row-vector convention: world = local * M, upper-left 3x3 stores R^T.
    m = np.array([[M[r][c] for c in range(3)] for r in range(3)], dtype=np.float64)
    return m.T


def _read_skeleton(stage: Usd.Stage) -> dict:
    """Pull joint hierarchy + rest from /Human/Joint_tree/Skeleton."""
    p = stage.GetPrimAtPath(SKELETON_PATH)
    if not p or p.GetTypeName() != "Skeleton":
        raise RuntimeError(
            f"Skeleton missing at {SKELETON_PATH}. Use snu_ysy_v3.usd (or any "
            f"asset with /Human/Joint_tree/Skeleton).")
    skel = UsdSkel.Skeleton(p)
    joint_paths: list[str] = list(skel.GetJointsAttr().Get())
    joint_names = [jp.split("/")[-1] for jp in joint_paths]
    path_to_idx = {jp: i for i, jp in enumerate(joint_paths)}
    parent_idx = []
    for jp in joint_paths:
        if "/" not in jp:
            parent_idx.append(-1)
        else:
            parent_idx.append(path_to_idx["/".join(jp.split("/")[:-1])])

    rest_local = list(skel.GetRestTransformsAttr().Get())  # parent-relative
    bind_world = list(skel.GetBindTransformsAttr().Get())  # world

    rest_local_t = np.array([_matrix_translation(M) for M in rest_local])
    rest_local_R = np.array([_matrix_rotation(M) for M in rest_local])
    bind_world_t = np.array([_matrix_translation(M) for M in bind_world])
    bind_world_R = np.array([_matrix_rotation(M) for M in bind_world])

    # Sanity: bind/rest rotations should all be identity for snu_ysy(3).
    max_dev = max(np.abs(R - np.eye(3)).max() for R in bind_world_R)
    if max_dev > 1e-3:
        print(f"[bake] WARNING: Skeleton bind rotations not identity "
              f"(max dev {max_dev:.4g}). FK assumes identity rest -- proceed with caution.")

    return {
        "skel_prim": p,
        "joint_paths": joint_paths,
        "joint_names": joint_names,
        "parent_idx": parent_idx,
        "rest_local_t": rest_local_t,
        "rest_local_R": rest_local_R,
        "bind_world_t": bind_world_t,
        "bind_world_R": bind_world_R,
    }


def write_muscle_activations(stage: Usd.Stage, diag_path: str,
                              max_frames: int | None = None) -> int:
    """Write per-frame activation as a time-sampled UsdGeom Primvar
    `primvars:muscle:activation` (Float, interpolation=constant) on each
    /Human/Muscles/<name>/curve BasisCurves prim.

    Using a Primvar (not a raw attribute) so downstream DCCs --
    Houdini/Solaris in particular -- pick it up as a detail attribute on
    the curve geometry. Interpolation=constant means one scalar covers the
    whole curve per timecode.

    Diag NPZ layout (from `eval.py --diag`):
        a            (T_diag, 284) float32  -- activation [0, 1]
        muscle_names (284,)        <U..

    Frame mismatch handling: walk_traj.npz has T+1 frames (rollout includes
    init), walk_diag.npz has T (one shorter). Trailing USD frames hold the
    last activation value.
    """
    print(f"[act ] Loading {diag_path}")
    d = np.load(diag_path, allow_pickle=False)
    a = d["a"]                                # (T_diag, 284)
    names = [str(n) for n in d["muscle_names"]]
    T_diag = a.shape[0]

    end_tc = stage.GetEndTimeCode()
    T_usd = int(end_tc) + 1
    if max_frames is not None:
        T_usd = min(T_usd, max_frames)

    written = 0
    missing = []
    for mi, name in enumerate(names):
        curve_prim = stage.GetPrimAtPath(f"/Human/Muscles/{name}/curve")
        if not curve_prim:
            missing.append(name); continue
        pv_api = UsdGeom.PrimvarsAPI(curve_prim)
        pv = pv_api.CreatePrimvar(
            "muscle:activation",
            Sdf.ValueTypeNames.Float,
            UsdGeom.Tokens.constant,
        )
        for ti in range(T_usd):
            di = min(ti, T_diag - 1)            # hold-last for trailing frames
            pv.Set(float(a[di, mi]), Usd.TimeCode(ti))
        written += 1
    if missing:
        print(f"[act ] {len(missing)} muscle curve prims missing in USD: {missing[:5]}...")
    print(f"[act ] Wrote primvars:muscle:activation on {written} muscle curves "
          f"({T_usd} frames; diag T={T_diag}).")
    return written


def skin_muscles(stage: Usd.Stage, muscle_xml: str,
                 joint_paths: list[str]) -> int:
    """Per-waypoint single-bone skinning of /Human/Muscles/<name>/curve.

    Each MASS muscle Waypoint has a `body` attribute naming the anchor bone.
    Curve points are 1:1 ordered with waypoints (verified within < 1e-3 mm
    in the source asset). Apply SkelBindingAPI per curve with elementSize=1
    weights (single-bone, weight=1) so UsdSkel skinning rigidly inherits each
    waypoint's parent-bone transform.

    /Human/Muscles is promoted from Scope to SkelRoot, with skel:skeleton rel
    pointing at the Joint_tree skeleton. Returns # curves successfully skinned.
    """
    print(f"[musc] Loading {muscle_xml}")
    tree = ET.parse(muscle_xml)
    mass_units = {}
    for u in tree.getroot():
        wps = [(c.get("body"),
                tuple(map(float, c.get("p").split())))
               for c in u if c.tag == "Waypoint"]
        mass_units[u.get("name")] = wps

    # Map joint leaf-name -> joint index in Skeleton.joints.
    name_to_jidx = {jp.split("/")[-1]: i for i, jp in enumerate(joint_paths)}

    muscles_prim = stage.GetPrimAtPath("/Human/Muscles")
    if not muscles_prim:
        print("[musc] /Human/Muscles not found; skipping")
        return 0

    # Promote Muscles Scope -> SkelRoot so UsdSkel processes child curves.
    muscles_prim.SetTypeName("SkelRoot")
    muscles_api = UsdSkel.BindingAPI.Apply(muscles_prim)
    muscles_api.CreateSkeletonRel().SetTargets([Sdf.Path(SKELETON_PATH)])

    # Curves are in world space at rest; geomBindTransform = identity.
    identity_bind = Gf.Matrix4d(1.0)

    skinned = 0
    bad_body: list[str] = []
    bad_count: list[str] = []
    for muscle_name, waypoints in mass_units.items():
        curve_path = f"/Human/Muscles/{muscle_name}/curve"
        curve_prim = stage.GetPrimAtPath(curve_path)
        if not curve_prim:
            continue
        curve = UsdGeom.BasisCurves(curve_prim)
        pts = curve.GetPointsAttr().Get()
        if pts is None or len(pts) != len(waypoints):
            bad_count.append(muscle_name)
            continue

        ji_arr = []
        jw_arr = []
        ok = True
        for body_name, _ in waypoints:
            ji = name_to_jidx.get(body_name)
            if ji is None:
                bad_body.append(f"{muscle_name}:{body_name}")
                ok = False
                break
            ji_arr.append(ji)
            jw_arr.append(1.0)
        if not ok:
            continue

        api = UsdSkel.BindingAPI.Apply(curve_prim)
        idx_pv = api.CreateJointIndicesPrimvar(False, 1)  # vertex interp, elementSize=1
        idx_pv.Set(Vt.IntArray(ji_arr))
        wt_pv = api.CreateJointWeightsPrimvar(False, 1)
        wt_pv.Set(Vt.FloatArray(jw_arr))
        api.CreateGeomBindTransformAttr(identity_bind)
        skinned += 1

    if bad_count:
        print(f"[musc] {len(bad_count)} curves with point/waypoint count mismatch (skipped): {bad_count[:5]}...")
    if bad_body:
        print(f"[musc] {len(bad_body)} waypoints with unknown body (skipped muscle): {bad_body[:5]}...")
    print(f"[musc] Skinned {skinned} muscle curves to skeleton.")
    return skinned


def bake_skelanim(traj_path: str, asset_path: str, out_usd: str,
                   max_frames: int | None = None,
                   muscle_xml: str | None = None,
                   diag_path: str | None = None) -> dict:
    """Author UsdSkelAnimation from MASS trajectory; bind to Joint_tree Skeleton."""
    t0 = time.time()
    print(f"[bake] Loading trajectory: {traj_path}")
    data = np.load(traj_path, allow_pickle=False)
    body_xpos = data["body_xpos"]      # (T, N, 3)
    body_xquat = data["body_xquat"]    # (T, N, 4) wxyz
    body_names = [str(n) for n in data["body_names"]]
    control_hz = int(data["control_hz"])
    name2idx = {n: i for i, n in enumerate(body_names)}
    T = body_xpos.shape[0]
    if max_frames is not None:
        T = min(T, max_frames)
    print(f"[bake] frames={T} bodies={len(body_names)} hz={control_hz}")

    print(f"[bake] Copying {asset_path} -> {out_usd}")
    os.makedirs(os.path.dirname(out_usd) or ".", exist_ok=True)
    shutil.copy(asset_path, out_usd)
    stage = Usd.Stage.Open(out_usd)
    stage.SetTimeCodesPerSecond(control_hz)
    stage.SetFramesPerSecond(control_hz)
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(float(T - 1))

    skel_info = _read_skeleton(stage)
    joint_paths = skel_info["joint_paths"]
    joint_names = skel_info["joint_names"]
    parent_idx = skel_info["parent_idx"]
    rest_local_t = skel_info["rest_local_t"]
    n_joints = len(joint_paths)

    # MASS body index per joint (must all resolve).
    missing = [n for n in joint_names if n not in name2idx]
    if missing:
        raise RuntimeError(f"MASS traj lacks bodies for joints: {missing}")
    mass_idx = np.array([name2idx[n] for n in joint_names], dtype=np.int64)

    # The source asset hides Joint_tree by default (its visualization aid
    # was intended to be the rigid /Human/Bone meshes). For the LBS pipeline
    # we want the opposite: show the skinned bone_mesh + muscle curves, hide
    # everything else.
    for show in ("/Human/Joint_tree", "/Human/Joint_tree/bone_mesh",
                 "/Human/Muscles"):
        prim = stage.GetPrimAtPath(show)
        if prim:
            UsdGeom.Imageable(prim).MakeVisible()
    for hide in ("/Human/Rigid_body_tree", "/Human/Bone",
                 "/Human/Joint_tree/joint_viz"):
        prim = stage.GetPrimAtPath(hide)
        if prim:
            UsdGeom.Imageable(prim).MakeInvisible()

    # Create SkelAnimation under the Skeleton.
    anim_path = f"{SKELETON_PATH}/Anim"
    anim = UsdSkel.Animation.Define(stage, anim_path)
    anim.GetJointsAttr().Set(joint_paths)

    binding_api = UsdSkel.BindingAPI.Apply(skel_info["skel_prim"])
    binding_api.CreateAnimationSourceRel().SetTargets([Sdf.Path(anim_path)])

    # Skin muscle curves (each waypoint -> single bone, weight 1).
    if muscle_xml:
        skin_muscles(stage, muscle_xml, joint_paths)

    # Per-frame activation: time-sampled muscle:activation on each Muscle prim.
    if diag_path:
        write_muscle_activations(stage, diag_path, max_frames=T)

    # Constant scales = identity for all frames.
    scales_const = Vt.Vec3hArray([Gf.Vec3h(1.0, 1.0, 1.0)] * n_joints)
    anim.GetScalesAttr().Set(scales_const)

    print(f"[bake] Writing SkelAnimation ({T} frames × {n_joints} joints) ...")
    rotations_attr = anim.GetRotationsAttr()
    translations_attr = anim.GetTranslationsAttr()

    # Cache per-frame world R/t for BVH export (keep float64).
    world_R_per_frame = np.zeros((T, n_joints, 3, 3), dtype=np.float64)
    local_R_per_frame = np.zeros((T, n_joints, 3, 3), dtype=np.float64)
    pelvis_world_t = np.zeros((T, 3), dtype=np.float64)

    for ti in range(T):
        rot_t = Rt.from_quat(_wxyz_to_xyzw(body_xquat[ti]))
        # World rotations indexed by joint order.
        for j in range(n_joints):
            world_R_per_frame[ti, j] = rot_t[mass_idx[j]].as_matrix()
        # Local rotation = R_parent_world^T @ R_world (root: identity parent).
        for j in range(n_joints):
            pi = parent_idx[j]
            if pi == -1:
                local_R_per_frame[ti, j] = world_R_per_frame[ti, j]
            else:
                local_R_per_frame[ti, j] = (
                    world_R_per_frame[ti, pi].T @ world_R_per_frame[ti, j]
                )

        # Translations: rest_local for non-root; root gets MASS world position.
        local_t = rest_local_t.copy()
        for j in range(n_joints):
            if parent_idx[j] == -1:
                local_t[j] = body_xpos[ti, mass_idx[j]]
                pelvis_world_t[ti] = body_xpos[ti, mass_idx[j]]

        # Pack per-joint quats.
        quats: list[Gf.Quatf] = []
        for j in range(n_joints):
            qx, qy, qz, qw = Rt.from_matrix(local_R_per_frame[ti, j]).as_quat()
            quats.append(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
        rotations_attr.Set(Vt.QuatfArray(quats), Usd.TimeCode(ti))

        trans = [Gf.Vec3f(float(local_t[j, 0]), float(local_t[j, 1]),
                          float(local_t[j, 2])) for j in range(n_joints)]
        translations_attr.Set(Vt.Vec3fArray(trans), Usd.TimeCode(ti))

        if (ti + 1) % 100 == 0:
            print(f"  frame {ti + 1}/{T}  ({time.time() - t0:.1f}s)")

    print(f"[bake] Saving USD to {out_usd}")
    stage.GetRootLayer().Save()
    print(f"[bake] USD done in {time.time() - t0:.1f}s")

    return {
        "T": T,
        "control_hz": control_hz,
        "joint_paths": joint_paths,
        "joint_names": joint_names,
        "parent_idx": parent_idx,
        "rest_local_t": rest_local_t,
        "local_R": local_R_per_frame,
        "pelvis_world_t": pelvis_world_t,
    }


def export_bvh(out_bvh: str, joint_paths: list[str], joint_names: list[str],
               parent_idx: list[int], rest_local_t: np.ndarray,
               local_R: np.ndarray, pelvis_world_t: np.ndarray,
               control_hz: int) -> None:
    """Write BVH using Joint_tree rest as OFFSETs and per-frame local R as MOTION.

    OFFSETs come from rest_local_t (parent-relative joint translations).
    MOTION rows: ROOT (Pelvis) gets 6 channels (Xpos Ypos Zpos Zrot Xrot Yrot);
    other joints get 3 (Zrot Xrot Yrot). All eulers are intrinsic ZXY.
    """
    print(f"[bvh ] Writing {out_bvh}")
    n_joints = len(joint_paths)
    children: dict[int, list[int]] = {i: [] for i in range(n_joints)}
    for c, p in enumerate(parent_idx):
        if p != -1:
            children[p].append(c)
    roots = [i for i, p in enumerate(parent_idx) if p == -1]
    if len(roots) != 1:
        raise RuntimeError(f"BVH needs single root; got {len(roots)}")
    root = roots[0]

    # HIERARCHY (DFS).
    lines: list[str] = ["HIERARCHY"]

    def emit(j: int, depth: int, is_root: bool) -> None:
        ind = "\t" * depth
        kw = "ROOT" if is_root else "JOINT"
        lines.append(f"{ind}{kw} {joint_names[j]}")
        lines.append(f"{ind}{{")
        if is_root:
            # Root OFFSET = world rest translation (Pelvis bind world).
            off = rest_local_t[j]
        else:
            off = rest_local_t[j]
        lines.append(f"{ind}\tOFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
        if is_root:
            lines.append(f"{ind}\tCHANNELS 6 Xposition Yposition Zposition "
                         f"Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{ind}\tCHANNELS 3 Zrotation Xrotation Yrotation")
        kids = children[j]
        for k in kids:
            emit(k, depth + 1, False)
        if not kids:
            lines.append(f"{ind}\tEnd Site")
            lines.append(f"{ind}\t{{")
            lines.append(f"{ind}\t\tOFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{ind}\t}}")
        lines.append(f"{ind}}}")

    emit(root, 0, True)

    # MOTION (DFS order matching emit).
    def dfs_order(j: int) -> list[int]:
        out = [j]
        for k in children[j]:
            out.extend(dfs_order(k))
        return out

    order = dfs_order(root)
    T = local_R.shape[0]
    lines.append("MOTION")
    lines.append(f"Frames: {T}")
    lines.append(f"Frame Time: {1.0 / float(control_hz):.7f}")

    for ti in range(T):
        row: list[str] = []
        # Root.
        ez, ex, ey = Rt.from_matrix(local_R[ti, root]).as_euler("ZXY",
                                                                degrees=True)
        row.extend([
            f"{pelvis_world_t[ti, 0]:.6f}",
            f"{pelvis_world_t[ti, 1]:.6f}",
            f"{pelvis_world_t[ti, 2]:.6f}",
            f"{ez:.6f}", f"{ex:.6f}", f"{ey:.6f}",
        ])
        for j in order[1:]:
            ez, ex, ey = Rt.from_matrix(local_R[ti, j]).as_euler("ZXY",
                                                                  degrees=True)
            row.extend([f"{ez:.6f}", f"{ex:.6f}", f"{ey:.6f}"])
        lines.append(" ".join(row))

    os.makedirs(os.path.dirname(out_bvh) or ".", exist_ok=True)
    with open(out_bvh, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[bvh ] {T} frames, {len(order)} joints, written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--asset", default="data/rl_mass/snuV2.usd")
    ap.add_argument("--muscle-xml", default="data/rl_mass/muscle284.xml",
                    help="MASS muscle definition; per-waypoint bone anchors are "
                         "used to skin the curve points. Pass empty string to skip.")
    ap.add_argument("--diag", default=None,
                    help="walk_diag.npz from eval.py --diag (carries 'a' "
                         "activation array). When given, time-samples "
                         "muscle:activation onto each muscle prim.")
    ap.add_argument("-o", "--output", required=True,
                    help="Output base path (writes .usd and .bvh)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-bvh", action="store_true")
    args = ap.parse_args()

    base = args.output[:-4] if args.output.endswith(".usd") else args.output
    out_usd = f"{base}.usd"
    out_bvh = f"{base}.bvh"

    muscle_xml = args.muscle_xml if args.muscle_xml else None
    bake = bake_skelanim(args.traj, args.asset, out_usd, args.max_frames,
                          muscle_xml=muscle_xml, diag_path=args.diag)

    if not args.no_bvh:
        export_bvh(
            out_bvh,
            bake["joint_paths"],
            bake["joint_names"],
            bake["parent_idx"],
            bake["rest_local_t"],
            bake["local_R"],
            bake["pelvis_world_t"],
            bake["control_hz"],
        )


if __name__ == "__main__":
    main()
