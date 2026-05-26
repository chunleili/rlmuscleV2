"""One-shot pipeline: rl_mass ckpt -> Human.usd muscle-driven animation.

Stages (all artifacts written under output/pipeline/<run_name>/):
  1. Eval rl_mass policy           -> traj.npz
  2. Bake UsdSkel from traj         -> source_skel.usd  (23 MASS joints)
  3. Retarget to Human.usd ragdoll  -> ragdoll_anim.usd (91 bone xforms)
  4. Convert ragdoll USD -> JSON    -> motion.json     (otto-format)
  5. example_drive (fullbody_drive) -> muscle.usd
  6. usd_to_mp4                     -> muscle.mp4

Defaults wire walk_max.pt -> data/muscle/config/fullbody_drive.json -> muscle.usd
(MP4 stage 6 is opt-in via --mp4).

Usage:
    uv run python tools/run_pipeline.py
    uv run python tools/run_pipeline.py --ckpt balance_max --frames 60
    uv run python tools/run_pipeline.py --skip-stages 1,2,3  # reuse cached
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SISTER_ASSET = Path(
    "/lus/lfs1aip2/scratch/u6ey/chunleili.u6ey/Dev/RLMuscle/data/snu/snu_ysy_v3.usd"
)


def _hdr(msg: str) -> None:
    print(f"\n========== {msg} ==========", flush=True)


# ---------------------------------------------------------------------------
# Stage 1: eval rl_mass
# ---------------------------------------------------------------------------

def stage1_eval(ckpt: str, max_steps: int, out_dir: Path) -> tuple[Path, Path]:
    """Run rl_mass eval, save trajectory + diagnostics NPZ."""
    _hdr(f"Stage 1: eval rl_mass ckpt={ckpt}")
    traj = out_dir / "traj.npz"
    diag = out_dir / "diag.npz"
    cmd = [
        sys.executable, "-m", "rl_mass.eval",
        "-d", "data/rl_mass/config.json",
        "-m", f"data/rl_mass/checkpoints/{ckpt}",
        "--max-steps", str(max_steps),
        "--save-traj", str(traj),
        "--diag", str(diag),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)
    return traj, diag


# ---------------------------------------------------------------------------
# Stage 2: bake UsdSkel
# ---------------------------------------------------------------------------

def stage2_bake_skel(traj: Path, asset: Path, out_usd: Path,
                     max_frames: int | None) -> None:
    """Author UsdSkelAnimation from MASS trajectory bound to Joint_tree."""
    _hdr(f"Stage 2: bake UsdSkel -> {out_usd.name}")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt
    from scipy.spatial.transform import Rotation as Rt

    SKEL_PATH = "/Human/Joint_tree/Skeleton"

    def _wxyz_to_xyzw(q):
        return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)

    def _mat_t(M):
        t = M.ExtractTranslation()
        return np.array([t[0], t[1], t[2]], dtype=np.float64)

    def _mat_R(M):
        m = np.array([[M[r][c] for c in range(3)] for r in range(3)], dtype=np.float64)
        return m.T  # USD row-vector

    data = np.load(traj, allow_pickle=False)
    body_xpos = data["body_xpos"]
    body_xquat = data["body_xquat"]
    body_names = [str(n) for n in data["body_names"]]
    control_hz = int(data["control_hz"])
    name2idx = {n: i for i, n in enumerate(body_names)}
    T = body_xpos.shape[0]
    if max_frames is not None:
        T = min(T, max_frames)
    print(f"  frames={T} bodies={len(body_names)} hz={control_hz}")

    out_usd.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(asset, out_usd)
    stage = Usd.Stage.Open(str(out_usd))
    stage.SetTimeCodesPerSecond(control_hz)
    stage.SetFramesPerSecond(control_hz)
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(float(T - 1))

    skel_prim = stage.GetPrimAtPath(SKEL_PATH)
    if not skel_prim or skel_prim.GetTypeName() != "Skeleton":
        raise RuntimeError(f"Skeleton missing at {SKEL_PATH} in {asset}")
    skel = UsdSkel.Skeleton(skel_prim)
    joint_paths = list(skel.GetJointsAttr().Get())
    joint_names = [jp.split("/")[-1] for jp in joint_paths]
    path_to_idx = {jp: i for i, jp in enumerate(joint_paths)}
    parent_idx = [
        -1 if "/" not in jp
        else path_to_idx["/".join(jp.split("/")[:-1])]
        for jp in joint_paths
    ]
    rest_local = list(skel.GetRestTransformsAttr().Get())
    rest_local_t = np.array([_mat_t(M) for M in rest_local])
    n_joints = len(joint_paths)

    missing = [n for n in joint_names if n not in name2idx]
    if missing:
        raise RuntimeError(f"MASS traj lacks bodies for joints: {missing}")
    mass_idx = np.array([name2idx[n] for n in joint_names], dtype=np.int64)

    anim_path = f"{SKEL_PATH}/Anim"
    anim = UsdSkel.Animation.Define(stage, anim_path)
    anim.GetJointsAttr().Set(joint_paths)
    UsdSkel.BindingAPI.Apply(skel_prim).CreateAnimationSourceRel().SetTargets(
        [Sdf.Path(anim_path)]
    )
    anim.GetScalesAttr().Set(
        Vt.Vec3hArray([Gf.Vec3h(1.0, 1.0, 1.0)] * n_joints)
    )
    rotations_attr = anim.GetRotationsAttr()
    translations_attr = anim.GetTranslationsAttr()

    t0 = time.time()
    for ti in range(T):
        rot_t = Rt.from_quat(_wxyz_to_xyzw(body_xquat[ti]))
        world_R = np.zeros((n_joints, 3, 3))
        for j in range(n_joints):
            world_R[j] = rot_t[mass_idx[j]].as_matrix()
        local_R = np.zeros((n_joints, 3, 3))
        local_t = rest_local_t.copy()
        for j in range(n_joints):
            pi = parent_idx[j]
            if pi == -1:
                local_R[j] = world_R[j]
                local_t[j] = body_xpos[ti, mass_idx[j]]
            else:
                local_R[j] = world_R[pi].T @ world_R[j]

        quats = []
        for j in range(n_joints):
            qx, qy, qz, qw = Rt.from_matrix(local_R[j]).as_quat()
            quats.append(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
        rotations_attr.Set(Vt.QuatfArray(quats), Usd.TimeCode(ti))
        trans = [Gf.Vec3f(float(local_t[j, 0]), float(local_t[j, 1]),
                          float(local_t[j, 2])) for j in range(n_joints)]
        translations_attr.Set(Vt.Vec3fArray(trans), Usd.TimeCode(ti))
        if (ti + 1) % 100 == 0:
            print(f"  {ti + 1}/{T} ({time.time() - t0:.1f}s)")

    stage.GetRootLayer().Save()
    print(f"  wrote {out_usd} ({out_usd.stat().st_size / 1024 / 1024:.1f} MB)")


# ---------------------------------------------------------------------------
# Stage 3: retarget Newton
# ---------------------------------------------------------------------------

def stage3_retarget(source_usd: Path, out_usd: Path, max_frames: int) -> None:
    """Run tools/retarget/retarget_newton.py as subprocess."""
    _hdr(f"Stage 3: Newton IK retarget -> {out_usd.name}")
    out_usd.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "retarget_newton.py",
        "--source", str(source_usd.resolve()),
        "--human-usd", str((PROJECT_ROOT / "data/Human/Human.usd").resolve()),
        "--joint-mapping", str((PROJECT_ROOT / "data/Human/joint_mapping.json").resolve()),
        "--out", str(out_usd.resolve()),
        "--max-frames", str(max_frames),
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT / "tools/retarget")


# ---------------------------------------------------------------------------
# Stage 4: baked ragdoll USD -> otto_motion.json
# ---------------------------------------------------------------------------

def stage4_to_motion_json(ragdoll_usd: Path, out_json: Path,
                          human_usd: Path,
                          bone_root: str = "/Human/Ragdoll/Bones") -> None:
    """Walk bone_root and dump per-frame world transforms in otto JSON format.

    The motion JSON's joint_names must match the ragdoll bone names that
    BoneAnimation looks up (1:1, parent-prim name of each Mesh under bone_root).
    rest_transforms come from human_usd at default time; per-frame transforms
    from ragdoll_usd at each animated time sample. We set joint_rest =
    bone_world_rest so BoneAnimation's offset reduces to identity and
    joint_world(t) == bone_world(t).
    """
    _hdr(f"Stage 4: bake -> motion JSON {out_json.name}")
    from pxr import Usd, UsdGeom

    def _gf_to_np(m):
        return np.array([[m[i][j] for j in range(4)] for i in range(4)],
                        dtype=np.float64)

    def _collect_bone_names(stage):
        """Mirror mesh_io._load_bone_usd traversal: every Mesh under bone_root
        contributes its parent prim's name as the bone id."""
        root = stage.GetPrimAtPath(bone_root)
        if not root or not root.IsValid():
            raise RuntimeError(f"{bone_root} missing in stage")
        names: list[str] = []
        mesh_prims: list = []
        seen: set[str] = set()
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            m = UsdGeom.Mesh(prim)
            if m.GetPointsAttr().Get() is None:
                continue
            parent = prim.GetParent()
            name = parent.GetName() if parent and parent.IsValid() else prim.GetName()
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
            mesh_prims.append(prim)
        return names, mesh_prims

    # 1) rest from Human.usd (default time = identity-rest xformOp)
    h_stage = Usd.Stage.Open(str(human_usd))
    h_names, h_meshes = _collect_bone_names(h_stage)
    h_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rest_transforms = [
        _gf_to_np(h_cache.GetLocalToWorldTransform(p)) for p in h_meshes
    ]

    # 2) per-frame from baked ragdoll USD
    r_stage = Usd.Stage.Open(str(ragdoll_usd))
    fps = float(r_stage.GetTimeCodesPerSecond() or 30.0)
    t0 = int(r_stage.GetStartTimeCode())
    t1 = int(r_stage.GetEndTimeCode())
    r_names, r_meshes = _collect_bone_names(r_stage)
    if r_names != h_names:
        raise RuntimeError(
            "bone-name traversal mismatch between Human.usd and ragdoll_anim.usd. "
            f"Human n={len(h_names)} ragdoll n={len(r_names)}"
        )

    frames = []
    for f in range(t0, t1 + 1):
        cache = UsdGeom.XformCache(float(f))
        xforms = [_gf_to_np(cache.GetLocalToWorldTransform(p)) for p in r_meshes]
        frames.append({"frame": f, "transforms": [m.flatten().tolist()
                                                  for m in xforms]})

    out = {
        "fps": fps,
        "frame_start": t0,
        "frame_end": t1,
        "joint_names": h_names,
        "rest_transforms": [m.flatten().tolist() for m in rest_transforms],
        "frames": frames,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(out, fh)
    print(f"  wrote {out_json} ({len(h_names)} bones, {len(frames)} frames, "
          f"@ {fps} fps)")


# ---------------------------------------------------------------------------
# Stage 5: example_drive
# ---------------------------------------------------------------------------

def stage5_example_drive(base_cfg: Path, motion_json: Path,
                         out_usd: Path, frames: int | None) -> None:
    """Run example_drive headless with a temp config pointing at our motion."""
    _hdr(f"Stage 5: example_drive -> {out_usd.name}")
    with open(base_cfg) as fh:
        cfg = json.load(fh)
    cfg["motion_json_path"] = str(motion_json.resolve())
    if frames is not None:
        cfg["nframes"] = int(frames)
    tmp_cfg = out_usd.parent / "muscle_cfg.json"
    out_usd.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_cfg, "w") as fh:
        json.dump(cfg, fh, indent=2)

    cmd = [
        sys.executable, "-m", "muscle.coupleOneWay.example_drive",
        "--config", str(tmp_cfg.resolve()),
        "--auto",
        "--output", str(out_usd.resolve()),
        "--device", "cuda",
    ]
    if frames is not None:
        cmd += ["--frames", str(frames)]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


# ---------------------------------------------------------------------------
# Stage 6: usd_to_mp4
# ---------------------------------------------------------------------------

def stage6_render_mp4(usd: Path, mp4: Path) -> None:
    _hdr(f"Stage 6: render MP4 -> {mp4.name}")
    usd_root = Path(
        "/lus/lfs1aip2/scratch/u6ey/chunleili.u6ey/Dev/RLMuscle/.conda_envs/usd"
    )
    ffmpeg_bin = Path(
        "/lus/lfs1aip2/scratch/u6ey/chunleili.u6ey/Dev/miniforge3/envs/lerobot/bin"
    )
    if not usd_root.exists():
        raise RuntimeError(f"USD render env missing: {usd_root}")

    # Manual Xvfb (no xvfb-run on this node)
    xvfb = subprocess.Popen(
        ["/usr/bin/Xvfb", ":99", "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(2)
        env = os.environ.copy()
        env["DISPLAY"] = ":99"
        env["USD_ROOT"] = str(usd_root)
        env["PATH"] = f"{ffmpeg_bin}{os.pathsep}{env.get('PATH', '')}"
        # MuscleSim's USD writes points on /root/muscle and /root/bone (no
        # animated prim xforms), so --follow can't track. Use a static camera.
        cmd = [
            sys.executable, "tools/usd_to_mp4.py", str(usd),
            "--follow", "none",
            "--output", str(mp4),
            "--stride", "1", "--width", "720",
            "--cam-distance", "3.0", "--cam-elevation", "0.5",
        ]
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)
    finally:
        xvfb.terminate()
        try:
            xvfb.wait(timeout=5)
        except subprocess.TimeoutExpired:
            xvfb.kill()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_stages(s: str) -> set[int]:
    if not s:
        return set()
    return {int(x) for x in s.split(",") if x.strip()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="walk_max",
                   help="Stem of .pt under data/rl_mass/checkpoints/")
    p.add_argument("--muscle-cfg", default="data/muscle/config/fullbody_drive.json")
    p.add_argument("--run-name", default=None,
                   help="Output sub-dir under output/pipeline/ (default: ckpt name)")
    p.add_argument("--frames", type=int, default=60,
                   help="Cap motion length (and example_drive --frames)")
    p.add_argument("--max-eval-steps", type=int, default=300,
                   help="rl_mass.eval --max-steps cap")
    p.add_argument("--asset", default=str(SISTER_ASSET),
                   help="UsdSkel template (snu_ysy_v3.usd from sister project)")
    p.add_argument("--skip-stages", default="",
                   help="Comma-separated stage indices to skip (e.g. 1,2,3)")
    p.add_argument("--mp4", action="store_true",
                   help="Also run stage 6 (usd_to_mp4 render). Off by default.")
    args = p.parse_args()

    run_name = args.run_name or args.ckpt
    out_root = PROJECT_ROOT / "output" / "pipeline" / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] run dir: {out_root}")

    skip = parse_stages(args.skip_stages)

    traj = out_root / "traj.npz"
    source_skel = out_root / "source_skel.usd"
    ragdoll = out_root / "ragdoll_anim.usd"
    motion_json = out_root / "motion.json"
    muscle_usd = out_root / "muscle.usd"
    mp4 = out_root / "muscle.mp4"

    if 1 not in skip:
        stage1_eval(args.ckpt, args.max_eval_steps, out_root)
    else:
        print("[skip] stage 1")

    if 2 not in skip:
        asset = Path(args.asset)
        if not asset.exists():
            raise RuntimeError(f"Source asset missing: {asset}")
        stage2_bake_skel(traj, asset, source_skel, max_frames=args.frames)
    else:
        print("[skip] stage 2")

    if 3 not in skip:
        stage3_retarget(source_skel, ragdoll, max_frames=args.frames)
    else:
        print("[skip] stage 3")

    if 4 not in skip:
        stage4_to_motion_json(
            ragdoll, motion_json,
            human_usd=PROJECT_ROOT / "data/Human/Human.usd",
        )
    else:
        print("[skip] stage 4")

    if 5 not in skip:
        stage5_example_drive(
            base_cfg=PROJECT_ROOT / args.muscle_cfg,
            motion_json=motion_json,
            out_usd=muscle_usd,
            frames=args.frames,
        )
    else:
        print("[skip] stage 5")

    if args.mp4 and 6 not in skip:
        stage6_render_mp4(muscle_usd, mp4)

    _hdr("DONE")
    print(f"traj        : {traj}")
    print(f"source_skel : {source_skel}")
    print(f"ragdoll     : {ragdoll}")
    print(f"motion_json : {motion_json}")
    print(f"muscle_usd  : {muscle_usd}")
    if args.mp4:
        print(f"mp4         : {mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
