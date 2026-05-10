"""Render a USD animation to MP4 via usdrecord (Hydra/Storm) + ffmpeg.

Uses the conda openusd environment's usdrecord for rendering, with xvfb-run
for headless OpenGL support. Outputs image sequence then stitches to MP4.

Usage:
    # Basic
    uv run python scripts/usd_to_mp4.py output/anim/example.usda

    # With follow camera (tracks a prim, side view)
    uv run python scripts/usd_to_mp4.py input.usd --follow /root/model/shapes/shape_1/instance_0

    # Skip frames for faster render (5x speedup)
    uv run python scripts/usd_to_mp4.py input.usd --stride 5

    # Full options
    uv run python scripts/usd_to_mp4.py input.usd -o out.mp4 --width 1280 --stride 5 \\
        --follow /root/model/shapes/shape_1/instance_0 --cam-distance 5 --cam-elevation 0.3
"""

import argparse
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# Project-local conda env with openusd (install: conda create -p .conda_envs/usd -c conda-forge openusd)
USD_ROOT = os.environ.get("USD_ROOT", str(PROJECT_ROOT / ".conda_envs" / "usd"))

CAM_PRIM_PATH = "/render_cam"


def get_stage_info(usd_path: str) -> tuple[int, int, float]:
    """Read frame range and fps from USD stage metadata."""
    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)
    start = int(stage.GetStartTimeCode())
    end = int(stage.GetEndTimeCode())
    fps = stage.GetTimeCodesPerSecond()
    return start, end, fps


def find_follow_target(usd_path: str) -> str | None:
    """Auto-detect the torso/pelvis prim to follow.
    Among animated prims, picks the one closest to the vertical center of mass
    (highest Z that isn't a head/extremity)."""
    from pxr import Usd, UsdGeom
    import numpy as np

    stage = Usd.Stage.Open(usd_path)
    start = int(stage.GetStartTimeCode())
    end = int(stage.GetEndTimeCode())

    candidates = []  # (prim_path, position)
    for p in stage.Traverse():
        xf = UsdGeom.Xformable(p)
        for op in xf.GetOrderedXformOps():
            if "translate" in str(op.GetOpName()) and op.GetNumTimeSamples() > 1:
                path = str(p.GetPath())
                # Prioritize pelvis/torso by name
                name = path.lower()
                if any(k in name for k in ("pelvis", "hips", "torso", "spine")):
                    print(f"Auto-detected follow target: {path} (pelvis/torso match)")
                    return path
                p0 = np.array(op.Get(start))
                candidates.append((path, p0))

    if not candidates:
        return None

    # Find the geometric center of all animated prims, pick the closest one
    all_pos = np.array([c[1] for c in candidates])
    center = all_pos.mean(axis=0)
    dists = [float(np.linalg.norm(c[1] - center)) for c in candidates]
    best_idx = int(np.argmin(dists))
    best = candidates[best_idx]
    print(f"Auto-detected follow target: {best[0]} (closest to center, dist={dists[best_idx]:.3f})")
    return best[0]


def add_follow_camera(
    usd_path: str,
    out_path: str,
    follow_prim: str,
    start: int,
    end: int,
    stride: int,
    cam_distance: float,
    cam_elevation: float,
) -> str:
    """Add a follow camera that tracks a prim's translate op. Returns path to new USD."""
    from pxr import Usd, UsdGeom, Gf

    shutil.copy(usd_path, out_path)
    stage = Usd.Stage.Open(out_path)

    # Find the translate op on the target prim
    prim = stage.GetPrimAtPath(follow_prim)
    if not prim:
        raise ValueError(f"Prim not found: {follow_prim}")

    xf = UsdGeom.Xformable(prim)
    translate_op = None
    transform_xop = None
    for op in xf.GetOrderedXformOps():
        name = str(op.GetOpName())
        if "translate" in name:
            translate_op = op
            break
        if name == "xformOp:transform":
            transform_xop = op

    if (not translate_op or translate_op.GetNumTimeSamples() == 0) and (
        transform_xop is None or transform_xop.GetNumTimeSamples() == 0
    ):
        raise ValueError(f"No animated translate / transform op on {follow_prim}")

    def _sample_pos(f):
        if translate_op is not None:
            v = translate_op.Get(f)
            return Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))
        M = transform_xop.Get(f)
        t = M.ExtractTranslation()
        return Gf.Vec3d(float(t[0]), float(t[1]), float(t[2]))

    up_axis = UsdGeom.GetStageUpAxis(stage)

    # Create camera prim
    cam = UsdGeom.Camera.Define(stage, CAM_PRIM_PATH)
    cam.GetFocalLengthAttr().Set(35.0)
    xf_cam = UsdGeom.Xformable(cam.GetPrim())
    xf_cam.ClearXformOpOrder()
    transform_op = xf_cam.AddTransformOp()

    # Camera offset & world_up depend on stage up-axis.
    # Z-up: side view from +Y, vertical = Z, lock Z (height).
    # Y-up: side view from +X, vertical = Y, lock Y (height).
    if up_axis == "Y":
        cam_offset = Gf.Vec3d(cam_distance, cam_elevation, 0.0)
        world_up = Gf.Vec3d(0, 1, 0)
        vertical_idx = 1
    else:
        cam_offset = Gf.Vec3d(0.0, cam_distance, cam_elevation)
        world_up = Gf.Vec3d(0, 0, 1)
        vertical_idx = 2

    # Sample initial vertical to lock vertical position (avoid bounce shake)
    init_pos = _sample_pos(start)
    locked_v = float(init_pos[vertical_idx])

    frames = list(range(start, end + 1, stride))
    for f in frames:
        pos = _sample_pos(f)
        # Follow horizontal axes; lock vertical to initial height.
        coords = [float(pos[0]), float(pos[1]), float(pos[2])]
        coords[vertical_idx] = locked_v
        target = Gf.Vec3d(*coords)
        cam_pos = target + cam_offset

        forward = (target - cam_pos).GetNormalized()
        right = Gf.Cross(forward, world_up).GetNormalized()
        up = Gf.Cross(right, forward).GetNormalized()

        mat = Gf.Matrix4d()
        mat.SetIdentity()
        mat[0] = (*right, 0)
        mat[1] = (*up, 0)
        mat[2] = (*(-forward), 0)
        mat[3] = (*cam_pos, 1)
        transform_op.Set(mat, f)

    stage.Save()
    print(f"Follow camera added: tracking {follow_prim}, {len(frames)} keyframes")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Render USD animation to MP4 via usdrecord")
    parser.add_argument("input", help="Path to .usd/.usda/.usdc file")
    parser.add_argument("-o", "--output", help="Output .mp4 path (default: same name as input)")
    parser.add_argument("--fps", type=int, default=0, help="Output FPS (0 = from USD metadata)")
    parser.add_argument("--start", type=int, default=-1, help="Start frame (-1 = from USD)")
    parser.add_argument("--end", type=int, default=-1, help="End frame (-1 = from USD)")
    parser.add_argument("--width", type=int, default=1280, help="Image width (default: 1280)")
    parser.add_argument("--complexity", choices=["low", "medium", "high", "veryhigh"], default="low")
    parser.add_argument("--stride", type=int, default=10, help="Frame stride (default: 10)")
    parser.add_argument("--follow", type=str, default="auto", help="Prim path to follow (default: auto-detect, 'none' to disable)")
    parser.add_argument("--cam-distance", type=float, default=5.0, help="Follow camera distance (default: 5.0)")
    parser.add_argument("--cam-elevation", type=float, default=0.3, help="Follow camera elevation offset (default: 0.3)")
    parser.add_argument("--camera", type=str, default=None, help="Use existing camera prim (overrides --follow)")
    parser.add_argument("--keep-frames", action="store_true", help="Keep intermediate PNG frames")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1

    # Read frame range from USD if not specified
    usd_start, usd_end, usd_fps = get_stage_info(str(input_path))
    start = args.start if args.start >= 0 else usd_start
    end = args.end if args.end >= 0 else usd_end
    fps = args.fps or int(usd_fps)
    stride = args.stride

    frame_spec = f"{start}:{end}" if stride == 1 else f"{start}:{end}x{stride}"
    print(f"Frames: {frame_spec}, FPS: {fps}, Width: {args.width}")

    # Determine camera and prepare USD
    render_usd = str(input_path)
    camera_name = args.camera
    tmp_usd = None

    if not args.camera:
        follow_prim = None
        if args.follow == "auto":
            follow_prim = find_follow_target(str(input_path))
        elif args.follow and args.follow != "none":
            follow_prim = args.follow

        if follow_prim:
            tmp_usd = str(input_path.with_name(input_path.stem + "_cam_tmp.usd"))
            render_usd = add_follow_camera(
                str(input_path), tmp_usd, follow_prim,
                start, end, stride, args.cam_distance, args.cam_elevation,
            )
            camera_name = "render_cam"

    # Create temp dir for PNG sequence
    tmp_dir = tempfile.mkdtemp(prefix="usdrecord_")
    frame_pattern = str(Path(tmp_dir) / "frame.####.png")

    # Build usdrecord command, wrapped with xvfb-run for headless OpenGL.
    # Prefer USD_ROOT/bin/python when present; otherwise fall back to sys.executable
    # so a USD_ROOT that only ships libs/binaries (e.g. ~/opt/openusd) works via the
    # project .venv's pxr-compatible Python ABI.
    _root_python = Path(USD_ROOT) / "bin" / "python"
    usd_python = str(_root_python) if _root_python.exists() else sys.executable
    usd_record = str(Path(USD_ROOT) / "bin" / "usdrecord")
    usdrecord_cmd = [
        usd_python, usd_record,
        "--frames", frame_spec,
        "--imageWidth", str(args.width),
        "--complexity", args.complexity,
    ]
    if camera_name:
        usdrecord_cmd.extend(["--camera", camera_name])
    usdrecord_cmd.extend([render_usd, frame_pattern])

    xvfb_bin = shutil.which("xvfb-run")
    if xvfb_bin:
        cmd = [xvfb_bin, "-a", "-s", "-screen 0 1920x1080x24"] + usdrecord_cmd
    else:
        # No xvfb: rely on usdrecord's GPU path (OK on a render node with NVIDIA
        # driver) or fall back to --disableGpu via env var.
        if os.environ.get("USDRECORD_DISABLE_GPU") == "1":
            usdrecord_cmd.insert(usdrecord_cmd.index("--frames"), "--disableGpu")
        cmd = usdrecord_cmd
        print("(xvfb-run not found — running usdrecord directly)")

    env = os.environ.copy()
    # Strip uv/venv activation so conda usdrecord uses its own pxr (not the caller's .venv pxr).
    for _v in ("VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"):
        env.pop(_v, None)
    env["PATH"] = ":".join(p for p in env.get("PATH", "").split(":") if "/.venv/" not in p)
    env["PXR_PLUGINPATH_NAME"] = f"{USD_ROOT}/lib/usd:{USD_ROOT}/plugin/usd"
    # Prepend USD_ROOT/lib so openusd's libs win over any venv-shipped ABI-mismatched copies.
    env["LD_LIBRARY_PATH"] = str(Path(USD_ROOT) / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    # Put openusd's bundled pxr first on PYTHONPATH so usdrecord imports the matching plugins.
    _usd_pypath = Path(USD_ROOT) / "lib" / "python"
    if _usd_pypath.exists():
        env["PYTHONPATH"] = str(_usd_pypath) + ":" + env.get("PYTHONPATH", "")
    else:
        # No bundled pxr under USD_ROOT (common with conda openusd) → don't leak caller's PYTHONPATH,
        # which would make the conda python find the caller's .venv pxr and fail to import UsdAppUtils.
        env.pop("PYTHONPATH", None)

    print("Running usdrecord...")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"usdrecord failed:\n{result.stderr}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return 1
    if result.stderr:
        print(result.stderr, end="")

    # Check that frames were generated
    pngs = sorted(Path(tmp_dir).glob("frame.*.png"))
    if not pngs:
        print("Error: usdrecord produced no frames")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return 1
    print(f"Rendered {len(pngs)} frames")

    # Rename frames to sequential numbering for ffmpeg
    for i, png in enumerate(pngs):
        seq_name = Path(tmp_dir) / f"seq.{i:04d}.png"
        png.rename(seq_name)

    # Stitch with ffmpeg (keep original fps — stride means faster playback)
    out_path = args.output or str(input_path.with_suffix(".mp4"))
    ffmpeg_input = str(Path(tmp_dir) / "seq.%04d.png")
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg  # type: ignore
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = "ffmpeg"
    ffmpeg_cmd = [
        ffmpeg_bin, "-y",
        "-framerate", str(fps),
        "-i", ffmpeg_input,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # libx264 needs even dims
        out_path,
    ]

    print(f"Encoding MP4: {out_path}")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg failed:\n{result.stderr}")
        return 1

    # Cleanup
    if not args.keep_frames:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    if tmp_usd and Path(tmp_usd).exists():
        Path(tmp_usd).unlink()

    print(f"Done: {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
