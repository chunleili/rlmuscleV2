"""Compare our Warp lion output against the taichi reference output.

The taichi reference (realtimeMuscle/engine/arap_hpbd.py) emits per-frame
.geo files when use_external_constraints=true; this script can read either
.geo or .ply, so it works regardless of which one is on disk.

Usage:
    # First, populate the reference:
    cd ~/Dev/realtimeMuscle
    python engine/arap_hpbd.py -use_json=1 \
        -json_path=./data/lion/config/lion_nomg.json \
        -end_frame=90 -export_mesh=1 -out_dir=result/lion_ref

    # Then verify (run from rlmuscleV2-physics root):
    python tests/verify_lion_parity.py \
        --ours   result/lion/mesh \
        --ref    ~/Dev/realtimeMuscle/result/lion_ref/mesh \
        --frames 1 10 45 90
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_positions(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        import meshio

        m = meshio.read(str(path))
        return np.asarray(m.points, dtype=np.float64)
    if suffix == ".geo":
        from muscle.util.geo_io import GeoFile

        return GeoFile(str(path)).positions.astype(np.float64)
    raise ValueError(f"unsupported extension: {path}")


def find_frame_file(directory: Path, frame: int) -> Path | None:
    for ext in (".ply", ".geo"):
        candidates = list(directory.glob(f"{frame:04d}{ext}"))
        if candidates:
            return candidates[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[1, 10, 45, 90])
    ap.add_argument("--rel_tol", type=float, default=1e-4,
                    help="max(per-vertex distance) / mesh_diagonal must be below this")
    args = ap.parse_args()

    ours_dir = Path(args.ours).expanduser()
    ref_dir = Path(args.ref).expanduser()

    failed = 0
    for f in args.frames:
        ours_p = find_frame_file(ours_dir, f)
        ref_p = find_frame_file(ref_dir, f)
        if ours_p is None:
            print(f"frame {f:>3d}: ours missing in {ours_dir}")
            failed += 1
            continue
        if ref_p is None:
            print(f"frame {f:>3d}: ref missing in {ref_dir}")
            failed += 1
            continue

        ours = load_positions(ours_p)
        ref = load_positions(ref_p)
        if ours.shape != ref.shape:
            print(f"frame {f:>3d}: shape mismatch ours={ours.shape} ref={ref.shape}")
            failed += 1
            continue

        d = np.linalg.norm(ours - ref, axis=1)
        diag = float(np.linalg.norm(ref.max(0) - ref.min(0)))
        max_err = float(d.max())
        rms_err = float(np.sqrt((d ** 2).mean()))
        rel_err = max_err / diag if diag > 0 else float("inf")
        status = "OK" if rel_err < args.rel_tol else "FAIL"
        print(
            f"frame {f:>3d}: "
            f"max={max_err:.3e}  rms={rms_err:.3e}  "
            f"rel={rel_err:.3e}  diag={diag:.3e}  [{status}]"
        )
        if status == "FAIL":
            failed += 1

    if failed:
        raise SystemExit(f"{failed} frame(s) failed parity")
    print("PARITY OK")


if __name__ == "__main__":
    main()
