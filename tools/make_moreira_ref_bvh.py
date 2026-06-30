"""Author a kinematics-corrected walk BVH from Moreira 2021 ankle angles.

The kinematics fairness gate (tools/validate_walk_kinematics.py) showed the
stock ``walk.bvh`` ankle over-rotates ~2x real (RightFoot Xrotation range 49°
vs Moreira 25°), while hip/knee already match. This script surgically replaces
ONLY the ankle channel (RightFoot / LeftFoot ``Xrotation``) with the Moreira
cohort-mean ankle profile — range/shape from real gait — keeping the root,
upper body, hip and knee channels untouched.

Phase alignment: the stock ankle channel ``B(t)`` is already phase-correct for
this gait (it tracks the same walk, just with the wrong amplitude). So we keep
``B``'s phase via circular cross-correlation and substitute its amplitude with
Moreira's:  ``new(t) = mean(B) + s * (A_aligned(t) - mean(A))``  where ``A`` is
the Moreira ankle resampled to the BVH frame count, ``A_aligned`` is circularly
shifted to best match ``B``, and ``s = ±1`` from the sign of the best lag's
correlation (the SNU/BVH ankle convention may be flipped vs Moreira).

Usage:
    uv run python tools/make_moreira_ref_bvh.py \
        --src data/rl_mass/motion/walk.bvh \
        --zip data/rl_mass/emg_moreira/raw/moreira_ascii.zip \
        --speed V4 \
        --out data/rl_mass/motion/walk_moreira.bvh
"""
from __future__ import annotations

import argparse
import re
import zipfile

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# BVH parsing (channel order + motion matrix)
# ──────────────────────────────────────────────────────────────────────────────

def parse_bvh(path: str):
    """Return (lines, channels, motion, motion_start).

    ``channels`` is a list of (joint_name, channel_name) in motion-column order.
    ``motion`` is (n_frames, n_channels). ``motion_start`` is the line index of
    the first motion data row (so the header can be re-emitted verbatim).
    """
    lines = open(path).read().split("\n")
    channels: list[tuple[str, str]] = []
    cur = None
    mi = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        m = re.match(r"(ROOT|JOINT)\s+(\S+)", s)
        if m:
            cur = m.group(2)
            continue
        if s.startswith("CHANNELS"):
            channels += [(cur, c) for c in s.split()[2:]]
        if s == "MOTION":
            mi = i
            break
    n_frames = int(lines[mi + 1].split()[1])
    start = mi + 3
    motion = np.array([[float(x) for x in lines[start + k].split()]
                       for k in range(n_frames)])
    return lines, channels, motion, start


def col_of(channels, joint: str, chan: str) -> int:
    for k, (j, c) in enumerate(channels):
        if j == joint and c == chan:
            return k
    raise KeyError(f"channel {joint}.{chan} not found")


# ──────────────────────────────────────────────────────────────────────────────
# Moreira ankle profile (cohort mean, one leg)
# ──────────────────────────────────────────────────────────────────────────────

def moreira_ankle(zip_path: str, speed: str, leg: str, n: int) -> np.ndarray:
    """Cohort-mean sagittal ankle angle (deg) for one leg → (n,) over 0-100%."""
    per_subject: list[np.ndarray] = []
    with zipfile.ZipFile(zip_path) as zf:
        subs: dict[str, list[str]] = {}
        for nm in zf.namelist():
            if f"/Processed_Data/{speed}/{leg}/Angles/" in nm and nm.endswith(".txt"):
                pid = nm.split("/Processed_Data/")[0].split("/")[-1]
                subs.setdefault(pid, []).append(nm)
        for pid in sorted(subs):
            trials = []
            for nm in sorted(subs[pid]):
                txt = zf.read(nm).decode().strip().split("\n")
                hdr = txt[0].split("\t")
                data = np.array([[float(v) for v in ln.split("\t")] for ln in txt[1:]])
                cols = [i for i, h in enumerate(hdr)
                        if h in ("St1_Ankle_X", "St2_Ankle_X")]
                col = data[:, cols].mean(axis=1)
                x = np.linspace(0, 100, len(col))
                trials.append(np.interp(np.linspace(0, 100, n), x, col))
            if trials:
                per_subject.append(np.mean(trials, axis=0))
    if not per_subject:
        raise ValueError(f"No Moreira {speed}/{leg} ankle subjects in {zip_path}")
    return np.stack(per_subject).mean(axis=0)


def align_amplitude(B: np.ndarray, A_cycle: np.ndarray) -> np.ndarray:
    """Replace ``B``'s amplitude with Moreira ``A`` keeping ``B``'s phase.

    ``A_cycle`` is the Moreira ankle over a single gait cycle (any length); it is
    resampled to ``len(B)``, circularly shifted to best correlate with ``B``, and
    its sign matched. Returns the new channel (same length/mean as ``B``).
    """
    nB = len(B)
    A = np.interp(np.linspace(0, 1, nB, endpoint=False),
                  np.linspace(0, 1, len(A_cycle), endpoint=False), A_cycle)
    Bc = B - B.mean()
    Ac = A - A.mean()
    # circular cross-correlation → best lag and its (signed) correlation
    corr = np.array([np.dot(Bc, np.roll(Ac, k)) for k in range(nB)])
    k = int(np.argmax(np.abs(corr)))
    s = float(np.sign(corr[k])) or 1.0
    A_aligned = np.roll(Ac, k) * s
    return B.mean() + A_aligned


def main():
    p = argparse.ArgumentParser(description="Author Moreira-corrected walk BVH")
    p.add_argument("--src", default="data/rl_mass/motion/walk.bvh")
    p.add_argument("--zip", default="data/rl_mass/emg_moreira/raw/moreira_ascii.zip")
    p.add_argument("--speed", default="V4")
    p.add_argument("--out", default="data/rl_mass/motion/walk_moreira.bvh")
    a = p.parse_args()

    lines, channels, motion, start = parse_bvh(a.src)
    n_frames = motion.shape[0]

    for leg, joint in (("R", "Character1_RightFoot"), ("L", "Character1_LeftFoot")):
        col = col_of(channels, joint, "Xrotation")
        B = motion[:, col].copy()
        A = moreira_ankle(a.zip, a.speed, leg, n_frames)
        new = align_amplitude(B, A)
        print(f"{joint:26s} Xrot  range {B.max()-B.min():5.1f}° -> "
              f"{new.max()-new.min():5.1f}°  (Moreira target ~{A.max()-A.min():.1f}°)")
        motion[:, col] = new

    # Re-emit: header verbatim, then corrected motion rows.
    out_lines = lines[:start]
    for k in range(n_frames):
        out_lines.append(" ".join(f"{v:.6f}" for v in motion[k]))
    # preserve any trailing lines (usually a single empty line)
    out_lines += lines[start + n_frames:]
    with open(a.out, "w") as f:
        f.write("\n".join(out_lines))
    print(f"wrote {a.out}  ({n_frames} frames)")


if __name__ == "__main__":
    main()
