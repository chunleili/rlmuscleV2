"""Convert an OpenSim Storage `.mot` motion to a TrajectoryRef `.npz`.

Selects the requested coordinate columns (in the given order = MJCF qpos
order), converts rotational coordinates degrees->radians when the file is
inDegrees=yes, and writes a single-episode npz in the format consumed by
rl_mass.finger.TrajectoryRef: keys `n_episodes`, `dt`, `episode_0_traj_qpos`.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def read_mot(path):
    """Parse a Storage .mot. Returns (col_names, time(T,), data(T,ncols), in_degrees)."""
    with open(path) as f:
        lines = f.read().splitlines()
    in_degrees = False
    i = 0
    for i, ln in enumerate(lines):
        s = ln.strip().lower()
        if s.startswith("indegrees"):
            in_degrees = s.split("=")[1].strip() == "yes"
        if s == "endheader":
            break
    header = lines[i + 1].split("\t")
    rows = [r for r in lines[i + 2:] if r.strip()]
    mat = np.array([[float(x) for x in r.split("\t")] for r in rows], dtype=np.float64)
    time = mat[:, 0]
    col_names = header[1:]
    data = mat[:, 1:]
    return col_names, time, data, in_degrees


def mot_to_npz(mot_path, out_path, coords, rotational):
    col_names, time, data, in_degrees = read_mot(mot_path)
    idx = [col_names.index(c) for c in coords]  # raises ValueError if missing
    q = data[:, idx].copy()
    if in_degrees:
        for j, c in enumerate(coords):
            if c in rotational:
                q[:, j] = np.deg2rad(q[:, j])
    dt = float(time[1] - time[0]) if len(time) > 1 else 1.0
    np.savez(
        out_path,
        n_episodes=np.int64(1),
        dt=np.float64(dt),
        episode_0_traj_qpos=q.astype(np.float64),
    )


def main():
    p = argparse.ArgumentParser(description="OpenSim .mot -> TrajectoryRef .npz")
    p.add_argument("-i", "--mot", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--coords", required=True,
                   help="Comma-separated coordinate names in qpos order")
    p.add_argument("--rotational", default="",
                   help="Comma-separated subset of --coords that are rotational")
    a = p.parse_args()
    coords = [c.strip() for c in a.coords.split(",") if c.strip()]
    rot = {c.strip() for c in a.rotational.split(",") if c.strip()}
    mot_to_npz(a.mot, a.out, coords, rot)
    print(json.dumps({"out": a.out, "coords": coords, "rotational": sorted(rot)}))


if __name__ == "__main__":
    main()
