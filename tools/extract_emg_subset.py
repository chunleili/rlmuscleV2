"""Extract a (speed, leg) EMG-envelope subset for all participants from the
Moreira 2021 raw ASCII zip into a flat, loader-friendly tree.

The 1.3 GB raw zip lives at ``data/rl_mass/emg_moreira/raw/moreira_ascii.zip``
(git-ignored). This script pulls just the ``EMG envelope`` text files for one
walking-speed code and one leg, across every participant that has them, and
rewrites them as::

    <out_dir>/<Participant>/T<k>.txt

so :class:`rl_mass.finger.gt.EMGLoader` can read each participant directory.
Re-runnable; existing files are overwritten.

Usage:
    uv run python tools/extract_emg_subset.py \
        --speed V4 --leg R \
        --out data/rl_mass/emg_moreira/V4_R
"""
from __future__ import annotations

import argparse
import os
import re
import zipfile

_DEFAULT_ZIP = "data/rl_mass/emg_moreira/raw/moreira_ascii.zip"
# .../<Participant8>/Processed_Data/<V4>/<R>/EMG envelope/<T3.txt>
_PAT = re.compile(
    r"(Participant\d+)/Processed_Data/(V\d+)/([LR])/EMG envelope/([^/]+\.txt)$"
)


def extract(zip_path: str, speed: str, leg: str, out_dir: str) -> dict:
    """Extract one (speed, leg) EMG-envelope subset for all participants.

    Returns a dict ``{participant: n_trials}``.
    """
    counts: dict[str, int] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            m = _PAT.search(info.filename)
            if not m:
                continue
            participant, vspeed, vleg, trial = m.groups()
            if vspeed != speed or vleg != leg:
                continue
            dest_dir = os.path.join(out_dir, participant)
            os.makedirs(dest_dir, exist_ok=True)
            with zf.open(info) as src, open(os.path.join(dest_dir, trial), "wb") as dst:
                dst.write(src.read())
            counts[participant] = counts.get(participant, 0) + 1
    return counts


def main():
    p = argparse.ArgumentParser(description="Extract Moreira EMG-envelope subset")
    p.add_argument("--zip", default=_DEFAULT_ZIP, help="path to moreira_ascii.zip")
    p.add_argument("--speed", default="V4", help="speed code (V1..V4)")
    p.add_argument("--leg", default="R", choices=["L", "R"], help="leg")
    p.add_argument("--out", required=True, help="output root dir")
    a = p.parse_args()

    counts = extract(a.zip, a.speed, a.leg, a.out)
    total_trials = sum(counts.values())
    print(f"extracted {a.speed}/{a.leg}: {len(counts)} participants, "
          f"{total_trials} trial files -> {a.out}")
    for k in sorted(counts, key=lambda s: int(s.replace("Participant", ""))):
        print(f"  {k}: {counts[k]}")


if __name__ == "__main__":
    main()
