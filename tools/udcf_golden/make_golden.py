"""Convert a UDCF_DUMP CSV (from the gaia_dump.patch instrumentation of
Gaia_UDCF) into tests/golden/udcf_pairs.npz.

Record layout (71 fields) is documented in
docs/superpowers/plans/2026-06-11-dcd-collision.md and must stay in sync
with the fprintf order in gaia_dump.patch.

Usage:
    .venv/bin/python tools/udcf_golden/make_golden.py /tmp/udcf_golden.csv \
        tests/golden/udcf_pairs.npz
"""
import sys

import numpy as np

MAX_STAGE0 = 4000      # interval rejects (already 1/97 subsampled in C++)
MAX_STAGE1 = 4000      # offset rejects
MAX_APPLIED = 12000    # stage 2 + 3
SEED = 0


def main(csv_path: str, npz_path: str) -> None:
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    assert raw.shape[1] == 71, f"expected 71 cols, got {raw.shape[1]}"

    stage = raw[:, 0].astype(np.int32)
    rng = np.random.default_rng(SEED)

    def take(mask, cap):
        idx = np.flatnonzero(mask)
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        return np.sort(idx)

    keep = np.concatenate([
        take(stage == 0, MAX_STAGE0),
        take(stage == 1, MAX_STAGE1),
        take((stage == 2) | (stage == 3), MAX_APPLIED),
    ])
    r = raw[keep].astype(np.float32)
    s = stage[keep]

    out = dict(
        stage=s,
        q0=r[:, 1:10].reshape(-1, 3, 3),
        q1=r[:, 10:19].reshape(-1, 3, 3),
        prev0=r[:, 19:28].reshape(-1, 3, 3),
        prev1=r[:, 28:37].reshape(-1, 3, 3),
        fixed0=r[:, 37].astype(np.int32),
        fixed1=r[:, 38].astype(np.int32),
        C1=r[:, 39],
        C2=r[:, 40],
        n=r[:, 41:44],
        bary_cur=r[:, 44:47],
        bary_closest=r[:, 47:50],
        mu_s=r[:, 50],
        mu_d=r[:, 51],
        mult=r[:, 52],
        post0=r[:, 53:62].reshape(-1, 3, 3),
        post1=r[:, 62:71].reshape(-1, 3, 3),
    )
    np.savez_compressed(npz_path, **out)
    uniq, cnt = np.unique(s, return_counts=True)
    print(f"wrote {npz_path}: {len(s)} records, stages {dict(zip(uniq.tolist(), cnt.tolist()))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
