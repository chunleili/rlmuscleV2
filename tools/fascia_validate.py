"""End-to-end fascia squeeze validation.

Proves (or disproves) that the fascia layer pulls the muscle surface inward
by comparing mean muscle-surface radius with fascia contraction OFF
(fascia_rest_scale = 1.0) vs ON (0.95).

Run:
    uv run python tools/fascia_validate.py
"""
import copy
import time

import numpy as np
import warp as wp

from muscle.coupleOneWayGmg.solver import OneWaySim, load_config
from muscle.util.geo_io import GeoFile

CONFIG = "data/humanV2/config/humanV2_fascia.json"
N_FRAMES = 30
DEVICE = "cuda:0"


def run(base_cfg, rest_scale, n_frames, device):
    cfg = copy.deepcopy(base_cfg)
    cfg["fascia_rest_scale"] = rest_scale
    cfg["end_frame"] = cfg["start_frame"] + n_frames - 1
    sim = OneWaySim(cfg, device=device)
    for frame in range(sim.start_frame, cfg["end_frame"] + 1):
        sim.step_frame(frame)
    return sim.positions_np(), sim.nvm


def mean_surface_radius(P, surf):
    """Mean radial distance of muscle surface verts from the vertical (Y) axis."""
    ms = P[surf]
    c = ms[:, [0, 2]].mean(axis=0)
    r = np.linalg.norm(ms[:, [0, 2]] - c, axis=1)
    return r.mean()


def main():
    wp.init()
    base_cfg = load_config(CONFIG)

    surf = GeoFile(base_cfg["rest_path"]).point_group("surface_points")
    print(f"surface_points: {len(surf)} vertices "
          f"(range [{surf.min()}, {surf.max()}])")

    device = DEVICE
    try:
        wp.get_device(device)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] {device} unavailable ({e}); falling back to CPU (slow).")
        device = None

    t0 = time.time()
    print(f"\n=== Running OFF baseline (rest_scale=1.0, {N_FRAMES} frames) ===")
    P_off, nvm = run(base_cfg, 1.0, N_FRAMES, device)
    t_off = time.time()
    print(f"=== Running ON  (rest_scale=0.95, {N_FRAMES} frames) ===")
    P_on, nvm_on = run(base_cfg, 0.95, N_FRAMES, device)
    t_on = time.time()

    assert nvm == nvm_on, f"nvm mismatch {nvm} vs {nvm_on}"
    for tag, P in (("OFF", P_off), ("ON", P_on)):
        if not np.isfinite(P).all():
            raise ValueError(f"{tag} result contains NaN/Inf")

    r_off = mean_surface_radius(P_off, surf)
    r_on = mean_surface_radius(P_on, surf)
    abs_change = r_on - r_off
    pct_change = 100.0 * abs_change / r_off

    fascia_off = P_off[nvm:]
    fascia_on = P_on[nvm:]
    fascia_disp = np.linalg.norm(fascia_on - fascia_off, axis=1).mean()

    runtime = t_on - t0
    print("\n" + "=" * 60)
    print("FASCIA SQUEEZE VALIDATION SUMMARY")
    print("=" * 60)
    print(f"device                : {device or 'cpu'}")
    print(f"frames                : {N_FRAMES}")
    print(f"muscle verts (nvm)    : {nvm}")
    print(f"surface verts         : {len(surf)}")
    print(f"merged verts (NV)     : {P_off.shape[0]}")
    print("-" * 60)
    print(f"mean surface radius OFF: {r_off:.6f}")
    print(f"mean surface radius ON : {r_on:.6f}")
    print(f"absolute change        : {abs_change:+.6f}")
    print(f"percent change         : {pct_change:+.3f}%")
    print(f"mean fascia disp OFF->ON: {fascia_disp:.6f}")
    print("-" * 60)
    print(f"OFF runtime            : {t_off - t0:.1f}s")
    print(f"ON  runtime            : {t_on - t_off:.1f}s")
    print(f"total runtime          : {runtime:.1f}s")
    print("-" * 60)
    if r_on < r_off:
        print(f"VERDICT: SQUEEZE CONFIRMED -- muscles pulled inward by "
              f"{-pct_change:.3f}% (mean radius).")
    else:
        print(f"VERDICT: CONCERN -- no squeeze. ON mean radius >= OFF "
              f"(change {pct_change:+.3f}%). Fascia did NOT contract the muscles.")
    print("=" * 60)


if __name__ == "__main__":
    main()
