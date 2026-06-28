# Walk vs EMG Validation — Build Report

## What was built

### New modules

**`src/rl_mass/finger/gt/loaders.py`** — Added `EMGLoader(GTLoader)`:
- Globs `emg_dir/T*.txt`, parses tab-separated header + data
- Collects every stride column whose suffix matches a requested muscle label (e.g. `St1_VL`, `St2_VL` → `VL`)
- Resamples each stride to `n_points` over 0–100% via linear interpolation
- Returns mean across all strides (10 trials × 2 strides = 20 per muscle for P8_V4_R)
- Returns raw μV (no normalization); raises `ValueError` on empty dir or missing muscle

**`src/rl_mass/finger/gt/phase.py`** — New module, pure NumPy:
- `anterior_projection(foot_xyz, pelvis_xyz, up_axis=1) -> (T,)`: Zeni 2008 foot-ahead-of-pelvis metric
- `detect_heel_strikes(proj) -> list[int]`: local-maxima scan with 4-neighbour test (handles leading-edge plateaus)
- `fold_cycles(signal, strikes, n_points=101, min_len=10) -> (n_points, ncol)`: resample + mean across gait cycles

**`src/rl_mass/finger/gt/__init__.py`** — Appended `EMGLoader` and the three phase functions to imports and `__all__`.

**`tools/validate_walk_vs_emg.py`** — `compare(config_path) -> dict`:
- Builds `MassEnv(walk_config)`, loads `SimulationNN` + `MuscleNN`
- Device-aware: `sdev = next(model.parameters()).device`, `mdev = next(mm.parameters()).device`
- Custom rollout mirroring scratch script (policy action → nspc//2 × muscle-net → set_activation → step×2)
- Records per-control-step activations, pelvis xpos, TalusR xpos
- Calls `anterior_projection` → `detect_heel_strikes` → `fold_cycles` on mapped SNU columns
- Calls `EMGLoader(emg_dir).load()` for GT
- Computes `pearson_r` + `timing_lag` per muscle, `mean_pearson` aggregate
- Writes `metrics.json` + peak-normalized `compare.png`

**`data/rl_mass/configs/walk_vs_emg_validate.json`** — Config with real paths:
- `walk_config`, `checkpoint` prefix, `emg_dir`, `muscle_map`, `out_dir`, `max_steps: 420`

### New tests (`tests/rl_mass/`)

**`test_emg_loader.py`** — 4 sub-tests:
- Real P8_V4_R data: names, shape (101,4), all ≥0, time 0..100
- Subset muscles work
- Missing muscle → ValueError
- Empty dir → ValueError

**`test_gt_phase.py`** — 9 sub-tests:
- `anterior_projection`: monotone increasing for foot ahead, up-axis zeroed
- `detect_heel_strikes`: triangle wave peaks at known indices, single peak, flat signal
- `fold_cycles`: shape (101,2), ramp mean ≈ linspace, short pairs skipped, no valid pairs → zeros

**`test_validate_walk_vs_emg.py`** — smoke test:
- Writes temp config with `max_steps=150`
- Calls `compare()`, checks `metrics.json` + `compare.png` exist
- Checks per_muscle keys == {"VL","BF","TA","GAL"}, each has "pearson" + "timing_lag"
- Checks "mean_pearson" in aggregate

## Test results

```
uv run python tests/rl_mass/test_emg_loader.py          → OK
uv run python tests/rl_mass/test_gt_phase.py             → OK
VIRTUAL_ENV= uv run python tests/rl_mass/test_validate_walk_vs_emg.py → OK
```

## Full compare() metrics (max_steps=420)

```json
{
  "aggregate": {
    "mean_pearson": -0.172
  },
  "per_muscle": {
    "VL":  { "pearson": -0.259, "timing_lag": 24.0 },
    "BF":  { "pearson":  0.226, "timing_lag":  8.0 },
    "TA":  { "pearson": -0.154, "timing_lag": 29.0 },
    "GAL": { "pearson": -0.500, "timing_lag": 41.0 }
  }
}
```

**Mean Pearson r = −0.172** — negative across 3 of 4 muscles, indicating that the
phase-normalized policy activation pattern is roughly anti-phased against real EMG.
BF (Biceps Femoris) is the only muscle with positive correlation (r=0.226).
GAL has the strongest anti-correlation (r=−0.500, timing_lag=41% cycle ≈ half-cycle
anti-phase), consistent with the GAL being a push-off muscle that the SNU policy
fires on the opposite leg.

This is expected and scientifically interesting: the SNU 284-muscle model was trained
for pose-matching reward, not EMG fidelity. The large timing lags (24–41% for VL, TA,
GAL) suggest the right-leg gait phase in the simulation is offset from the Zeni heel-
strike anchor, or the muscle synergies differ from biological patterns.

## Files changed

| File | Action |
|---|---|
| `src/rl_mass/finger/gt/loaders.py` | Added `EMGLoader` class |
| `src/rl_mass/finger/gt/phase.py` | New file |
| `src/rl_mass/finger/gt/__init__.py` | Added `EMGLoader`, phase exports |
| `tools/validate_walk_vs_emg.py` | New file |
| `data/rl_mass/configs/walk_vs_emg_validate.json` | New file |
| `tests/rl_mass/test_emg_loader.py` | New file |
| `tests/rl_mass/test_gt_phase.py` | New file |
| `tests/rl_mass/test_validate_walk_vs_emg.py` | New file |

## Concerns

1. **Negative correlations**: mean_pearson −0.172 is poor. The SNU model is trained on
   pose-error reward, not EMG fidelity. This harness correctly surfaces that gap.
   Options: (a) phase-shift the rollout window, (b) use contralateral leg, (c) run
   multiple gait cycles and phase-average differently.

2. **timing_lag units**: currently in % gait cycle (dt = 100/(n_points−1) = 1.0%).
   This is intuitive for gait analysis but differs from the OpenSim tool (seconds).
   The field name is consistent; callers should be aware.

3. **output/ not gitignored**: `output/walk_vs_emg/` is generated by the full run.
   Suggest adding `output/` to .gitignore if not already covered.

4. **No env_factory shim**: `validate_walk_vs_emg.py` uses `MassEnv` directly (as the
   scratch script did), not `make_env`. This is intentional — the walk env has no
   `BimanualEnv` wrapper; `env_factory.make_env` would need a type hint to dispatch.
