# Warp Backend

`rl_mass.warp` is the batched GPU backend built on `mujoco_warp` + NVIDIA Warp.
It keeps the high-level MASS control logic intact and ports the hot batched paths
incrementally.

## Status

Implemented:
- Shared batched `WarpEngine`
- Batched muscle force / JtA path
- `WarpVecEnv` for `--vec_env_mode warp --no_packed`
- Batched DART position differences on GPU via `warp.svd3`
- GPU-side SPD mass/bias/constraint/solve path
- Batched motion target sampling / state assembly
- Batched reward / termination query path
- Regression coverage for engine rollout, desired torques, state/reward/termination, and 1-env control step

Not implemented yet:
- Packed `run_control_step()` path for warp
- Formal multi-task benchmark matrix beyond `swim.json`
- Fully device-resident rollout loop without `.cpu().numpy()` hops in policy/muscle inference
- GPU-side muscle tuple / Jacobian collection path

## Key Files

- `src/rl_mass/warp/engine.py`: batched MuJoCo/Warp engine wrapper
- `src/rl_mass/warp/muscle.py`: batched muscle update / force / JtA
- `src/rl_mass/warp/vec_env.py`: vec-env wrapper and current batched rollout path
- `src/rl_mass/warp/dart_diff.py`: GPU DART `getPositionDifferences` equivalent
- `tools/profile_warp.py`: perf sweep / rollout benchmark

## Phase Summary

### Phase 1

- Built the shared batched `WarpEngine` on top of `mujoco_warp`
- Added `WarpVecEnv` bootstrap path for `--vec_env_mode warp --no_packed`
- Moved batched physics stepping and shared world state storage onto GPU

### Phase 2

- Ported batched muscle update / apply / JtA into `rl_mass.warp.muscle`
- Integrated shared batched muscle execution into `WarpVecEnv`
- Added runtime and smoke coverage for warp vec-env rollout

### Phase 3

- Added GPU batched DART pose difference kernel in `rl_mass.warp.dart_diff`
- Integrated it into desired torque computation
- Added profiling/docs and CPU-vs-warp regression coverage
- Conclusion from profiling: the remaining bottleneck was CPU SPD/control logic

### Phase 4

- Moved SPD mass matrix / bias / constraint / solve path to GPU
- Cached BVH motion in memory and added batched motion target sampling
- Batched `get_states()`, `set_actions()`, reward, and termination logic
- Added batched reward/eoe queries to the training loop to avoid per-env round-trips

## Current Design

The warp backend still reuses parts of `MassEnv` and `Character` for episode
bookkeeping and some fallback logic, but the hot rollout path is now mostly
batched:

1. Physics state is stored once in a tiled `mujoco_warp` data structure.
2. Muscle update/apply/JtA run batched on GPU.
3. DART rotational position differences for SPD run batched on GPU.
4. SPD mass matrix, bias, constraint forces, and linear solve run batched on GPU.
5. Motion target sampling, state assembly, reward, and termination are batched in `WarpVecEnv`.

The main remaining sync points are no longer the SPD path. The largest
remaining CPU participation is in:
- policy/muscle inference `.cpu().numpy()` hops
- per-env reset / bookkeeping paths
- muscle tuple / Jacobian collection
- missing packed warp rollout path

## Verification

Relevant tests:

- `tests/test_warp_vs_cpu.py`
- `tests/test_warp_dart_diff.py`
- `tests/test_warp_vec_env_runtime.py`
- `tests/test_warp_train_eval_smoke.py`

Typical commands:

```bash
PYTHONPATH=src /home/chunleli/Dev/rlmuscleV2/.venv/bin/python -m unittest tests.test_warp_dart_diff -v
PYTHONPATH=src /home/chunleli/Dev/rlmuscleV2/.venv/bin/python -m unittest tests.test_warp_vec_env_runtime -v
PYTHONPATH=src /home/chunleli/Dev/rlmuscleV2/.venv/bin/python -m unittest tests.test_warp_vs_cpu -v
PYTHONPATH=src /home/chunleli/Dev/rlmuscleV2/.venv/bin/python -m unittest tests.test_warp_train_eval_smoke -v
```

## Benchmarking

Use:

```bash
PYTHONPATH=src /home/chunleli/Dev/rlmuscleV2/.venv/bin/python tools/profile_warp.py \
  --config data/rl_mass/configs/swim.json \
  --num-envs 1 4 8 16 \
  --buffer-size 256 \
  --json-out output/warp_profile.json
```

The script reports:
- core env API timings (`get_muscle_torques`, `get_desired_torques`, `steps`)
- one `PPO.GenerateTransitions()` rollout timing
- sampled GPU util / memory via `nvidia-smi` when available

Latest captured outputs:
- `output/warp_profile_envs.json`
- `output/warp_profile_buffer128.json`
- `output/warp_profile_phase4_20260522.json`
- `output/cpu_rollout_compare_phase4_20260522.json`

### 2026-05-22 Phase 3 Snapshot (`swim.json`, RTX 4090, warp backend, `--no_packed`)

`buffer_size=32`

| num_envs | `steps_at_once` | `GenerateTransitions` | avg GPU util during rollout |
|---|---:|---:|---:|
| 1 | 0.673 s | 21.720 s | 6.7% |
| 8 | 3.027 s | 20.276 s | 5.9% |

Core API timings stayed almost flat between 1 and 8 envs:
- `get_desired_torques`: 0.042 s -> 0.047 s
- `get_muscle_torques`: 0.043 s -> 0.055 s

Interpretation:
- batched GPU kernels are not the dominant bottleneck now
- raising env count from 1 to 8 barely improves rollout wall time
- GPU utilization remains low, so the remaining cost is the per-world CPU SPD path

`buffer_size=128`, `num_envs=8`

| num_envs | buffer | `GenerateTransitions` | avg GPU util during rollout |
|---|---:|---:|---:|
| 8 | 128 | 119.171 s | 14.3% |

Interpretation:
- larger rollout buffers scale wall time roughly as expected
- even on a longer run, GPU utilization is still modest instead of saturating
- current warp backend is functional, but not yet throughput-optimized end-to-end

One attempted `num_envs=16` rollout sweep was intentionally aborted after confirming
the same scaling pattern: more envs were increasing wall time in the current
CPU-synchronized SPD design rather than unlocking higher GPU occupancy.

### 2026-05-22 Phase 4 Snapshot (`swim.json`, RTX 4090, warp backend, `--no_packed`)

`buffer_size=32`

| backend | num_envs | `get_desired_torques` | `steps_at_once` | `GenerateTransitions` |
|---|---:|---:|---:|---:|
| warp | 1 | 0.079 s | 0.224 s | 16.016 s |
| warp | 8 | 0.027 s | 1.253 s | 24.496 s |
| cpu serial | 1 | n/a | n/a | 1.058 s |
| cpu serial | 8 | n/a | n/a | 0.995 s |
| cpu subproc | 8 | n/a | n/a | 0.276 s |

Warp GPU utilization during rollout stayed modest:
- `num_envs=1`: avg GPU util about `9.8%`, peak `44%`
- `num_envs=8`: avg GPU util about `9.3%`, peak `72%`

Interpretation:
- `phase4` did remove the old CPU SPD bottleneck from the core control math
- this shows up clearly in `get_desired_torques`, especially at `num_envs=8`
- but end-to-end rollout is still dominated by remaining host/device traffic and Python bookkeeping
- the current `warp --no_packed` path is still slower than the CPU rollout baselines on `swim.json`

In other words: after `phase4`, the problem is no longer just SPD. The next
throughput targets are:
- device-resident policy/muscle inference handoff
- packed warp rollout path
- muscle tuple / Jacobian collection
- per-env reset / bookkeeping cleanup

## Latest Notes

Phase 3 moved batched DART pose differences to GPU. Phase 4 then moved the SPD
solve, motion sampling, state assembly, and reward/termination queries into
batched warp-side paths.

The latest measurements changed the diagnosis:
- the old CPU SPD hotspot is no longer the main bottleneck
- warp control math scales much better with env count than before
- end-to-end rollout still underutilizes the GPU and remains slower than CPU baselines

So the next optimization target is the full rollout dataflow rather than one
isolated kernel:
- `actions -> targets -> torques -> activations -> step`
- minimizing `.cpu().numpy()` conversions
- adding a packed warp rollout path comparable to CPU `subproc`
