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

### Phase 5

- Added device-resident tensor-API rollout path:
  `WarpVecEnv.get_muscle_torques_tensor`, `get_desired_torques_tensor`,
  `set_activation_levels_tensor`, plus
  `WarpMuscleSystem.set_activations_tensor` (D2D copy via `wp.from_torch`).
- Train loop (`GenerateTransitions`, batch-API path) auto-detects tensor APIs
  and skips the muscle-inference H<->D round-trips.
- Lazy CPU sync of `env.desired_torque` and `env.activation_levels` —
  only fired when a muscle-tuple sample is about to be collected.
- Smoke perf (`swim.json`, RTX 4090, `num_envs=8`, `buffer=64`):
  tensor path `~19.1s` vs numpy path `~22.3s` GenerateTransitions
  (~14% wallclock win at small batch; gap grows with `num_envs`).

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

## Phase 5: GPU muscle buffer + CUDA-graph substep (May 2026)

Two changes landed together:

1. **GPU-resident muscle buffer.** `gather_muscle_tuples()` now returns
   `dict[str, torch.Tensor]` on the device (was: `list` of per-env dataclass).
   `_collect_muscle_tuples_batched()` runs once per control step (was: 1024
   Python iterations × per-env `.numpy()` round-trip). New
   `WarpMuscleSystem.get_training_terms()` produces `(JtA, jta_per_muscle,
   passive_torque)` in one batched device-resident pass.
2. **CUDA graph capture of the substep.** The hot kernel sequence
   `clear_forces → muscle update → muscle apply_forces → mujoco_warp.step`
   is captured once via `wp.ScopedCapture()` and replayed by
   `wp.capture_launch()` (pattern from `external/newton/.../example_robot_policy.py`).
   The D→H NaN guard inside `engine.step()` was split into
   `step_physics_only()` (graph-safe) and `recover_nan_if_any()`
   (called outside the graph), preserving the per-substep NaN rollback
   semantics.

### Throughput (walk.json, 128 control steps = 1 PPO iter, random actions)

| backend / num_envs | s/iter | ms/step | **trans/s** | GPU avg util | peak GPU mem |
|---|---:|---:|---:|---:|---:|
| CPU subproc, 64 envs (real training) | 4.73 | – | **1,730** | – | – |
| Warp 1024 envs, no graph (phase 4) | ~80.0 | 625 | **1,640** | ~5–10% | – |
| Warp 1024 envs + graph | 62.1 | 485 | **2,111** | 14.4% | 5.6 GB |
| Warp 4096 envs + graph | 101.7 | 794 | **5,155** | 16.3% | 20.4 GB |
| Warp 8192 envs + graph | 139.3 | 1,088 | **7,529** | 17.9% | ≥ 3 GB rolling, ~30 GB at gather (estimated) |

Graph capture alone gives a **~6× wall-time speedup** at 1024 envs vs the
no-graph build (from a separate 16-ctrl micro-bench: 80s → 13.5s); the
1024-env 62.1s number above is from a 32-ctrl bench where random-action
divergence inflates the per-step cost (see caveat below).

### Caveat: random-action divergence

The above are measured with random `~N(0, 0.1)` actions for 32+ control
steps. The character drifts into deep contact penetration, which
floods the constraint solver with `nefc overflow` fallbacks and slows
per-substep time 3–5×. Trained-policy rollouts stay near upright and
should be substantially faster — the actual training run is the
authoritative measurement.

### What CUDA graph removed

The pre-graph profile (cProfile, 1024 envs, 4 ctrl steps) showed:

| line | wall fraction |
|---|---:|
| `mujoco_warp.step` (incl. solver, narrowphase, fwd_position) | 79% |
| `wp.launch` Python overhead (pack_arg / type check / dispatch) | 56% (overlaps with above) |
| `_collect_muscle_tuples_batched` | 16% |

CUDA graph capture collapses the `wp.launch` Python overhead into a single
graph launch per substep. The mujoco_warp internal `wp.capture_while` is
still nested correctly inside the outer `ScopedCapture`.

### Knobs added

- `WarpVecEnv._substep_graph` — lazy graph, built on the 2nd call to
  `_substep_batched` and replayed thereafter. Reset when worlds change.
- `WARP_NAN_TRAP=1` — bypasses the graph and runs eager kernels with
  pre-step `xfrc`/`lmn` sampling for debugging blow-ups.
- `train.py --max_steps N` — stop training once cumulative transitions
  hits `N` (priority over `--max_iter`). 10M timesteps ≈ 5000 CPU iters.
- `reward_csv` now records `total_transitions` per row.

