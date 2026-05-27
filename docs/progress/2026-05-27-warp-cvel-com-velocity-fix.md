# Warp COM-Velocity Bug Fix (cvel ≠ jacBodyCom @ qvel)

**Date**: 2026-05-27
**Branch**: `worktree-warp`
**Files**: `src/rl_mass/warp/engine.py`, `src/rl_mass/warp/gpu_env.py`
**Validation run**: `output/walk/runs/20260526_200734-nid010725-n64-h32-fix-comvel/`

## Symptom

GPU/Warp PPO training dramatically underperformed the CPU baseline on the
`walk` task at matched (n_envs, horizon):

| Backend           | n_envs | max_return @ iter ~5000 |
| ----------------- | ------ | ----------------------- |
| CPU (MuJoCo)      | 64     | 88                      |
| Warp (pre-fix)    | 64     | 26.66 (plateaued ~5–7 early) |
| Warp (large)      | 16384  | 3.65 (worse, not better)|

Same architecture, same config, same reward formulation — only the physics
backend differed. Reward growth was flat after ~iter 500 on warp.

## Root cause

`WarpEngine.get_body_com_velocity` read `data.cvel[bid, 3:6]`, while the CPU
engine computed `mj_jacBodyCom(bid) @ qvel`. These are **not** the same
physical quantity, and the discrepancy is not a warp bug — it's also true on
CPU MuJoCo:

- `cvel[bid][3:6]` is the translation component of the **spatial twist of
  body `bid` expressed at `subtree_com[bid]`**. The reference point moves
  with the subtree; for any body whose subtree contains more than one body,
  this is not `d(xipos)/dt`.
- `mj_jacBodyCom(bid) @ qvel` is the world-frame linear velocity of the
  body's own COM (`xipos[bid]`) — the canonical "COM linear velocity".

We empirically tested four candidate transforms to reconcile them
(A=`jacp@qvel`, B=`cvel[3:]`, C=`B + ω×(xipos − subtree_com)`,
D=`B + ω×(xipos − xpos)`); none agreed for all bodies. They are genuinely
different quantities, not off by a frame.

The state vector `s = [p (66), v (69), φ (1)]` feeds `v` into the policy and
the tracking reward (`r_v`, and now `r_com` after recent reward changes).
Feeding the wrong `v` for 22 non-root bodies + root corrupted both the
observation and the gradient signal — the policy plateaued because the
critic was learning a velocity field that didn't match the dynamics it was
acting on.

## Diagnostic trail

1. `tools/diag_cpu_vs_warp.py` — initialized CPU and warp to identical
   `(qpos, qvel)` and printed component-wise state deltas. Showed
   `Δstate.v = 0.806`, `Δstate.p = 0` — isolated the bug to the velocity
   path.
2. `tools/diag_cvel_semantics.py` — confirmed the same `cvel ≠ jacp@qvel`
   gap on **CPU MuJoCo** with no warp involved, ruling out an
   integration/precision issue and pinpointing the formula.
3. Reward decomposition at reset (`r_v` term) showed the velocity reward
   was the dominant deviator; positional reward matched to ~1e-15.

## Fix

Two patches on the `worktree-warp` branch:

### `engine.py`
- `get_body_com_velocity(bid)` and `get_body_com_velocity_world(world, bid)`
  now sync each world to `cpu_data` and return
  `mj_jacBodyCom(bid) @ qvel`. Matches the CPU engine bit-for-bit modulo
  fp32→fp64 cast.
- New `get_body_com_velocities_batch(bids)` — for a list of body ids, syncs
  each world once (one `mj_forward` per world) and then loops
  `mj_jacBodyCom` per body. This is the cheap shape: the per-world sync
  dominates; adding bodies is essentially free.

### `gpu_env.py`
- `WarpGPUEnv.get_states_tensor` previously read `cvel` directly via
  `wp.to_torch(self._engine.data.cvel)`, bypassing the engine getter.
  Replaced with a single call to `get_body_com_velocities_batch` over the
  combined `[non_root_body_ids, root_bid]` index list. Result is moved to
  the torch tensor's device and dtype and sliced back into `body_vel` and
  `root_vel`.

## Verification

`tools/diag_cpu_vs_warp.py` after fix:

```
Δstate.p = 0.0
Δstate.v = 3.66e-08         # was 0.806
Δreward  = 7e-15            # numerical noise
```

Training validation: fresh warp run at `n=64, h=32` with the fix vs. the
pre-fix run and the CPU baseline:

| iter | CPU baseline | Warp pre-fix | Warp post-fix |
| ---- | ------------ | ------------ | -------------- |
| 1150 | 38.78        | 7.08         | 17.90          |
| 1310 | ~43          | ~7           | 24.83          |

Pre-fix growth rate at this stage: ~0.008/iter (flat). Post-fix: ~0.058/iter,
linear, no plateau. CPU: ~0.105/iter. Post-fix reaches ~46% of CPU at
matching iter and ~2.5× the pre-fix curve — confirms the v-channel was the
dominant blocker. The remaining gap is attributed to fp32 solver divergence
(Phase-2 diag showed Δr ≈ 1.7% at 60 zero-action substeps).

## Cost and follow-up

- **Perf cost**: per-world `mj_forward` in `get_states_tensor` makes the
  rollout slower — iter time at n=64 went 6 s → 9 s (+50%). Cost scales
  linearly with `num_worlds`; the fix is unsuitable for large n_envs.
- **Follow-up (tracked, task #34)**: port `jacBodyCom @ qvel` to a Warp
  GPU kernel. Inputs needed: `body_jntadr, body_jntnum, jnt_dofadr,
  jnt_type, body_parentid, cdof, qvel, subtree_com, xipos`. The kernel
  walks the joint chain per body and accumulates `cdof[dof] × qvel[dof]`
  with the moment-arm shift `(xipos[b] − subtree_com[body_jntbody])`.
  Once that lands, the cpu sync fallback can be deleted.
- Pre-fix runs and any checkpoints trained on warp before this fix are
  poisoned — their critic learned the wrong velocity field. Discard for
  finetuning; only CPU checkpoints (or post-fix warp) are valid sources.
