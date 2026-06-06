# rl_mass on MyoBimanualArm (fingers, 1D muscles) — Design

Date: 2026-06-06
Branch: `finger`
Status: Approved (brainstorm), pending implementation plan

## Goal

Train the existing rl_mass pipeline (PPO + MuscleNN) to imitate reference
motion on a **finger-articulated skeleton**, driven by **1D Hill-type
muscles**. Source the skeleton + muscles from musclemimic's **MyoBimanualArm**
model (two arms + fully articulated hands), and source reference motion from
musclemimic's pre-retargeted bimanual trajectories.

This is the first proof of the finger + 1D-muscle pipeline in rl_mass. Full-body
and BVH-driven motion are explicitly out of scope (see Non-Goals).

## Decisions (from brainstorming)

- **Scope:** MyoBimanualArm — two arms + hands. Fixed-base (no floating root).
  (Supersedes an earlier "full body + fingers" direction; user chose bimanual.)
- **Asset source:** Port musclemimic assets (skeleton + muscles), not re-author
  in rl_mass.
- **Motion source:** Reuse musclemimic's existing retargeted bimanual `.npz`
  trajectories directly (already in MyoBimanualArm MuJoCo qpos space) — no new
  retargeting.
- **Muscle engine:** Load the MJCF as-is and read muscle length + moment-arms
  from **MuJoCo-native spatial tendons** (`ten_length`, `ten_J`), preserving
  wrap geometry exactly. Apply rl_mass Hill force curves on top. No waypoint
  re-authoring, no fidelity loss.

## Verified facts (grounding)

MuJoCo 3.8.0 (repo env):
- `MjData.ten_length` (per-tendon length) and `MjData.ten_J`
  (tendon Jacobian = moment-arm matrix, ntendon × nv) are available natively,
  computed with wrapping.

MyoBimanualArm MJCF
(`/home/chunleli/Dev/musclemimic_models/musclemimic_models/model/arm/myoarm_bimanual.xml`):
- `nq = nv = 76` — **all hinge joints, no free root** (fixed thorax base).
- `ntendon = 134`, `nu = 126` muscle actuators. The 8 extra tendons are
  kinematic-coupling tendons (not muscles) → excluded from the muscle set.
- `nbody = 77`, `njnt = 76`.
- Joints per side (×2): shoulder girdle (sternoclavicular, scapula,
  acromioclavicular, humerus rotations), `elbow_flex`, `pro_sup`, wrist
  (`deviation`, `flexion`), thumb (`cmc_flexion`, `cmc_abduction`, `mp_flexion`,
  `ip_flexion`), fingers 2–5 (`mcpN_flexion`, `mcpN_abduction`, `pmN_flexion`,
  `mdN_flexion`).
- Loads cleanly via `MjModel.from_xml_path` (resolves its own includes + meshes).

Bimanual trajectory `.npz`
(`/home/chunleli/Dev/musclemimic/trajectory_data_bimanual/bimanual_episodes_mjx_20260601_220203.npz`):
- `episode_N_traj_qpos` = `(frames, 76)` float32 — reference qpos, already in
  MyoBimanualArm MuJoCo qpos order.
- `episode_N_muscle_activations` = `(frames, 126)` — reference activations
  (optional supervision / sanity reference).
- `dt`, `total_fps`, `joint_names` (76) present. Multiple episodes per file.

rl_mass muscle contract (existing, to be preserved):
- `MuscleTuple(JtA, L, b, tau_des)` feeds `MuscleNN`. `L` is the
  (active_dof × n_muscle) moment-arm × active-force matrix; `b` passive
  forces; `tau_des` SPD desired torques. Force model is affine in activation:
  `tau = L @ a + b`. Hill curves: `g_al(l_m/l_m0)`, `g_pl(l_m/l_m0)` in
  `src/rl_mass/muscle.py` (no force-velocity term).
- Body/DOF structure currently hardcoded via `MASS_BODY_NAMES` / DART
  convention in `src/rl_mass/joint_utils.py`; humanoid path routes BVH → DART →
  MuJoCo.

## Architecture

Add a **config-driven native-MuJoCo case**, isolated from the existing
humanoid/DART/BVH path so walk/dance keep working. Selected by a config flag
`skeleton_format: "mujoco_native"`.

### Components

1. **Case dir `data/rl_mass_bimanual/`**
   - `config.json`: `skeleton_format: "mujoco_native"`, path to bimanual MJCF,
     path to bimanual `.npz`, reward weights (`w_q`, `w_v`), SPD `kp`,
     control/sim Hz.

2. **Native model load (`env.py`, new branch)**
   - Load MJCF directly via `MjModel.from_xml_path`. Read `nq/nv/njnt/joint_names`
     from the model. `root_joint_dof = 0`; all 76 DOF active. Bypass
     `MASS_BODY_NAMES` / DART entirely on this branch.

3. **`TrajectoryRef` (new)**
   - Loads `.npz`; serves target `qpos(t)` / `qvel(t)` by interpolating
     `traj_qpos` frames at `dt` (qvel by finite difference). Replaces
     `Character` / BVH for this case. No retargeting. Handles multiple episodes
     (select / concatenate).

4. **`MujocoTendonMuscleSystem` (new muscle backend)**
   - One muscle per actuator (126); map actuator → its tendon via
     `actuator_trnid` / `tendon_actuatorid`; exclude the 8 non-actuated tendons.
   - Each control step after `mj_forward`: read `data.ten_length` and
     `data.ten_J`. Normalize length (`tendon_length0` / `lengthrange`) → `l_m`;
     apply rl_mass `g_al` / `g_pl`. `f0` from the muscle actuator's gain params.
   - Emit the existing `MuscleTuple(JtA, L, b, tau_des)` contract — drop-in for
     the current `MuscleNN`. Wrapping preserved exactly.

5. **SPD + reward**
   - SPD over 76 DOF → `tau_des` from tracking the reference qpos.
   - Reward = `w_q`·joint-angle tracking + `w_v`·joint-velocity tracking.
   - Drop COM and end-effector terms (no floating base).

6. **Train / eval**
   - Reuse `train.py -d data/rl_mass_bimanual/config.json`.
   - Action space = 126 muscle activations.
   - Eval/USD-bake: out of scope for v1 (see Non-Goals); revisit after training
     converges.

## Open questions / risks (resolve during implementation)

- **`f0` extraction:** exact mapping from MuJoCo muscle `gainprm`/`biasprm`
  (and/or the `force` attr) to rl_mass `f0`. Verify against a known muscle.
- **`ten_J` convention:** confirm sign and units of `data.ten_J` vs rl_mass `L`
  (moment-arm sign, normalization). Validate that `tau = L @ a + b` reproduces
  expected joint torques on a static pose.
- **Tendon→muscle mapping:** confirm the 8 non-actuated tendons are coupling
  tendons and correctly excluded; confirm 1:1 actuator↔tendon for the 126.
- **Length normalization:** choose `l_m0` (optimal fiber length) source —
  `lengthrange` midpoint vs `tendon_length0` vs gainprm — and `l_t0`.
- **Contacts:** confirm bimanual model needs no special contact handling for
  imitation (anchored arms, minimal contact).

## Update (2026-06-06): constraint-dependent DOFs excluded

Implementation surfaced that MyoBimanualArm encodes the **scapulohumeral
rhythm** via 22 `mjEQ_JOINT` equality constraints: 22 shoulder-girdle DOFs
(sternoclavicular, unrotscap, acromioclavicular, unrothum, shoulder1 — both
sides) are slaved to `shoulder_elv`/`elv_angle`. These DOFs are driven by
MuJoCo's constraint forces, not muscles, so the muscle moment matrix `L` is
only rank 64 over the 76 DOFs and SPD demands torque on 22 unspannable
directions (Task-7 conditioning gate: residual 0.81).

Resolution (chosen by user): exclude the 22 constraint-dependent DOFs from the
muscle-tuple target and the action space — mirroring how rl_mass excludes the
unactuated root DOFs. `BimanualEnv` computes `active_dof` (54 independent DOFs)
from the `mjEQ_JOINT` constraints' dependent joints; the policy action and
`tau_des`/`L`/`b` operate on those 54. `JtA` stays full (input feature, not a
target). On the 54 independent DOFs the basis is full rank and the
torque-matching LS residual is **0.0**. Reward still tracks all 76 qpos (the
dependent DOFs follow their drivers). `num_action` = 54.

## Non-Goals (v1)

- Full-body (MyoFullBody) skeleton.
- BVH → MyoBimanualArm retargeting (reuse `.npz` only).
- Eval / USD / mp4 baking.
- Modifying the existing humanoid/DART/BVH walk/dance path beyond adding the
  isolated native branch.

## Success criteria

- `train.py -d data/rl_mass_bimanual/config.json` launches and runs PPO on the
  76-DOF bimanual skeleton with 126 tendon-backed muscles.
- Muscle length/moment-arms sourced from MuJoCo tendons; `MuscleTuple` validated
  (`L @ a + b` reproduces expected torque on a test pose).
- Tracking reward increases over training (policy learns to imitate the
  reference trajectory), num_tuple stable (per project memory on muscle loss
  spikes).
