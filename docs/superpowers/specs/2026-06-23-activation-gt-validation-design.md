# Muscle-Activation Ground-Truth Validation Harness — Design

**Date:** 2026-06-23
**Branch:** `finger`
**Status:** Approved design, pre-implementation

## Goal

Quantify how close rl_mass's learned muscle activations are to a trusted
biomechanics ground truth. First instantiation: the OpenSim **Arm26** model
(2 DOF, 6 muscles) compared against OpenSim **Static Optimization (SO)**
activations. The comparison core is **GT-source-agnostic** so EMG data can plug
in later without touching the metrics/plots code.

This is a **validation harness only** — it does not change training. No GT term
is fed into the loss.

## Why this matters

Muscle redundancy means many activation patterns produce the same joint torque.
An RL policy can track a motion with activations that are *not* physiologically
plausible. The only way to know whether rl_mass learned biologically sensible
activations is to compare against a ground truth. Two GT sources are in scope;
this design covers the cleaner one first.

## What we already have (verified on disk)

All under `/home/chunleli/Dev/OpenSimExample/Models/Arm26/`:

- `InverseKinematics/arm26.osim` — the model: 3 bodies (`base`, `r_humerus`,
  `r_ulna_radius_hand`), 2 actuated DOFs (`r_shoulder_elev`, `r_elbow_flex`),
  6 `Thelen2003Muscle` (`TRIlong`, `TRIlat`, `TRImed`, `BIClong`, `BICshort`,
  `BRA`), 32 path points, wrap cylinders/ellipsoids. Fixed base (base welded to
  ground). No equality constraints.
- `OutputReference/InverseKinematics/arm26_InverseKinematics.mot` — the motion:
  121 frames @120 Hz, 2 coordinate columns, **degrees**. Elbow-flexion motion.
- `OutputReference/StaticOptimization/PhysiologicalResults/arm26_StaticOptimization_activation.sto`
  — **GT activations already computed**: 121 rows @120 Hz, columns are the 6
  muscle names verbatim, activation in `[0,1]`, `inDegrees=no`. A matching
  `_force.sto` (per-muscle force GT) sits beside it.

Consequence: **OpenSim does not need to be run.** (It is not installed/built in
this environment anyway — no `opensim` python module, no built `opensim-cmd`.)
The GT is a static input file.

The only missing artifact is an **arm26 MJCF**; MuJoCo cannot read `.osim`.

## Approach

A small `.osim → MJCF` parser (decided over MyoConverter and hand-authoring)
covering only the subset Arm26 needs, reusable later for Gait2354. Then reuse
the existing `rl_mass.finger` native path (fixed base, tendon-backed Hill
muscles) to train on Arm26, and a new harness to compare rollout activations
against the SO `.sto`.

## Components

### C1. `.osim → MJCF` parser  (`tools/osim2mjcf.py`)

Standalone script. Parses the Arm26-relevant subset of OpenSim XML and emits a
MuJoCo MJCF + assets:

- **Bodies** ← `BodySet/Body`: name, mass, mass_center, inertia, attached
  meshes (`.vtp` → reference by name; mesh conversion out of scope for v1, use
  primitive/site geometry for collision-free visualization or skip visuals).
- **Joints** ← `JointSet`: `WeldJoint` → MuJoCo weld/fixed; `CustomJoint` with a
  single non-constant `TransformAxis` → MuJoCo `hinge` about that axis, using
  the joint's `PhysicalOffsetFrame` translations for placement and the
  `Coordinate` range. (Arm26's two CustomJoints each reduce to one hinge.)
- **Muscles** ← `ForceSet/Thelen2003Muscle`: each becomes a MuJoCo **spatial
  tendon** routed through sites placed at every `PathPoint` (`location` in its
  `socket_parent_frame` body), plus a muscle **actuator** carrying the Thelen
  parameters needed by `TendonMuscleSystem` (`max_isometric_force` →
  actuator force scale `f0`; `optimal_fiber_length`, `tendon_slack_length`,
  `pennation_angle` recorded for the Hill curves).
- **Wrap objects** ← `WrapCylinder`/`WrapEllipsoid`: emit MuJoCo tendon wrap
  geoms (`sphere`/`cylinder`) with the object's transform; attach via tendon
  `<geom>` wrap entries. Ellipsoid approximated by its dominant-radius sphere
  if MuJoCo lacks a direct ellipsoid tendon-wrap (document the approximation).

The parser is intentionally minimal: it errors loudly on any OpenSim construct
it does not handle rather than silently dropping it, so Gait2354's extra
constructs (coordinate-coupler constraints, multi-axis CustomJoints, free root)
surface explicitly when that model is attempted.

### C2. Motion loader  (`.mot → qpos npz`)  (`tools/mot2npz.py`)

Reads an OpenSim `.mot` (Storage format), converts coordinate columns to the
MJCF qpos order, degrees → radians for rotational coordinates, and writes the
`.npz` trajectory consumed by `rl_mass.finger.TrajectoryRef`. For Arm26: 2
columns → 2 qpos, 120 Hz. The npz carries `qpos (T, nq)` and the sample rate so
`TrajectoryRef.pos_vel` can finite-difference velocity exactly as for the
bimanual case.

### C3. Arm26 training config + run  (`data/rl_mass_arm26/config.json`)

A `BimanualEnv`-format config (`skeleton_format="mujoco_native"`) pointing at
the generated MJCF + npz. Arm26 has no equality constraints, so `active_dof`
is all 2 DOFs and `BimanualEnv` is reused unchanged. Train with the existing
PPO pipeline; the model is tiny (state 5, action 2, 6 muscles) and converges
fast. Produces `nn/max.pt` + `nn/max_muscle.pt`.

### C4. Validation harness  (`tools/validate_vs_gt.py`)  — primary deliverable

Pipeline:

1. **Rollout** — load checkpoint, run the trained policy deterministically on
   the reference motion, recording per-control-step activations `a_RL (T, nu)`
   in actuator order. Reuses the rollout loop from
   `tools/bake_bimanual_usd.py` (extract the shared rollout into a helper so
   both tools call it).
2. **Load GT** — `GTLoader.load()` returns `a_GT (T_gt, m)` plus muscle names
   and timestamps. `OpenSimStoLoader` parses the `.sto` (skip header to
   `endheader`, read `time` + named muscle columns).
3. **Align** — map `a_RL` muscles to `a_GT` muscles via `MuscleNameMap`
   (identity for Arm26, since names match 1:1). Resample both onto a common
   time grid (rl_mass `control_hz` vs SO 120 Hz) by linear interpolation over
   the overlapping interval.
4. **Metrics** (per muscle, plus aggregate): Pearson `r`, RMSE, R²,
   on/off-timing lag (argmax of normalized cross-correlation), and a
   co-contraction index across the muscle set. 
5. **Output** — `metrics.json` (per-muscle + aggregate), a per-muscle overlay
   figure `compare.png` (a_RL vs a_GT vs time), and `compare.csv` (aligned
   series). Print a summary table.

**GT-source-agnostic boundary:** steps 3–5 know nothing about OpenSim or EMG.
The only source-specific code is a `GTLoader` implementation and a
`MuscleNameMap`. EMG support later = an `EMGLoader` (bandpass → rectify →
low-pass envelope → MVC-normalize) + a channel↔muscle map, with no change to
the metrics/plot core.

## File structure

```
tools/osim2mjcf.py            # C1  .osim -> MJCF (Arm26 subset, errors on unknown)
tools/mot2npz.py              # C2  .mot -> qpos npz
tools/validate_vs_gt.py       # C4  harness: rollout vs GT -> metrics + plots
tools/_rollout.py             # shared rollout helper (factored from bake tool)
src/rl_mass/finger/gt/
  __init__.py
  loaders.py                  # GTLoader interface + OpenSimStoLoader (EMGLoader later)
  metrics.py                  # Pearson/RMSE/R2/timing/co-contraction
  align.py                    # name map + time resample
data/rl_mass_arm26/
  config.json                 # BimanualEnv config for Arm26
  arm26.xml                   # generated MJCF (committed artifact)
  arm26_motion.npz            # generated trajectory (committed artifact)
```

`tools/bake_bimanual_usd.py` is refactored to import the shared rollout helper.

## Data flow

```
arm26.osim ──C1──> arm26.xml (+ Thelen params)
arm26.osim ──C1──> muscle f0 / Hill params
arm26_IK.mot ──C2──> arm26_motion.npz
arm26.xml + npz ──C3(train)──> nn/max.pt, nn/max_muscle.pt
checkpoint ──C4(rollout)──> a_RL (T, 6)
arm26_StaticOptimization_activation.sto ──C4(GTLoader)──> a_GT (T, 6)
a_RL + a_GT ──C4(align+metrics)──> metrics.json, compare.png, compare.csv
```

## Decisions

- **MJCF source: small custom parser** (not MyoConverter, not hand-authoring).
  Reusable for Gait2354; 1:1 name control; no heavy OpenSim/conda toolchain.
- **GT variant: PhysiologicalResults** `.sto` (the physiologically-bounded SO
  solution), not the Nonphysiological variant.
- **Comparison level for v1: closed-loop (L2)** — policy rollout on the same
  reference motion vs SO on that motion. A solver-only (L1) comparison (feed
  identical inverse-dynamics torques to `MuscleNN` and to SO) is a documented
  future refinement, cleaner but needs an ID torque export.
- **Reuse `BimanualEnv` unchanged** — Arm26 fits the finger archetype (fixed
  base, no equality constraints, tendon muscles).

## Known confounds (report, do not hide)

- MuJoCo tendon wrap geometry ≠ OpenSim `GeometryPath`; ellipsoid wraps are
  approximated. Moment arms — and therefore activations — will differ somewhat.
- Thelen Hill curves vs rl_mass `g_al`/`g_pl` curves differ in shape; identical
  parameters do not guarantee identical force-length behavior.
- SO minimizes Σa²; rl_mass learns a least-effort-like activation. Pattern and
  timing should agree; absolute magnitudes may not.

These differences are the measurement, not bugs. The harness quantifies them.

## Success criteria

1. `osim2mjcf.py` produces an `arm26.xml` that loads in MuJoCo, with 2 hinge
   DOFs and 6 spatial-tendon muscles named identically to the `.osim`.
2. `BimanualEnv` trains on Arm26 to a healthy return (motion tracked).
3. `validate_vs_gt.py` runs end-to-end and emits `metrics.json` +
   `compare.png` + `compare.csv` with per-muscle Pearson r / RMSE / R² /
   timing-lag and an aggregate row, muscles aligned 1:1 by name.
4. The metrics/plot core has zero OpenSim-specific imports; the OpenSim
   dependency lives only in `OpenSimStoLoader`.

## Out of scope (future)

- EMG GT (`EMGLoader` + channel map) on a lower-limb model.
- Gait2354 (floating root, coupler constraints, 54 muscles) — exercises the
  parser's error paths and the floating-base path.
- L1 solver-only comparison (inverse-dynamics torque export).
- `.vtp → MuJoCo mesh` conversion for faithful visuals.
- GT-supervised training (feeding GT into the loss).
