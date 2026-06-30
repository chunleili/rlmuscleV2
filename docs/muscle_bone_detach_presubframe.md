# Muscle–bone detach fix (pre-subframe pin-to-target)

Walk-driven otto muscle visibly **detached** from the bone (hips/thighs lagging
behind the skeleton), reading like "too much inertia / hysteresis". `quasi_static`
was already on but felt useless. This documents the root cause (found by
elimination) and the fix.

## Root cause — by elimination

Do NOT re-chase the wrong leads:

- **NOT inertia.** `quasi_static: true` already zeroes the velocity each step.
  Toggling it changes nothing — the lag is not momentum.
- **NOT the bone target.** The per-vert `bone_slide` target (each muscle vert
  rigidly bound to its nearest bone-surface triangle at the start frame, then
  reconstructed every frame from that triangle's current pose) is **correct**:
  offline it stays ~4.8 mm (p50) from the bone surface EVERY frame, including the
  violent pre-roll. The frozen single-triangle binding does **not** swing/fling
  (an 8-nearest-face average gave identical drift). The earlier
  "single-triangle swing" theory was wrong.
- **REAL cause: the attach constraint converges too slowly.** `belly_attach`
  is an XPBD `ATTACH` solved on the Jacobi path with `omega = 0.1` SOR. Information
  propagates ~one ring per iteration; 200 iters/frame cannot catch fast walk
  motion. Instrumenting `|muscle_pos − target_pos|`: it jumps to **p50 = 176 mm
  the first frame the bone moves**, grows to **673 mm** mid-pre-roll, settles
  ~**50 mm** in walk. The muscle barely follows; the target is fine.

Lever tests at frame −20 (`|muscle−target|`):

| change | p50 |
| --- | --- |
| baseline | 497 mm |
| `omega = 0.5` | 497 mm (no effect) |
| `fine_iterations = 80` (4×) | 384 mm (only −23%) |
| `use_colored_gs = true` | 901 mm (worse) |
| `region = "full"` (hard-pin ALL) | **0.0 mm** (+ inversion 5%→0.6%) |

`region=full` proves the target is 100% reachable — but it hard-pins every vert
(kinematic lock), which kills the FEM: no fiber bulge, no collision give. So it
is not an acceptable final answer, only a diagnostic.

## The fix — `presubframe_pintotarget`

Since the target is a correct "skinned" position, don't make the slow constraint
chase it — **move the muscle there directly before solving**:

> Each (sub)frame, before the XPBD iterations, blend every muscle vert toward its
> bone target. The muscle starts the solve already on the bone; XPBD then applies
> fiber / collision / ARAP as **local** secondary deformation off that skinned base.

It is literally `apply_pintotarget`'s `pos = target` overwrite, with two changes:

1. **Soft, not locked.** The verts are left free (not made kinematic), so XPBD
   relaxes them afterwards. `region=full` = this preset **plus a lock** on
   everyone (rigid, no FEM); `presubframe_pintotarget` = the preset **without**
   the lock (keeps FEM).
2. **Over ALL verts.** Pinned tendon verts are already at target (set by
   `apply_pintotarget` the line before), so blending them toward the same target
   is a no-op, and their kinematic lock is untouched. Running over all verts (not
   just the freed belly) keeps it correct with any/no pin mask.

```
old:  [bone moves] → XPBD slowly pulls → can't keep up → detaches 100s of mm
new:  [bone moves] → preset muscle onto target → XPBD fixes only local detail → stays attached
```

## Code

- `src/muscle/mg/kernel.py` — `presubframe_pintotarget_kernel`:
  `pos[v] = pos[v] + beta * (target_pos + rest_offset − pos[v])` over all verts.
- `src/muscle/mg/mg.py` — `MultiGridSim.apply_pintotarget_all()` (soft, all verts;
  vs hard masked `apply_pintotarget`), launched from the shared `_kinematic_step(t)`
  helper (bone targets → hard `apply_pintotarget` → soft `apply_pintotarget_all`) used by
  both `step_frame()` (once/frame) and `substep_at_frac_frame()` (once/substep).
- Config key `presubframe_pintotarget` (float `beta`, **0 = off, the default** —
  plain `otto.json` is unchanged; `1.0` = snap fully to target then relax). Set in
  `data/humanV2/config/otto_walk.json`.

The target itself comes from the existing bone-slide machinery
(`_build_bone_slide_binding` / `bone_slide_target_kernel`, fed per frame by
`bone_source.positions(t)`), unchanged by this fix.

## Results (`otto_walk.json`, frames −24..200)

| metric | before | after (`presubframe_pintotarget = 1.0`) |
| --- | --- | --- |
| `|muscle − target|` p50 | 497 mm | **0.1 mm** |
| inverted tets, peak | 6.1 % | **0.9 %** |
| muscle–bone penetration | — | **0.0 mm** all frames |

## Caveat — fiber bulge

Presetting every substep + one iter/substep leaves little relaxation, so
activation-driven fiber bulge stays small. This is **not** a regression: fiber
contraction was already dead under the Jacobi path (it needs
`use_colored_gs = true`; see `docs`/memory). If fiber bulge is wanted later,
enable colored GS and likely run the preset once per frame (not per substep) to
leave more relax iterations.
