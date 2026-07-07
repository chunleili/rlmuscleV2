> **⚠️ Historical design doc.** References to `OneWaySim` /
> `coupleOneWayGmg/solver.py` and its multigrid phases predate the move to
> `src/muscle/mg/mg.py` (`MultiGridSim`); those module paths no longer exist. For
> the current multigrid scheme see [`../../humanV2_cage_multigrid.md`](../../humanV2_cage_multigrid.md).

# Unified XPBD Engine — design

**Date:** 2026-06-13
**Status:** approved-design (pending spec review)

## Goal

Merge the three XPBD substep loops in the codebase into **one engine**:

- `MuscleSim` (`muscle_warp.py`) — production human-muscle sim (fiber/Millard/ARAP/volume/SNH, force+gravity, bone-pin).
- `OneWaySim` (`coupleOneWayGmg/solver.py`) — quasi-static pin-to-target ARAP-HPBD (lion / humanV2 / fascia), multigrid + fascia glue/anchor.
- `MassSpringTet` (`collision/udcf/massspring.py`) — Gaia S01 dynamic MassSpring (edge springs + hard volume + inversion), driven by the U-drop self-collision demo.

They are **not three algorithms** — they are three *programs* over the same skeleton:

```
for substep:
    integrate(mode)                 # force+gravity | gravity | none(quasi-static)
    [kinematic snap: pin / pintotarget]
    clear dP
    for iter:
        project typed-constraints   # jacobi-accum | colored-GS | serial-GS
        [apply_dP]
        collision extras            # DCD | SDF | boundary
    [post-smooth] [inversion-repair] [multigrid] [glue]
    update_velocities(damp mode)
```

`MuscleSim` already implements ~all of this: a typed-constraint registry
(`cons` + `cons_ranges{type:(off,count)}` + `_dispatch_constraints`), a
jacobi/colored-GS style knob, a `constraint_solvers[]` collision-extras hook,
and `integrate` / `update_velocities` / `apply_pintotarget` /
`repair_inverted_tets`. The other two are subsets/variants.

## Hard invariant: bit-wise identity

**For every sim, the merged engine must produce output bit-for-bit identical
to the pre-merge code**, frame-by-frame, same input. This is the acceptance
gate for every migration stage — a stage that diverges by a single ULP does
not land.

Consequences:

- **No kernel is "unified" unless the two kernels are already FP-identical.**
  MuscleSim's `solve_tetarap_svd_kernel` and OneWay's `arap_svd_kernel` differ
  in arithmetic; they stay as **two distinct constraint types**, selected by the
  program. **Redundant kernels are allowed and preferred over risking a diff.**
- Per-step path is selected by **optional flags / program phases**. All existing
  config defaults reproduce current behavior exactly (new behavior is opt-in).
- Dedup is opportunistic, only for kernels proven identical (e.g. OneWay's ARAP
  vs MuscleSim's are *not* merged; a true exact duplicate may be).

## The framework: `XpbdEngine`

**Locked decisions:**
- `XpbdEngine` is defined **in-place in `muscle_warp.py`**; `MuscleSim`
  subclasses it same-file (no moving the 5000-line core across files — lowest
  Stage-1 bit-wise risk).
- A **program phase is a bound method / closure** (`program = [self.integrate,
  self.solve_typed, ...]`), matching current method calls 1:1 — easiest to
  prove bit-identical. No tagged-op machinery.

Extract MuscleSim's reusable core into a base `XpbdEngine` (host class, no
sim-specific loaders). It owns:

- **State buffers:** `pos, pprev, vel, dP, dPw, mass/inv_mass, tet_indices,
  stopped, pos_mid` (snapshot), surface tris.
- **Constraint registry + dispatch:** `cons` + `cons_ranges` +
  `_dispatch_constraints(type_ranges)` (moved verbatim from MuscleSim — same
  kernels). New types append cases: `DISTANCE` (edge spring; also covers fascia
  glue), `TETVOLUME` gains a `hard` compliance path, `INVERSION`, plus OneWay's
  `ARAP_SVD_ONEWAY` / `VOLUME_ONEWAY` registered as **separate** types
  (redundant kernels, bit-exact to OneWay).
- **Style knob:** jacobi (`dP/dPw` + `apply_dP`), `use_colored_gs`, serial-GS.
- **Collision extras:** `constraint_solvers[]` — DCD / SDF / boundary all route
  here as callables.
- **Primitives:** `integrate` (mode flag: `force` | `gravity` | `none`),
  `update_velocities` (mode: `plain` | `damped(exp,const)`),
  `apply_pintotarget`, `repair_inverted_tets`.
- **Program runner:** `self.program` = ordered list of phases (bound callables /
  tagged ops). `step()` runs the program across `num_substeps`. **Default
  program for each sim reproduces its current exact order**, so `step()` stays
  bit-identical.

### Three sims = three programs over one engine

```
MuscleSim   = [integrate(force), pintotarget, solve(typed), apply_dP,
               collide(sdf), post_smooth, repair, pintotarget, update_vel]
OneWaySim   = [pin, multigrid?, solve([arap_oneway, vol_oneway]), apply_dP,
               glue?, collide(sdf)?]                        # integrate=none
MassSpring  = [integrate(gravity), solve([dist_spring, vol_hard], GS),
               collide(boundary), repair(inversion), collide(dcd),
               update_vel(damped)]
```

Each sim subclasses `XpbdEngine`, keeps **its own loader/init/state** and its
own kernels, and defines `build_program()`. Kernels stay in their current
modules (`coupleOneWayGmg`, `collision/udcf`), imported and registered — not
moved into `muscle_warp.py`.

### Layering

`collision/` stays pipeline-independent (raw collider + kernels). `XpbdEngine`
+ the three sims live in the muscle pipeline; the engine imports collision, not
vice-versa. The U-drop example becomes a thin driver over `MassSpringSim`.

## Migration — staged, each gated by bit-wise regression

**Stage 0 — golden harness.** Capture per-frame `pos` (and `vel`) golden
arrays from the *current* code for: MuscleSim(humanV2 muscle run), OneWaySim
(lion, humanV2, fascia), MassSpring (U-drop vs Gaia `testresults_U/A*.ply`).
Harness asserts `np.array_equal` (exact) frame-by-frame.

**Stage 1 — extract engine.** Refactor MuscleSim to subclass `XpbdEngine`;
`step()` → `run_program(self.program)`; default program == current order.
*No new features.* Gate: MuscleSim golden bit-identical.

**Stage 2 — OneWaySim onto engine.** Register OneWay kernels as types/phases
(redundant ARAP/volume types), multigrid/fascia/glue as phases; `OneWaySim`
becomes an `XpbdEngine` subclass with the program above. Keep `OneWaySim`
class name + constructor (back-compat: configs/examples unchanged). Gate:
lion / humanV2 / fascia golden bit-identical.

**Stage 3 — MassSpring onto engine.** Add `DISTANCE` type, `TETVOLUME` hard
flag, `INVERSION` type, `integrate(gravity)` + `update_vel(damped)` variants,
boundary + DCD extras; build `MassSpringSim`. Rewire `collision/udcf/example/
u_drop.py` to drive `MassSpringSim`. Gate: U-drop vs Gaia golden bit-identical
(== current bespoke driver's L2).

**Stage 4 — opportunistic cleanup.** Dedup only kernels proven exactly
identical; collision uniformly via extras. Gate: all goldens still identical.

## Risks

- **MuscleSim is the 5000-line production sim.** Extracting `XpbdEngine` is the
  riskiest step; the Stage-1 bit-wise gate is the protection — if it can't be
  made identical, the extraction is wrong and is reverted, not the test relaxed.
- **Hidden state coupling** (e.g. `stopped`, `reaction_accum`, Millard curve
  buffers) may resist clean extraction. Mitigation: extract by *moving*, not
  rewriting; the base holds buffers, the subclass populates them.
- **Float order in jacobi atomics** is deterministic on a fixed device/Warp
  build but can differ across them. Goldens are captured and compared on the
  same device (`cuda:0`).

## Out of scope

- New physics. This is pure restructure + the already-validated MassSpring path.
- Touching the SDF collision pipeline's kernels (already working).
- Merging OneWay's ARAP with MuscleSim's ARAP (kept redundant by design).
