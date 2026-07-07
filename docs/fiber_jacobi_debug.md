# Debug: humanV2 fiber contraction "invisible" vs bicep — is GPU Jacobi the cause?

Worktree: `.claude/worktrees/debug-fiber-jacobi` (branch `worktree-debug-fiber-jacobi`,
reset to main `2b73ad9`). Probes: `tools/fiber_probe_bicep.py`,
`tools/fiber_probe_humanV2.py`, `tools/viz_fiber_disp.py`.

## Method
Isolate the fiber constraint's effect: settle passively (activation 0) → P0, then
hold full activation (=1) → P1. `disp = P1 - P0` is the pure fiber response
(bone held static, no gravity). Metric = per-vertex displacement on the target
muscle + per-tet along-fiber stretch `|F·f0|` (1.0 = rest).

Both sims call the SAME fiber kernel `solve_tetfibernorm_kernel`
(`create_tet_fiber_constraint`, identical math). bicep=`MuscleSim` (3938 tets),
humanV2=`MultiGridSim` (468k tets, biceps brachii = 7876 tets).

## Result 1 — bicep contracts under every solver mode (GPU Jacobi is NOT broken)

| mode        | device | max_disp | len_shrink | girth_grow |
|-------------|--------|----------|-----------|-----------|
| GS          | cpu    | 31.6 mm  | 5.28 %    | 1.97 %    |
| **Jacobi**  | cuda:0 | 25.0 mm  | 6.28 %    | 1.58 %    |
| colored-GS  | cuda:0 | 28.9 mm  | 5.61 %    | 4.38 %    |

→ GPU Jacobi and GPU colored-GS both produce a correct, visible contraction on
bicep. The Jacobi path itself is not wrong.

## Result 2 — humanV2 biceps with fiber act=1 barely moves (reproduces the bug)

Real config + fiber act=1, GPU Jacobi, bone static, 20 fine iters:
`max_disp = 1.1 mm`, fiber shrink **0.22 %** (target_stretch = 0.60 = wants 40 %).
= invisible.

### Two compounding causes (neither is "Jacobi giving wrong numbers")

**(a) All-vertex belly-attach clamp — config/asset bug.**
`humanV2.json` sets `muscleToBonePin region="pinmask"`, but the `.geo` rest asset
has **no `pinmask` attribute** → fallback = **0 hard pins → all 157 369 verts get
the compliant belly-attach (k=1e6) tying every vertex to its bone/rest target.**
The whole muscle is soft-clamped to its rest shape, so contraction has no free
verts to move (bicep only attaches the ~127 tendon-END verts; its belly is free).

Disabling belly-attach (k=0), same Jacobi solver: shrink 0.22 %→**1.03 %**,
disp 1.1→2.3 mm.

**(b) Jacobi under-convergence on the large mesh.**
Even unclamped, per-tet fiber shortening is local (~1 %); propagating it into
visible global displacement needs far more iterations than the default 20:

| condition (belly OFF, Jacobi) | fine iters | fiber shrink | max_disp |
|-------------------------------|-----------:|-------------:|---------:|
| baseline                      | 20         | 1.03 %       | 2.3 mm   |
| more iters                    | **200**    | 1.02 %       | **9.6 mm** |

Local stretch stays ~1 % but displacement grows 4× — the low-frequency (bulk)
mode is what Jacobi resolves slowly. bicep (3938 tets) converges within its
substeps; humanV2's biceps inside a 468k-tet mesh does not in 20 iters.

### Ruled out
- **Stiffer fiber**: ×100 (`fiber_stiffness_scale` 1→100, eff ~1e11) → identical
  1.03 % / 2.3 mm. Saturated — not a stiffness deficit.
  (Note: mg hardcodes `fiber_stiffness_scale=1.0`; bicep uses 10000 — but this is
  not the limiter.)
- **Weaker ARAP**: `tetarapsvd` stiffness ×1/1000 (1e6→1e3) → identical. ARAP-SVD
  is not locking the shape.
- **omega=0.1**: only under-relaxes the m2m-glue kernel; the main `apply_dP`
  applies the full averaged correction (no omega). Not a factor here.

## Result 3 — Jacobi vs colored-GS on humanV2 (matched, belly OFF, settle 8, 20 iters)

| solver      | max_disp | fiber shrink |
|-------------|---------:|-------------:|
| Jacobi      | 3.06 mm  | 1.17 %       |
| colored-GS  | **4.09 mm** | 1.03 %    |

colored-GS propagates the contraction ~33 % further per iteration than Jacobi
(Gauss–Seidel > Jacobi convergence rate), confirming the convergence story — but
the gain is modest vs simply raising iterations (200 iters → 9.6 mm). And mg's
colored-GS is impractically slow here: graph coloring gives **116 colors**
(~90 s to build) + 116 sequential kernel launches per substep → launch-bound.

## Result 4 — the `projection` multigrid makes it WORSE (rules it out as a fix)

Real config + `use_multigrid=true`, **`mg_mode=projection`** (coarse cage 28131
NV): max_disp **0.18 mm**, shrink ~0 %. The projection V-cycle restricts the bone
target onto the cage and prolongs, re-imposing the bone/rest shape each substep —
same clamping family as the all-vert belly-attach, stronger. Not a fix.

> This applies to `mg_mode=projection` only. The current `mg_mode=hpbd`
> (state-sync residual smoother + delta-prolong, see
> [`humanV2_cage_multigrid.md`](humanV2_cage_multigrid.md)) does not teleport to
> the bone target, so it does not clamp the same way — but fiber contraction was
> never MG's job; fixes 1–2 below are the real levers.

## Verdict & fixes

**GPU Jacobi is not producing wrong results.** The humanV2 contraction is correct
in direction; it is (1) clamped by an unintended all-vertex belly-attach and
(2) under-propagated by Jacobi at 20 iters on a 468k-tet mesh.

Fix, in order of leverage:
1. **Free the belly.** Author a `pinmask` on the humanV2 rest asset (pin only
   tendon ends), or set `muscleToBonePin region="tendonmask"` with a real
   threshold (e.g. 0.5 — current `tendon_pin_threshold=1.1` > max mask 1.0 pins
   nothing) so only tendons are hard-pinned and the belly is free (bicep pattern).
2. **Add the fiber constraint + activation** to humanV2.json (it has none; the
   muscle can never contract as shipped) — `{"type":"fiber","fiber_model":
   "fibernorm","stiffness":1e8,"contraction_ratio":0.4}` + an activation source.
3. **Converge harder on the big mesh**: raise `fine_iterations` (20→100+), or use
   colored-GS (faster/iter but needs the 116-color launch cost optimized). Do NOT
   rely on `mg_mode=projection` for fiber — it suppresses contraction (hpbd does
   not, but MG is not the fiber lever regardless).
4. Not the problem: `fiber_stiffness_scale` (mg hardcodes 1.0 vs bicep 10000 —
   irrelevant, saturated), ARAP-SVD stiffness, omega.

## Acceptance render (`tools/fiber_acceptance_humanV2.py`)

Time-sampled USD proving the fix produces a stable, visible biceps contraction.
Config: tendon-tip hard pin (`region=tendonmask`, thr **0.9** → 13.8 % pinned —
anchors the muscle without over-constraining the belly), freed belly
(`belly_attach=0`), fibernorm fiber `k=1e8` ratio 0.6, `fine_iterations=100`,
static bone at frame 1, activation stepped 0→1 over 5 levels, each settled to
equilibrium (40 substeps/level).

Result — monotonic, drift-free: act 0 → 0.4 → 1.0 gives biceps
`max_disp 0 → 5.98 → 12.64 mm` (mean 6.9 mm). Outputs:
`output/humanV2_fiber_accept.usd` (muscle points + displayColor heatmap +
static grey bones, 6 timesamples) and `.png` (rest/mid/final panels).

### Pitfalls found while building the acceptance render
- **belly_attach=0 with NO hard pin → runaway drift** (biceps "displacement"
  blew up to 254 mm): with zero bone anchoring, inertia + sustained activation
  push the free muscle away unboundedly. Fix = anchor the tendon tips.
- **`region=tendonmask` thr 0.5 (23 % pinned) → contraction dies** (shrink
  0.006 %): the pin set reaches into the belly and hard-freezes those tets. thr
  0.9 (tips only) keeps the belly free → 15.9 mm.
- **quasi_static removes overshoot but also the contraction** (2.6 mm): with no
  inertia, plain Jacobi is too slow to reach equilibrium in a feasible iter
  budget. Kept inertia + tip anchor + per-level settle instead.
- These reinforce the core finding: on the 468k-tet mesh the visible result is
  governed by Jacobi convergence, not by any single stiffness.

---

# Update (session 2): clean factorial — belly-attach vs iteration, and the real #1

Session 1 left two suspects (belly-attach clamp, Jacobi under-convergence) tangled
because it measured in the **dynamic** regime (inertia on), where displacement is a
moving snapshot, not equilibrium. Session 2 re-ran the same probe with
`--quasi-static` (no inertia → true static equilibrium) to disentangle them, then
found a deeper third cause. Activated **one side only** (`R_bicepsBrachialis`) to
avoid the L/R asymmetry, tip-pin thr 0.9, `belly_k=0` unless noted.

## Experiment A — vary `fine_iterations` (isolate iteration)

| regime | 20 | 50 | 100 | 200 | 400 |
|--------|----:|----:|----:|----:|----:|
| dynamic max_disp | 8.66 | 13.37 | 13.22 | 16.50 | 12.06 |
| **quasi-static max_disp** | — | **2.598** | **2.599** | **2.603** | 2.615 |

Static bulge is **iteration-independent** (2.60 mm across 50→400, <1 %). The dynamic
"20→50 doubles" was **inertial overshoot amplitude**, not bulge convergence.
→ **Iteration is NOT the cause.** (Contradicts session-1 Result 2b, which was a
dynamic artifact.)

## Experiment B — vary `belly_attach` stiffness (isolate belly-attach)

`belly_attach` with `stiffness=0` is INERT (`solve_kernels.py:915` `if cstiffness
<= 0.0: return`), NOT a hard pin. `belly_attach` and `muscleToBonePin` strictly
PARTITION the muscle verts (single `pin_active` mask, `mg.py:588-591`): raising
`tendon_pin_threshold` → fewer hard pins → more belly-attach.

| belly_k (fine_iters=200) | dynamic max_disp | quasi-static max_disp |
|--------------------------|-----------------:|----------------------:|
| 0     | 16.50 | 2.603 |
| 1e4   | 1.65  | —     |
| 1e6 (humanV2.json) | 1.65 | 1.633 |
| 1e8   | 1.65  | —     |

Static: belly-attach cuts bulge 2.60→1.63 mm (**~37 %**), saturating (1e4≈1e6≈1e8).
Dynamic: it damps the free-belly overshoot 16.5→1.65 mm (~10×). → belly-attach is a
real suppressor, but most of its apparent "10× kill" is damping inertial overshoot.

## The real #1 — `fibernorm` compliance is normalized by rest VOLUME

Even belly-off + fully converged, static bulge is only **2.6 mm** and along-fiber
shrink only **0.18 %** while `target_stretch=0.4` asks for 60 % shortening. Cause:
`fibernorm` sets `alpha = 1/(k · restlength · dt²)` where `restlength = tet rest
volume` (`builder.py:226`, `solve_kernels.py:312-314`). With V0 ~ 1e-9, alpha is
~1e9× too large → the XPBD constraint is far too soft → it never reaches its target.
(The reference `pbd_constraints.cl:1128` notes this volume term is the physically
correct FEM normalization — but then `k` must be ~1e17, not 1e8, to bite.)

### Fix — HOBAK anisotropic-ARAP (`fiber_model="aniso_arap"`, TETARAPANISO)

`solve_tet_aniso_arap_kernel` (`solve_kernels.py:1311`) is the reference's
**branch-1** (quadratic HOBAK `0.5*(s − SI4·fiberscale)²`, `s=|F·f0|`,
`SI4=sign(detF)`, guard `detF<=0 or s>1.5`). Crucially `alpha = 1/(k · dt²)` — **no
volume term** — so it reaches the target.

| model (R bicep, tip-pin, belly 0, ratio 0.6, iters 100) | k | max_disp | shrink |
|---------------------------------------------------------|----|---------:|-------:|
| fibernorm (quasi-static) | 1e8 | 2.60 mm | 0.18 % |
| **aniso_arap (quasi-static)** | 1e8 | **25.54** | **1.52 %** |
| aniso_arap (quasi-static) | 1e6 | 25.55 | 1.48 % |
| aniso_arap (quasi-static) | 1e4 | 25.56 | 1.50 % |
| fibernorm (dynamic) | 1e8 | 14.60 | 0.34 % |
| aniso_arap (dynamic) | 1e6 | 28.98 | 1.78 % |

aniso_arap gives **~10× the static bulge** and is **k-insensitive** (1e4=1e6=1e8) =
target-reached. Its contraction is TARGET-DRIVEN (25 mm at static), not dependent on
inertial overshoot like fibernorm. Probe it with
`tools/fiber_probe_humanV2.py --fiber-model aniso_arap`.

## `fibernorm` vs `pbd_constraints.cl:1093` (`tetFiberUpdateXPBD`)

The reference fiber constraint has two inline branches: **branch-1** anisotropic-ARAP
(HOBAK, `L1146-1176`: `I5`, `I4/SI4` inversion sign, singularity guard, target
`fiberscale=rv.w`) and **branch-2** linear passive (`L1177-1204`: precomputed
`wᵀDm⁻ᵀ`, `√I5`, no target). The Warp `fibernorm` = **branch-2 + a grafted target**
(`C=|Fw|−target_stretch`, `solve_kernels.py:362`); it OMITS branch-1's `SI4`
inversion sign, singularity guard, and `det(F)` check → no robustness under strong
contraction/inversion. Warp instead exposes branch-1 as the separate
`aniso_arap`/TETARAPANISO constraint. Isotropic `tetarapsvd` ARAP runs alongside
either.

## Cause ranking (why humanV2 showed no visible contraction)

1. **`fibernorm` rest-volume compliance** — effective stiffness `k·V0` ~1e9× too
   small → never reaches target (0.18 % shrink). **Fix: `aniso_arap`** (default now).
2. **`belly_attach` (k=1e6)** — clamps 86 % of belly verts to their bone-rest target
   (static −37 %, dynamic damps overshoot). Fix: free belly, tip-pin only.
3. **`fine_iterations=20`** — affects only the dynamic response amplitude; static
   bulge converges by 50. Minor.
- **Ruled out: GPU Jacobi.** bicep contracts under all 3 solvers; humanV2 static
  bulge is iteration-independent. Jacobi is not producing wrong numbers.

Defaults changed: `mg.py` `fiber_model` default `fibernorm → aniso_arap`;
`example/humanV2_fiber.py` and `tools/fiber_probe_humanV2.py --fiber-model` likewise.
`otto.json`/`otto_walk.json` keep explicit `fibernorm` (unchanged).

