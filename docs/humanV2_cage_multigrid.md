# humanV2 Cage Multigrid — Conforming Voxel Cage + Galerkin Restrict

How the humanV2 muscle solver accelerates its quasi-static ARAP solve with a
per-muscle embedding cage, and **why the naive multigrid that ships in most PBD
code makes things worse here**. This is a principle doc — the measured numbers
are from the build that produced it.

Assets / code:
- `tools/gen_coarse_cage.py` — bakes the cage + transfer operators → `data/humanV2/model/muscle/coarse_cage.npz`
- `src/muscle/coupleOneWayGmg/solver.py` — `_init_coarse_cage`, `restrict_galerkin_scatter_kernel`, `block_solve_kernel`, the `pre_phase` cage branch
- enable via config: `use_multigrid: true`, `pin_belly_attach: true`, `coarse_npz_path: ".../coarse_cage.npz"`

---

## 1. The problem

humanV2 is **pin-driven**: every muscle vertex is pulled toward a per-frame bone
target. Tendons (`tendonmask > 0`, ~32%) are hard-pinned to the bone; the belly
(~68%) rides a compliant XPBD *attach* constraint so collision/fascia can push it
off the target. The elastic model is ARAP-SVD projected with XPBD parallel-Jacobi.

Multigrid is supposed to accelerate the elastic solve: relax low-frequency error
on a cheap coarse mesh, prolong the correction back to the fine mesh. The classic
geometric-MG V-cycle is:

```
restrict (fine → coarse) → coarse smooth → prolong (coarse → fine) → re-pin
```

Two things make this *backfire* on muscle, and both were measured, not assumed.

### Failure 1 — the stock coarse mesh is not a cage (→ spikes)

The shipped `restpos_lowres.geo` is a surface-fit decimation, roughly the same
*volume* as the fine mesh, not an enclosing cage. Measured (fine `restpos.geo`
vs the lowres):

| metric | value |
|---|---|
| fine verts **inside** a coarse tet | **38.2 %** |
| fine verts **outside** all coarse tets (→ nearest-tet fallback) | **61.8 %** |
| prolongation barycentric weight range | **−29.7 … +32.5** |
| prolong source tet from a **different muscle** | **21.5 %** |

A point outside its coarse tet is **extrapolated**: barycentric weights leave
`[0,1]` and blow up to ±32. At rest this reconstructs exactly (affine), so it
*looks* fine — but under deformation a coarse displacement `d` on a weight-32
vertex becomes `32·d` on the fine vertex. That amplification is the gmg **尖刺
(spikes)**, concentrated where the cage fails to enclose and at the tendon/belly
seam.

### Failure 2 — full pin makes MG a no-op, free pin makes it diverge

`pre_phase` re-applies the pin *after* prolong. In **full-pin** mode the pin
overwrites every vertex, so the prolonged coarse correction is erased → MG is an
exact no-op. MG can only ever act on the **free belly**. But a free belly fed
through a naive restrict diverges: measured ARAP residual RMS after the coarse
step was **0.67**, vs **0.006** for no-MG — ~100× worse, NaN within a few frames.

So the honest baseline was: *naive cage-MG is strictly harmful for this problem.*

---

## 2. The fix, part A — a conforming voxel cage

A multigrid/embedding cage must **enclose** the object so every fine vertex is a
*convex* combination of coarse vertices (weights in `[0,1]`, no extrapolation).
Surface-fitting (fTetWild, decimation) does **not** give this — measured, an
fTetWild per-muscle cage still left ~half the verts outside.

The construction that does is a **voxel embedding cage**, built **per muscle**
(`tools/gen_coarse_cage.py`):

1. voxelize the muscle's fine verts at edge length `h` (default 0.045 m);
2. keep occupied voxels, split each into 6 tets.

Every fine vertex lies in its own voxel **by construction** → it is contained in a
**same-muscle** coarse tet with barycentric weights in `[0,1]`. Measured on all
199 muscles:

| metric | lowres (old) | voxel cage |
|---|---|---|
| fine verts inside | 38.2 % | **100 %** |
| prolong weight range | −29.7…+32.5 | **0 … 0.98** |
| cross-muscle coupling | 21.5 % | **0** (per-muscle) |
| coarse size | 13393 V / 35575 T | 10352 V / 19014 T |

This kills the spikes geometrically: a convex combination **cannot** amplify.
Per-muscle cages also mean a coarse tet never spans the gap between two muscles.

The cage gives the **prolongation operator** `P` (fine = `P` · coarse), stored as
`cage_idx` (fine vert → coarse tet) + `uvw` (barycentric).

---

## 3. The fix, part B — Galerkin least-squares restrict (the crux)

Fixing the cage removed the spikes but the solve *still* diverged. The decisive
measurement: the strain came from the **restrict**, not the prolong.

The intuitive restrict — set each coarse vertex from its nearest fine vertex
(*gather*) — jitters the cage: adjacent corners grab unrelated fine targets, and
corners outside the muscle grab whatever is nearest. The resulting cage
deformation is itself high-strain, and **any** prolong (voxel barycentric *or*
smooth MVC/Green coordinates) faithfully interpolates that bad cage. We tested
MVC (Mean Value Coordinates) — it did **not** help, because the operator's
smoothness was never the bottleneck.

The right restrict is a **least-squares projection**: find the cage deformation
`Cdef` whose prolongation best reproduces the bone target `T`,

```
Cdef = argmin_C ‖ P·C − T ‖²   ⇒   Cdef = (PᵀP)⁻¹ Pᵀ T
```

This is exactly the **Galerkin restriction** `R = Pᵀ` with the `(PᵀP)⁻¹` mass
normalization. Geometrically it projects the bone target onto the cage's
deformation subspace — the **smoothest** field the cage can represent that still
reproduces `T`. Because it lives in the low-dimensional cage subspace, it filters
out the per-vertex high-frequency content that injects ARAP strain.

`P` is block-diagonal per muscle (a fine vert only weights its own muscle's cage),
so `PᵀP` is block-diagonal; we bake the per-muscle dense inverse `Minv =
pinv(PᵀP)` (`pinv` tolerates cage corners no fine vert references — their value is
irrelevant, prolong never reads them). Total `Minv` storage: **3.1 MB**.

Per substep the restrict is two cheap launches:
- `restrict_galerkin_scatter_kernel`: `b = Pᵀ T` — scatter each fine target into
  its coarse tet's 4 corners, weighted by the prolong weights (atomic add);
- `block_solve_kernel`: `Cdef = Minv · b` — per-muscle dense block mat-vec.

### Injected-strain comparison (single muscle, real bone frame)

ARAP residual RMS of the prolonged fine config (lower = smoother, less strain):

| restrict \ muscle | 5 | 50 |
|---|---|---|
| true bone target (reference) | 0.178 | 0.034 |
| voxel + **gather** | 0.657 | 0.874 |
| MVC + gather | 0.732 | 0.889 |
| **voxel + lsq (Galerkin)** | **0.106** | **0.029** |
| MVC + lsq | 0.214 | 0.111 |

lsq cuts injected strain ~6× and lands **at or below** the target's own strain.
MVC/Green coordinates are unnecessary — the plain voxel cage with the right
restrict wins.

---

## 4. How it runs each substep

`pre_phase` (cage + `pin_belly_attach`), the **full-pin scaffold** — coarse runs
no ARAP, it is purely a bone-driven target projector:

```
scatter_target            fine_target_dense[v] = bone target               (dense T)
b = Pᵀ T                  restrict_galerkin_scatter_kernel                 (coarse)
Cdef = Minv · b           block_solve_kernel  → coarse_target              (Galerkin lsq)
coarse_pos ← Cdef         copy
fine ← P · coarse_pos     prolong_kernel  (every muscle vert)              (smooth belly start)
re-pin tendons            apply_pintotarget  (pin_active hard only)
```

Then `iter_phase` runs ARAP + volume + fiber + **belly attach** + collision from
that start, and `post_phase` does fascia + collision. The belly's per-substep
initial guess is now the smooth cage projection of the bone target — low strain,
no spikes, no blocky imprint.

There is no free coarse ARAP and no V-cycle iteration: the "multigrid" here is a
single linear *target-smoothing* operator `P (PᵀP)⁻¹ Pᵀ` applied to the bone
target. That is what a pin-dominated quasi-static problem actually needs.

---

## 5. Result — convergence vs the non-MG baseline

In-solver ARAP residual RMS vs fine-iteration count (attach system, ω = 0.1):

| iter | non-MG (baseline) | cage-MG **gather** | cage-MG **lsq** |
|---|---|---|---|
| 0 | 5.70e-3 | 6.73e-1 | **2.92e-3** |
| 5 | 1.78e-3 | 1.56e-1 | 1.34e-3 |
| 10 | 1.54e-3 | 7.40e-2 | 1.14e-3 |
| 30 | 1.27e-3 | 3.37e-2 | **9.18e-4** |

cage-MG-lsq beats the non-MG baseline at **every** iteration: ~2× lower initial
residual (better start), ~1.4× lower final residual, and it reaches the baseline's
30-iter value in ~5–10 iters → **3–6× fewer iterations** for the same accuracy.
gather stays ~50–100× worse, confirming the bottleneck was the restrict.

---

## 6. Lessons

- **A coarse mesh is not a cage.** Multigrid/embedding needs *containment*
  (weights in `[0,1]`); surface-fit decimation gives extrapolation → spikes. Build
  an enclosing voxel cage, per muscle.
- **The transfer operator quality lives in the restrict, not the prolong.** A
  smooth prolong (MVC/Green) over a jittery cage is still bad. Use the Galerkin
  least-squares restrict `(PᵀP)⁻¹Pᵀ` — project the target onto the cage subspace.
- **Match the method to the problem.** This is pin-dominated quasi-statics, so the
  win is a one-shot smooth target projection, not a full elastic V-cycle.

## 7. Regenerate / tune

```
PYTHONPATH=src uv run python tools/gen_coarse_cage.py \
    --rest data/humanV2/model/muscle/restpos.geo \
    --out  data/humanV2/model/muscle/coarse_cage.npz \
    --voxel 0.045
```

`--voxel` is the resolution/smoothness knob (smaller = finer cage, less
smoothing, more coarse DOF). Set `debug_residual: true` in the config to log the
per-iteration ARAP residual for re-measuring convergence.
