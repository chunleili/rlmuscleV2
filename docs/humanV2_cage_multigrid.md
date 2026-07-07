# humanV2 Multigrid — HPBD State-Sync V-Cycle on an Enclosing Cage

How the humanV2 muscle solver (`src/muscle/mg/mg.py`, `MultiGridSim`) accelerates
its ARAP solve with a coarse embedding cage, and **the recipe that finally makes a
coarse-relaxation multigrid beat the single-level solver under a bone boundary
condition**. This is a principle doc; the numbers are from the build that
produced it (branch `mg`).

> **Current status.** `mg_mode` selects the scheme:
> - `off` — no multigrid (single-level fine XPBD).
> - `projection` — the original one-shot target smoother (restrict the bone
>   target onto the cage, prolong; no coarse relaxation). Stable but biased; kept
>   as a fallback.
> - `hpbd` — **the validated scheme described here**: a per-substep V-cycle with
>   Galerkin state-restrict, coarse ARAP relaxation, and delta-prolongation. It
>   Pareto-beats `off` on shape/inversions under the real (attach) boundary
>   condition, once three consistency conditions hold (§4).
>
> The cage npz (`data/humanV2/model/muscle/coarse_cage.npz`) is gitignored and
> auto-baked on first run if missing (~90 s fTetWild inflate cage).

Assets / code:
- `tools/gen_coarse_cage.py` — bakes the enclosing cage + transfer operators.
- `src/muscle/mg/mg.py` — `MultiGridSim`, `_init_coarse`, `_hpbd_vcycle`,
  `_restrict` (Galerkin state-restrict), `_coarse_solve`, `_prolong` (delta).
- `src/muscle/mg/kernel.py` — `restrict_galerkin_scatter_kernel`,
  `block_solve_kernel`, `solve_tetarap_svd_kernel` (coarse ARAP),
  `prolong_delta_kernel`, `arap_residual_accum_kernel`.
- config: `use_multigrid: true`, `mg_mode: "hpbd"`, `mg_coarse_iters`,
  `mg_coarse_attach_beta: 0.0`, `coarse_npz_path`.

---

## 1. The problem

humanV2 is **bone-driven**: every muscle vertex is pulled toward a per-frame bone
target through a compliant XPBD *belly-attach* (stiffness 1e6). This spring is a
**Robin** (elastic-support) boundary condition — traction ∝ `k(target − x)`, a
function of the displacement — not Neumann (which prescribes a
displacement-independent traction); in the limits `k→∞` it becomes Dirichlet
(hard pin) and `k→0` becomes Neumann (free). There are no hard pins on this asset
(the `pinmask` primvar is empty). The elastic model is ARAP-SVD
projected with XPBD parallel-Jacobi.

Multigrid should accelerate the elastic solve: relax the low-frequency error on a
cheap coarse mesh and prolong the correction back to the fine mesh. The V-cycle is

```
restrict (fine → coarse) → coarse solve → prolong (coarse → fine) → fine post-smooth
```

Two independent things have to be right for this to help rather than hurt: the
**transfer operators** (§2–3), and the **consistency of the coarse problem**
(§4). The second is what took the longest and is the real contribution.

---

## 2. The enclosing cage (transfer geometry)

A multigrid/embedding cage must **enclose** the object so every fine vertex is a
*convex* combination of coarse vertices (barycentric weights in `[0,1]`, no
extrapolation). A surface-fit decimation does **not** give this — a fine vertex
outside its coarse tet is extrapolated, its weights leave `[0,1]` and blow up, and
a coarse displacement `d` on a weight-32 vertex becomes `32·d` on the fine vertex
(the historical gmg **尖刺 / spikes**).

The shipped cage is an **inflate** cage (`gen_coarse_cage.py --method inflate`),
anatomy-conforming, ~99.99% containment: coarse NV=28131, NT=61108 enclosing the
fine NV=157369, NT=468162. Every fine vertex is a convex combination of its
enclosing coarse tet's 4 corners → the prolongation operator `P` (`fine = P ·
coarse`) has weights in `[0,1]` and cannot amplify. `P` is stored as `cage_idx`
(fine vert → coarse tet) + `uvw` (barycentric).

> The old per-muscle **voxel** cage is deprecated; the inflate cage supersedes it.
> A **c2f** restrict (coarse-verts-embedded-in-fine-tets, the bunny/realtimeMuscle
> scheme) was tested and **rejected** for this cage: it needs a decimation /
> co-located coarse (coarse ⊂ fine), but the inflate cage is the opposite
> containment (fine ⊂ coarse) → 69% of coarse verts extrapolate, |w| up to 15 →
> blowup. Galerkin restrict (below) is the correct operator for an enclosing cage.

---

## 3. Galerkin least-squares restrict

The intuitive restrict — set each coarse vertex from its nearest fine vertex
(*gather*) — jitters the cage: adjacent corners grab unrelated fine positions and
the resulting cage deformation is itself high-strain; any prolong then faithfully
interpolates that bad cage.

The right restrict is a **least-squares projection**: find the cage state `x^c`
whose prolongation best reproduces the fine field `x^f`,

```
x^c = argmin_C ‖ P·C − x^f ‖²   ⇒   x^c = (PᵀP)⁻¹ Pᵀ x^f
```

This is the **Galerkin restriction** `R = Pᵀ` with the `(PᵀP)⁻¹` mass
normalization — the orthogonal projection of `x^f` onto the cage subspace, i.e.
the smoothest field the cage can represent. `P` is block-diagonal per muscle, so
`PᵀP` is block-diagonal; we bake the per-muscle dense inverse `Minv = pinv(PᵀP)`
(≈3 MB). Per call the restrict is two launches: `restrict_galerkin_scatter_kernel`
(`b = Pᵀ x^f`) then `block_solve_kernel` (`x^c = Minv · b`). Derivation:
[`restrict.md`](restrict.md).

In `hpbd` mode this operator restricts the **fine state** (`sim.pos`) each
substep — the "state-sync" in §4 — not just the bone target.

---

## 4. Why coarse-relax MG failed before, and the fix

Running a **coarse ARAP relaxation** (not just a one-shot projection) used to blow
the fine residual up ~100× and NaN within a few frames under the bone BC. That was
blamed on cage inversion / the enclosing cage being unfit for relaxation. It was
**none of those** — it was three separate *consistency* violations. The `hpbd`
mode fixes all three; each is necessary.

**(a) State-sync restrict (the double-drive bug).** The coarse level must be
re-synced to the fine state *before* it relaxes, so its per-substep displacement
is a true residual correction that vanishes when the fine is already smooth.
Without it the coarse ran as a free sim whose per-substep delta equalled the
bone-target increment; delta-prolong then added a **second** copy of the bone
motion on top of the fine's own belly-attach → double-drive → blowup. Fix:
`_restrict` sets `coarse_pos = (PᵀP)⁻¹Pᵀ x_fine` (Galerkin state-restrict, §3)
each substep before saving `coarse_prev`. This is the *implicit* property of a
conforming (nested coarse ⊂ fine) hierarchy — restrict = identity there — emulated
on the non-nested inflate cage by the explicit Galerkin projection.

**(b) FAS-consistent coarse rest shape.** The coarse ARAP rest must be
`coarse_B = restrict(fine_rest)`, i.e. built from `q0 = (PᵀP)⁻¹Pᵀ x_rest`, **not**
the baked cage rest. If the coarse rest is the cage's own rest, then
`restrict(fine_rest) ≠ cage_rest`, so the coarse ARAP injects strain even when the
fine is at global rest → the coarse delta never vanishes → bias. Built once in
`_init_coarse`: Galerkin-restrict `pos0` to `q0`, then
`coarse_B = _rest_B_p3ref(q0, coarse_tet)`.

**(c) No coarse attach BC (`mg_coarse_attach_beta = 0`).** A coarse soft-attach
toward `restrict(bone target)` is **harmful**: on the enclosing cage the
under-referenced outer nodes get ill-conditioned target values, so the attach
yanks the cage and delta-prolong sprays that error into the fine. The state-sync
restrict (a) already carries the fine's BC state into the coarse every substep, so
the coarse needs no boundary condition of its own — it only does low-frequency
ARAP smoothing of the synced state.

Delta-prolongation (Müller 2008 *Hierarchical Position Based Dynamics* §7,
eq 7) is `fine.pos[i] += Σ_j w_ij (coarse_pos[j] − coarse_prev[j])`. It is
unbiased **iff the coarse delta → 0 at the fine equilibrium** — which is exactly
what (a)+(b)+(c) guarantee.

---

## 5. How the boundary condition is handled (per level)

This is the crux of an unbiased V-cycle, so it gets its own section. **The BC
lives on the fine level only; the coarse level is never given a boundary condition
of its own.** Three cases by BC type:

### Fine level — where the BC actually acts

The compliant belly-attach (§1, the Robin BC) is a constraint in the fine
post-smooth. `_update_belly_attach_targets` refreshes each vert's target
(bone-surface projection) every substep, and the ATTACH constraint pulls `x`
toward it with XPBD compliance `α = 1/(k·dt²)`. This is the **only** place the
Robin BC is enforced. Hard pins (`pinmask`, if present) are the fine Dirichlet BC,
overwritten by `apply_pintotarget`.

### Coarse level, the Robin part — carried implicitly by state-sync, NOT re-applied

The coarse never sees the bone target. It sees the fine **state** through the
Galerkin state-restrict `coarse_pos = (PᵀP)⁻¹Pᵀ x_fine` (§3, §4a), and that fine
state already reflects the applied belly-attach. So the coarse inherits the
BC-satisfying configuration for free and only smooths its low-frequency error.

Giving the coarse its own attach (`mg_coarse_attach_beta > 0`, a `dense_soft_attach`
toward `restrict(bone target)`) is **harmful** and disabled (`beta = 0`):
- **Double-drive / τ-inconsistency.** The fine already applies the BC and
  state-sync carries it up; a coarse re-drive makes the coarse solve a *different*
  problem, so its per-substep delta does not vanish at the fine equilibrium →
  delta-prolong biases the solution (this is the bias that plagued earlier builds).
- **Ill-conditioning.** `restrict(bone target)` is ill-conditioned on the enclosing
  cage's under-referenced outer nodes; the attach yanks the cage there and
  delta-prolong sprays the error into the fine.

(Contrast `mg_mode=projection`: it *does* teleport the coarse to `restrict(bone
target)` each substep — a hard target enforcement — which is exactly why it is the
more biased, less robust mode.)

### Coarse level, the Dirichlet part — kinematic pin (for pinmask assets)

If the fine asset has hard pins, those pins **are** re-imposed on the coarse
(essential BCs must be, see below). `_init_coarse` builds `coarse_pin_active` by
scattering the fine `pin_active` onto cage corners weighted by `|w|` and
thresholding (`mg_coarse_pin_frac`); `_coarse_kinematic_pin` overwrites those
coarse nodes to the restricted bone pose **before** the ARAP sweeps and holds them
(inv_mass 0) during, so the coarse relaxes from a correct pinned boundary. humanV2
has 0 pins → no-op here. (This is also why the synthetic `pin` bench regime loses:
its pins are injected *after* construction, so the coarse never learns them.)

### Why the split

A Robin/Neumann BC is a **natural** condition — it enters through the forces, which
the fine integrates and the state-sync faithfully restricts; re-applying it on the
coarse would double-drive. A Dirichlet BC is an **essential** condition — it
constrains the solution space itself, so the coarse solve must be told explicitly
(the kinematic pin) or it relaxes into the wrong (unpinned) nullspace.

## 6. The hpbd V-cycle, per substep

`_hpbd_vcycle` (`src/muscle/mg/mg.py`):

```
restrict          coarse_pos = (PᵀP)⁻¹ Pᵀ x_fine        Galerkin STATE restrict   (a)
snapshot          coarse_prev ← coarse_pos
coarse solve      _coarse_solve():
                    (kinematic-pin any Dirichlet coarse nodes — none on this asset)
                    mg_coarse_iters × ARAP-SVD sweep on coarse_B=restrict(rest)     (b)
                    NO coarse attach (beta=0)                                        (c)
prolong           fine.pos[i] += Σ_j w_ij (coarse_pos[j] − coarse_prev[j])   delta-prolong
```

Then the normal fine iteration (ARAP + volume + fiber + belly-attach + collision)
runs as the **post-smooth**. Coarse mass is the fine total (70 kg) redistributed
by coarse tet volume. There is no forced target teleport and no free coarse
drive — the coarse is a *residual smoother* re-anchored to the fine each substep.

---

## 7. Results

**Round-trip `cycle` regime** (rest → far bone pose → rest, so the final
residual-vs-rest is a *meaningful* recovery metric, not a folding-blind number).
Final ARAP-C RMS / inverted-tet count, `off` vs `hpbd`:

| regime | off | hpbd | note |
|---|---|---|---|
| squash (no BC) | 0.160 / 300 inv | **0.120 / 152 inv** | hpbd better + fewer inversions |
| attach (Robin / elastic-support, the real BC) | 0.092 / 97 inv | **0.081 / 24 inv** | hpbd better + 4× fewer inversions |
| **cycle** (round-trip) | 0.117 / 273 inv | **0.034 / 3 inv** | **3.4× better recovery, ~zero inversions** |
| pin (Dirichlet, synthetic) | 0.889 / 3887 inv | 1.92 / 5042 inv | hpbd loses (see below) |

`hpbd` Pareto-beats `off` on the three non-synthetic regimes with far fewer
inversions. On the full 200-frame otto motion it beats `off` on bone-tracking
(~2.4× tighter) and crushes the `projection` mode on robustness (~890 vs 2493
inverted tets — `projection`'s absolute cage teleport is far more violent than the
gentle delta-prolong).

The **pin** regime still loses, but this is a *synthetic* bench artifact: it
hard-pins 15% of fine verts *after* construction, so `coarse_pin_active` (built in
`_init_coarse`) never learns those pins and the coarse smooths across the pinned
boundary. The real humanV2 BC is attach, which hpbd wins; a pinmask asset would
need the coarse pins rebuilt post-construction.

Validation harnesses:
- `src/example/hpbd_conv_curve.py` — 4-regime C-residual + inversion convergence
  curves (writes CSV + `output/hpbd_conv/conv_cresidual_F*.png`).
- `src/example/hpbd_cycle_anim.py` — bakes the round-trip as a point-animated USD
  (`off` vs `hpbd`) for visual A/B.

---

## 8. Lessons

- **Consistency, not the cage, was the wall.** A coarse relaxation MG under a BC
  needs: (a) the coarse re-synced to the fine state each step, (b) a coarse rest
  equal to `restrict(fine_rest)`, (c) no independent coarse BC. Miss any one and
  delta-prolong biases or diverges. All three make the coarse delta vanish at
  equilibrium → the prolonged correction is a true residual correction.
- **Validate with a shape / inversion / round-trip-recovery metric, not ARAP
  residual alone.** The ARAP C-residual is blind to folding, global rotation, and
  crumpling — a smoothly varying rotation field folds the body at ~zero per-tet
  energy. The round-trip `cycle` (ends at rest) is the honest test. An *exact
  global* coarse solve was tried and **reverted**: it drove the residual ~30–60×
  lower while visibly crumpling the mesh (volume-preserved, locally-rigid, bbox
  collapsed). The coarse solver stays a **pooled ARAP-SVD Jacobi** sweep — its
  gentleness protects a free/under-constrained coarse body from jumping into a
  fold.
- **The transfer operator quality lives in the restrict, not the prolong.** The
  Galerkin lsq restrict `(PᵀP)⁻¹Pᵀ` (projection onto the cage subspace) is what
  makes an enclosing cage work; c2f/gather do not.
- **Drive the round-trip by real bone frames, not a straight-line position lerp.**
  A linear world-space lerp of a limb's rotation chords through the arc and
  collapses the distal targets mid-cycle, spaghettifying thin forearm/hand
  muscles. Sweeping the real bone frame index makes the attach target ride the
  true arc; `hpbd_cycle_anim.py` does this.

## 9. Regenerate / validate

```
# bake the enclosing cage (auto-runs on first sim if the npz is missing)
PYTHONPATH=src uv run python tools/gen_coarse_cage.py \
    --rest data/humanV2/model/muscle/restpos.geo \
    --out  data/humanV2/model/muscle/coarse_cage.npz --method inflate

# convergence curves (off vs hpbd, 4 regimes)
PYTHONPATH=src .venv/bin/python -m example.hpbd_conv_curve \
    --config data/humanV2/config/humanV2_hpbd_conv.json

# round-trip A/B animation (off vs hpbd) → output/hpbd_conv/muscle_cycle_*.usd
PYTHONPATH=src .venv/bin/python -m example.hpbd_cycle_anim \
    --config data/humanV2/config/humanV2_hpbd_conv.json
```
