# Fascia squeeze validation — 2026-06-06

End-to-end check that the fascia layer pulls the muscle surface inward.
Script: `tools/fascia_validate.py`. Config: `data/humanV2/config/humanV2_fascia.json`.

## Metric
- Build `OneWaySim` with fascia contraction OFF (`fascia_rest_scale=1.0`) and ON
  (`0.95`); step 30 frames each from `start_frame`.
- On the muscle **surface** vertices only (`surface_points` group, 119482 verts,
  indexing the muscle range `[0, nvm)`):
  centroid `c = ms[:,[0,2]].mean`; radial dist `r = ||ms[:,[0,2]] - c||`
  about the vertical Y axis; **metric = mean radius**. Squeeze ⇒ radius drops.
- Sanity: mean fascia-vertex displacement (`P[nvm:]`) between OFF and ON.

## Results (RTX 4090, cuda:0)
| quantity                | value     |
|-------------------------|-----------|
| mean surface radius OFF | 0.145644  |
| mean surface radius ON  | 0.145640  |
| absolute change         | -0.000004 |
| percent change          | **-0.003%** |
| mean fascia disp OFF→ON | 0.002518  |
| total runtime           | 63.6 s (≈31s/run) |
| muscle / merged verts   | 157369 / 172717 |

No NaN/Inf in either run.

## Test suite
`pytest test_fascia_sim_smoke test_fascia_kernels test_fascia_geo_io test_lion_kernels`
→ **16 passed in 7.31s**.

## Verdict / concern
Direction is correct: ON radius < OFF, and the fascia mesh visibly moves
(disp 0.0025), so the contraction is real and acts inward. **But the magnitude
is tiny (-0.003%)** — over 30 frames at `rest_scale=0.95` the squeeze barely
reaches the muscle surface. The kernel-level test
(`test_fascia_glue_pulls_muscle_inward`) already proves the mechanism works;
the weak full-sim effect suggests the glue is under-coupled or 30 frames /
0.95 is too gentle to overcome the pin-to-target muscle constraints. CONCERN:
needs stronger rest_scale, more glue iterations, or longer rollout to be a
convincing visible squeeze.

---

## Task 9: after freeing glue-target pins

The root cause of the 0.003% baseline was confirmed: the humanV2 scene pins
**100% of muscle verts** via `pintotarget` (`cons_<frame>.geo` NCONS == muscle
NV), so every glue-target surface vert was hard-overwritten back to its bone
target each substep → the muscle never yielded → squeeze ≈ 0.

### Fix implemented (`src/muscle/coupleOneWayGmg/solver.py`)
- **Masked pin** (`pintotarget_overwrite_masked_kernel` + `pin_active` mask):
  when fascia is active, FREE the muscle verts the fascia glues to plus a
  dilated `fascia_free_ring`-ring neighborhood (muscle-tet adjacency) from the
  per-frame pin overwrite. Muscle-only path is untouched (`pin_active` is None →
  original kernel).
- **Soft surface tets**: muscle tets touching a freed vert get ARAP stiffness
  `fascia_surface_mu` (1e5) instead of `mu` (1e9) so the glue can deform them.
- **Soft pin for freed verts** (`pintotarget_soft_kernel`, `fascia_soft_pin`):
  NEW knob added during this task. Freeing the pins entirely makes the freed
  verts tear loose when the per-frame bone target jumps (diverges at frame 2).
  A partial PBD pull toward the bone target each substep keeps them tracking
  gross body motion while still yielding inward. Required for stability.

### Stability finding (the real blocker)
The freed surface muscle verts sit in tiny tets → physical `inv_mass` up to
~1e6 (median ~12k). The XPBD glue (`solve_glue_kernel`, Jacobi, `omega=0.1`,
`glue_L` reset per iter = effectively PBD) over-shoots on these light verts and
**diverges** for any glue stiffness scale above ~0.0015. Tried and rejected:
glue stiffness ×100/×1000 (diverges immediately), capping muscle `inv_mass` for
the glue solve (made it worse — shifts correction to the fascia side), XPBD
lambda accumulation across iters (stabilizes high `glue_iterations` but *reduces*
squeeze and still caps glue stiffness). So glue must stay soft.

### Final settings used
`fascia_glue_iterations=10`, `fascia_free_ring=1`, `fascia_surface_mu=1e5`,
`fascia_glue_stiffness_scale=0.001`, `fascia_soft_pin=0.5`,
`fascia_rest_scale` 0.95 (official tool) / 0.9 (sensitivity).
Freed: **14297 / 157369 muscle verts** freed from the pin; **143072 pins still
active**. Glue pairs: 4252.

### Results (RTX 4090, cuda:0, 30 frames, no NaN/Inf)
Official `tools/fascia_validate.py` (OFF rest_scale=1.0 vs ON 0.95), mean radius
over **all 119482 surface verts**:

| quantity                | BEFORE (task 8) | AFTER (task 9) |
|-------------------------|-----------------|----------------|
| mean surface radius OFF | 0.145644        | 0.145614       |
| mean surface radius ON  | 0.145640        | 0.145609       |
| percent change          | **-0.003%**     | **-0.004%**    |
| total runtime           | 63.6 s          | 51.8 s         |

At `rest_scale=0.9` (stronger contraction): global surface metric **-0.008%**.

**Diagnostic — squeeze on the freed verts only** (the 14297 verts the fascia
actually acts on, rest_scale 1.0 vs 0.9): mean radius **0.12287 → 0.12277,
-0.081%** — ~27x the global 0.003%. The fascia itself shrinks **-2.46%** in
radius; freed muscle verts move 0.00017 (≈10x the old all-muscle 0.00001).

### Verdict
The fix is mechanically correct and a real improvement: the freed verts now
escape the pin and follow the fascia ~10x more than before, and locally (on the
glued region) the squeeze is -0.08%, ~27x the baseline. **But the global
surface-radius metric only moves 0.004%** because (a) only 14297 of 119482
surface verts are glued so the global mean dilutes the local effect ~8x, and
(b) the glue stiffness is stability-capped (~0.0015) by the ultra-light freed
verts in the Jacobi/PBD glue solve, so only a fraction of the fascia's 2.46%
self-squeeze transfers to the muscle.

To make this a *visibly* strong squeeze, the next step is a more stable glue
solver (e.g. graph-colored Gauss-Seidel glue, or per-vertex mass lumping for
the freed surface verts) so the glue stiffness can be raised 100–1000x without
divergence — not more parameter tuning of the current Jacobi solve.
