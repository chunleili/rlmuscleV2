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
