# Fascia + Muscle Two-Way Simulation — Design

Date: 2026-06-06
Status: approved (brainstorm), pending spec review

## Goal

Add a **fascia** layer to the humanV2 muscle sim. Fascia is a tetrahedral shell
shrinkwrapped around the muscles. It **contracts inward (收缩)** — its rest shape
shrinks — so it squeezes the enclosed muscles together and keeps them tight, no
splaying. Models the Houdini Otis "fascia/tissue core" recipe.

Key correction baked into this design: the mechanism is **inward contraction**,
NOT per-frame nearest-neighbour *search*. Fascia↔muscle pairing is established
ONCE at build time; the runtime behaviour is contraction transmitted through a
fixed soft coupling.

## Decisions (from brainstorm)

| Decision | Choice |
|---|---|
| Coupling | **Two-way unified** — muscle + fascia in one solver; fascia tugs muscle |
| Integration | **Extend `src/muscle/coupleOneWayGmg`** (humanV2 production path) |
| Contraction | **Rest-scale shrink** of fascia tets (volume/shape) |
| Anchor | **Ride driven muscle** — fascia `rigid`/bone pts follow nearest muscle vertex |
| V1 scope | **Full design** (merged mesh + per-tet stiffness + scaled-B + glue + anchor + USD) |

## Inputs

- Muscle: `data/humanV2/model/muscle/restpos.geo` — 157,369 verts, 468,162 tets.
  Point groups `surface_points`, `interior_points`. Bone-driven via
  `cons_<frame>.geo` pin-to-target. Co-registered with fascia (same bounds, Y-up).
- Fascia: `~/Dev/fascia.geo` → copy into repo as
  `data/humanV2/model/fascia/fascia.geo` — 15,348 verts, 66,785 tets (+17,754
  surface polys, ignored for sim). Per-point attrs:
  `solidshapestiffness`(5000), `solidvolumestiffness`(25), `solidmassdensity`(900),
  `soliddampingratio`, `materialW`, `tpose`,
  `tissuetomuscledistance`(0.01)/`...stiffness`(1000)/`...damping`,
  `tissuetobonedistance`(0.01)/`...stiffness`(1000)/`...damping`.
  Point groups: `rigid` (ordered idx list), `tissue_bone_points` (boolRLE),
  `tissue_inner_points` (boolRLE). Prim groups: `fascia`, `tissue_core`,
  `tissue`, `tissue_core_surface`.

## Architecture

### 1. One merged particle system
Append fascia verts after muscle verts. Fascia tet indices offset by `muscle.NV`.
Single `pos` (≈172k verts), single `tet_idx` (≈535k tets). A per-tet `mat_tag`
array (0=muscle, 1=fascia) selects per-tet rest-scale and stiffness. The existing
`arap_svd_kernel` runs over all tets unchanged in shape — only its per-tet inputs
(`B`, `alpha`) differ by material.

### 2. Contraction via scaled rest matrix (no new contraction kernel)
ARAP-SVD drives `F = Ds·B` toward σ=1, i.e. current edges → rest edges encoded in
`B`. For a fascia tet, build `B = inv(s · Dm0)` with shrink factor `s = fascia_rest_scale < 1`.
ARAP then wants `Ds → s·Dm0` → the tet contracts to `s×` its loaded rest size each
solve. Whole fascia shell contracts uniformly inward. `solidvolumestiffness` is
subsumed (ARAP-SVD already controls volume through σ→1). Muscle tets keep `s=1`.

`precompute_rest_kernel` gains a per-tet scale input (1.0 for muscle, `s` for
fascia) applied to `Dm` before inversion. Rest volume / mass use the **unscaled**
`Dm0` so mass and density stay physical.

### 3. Per-tet stiffness (replaces uniform `mu`)
Current `alpha[t] = 1/(mu·V0)` uses one global `mu`. Make `alpha` per-tet:
- muscle tets: `mu = cfg.mu` (1e9), unchanged behaviour.
- fascia tets: `solidshapestiffness` (5000) from the fascia attr.

`precompute_rest_kernel` takes a per-tet stiffness array instead of scalar `mu`.

### 4. Tissue-to-muscle glue — the one new kernel
Build time (one-shot, CPU): KDTree query each fascia `tissue_inner_points` vertex
against muscle `surface_points` → nearest muscle vertex `j`. Emit pairs
`(fascia_i, muscle_j, restlen = tissuetomuscledistance≈0.01, compliance from
tissuetomusclestiffness)`. `tissue_bone_points` likewise pair to nearest muscle
vertex with `tissuetobonestiffness` (stiffer), acting as the near-bone hold.

Runtime: new `solve_glue_kernel` — XPBD **particle-particle distance constraint**
between two sim DOFs `pos[i]` and `pos[j]`:
```
n = pos[i] - pos[j];  d = |n|;  C = d - restlen
dlambda = -(C + a~·L) / (w_i + w_j + a~)
pos[i] += omega · w_i · dlambda · n/d
pos[j] -= omega · w_j · dlambda · n/d   # two-way: muscle moves too
```
Correction splits by inv_mass → fascia contraction pulls muscle inward (anti-splay)
and muscle pushes back. Per-pair `L` slot (one writer, no race). Color/atomic:
fascia_i unique per pair (inner pts pair once); muscle_j may repeat → use
`wp.atomic_add` on `pos`.

### 5. Anchor — ride the driven muscle
No bone mesh in this pipeline (bones only drive muscle through `pintotarget`). So:
- Fascia `rigid` pts: at build time bind each to its nearest muscle vertex
  (record that muscle index). Each frame, after muscle `pintotarget`, hard-overwrite
  `pos[fascia_rigid] = pos[bound_muscle_vertex] + rest_offset`. Mirrors
  `pintotarget_overwrite_kernel`. Keeps fascia from floating/rotating.

### 6. Frame loop (extends `OneWaySim`)
Per frame:
1. `load_frame_targets(frame)` + muscle `apply_pintotarget()` (existing).
2. `apply_fascia_anchor()` — overwrite fascia rigid pts to follow bound muscle verts.
3. `for _ in nsubsteps: substep()`:
   - multigrid V-cycle on **muscle only** (fascia is small; fine-level only).
   - re-apply muscle pintotarget + fascia anchor (prolong overwrites pinned).
   - fine ARAP over **all** tets (per-tet `B`, `alpha`).
   - **glue iters** (`fascia_glue_iterations`, default = fine_iters): launch
     `solve_glue_kernel`.

## New / changed components

| File | Change |
|---|---|
| `src/muscle/util/geo_io.py` | Extend `GeoFile`: handle `Tetrahedron_run` block (slice first N·4 of pointref indices, ignore Polygon_run); read named point attrs generically; decode point groups (`ordered` list + `boolRLE`). Keep existing trimmed API working. |
| `src/muscle/coupleOneWayGmg/solver.py` | New `precompute_rest_kernel` signature (per-tet scale + per-tet stiffness arrays); new `solve_glue_kernel`; new `fascia_anchor_kernel`; `OneWaySim` loads + merges fascia, builds glue pairs (scipy cKDTree) + anchor binding, extends `substep`/`step_frame`. |
| `data/humanV2/config/humanV2_fascia.json` | New config: extends gmg + `fascia_path`, `fascia_rest_scale`(0.95), `fascia_glue_iterations`. |
| `data/humanV2/model/fascia/fascia.geo` | Copy of `~/Dev/fascia.geo`. (Large; respect existing model.zip / gitignore convention — do NOT commit raw; document extraction.) |
| driver / `__main__` | Run sequence, export merged surfaces to USD (muscle + fascia, tagged colors). |

## Design for isolation
- `geo_io` stays a pure reader (no sim coupling); new group/attr access is additive,
  existing `positions`/`tet_indices`/`pts`/`target_pos` properties unchanged.
- `solve_glue_kernel` and `fascia_anchor_kernel` are standalone Warp kernels with
  explicit array inputs — testable in isolation on tiny synthetic meshes.
- Fascia build (merge, KDTree pairing, anchor binding) is one method
  `OneWaySim._init_fascia(...)`, guarded by `cfg.get("fascia_path")` so the plain
  humanV2 path is untouched when absent.

## Error handling
- Missing `fascia.geo` → `_check_unzipped`-style clear message.
- Empty `tissue_inner_points`/`rigid` group → warn + skip glue/anchor (sim still runs).
- KDTree pairs whose rest distance ≫ `tissuetomuscledistance` → clamp restlen to the
  measured rest distance (don't yank distant points); log count.
- Degenerate fascia tets (det≈0) → `valid` mask, skip (mirrors muscle path).

## Testing / validation
- Unit: `solve_glue_kernel` on a 2-particle pair converges to restlen; mass-weighted
  split correct (heavier particle moves less).
- Unit: scaled-B ARAP shrinks a single free tet to `s×` rest volume.
- Unit: `geo_io` decodes fascia.geo — vert/tet counts (15348/66785), one boolRLE
  group cardinality, `rigid` ordered list length.
- Integration: run humanV2 frames with `fascia_rest_scale=1.0` (off) vs `0.95` (on);
  export USD; confirm muscle surface cross-section tightens (measure mean radial
  distance of muscle surface verts to limb axis: on < off). Per project default,
  output USD via bake, not mp4.

## Out of scope (YAGNI)
- Self-collision / contact between fascia and muscle surfaces (glue + contraction
  give the squeeze; contact deferred).
- Outer fat/skin layer (`tissue`/`tissue_solid`) — fascia = core layer only.
- Anisotropic `materialW` fiber stiffness on fascia (isotropic shrink first).
- Separate bone mesh asset (ride driven muscle instead).
- Time-varying / per-region shrink ramp (constant global `s` first).
