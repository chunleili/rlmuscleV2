# humanV2 Pipeline

End-to-end walkthrough of the `example.humanV2` demo, from the CLI entry to the
USD export, with the exact function jumps. The demo is a single json-driven
entry point (`--config` is the only argument); the baseline config
(`data/humanV2/config/humanV2.json`) is **fascia shell + animated-bone SDF
collision, multigrid OFF**.

> **⚠️ Note:** the physics solver moved from `coupleOneWayGmg/solver.py`
> (`OneWaySim`) to **`src/muscle/mg/mg.py` (`MultiGridSim`)**; the `solver.py:NNNN`
> line jumps below are stale. The flow is otherwise unchanged. For the multigrid
> details in particular see [`humanV2_cage_multigrid.md`](humanV2_cage_multigrid.md)
> (the current `mg_mode: off|projection|hpbd` scheme).

Two source files carry the whole flow:

- `src/example/humanV2.py` — the demo driver (build, collider, render loop).
- `src/muscle/mg/mg.py` — `MultiGridSim`, the ARAP-HPBD bone-driven physics
  (was `coupleOneWayGmg/solver.py` `OneWaySim`).

## Data flow (one glance)

```
restpos.geo ─┐
fascia.geo ──┴→ OneWaySim(merge) → pos[NV]
cons_<f>.geo → target_pos → pin
                ↓ per frame
  pin → [ARAP + volume? + fiber?] x fine_iters → glue → self? → SDF bone push
                ↓
  pos → inside_depth (meter) → USD (muscle / fascia / bones + primvars)
bone_<f>.geo → wp.Mesh (collider, winding flipped) ┘
```

`NV` = merged vertex count (muscle `< nvm`, fascia `>= nvm`). `NT` = merged tet
count (muscle tets first, fascia tets appended).

## 0. Entry — `main()` (`humanV2.py:640`)

1. Parse the single `--config` argument.
2. `wp.init()` → `load_config()` (`solver.py:1614`) — loads JSON, resolves
   relative asset paths against cwd.
3. `demo = cfg["demo"]` — run-level settings (device, out path, collision,
   self_collision).
4. `build_humanV2_sim(cfg, demo)` → build the bone collider → build the optional
   self-collider → assemble `collision_solvers` → `run_humanV2_loop()`.

## 1. Build the sim — `build_humanV2_sim()` (`humanV2.py:325`)

Wraps `OneWaySim(cfg)` (`solver.py:980`). The constructor:

**a. Read the rest mesh** (`solver.py:1048`) — `GeoFile(rest_path)` gives
`pos / pos_rest / pos_mid / tet_idx`. `NV` = muscle verts, `NT` = muscle tets.

**b. Precompute rest quantities** → `precompute_rest_kernel` (`solver.py:23`).
Per tet: `B = inv(Dm)`, `rest_volume V0`, `alpha = 1/(mu·V0)` (XPBD compliance).

**c. Mass** (`solver.py:1069`) — `density = total_mass / ΣV0`; per-vertex mass =
Σ over incident tets of `density·V0/4` → `inv_mass`.

**d. Static attribute cache** (`solver.py:1094`) — `materialW_np` (fiber
direction), `tendonmask_np`, `cd_np`, `muscle_id_np` (per-tet) +
`muscle_id_strings`. Used later for USD baking and named-muscle activation.

**e. Optional fiber** (`solver.py:1110`; OFF in baseline) — the two passive
Houdini models `fibernorm` / `aniso_arap`. Builds `fiber_alpha = 1/(k_fib·V0)`,
`fiber_tendonmask` (per-tet mean), `fiber_activation` (default 0). `_fiber_nt =
muscle NT` is captured *before* any fascia merge so the fiber kernel always runs
over the first `_fiber_nt` (muscle) tets. The active `fibermillard` model is
gated off — it needs per-tet activation the asset does not carry.

**f. Pin-to-target** (`solver.py:1148`) — read `cons_<start>.geo` → `pts` (pin
vertex indices; humanV2 pins **all** NV), `target_pos`, and
`rest_offset = restpos[pts] - target0`. Topology (`pts`) is constant across all
`cons_<frame>.geo`; only `target_pos` changes per frame. Optional partial
pinning via `tendon_pin_threshold` frees the belly to a soft pull
(`pin_soft_stiffness`) while tendon regions stay hard-pinned.

**g. Optional multigrid** (`solver.py:1182`) → `_init_coarse()` (`solver.py:1348`).
Loads the coarse mesh and brute-force computes both rest-pose barycentric maps
via `compute_barycentric_kernel` (`solver.py:695`): fine→coarse (`cage_idx /
uvw`, for prolongation) and coarse→fine (`cage_idx_c2f / uvw_c2f`, for
restriction). `mg_fine_nv = muscle NV` is captured here (before the fascia merge
mutates `self.NV`) so prolong stays in bounds. OFF in baseline. (Timing note:
both maps together cost ~0.2s on a 4090 — not an init bottleneck; the cold-start
cost is geo JSON parsing, cached to `.npy`.)

**h. Fascia** (`solver.py:1190`) → `_init_fascia()` (`solver.py:1195`):

- Read the fascia geo.
- Glue pairing. Baseline `fascia_glue_mode="surface"` → `_pair_surface()`
  (`solver.py:1316`): a cKDTree pairs **every muscle surface vertex**
  (`point_group("surface_points")`) to its nearest fascia vertex (~119482
  pairs), `restlen = min(0.01, dist)`.
- **Free ring** (`solver.py:1257`): the glued muscle verts plus a dilated 1-ring
  are freed from the pin (`pin_active = 0`) so the glue can pull them inward;
  those tets are softened (`surface_mu = 1e5` instead of `mu`).
- `build_merged_state()` (`solver.py:869`): muscle + fascia become **one**
  particle system (fascia verts appended after `nvm`, fascia tet indices
  `+= nvm`). `precompute_rest_scaled_kernel` (`solver.py:56`) is per-tet: fascia
  uses `fascia_rest_scale = 0.95` to shrink its rest shape (so it contracts and
  squeezes the muscle inward), with fascia stiffness = `solidshapestiffness`.
- After the merge, `self.pos / tet_idx / B / ... / NV / NT` all point at the
  merged state.

Back in `build_humanV2_sim`:

- `sim.preload_targets(frames)` (`solver.py:1441`): parse every frame's
  `cons_*.geo` once into RAM (`.npy`-backed) so the step loop is pure GPU work.
- Split surface triangles: muscle tets vs fascia tets each go through
  `vectorized_surface_tris()`, rendered as separate USD prims.

## 2. Bone collider — `AnimatedBoneMeshCollider` (`humanV2.py:219`)

- Read `bone_1.geo`; **flip triangle winding** `tri[:, [0, 2, 1]]`
  (`humanV2.py:241`). SNU bones are wound inward relative to Warp's signed
  query; without the flip `mesh_query_point_sign_normal` reports ~155640 false
  "inside" verts at rest vs the true ~1729.
- Build a `wp.Mesh` once (topology constant; per frame only `points` is
  refreshed + `refit()`).
- `coll_inv_mass`: fascia verts (`>= nvm`) set to 0 → excluded from contact
  resolution (fascia never touches bone; any "contact" it reports is a sign-flip
  spike).
- `meter_mask` (`humanV2.py:267`): muscle = 1, fascia = 0, used by the
  **measurement** kernels (independent of `inv_mass`, so pinned muscle verts are
  still measured).
- `preload()`: all per-frame bone poses into RAM.

The collider is **always** built — it is the penetration **meter**
(`inside_depth`) even when contact resolution is disabled.

## 3. Assemble the solver chain (`humanV2.py:660`)

- `_make_self_collider()` (`humanV2.py:544`) reads `demo.self_collision`.
  Baseline `method = "none"` → all None. `dcd` (UDCF triangle-triangle, optional
  CCD) and `intermuscle` (per-muscle closest-point) return a push function +
  optional `mm_meter`.
- `post = [self_collider?, collider?]`: the self push runs **first**, the SDF
  bone runs **last** (final word). `sim.collision_solvers = post`.

## 4. Main loop — `run_humanV2_loop()` (`humanV2.py:400`)

Per frame:

1. `collider.set_frame(frame)` (`humanV2.py:282`): swap bone `points` +
   `mesh.refit()`.
2. `pre_step(frame)`: refresh self-collider (none in baseline).
3. Activation ramp (only if a muscle is named via `activate`).
4. **`sim.step_frame(frame)`** — the physics (section 5).
5. `collider.inside_depth()` (`humanV2.py:312`) → `_mark_inside_kernel`
   (`humanV2.py:70`): per muscle-vertex bone-penetration depth.
6. `is_collide = depth > collide_eps` (muscle-bone); `mm_meter()` if present
   (muscle-muscle self penetration, measured **separately**).
7. USD export: `render_mesh` for muscle / fascia / bones; write the
   `is_collide_muscle_bone` / `pen_depth_*` / `sum_*` primvars; on the first
   frame, `_bake_static_primvars()` (`humanV2.py:137`) writes
   materialW / tendonmask / muscle_id as default-timecode primvars.

## 5. One physics frame — `OneWaySim.step_frame()` (`solver.py:1602`)

```
load_frame_targets(frame)      # target_pos.assign(cached)
apply_pintotarget()            # one hard pin at frame start
apply_anchor(merged)           # fascia rigid anchors (if any)
for _ in range(nsubsteps):     # 10
    substep()
```

`substep()` (`solver.py:1596`) runs the three-phase `program`
(`build_program`, `solver.py:1491`):

**pre_phase**
- Multigrid (if ON): the V-cycle depends on `mg_mode` — `projection` restricts
  the bone target onto the cage and prolongs (one-shot smoother); **`hpbd`**
  Galerkin state-restricts the fine state, relaxes coarse ARAP, and
  delta-prolongs the correction. See
  [`humanV2_cage_multigrid.md`](humanV2_cage_multigrid.md).
- Else partial-pin: `apply_pintotarget`.
- `apply_anchor`; zero `L` / `vol_L` / `fiber_L`.

**iter_phase** (`solver.py:1538`, run `fine_iters` times — 2 in baseline)
- `wp.copy(pos_mid, pos)` (snapshot).
- `arap_svd_kernel` (`solver.py:101`): ARAP-SVD shape match, parallel-Jacobi
  atomic projection.
- Optional `solve_volume_kernel` (`solver.py:193`): hydrostatic `det(F) = 1`.
- Optional fiber kernel (`fibernorm` / `aniso_arap`) over the first `_fiber_nt`
  tets.
- `collision_solvers_iter` (UDCF DCD per-iteration hook; none in baseline).

**post_phase** (`solver.py:1584`)
- `solve_glue()` (`solver.py:945`): fascia↔muscle XPBD distance constraint ×
  `fascia_glue_iters` (2) → `solve_glue_kernel` (`solver.py:648`).
- `collision_solvers` (post): self push first → bone collider `__call__`
  (`humanV2.py:296`) last → `solve_mesh_contact_pos_kernel` (sign-normal SDF
  query, push every inside vertex out to the surface; `max_push <= 0` = push
  regardless of depth).

## Key knobs (json)

| key | effect | baseline |
|-----|--------|----------|
| `use_fascia` | merge the fascia squeeze shell | true |
| `use_multigrid` / `mg_mode` | coarse-mesh V-cycle in pre_phase (`off`/`projection`/`hpbd`) | false |
| `use_fiber` / `fiber_model` | passive along-fiber stiffness | off |
| `fascia_rest_scale` | fascia rest shrink (squeeze strength) | 0.95 |
| `nsubsteps` / `fine_iterations` | substeps per frame / ARAP iters per substep | 10 / 2 |
| `demo.collision.enabled` | SDF bone push (meter still runs) | true |
| `demo.collision.max_push` | reject contacts deeper than this (`<= 0` = off) | -1.0 |
| `demo.self_collision.method` | `none` / `dcd` / `intermuscle` | none |

## Ablation configs

Each differs from the baseline by exactly one knob (and its own `out` path):

- `humanV2.json` — baseline (fascia + SDF + no MG).
- `humanV2_gmg.json` — `use_multigrid: true`.
- `humanV2_nofascia.json` — `use_fascia: false`.
- `humanV2_nocoll.json` — `demo.collision.enabled: false`.
