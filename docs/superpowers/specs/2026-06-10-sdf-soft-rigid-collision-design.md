# SDF soft-vs-rigid collision demo

Date: 2026-06-10
Branch: `coll`

## Goal

Simplest classic PBD soft-body-vs-rigid-body collision, as a base for muscle–bone
coupling (muscle = soft tet body, bone = rigid). Replaces the heavy triangle–triangle
UDCF path (`udcf_warp.py`, 820 lines) with a particle-vs-SDF push for the rigid case.

The existing UDCF files stay untouched — this is a new, parallel module.

## Why SDF, not triangle–triangle

Bone is rigid. Classic PBD soft-vs-rigid = each soft-body vertex queries the rigid
body's signed distance field; if it penetrates, push it out along the SDF gradient.
This is what Newton's `eval_particle_body_contact` and Bender's particle-rigid
contact do. No broadphase, no candidate pairs, no triangle winding, no self-collision
filter — interior verts have `phi > 0` and are automatic no-ops, so no surface
extraction either.

## Architecture — 2 new files + 1 config

### `src/muscle/collision/sdf_warp.py` (~55 lines)

Analytic SDFs as `@wp.func`:
- `sdf_plane(p, y0) -> (phi, n)` : `phi = p.y - y0`, `n = (0,1,0)`
- `sdf_sphere(p, c, r) -> (phi, n)` : `phi = |p-c| - r`, `n = normalize(p-c)`

One kernel `solve_sdf_contact_kernel`:
```
inputs: pos, pprev, inv_mass,
        shape_type (0=plane,1=sphere), plane_y, sphere_c, sphere_r,
        contact_offset,
        is_collide (out), contact_normal (out)
per vertex i:
  if inv_mass[i] == 0: is_collide[i]=0; return
  (phi, n) = sdf(pos[i], shape)
  if phi < contact_offset:
    dP = n * (contact_offset - phi)      # push out to the surface
    pos[i]   += dP
    pprev[i] += dP                       # push BOTH → velocity-invariant
    is_collide[i] = 1
    contact_normal[i] = n
  else:
    is_collide[i] = 0
```

Position-level penetration resolution. One write per vertex (each vertex vs a static
rigid is independent — no atomics, no Jacobi averaging). The kernel does NOT touch
velocity or friction.

### Velocity / friction — reuse existing machinery

`MuscleSim.update_velocities()` already branches to
`update_velocities_with_contact_kernel(pos, pprev, vel, dt, is_collide,
contact_normal, restitution_n, restitution_t)` when `sim.is_collide` /
`sim.contact_normal` are set. That kernel is the standard PBD velocity solve:
- approaching (`vn < 0`): `vn_new = -restitution_n * vn`  (normal restitution)
- tangential: `v_t_new = restitution_t * v_t`             (`rt<1` ⇒ friction)

So the SDF kernel only fills `is_collide` + `contact_normal`; restitution and
friction come for free from the existing, tested velocity kernel. Defaults:
`restitution_n = 0.0` (muscle doesn't bounce off bone), `restitution_t = 0.8`
(mild sliding friction).

### `src/muscle/collision/example_sdf_drop.py` (~130 lines)

- Reuse `_load_u_mesh`, `_build_plane_mesh_xz` from `example_u_collision`.
- `SdfCollisionDemo`: owns shape params + `is_collide` / `contact_normal` buffers.
  `collision_solver()` launches the one kernel each substep.
- Wire `sim.constraint_solvers = [demo.collision_solver]`, plus
  `sim.is_collide / sim.contact_normal / sim.restitution_n / sim.restitution_t`
  (same hook the UDCF demo uses).
- CLI `--shape plane|sphere`, `--config data/collisionTest/sdf_drop.json`,
  optional WarpRenderer USD output.

### `data/collisionTest/sdf_drop.json`

Minimal config: reuse U.geo, volume + tetarap constraints, a `plane` block and a
`sphere` block (center, radius), `contact_offset`, restitution params.

## Data flow (per substep, inside `MuscleSim.step`)

gravity + predict → constraint_solvers `[ARAP, volume, sdf_contact]` →
`update_velocities` (with-contact branch). The SDF push moves `pos` and `pprev`
equally, so the push itself injects no velocity; the velocity kernel then applies
normal restitution + tangential friction using `is_collide` / `contact_normal`.

## Testing — `tests/test_sdf_drop.py` (headless, no renderer)

Run a short sim for each shape, then assert:
1. **No explosion**: `max|vel|` finite and below a bound; no NaN in `pos`.
2. **Penetration resolved**: after settling, `min phi >= -tol`.
3. **Settled**: total kinetic energy at the end below a small threshold.
4. **Did not sink through**: plane → `min y >= -tol`; sphere → `min |p-c| >= r-tol`.

CPU device, ~120 steps, small mesh (U.geo) → runs in a few seconds.

## Out of scope (YAGNI)

- Baked grid SDF for arbitrary bone meshes (later; same kernel, swap the `@wp.func`).
- Capsule / box SDFs (trivial to add later as more shape types).
- Two-way coupling (bone reacts to muscle) — bone is static here.
- Self-collision — that's what the UDCF path is for; kept separate.
