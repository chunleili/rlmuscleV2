# Inter-muscle collision (closest_point)

`src/muscle/collision/closest_point/intermuscle.py` — resolve muscle-A-vs-muscle-B
penetration only (never muscle-vs-itself), as an optional self-collision mode of
the humanV2 demo (`demo.self_collision.method = "intermuscle"`).

## Why not the pseudonormal sign

The first implementation decided inside/outside with
`wp.mesh_query_point_sign_normal` — the angle-weighted pseudonormal sign of the
nearest point on the *other* muscle's surface. On the densely-packed, folded RL
poses this **flips sign** (thin / concave / near-non-manifold muscle surfaces),
reporting verts that are OUTSIDE as inside. Consequences measured on humanV2
(50 frames):

- The depth meter saturated at the search radius (50 mm, then 200 mm when the
  radius was raised) — i.e. it reported 20 cm "penetration" that does not exist.
- Raising `max_dist` or iterating the push made the count **rise**, not fall
  (`md=0.02 → 1371`, `md=0.10 → 19625`): a larger radius surfaces more
  false-positive signs, and pushing on them manufactures real churn.
- The count could not be driven down — minimal pushing was the least-bad setting.

The push logic was fine; the **inside decision it relied on was lying**.

## The fix — tet-barycentric inside, geometric closest-surface push

The muscles are tet meshes, so inside/outside is exact volumetric containment, no
sign to flip: `x` is inside muscle B iff it falls inside one of B's tets
(barycentric coords all ≥ 0, sum ≤ 1). This holds for thin / concave / even
slightly-inverted tets.

The key is to **decouple** the two questions:

1. **Inside decision** — a global `wp.Bvh` over every muscle tet's AABB (tagged
   by muscle id) is both the broadphase and the test. `bvh_query_aabb(x, x)`
   returns the candidate tets containing `x`; a barycentric test
   (`_tet_inside_score`) on the non-self ones decides containment. Exact and
   **unbounded** — a vertex 5 cm deep is still detected.
2. **Push target** — the nearest point on the containing muscle's boundary
   surface (its own per-muscle `wp.Mesh`), found with a plain closest-point
   query. Only the *geometric* closest point (face + bary) is used, never the
   query's sign, so the surface search radius (`surf_dist`) can be large enough
   to reach a deep vertex **without** re-introducing any sign-flip. The push
   moves `x` to that surface point plus a small outward `offset`.

Because the inside test no longer lies and the push can see deep penetration,
**iterating now converges** instead of diverging.

## Pipeline

Build (`__init__`):
- per-tet `muscle_id` → `vert_muscle` (per-vertex), `tet_muscle` (per-tet tag).
- global tet arrays + AABB buffers for the inside-test BVH.
- per-muscle surface `wp.Mesh` (points **shared** with `sim.pos`) for the push
  target. Winding no longer matters (only the geometric closest point is used).

Per frame (`refresh`): per-muscle AABB + adjacency → `active` mask (one host
sync), used only to gate which surface meshes get refit.

Per resolve pass (`__call__`, post_phase, self before SDF-bone, `stride`-gated,
last substep always runs):
1. refit the active surface meshes.
2. ×`iters`: rebuild the tet BVH on current positions → `_push_kernel`
   (one thread per muscle vertex: `_inside_target` finds the deepest non-self
   muscle it is inside; push to that muscle's nearest surface + `offset`,
   clamped to `push_cap`).

Meters: `inside_count` / `inside_depth` use the same barycentric gate, so the
reported count and depth are honest (no false positives). `inside_depth` is the
muscle-muscle penetration primvar (`pen_depth_muscle_muscle`) in the USD export.

## Results (humanV2, 50 frames, RTX 4090)

Honest barycentric meter, final inter-muscle inside-vertex count:

| settings | inside_count | max depth | sim time |
|----------|-------------:|----------:|---------:|
| ~1 push/frame (`stride=100`) | 1413 | 2.5 mm | 1.8 s |
| `stride=2`, `iters=1` | 1456 | 2.9 mm | 3.2 s |
| `stride=1`, `iters=4` | 445 | 1.0 mm | 6.0 s |
| `stride=1`, `iters=4`, `push_cap=0.003` | 419 | 0.9 mm | 6.0 s |
| `stride=1`, `iters=8`, `push_cap=0.002` | **281** | 0.8 mm | 8.0 s |

More iterations now reduce the count monotonically (the opposite of the
pseudonormal version). The residual (~280–450) is shallow (< 1 mm) penetration
that the per-substep hard pin re-creates faster than one resolve pass clears.

## Config (`demo.self_collision`)

| key | meaning | default |
|-----|---------|---------|
| `method` | `intermuscle` | — |
| `max_dist` | resolver scale; `surf_dist` defaults to `max(0.2, 10·max_dist)` | 0.02 |
| `offset` | outward margin past the surface (`contact_offset`) | 1e-4 |
| `iters` | push passes per resolve (more = lower count, slower) | 4 |
| `stride` | run every Nth substep (last always runs) | 1 |
| `push_cap` | per-pass displacement clamp (0 = unlimited) | 0.0 |
| `surf_dist` | explicit surface search radius (overrides the default) | — |
| `meter_max_dist` | surface radius for the depth meter only | = `surf_dist` |

Baseline config: `data/humanV2/config/humanV2_intermuscle.json`.

## Known cost / next (speed)

The tet BVH is rebuilt every push pass (`iters` × every substep). At `iters=4`
that is ~40 rebuilds of a 468k-tet BVH per frame — the dominant cost. Borrowing
the udcf optimizations (build once per substep and reuse across `iters`, gate the
per-frame meter to export-only, trim the active-mesh refit) is the open speed
work.
