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
2. **Push target** — the nearest point on the containing muscle `mb`'s boundary
   surface, found by a **bounded scan over `mb`'s own boundary faces** (no
   per-muscle `wp.Mesh`, no refit). `muscle_id` is sorted ascending, so each
   muscle's tets — and hence its boundary faces — form a contiguous block
   `[start, start+count)`; the scan walks only that range, AABB-culling each face
   against `x ± surf_dist` and point-tri testing the survivors. The push moves
   `x` to the nearest surface point plus a small outward `offset`.

Because the inside test no longer lies and the push can see deep penetration,
**iterating now converges** instead of diverging.

## Pipeline

Build (`__init__`):
- per-tet `muscle_id` → `vert_muscle` (per-vertex), `tet_muscle` (per-tet tag).
- global tet arrays + AABB buffers for the inside-test BVH.
- compact **per-muscle boundary-face list** (`_bface` + `_bf_start/_bf_count`):
  a tet face is a boundary face iff it occurs once within its muscle's tets;
  `muscle_id` is sorted, so these come out grouped by muscle into a contiguous
  range. ~13 % of tet faces are boundary. No `wp.Mesh`, no per-frame refit.

Per frame (`refresh`): per-muscle AABB + adjacency → `active`/pairs report (one
host sync). No longer gates any refit (there is none).

Per resolve pass (`__call__`, post_phase, self before SDF-bone, `stride`-gated,
last substep always runs):
1. rebuild the tet BVH + per-tet AABBs on current positions (once per call,
   reused across `iters`).
2. ×`iters`: `_push_kernel` (one thread per muscle vertex: `_inside_target`
   finds the deepest non-self muscle it is inside; `_closest_surface` scans that
   muscle's boundary-face range, AABB-culled, for the nearest surface point; push
   to it + `offset`, clamped to `push_cap`).

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
| `max_dist` | resolver scale; `surf_dist` defaults to `max(0.05, 5·max_dist)` | 0.02 |
| `offset` | outward margin past the surface (`contact_offset`) | 1e-4 |
| `iters` | push passes per resolve (more = lower count, slower) | 4 |
| `stride` | run every Nth substep (last always runs) | 1 |
| `push_cap` | per-pass displacement clamp (0 = unlimited) | 0.0 |
| `surf_dist` | boundary-face AABB-cull radius (only spans penetration depth) | — |
| `meter_max_dist` | surface radius for the depth meter only | = `surf_dist` |

Baseline config: `data/humanV2/config/humanV2_intermuscle.json`
(`iters=4`, `stride=1`, `push_cap=0.003`).

## Speed — surface search without a mesh

The push target needs a nearest-point-on-surface query. The first version gave
each muscle its own `wp.Mesh` (a per-muscle BVH, distance-pruned `O(log)`
queries) but paid ~199 host-side `refit()` launches per substep. Removing the
meshes — driving the surface search off the **one** tet structure instead, keyed
by `muscle_id` — went through three forms (humanV2 50f, no export, RTX 4090):

| surface-search structure | sim time | count | depth |
|--------------------------|---------:|------:|------:|
| per-muscle `wp.Mesh` (+199 refit/substep) | 3.3 s | 397 | 2.4 mm |
| global tet-BVH expanding `bvh_query_aabb(x±r)` | 76 s | 290 | — |
| scan target muscle's whole tet range, AABB-culled | 13.7 s | 302 | 1.1 mm |
| **scan target muscle's BOUNDARY FACES, AABB-culled** | **5.7 s** | **303** | **1.1 mm** |

Lessons:

- **`bvh_query_aabb` has no distance pruning** — a box query visits *every* tet
  in the box, so reusing the global tet BVH for the surface search is 23×
  *slower* than the per-muscle mesh (which prunes by distance). Dead end.
- **Bounding the scan to one muscle** (contiguous tet/face range, possible
  because `muscle_id` is sorted and each muscle's verts/tets are one block) is
  what makes a mesh-free linear scan viable — 76 s → 13.7 s.
- **Scanning only the boundary faces** (~13 % of tet faces; interior tets never
  yield a surface point) with a per-face AABB cull is another 2.4× — 13.7 →
  5.7 s, at *better* quality than the mesh version (count 397 → 303, depth
  2.4 → 1.1 mm, because the scan point-projects on the live geometry with no
  refit-staleness and no sign-flip).

Net: ~1.7× the mesh version's wall-clock but no `wp.Mesh`/refit and better
penetration numbers. The linear scan is `O(boundary faces of mb)`; a per-muscle
boundary-face BVH would restore `O(log)` but reintroduce per-muscle structures
and per-pass rebuilds, not worth it. The remaining cost is now the global tet
BVH rebuild + the `_inside_target` query over all 157 k muscle verts, not the
surface scan.
