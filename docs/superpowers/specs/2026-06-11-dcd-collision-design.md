# DCD (UDCF) Muscle-vs-Bone Collision — Design

**Date:** 2026-06-11
**Status:** approved design, pre-implementation
**Goal:** replace the SDF-based muscle-vs-bone collision with a DCD pipeline that
faithfully replicates Gaia_UDCF's triangle-triangle collision (detection + response),
integrated into OneWaySim's substep iteration loop. CCD is out of scope but gets a
call-site interface. Numerical parity with the C++ reference is a hard requirement
so divergence can be bisected.

## Background

- Reference: `~/Dev/Gaia_UDCF` (CPU PBD fork of GAIA; run via `PBDDynamics ... --CPU`).
  Active code path: `PBDPhysics::runStepCPU()` (PBDPhysics.cpp:622-719).
- A prior port (`src/muscle/collision/udcf_warp.py` + orphaned
  `example_u_collision.py`) deviated algorithmically (dominant-triangle + bary²
  weighting instead of smaller-of-C1/C2 + equal push) and ran collision once per
  substep *after* all constraints instead of interleaved per iteration. Revived demo
  shows floor tunneling (min_y → −0.20 m) and diverging self-penetration
  (max_bary → +0.204).
- Current production SDF path (`sdf_warp.py` / `AnimatedBoneMeshCollider`): per-vertex
  `mesh_query_point_sign_normal` pushout, single pass via the `collision_solvers`
  hook at the end of `OneWaySim.substep()` (solver.py:1233).
- Perf envelope (RTX 4090, humanV2, NV=157369 NT=468162): sim ≈ 1.6 s/frame; SDF
  collision adds ~24 ms (~1%). Memory is not a constraint (452 MB whole process).
  Large headroom for a heavier DCD pipeline.

## Reference pipeline (what we replicate)

### Loop structure (PBDPhysics.cpp:622-719)

```
for substep in 0..numSubsteps:
    if substep % collisionDetectionSubSteps == 0: broadPhase()
    for iIter in 0..iterations:
        if iIter == 0: applyInitialGuesses()
        projectConstraintsCPU()          # material (spring + volume)
        BoundarySolveCPU(); InversionSolveCPU()
        if substep % collisionDetectionSubSteps == 0: broadPhase()
        narrowPhase()                    # UDCF detect + respond
        applyConstraints()               # re-pin fixed verts
    updateVelocities()
```

Key structural fact: collision runs **inside the iteration loop**, interleaved with
material constraints. The prior port's once-at-the-end placement is the root cause of
its instability.

### narrowPhase (PBDPhysics.cpp:1264-1762)

Per candidate triangle pair (T0, T1) from broadphase:

1. Plane for each triangle: `N = normalize(cross(e1, e2))`, `d = −N·q0`.
2. **Möller interval reject** (`gen_t`, :1361): signed distances of T0's verts to
   T1's plane (threshold 1e-5); all same sign → no intersection. Otherwise compute
   parametric intervals [t1,t2], [t3,t4] of both triangles on the intersection line
   `D = normalize(N1×N2)`; non-overlapping intervals (`!(t2≥t3 && t4≥t1)`) → reject.
3. **Candidate contact points**, three kinds per direction (:1594-1674):
   - vertex-face: T0 vertex projected onto T1's plane lies inside T1
     → depth = `−(N2·v + d2)`;
   - Q-points: T1 vertex shot along N2 onto T0's plane, lies inside T0 (same depth
     formula evaluated at the intersection point);
   - edge-edge: projected T0 edge crosses T1 edge in T1's plane
     (`line_intersection_on_same_plane`) → depth at the crossing point on the
     original edge.
   Symmetric set for T1 against T0's plane.
4. `C1 = max` candidate depth of T0 below T1's plane (record barycentric coords of
   the contact point on **both** triangles: bary1 on T0, bary2 on T1); `C2`
   symmetric.
5. Accept only if **both** `C1 > 1e-6` and `C2 > 1e-6` (one-sided positives are
   grazing false positives). Choose the **smaller**: `C1 < C2` → depth `C1`,
   gradient `n = N2` (the *opposing* plane normal); else `C2`, `n = N1`.

### Response (applyNarrowCollisionConstraintsCPU, PBDPhysics.cpp:2894-3116)

1. `C *= collisionConstraintMultiplier` (1.01 — 1% over-push, anti-sticking).
2. `isSolid` per side = any of the 3 face verts fixed. Both sides movable →
   `C *= 0.5` (split); one side solid → movable side takes full C.
3. **Normal pushout: all 3 verts of each movable face moved equally** by `±C·n`.
   No mass weighting, no barycentric weighting (the XPBD-weighted variant is
   commented out in the reference).
4. **Friction**: per-vert velocity `x − xPrev`; contact-point velocity on each face
   via barycentric interpolation; relative tangential velocity
   `diff = tangential(v_contact1 − v_contact0)`; if `|diff| > C·μ_static` → slipping,
   clamp to `C·μ_dynamic`; else remove fully (static stick). Correction split 50/50
   between the faces, each face's 3 verts moved equally. Reference applies friction
   even to fixed verts and relies on `applyConstraints()` to re-pin them.
5. Defaults: `collisionConstraintMultiplier = 1.01`, `μ_static = 0.5`,
   `μ_dynamic = 0.4`, `collisionDetectionSubSteps = 1`, `iterations = 3`.

### Reference bugs — do NOT copy

| Where | Bug | Our behavior |
|---|---|---|
| PBDTetMeshMassSpring.cpp:541 | `vertex(i)` uses loop index instead of `vId` | n/a (we don't port InversionSolve) |
| PBDTetMeshMassSpring.cpp:206 | `gradVol` is 9×1, `segment<3>(9)` is out of bounds | n/a |
| PBDPhysics.cpp:3055 | unconditional `dynamicNorm = depth·μd` makes the :3045 clamp dead code | implement the :3045 intent: clamp only when slipping |

Material constraints (spring/volume) and InversionSolve are **not ported** — our GMG
ARAP-SVD + XPBD volume already fill those roles.

## Mapping to OneWaySim (muscle-vs-bone, one-way)

| Reference | Ours |
|---|---|
| two deformable tet meshes | muscle surface tris (movable) vs bone tris (fixed, animated per frame) |
| isSolid side | bone is always solid → muscle face takes full C, bone never moves |
| gradient = opposing normal | C1 chosen (muscle into bone's plane) → push muscle along N_bone; C2 chosen → push muscle along −N_muscle... see note below |
| `x − xPrev` for friction | quasi-static GMG: use the substep-start snapshot (`pos_mid`) as prevPos |
| serial Gauss-Seidel contact apply | GPU Jacobi: accumulate `dP`, `dP_count` per vert, apply `dP/count` after the kernel |
| per-iteration narrowPhase | hook inside OneWaySim's fine-iteration loop |

Direction note: in the reference, whichever side is chosen, *both* faces get equal
and opposite pushes (modulo isSolid). With pair order (T0 = muscle, T1 = bone):
when C1 wins (muscle below bone's plane), the muscle face moves `+C·N_bone`; when
C2 wins (bone face below muscle's plane), the reference pushes the bone face
`+C·N_muscle` and the muscle face `−C·N_muscle` — with bone solid, the muscle face
takes the full `−C·N_muscle`. Friction: bone contact-point velocity comes from the
bone animation (prev vs current bone frame positions), muscle side from `x − pos_mid`.

GS→Jacobi divergence is a known, accepted deviation: golden comparison is anchored at
single-contact level (independent contacts), multi-contact accumulation may differ.

## Module design

New file `src/muscle/collision/udcf_dcd.py`. `udcf_warp.py` stays untouched as a
deprecated reference. No dependency on the deleted `muscle.{config,geo,log,vis_warp}`
modules.

```
class UdcfDcdCollider:
    def __init__(self, sim, bone_source, params: UdcfParams)
        # muscle surface tris from sim (build_surface_tris), bone tris from
        # bone_source (AnimatedBoneMeshCollider-style frame sequence or static mesh)
        # buffers: pair list (fixed cap + overflow counter), contact records,
        # dP / dP_count accumulators, hashgrid
    def set_frame(self, frame)        # update bone verts (prev + cur for friction)
    def broadphase(self)              # centroid hashgrid + max-radius query,
                                      # emit muscle-tri × bone-tri candidate pairs
    def narrow(self)                  # narrow_kernel: Möller reject → 3 candidate
                                      # kinds → C1/C2 → smaller positive → contact
                                      # record {C, n, bary_m, bary_b, face ids}
    def respond(self)                 # respond_kernel: equal push to 3 muscle verts
                                      # (full C, bone solid), friction, Jacobi apply
    def solve(self, i_iter)           # i_iter==0: broadphase; always: narrow+respond
    ccd: CcdInterface | None = None   # interface only; solve() calls
                                      # self.ccd.solve() if set

@dataclass
class UdcfParams:
    multiplier: float = 1.01
    friction_static: float = 0.5
    friction_dynamic: float = 0.4
    offset: float = 1e-6
    max_pairs: int = 1 << 20          # cap, overflow → warn once per frame
    broadphase_every_substep: bool = True   # maps collisionDetectionSubSteps=1

class CcdInterface(Protocol):
    def solve(self, ...) -> None
```

`narrow` and `respond` are **separate kernels** (mirrors the reference split; lets
the golden tests validate each independently). Contact records live in a fixed-size
buffer with an atomic counter; overflow clamps and logs.

Degenerate guards: triangle normal `‖cross‖ < eps` → skip pair; `‖N1×N2‖ < 1e-6`
(parallel planes) → the reference's gen_t denominators go bad → skip pair.

## Solver integration

`OneWaySim.substep()` gains a per-iteration hook alongside the existing end-of-substep
one:

- existing: `collision_solvers` — runs once after the constraint loop (SDF path,
  unchanged, stays the default).
- new: `collision_solvers_iter` — list of callables `fn(i_iter)` invoked inside the
  fine-iteration loop, after material/volume/glue each iteration.

Config switch in the case JSON selects SDF vs DCD; both paths coexist. The DCD
collider registers `solve` on `collision_solvers_iter`.

## Numerical comparison harness (parity with the C++ reference)

Trajectory-level comparison is impossible (different material solvers). Collision is
a pure function of its inputs, so parity is established function-level:

1. **Golden data generation**: patch Gaia_UDCF (`narrowPhase` +
   `applyNarrowCollisionConstraintsCPU`) to dump per-contact records when
   `UDCF_DUMP=<path>` is set: inputs (both triangles' 6×3 vert positions, prevPos,
   isFixed flags, params) and outputs (C1, C2, chosen C, gradient n, bary1, bary2,
   post-response positions of all 6 verts). Rebuild with `build.sh`, run the
   `S02_SquishyBallDropping/4LowResSquishyBallColliding_SeeNumCollisions` scene,
   capture a few thousand records → `tests/golden/udcf_pairs.npz` (committed; keep
   under a few MB by capping record count).
2. **Replay tests** (`tests/test_udcf_dcd.py`):
   - feed golden inputs as a batch to `narrow_kernel`; compare C, gradient, barys
     (float32, atol ≈ 1e-5; relative for C);
   - feed golden contacts to `respond_kernel` (single contact each, isFixed flags
     honored, GS-vs-Jacobi irrelevant at single-contact level); compare position
     deltas of all 6 verts.
   - rejected pairs are part of the golden set too (records with C1/C2 ≤ offset):
     assert our kernel also rejects them.
3. **Bisection story**: golden passes but scene misbehaves → integration/iteration
   structure; golden fails → detection or response math, and the failing record
   pinpoints the triangle pair.

Patch to Gaia_UDCF lives in our repo as `tools/udcf_golden/gaia_dump.patch` plus a
generation script, so the golden file is reproducible.

## Testing

1. Golden replay tests (above) — the parity gate.
2. **U-demo regression**: rebuild the self-collision U demo on the new collider
   (self-collision = muscle-vs-muscle is a degenerate config where both sides are
   movable; supported by the same kernels with C·0.5 split). Pass criterion:
   `max_bary` penetration metric converges instead of diverging (old port:
   +0.057 → +0.204 monotonic).
3. **humanV2 smoke**: muscle-vs-bone on the production case; penetration-depth
   statistics vs the SDF baseline; no NaNs, no exploding verts over 30 frames.
4. **Perf**: ms/frame vs SDF baseline (+24 ms). Budget: DCD may cost several × SDF;
   acceptable while total stays within ~10% of the 1.6 s/frame sim cost.

## Out of scope

- CCD (interface reserved only).
- Two-way muscle-bone coupling, fascia interaction.
- Porting GAIA material/inversion solvers.
- GPU GS coloring (Jacobi accepted; revisit only if U-demo regression fails to
  converge).
