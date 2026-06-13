# UDCF DCD Muscle-vs-Bone Collision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace SDF muscle-vs-bone collision with a Warp port of Gaia_UDCF's triangle-triangle DCD (detect + respond), interleaved into OneWaySim's fine-iteration loop, with record-level numerical parity against the C++ reference.

**Architecture:** New self-contained module `src/muscle/collision/udcf_dcd.py` (broadphase emit → narrow kernel → respond kernel → Jacobi apply). Golden data dumped from an instrumented Gaia_UDCF build gates parity via replay tests. `solver.py` gains a per-iteration collision hook; the SDF path stays untouched.

**Tech Stack:** NVIDIA Warp (fp32 kernels), numpy, pytest. Reference: `~/Dev/Gaia_UDCF` C++ (CPU build).

**Spec:** `docs/superpowers/specs/2026-06-11-dcd-collision-design.md`

---

## Context for the implementer

- Reference narrow phase: `~/Dev/Gaia_UDCF/Simulator/Modules/PBD/PBDPhysics.cpp` — `narrowPhase()` lines 1264-1762, `applyNarrowCollisionConstraintsCPU()` lines 2894-3116, loop `runStepCPU()` 622-719. Read these before writing kernels; the kernels below mirror them line-for-line.
- Our solver: `src/muscle/coupleOneWayGmg/solver.py` — `substep()` at lines 1167-1234; the existing end-of-substep hook is `collision_solvers` (line 1233).
- Existing Warp tri-tri port (deviant, do NOT modify, only reuse `centroid_radius_kernel` + `max_radius_reduce_kernel` by import): `src/muscle/collision/udcf_warp.py`.
- Reference quirks we replicate exactly (parity > taste): candidate depths are computed with `offset = 0` and only the final accept test uses `1e-6` (offset is assigned *after* the candidate loop, PBDPhysics.cpp:1698); friction interpolates the *closest* face's velocities with the *cur* face's barycentric coords and vice versa (PBDPhysics.cpp:3032/3035); slipping friction magnitude is always `pen·μd` (the :3045 clamp and the :3055 override are equivalent because μs > μd).
- Reference bugs we do NOT replicate: stale `Q[i][j]` reuse across pairs when `find_intersection_point` returns false (we use validity flags — may cause rare golden mismatches, hence the 99.9% pass criterion); `InversionSolveCPU` vertex-index bug (not ported at all).
- All Python runs from repo root with `PYTHONPATH=src` and `.venv/bin/python` / `.venv/bin/pytest`.
- Deviation from the spec's "config switch in the case JSON": colliders in this repo are wired by example scripts, not by `OneWaySim` config (the SDF path works the same way), so the SDF/DCD switch is the demo's `--mode` flag (Task 9). The solver itself only grows the neutral `collision_solvers_iter` hook.

## File structure

| File | Action | Responsibility |
|---|---|---|
| `tools/udcf_golden/gaia_dump.patch` | Create | Instrumentation diff for Gaia_UDCF (reproducibility) |
| `tools/udcf_golden/make_golden.py` | Create | CSV → `tests/golden/udcf_pairs.npz` converter |
| `tests/golden/udcf_pairs.npz` | Create (generated) | Golden contact records (≤ 20k, few MB) |
| `src/muscle/collision/udcf_dcd.py` | Create | Geometry funcs, narrow/respond/apply/emit kernels, `UdcfParams`, `CcdInterface`, `UdcfDcdCollider` |
| `tests/test_udcf_dcd.py` | Create | Golden replay + synthetic unit tests |
| `src/muscle/coupleOneWayGmg/solver.py` | Modify (~1206-1234) | `collision_solvers_iter` per-iteration hook |
| `src/muscle/collision/example_dcd_u.py` | Create | U-shape self-collision regression demo |
| `src/muscle/collision/example_humanV2_bone_dcd.py` | Create | Production muscle-vs-bone demo + metrics |

Golden CSV record layout (71 comma-separated fields, one line per dumped pair; this exact order is shared by the C++ dump, `make_golden.py`, and the tests):

```
stage,                       # 0=gen_t/interval reject (subsampled 1/97), 1=offset reject, 2=applied C1-branch, 3=applied C2-branch
q0 (9), q1 (9),              # T0 / T1 vert positions at narrow time, pair order
prev0 (9), prev1 (9),        # vertexPrevPos of the same verts
fixed0, fixed1,              # 1 if any vert of T0 / T1 isFixed
C1, C2,                      # max penetration depths (0 when never set)
n (3),                       # chosen gradient (N2 for stage 2, N1 for stage 3; zeros for rejects)
bary_cur (3), bary_closest (3),  # barycentrics exactly as passed to apply (zeros for rejects)
mu_s, mu_d, mult,            # friction_static, friction_dynamic, collisionConstraintMultiplier
post0 (9), post1 (9)         # T0 / T1 vert positions AFTER apply, pair order (pre positions for rejects)
```

Stage→apply-order mapping: stage 2 → cur = T0, closest = T1; stage 3 → cur = T1, closest = T0.

---

### Task 1: Instrument Gaia_UDCF and capture the golden CSV

**Files:**
- Modify: `~/Dev/Gaia_UDCF/Simulator/Modules/PBD/PBDPhysics.cpp` (outside our repo; backed up + diffed)
- Create: `tools/udcf_golden/gaia_dump.patch`

- [ ] **Step 1: Back up the reference file**

```bash
cp ~/Dev/Gaia_UDCF/Simulator/Modules/PBD/PBDPhysics.cpp \
   ~/Dev/Gaia_UDCF/Simulator/Modules/PBD/PBDPhysics.cpp.orig
```

- [ ] **Step 2: Add the dump helper at file scope**

In `PBDPhysics.cpp`, directly after the last `#include` line at the top of the file, insert:

```cpp
// ---- UDCF golden dump (enable with env UDCF_DUMP=<csv path>) ----
#include <cstdio>
#include <cstdlib>
namespace {
    inline FILE* udcfDumpFile() {
        static FILE* f = nullptr;
        static bool tried = false;
        if (!tried) {
            tried = true;
            const char* p = std::getenv("UDCF_DUMP");
            if (p) f = fopen(p, "w");
        }
        return f;
    }
    int udcfDumpCount = 0;
    int udcfRejectSkip = 0;
    const int UDCF_DUMP_MAX = 200000;
    inline int udcfEnvInt(const char* name, int dflt) {
        const char* p = std::getenv(name);
        return p ? std::atoi(p) : dflt;
    }
    // Frame window gate (0-based frameId; defaults dump every frame).
    inline bool udcfFrameInWindow(int frameId) {
        static int lo = udcfEnvInt("UDCF_DUMP_FRAME_MIN", -1);
        static int hi = udcfEnvInt("UDCF_DUMP_FRAME_MAX", 1 << 30);
        return frameId >= lo && frameId <= hi;
    }
}
```

- [ ] **Step 3: Add a dump lambda inside narrowPhase**

In `narrowPhase()`, immediately after the line `float d2 = -N2.dot(q[1][0]);` (PBDPhysics.cpp:1519), insert:

```cpp
        // ---- golden dump lambda: writes one CSV record in the layout
        // documented in docs/superpowers/plans/2026-06-11-dcd-collision.md.
        // For stage>=2 call it AFTER applyNarrowCollisionConstraintsCPU so
        // the post columns hold post-response positions.
        auto udcfDump = [&](int stage, float C1v, float C2v,
                            const Vec3& nrm, const Vec3& bCur, const Vec3& bClo,
                            int frictionMeshID) {
            FILE* fdump = udcfDumpFile();
            if (!fdump || udcfDumpCount >= UDCF_DUMP_MAX) return;
            if (!udcfFrameInWindow(frameId)) return;  // frameId is a PBDPhysics member
            ++udcfDumpCount;
            Vec3I v0ids = m_faces0->surfaceFacesTetMeshVIds.col(iSurfV0);
            Vec3I v1ids = m_faces1->surfaceFacesTetMeshVIds.col(iSurfV1);
            int fixed0 = 0, fixed1 = 0;
            for (int j = 0; j < 3; j++) {
                if (m_particles0->isFixed(v0ids[j])) fixed0 = 1;
                if (m_particles1->isFixed(v1ids[j])) fixed1 = 1;
            }
            FloatingType mus = objectParamsList.getObjectParam(frictionMeshID).friction_static;
            FloatingType mud = objectParamsList.getObjectParam(frictionMeshID).friction_dynamic;
            fprintf(fdump, "%d", stage);
            for (int j = 0; j < 3; j++)
                fprintf(fdump, ",%.9g,%.9g,%.9g", q[0][j][0], q[0][j][1], q[0][j][2]);
            for (int j = 0; j < 3; j++)
                fprintf(fdump, ",%.9g,%.9g,%.9g", q[1][j][0], q[1][j][1], q[1][j][2]);
            for (int j = 0; j < 3; j++) {
                Vec3 pv = m_particles0->vertexPrevPos(v0ids[j]);
                fprintf(fdump, ",%.9g,%.9g,%.9g", pv[0], pv[1], pv[2]);
            }
            for (int j = 0; j < 3; j++) {
                Vec3 pv = m_particles1->vertexPrevPos(v1ids[j]);
                fprintf(fdump, ",%.9g,%.9g,%.9g", pv[0], pv[1], pv[2]);
            }
            fprintf(fdump, ",%d,%d,%.9g,%.9g", fixed0, fixed1, C1v, C2v);
            fprintf(fdump, ",%.9g,%.9g,%.9g", nrm[0], nrm[1], nrm[2]);
            fprintf(fdump, ",%.9g,%.9g,%.9g", bCur[0], bCur[1], bCur[2]);
            fprintf(fdump, ",%.9g,%.9g,%.9g", bClo[0], bClo[1], bClo[2]);
            fprintf(fdump, ",%.9g,%.9g,%.9g", mus, mud,
                    physicsParams().collisionConstraintMultiplier);
            for (int j = 0; j < 3; j++) {
                Vec3 pp = m_particles0->vertex(v0ids[j]);
                fprintf(fdump, ",%.9g,%.9g,%.9g", pp[0], pp[1], pp[2]);
            }
            for (int j = 0; j < 3; j++) {
                Vec3 pp = m_particles1->vertex(v1ids[j]);
                fprintf(fdump, ",%.9g,%.9g,%.9g", pp[0], pp[1], pp[2]);
            }
            fprintf(fdump, "\n");
        };
        const Vec3 udcfZero3(0.0f, 0.0f, 0.0f);
```

- [ ] **Step 4: Dump interval rejects (subsampled)**

Replace these three reject statements (PBDPhysics.cpp:1561-1566):

```cpp
		if (t1 == -1 && t2 == -1)
			continue;
		if (t3 == -1 && t4 == -1)
			continue;
		if (!(t2 >= t3 && t4 >= t1))
			continue;
```

with:

```cpp
		if ((t1 == -1 && t2 == -1) || (t3 == -1 && t4 == -1) || !(t2 >= t3 && t4 >= t1))
		{
			if (++udcfRejectSkip % 97 == 0)
				udcfDump(0, 0.0f, 0.0f, udcfZero3, udcfZero3, udcfZero3, iMesh0);
			continue;
		}
```

- [ ] **Step 5: Dump offset rejects and applied contacts**

Replace the offset-reject statement (PBDPhysics.cpp:1698-1700):

```cpp
		offset = 1e-6;
		if (C1 <= offset || C2 <= offset)
			continue;
```

with:

```cpp
		offset = 1e-6;
		if (C1 <= offset || C2 <= offset)
		{
			udcfDump(1, C1, C2, udcfZero3, udcfZero3, udcfZero3, iMesh0);
			continue;
		}
```

Then add a dump call after EACH of the two `applyNarrowCollisionConstraintsCPU(...)` calls (PBDPhysics.cpp:1727 and :1744). After the C1-branch call add:

```cpp
			udcfDump(2, C1, C2, grads, barycentric1, barycentric2, iMesh0);
```

After the C2-branch call add:

```cpp
			udcfDump(3, C1, C2, grads, barycentric1, barycentric2, iMesh1);
```

(`barycentric1/2` in each branch are the locals already declared there; they hold bary3/bary4 in the C2 branch, which is exactly the apply-order convention.)

- [ ] **Step 6: Rebuild**

```bash
make -j --directory ~/Dev/Gaia_UDCF/Simulator/PBDDynamics/build
```

Expected: links `PBDDynamics` with no errors. If `objectParamsList.getObjectParam` or `isFixed`/`vertexPrevPos` names fail to compile, grep their actual spelling in `PBDPhysics.cpp` (they are all used elsewhere in the file: `getObjectParam` at the friction lookup in `applyNarrowCollisionConstraintsCPU` ~:3042, `isFixed` ~:2958, `vertexPrevPos` ~:3011) and match it.

- [ ] **Step 7: Run the S01_UCollisionDemo scene with dumping on**

The golden scene is `Simulator/PBDDynamics/ParameterExamples/S01_UCollisionDemo` (U-shape drop, 60 frames, 20 substeps, self-collision; mirrors our own `data/collisionTest/u_collision.json`). Contacts are richest in frames 55-60, so the dump is gated to 0-based `frameId` 54-59:

```bash
cd ~/Dev/Gaia_UDCF
mkdir -p /tmp/udcf_out
UDCF_DUMP=/tmp/udcf_golden.csv UDCF_DUMP_FRAME_MIN=54 UDCF_DUMP_FRAME_MAX=59 \
./Simulator/PBDDynamics/build/PBDDynamics \
  Simulator/PBDDynamics/ParameterExamples/S01_UCollisionDemo/Models.json \
  Simulator/PBDDynamics/ParameterExamples/S01_UCollisionDemo/Parameters.json \
  /tmp/udcf_out -R ~/Dev/Gaia_UDCF --CPU
wc -l /tmp/udcf_golden.csv
awk -F, '{c[$1]++} END {for (s in c) print "stage", s, c[s]}' /tmp/udcf_golden.csv
```

Run this in the BACKGROUND (`run_in_background` Bash) and poll its output — it is a 60-frame CPU sim and may take many minutes; do not block on it.

Notes:
- The scene sets `allowCCD: true`. That is fine for golden capture: every record snapshots its own inputs (q/prev) at narrow time and its own post-apply outputs inside `narrowPhase()`, so CCD elsewhere in the step cannot contaminate a record. Do NOT edit the scene JSONs.
- `collisionConstraintMultiplier` is absent from this Parameters.json; the dump records whatever default the build uses, and downstream tasks read the recorded value — no action needed.

Expected: stages 0, 1 present and at least ~1000 records with stage 2 or 3. If stage 2/3 counts are near zero, widen the window (`UDCF_DUMP_FRAME_MIN=40`) and rerun; if still near zero, STOP and report — the scene may not be self-colliding the way we expect.

- [ ] **Step 8: Save the instrumentation diff into our repo**

```bash
mkdir -p /mnt/storage_disk_1/chunleli/Dev/rlmuscleV2/tools/udcf_golden
diff -u ~/Dev/Gaia_UDCF/Simulator/Modules/PBD/PBDPhysics.cpp.orig \
        ~/Dev/Gaia_UDCF/Simulator/Modules/PBD/PBDPhysics.cpp \
  > /mnt/storage_disk_1/chunleli/Dev/rlmuscleV2/tools/udcf_golden/gaia_dump.patch || true
```

(`diff -u` exits 1 when files differ — the `|| true` keeps the shell happy.)

- [ ] **Step 9: Commit**

```bash
cd /mnt/storage_disk_1/chunleli/Dev/rlmuscleV2
git add tools/udcf_golden/gaia_dump.patch
git commit -m "tools(collision): Gaia_UDCF golden-dump instrumentation patch"
```

---

### Task 2: Golden converter + committed npz

**Files:**
- Create: `tools/udcf_golden/make_golden.py`
- Create: `tests/golden/udcf_pairs.npz` (generated)

- [ ] **Step 1: Write the converter**

`tools/udcf_golden/make_golden.py`:

```python
"""Convert a UDCF_DUMP CSV (from the gaia_dump.patch instrumentation of
Gaia_UDCF) into tests/golden/udcf_pairs.npz.

Record layout (71 fields) is documented in
docs/superpowers/plans/2026-06-11-dcd-collision.md and must stay in sync
with the fprintf order in gaia_dump.patch.

Usage:
    .venv/bin/python tools/udcf_golden/make_golden.py /tmp/udcf_golden.csv \
        tests/golden/udcf_pairs.npz
"""
import sys

import numpy as np

MAX_STAGE0 = 4000      # interval rejects (already 1/97 subsampled in C++)
MAX_STAGE1 = 4000      # offset rejects
MAX_APPLIED = 12000    # stage 2 + 3
SEED = 0


def main(csv_path: str, npz_path: str) -> None:
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    assert raw.shape[1] == 71, f"expected 71 cols, got {raw.shape[1]}"

    stage = raw[:, 0].astype(np.int32)
    rng = np.random.default_rng(SEED)

    def take(mask, cap):
        idx = np.flatnonzero(mask)
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        return np.sort(idx)

    keep = np.concatenate([
        take(stage == 0, MAX_STAGE0),
        take(stage == 1, MAX_STAGE1),
        take((stage == 2) | (stage == 3), MAX_APPLIED),
    ])
    r = raw[keep].astype(np.float32)
    s = stage[keep]

    out = dict(
        stage=s,
        q0=r[:, 1:10].reshape(-1, 3, 3),
        q1=r[:, 10:19].reshape(-1, 3, 3),
        prev0=r[:, 19:28].reshape(-1, 3, 3),
        prev1=r[:, 28:37].reshape(-1, 3, 3),
        fixed0=r[:, 37].astype(np.int32),
        fixed1=r[:, 38].astype(np.int32),
        C1=r[:, 39],
        C2=r[:, 40],
        n=r[:, 41:44],
        bary_cur=r[:, 44:47],
        bary_closest=r[:, 47:50],
        mu_s=r[:, 50],
        mu_d=r[:, 51],
        mult=r[:, 52],
        post0=r[:, 53:62].reshape(-1, 3, 3),
        post1=r[:, 62:71].reshape(-1, 3, 3),
    )
    np.savez_compressed(npz_path, **out)
    uniq, cnt = np.unique(s, return_counts=True)
    print(f"wrote {npz_path}: {len(s)} records, stages {dict(zip(uniq.tolist(), cnt.tolist()))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 2: Generate and sanity-check the npz**

```bash
mkdir -p tests/golden
.venv/bin/python tools/udcf_golden/make_golden.py /tmp/udcf_golden.csv tests/golden/udcf_pairs.npz
ls -la tests/golden/udcf_pairs.npz
```

Expected: prints record/stage counts; file well under 10 MB. If stage 2+3 < 1000, regenerate the CSV with more frames (Task 1 Step 7).

- [ ] **Step 3: Commit**

```bash
git add tools/udcf_golden/make_golden.py tests/golden/udcf_pairs.npz
git commit -m "test(collision): golden UDCF contact records from Gaia_UDCF reference"
```

---

### Task 3: Geometry funcs + narrow kernel (golden replay TDD)

**Files:**
- Create: `src/muscle/collision/udcf_dcd.py` (geometry + narrow kernel portion)
- Create: `tests/test_udcf_dcd.py` (narrow replay test)

- [ ] **Step 1: Write the failing narrow replay test**

`tests/test_udcf_dcd.py`:

```python
"""Golden replay tests: our Warp UDCF DCD kernels vs the instrumented
Gaia_UDCF C++ reference (tests/golden/udcf_pairs.npz).

Parity is asserted record-level. A tiny mismatch budget (0.1%) absorbs the
one known reference bug we deliberately do not replicate (stale Q[i][j]
reuse across pairs in narrowPhase) plus fp32 candidate ties.
"""
from pathlib import Path

import numpy as np
import pytest
import warp as wp

wp.init()

GOLDEN = Path(__file__).parent / "golden" / "udcf_pairs.npz"

skip_no_golden = pytest.mark.skipif(not GOLDEN.exists(),
                                    reason="golden npz not generated")

ATOL = 2e-5          # absolute tolerance on depths / barys / normals
PASS_RATE = 0.999    # allowed mismatch budget
TIE_EPS = 5e-8       # C1-vs-C2 tie window: below this the branch pick is
                     # fp noise (CPU-Eigen vs GPU-Warp last-bit divergence)


def _load():
    return dict(np.load(GOLDEN))


def _run_narrow(g):
    """Feed every golden pair (T0, T1) through narrow_kernel as one batch.

    Each record gets its own 6 verts; record i uses verts [6i..6i+6) and
    tris (2i, 2i+1), so contacts map back to records via vert_id // 6.
    """
    from muscle.collision.udcf_dcd import narrow_kernel

    n = len(g["stage"])
    x_np = np.stack([g["q0"], g["q1"]], axis=1).reshape(-1, 3)
    tris_np = np.arange(n * 6, dtype=np.int32).reshape(n * 2, 3)
    pair_a = np.arange(0, 2 * n, 2, dtype=np.int32)
    pair_b = np.arange(1, 2 * n, 2, dtype=np.int32)

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    x = wp.array(x_np.astype(np.float32), dtype=wp.vec3, device=dev)
    tris = wp.array2d(tris_np, dtype=wp.int32, device=dev)
    pa = wp.array(pair_a, dtype=wp.int32, device=dev)
    pb = wp.array(pair_b, dtype=wp.int32, device=dev)
    n_pairs = wp.array([n], dtype=wp.int32, device=dev)

    max_c = n
    n_contacts = wp.zeros(1, dtype=wp.int32, device=dev)
    c_cur = wp.zeros((max_c, 3), dtype=wp.int32, device=dev)
    c_closest = wp.zeros((max_c, 3), dtype=wp.int32, device=dev)
    c_C = wp.zeros(max_c, dtype=wp.float32, device=dev)
    c_C1 = wp.zeros(max_c, dtype=wp.float32, device=dev)
    c_C2 = wp.zeros(max_c, dtype=wp.float32, device=dev)
    c_n = wp.zeros(max_c, dtype=wp.vec3, device=dev)
    c_b1 = wp.zeros(max_c, dtype=wp.vec3, device=dev)
    c_b2 = wp.zeros(max_c, dtype=wp.vec3, device=dev)

    wp.launch(narrow_kernel, dim=n,
              inputs=[pa, pb, n_pairs, tris, x,
                      wp.float32(1.0e-6), wp.int32(max_c),
                      n_contacts, c_cur, c_closest,
                      c_C, c_C1, c_C2, c_n, c_b1, c_b2],
              device=dev)
    wp.synchronize()
    nc = int(n_contacts.numpy()[0])
    return dict(nc=nc, cur=c_cur.numpy()[:nc], closest=c_closest.numpy()[:nc],
                C=c_C.numpy()[:nc], C1=c_C1.numpy()[:nc], C2=c_C2.numpy()[:nc],
                n=c_n.numpy()[:nc], b1=c_b1.numpy()[:nc], b2=c_b2.numpy()[:nc])


@skip_no_golden
def test_narrow_golden_replay():
    g = _load()
    out = _run_narrow(g)
    n = len(g["stage"])
    applied = (g["stage"] >= 2)

    # Map each emitted contact back to its golden record.
    got = np.zeros(n, dtype=bool)
    rec_of_contact = out["cur"][:, 0] // 6
    mismatches = []
    for ci, ri in enumerate(rec_of_contact):
        got[ri] = True
        st = 2 if (out["cur"][ci, 0] % 6) < 3 else 3   # cur==T0 → stage2
        depths_ok = (abs(out["C1"][ci] - g["C1"][ri]) <= ATOL
                     and abs(out["C2"][ci] - g["C2"][ri]) <= ATOL)
        if st == g["stage"][ri]:
            ok = (depths_ok
                  and np.allclose(out["n"][ci], g["n"][ri], atol=ATOL)
                  and np.allclose(out["b1"][ci], g["bary_cur"][ri], atol=1e-3)
                  and np.allclose(out["b2"][ci], g["bary_closest"][ri], atol=1e-3))
        else:
            # Branch flip is acceptable ONLY on a C1≈C2 tie (≤ TIE_EPS, on
            # either side's arithmetic): the reference's pick is last-bit fp
            # noise there, and the golden n/bary belong to the other branch
            # so only the depths are comparable.
            ok = depths_ok and (
                abs(g["C1"][ri] - g["C2"][ri]) <= TIE_EPS
                or abs(out["C1"][ci] - out["C2"][ci]) <= TIE_EPS)
        if not ok:
            mismatches.append(ri)

    false_neg = np.flatnonzero(applied & ~got)        # golden applied, we rejected
    false_pos = np.flatnonzero(~applied & got)        # golden rejected, we applied
    bad = len(mismatches) + len(false_neg) + len(false_pos)
    rate = 1.0 - bad / n
    assert rate >= PASS_RATE, (
        f"parity {rate:.4%} < {PASS_RATE:.1%}: value-mismatch={len(mismatches)} "
        f"false-neg={len(false_neg)} false-pos={len(false_pos)} of {n}; "
        f"first bad records: {(mismatches + list(false_neg) + list(false_pos))[:10]}")
```

- [ ] **Step 2: Run to verify it fails**

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_udcf_dcd.py::test_narrow_golden_replay -x -q
```

Expected: FAIL with `ModuleNotFoundError`/`ImportError` (no `udcf_dcd` yet).

- [ ] **Step 3: Implement geometry funcs + narrow kernel**

Create `src/muscle/collision/udcf_dcd.py`:

```python
"""Faithful Warp port of Gaia_UDCF's triangle-triangle DCD.

Reference: ~/Dev/Gaia_UDCF/Simulator/Modules/PBD/PBDPhysics.cpp
  narrowPhase()                          :1264-1762
  applyNarrowCollisionConstraintsCPU()   :2894-3116
  runStepCPU() loop structure            :622-719

Pipeline (host orchestrated by UdcfDcdCollider, one call per solver
fine-iteration):

  broadphase (hashgrid, per substep)  →  narrow_kernel (per candidate pair)
  →  respond_kernel (per contact)     →  apply_kernel (Jacobi dP/count)

Replicated reference quirks (numerical parity with the golden dump beats
taste — see docs/superpowers/specs/2026-06-11-dcd-collision-design.md):
  * candidate depths use offset=0; only the final accept test uses 1e-6
  * smaller-positive of (C1, C2) wins; tie goes to the C2 branch
  * response pushes all 3 verts of each movable face EQUALLY (no mass or
    bary weighting); both movable → half each; one fixed → full to movable
  * friction interpolates closest-face velocities with the cur face's
    barys and vice versa (reference :3032/:3035)
Deviations (documented, accepted):
  * Jacobi accumulation (dP / count) instead of serial Gauss-Seidel
  * invalid plane-line intersections are skipped instead of reusing the
    previous pair's stale Q point (reference bug)
  * parallel-plane pairs (|N1×N2| < 1e-8) are skipped

All arithmetic fp32.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import warp as wp

from muscle.collision.udcf_warp import (
    centroid_radius_kernel,
    max_radius_reduce_kernel,
)

THRESH = wp.constant(1.0e-5)   # gen_t same-side threshold (reference :1366)


# ======================================================================
# Geometry helpers — line-for-line ports of narrowPhase()'s `internal`
# ======================================================================

@wp.func
def _pick(a: wp.vec3, b: wp.vec3, c: wp.vec3, i: int) -> wp.vec3:
    if i == 0:
        return a
    if i == 1:
        return b
    return c


@wp.func
def _inside_tri(p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, p: wp.vec3) -> int:
    # reference inside_triangle_on_same_plane (:1302)
    c1 = wp.cross(p2 - p1, p - p2)
    c2 = wp.cross(p3 - p2, p - p3)
    c3 = wp.cross(p1 - p3, p - p1)
    if wp.dot(c1, c2) > 0.0 and wp.dot(c1, c3) > 0.0:
        return 1
    return 0


@wp.func
def _bary(A: wp.vec3, B: wp.vec3, C: wp.vec3, P: wp.vec3) -> wp.vec3:
    # reference computeBarycentric (:1395) — dot-product form, NOT the
    # area form used by udcf_warp._bary_tri
    v0 = B - A
    v1 = C - A
    v2 = P - A
    d00 = wp.dot(v0, v0)
    d01 = wp.dot(v0, v1)
    d11 = wp.dot(v1, v1)
    d20 = wp.dot(v2, v0)
    d21 = wp.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if denom == 0.0:
        return wp.vec3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return wp.vec3(1.0 - v - w, v, w)


@wp.func
def _line_isect(p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, p4: wp.vec3,
                v1: wp.vec3, v2: wp.vec3):
    # reference line_intersection_on_same_plane (:1270): edges (p1,p2) and
    # (p3,p4) are co-planar projections; the hit is evaluated on the
    # ORIGINAL edge (v1,v2). Returns (valid, point).
    d1 = p2 - p1
    d2 = p4 - p3
    nrm = wp.cross(d1, d2)
    if wp.length(nrm) < 1.0e-6:
        return 0, wp.vec3(0.0, 0.0, 0.0)
    denom = wp.dot(nrm, nrm)
    if denom < 1.0e-6:
        return 0, wp.vec3(0.0, 0.0, 0.0)
    v = p3 - p1
    t1 = wp.dot(wp.cross(v, d2), nrm) / denom
    t2 = wp.dot(wp.cross(v, d1), nrm) / denom
    if t2 >= 0.0 and t2 <= 1.0 and t1 >= 0.0 and t1 <= 1.0:
        return 1, v1 + t1 * (v2 - v1)
    return 0, wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _find_isect_pt(N: wp.vec3, p0: wp.vec3, p: wp.vec3, line_dir: wp.vec3):
    # reference find_intersection_point (:1293)
    denom = wp.dot(N, line_dir)
    if denom == 0.0:
        return 0, wp.vec3(0.0, 0.0, 0.0)
    t = wp.dot(N, p0 - p) / denom
    return 1, p + t * line_dir


@wp.func
def _gen_t(N: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3,
           d: float, D: wp.vec3):
    # reference gen_t (:1361): signed distances vs plane (N, d), same-side
    # reject with THRESH, reorder so sv1 is the lone-side vertex, then the
    # Möller interval [t_lo, t_hi] along D. Returns (valid, t_lo, t_hi).
    d0 = wp.dot(N, v0) + d
    d1 = wp.dot(N, v1) + d
    d2 = wp.dot(N, v2) + d
    if d0 <= THRESH and d1 <= THRESH and d2 <= THRESH:
        return 0, 0.0, 0.0
    if d0 >= -THRESH and d1 >= -THRESH and d2 >= -THRESH:
        return 0, 0.0, 0.0

    sv0 = v0
    sv1 = v1
    sv2 = v2
    sd0 = d0
    sd1 = d1
    sd2 = d2
    # reference swap (:1322): two cascaded if/else-if blocks
    if (d0 <= 0.0 and d1 >= 0.0 and d2 >= 0.0) or \
       (d0 >= 0.0 and d1 <= 0.0 and d2 <= 0.0):
        sv0 = v1
        sv1 = v0
        sd0 = d1
        sd1 = d0
    elif (d0 >= 0.0 and d1 >= 0.0 and d2 <= 0.0) or \
         (d0 <= 0.0 and d1 <= 0.0 and d2 >= 0.0):
        sv1 = v2
        sv2 = v1
        sd1 = d2
        sd2 = d1

    p0 = wp.dot(D, sv0)
    p1 = wp.dot(D, sv1)
    p2 = wp.dot(D, sv2)
    tA = p0 + (p1 - p0) * wp.abs(sd0 / (sd0 - sd1))
    tB = p2 + (p1 - p2) * wp.abs(sd2 / (sd2 - sd1))
    if tA > tB:
        return 1, tB, tA
    return 1, tA, tB


# ======================================================================
# Narrow phase kernel — port of narrowPhase()'s per-pair body
# ======================================================================

@wp.kernel
def narrow_kernel(
    pair_a: wp.array(dtype=wp.int32),
    pair_b: wp.array(dtype=wp.int32),
    n_pairs: wp.array(dtype=wp.int32),
    tris: wp.array2d(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    offset_eps: wp.float32,            # reference final accept offset (1e-6)
    max_contacts: wp.int32,
    n_contacts: wp.array(dtype=wp.int32),
    c_cur: wp.array2d(dtype=wp.int32),       # (max_contacts, 3) vert ids, +C·n side
    c_closest: wp.array2d(dtype=wp.int32),   # (max_contacts, 3) vert ids, −C·n side
    c_C: wp.array(dtype=wp.float32),         # chosen depth (pre-multiplier)
    c_C1: wp.array(dtype=wp.float32),        # diagnostics / golden replay
    c_C2: wp.array(dtype=wp.float32),
    c_n: wp.array(dtype=wp.vec3),            # gradient = opposing plane normal
    c_b1: wp.array(dtype=wp.vec3),           # bary as apply's barycentric1
    c_b2: wp.array(dtype=wp.vec3),           # bary as apply's barycentric2
):
    k = wp.tid()
    if k >= n_pairs[0]:
        return
    ta = pair_a[k]
    tb = pair_b[k]
    q00 = x[tris[ta, 0]]
    q01 = x[tris[ta, 1]]
    q02 = x[tris[ta, 2]]
    q10 = x[tris[tb, 0]]
    q11 = x[tris[tb, 1]]
    q12 = x[tris[tb, 2]]

    n1f = wp.cross(q01 - q00, q02 - q00)
    n2f = wp.cross(q11 - q10, q12 - q10)
    l1 = wp.length(n1f)
    l2 = wp.length(n2f)
    if l1 == 0.0 or l2 == 0.0:           # degenerate triangle
        return
    N1 = n1f / l1
    N2 = n2f / l2
    d1 = -wp.dot(N1, q00)
    d2 = -wp.dot(N2, q10)

    cp = wp.cross(N1, N2)
    lcp = wp.length(cp)
    if lcp < 1.0e-8:                     # parallel planes (deviation: skip)
        return
    D = cp / lcp

    ok1, t1, t2 = _gen_t(N2, q00, q01, q02, d2, D)
    ok2, t3, t4 = _gen_t(N1, q10, q11, q12, d1, D)
    if ok1 == 0 or ok2 == 0:
        return
    if not (t2 >= t3 and t4 >= t1):
        return

    C1 = float(0.0)
    C2 = float(0.0)
    b1 = wp.vec3(0.0, 0.0, 0.0)
    b2 = wp.vec3(0.0, 0.0, 0.0)
    b3 = wp.vec3(0.0, 0.0, 0.0)
    b4 = wp.vec3(0.0, 0.0, 0.0)

    for j in range(3):
        q0j = _pick(q00, q01, q02, j)
        q1j = _pick(q10, q11, q12, j)
        # vertex-face: T0 vert projected onto T1's plane (reference P[0][j])
        P0j = q0j - wp.dot(N2, q0j - q10) * N2
        if _inside_tri(q10, q11, q12, P0j) == 1:
            cand = wp.dot(N2, q0j) + d2
            if C1 < -cand:
                C1 = -cand
                b1 = _bary(q00, q01, q02, q0j)
                b2 = _bary(q10, q11, q12, q0j - cand * N2)
        # Q point: T1 vert shot along N2 onto T0's plane (reference Q[0][j])
        okq, Q0j = _find_isect_pt(N1, q00, q1j, N2)
        if okq == 1 and _inside_tri(q00, q01, q02, Q0j) == 1:
            cand = wp.dot(N2, Q0j) + d2
            if C1 < -cand:
                C1 = -cand
                b1 = _bary(q00, q01, q02, Q0j)
                b2 = _bary(q10, q11, q12, Q0j - cand * N2)
        # symmetric direction (reference P[1][j] / Q[1][j])
        P1j = q1j - wp.dot(N1, q1j - q00) * N1
        if _inside_tri(q00, q01, q02, P1j) == 1:
            cand = wp.dot(N1, q1j) + d1
            if C2 < -cand:
                C2 = -cand
                b3 = _bary(q10, q11, q12, q1j)
                b4 = _bary(q00, q01, q02, q1j - cand * N1)
        okq, Q1j = _find_isect_pt(N2, q10, q0j, N1)
        if okq == 1 and _inside_tri(q10, q11, q12, Q1j) == 1:
            cand = wp.dot(N1, Q1j) + d1
            if C2 < -cand:
                C2 = -cand
                b3 = _bary(q10, q11, q12, Q1j)
                b4 = _bary(q00, q01, q02, Q1j - cand * N1)
        # edge-edge (reference inner k loop)
        jn = (j + 1) % 3
        q0jn = _pick(q00, q01, q02, jn)
        q1jn = _pick(q10, q11, q12, jn)
        P0jn = q0jn - wp.dot(N2, q0jn - q10) * N2
        P1jn = q1jn - wp.dot(N1, q1jn - q00) * N1
        for kk in range(3):
            kn = (kk + 1) % 3
            q1k = _pick(q10, q11, q12, kk)
            q1kn = _pick(q10, q11, q12, kn)
            oke, out = _line_isect(P0j, P0jn, q1k, q1kn, q0j, q0jn)
            if oke == 1:
                cand = wp.dot(N2, out) + d2
                if C1 < -cand:
                    C1 = -cand
                    b1 = _bary(q00, q01, q02, out)
                    b2 = _bary(q10, q11, q12, out - cand * N2)
            q0k = _pick(q00, q01, q02, kk)
            q0kn = _pick(q00, q01, q02, kn)
            oke, out = _line_isect(P1j, P1jn, q0k, q0kn, q1j, q1jn)
            if oke == 1:
                cand = wp.dot(N1, out) + d1
                if C2 < -cand:
                    C2 = -cand
                    b3 = _bary(q10, q11, q12, out)
                    b4 = _bary(q00, q01, q02, out - cand * N1)

    # reference :1698-1745: both must exceed offset; smaller wins (tie → C2)
    if C1 <= offset_eps or C2 <= offset_eps:
        return
    slot = wp.atomic_add(n_contacts, 0, 1)
    if slot >= max_contacts:
        return
    if C1 < C2:
        for j in range(3):
            c_cur[slot, j] = tris[ta, j]
            c_closest[slot, j] = tris[tb, j]
        c_C[slot] = C1
        c_n[slot] = N2
        c_b1[slot] = b1
        c_b2[slot] = b2
    else:
        for j in range(3):
            c_cur[slot, j] = tris[tb, j]
            c_closest[slot, j] = tris[ta, j]
        c_C[slot] = C2
        c_n[slot] = N1
        c_b1[slot] = b3
        c_b2[slot] = b4
    c_C1[slot] = C1
    c_C2[slot] = C2
```

- [ ] **Step 4: Run the narrow replay test**

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_udcf_dcd.py::test_narrow_golden_replay -x -q
```

Expected: PASS (parity ≥ 99.9%). If it fails, the assertion message lists the first bad record indices — debug by printing that record's `q0/q1/C1/C2/stage` from the npz and stepping the math by hand against PBDPhysics.cpp. Common causes: `_gen_t` swap-branch mismatch, plane `d` sign, missing `(j+1)%3` projected endpoint.

- [ ] **Step 5: Commit**

```bash
git add src/muscle/collision/udcf_dcd.py tests/test_udcf_dcd.py
git commit -m "feat(collision): UDCF DCD narrow-phase kernel, golden-parity tested"
```

---

### Task 4: Respond kernel (golden replay + synthetic solid-side TDD)

**Files:**
- Modify: `src/muscle/collision/udcf_dcd.py` (append respond + apply kernels)
- Modify: `tests/test_udcf_dcd.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_udcf_dcd.py`:

```python
def _run_respond(x_np, prev_np, im_np, cur_ids, closest_ids, C, n, b1, b2,
                 mult, mu_s, mu_d):
    """One respond_kernel launch over a batch of single contacts."""
    from muscle.collision.udcf_dcd import respond_kernel, apply_kernel

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    nv = x_np.shape[0]
    nc = len(C)
    x = wp.array(x_np.astype(np.float32), dtype=wp.vec3, device=dev)
    xprev = wp.array(prev_np.astype(np.float32), dtype=wp.vec3, device=dev)
    im = wp.array(im_np.astype(np.float32), dtype=wp.float32, device=dev)
    n_contacts = wp.array([nc], dtype=wp.int32, device=dev)
    c_cur = wp.array2d(cur_ids.astype(np.int32), dtype=wp.int32, device=dev)
    c_closest = wp.array2d(closest_ids.astype(np.int32), dtype=wp.int32, device=dev)
    c_C = wp.array(C.astype(np.float32), dtype=wp.float32, device=dev)
    c_n = wp.array(n.astype(np.float32), dtype=wp.vec3, device=dev)
    c_b1 = wp.array(b1.astype(np.float32), dtype=wp.vec3, device=dev)
    c_b2 = wp.array(b2.astype(np.float32), dtype=wp.vec3, device=dev)
    mu_s_a = wp.array(mu_s.astype(np.float32), dtype=wp.float32, device=dev)
    mu_d_a = wp.array(mu_d.astype(np.float32), dtype=wp.float32, device=dev)
    dP = wp.zeros(nv, dtype=wp.vec3, device=dev)
    dPw = wp.zeros(nv, dtype=wp.float32, device=dev)

    wp.launch(respond_kernel, dim=nc,
              inputs=[n_contacts, c_cur, c_closest, c_C, c_n, c_b1, c_b2,
                      x, xprev, im, wp.float32(mult), mu_s_a, mu_d_a,
                      dP, dPw],
              device=dev)
    wp.launch(apply_kernel, dim=nv, inputs=[x, dP, dPw], device=dev)
    wp.synchronize()
    return x.numpy()


@skip_no_golden
def test_respond_golden_replay():
    g = _load()
    sel = np.flatnonzero(g["stage"] >= 2)
    nrec = len(sel)
    # 6 isolated verts per record: [T0 verts, T1 verts]
    x_np = np.stack([g["q0"][sel], g["q1"][sel]], axis=1).reshape(nrec, 6, 3)
    prev_np = np.stack([g["prev0"][sel], g["prev1"][sel]], axis=1).reshape(nrec, 6, 3)
    base = (np.arange(nrec) * 6)[:, None]
    t0_ids = base + np.array([[0, 1, 2]])
    t1_ids = base + np.array([[3, 4, 5]])
    st = g["stage"][sel]
    cur_ids = np.where((st == 2)[:, None], t0_ids, t1_ids)
    closest_ids = np.where((st == 2)[:, None], t1_ids, t0_ids)
    fixed_t0 = g["fixed0"][sel]
    fixed_t1 = g["fixed1"][sel]
    im_np = np.ones(nrec * 6, dtype=np.float32)
    im_np[(t0_ids[fixed_t0 == 1]).reshape(-1)] = 0.0
    im_np[(t1_ids[fixed_t1 == 1]).reshape(-1)] = 0.0
    mult = float(g["mult"][sel][0])
    assert np.allclose(g["mult"][sel], mult), "mixed multipliers in golden set"

    x_out = _run_respond(
        x_np.reshape(-1, 3), prev_np.reshape(-1, 3), im_np,
        cur_ids, closest_ids,
        g["C"][sel] if "C" in g else np.minimum(g["C1"][sel], g["C2"][sel]),
        g["n"][sel], g["bary_cur"][sel], g["bary_closest"][sel],
        mult, g["mu_s"][sel], g["mu_d"][sel],
    ).reshape(nrec, 6, 3)

    post = np.stack([g["post0"][sel], g["post1"][sel]], axis=1).reshape(nrec, 6, 3)
    err = np.abs(x_out - post).max(axis=(1, 2))
    bad = np.flatnonzero(err > ATOL)
    rate = 1.0 - len(bad) / nrec
    assert rate >= PASS_RATE, (
        f"respond parity {rate:.4%}: {len(bad)}/{nrec} records exceed "
        f"{ATOL} (worst {err.max():.2e} at record {int(err.argmax())})")


def test_respond_solid_side_full_push():
    """One side fixed → movable face takes FULL C·mult along n; fixed face
    does not move (reference :2966-2967 skips the 0.5 split, :3009/:3018
    skip the solid side; friction on fixed verts is nulled by re-pinning,
    which our inv_mass guard reproduces)."""
    n = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    C = np.array([0.01], dtype=np.float32)
    mult = 1.01
    # cur = movable tri at y=0; closest = fixed tri slightly above
    x_np = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],   # cur (movable)
        [0.0, 0.005, 0.0], [1.0, 0.005, 0.0], [0.0, 0.005, 1.0],  # closest (fixed)
    ], dtype=np.float32)
    prev_np = x_np.copy()                      # zero velocity → zero friction
    im_np = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
    cur_ids = np.array([[0, 1, 2]])
    closest_ids = np.array([[3, 4, 5]])
    b = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float32)
    mu = np.array([0.5], dtype=np.float32)

    x_out = _run_respond(x_np, prev_np, im_np, cur_ids, closest_ids,
                         C, n, b, b, mult, mu, mu * 0.8)
    push = C[0] * mult
    assert np.allclose(x_out[0:3, 1], push, atol=1e-6), x_out[0:3]
    assert np.allclose(x_out[3:6], x_np[3:6], atol=1e-7), "fixed face moved"


def test_respond_both_movable_half_split():
    """Both movable → C·mult·0.5 each, opposite directions."""
    n = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    C = np.array([0.01], dtype=np.float32)
    mult = 1.01
    x_np = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
        [0.0, 0.005, 0.0], [1.0, 0.005, 0.0], [0.0, 0.005, 1.0],
    ], dtype=np.float32)
    prev_np = x_np.copy()
    im_np = np.ones(6, dtype=np.float32)
    b = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=np.float32)
    mu = np.array([0.5], dtype=np.float32)

    x_out = _run_respond(x_np, prev_np, im_np, np.array([[0, 1, 2]]),
                         np.array([[3, 4, 5]]), C, n, b, b, mult, mu, mu * 0.8)
    half = C[0] * mult * 0.5
    assert np.allclose(x_out[0:3, 1] - x_np[0:3, 1], +half, atol=1e-6)
    assert np.allclose(x_out[3:6, 1] - x_np[3:6, 1], -half, atol=1e-6)
```

Note: the golden npz has no `"C"` key — the `g["C"] if "C" in g` guard resolves to `min(C1, C2)`, which is the reference's chosen depth.

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_udcf_dcd.py -k respond -x -q
```

Expected: FAIL with `ImportError: cannot import name 'respond_kernel'`.

- [ ] **Step 3: Implement respond + apply kernels**

Append to `src/muscle/collision/udcf_dcd.py`:

```python
# ======================================================================
# Response — port of applyNarrowCollisionConstraintsCPU (:2894-3116)
# ======================================================================

@wp.kernel
def respond_kernel(
    n_contacts: wp.array(dtype=wp.int32),
    c_cur: wp.array2d(dtype=wp.int32),
    c_closest: wp.array2d(dtype=wp.int32),
    c_C: wp.array(dtype=wp.float32),         # pre-multiplier depth
    c_n: wp.array(dtype=wp.vec3),
    c_b1: wp.array(dtype=wp.vec3),           # apply's barycentric1
    c_b2: wp.array(dtype=wp.vec3),           # apply's barycentric2
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=wp.float32),
    multiplier: wp.float32,
    mu_s: wp.array(dtype=wp.float32),        # per-contact (golden replay
    mu_d: wp.array(dtype=wp.float32),        # carries per-mesh values)
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
):
    k = wp.tid()
    if k >= n_contacts[0]:
        return
    pen = c_C[k]                       # reference penetrationDepth arg
    C = pen * multiplier               # :2901
    if C == 0.0:
        return
    nrm = c_n[k]

    cu0 = c_cur[k, 0]
    cu1 = c_cur[k, 1]
    cu2 = c_cur[k, 2]
    cl0 = c_closest[k, 0]
    cl1 = c_closest[k, 1]
    cl2 = c_closest[k, 2]

    solid_cur = 0
    solid_clo = 0
    if inv_mass[cu0] == 0.0 or inv_mass[cu1] == 0.0 or inv_mass[cu2] == 0.0:
        solid_cur = 1
    if inv_mass[cl0] == 0.0 or inv_mass[cl1] == 0.0 or inv_mass[cl2] == 0.0:
        solid_clo = 1
    if solid_cur == 0 and solid_clo == 0:    # :2966
        C = C * 0.5

    # normal push (local; written to x only via dP) + post-push velocities
    push_clo = wp.vec3(0.0, 0.0, 0.0)
    if solid_clo == 0:
        push_clo = -C * nrm
    push_cur = wp.vec3(0.0, 0.0, 0.0)
    if solid_cur == 0:
        push_cur = C * nrm
    v0 = (x[cl0] + push_clo) - x_prev[cl0]   # :3005-3023 velocities[0..5]
    v1 = (x[cl1] + push_clo) - x_prev[cl1]
    v2 = (x[cl2] + push_clo) - x_prev[cl2]
    v3 = (x[cu0] + push_cur) - x_prev[cu0]
    v4 = (x[cu1] + push_cur) - x_prev[cu1]
    v5 = (x[cu2] + push_cur) - x_prev[cu2]

    g = nrm / wp.length(nrm)                 # :3029-3030
    b1 = c_b1[k]
    b2 = c_b2[k]
    # reference quirk (:3032/:3035): closest velocities × barycentric1,
    # cur velocities × barycentric2 — replicated for golden parity
    vclo = v0 * b1[0] + v1 * b1[1] + v2 * b1[2]
    perp_clo = vclo - g * wp.dot(vclo, g)
    vcur = v3 * b2[0] + v4 * b2[1] + v5 * b2[2]
    perp_cur = vcur - g * wp.dot(vcur, g)
    diff = perp_cur - perp_clo               # :3038
    dnorm = wp.length(diff)
    static_norm = pen * mu_s[k]              # :3058
    if dnorm > static_norm and dnorm > 0.0:  # :3060 slipping → pen·μd
        diff = diff * (pen * mu_d[k] / dnorm)
    diffV = diff * 0.5                       # :3072 cur faces get −diffV
    diffT = diff - diffV                     # :3077 closest faces get +diffT

    # accumulate; inv_mass==0 verts skipped — equivalent to the reference
    # moving fixed verts then re-pinning them in applyConstraints()
    if inv_mass[cu0] > 0.0:
        wp.atomic_add(dP, cu0, push_cur - diffV)
        wp.atomic_add(dPw, cu0, 1.0)
    if inv_mass[cu1] > 0.0:
        wp.atomic_add(dP, cu1, push_cur - diffV)
        wp.atomic_add(dPw, cu1, 1.0)
    if inv_mass[cu2] > 0.0:
        wp.atomic_add(dP, cu2, push_cur - diffV)
        wp.atomic_add(dPw, cu2, 1.0)
    if inv_mass[cl0] > 0.0:
        wp.atomic_add(dP, cl0, push_clo + diffT)
        wp.atomic_add(dPw, cl0, 1.0)
    if inv_mass[cl1] > 0.0:
        wp.atomic_add(dP, cl1, push_clo + diffT)
        wp.atomic_add(dPw, cl1, 1.0)
    if inv_mass[cl2] > 0.0:
        wp.atomic_add(dP, cl2, push_clo + diffT)
        wp.atomic_add(dPw, cl2, 1.0)


@wp.kernel
def apply_kernel(
    pos: wp.array(dtype=wp.vec3),
    dP: wp.array(dtype=wp.vec3),
    dPw: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    w = dPw[i]
    if w > 0.0:
        pos[i] = pos[i] + dP[i] / w
```

- [ ] **Step 4: Run the tests**

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_udcf_dcd.py -k respond -x -q
```

Expected: 3 PASS. If `test_respond_golden_replay` fails on friction-bearing records but passes on near-zero-velocity ones, recheck the b1/b2 quirk pairing and that `pen` (not `C·mult`) feeds the friction norms.

- [ ] **Step 5: Commit**

```bash
git add src/muscle/collision/udcf_dcd.py tests/test_udcf_dcd.py
git commit -m "feat(collision): UDCF DCD response kernel with friction, golden-parity tested"
```

---

### Task 5: Cross-group broadphase emit kernel

**Files:**
- Modify: `src/muscle/collision/udcf_dcd.py` (append kernel)
- Modify: `tests/test_udcf_dcd.py` (append test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_udcf_dcd.py`:

```python
def test_emit_pairs_groups():
    """Group rules: cross-group always; movable-self only when allowed;
    static-static never; shared-vert self pairs filtered."""
    from muscle.collision.udcf_dcd import emit_pairs_groups_kernel
    from muscle.collision.udcf_warp import centroid_radius_kernel

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    # 4 tris stacked at the origin: 0,1 movable (group 0; tri 1 shares vert 1
    # with tri 0), 2,3 static (group 1)
    x_np = np.array([
        [0, 0, 0], [1, 0, 0], [0, 0, 1],     # tri0 (movable)
        [0.1, 0.01, 0.1], [1, 0.01, 1],      # tri1 = [1, 3, 4] shares vert 1
        [0, 0.02, 0], [1, 0.02, 0], [0, 0.02, 1],   # tri2 (static)
        [0, 0.03, 0], [1, 0.03, 0], [0, 0.03, 1],   # tri3 (static)
    ], dtype=np.float32)
    tris_np = np.array([[0, 1, 2], [1, 3, 4], [5, 6, 7], [8, 9, 10]],
                       dtype=np.int32)
    group_np = np.array([0, 0, 1, 1], dtype=np.int32)

    x = wp.array(x_np, dtype=wp.vec3, device=dev)
    tris = wp.array2d(tris_np, dtype=wp.int32, device=dev)
    group = wp.array(group_np, dtype=wp.int32, device=dev)
    cent = wp.zeros(4, dtype=wp.vec3, device=dev)
    radii = wp.zeros(4, dtype=wp.float32, device=dev)
    wp.launch(centroid_radius_kernel, dim=4, inputs=[tris, x, cent, radii],
              device=dev)
    grid = wp.HashGrid(8, 8, 8, device=dev)
    grid.build(cent, 4.0)

    def emit(allow_self):
        pa = wp.zeros(64, dtype=wp.int32, device=dev)
        pb = wp.zeros(64, dtype=wp.int32, device=dev)
        n = wp.zeros(1, dtype=wp.int32, device=dev)
        wp.launch(emit_pairs_groups_kernel, dim=4,
                  inputs=[grid.id, cent, radii, tris, group,
                          wp.int32(allow_self), wp.float32(4.0),
                          wp.float32(0.1), wp.int32(64), pa, pb, n],
                  device=dev)
        wp.synchronize()
        nn = int(n.numpy()[0])
        return set(map(tuple, np.stack([pa.numpy()[:nn], pb.numpy()[:nn]], 1)))

    pairs = emit(allow_self=0)
    # cross pairs only: (0,2),(0,3),(1,2),(1,3); never (2,3); no self
    assert pairs == {(0, 2), (0, 3), (1, 2), (1, 3)}, pairs
    pairs = emit(allow_self=1)
    # self allowed, but (0,1) shares vert 1 → adjacency-filtered
    assert pairs == {(0, 2), (0, 3), (1, 2), (1, 3)}, pairs
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_udcf_dcd.py::test_emit_pairs_groups -x -q
```

Expected: FAIL with ImportError.

- [ ] **Step 3: Implement**

Append to `src/muscle/collision/udcf_dcd.py`:

```python
# ======================================================================
# Broadphase: candidate pairs with movable/static group rules
# ======================================================================

@wp.kernel
def emit_pairs_groups_kernel(
    grid: wp.uint64,
    centroids: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=wp.float32),
    tris: wp.array2d(dtype=wp.int32),
    tri_group: wp.array(dtype=wp.int32),     # 0 = movable, 1 = static
    allow_self: wp.int32,                    # movable-vs-movable pairs
    query_radius: wp.float32,
    thickness: wp.float32,
    max_pairs: wp.int32,
    pair_a: wp.array(dtype=wp.int32),
    pair_b: wp.array(dtype=wp.int32),
    n_pairs: wp.array(dtype=wp.int32),
):
    """Unique candidate pairs (i < j). Rules:
      static-static          → never (neither side can move)
      movable-self (0 vs 0)  → only when allow_self, with shared-vertex
                               adjacency filter (vert spaces of distinct
                               groups are disjoint, so cross pairs never
                               share verts and skip the 9-way compare)
      cross (0 vs 1)         → always
    Counter saturates at max_pairs; host checks n_pairs for overflow."""
    i = wp.tid()
    ci = centroids[i]
    ri = radii[i]
    gi = tri_group[i]
    q = wp.hash_grid_query(grid, ci, query_radius)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if j <= i:
            continue
        gj = tri_group[j]
        if gi == 1 and gj == 1:
            continue
        if gi == 0 and gj == 0:
            if allow_self == 0:
                continue
            ai0 = tris[i, 0]
            ai1 = tris[i, 1]
            ai2 = tris[i, 2]
            bj0 = tris[j, 0]
            bj1 = tris[j, 1]
            bj2 = tris[j, 2]
            if (ai0 == bj0 or ai0 == bj1 or ai0 == bj2 or
                    ai1 == bj0 or ai1 == bj1 or ai1 == bj2 or
                    ai2 == bj0 or ai2 == bj1 or ai2 == bj2):
                continue
        if wp.length(ci - centroids[j]) > ri + radii[j] + thickness:
            continue
        slot = wp.atomic_add(n_pairs, 0, 1)
        if slot < max_pairs:
            pair_a[slot] = i
            pair_b[slot] = j
```

- [ ] **Step 4: Run, expect PASS, commit**

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_udcf_dcd.py::test_emit_pairs_groups -x -q
git add src/muscle/collision/udcf_dcd.py tests/test_udcf_dcd.py
git commit -m "feat(collision): group-aware broadphase pair emission for DCD"
```

---### Task 6: UdcfDcdCollider orchestration

**Files:**
- Modify: `src/muscle/collision/udcf_dcd.py` (append params + collider)
- Modify: `tests/test_udcf_dcd.py` (append integration smoke)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_udcf_dcd.py`:

```python
def test_collider_resolves_penetration():
    """A movable triangle pushed into a static one separates over a few
    solve() iterations along the static face's normal."""
    from muscle.collision.udcf_dcd import UdcfDcdCollider, UdcfParams

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    # movable tri tilted through the static one (guarantees both C1>0,C2>0)
    mov_np = np.array([[-0.5, -0.02, -0.3], [0.5, 0.04, -0.3], [0.0, -0.02, 0.6]],
                      dtype=np.float32)
    static_v = np.array([[-1, 0, -1], [1, 0, -1], [0, 0, 1.5]], dtype=np.float32)
    static_t = np.array([[0, 1, 2]], dtype=np.int32)

    pos = wp.array(mov_np, dtype=wp.vec3, device=dev)
    inv_mass = wp.array(np.ones(3, dtype=np.float32), dtype=wp.float32,
                        device=dev)
    coll = UdcfDcdCollider(pos, inv_mass,
                           surface_tris=np.array([[0, 1, 2]], dtype=np.int32),
                           static_verts=static_v, static_tris=static_t,
                           params=UdcfParams(max_pairs=64, max_contacts=64),
                           device=dev)
    depth0 = -float(pos.numpy()[:, 1].min())
    for it in range(8):
        coll.solve(it if it == 0 else 1)   # broadphase once, then iterate
    wp.synchronize()
    stats = coll.stats()
    assert stats["n_pairs"] >= 1
    assert stats["n_contacts"] >= 1
    depth1 = -float(pos.numpy()[:, 1].min())
    assert depth1 < depth0 * 0.5, (depth0, depth1)
```

- [ ] **Step 2: Run to verify failure** (`ImportError: UdcfDcdCollider`)

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_udcf_dcd.py::test_collider_resolves_penetration -x -q
```

- [ ] **Step 3: Implement params, CCD interface, collider**

Append to `src/muscle/collision/udcf_dcd.py`:

```python
# ======================================================================
# Host orchestration
# ======================================================================

@dataclass
class UdcfParams:
    multiplier: float = 1.01          # collisionConstraintMultiplier
    friction_static: float = 0.5
    friction_dynamic: float = 0.4
    offset: float = 1.0e-6            # final accept threshold
    thickness: float = 0.0            # broadphase sphere slack
    cell_size: Optional[float] = None  # None → 2·max_radius + thickness
    max_pairs: int = 1 << 20
    max_contacts: int = 1 << 18
    allow_self: bool = False          # movable-vs-movable pairs
    grid_dim: int = 128


class CcdInterface(Protocol):
    """Reserved continuous-collision hook. Implementations get called once
    per solve() after the DCD pass with the collider as argument."""

    def solve(self, collider: "UdcfDcdCollider") -> None: ...


class UdcfDcdCollider:
    """Gaia_UDCF DCD between a movable vert set and a static (per-frame
    animated) mesh, plus optional movable self-collision.

    ``pos`` is the live solver position array (e.g. ``OneWaySim.pos``) —
    the collider reads AND writes it in :meth:`solve`. Register on the
    solver's per-iteration hook::

        coll = UdcfDcdCollider(sim.pos, inv_mass, surf_tris, bone_v, bone_t)
        sim.collision_solvers_iter = [coll.solve]

    Static-mesh winding: triangle ``cross(e1, e2)`` normals must point OUT
    of the static body (SNU bone .geo files need the ``[:, [0, 2, 1]]``
    column swap, same as AnimatedBoneMeshCollider).
    """

    def __init__(self, pos: wp.array, inv_mass: wp.array,
                 surface_tris: np.ndarray,
                 static_verts: np.ndarray, static_tris: np.ndarray,
                 params: UdcfParams | None = None,
                 device: str | None = None,
                 ccd: CcdInterface | None = None):
        self.params = params or UdcfParams()
        self.device = device or str(pos.device)
        self.pos = pos
        self.ccd = ccd
        p = self.params

        self.n_mov = int(pos.shape[0])
        self.n_static = int(static_verts.shape[0])
        n_total = self.n_mov + self.n_static

        mov_tris = np.asarray(surface_tris, dtype=np.int32).reshape(-1, 3)
        sta_tris = (np.asarray(static_tris, dtype=np.int32).reshape(-1, 3)
                    + self.n_mov)
        all_tris = np.vstack([mov_tris, sta_tris])
        self.n_tris = int(all_tris.shape[0])
        self.tris = wp.array2d(all_tris, dtype=wp.int32, device=self.device)
        group_np = np.zeros(self.n_tris, dtype=np.int32)
        group_np[mov_tris.shape[0]:] = 1
        self.tri_group = wp.array(group_np, dtype=wp.int32, device=self.device)

        im_np = np.concatenate([inv_mass.numpy().astype(np.float32),
                                np.zeros(self.n_static, dtype=np.float32)])
        self.inv_mass = wp.array(im_np, dtype=wp.float32, device=self.device)

        x_np = np.zeros((n_total, 3), dtype=np.float32)
        x_np[self.n_mov:] = static_verts
        self.x = wp.array(x_np, dtype=wp.vec3, device=self.device)
        self.x_prev = wp.clone(self.x)

        self.centroids = wp.zeros(self.n_tris, dtype=wp.vec3, device=self.device)
        self.radii = wp.zeros(self.n_tris, dtype=wp.float32, device=self.device)
        self._max_radius = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.pair_a = wp.zeros(p.max_pairs, dtype=wp.int32, device=self.device)
        self.pair_b = wp.zeros(p.max_pairs, dtype=wp.int32, device=self.device)
        self.n_pairs = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.grid = wp.HashGrid(p.grid_dim, p.grid_dim, p.grid_dim,
                                device=self.device)

        mc = p.max_contacts
        self.n_contacts = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.c_cur = wp.zeros((mc, 3), dtype=wp.int32, device=self.device)
        self.c_closest = wp.zeros((mc, 3), dtype=wp.int32, device=self.device)
        self.c_C = wp.zeros(mc, dtype=wp.float32, device=self.device)
        self.c_C1 = wp.zeros(mc, dtype=wp.float32, device=self.device)
        self.c_C2 = wp.zeros(mc, dtype=wp.float32, device=self.device)
        self.c_n = wp.zeros(mc, dtype=wp.vec3, device=self.device)
        self.c_b1 = wp.zeros(mc, dtype=wp.vec3, device=self.device)
        self.c_b2 = wp.zeros(mc, dtype=wp.vec3, device=self.device)
        # respond_kernel takes per-contact friction (golden replay needs
        # per-mesh values); production fills them with the params constants
        self.mu_s = wp.full(mc, p.friction_static, dtype=wp.float32,
                            device=self.device)
        self.mu_d = wp.full(mc, p.friction_dynamic, dtype=wp.float32,
                            device=self.device)

        self.dP = wp.zeros(n_total, dtype=wp.vec3, device=self.device)
        self.dPw = wp.zeros(n_total, dtype=wp.float32, device=self.device)

        self._x_mov = self.x[:self.n_mov]
        self._xp_mov = self.x_prev[:self.n_mov]
        self._x_sta = self.x[self.n_mov:]
        self._xp_sta = self.x_prev[self.n_mov:]
        self._overflow_warned = False

    def set_static_verts(self, verts: np.ndarray) -> None:
        """New static pose (per-frame bone animation). The previous pose
        becomes x_prev for friction's contact velocity."""
        wp.copy(self._xp_sta, self._x_sta)
        self._x_sta.assign(np.asarray(verts, dtype=np.float32))

    def _broadphase(self) -> None:
        p = self.params
        wp.launch(centroid_radius_kernel, dim=self.n_tris,
                  inputs=[self.tris, self.x, self.centroids, self.radii],
                  device=self.device)
        if p.cell_size is None:
            wp.launch(max_radius_reduce_kernel, dim=1,
                      inputs=[self.radii, self._max_radius], device=self.device)
            cell = max(2.0 * float(self._max_radius.numpy()[0]) + p.thickness,
                       1.0e-4)
        else:
            cell = p.cell_size
        self.grid.build(self.centroids, cell)
        self.n_pairs.zero_()
        wp.launch(emit_pairs_groups_kernel, dim=self.n_tris,
                  inputs=[self.grid.id, self.centroids, self.radii, self.tris,
                          self.tri_group, wp.int32(int(p.allow_self)),
                          wp.float32(cell), wp.float32(p.thickness),
                          wp.int32(p.max_pairs),
                          self.pair_a, self.pair_b, self.n_pairs],
                  device=self.device)

    def solve(self, i_iter: int = 0) -> None:
        """One DCD pass: refresh movable verts, (re)build pairs on the
        first iteration of each substep, then narrow → respond → apply."""
        p = self.params
        if i_iter == 0:
            wp.copy(self._xp_mov, self.pos)   # substep-start snapshot
        wp.copy(self._x_mov, self.pos)
        if i_iter == 0:
            self._broadphase()
        self.n_contacts.zero_()
        wp.launch(narrow_kernel, dim=p.max_pairs,
                  inputs=[self.pair_a, self.pair_b, self.n_pairs, self.tris,
                          self.x, wp.float32(p.offset),
                          wp.int32(p.max_contacts),
                          self.n_contacts, self.c_cur, self.c_closest,
                          self.c_C, self.c_C1, self.c_C2,
                          self.c_n, self.c_b1, self.c_b2],
                  device=self.device)
        self.dP.zero_()
        self.dPw.zero_()
        wp.launch(respond_kernel, dim=p.max_contacts,
                  inputs=[self.n_contacts, self.c_cur, self.c_closest,
                          self.c_C, self.c_n, self.c_b1, self.c_b2,
                          self.x, self.x_prev, self.inv_mass,
                          wp.float32(p.multiplier), self.mu_s, self.mu_d,
                          self.dP, self.dPw],
                  device=self.device)
        wp.launch(apply_kernel, dim=self.n_mov,
                  inputs=[self.pos, self.dP, self.dPw], device=self.device)
        if self.ccd is not None:
            self.ccd.solve(self)

    def stats(self) -> dict:
        """Host-side diagnostics (one device sync). Warns once per overflow."""
        np_ = int(self.n_pairs.numpy()[0])
        nc = int(self.n_contacts.numpy()[0])
        p = self.params
        if (np_ > p.max_pairs or nc > p.max_contacts) and not self._overflow_warned:
            self._overflow_warned = True
            print(f"[udcf_dcd] WARNING: overflow pairs={np_}/{p.max_pairs} "
                  f"contacts={nc}/{p.max_contacts} — raise UdcfParams caps")
        return {"n_pairs": min(np_, p.max_pairs),
                "n_contacts": min(nc, p.max_contacts)}
```

Two Warp API points to verify while implementing (adjust if the installed Warp disagrees):
- 1-D array slicing views (`self.x[:self.n_mov]`), `wp.copy(view, src)`, and `view.assign(np_array)` in `set_static_verts`. If views reject any of these, fall back to whole-array `wp.copy(dest, src, dest_offset=..., count=...)` or keep separate movable/static arrays and pass both to the kernels via an index offset;
- `narrow_kernel` is launched with `dim=max_pairs` and self-guards on `n_pairs[0]`; `n_pairs[0]` may exceed `max_pairs` when the emit counter saturates, so the guard in `narrow_kernel` must be `if k >= n_pairs[0]: return` *plus* the implicit `k < max_pairs` from the launch dim — pairs beyond the cap simply were never stored. That is already the semantics of the code above.

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_udcf_dcd.py::test_collider_resolves_penetration -x -q
```

- [ ] **Step 5: Run the whole test file, commit**

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_udcf_dcd.py -q
git add src/muscle/collision/udcf_dcd.py tests/test_udcf_dcd.py
git commit -m "feat(collision): UdcfDcdCollider host orchestration with CCD interface"
```

---

### Task 7: solver.py per-iteration hook

**Files:**
- Modify: `src/muscle/coupleOneWayGmg/solver.py:1206-1234`
- Modify: `tests/test_udcf_dcd.py` (append hook test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_udcf_dcd.py` (lion data pattern copied from `tests/test_lion_sim_smoke.py`):

```python
import json
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
_LION = _ROOT / "data" / "lion" / "model"
_LION_CFG = _ROOT / "data" / "lion" / "config" / "lion_nomg.json"

skip_no_lion = pytest.mark.skipif(not (_LION / "restpos.geo").exists(),
                                  reason="lion .geo blobs not present")


@skip_no_lion
def test_solver_iter_hook_called_per_fine_iteration():
    from muscle.coupleOneWayGmg.solver import OneWaySim

    cfg = json.load(open(_LION_CFG))
    cfg["geo_dir"] = str(_LION)
    sim = OneWaySim(cfg)

    calls = []
    sim.collision_solvers_iter = [lambda it: calls.append(it)]
    end_calls = []
    sim.collision_solvers = [lambda: end_calls.append(1)]

    sim.step_frame(sim.start_frame)
    expected = sim.fine_iters * sim.nsubsteps
    assert len(calls) == expected, (len(calls), expected)
    # each substep counts 0..fine_iters-1
    assert calls[:sim.fine_iters] == list(range(sim.fine_iters))
    assert len(end_calls) == sim.nsubsteps   # old hook untouched
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_udcf_dcd.py::test_solver_iter_hook_called_per_fine_iteration -x -q
```

Expected: FAIL — `calls` stays empty (hook not wired).

(If `sim.fine_iters` / `sim.nsubsteps` attribute names differ, grep `fine_iters` and `nsubsteps` in `solver.py` — both appear in `__init__` — and fix the test, not the solver.)

- [ ] **Step 3: Implement the hook**

In `src/muscle/coupleOneWayGmg/solver.py`, change the fine-iteration loop (currently `for _ in range(self.fine_iters):` at line 1206) to:

```python
        for it in range(self.fine_iters):
```

and append inside that loop, after the `if self.use_volume:` block (i.e. as the last statement of the loop body):

```python
            # Per-iteration collision hook (UDCF DCD): interleaves contact
            # resolution with ARAP/volume the way the reference PBD loop
            # interleaves narrowPhase() with projectConstraints().
            for fn in getattr(self, "collision_solvers_iter", None) or []:
                fn(it)
```

- [ ] **Step 4: Run the test + existing solver smoke tests**

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_udcf_dcd.py::test_solver_iter_hook_called_per_fine_iteration \
    tests/test_lion_sim_smoke.py -q
```

Expected: all PASS (regression: lion smoke unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/muscle/coupleOneWayGmg/solver.py tests/test_udcf_dcd.py
git commit -m "feat(solver): per-fine-iteration collision hook for DCD interleaving"
```

---

### Task 8: U-shape self-collision regression demo

**Files:**
- Create: `src/muscle/collision/example_dcd_u.py`

- [ ] **Step 1: Write the demo**

`src/muscle/collision/example_dcd_u.py`:

```python
"""U-shape drop regression for the UDCF DCD collider (self + plane contact).

Successor of the bit-rotted example_u_collision.py (whose collider diverged:
max_bary +0.057 → +0.204 monotonic, floor tunneling to min_y −0.20 m).
Pass criteria printed as a final METRICS line:
  min_y_overall > -0.05   (no floor tunneling)
  max_bary_final < 0.10   (self-penetration bounded, not diverging)

Run::

    PYTHONPATH=src .venv/bin/python -m muscle.collision.example_dcd_u \
        --config data/collisionTest/u_collision.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp

from muscle.collision.udcf_dcd import UdcfDcdCollider, UdcfParams
from muscle.muscle_warp import MuscleSim
from muscle.util.geo_io import GeoFile
from muscle.util.mesh_io import build_surface_tris


def _build_plane_mesh_xz(y, half_extent, subdiv):
    n = int(max(1, subdiv))
    xs_1d = np.linspace(-half_extent, half_extent, n + 1)
    zs_1d = np.linspace(-half_extent, half_extent, n + 1)
    xs, zs = np.meshgrid(xs_1d, zs_1d, indexing="xy")
    verts = np.stack([xs.flatten(), np.full(xs.size, y), zs.flatten()],
                     axis=1).astype(np.float32)
    tris = []
    for j in range(n):
        for i in range(n):
            v00 = j * (n + 1) + i
            v10 = v00 + 1
            v01 = (j + 1) * (n + 1) + i
            v11 = v01 + 1
            tris.append([v00, v11, v10])   # CCW from +Y → normals point up
            tris.append([v00, v01, v11])
    return verts, np.asarray(tris, dtype=np.int32)


def _scan_max_bary(pos_np, tets, vert_side, tet_side):
    """Worst (vert inside opposite-arm tet) barycentric — >0 means
    self-penetration. Same diagnostic as the old demo."""
    V0 = pos_np[tets[:, 0]]
    M = np.stack([pos_np[tets[:, 1]] - V0, pos_np[tets[:, 2]] - V0,
                  pos_np[tets[:, 3]] - V0], axis=-1)
    det = np.linalg.det(M)
    ok = np.abs(det) > 1e-12
    Minv = np.zeros_like(M)
    Minv[ok] = np.linalg.inv(M[ok])
    b123 = np.einsum("tik,vk->vti", Minv, pos_np) - \
        np.einsum("tik,tk->ti", Minv, V0)[None]
    b0 = 1.0 - b123.sum(axis=2)
    min_bary = np.minimum.reduce([b0, b123[..., 0], b123[..., 1], b123[..., 2]])
    mask = ((vert_side[:, None] * tet_side[None, :]) < 0) & ok[None, :]
    return float(np.where(mask, min_bary, -np.inf).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/collisionTest/u_collision.json")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--coll_iters", type=int, default=3,
                    help="DCD sweeps per substep (reference iterations=3)")
    args = ap.parse_args()

    wp.init()
    cfg = json.load(open(args.config))
    nsteps = args.steps or int(cfg["nsteps"])
    plane_cfg = cfg["collision"]["plane"]

    g = GeoFile(cfg["geo_path"])
    verts = np.asarray(g.positions, dtype=np.float32)
    tets = np.asarray(g.tet_indices, dtype=np.int32)
    verts[:, 0] -= 0.5 * (verts[:, 0].min() + verts[:, 0].max())
    verts[:, 2] -= 0.5 * (verts[:, 2].min() + verts[:, 2].max())
    verts[:, 1] += float(plane_cfg["drop_height"]) - verts[:, 1].min()
    surf = build_surface_tris(tets, verts)

    fiber = np.zeros((tets.shape[0], 3), dtype=np.float32)
    fiber[:, 0] = 1.0
    sim = MuscleSim.from_procedural(
        verts, tets, fiber,
        constraint_configs=list(cfg["constraints"]),
        dts=float(cfg["dt"]), num_substeps=int(cfg["num_substeps"]),
        device=args.device, density=float(cfg["density"]),
        veldamping=float(cfg["veldamping"]),
        gravity=-abs(float(cfg["gravity_z"])),
    )

    plane_v, plane_t = _build_plane_mesh_xz(
        0.0, float(plane_cfg["half_extent"]), int(plane_cfg["subdiv"]))

    mass = sim.mass.numpy()
    inv_mass = wp.array(
        np.where(mass > 0, 1.0 / np.where(mass > 0, mass, 1.0), 0.0)
        .astype(np.float32), dtype=wp.float32, device=args.device)
    coll = UdcfDcdCollider(
        sim.pos, inv_mass, surface_tris=np.asarray(sim.surface_tris, np.int32),
        static_verts=plane_v, static_tris=plane_t,
        params=UdcfParams(allow_self=True,
                          max_pairs=int(cfg["collision"]["max_pairs"]),
                          thickness=float(cfg["collision"]["thickness"])),
        device=args.device)

    def collision_solver():
        for i in range(args.coll_iters):
            coll.solve(i)

    sim.constraint_solvers = [collision_solver]

    rest = sim.pos.numpy().copy()
    vert_side = np.sign(rest[:, 0]).astype(np.int32)
    tet_side = np.sign(rest[sim.tet_np].mean(axis=1)[:, 0]).astype(np.int32)

    min_y_overall = 1e9
    mb = 0.0
    for step in range(1, nsteps + 1):
        sim.step_cnt = step
        sim.step()
        if step % 10 == 0 or step == 1 or step == nsteps:
            p = sim.pos.numpy()
            mb = _scan_max_bary(p, sim.tet_np, vert_side, tet_side)
            min_y = float(p[:, 1].min())
            min_y_overall = min(min_y_overall, min_y)
            s = coll.stats()
            print(f"step={step:4d} min_y={min_y:+.4f} max_bary={mb:+.4f} "
                  f"pairs={s['n_pairs']} contacts={s['n_contacts']} "
                  f"{'PEN!' if mb > 0 else 'ok'}")

    ok = (min_y_overall > -0.05) and (mb < 0.10)
    print(f"METRICS min_y_overall={min_y_overall:+.4f} max_bary_final={mb:+.4f} "
          f"=> {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Note: `MuscleSim.from_procedural` and attribute names (`mass`, `surface_tris`, `tet_np`, `constraint_solvers`, `step_cnt`, `step`) follow the old demo (`example_u_collision.py:387-466`); verify against `src/muscle/muscle_warp.py` if anything errors, and adapt the demo (not the collider).

- [ ] **Step 2: Run the regression**

```bash
PYTHONPATH=src .venv/bin/python -m muscle.collision.example_dcd_u \
    --config data/collisionTest/u_collision.json
echo "exit=$?"
```

Expected: `METRICS ... => PASS`, exit 0. The old port's failure signature (monotonic `max_bary` growth, `min_y` ≈ −0.20) must be gone.

Fallback if the body gains energy and launches upward (collision push enters `update_velocities` as velocity): add `--sync_prev` mode to the demo that mirrors the dP into `sim.pprev` after each `collision_solver()` call (see `example_u_collision.py:59-75` `_apply_collision_dP_kernel` for the pattern) and re-run. Reference PBD does not sync, so try without first; document which mode passes in the commit message.

- [ ] **Step 3: Commit**

```bash
git add src/muscle/collision/example_dcd_u.py
git commit -m "test(collision): U-drop self-collision regression for DCD collider"
```

---

### Task 9: humanV2 muscle-vs-bone demo + perf/penetration stats

**Files:**
- Create: `src/muscle/collision/example_humanV2_bone_dcd.py`

- [ ] **Step 1: Write the demo**

`src/muscle/collision/example_humanV2_bone_dcd.py`:

```python
"""humanV2 muscle vs animated bone with the UDCF DCD collider.

Same scenario as example_humanV2_bone_anim (SDF baseline) but contact runs
inside OneWaySim's fine-iteration loop via collision_solvers_iter.
AnimatedBoneMeshCollider is kept UNREGISTERED purely as a penetration
gauge (inside_depth) so SDF/DCD numbers are directly comparable.

Run::

    PYTHONPATH=src .venv/bin/python -m muscle.collision.example_humanV2_bone_dcd \
        --config data/humanV2/config/humanV2_gmg.json \
        --bone_dir data/humanV2/model/bone --end_frame 20
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import warp as wp

from muscle.collision.example_humanV2_bone_anim import (
    AnimatedBoneMeshCollider, _load_bone_verts)
from muscle.collision.udcf_dcd import UdcfDcdCollider, UdcfParams
from muscle.coupleOneWayGmg.example import _vectorized_surface_tris
from muscle.coupleOneWayGmg.solver import OneWaySim, load_config
from muscle.util.geo_io import GeoFile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="data/humanV2/config/humanV2_gmg.json")
    ap.add_argument("--bone_dir", default="data/humanV2/model/bone")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--end_frame", type=int, default=20)
    ap.add_argument("--mode", choices=["dcd", "sdf", "off"], default="dcd")
    args = ap.parse_args()

    wp.init()
    cfg = load_config(args.config)
    cfg["end_frame"] = args.end_frame
    sim = OneWaySim(cfg, device=args.device)

    # measurement gauge (and the SDF baseline when --mode sdf)
    gauge = AnimatedBoneMeshCollider(sim, args.bone_dir, max_dist=0.2)

    bone_dir = Path(args.bone_dir)
    g0 = GeoFile(str(bone_dir / "bone_1.geo"))
    bone_tris = np.asarray(g0.tet_indices, dtype=np.int32)[:, [0, 2, 1]]
    bone_v0 = _load_bone_verts(bone_dir, 1)
    surf = _vectorized_surface_tris(sim.tet_idx.numpy()).astype(np.int32)

    dcd = None
    if args.mode == "dcd":
        # full inv_mass like the SDF collider: humanV2 is fully pin-driven,
        # the push must win over the pin within the iteration
        dcd = UdcfDcdCollider(sim.pos, gauge.coll_inv_mass, surf,
                              bone_v0, bone_tris,
                              params=UdcfParams(), device=args.device)
        sim.collision_solvers_iter = [dcd.solve]
    elif args.mode == "sdf":
        sim.collision_solvers = [gauge]

    print(f"mode={args.mode} NV={sim.NV} NT={sim.NT} "
          f"muscle_surf_tris={surf.shape[0]} bone_tris={bone_tris.shape[0]}")

    t_total = 0.0
    for frame in range(sim.start_frame, sim.end_frame + 1):
        gauge.set_frame(frame)                     # gauge follows animation
        if dcd is not None:
            dcd.set_static_verts(_load_bone_verts(bone_dir, frame))
        wp.synchronize()
        t0 = time.perf_counter()
        sim.step_frame(frame)
        wp.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        t_total += dt_ms
        depth = gauge.inside_depth()
        extra = ""
        if dcd is not None:
            s = dcd.stats()
            extra = f" pairs={s['n_pairs']} contacts={s['n_contacts']}"
        print(f"frame {frame:3d} {dt_ms:7.1f}ms inside={int((depth > 0).sum())} "
              f"max_depth={depth.max() * 1000:.1f}mm{extra}")
        assert not np.isnan(sim.positions_np()).any(), f"NaN at frame {frame}"

    n = sim.end_frame - sim.start_frame + 1
    print(f"METRICS mode={args.mode} mean_ms_per_frame={t_total / n:.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run A/B and record the numbers**

```bash
PYTHONPATH=src .venv/bin/python -m muscle.collision.example_humanV2_bone_dcd --mode off --end_frame 10
PYTHONPATH=src .venv/bin/python -m muscle.collision.example_humanV2_bone_dcd --mode sdf --end_frame 10
PYTHONPATH=src .venv/bin/python -m muscle.collision.example_humanV2_bone_dcd --mode dcd --end_frame 10
```

Acceptance:
- `dcd` mode: no NaN, no exploding verts, `max_depth` comparable to or lower than `sdf` mode by frame 10;
- perf: `dcd` mean ms/frame within 10% of the ~1600 ms sim cost over `off` mode (i.e. ≤ ~160 ms collision overhead; SDF baseline was +24 ms). If over budget, first lever: `UdcfParams.cell_size` fixed value (skips the per-substep readback), second lever: broadphase every other substep (call `solve` with a frozen pair list — broadphase already only runs at `i_iter == 0`).

- [ ] **Step 3: Commit**

```bash
git add src/muscle/collision/example_humanV2_bone_dcd.py
git commit -m "feat(collision): humanV2 muscle-vs-bone DCD demo with SDF A/B metrics"
```

---

### Task 10: Full suite + wrap-up

- [ ] **Step 1: Run everything**

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_udcf_dcd.py tests/test_lion_sim_smoke.py tests/test_sdf_drop.py -q
```

Expected: all PASS (SDF path regression-free).

- [ ] **Step 2: Restore the reference repo (keep the patch file)**

```bash
cp ~/Dev/Gaia_UDCF/Simulator/Modules/PBD/PBDPhysics.cpp.orig \
   ~/Dev/Gaia_UDCF/Simulator/Modules/PBD/PBDPhysics.cpp
```

(Reapply later with `patch -p0 < tools/udcf_golden/gaia_dump.patch` from the Gaia_UDCF root if the golden set ever needs regenerating.)

- [ ] **Step 3: Final commit if anything is pending**

```bash
git status --short
```

Expected: clean (everything committed per task).
