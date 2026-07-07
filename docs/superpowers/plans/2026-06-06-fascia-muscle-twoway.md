> **⚠️ Historical plan.** The multigrid / `gmg` / `restpos_lowres.geo` references
> predate `src/muscle/mg/mg.py` (`MultiGridSim`) with the inflate cage. Current
> multigrid: [`../../humanV2_cage_multigrid.md`](../../humanV2_cage_multigrid.md).

# Fascia + Muscle Two-Way Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fascia tet shell to the humanV2 muscle sim that contracts inward (rest-scale shrink) and, through two-way XPBD glue, squeezes the muscles together so they stay tight and don't splay.

**Architecture:** Muscle + fascia live in ONE merged particle system inside `coupleOneWayGmg`. Fascia contraction reuses the existing ARAP-SVD kernel by feeding it a scaled rest matrix `B = inv(s·Dm0)` (`s` = shrink factor). A new particle↔particle XPBD distance kernel binds fascia inner points to nearest muscle surface vertices (two-way). Fascia `rigid` points ride the bone-driven muscle. New kernels are added (not signature-changed) so the existing lion path/tests are untouched.

**Tech Stack:** Python, NVIDIA Warp 1.13 (`uv run`), scipy cKDTree, Houdini JSON `.geo`, USD/PLY export. Tests: pytest on default (CPU) Warp device.

---

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `src/muscle/util/geo_io.py` | `.geo` reader. Add: tets from `Tetrahedron_run`, generic named point attrs, point-group decode (`ordered` list + `boolRLE`). Keep existing trimmed API intact. | Modify |
| `src/muscle/coupleOneWayGmg/solver.py` | Add kernels `precompute_rest_scaled_kernel`, `solve_glue_kernel`, `fascia_anchor_kernel`; extend `OneWaySim` with `_init_fascia`, `apply_fascia_anchor`, `solve_glue`, fascia-aware `substep`. | Modify |
| `data/humanV2/config/humanV2_fascia.json` | Scene config: gmg params + `fascia_path`, `fascia_rest_scale`, `fascia_glue_iterations`. | Create |
| `src/muscle/coupleOneWayGmg/example.py` | Optional dual-layer USD/PLY export (muscle + fascia tagged). | Modify |
| `tests/test_fascia_geo_io.py` | geo_io fascia decode tests. | Create |
| `tests/test_fascia_kernels.py` | Unit tests for the 3 new kernels. | Create |
| `tests/test_fascia_sim_smoke.py` | Synthetic merged-mesh integration smoke + (skipped) humanV2 smoke. | Create |

**Asset prep (do once, before Task 1 tests that need real data):**
```bash
mkdir -p data/humanV2/model/fascia
cp ~/Dev/fascia.geo data/humanV2/model/fascia/fascia.geo
# humanV2 muscle assets are zipped — extract for integration runs:
(cd data/humanV2 && unzip -o model.zip)   # yields model/muscle/restpos.geo + cons/
```
fascia.geo (~5 MB) and the extracted humanV2 model are gitignored per the repo's model.zip convention — do NOT commit the raw geo.

---

## Task 1: geo_io — read fascia tets, attrs, and groups

**Files:**
- Modify: `src/muscle/util/geo_io.py`
- Test: `tests/test_fascia_geo_io.py`

Background on the format (verified against `~/Dev/fascia.geo`):
- Root pair-list keys include `topology`, `attributes`, `pointgroups`, `primitivecount`.
- `topology.pointref.indices` is a flat int list. `primitives` is a list of run blocks; block 0 header `{"type":"Tetrahedron_run"}` with body `{"startvertex":0,"nprimitives":NT}` — its tets are the first `NT*4` entries of `indices`. A `Polygon_run` block may follow (ignored).
- Point attributes: each entry `[meta_pairs, data_pairs]`; `meta.name`, `data.values[5]` holds payload. Vector attrs (`P`,`materialW`,`tpose`) → shape `(NV,3)` directly at `values[5]`. Scalar attrs → shape `(1,NV)` at `values[5]` (transpose to `(NV,)`).
- Point groups: each `[meta_pairs, sel_pairs]`; `meta.name`; `sel.selection` is a pair-list with one key: `"ordered"` → `{... "ordered":[idx,...]}` OR `"unordered"` → `["boolRLE",[count,bool,count,bool,...]]`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fascia_geo_io.py
"""Tests for geo_io fascia (mixed tet+poly, named attrs, point groups)."""
from pathlib import Path

import numpy as np
import pytest

from muscle.util.geo_io import GeoFile

FASCIA = Path(__file__).resolve().parents[1] / "data" / "humanV2" / "model" / "fascia" / "fascia.geo"

skip_no_data = pytest.mark.skipif(
    not FASCIA.exists(),
    reason="fascia.geo not present (cp ~/Dev/fascia.geo data/humanV2/model/fascia/)",
)


@skip_no_data
def test_fascia_tets_and_positions():
    g = GeoFile(str(FASCIA))
    assert g.positions.shape == (15348, 3)
    assert g.positions.dtype == np.float32
    # Tetrahedron_run only: 66785 tets, polygons excluded.
    assert g.tet_indices.shape == (66785, 4)
    assert g.tet_indices.min() >= 0
    assert g.tet_indices.max() < 15348


@skip_no_data
def test_fascia_named_point_attrs():
    g = GeoFile(str(FASCIA))
    d = g.point_attr("tissuetomuscledistance")
    assert d.shape == (15348,)
    assert np.allclose(d, 0.01, atol=1e-6)
    mat = g.point_attr("materialW")
    assert mat.shape == (15348, 3)
    assert g.point_attr("solidshapestiffness").mean() == pytest.approx(5000.0, rel=1e-3)


@skip_no_data
def test_fascia_point_groups():
    g = GeoFile(str(FASCIA))
    rigid = g.point_group("rigid")          # ordered list
    inner = g.point_group("tissue_inner_points")  # boolRLE
    bone = g.point_group("tissue_bone_points")     # boolRLE
    for grp in (rigid, inner, bone):
        assert grp.dtype == np.int32
        assert grp.ndim == 1
        assert grp.min() >= 0 and grp.max() < 15348
        assert len(np.unique(grp)) == len(grp)  # no dupes
    assert len(rigid) > 0 and len(inner) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fascia_geo_io.py -v`
Expected: FAIL — `GeoFile` has no `point_attr`/`point_group`; `tet_indices` wrong shape (current code reshapes ALL prims, mixing polys).

- [ ] **Step 3: Implement geo_io changes**

In `src/muscle/util/geo_io.py`, replace the tet-topology block in `GeoFile.__init__` and add generic attr + group storage. Keep the existing `P/gluetoanimation/pts/target_pos` handling working.

```python
# --- inside GeoFile.__init__, after the existing pointattrs loop ---
# Generic named point-attribute capture (in addition to the trimmed fields).
self._attrs: dict[str, np.ndarray] = {}
for entry in pointattrs:
    meta = _pair_list_to_dict(entry[0])
    data = _pair_list_to_dict(entry[1])
    name = meta["name"]
    if "values" not in data:
        continue
    payload = np.asarray(data["values"][5])
    if payload.ndim == 2 and payload.shape[0] == 1:   # scalar attr stored (1,NV)
        payload = payload[0]
    self._attrs[name] = payload

# --- replace the tet-indices section ---
topo = _pair_list_to_dict(root["topology"])
pointref = _pair_list_to_dict(topo["pointref"])
indices = np.asarray(pointref["indices"], dtype=np.int32)

n_tets = 0
prims = root.get("primitives", [])
for blk in prims:
    hdr = _pair_list_to_dict(blk[0])
    if hdr.get("type") == "Tetrahedron_run":
        body = _pair_list_to_dict(blk[1])
        n_tets = int(body["nprimitives"])
        break
if n_tets > 0:
    tet_indices = indices[: n_tets * 4].reshape(n_tets, 4)
else:
    # Legacy path: whole file is one prim type (lion restpos).
    primitive_count = int(root.get("primitivecount", 0))
    if primitive_count > 0 and indices.size:
        vpp = indices.size // primitive_count
        tet_indices = indices.reshape(primitive_count, vpp)
    else:
        tet_indices = indices.reshape(0, 0)

# --- point groups ---
self._groups: dict[str, np.ndarray] = {}
for grp in root.get("pointgroups", []):
    gmeta = _pair_list_to_dict(grp[0])
    gsel = _pair_list_to_dict(grp[1])["selection"]
    seld = _pair_list_to_dict(gsel)
    if "ordered" in seld:
        idx = np.asarray(seld["ordered"], dtype=np.int32)
    elif "unordered" in seld:
        idx = _decode_bool_rle(seld["unordered"])
    else:
        continue
    self._groups[gmeta["name"]] = idx
```

Add module-level helper + accessor methods:

```python
def _decode_bool_rle(unordered) -> np.ndarray:
    """Decode Houdini ['boolRLE', [count, bool, count, bool, ...]] to int32 indices."""
    if not unordered or unordered[0] != "boolRLE":
        return np.zeros(0, dtype=np.int32)
    runs = unordered[1]
    out = []
    pos = 0
    for k in range(0, len(runs), 2):
        count = int(runs[k])
        val = bool(runs[k + 1])
        if val:
            out.extend(range(pos, pos + count))
        pos += count
    return np.asarray(out, dtype=np.int32)
```

```python
    # --- inside GeoFile, new public methods ---
    def point_attr(self, name: str) -> np.ndarray:
        if name not in self._attrs:
            raise KeyError(f"point attribute '{name}' not in {sorted(self._attrs)}")
        return self._attrs[name]

    def has_point_attr(self, name: str) -> bool:
        return name in self._attrs

    def point_group(self, name: str) -> np.ndarray:
        if name not in self._groups:
            raise KeyError(f"point group '{name}' not in {sorted(self._groups)}")
        return self._groups[name]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fascia_geo_io.py tests/test_geo_io.py -v`
Expected: fascia tests PASS; existing lion `test_geo_io.py` still PASS (no regression).

- [ ] **Step 5: Commit**

```bash
git add src/muscle/util/geo_io.py tests/test_fascia_geo_io.py
git commit -m "feat(geo_io): read fascia tets, named point attrs, point groups"
```

---

## Task 2: Scaled rest-matrix kernel (contraction)

**Files:**
- Modify: `src/muscle/coupleOneWayGmg/solver.py`
- Test: `tests/test_fascia_kernels.py`

A new precompute kernel that bakes a per-tet shrink scale and per-tet stiffness, leaving the lion `precompute_rest_kernel` untouched. `arap_svd_kernel` is reused unchanged (it already reads per-tet `B[]` and `alpha[]`).

- [ ] **Step 1: Write failing test**

```python
# tests/test_fascia_kernels.py
"""Unit tests for fascia Warp kernels."""
import numpy as np
import warp as wp

wp.init()

_REG_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)


def _tet_volume(p, idx=(0, 1, 2, 3)):
    a, b, c, d = (p[i] for i in idx)
    return abs(np.dot(np.cross(b - a, c - a), d - a)) / 6.0


def test_scaled_precompute_shrinks_rest():
    """B = inv(s*Dm); rest_volume uses UNSCALED Dm; alpha = 1/(stiff*V0)."""
    from muscle.coupleOneWayGmg.solver import precompute_rest_scaled_kernel

    pos = wp.array(_REG_TET, dtype=wp.vec3)
    tets = wp.array(np.array([[0, 1, 2, 3]], dtype=np.int32), dtype=wp.vec4i)
    scale = wp.array(np.array([0.5], dtype=np.float32), dtype=wp.float32)
    stiff = wp.array(np.array([5000.0], dtype=np.float32), dtype=wp.float32)
    B = wp.zeros(1, dtype=wp.mat33)
    V0 = wp.zeros(1, dtype=wp.float32)
    alpha = wp.zeros(1, dtype=wp.float32)

    wp.launch(precompute_rest_scaled_kernel, dim=1,
              inputs=[pos, tets, scale, stiff, B, V0, alpha])
    wp.synchronize()

    # Dm = I (regular tet) → B = inv(0.5*I) = 2*I
    np.testing.assert_allclose(B.numpy()[0], 2.0 * np.eye(3), atol=1e-5)
    np.testing.assert_allclose(V0.numpy()[0], 1.0 / 6.0, atol=1e-7)   # unscaled
    np.testing.assert_allclose(alpha.numpy()[0], 1.0 / (5000.0 * (1.0 / 6.0)), rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fascia_kernels.py::test_scaled_precompute_shrinks_rest -v`
Expected: FAIL — `precompute_rest_scaled_kernel` undefined.

- [ ] **Step 3: Implement the kernel**

Add to `src/muscle/coupleOneWayGmg/solver.py` (after `precompute_rest_kernel`):

```python
@wp.kernel
def precompute_rest_scaled_kernel(
    pos_rest: wp.array(dtype=wp.vec3),
    tet_indices: wp.array(dtype=wp.vec4i),
    scale: wp.array(dtype=wp.float32),      # per-tet shrink factor (1.0 = no shrink)
    stiffness: wp.array(dtype=wp.float32),  # per-tet ARAP stiffness (muscle: mu; fascia: solidshapestiffness)
    B: wp.array(dtype=wp.mat33),
    rest_volume: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
):
    """Like precompute_rest_kernel but per-tet: B = inv(scale*Dm),
    rest_volume from UNSCALED Dm (physical mass), alpha = 1/(stiffness*V0)."""
    t = wp.tid()
    idx = tet_indices[t]
    x0 = pos_rest[idx[0]]
    x1 = pos_rest[idx[1]]
    x2 = pos_rest[idx[2]]
    x3 = pos_rest[idx[3]]
    e1 = x1 - x0
    e2 = x2 - x0
    e3 = x3 - x0
    Dm = wp.mat33(
        e1[0], e2[0], e3[0],
        e1[1], e2[1], e3[1],
        e1[2], e2[2], e3[2],
    )
    det = wp.determinant(Dm)
    V0 = wp.abs(det) / 6.0
    s = scale[t]
    Dm_s = wp.mat33(
        s * e1[0], s * e2[0], s * e3[0],
        s * e1[1], s * e2[1], s * e3[1],
        s * e1[2], s * e2[2], s * e3[2],
    )
    B[t] = wp.inverse(Dm_s)
    rest_volume[t] = V0
    k = stiffness[t]
    alpha[t] = 1.0 / (k * V0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fascia_kernels.py::test_scaled_precompute_shrinks_rest -v`
Expected: PASS.

- [ ] **Step 5: Add a convergence test (free tet shrinks under ARAP)**

Append to `tests/test_fascia_kernels.py`:

```python
def test_free_tet_contracts_to_scale():
    """A free tet driven by ARAP toward a 0.5-scaled rest converges to 1/8 volume."""
    from muscle.coupleOneWayGmg.solver import (
        precompute_rest_scaled_kernel, arap_svd_kernel)

    pos_np = _REG_TET.copy()
    pos = wp.array(pos_np, dtype=wp.vec3)
    pos_mid = wp.clone(pos)
    tets = wp.array(np.array([[0, 1, 2, 3]], dtype=np.int32), dtype=wp.vec4i)
    scale = wp.array(np.array([0.5], dtype=np.float32), dtype=wp.float32)
    stiff = wp.array(np.array([1.0e9], dtype=np.float32), dtype=wp.float32)
    inv_mass = wp.array(np.ones(4, dtype=np.float32), dtype=wp.float32)
    B = wp.zeros(1, dtype=wp.mat33)
    V0 = wp.zeros(1, dtype=wp.float32)
    alpha = wp.zeros(1, dtype=wp.float32)
    L = wp.zeros(1, dtype=wp.float32)

    wp.launch(precompute_rest_scaled_kernel, dim=1,
              inputs=[pos, tets, scale, stiff, B, V0, alpha])
    dt = 0.004166
    for _ in range(2000):
        wp.copy(pos_mid, pos)
        wp.launch(arap_svd_kernel, dim=1,
                  inputs=[pos_mid, tets, B, inv_mass, alpha, L, dt, 0.2, pos])
    wp.synchronize()

    v = _tet_volume(pos.numpy())
    np.testing.assert_allclose(v, (0.5 ** 3) * (1.0 / 6.0), rtol=0.05)
```

Run: `uv run pytest tests/test_fascia_kernels.py -v`
Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add src/muscle/coupleOneWayGmg/solver.py tests/test_fascia_kernels.py
git commit -m "feat(solver): per-tet scaled rest-matrix kernel for fascia contraction"
```

---

## Task 3: Tissue-to-muscle glue kernel (two-way)

**Files:**
- Modify: `src/muscle/coupleOneWayGmg/solver.py`
- Test: `tests/test_fascia_kernels.py`

XPBD particle↔particle distance constraint. Each pair has one `L` slot. `pos[j]` (muscle) may be shared across pairs → use `wp.atomic_add`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fascia_kernels.py`:

```python
def _run_glue(pos_np, pair_i, pair_j, restlen, inv_mass_np, stiff=1.0e6, iters=500):
    from muscle.coupleOneWayGmg.solver import solve_glue_kernel
    pos = wp.array(pos_np.astype(np.float32), dtype=wp.vec3)
    pi = wp.array(np.asarray(pair_i, np.int32), dtype=wp.int32)
    pj = wp.array(np.asarray(pair_j, np.int32), dtype=wp.int32)
    rl = wp.array(np.asarray(restlen, np.float32), dtype=wp.float32)
    comp = wp.array((1.0 / np.asarray(stiff, np.float32) * np.ones(len(pair_i))).astype(np.float32),
                    dtype=wp.float32)
    L = wp.zeros(len(pair_i), dtype=wp.float32)
    inv_mass = wp.array(inv_mass_np.astype(np.float32), dtype=wp.float32)
    dt = 0.004166
    for _ in range(iters):
        L.zero_()
        wp.launch(solve_glue_kernel, dim=len(pair_i),
                  inputs=[pi, pj, rl, comp, inv_mass, L, dt, 1.0, pos])
    wp.synchronize()
    return pos.numpy()


def test_glue_pulls_to_restlen_symmetric():
    """Equal masses → both move equally, final distance == restlen, midpoint fixed."""
    pos_np = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    out = _run_glue(pos_np, [0], [1], [0.2], np.array([1.0, 1.0]))
    d = np.linalg.norm(out[0] - out[1])
    np.testing.assert_allclose(d, 0.2, rtol=0.02)
    np.testing.assert_allclose((out[0] + out[1]) / 2.0, [0.5, 0, 0], atol=1e-3)


def test_glue_mass_weighted_split():
    """Lighter particle (larger inv_mass) moves farther, in inv_mass ratio."""
    pos_np = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    out = _run_glue(pos_np, [0], [1], [0.0], np.array([2.0, 1.0]))
    disp_i = np.linalg.norm(out[0] - pos_np[0])
    disp_j = np.linalg.norm(out[1] - pos_np[1])
    np.testing.assert_allclose(disp_i / disp_j, 2.0, rtol=0.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fascia_kernels.py -k glue -v`
Expected: FAIL — `solve_glue_kernel` undefined.

- [ ] **Step 3: Implement the kernel**

Add to `src/muscle/coupleOneWayGmg/solver.py`:

```python
@wp.kernel
def solve_glue_kernel(
    pair_i: wp.array(dtype=wp.int32),    # fascia vertex (unique per pair)
    pair_j: wp.array(dtype=wp.int32),    # muscle vertex (may repeat across pairs)
    restlen: wp.array(dtype=wp.float32),
    compliance: wp.array(dtype=wp.float32),  # 1/stiffness
    inv_mass: wp.array(dtype=wp.float32),
    L: wp.array(dtype=wp.float32),
    dt: float,
    omega: float,
    pos: wp.array(dtype=wp.vec3),
):
    """XPBD two-way distance constraint between pos[pair_i] and pos[pair_j]."""
    c = wp.tid()
    i = pair_i[c]
    j = pair_j[c]
    n = pos[i] - pos[j]
    d = wp.length(n)
    if d < 1.0e-9:
        return
    dirn = n / d
    C = d - restlen[c]
    wi = inv_mass[i]
    wj = inv_mass[j]
    alpha_tilde = compliance[c] / (dt * dt)
    denom = wi + wj + alpha_tilde
    if denom < 1.0e-12:
        return
    dlambda = -(C + alpha_tilde * L[c]) / denom
    L[c] = L[c] + dlambda
    f = omega * dlambda
    wp.atomic_add(pos, i, (f * wi) * dirn)
    wp.atomic_add(pos, j, (-f * wj) * dirn)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fascia_kernels.py -k glue -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/muscle/coupleOneWayGmg/solver.py tests/test_fascia_kernels.py
git commit -m "feat(solver): two-way XPBD glue kernel (fascia<->muscle distance)"
```

---

## Task 4: Fascia anchor kernel (ride driven muscle)

**Files:**
- Modify: `src/muscle/coupleOneWayGmg/solver.py`
- Test: `tests/test_fascia_kernels.py`

Hard-overwrite fascia `rigid` points to follow their bound muscle vertex plus the rest offset captured at build time. Mirrors `pintotarget_overwrite_kernel` but the target is a vertex already in `pos`.

- [ ] **Step 1: Write failing test**

Append to `tests/test_fascia_kernels.py`:

```python
def test_anchor_follows_bound_muscle_vertex():
    from muscle.coupleOneWayGmg.solver import fascia_anchor_kernel
    # pos: [muscle0, muscle1, fascia0]; fascia0 anchored to muscle1 with offset (0,0.1,0)
    pos = wp.array(np.array([[0, 0, 0], [2, 0, 0], [9, 9, 9]], dtype=np.float32), dtype=wp.vec3)
    anchor_fascia = wp.array(np.array([2], dtype=np.int32), dtype=wp.int32)
    anchor_muscle = wp.array(np.array([1], dtype=np.int32), dtype=wp.int32)
    rest_offset = wp.array(np.array([[0, 0.1, 0]], dtype=np.float32), dtype=wp.vec3)
    wp.launch(fascia_anchor_kernel, dim=1,
              inputs=[anchor_fascia, anchor_muscle, rest_offset, pos])
    wp.synchronize()
    np.testing.assert_allclose(pos.numpy()[2], [2, 0.1, 0], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fascia_kernels.py -k anchor -v`
Expected: FAIL — `fascia_anchor_kernel` undefined.

- [ ] **Step 3: Implement the kernel**

```python
@wp.kernel
def fascia_anchor_kernel(
    anchor_fascia: wp.array(dtype=wp.int32),
    anchor_muscle: wp.array(dtype=wp.int32),
    rest_offset: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
):
    """pos[anchor_fascia[i]] = pos[anchor_muscle[i]] + rest_offset[i]."""
    i = wp.tid()
    pos[anchor_fascia[i]] = pos[anchor_muscle[i]] + rest_offset[i]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fascia_kernels.py -k anchor -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/muscle/coupleOneWayGmg/solver.py tests/test_fascia_kernels.py
git commit -m "feat(solver): fascia anchor kernel (rigid pts ride driven muscle)"
```

---

## Task 5: OneWaySim `_init_fascia` — merge mesh, build pairs & anchors

**Files:**
- Modify: `src/muscle/coupleOneWayGmg/solver.py` (`OneWaySim.__init__`, new `_init_fascia`)
- Test: `tests/test_fascia_sim_smoke.py`

After the existing muscle init, if `cfg["fascia_path"]` is set, load fascia, append its verts/tets to the merged GPU arrays, rebuild `B`/`alpha` via the scaled kernel for ALL tets, and build glue pairs + anchor bindings on CPU.

- [ ] **Step 1: Write a failing synthetic integration test**

```python
# tests/test_fascia_sim_smoke.py
"""Synthetic merged-mesh fascia smoke test (no real assets, runs on CPU)."""
import numpy as np
import warp as wp

wp.init()


def test_fascia_glue_pulls_muscle_inward(tmp_path, monkeypatch):
    """Two muscle tets + one shrinking fascia tet glued to a muscle surface vert.
    After stepping, the glued muscle vertex moves toward the fascia (inward)."""
    from muscle.coupleOneWayGmg import solver as S

    # Build a tiny merged system by hand via the low-level helper.
    # Muscle: a unit tet at origin (verts 0..3). Fascia: a unit tet offset +x,
    # scaled to shrink. Glue: fascia vert -> nearest muscle surface vert.
    muscle_pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    fascia_pos = muscle_pos + np.array([1.5, 0, 0], dtype=np.float32)
    sim = S.OneWaySim.__new__(S.OneWaySim)  # bypass full init; use helper directly
    state = S.build_merged_state(
        muscle_pos=muscle_pos,
        muscle_tet=np.array([[0, 1, 2, 3]], np.int32),
        fascia_pos=fascia_pos,
        fascia_tet=np.array([[0, 1, 2, 3]], np.int32),
        fascia_rest_scale=0.5,
        muscle_stiffness=1.0e9,
        fascia_stiffness=5000.0,
        glue_fascia=np.array([0], np.int32),       # fascia local vert 0
        glue_muscle=np.array([1], np.int32),       # muscle vert 1 (the +x corner)
        glue_restlen=np.array([0.1], np.float32),
        glue_stiffness=np.array([1000.0], np.float32),
        anchor_fascia=np.zeros(0, np.int32),
        anchor_muscle=np.zeros(0, np.int32),
        total_mass=1.0, device=None,
    )
    p0 = state.pos.numpy().copy()
    S.run_substeps(state, nsubsteps=20, fine_iters=3, glue_iters=4,
                   dt=0.004166, omega=0.2)
    p1 = state.pos.numpy()
    assert not np.isnan(p1).any()
    # muscle vert 1 (index 1) pulled toward fascia vert 0 (index 4) → +x increases
    assert p1[1][0] > p0[1][0]
```

NOTE: `build_merged_state` and `run_substeps` are thin module-level helpers
introduced for testability; `OneWaySim` uses them internally (DRY).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fascia_sim_smoke.py -v`
Expected: FAIL — `build_merged_state`/`run_substeps` undefined.

- [ ] **Step 3: Implement merged-state helpers + `_init_fascia`**

Add to `src/muscle/coupleOneWayGmg/solver.py` a small dataclass and helpers:

```python
from dataclasses import dataclass

@dataclass
class MergedState:
    pos: wp.array
    pos_mid: wp.array
    pos_rest: wp.array
    tet_idx: wp.array
    B: wp.array
    rest_volume: wp.array
    alpha: wp.array
    L: wp.array
    inv_mass: wp.array
    NT: int
    # glue
    glue_fascia: wp.array
    glue_muscle: wp.array
    glue_restlen: wp.array
    glue_compliance: wp.array
    glue_L: wp.array
    n_glue: int
    # anchors
    anchor_fascia: wp.array
    anchor_muscle: wp.array
    anchor_offset: wp.array
    n_anchor: int


def build_merged_state(muscle_pos, muscle_tet, fascia_pos, fascia_tet,
                       fascia_rest_scale, muscle_stiffness, fascia_stiffness,
                       glue_fascia, glue_muscle, glue_restlen, glue_stiffness,
                       anchor_fascia, anchor_muscle, total_mass, device):
    """Concatenate muscle+fascia into one particle system and precompute rest.

    glue_fascia/anchor_fascia are LOCAL fascia vertex indices; they are offset
    by muscle vertex count here. glue_muscle/anchor_muscle are muscle indices.
    """
    nvm = muscle_pos.shape[0]
    pos_np = np.vstack([muscle_pos, fascia_pos]).astype(np.float32)
    tet_np = np.vstack([muscle_tet, fascia_tet + nvm]).astype(np.int32)
    nt_m = muscle_tet.shape[0]
    nt_f = fascia_tet.shape[0]
    NT = nt_m + nt_f
    NV = pos_np.shape[0]

    scale_np = np.ones(NT, dtype=np.float32)
    scale_np[nt_m:] = fascia_rest_scale
    stiff_np = np.empty(NT, dtype=np.float32)
    stiff_np[:nt_m] = muscle_stiffness
    stiff_np[nt_m:] = fascia_stiffness

    pos = wp.array(pos_np, dtype=wp.vec3, device=device)
    pos_mid = wp.clone(pos)
    pos_rest = wp.array(pos_np.copy(), dtype=wp.vec3, device=device)
    tet_idx = wp.array(tet_np, dtype=wp.vec4i, device=device)
    B = wp.zeros(NT, dtype=wp.mat33, device=device)
    rest_volume = wp.zeros(NT, dtype=wp.float32, device=device)
    alpha = wp.zeros(NT, dtype=wp.float32, device=device)
    L = wp.zeros(NT, dtype=wp.float32, device=device)
    scale = wp.array(scale_np, dtype=wp.float32, device=device)
    stiff = wp.array(stiff_np, dtype=wp.float32, device=device)
    wp.launch(precompute_rest_scaled_kernel, dim=NT,
              inputs=[pos_rest, tet_idx, scale, stiff, B, rest_volume, alpha],
              device=device)

    V0 = rest_volume.numpy()
    density = total_mass / float(V0.sum())
    mass_v = np.zeros(NV, dtype=np.float32)
    per_tet_mass = density * V0 / 4.0
    for k in range(4):
        np.add.at(mass_v, tet_np[:, k], per_tet_mass)
    inv_mass = wp.array((1.0 / mass_v).astype(np.float32), dtype=wp.float32, device=device)

    gf = (np.asarray(glue_fascia, np.int32) + nvm)
    gm = np.asarray(glue_muscle, np.int32)
    n_glue = len(gf)
    af = (np.asarray(anchor_fascia, np.int32) + nvm)
    am = np.asarray(anchor_muscle, np.int32)
    offset_np = (pos_np[af] - pos_np[am]).astype(np.float32) if len(af) else np.zeros((0, 3), np.float32)

    return MergedState(
        pos=pos, pos_mid=pos_mid, pos_rest=pos_rest, tet_idx=tet_idx,
        B=B, rest_volume=rest_volume, alpha=alpha, L=L, inv_mass=inv_mass, NT=NT,
        glue_fascia=wp.array(gf, dtype=wp.int32, device=device),
        glue_muscle=wp.array(gm, dtype=wp.int32, device=device),
        glue_restlen=wp.array(np.asarray(glue_restlen, np.float32), dtype=wp.float32, device=device),
        glue_compliance=wp.array((1.0 / np.asarray(glue_stiffness, np.float32)).astype(np.float32),
                                 dtype=wp.float32, device=device),
        glue_L=wp.zeros(max(n_glue, 1), dtype=wp.float32, device=device),
        n_glue=n_glue,
        anchor_fascia=wp.array(af, dtype=wp.int32, device=device),
        anchor_muscle=wp.array(am, dtype=wp.int32, device=device),
        anchor_offset=wp.array(offset_np, dtype=wp.vec3, device=device),
        n_anchor=len(af),
    )


def apply_anchor(state, device=None):
    if state.n_anchor:
        wp.launch(fascia_anchor_kernel, dim=state.n_anchor,
                  inputs=[state.anchor_fascia, state.anchor_muscle,
                          state.anchor_offset, state.pos], device=device)


def solve_glue(state, dt, omega, iters, device=None):
    for _ in range(iters):
        state.glue_L.zero_()
        if state.n_glue:
            wp.launch(solve_glue_kernel, dim=state.n_glue,
                      inputs=[state.glue_fascia, state.glue_muscle,
                              state.glue_restlen, state.glue_compliance,
                              state.inv_mass, state.glue_L, dt, omega, state.pos],
                      device=device)


def run_substeps(state, nsubsteps, fine_iters, glue_iters, dt, omega, device=None):
    """Standalone (no multigrid, no pin) loop — used by tests."""
    for _ in range(nsubsteps):
        apply_anchor(state, device)
        state.L.zero_()
        for _ in range(fine_iters):
            wp.copy(state.pos_mid, state.pos)
            wp.launch(arap_svd_kernel, dim=state.NT,
                      inputs=[state.pos_mid, state.tet_idx, state.B,
                              state.inv_mass, state.alpha, state.L, dt, omega, state.pos],
                      device=device)
        solve_glue(state, dt, omega, glue_iters, device)
```

Then add `OneWaySim._init_fascia` (called at end of `__init__` when `cfg.get("fascia_path")`), which: loads fascia via `GeoFile`, reads attrs/groups, runs cKDTree pairing of `tissue_inner_points`+`tissue_bone_points` against muscle `surface_points`, clamps restlen to `min(tissuetomuscledistance, measured_rest_dist)`, binds `rigid` to nearest muscle vertex, calls `build_merged_state`, and swaps the sim's GPU arrays to the merged ones. Code:

```python
    def _init_fascia(self, rest_muscle_pos):
        from scipy.spatial import cKDTree
        import logging
        log = logging.getLogger(__name__)

        _check_unzipped(self.fascia_path)
        fg = GeoFile(self.fascia_path)
        fascia_pos = fg.positions
        fascia_tet = fg.tet_indices

        # Muscle surface vertices to pair against.
        mrest = GeoFile(self.rest_path)
        surf = mrest.point_group("surface_points")  # muscle surface vert ids
        surf_tree = cKDTree(rest_muscle_pos[surf])

        def pair(group_name, dist_attr, stiff_attr):
            if group_name not in fg._groups:
                return (np.zeros(0, np.int32),) * 1 + (np.zeros(0, np.int32),
                        np.zeros(0, np.float32), np.zeros(0, np.float32))
            fids = fg.point_group(group_name)
            d, loc = surf_tree.query(fascia_pos[fids], k=1)
            mids = surf[loc].astype(np.int32)
            want = fg.point_attr(dist_attr)[fids].astype(np.float32)
            restlen = np.minimum(want, d.astype(np.float32))  # don't yank distant pts
            stiff = fg.point_attr(stiff_attr)[fids].astype(np.float32)
            return fids.astype(np.int32), mids, restlen, stiff

        gi_f, gi_m, gi_r, gi_s = pair("tissue_inner_points",
                                      "tissuetomuscledistance", "tissuetomusclestiffness")
        gb_f, gb_m, gb_r, gb_s = pair("tissue_bone_points",
                                      "tissuetobonedistance", "tissuetobonestiffness")
        glue_fascia = np.concatenate([gi_f, gb_f])
        glue_muscle = np.concatenate([gi_m, gb_m])
        glue_restlen = np.concatenate([gi_r, gb_r])
        glue_stiff = np.concatenate([gi_s, gb_s])

        # Rigid anchors ride nearest muscle vertex (any muscle vert, not just surface).
        if "rigid" in fg._groups:
            rid = fg.point_group("rigid")
            full_tree = cKDTree(rest_muscle_pos)
            _, aloc = full_tree.query(fascia_pos[rid], k=1)
            anchor_fascia = rid.astype(np.int32)
            anchor_muscle = aloc.astype(np.int32)
        else:
            anchor_fascia = np.zeros(0, np.int32)
            anchor_muscle = np.zeros(0, np.int32)

        log.info(f"fascia: {fascia_pos.shape[0]} verts, {fascia_tet.shape[0]} tets; "
                 f"glue pairs={len(glue_fascia)} (inner={len(gi_f)}, bone={len(gb_f)}); "
                 f"anchors={len(anchor_fascia)}; rest_scale={self.fascia_rest_scale}")

        self.nvm = rest_muscle_pos.shape[0]
        self.merged = build_merged_state(
            muscle_pos=rest_muscle_pos, muscle_tet=mrest.tet_indices,
            fascia_pos=fascia_pos, fascia_tet=fascia_tet,
            fascia_rest_scale=self.fascia_rest_scale,
            muscle_stiffness=self.mu, fascia_stiffness=float(fg.point_attr("solidshapestiffness").mean()),
            glue_fascia=glue_fascia, glue_muscle=glue_muscle,
            glue_restlen=glue_restlen, glue_stiffness=glue_stiff,
            anchor_fascia=anchor_fascia, anchor_muscle=anchor_muscle,
            total_mass=self.total_mass, device=self._device)
        # Point the sim's fine-level arrays at the merged system.
        self.pos = self.merged.pos
        self.pos_mid = self.merged.pos_mid
        self.tet_idx = self.merged.tet_idx
        self.B = self.merged.B
        self.alpha = self.merged.alpha
        self.L = self.merged.L
        self.inv_mass = self.merged.inv_mass
        self.NT = self.merged.NT
        self.NV = self.pos.shape[0]
```

In `OneWaySim.__init__`, read the new config keys near the top:
```python
        self.fascia_path = cfg.get("fascia_path")
        if self.fascia_path is not None and not Path(self.fascia_path).is_absolute():
            self.fascia_path = str(Path.cwd() / self.fascia_path)
        self.fascia_rest_scale = float(cfg.get("fascia_rest_scale", 0.95))
        self.fascia_glue_iters = int(cfg.get("fascia_glue_iterations", self.fine_iters))
```
and at the very end of `__init__`, after multigrid init:
```python
        if self.fascia_path:
            self._init_fascia(rest.positions.copy())
```
Also add `load_config` keys: extend its path-resolution loop to include `"fascia_path"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fascia_sim_smoke.py -v`
Expected: PASS (muscle vert pulled toward fascia).

- [ ] **Step 5: Run full kernel + geo suites for no regression**

Run: `uv run pytest tests/test_fascia_kernels.py tests/test_fascia_geo_io.py tests/test_lion_kernels.py tests/test_geo_io.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/muscle/coupleOneWayGmg/solver.py tests/test_fascia_sim_smoke.py
git commit -m "feat(solver): merge fascia into OneWaySim (pairing, anchors, merged state)"
```

---

## Task 6: Wire fascia into `OneWaySim.substep` / `step_frame`

**Files:**
- Modify: `src/muscle/coupleOneWayGmg/solver.py` (`substep`)
- Test: `tests/test_fascia_sim_smoke.py`

Multigrid stays muscle-only: its restrict/prolong launch dims and barycentric maps cover muscle verts only (indices `< self.nvm`). Fascia is fine-level + glue. After fine ARAP, run glue iters. Anchor is applied each substep (prolong/pin can move muscle, anchor re-rides).

- [ ] **Step 1: Write failing test (fascia path active in substep)**

Append to `tests/test_fascia_sim_smoke.py`:

```python
def test_substep_runs_glue_when_fascia_present(monkeypatch):
    """OneWaySim.substep applies glue iters when a merged state exists."""
    from muscle.coupleOneWayGmg import solver as S
    calls = {"glue": 0}
    real_solve_glue = S.solve_glue

    def spy(state, dt, omega, iters, device=None):
        calls["glue"] += 1
        return real_solve_glue(state, dt, omega, iters, device)

    monkeypatch.setattr(S, "solve_glue", spy)

    sim = S.OneWaySim.__new__(S.OneWaySim)
    # Minimal fields substep touches in the no-multigrid + fascia branch:
    sim.use_multigrid = False
    sim.fine_iters = 2
    sim.fascia_glue_iters = 3
    sim.dt = 0.004166
    sim.omega = 0.2
    sim._device = None
    sim.merged = S.build_merged_state(
        muscle_pos=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32),
        muscle_tet=np.array([[0, 1, 2, 3]], np.int32),
        fascia_pos=np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1]], np.float32),
        fascia_tet=np.array([[0, 1, 2, 3]], np.int32),
        fascia_rest_scale=0.9, muscle_stiffness=1e9, fascia_stiffness=5000.0,
        glue_fascia=np.array([0], np.int32), glue_muscle=np.array([1], np.int32),
        glue_restlen=np.array([0.1], np.float32), glue_stiffness=np.array([1000.0], np.float32),
        anchor_fascia=np.zeros(0, np.int32), anchor_muscle=np.zeros(0, np.int32),
        total_mass=1.0, device=None)
    sim.pos = sim.merged.pos
    sim.pos_mid = sim.merged.pos_mid
    sim.tet_idx = sim.merged.tet_idx
    sim.B = sim.merged.B
    sim.alpha = sim.merged.alpha
    sim.L = sim.merged.L
    sim.inv_mass = sim.merged.inv_mass
    sim.NT = sim.merged.NT

    sim.substep()
    assert calls["glue"] == 1
    assert not np.isnan(sim.pos.numpy()).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fascia_sim_smoke.py::test_substep_runs_glue_when_fascia_present -v`
Expected: FAIL — current `substep` has no fascia branch (`self.merged` / `apply_anchor` / glue not invoked; AttributeError or glue count 0).

- [ ] **Step 3: Modify `substep`**

Guard fascia work behind `getattr(self, "merged", None)`. Insert anchor before the relax loop and glue after the fine ARAP loop:

```python
    def substep(self) -> None:
        has_fascia = getattr(self, "merged", None) is not None
        if self.use_multigrid:
            # ... existing V-cycle, unchanged ...
            self.apply_pintotarget()
        if has_fascia:
            apply_anchor(self.merged, self._device)
        self.L.zero_()
        for _ in range(self.fine_iters):
            wp.copy(self.pos_mid, self.pos)
            wp.launch(arap_svd_kernel, dim=self.NT,
                      inputs=[self.pos_mid, self.tet_idx, self.B,
                              self.inv_mass, self.alpha, self.L,
                              self.dt, self.omega, self.pos],
                      device=self._device)
        if has_fascia:
            solve_glue(self.merged, self.dt, self.omega, self.fascia_glue_iters, self._device)
```

Also in `step_frame`, apply anchor right after the per-frame muscle pin (so frame-0 fascia starts attached):
```python
    def step_frame(self, frame: int) -> None:
        self.load_frame_targets(frame)
        self.apply_pintotarget()
        if getattr(self, "merged", None) is not None:
            apply_anchor(self.merged, self._device)
        for _ in range(self.nsubsteps):
            self.substep()
```

Multigrid note: the V-cycle launches (`restrict_direct_kernel` dim=`coarse_NV`, `prolong_kernel` dim=`self.NV`) must use the MUSCLE vertex count, not the merged total. In `_init_fascia`, after building merged state, restore the multigrid fine dim by keeping a separate `self.mg_fine_nv = self.nvm` and changing the `prolong_kernel` launch dim from `self.NV` to `getattr(self, "mg_fine_nv", self.NV)`. (cage maps were built against muscle rest positions only.) Apply that one-line dim change in the multigrid block.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fascia_sim_smoke.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/muscle/coupleOneWayGmg/solver.py tests/test_fascia_sim_smoke.py
git commit -m "feat(solver): fascia anchor+glue wired into substep/step_frame"
```

---

## Task 7: Config + dual-layer export + humanV2 smoke

**Files:**
- Create: `data/humanV2/config/humanV2_fascia.json`
- Modify: `src/muscle/coupleOneWayGmg/example.py`
- Test: `tests/test_fascia_sim_smoke.py`

- [ ] **Step 1: Create the config**

```json
{
    "start_frame": 1,
    "end_frame": 200,
    "geo_dir": "data/humanV2/model",
    "cons_dir": "data/humanV2/model/cons",
    "rest_path": "data/humanV2/model/muscle/restpos.geo",
    "coarse_rest_path": "data/humanV2/model/muscle/restpos_lowres.geo",
    "fascia_path": "data/humanV2/model/fascia/fascia.geo",
    "fascia_rest_scale": 0.95,
    "fascia_glue_iterations": 2,
    "use_pintotarget": true,
    "quasi_static": true,
    "use_external_constraints": true,
    "use_multigrid": true,
    "omega": 0.1,
    "mu": 1e9,
    "delta_t": 0.04166,
    "damping_coeff": 1.0,
    "gravity": [0.0, 0.0, 0.0],
    "total_mass": 70.0,
    "mass_density": -1.0,
    "coarse_iterations": 5,
    "fine_iterations": 2,
    "nsubsteps": 10
}
```

- [ ] **Step 2: Write a (skipped-without-assets) humanV2 smoke test**

Append to `tests/test_fascia_sim_smoke.py`:

```python
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
HV2 = ROOT / "data" / "humanV2" / "model"
FASCIA_CFG = ROOT / "data" / "humanV2" / "config" / "humanV2_fascia.json"

skip_no_hv2 = pytest.mark.skipif(
    not (HV2 / "muscle" / "restpos.geo").exists()
    or not (HV2 / "fascia" / "fascia.geo").exists(),
    reason="humanV2 model.zip not extracted / fascia.geo not copied",
)


@skip_no_hv2
def test_humanV2_fascia_one_frame_no_nan():
    from muscle.coupleOneWayGmg.solver import OneWaySim, load_config
    cfg = load_config(str(FASCIA_CFG))
    cfg["end_frame"] = cfg["start_frame"]
    sim = OneWaySim(cfg)            # GPU if available; CPU fallback fine for 1 frame
    assert sim.NV > 157369         # muscle + fascia merged
    assert sim.merged.n_glue > 0
    rest = sim.positions_np().copy()
    sim.step_frame(sim.start_frame)
    after = sim.positions_np()
    assert not np.isnan(after).any() and not np.isinf(after).any()
    # fascia verts (index >= nvm) moved inward (shrink) — mean radial drift < 0
    fascia = after[sim.nvm:]
    fascia0 = rest[sim.nvm:]
    assert np.linalg.norm(fascia - fascia0, axis=1).mean() > 1e-5
```

- [ ] **Step 3: Run it (will skip if assets absent; do not block on GPU)**

Run: `uv run pytest tests/test_fascia_sim_smoke.py -v`
Expected: synthetic tests PASS; `test_humanV2_*` PASS if assets present else SKIP.

- [ ] **Step 4: Add dual-layer export flag to example.py**

In `main()`, add `--fascia` handling so muscle and fascia surfaces export separately (split by `sim.nvm`):

```python
    ap.add_argument("--split_fascia", action="store_true",
                    help="export muscle and fascia as separate PLY/USD streams")
    # ... after sim build ...
    nvm = getattr(sim, "nvm", sim.NV)
    if args.split_fascia and nvm < sim.NV:
        tets = sim.tet_idx.numpy()
        muscle_tets = tets[(tets < nvm).all(axis=1)]
        fascia_tets = tets[(tets >= nvm).all(axis=1)]
        muscle_faces = _vectorized_surface_tris(muscle_tets)
        fascia_faces = _vectorized_surface_tris(fascia_tets)
        # per frame:
        #   save_ply(mesh_out/f"muscle_{frame:04d}.ply", sim.positions_np(), muscle_faces)
        #   save_ply(mesh_out/f"fascia_{frame:04d}.ply", sim.positions_np(), fascia_faces)
```
Wire those two `save_ply` calls into the existing per-frame export branch when `args.split_fascia`.

- [ ] **Step 5: Commit**

```bash
git add data/humanV2/config/humanV2_fascia.json src/muscle/coupleOneWayGmg/example.py tests/test_fascia_sim_smoke.py
git commit -m "feat: humanV2 fascia config, dual-layer export, integration smoke test"
```

---

## Task 8: End-to-end validation (manual — needs GPU + assets)

**Files:** none (validation only). Produces evidence the squeeze works.

- [ ] **Step 1: Extract assets + copy fascia**

```bash
(cd data/humanV2 && unzip -o model.zip)
mkdir -p data/humanV2/model/fascia && cp ~/Dev/fascia.geo data/humanV2/model/fascia/fascia.geo
```

- [ ] **Step 2: Run fascia OFF (baseline) and ON over a short range**

```bash
# OFF: rest_scale 1.0 (no contraction)
uv run python -m muscle.coupleOneWayGmg.example \
  --config data/humanV2/config/humanV2_fascia.json \
  --out_dir result/humanV2_fascia_off --end_frame 30 --split_fascia
# edit a temp config or pass override: set fascia_rest_scale 1.0 for OFF, 0.95 for ON
uv run python -m muscle.coupleOneWayGmg.example \
  --config data/humanV2/config/humanV2_fascia.json \
  --out_dir result/humanV2_fascia_on --end_frame 30 --split_fascia
```
Expected: completes without NaN; prints `NV=` larger than muscle-only (≈172,717).

- [ ] **Step 3: Quantify the squeeze**

Measure mean radial distance of muscle surface verts to the limb's long axis at a fixed frame, OFF vs ON. ON should be smaller (muscles pulled tighter). Write a tiny throwaway script reading the muscle PLYs; record the number in the commit message / a note. Per project default, also bake a USD for visual inspection (not mp4):
```bash
MM_EXPORT_USD=1 uv run python -m muscle.coupleOneWayGmg.example \
  --config data/humanV2/config/humanV2_fascia.json --out_dir result/humanV2_fascia_on --end_frame 30
```

- [ ] **Step 4: Final verification run of the whole test suite**

Run: `uv run pytest tests/ -v`
Expected: all PASS or SKIP (no failures). Confirms lion path intact + fascia path green.

- [ ] **Step 5: Commit any validation notes / finish branch**

```bash
git add -A && git commit -m "docs: fascia squeeze validation notes (OFF vs ON radial tightening)"
```
Then use the `superpowers:finishing-a-development-branch` skill to decide merge/PR.

---

## Self-Review

**Spec coverage:**
- Merged particle system → Task 5 (`build_merged_state`). ✓
- Scaled-B contraction → Task 2. ✓
- Per-tet stiffness → Task 2 (`stiffness[]` in scaled kernel) + Task 5 (muscle=mu, fascia=solidshapestiffness). ✓
- Tissue-to-muscle glue (new kernel) → Task 3 + pairing Task 5. ✓
- Anchor / ride driven muscle → Task 4 + binding Task 5 + wiring Task 6. ✓
- geo_io tets/attrs/groups → Task 1. ✓
- Frame loop / multigrid muscle-only → Task 6. ✓
- Config + USD/PLY dual export → Task 7. ✓
- Error handling (missing assets, empty groups, restlen clamp, degenerate tets) → `_check_unzipped` (Task 5), empty-group guards (Task 5 `pair`/anchor branches), `np.minimum` clamp (Task 5), ARAP det handled by inverse (existing). ✓
- Validation OFF vs ON → Task 8. ✓

**Divergence from spec (intentional):** spec said "precompute_rest_kernel gains a param"; plan instead ADDS `precompute_rest_scaled_kernel` to avoid breaking the lion `test_lion_kernels.py` signature tests. Same requirement met, better isolation.

**Placeholder scan:** no TBD/TODO; every code step shows full code. ✓

**Type/name consistency:** `MergedState` fields, `build_merged_state`/`run_substeps`/`apply_anchor`/`solve_glue`, kernel names (`precompute_rest_scaled_kernel`, `solve_glue_kernel`, `fascia_anchor_kernel`), `GeoFile.point_attr`/`point_group`/`has_point_attr`, `OneWaySim.nvm`/`merged`/`fascia_rest_scale`/`fascia_glue_iters` used consistently across Tasks 1–8. ✓
