# Unified XPBD Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `MuscleSim`, `OneWaySim`, and the Gaia `MassSpring` U-drop into one `XpbdEngine` (typed-constraint dispatch + collision-extras hook + a configurable substep *program*), with output **bit-wise identical** to the pre-merge code at every stage.

**Architecture:** `XpbdEngine` is a base class defined in-place in `muscle_warp.py` holding the shared state buffers, the typed-constraint registry + `_dispatch_constraints`, jacobi/colored-GS style, the `constraint_solvers[]` collision hook, the `integrate`/`update_velocities`/`apply_pintotarget`/`repair_inverted_tets` primitives, and a **program runner** (`step()` walks an ordered list of bound-method phases). The three sims subclass it, keep their own loaders/kernels, and each define a `build_program()` whose default reproduces its current exact substep order. Kernels stay in their current modules. Bit-wise identity is enforced by a golden-frame regression harness — every stage either passes `np.array_equal` against goldens captured from the pre-refactor code, or it does not land.

**Tech Stack:** Python, NVIDIA Warp (`wp.array`, `wp.launch`, `cuda:0`), NumPy, pytest. XPBD constraint projection.

---

## File structure

- `src/muscle/muscle_warp.py` — **modify.** Add `class XpbdEngine` (extracted base). `MuscleSim(XpbdEngine)`. Add `integrate` mode flag, `update_velocities` damped variant, new constraint-type cases (`DISTANCE`, hard `TETVOLUME`, `INVERSION`) in `_dispatch_constraints`, and the program runner `step()`/`run_program()`.
- `src/muscle/coupleOneWayGmg/solver.py` — **modify.** `OneWaySim(XpbdEngine)`; register OneWay ARAP/volume/glue/multigrid as phases; default program == current `substep()`.
- `src/muscle/collision/udcf/massspring_sim.py` — **create.** `MassSpringSim(XpbdEngine)` wrapping the validated MassSpring material as a program (integrate(gravity) → spring/vol_hard/boundary/inversion/dcd → update_vel(damped)).
- `src/muscle/collision/udcf/example/u_drop.py` — **modify.** Drive `MassSpringSim` instead of hand-launching kernels.
- `tests/golden/capture_engine_goldens.py` — **create.** Runs each sim N frames on current code, writes `output/golden/<sim>.npz` (pos+vel per frame). Gitignored artifacts.
- `tests/test_engine_parity.py` — **create.** Loads `output/golden/<sim>.npz`, re-runs the (refactored) sim, asserts `np.array_equal` per frame. Skips if golden npz or data absent.

> **Golden artifacts are local, not committed** (positions×frames are large). They live under `output/golden/` (already gitignored via `output/`). Capture once at Stage 0 on the pre-refactor HEAD; do not delete between stages.

---

## Stage 0 — Golden regression harness

### Task 0: Capture-and-compare harness

**Files:**
- Create: `tests/golden/capture_engine_goldens.py`
- Create: `tests/test_engine_parity.py`

- [ ] **Step 1: Write the golden capture script**

```python
# tests/golden/capture_engine_goldens.py
"""Capture per-frame pos/vel goldens from the CURRENT (pre-refactor) sims.

Run ONCE on pre-refactor HEAD:
    PYTHONPATH=src python tests/golden/capture_engine_goldens.py

Writes output/golden/<sim>.npz with arrays pos[F,N,3] (+ vel where dynamic).
These are the bit-wise reference for tests/test_engine_parity.py.
"""
import json
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "golden"
DEVICE = "cuda:0"
N_FRAMES = 5


def _save(name, pos_list, vel_list=None):
    OUT.mkdir(parents=True, exist_ok=True)
    d = {"pos": np.stack(pos_list).astype(np.float32)}
    if vel_list is not None:
        d["vel"] = np.stack(vel_list).astype(np.float32)
    np.savez(OUT / f"{name}.npz", **d)
    print(f"  wrote {OUT / f'{name}.npz'}  pos{d['pos'].shape}")


def capture_oneway():
    from muscle.coupleOneWayGmg.solver import OneWaySim
    cfg = json.load(open(ROOT / "data/lion/config/lion_nomg.json"))
    cfg["geo_dir"] = str(ROOT / "data/lion/model")
    cfg["end_frame"] = N_FRAMES
    sim = OneWaySim(cfg, device=DEVICE)
    pos = []
    for f in range(sim.start_frame, sim.start_frame + N_FRAMES):
        sim.step_frame(f)
        pos.append(sim.positions_np().copy())
    _save("oneway_lion", pos)


def capture_massspring():
    import numpy as _np
    from muscle.collision.udcf.massspring import MassSpringTet, MassSpringParams
    from muscle.collision.udcf.solver import UdcfDcdCollider, UdcfParams
    from muscle.collision.udcf.kernel import box_boundary_kernel
    from muscle.util.geo_io import GeoFile
    from muscle.util.mesh_io import build_surface_tris
    g = GeoFile(str(ROOT / "data/collisionTest/U.geo"))
    verts = _np.asarray(g.positions, _np.float32).copy(); verts[:, 1] += 0.2
    tets = _np.asarray(g.tet_indices, _np.int32)
    dt = 1.0 / 60.0; nsub = 20
    ms = MassSpringTet(verts, tets, dt / nsub,
                       params=MassSpringParams(), device=DEVICE)
    v0 = _np.zeros((verts.shape[0], 3), _np.float32); v0[:, 1] = -5.0
    ms.vel.assign(v0)
    surf = build_surface_tris(tets, verts).astype(_np.int32)
    mp = max(65536, surf.shape[0] * (surf.shape[0] - 1) // 2)
    dv = _np.array([[1e4, 1e4, 1e4], [1e4 + 1, 1e4, 1e4], [1e4, 1e4, 1e4 + 1]], _np.float32)
    dcd = UdcfDcdCollider(ms.pos, ms.inv_mass, surface_tris=surf,
                          static_verts=dv, static_tris=_np.array([[0, 1, 2]], _np.int32),
                          params=UdcfParams(allow_self=True, max_pairs=mp, thickness=1e-3,
                                            jacobi_avg=False, gauss_seidel=True, max_push=0.0),
                          device=DEVICE)
    lo = wp.vec3(-2.0, 0.0, -2.0); hi = wp.vec3(2.0, 5.0e4, 2.0)
    pos, vel = [], []
    for _f in range(N_FRAMES):
        for _ in range(nsub):
            ms.integrate(); ms.solve_material()
            wp.launch(box_boundary_kernel, dim=verts.shape[0],
                      inputs=[ms.pos, ms.pprev, lo, hi, wp.float32(0.5), wp.float32(0.4)],
                      device=DEVICE)
            ms.solve_inversion(); dcd.solve(0); ms.update_velocities()
        pos.append(ms.pos.numpy().copy()); vel.append(ms.vel.numpy().copy())
    _save("massspring_u", pos, vel)


if __name__ == "__main__":
    wp.init()
    for name, fn in [("oneway_lion", capture_oneway),
                     ("massspring_u", capture_massspring)]:
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  SKIP {name}: {e}")
```

- [ ] **Step 2: Write the parity test**

```python
# tests/test_engine_parity.py
"""Bit-wise parity: refactored sims must match output/golden/<sim>.npz exactly."""
import json
from pathlib import Path

import numpy as np
import pytest
import warp as wp

wp.init()
ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "output" / "golden"
DEVICE = "cuda:0"


def _need(name):
    p = GOLD / f"{name}.npz"
    if not p.exists():
        pytest.skip(f"golden {p} missing — run capture_engine_goldens.py on pre-refactor HEAD")
    return np.load(p)


def test_oneway_lion_bitwise():
    if not (ROOT / "data/lion/model/restpos.geo").exists():
        pytest.skip("lion data absent")
    g = _need("oneway_lion")
    from muscle.coupleOneWayGmg.solver import OneWaySim
    cfg = json.load(open(ROOT / "data/lion/config/lion_nomg.json"))
    cfg["geo_dir"] = str(ROOT / "data/lion/model"); cfg["end_frame"] = g["pos"].shape[0]
    sim = OneWaySim(cfg, device=DEVICE)
    for i, f in enumerate(range(sim.start_frame, sim.start_frame + g["pos"].shape[0])):
        sim.step_frame(f)
        assert np.array_equal(sim.positions_np(), g["pos"][i]), f"oneway frame {i} diverged"


def test_massspring_u_bitwise():
    if not (ROOT / "data/collisionTest/U.geo").exists():
        pytest.skip("U.geo absent")
    g = _need("massspring_u")
    from muscle.collision.udcf.massspring_sim import MassSpringSim  # Stage 3
    sim = MassSpringSim.u_drop(str(ROOT / "data/collisionTest/U.geo"), device=DEVICE)
    for i in range(g["pos"].shape[0]):
        sim.step()
        assert np.array_equal(sim.pos.numpy(), g["pos"][i]), f"massspring frame {i} diverged"
        assert np.array_equal(sim.vel.numpy(), g["vel"][i]), f"massspring vel {i} diverged"
```

- [ ] **Step 3: Capture goldens on current code**

Run: `PYTHONPATH=src python tests/golden/capture_engine_goldens.py`
Expected: prints `wrote output/golden/oneway_lion.npz pos(5, …, 3)` and `…massspring_u.npz`. (Either line may print `SKIP` if that dataset isn't extracted — capture what's available.)

- [ ] **Step 4: Verify the OneWay parity test passes against current code**

Run: `PYTHONPATH=src python -m pytest tests/test_engine_parity.py::test_oneway_lion_bitwise -v`
Expected: PASS (sim unchanged, matches its own golden). `test_massspring_u_bitwise` errors on missing `massspring_sim` import — that's expected until Stage 3; leave it.

- [ ] **Step 5: Commit**

```bash
git add tests/golden/capture_engine_goldens.py tests/test_engine_parity.py
git commit -m "test(engine): bit-wise golden harness for unified-engine merge"
```

---

## Stage 1 — Extract `XpbdEngine`; MuscleSim subclasses it

Goal: pure restructure. `MuscleSim` behavior byte-identical. The golden for MuscleSim is its humanV2 run; capture is added here (kept separate from Stage 0 since it needs the humanV2 asset).

### Task 1: Capture the MuscleSim golden

**Files:**
- Modify: `tests/golden/capture_engine_goldens.py`
- Modify: `tests/test_engine_parity.py`

- [ ] **Step 1: Add a MuscleSim capture function**

Add to `capture_engine_goldens.py`, and call it in `__main__`. Use the existing humanV2 muscle config (the one `src/muscle/example.py` loads); fill the exact config path used by that script:

```python
def capture_musclesim():
    from muscle.muscle_warp import MuscleSim
    from muscle.util.util import load_config
    cfg = load_config(str(ROOT / "data/humanV2/config/humanV2_gmg.json"))
    sim = MuscleSim(cfg)            # device per cfg; uses cfg.device or default
    pos = []
    for _ in range(N_FRAMES):
        sim.step()
        pos.append(sim.pos.numpy().copy())
    _save("musclesim_humanv2", pos)
```

> If `MuscleSim` requires a different config/loader than the humanV2 GMG json, use the exact config `src/muscle/example.py`'s `load_config` consumes — match that script's invocation so the golden reflects a real run.

- [ ] **Step 2: Add the MuscleSim parity test**

```python
def test_musclesim_humanv2_bitwise():
    cfg_path = ROOT / "data/humanV2/config/humanV2_gmg.json"
    if not cfg_path.exists():
        pytest.skip("humanV2 config absent")
    g = _need("musclesim_humanv2")
    from muscle.muscle_warp import MuscleSim
    from muscle.util.util import load_config
    sim = MuscleSim(load_config(str(cfg_path)))
    for i in range(g["pos"].shape[0]):
        sim.step()
        assert np.array_equal(sim.pos.numpy(), g["pos"][i]), f"musclesim frame {i} diverged"
```

- [ ] **Step 3: Capture + verify against current MuscleSim**

Run: `PYTHONPATH=src python tests/golden/capture_engine_goldens.py` then
`PYTHONPATH=src python -m pytest tests/test_engine_parity.py::test_musclesim_humanv2_bitwise -v`
Expected: capture writes `musclesim_humanv2.npz`; test PASSES (unchanged sim vs own golden).

- [ ] **Step 4: Commit**

```bash
git add tests/golden/capture_engine_goldens.py tests/test_engine_parity.py
git commit -m "test(engine): add MuscleSim humanV2 golden"
```

### Task 2: Introduce `XpbdEngine` base by inheritance-no-op

Strategy: create `XpbdEngine` as an **empty marker base first**, make `MuscleSim` inherit it, confirm identical. Then move members up one cluster at a time, re-running the golden after each move. Moving (not rewriting) guarantees bit-identity.

**Files:**
- Modify: `src/muscle/muscle_warp.py` (around `class MuscleSim`, line ~3548)

- [ ] **Step 1: Add empty base and inherit**

```python
class XpbdEngine(ConstraintBuilderMixin):
    """Shared XPBD substep engine: state buffers, typed-constraint dispatch,
    jacobi/colored-GS style, collision-extras hook, and a program runner.
    Subclasses own their loaders/kernels and define build_program()."""
    pass


class MuscleSim(XpbdEngine):
    ...  # body unchanged
```

- [ ] **Step 2: Run MuscleSim golden**

Run: `PYTHONPATH=src python -m pytest tests/test_engine_parity.py::test_musclesim_humanv2_bitwise -v`
Expected: PASS (inheritance is a no-op).

- [ ] **Step 3: Commit**

```bash
git add src/muscle/muscle_warp.py
git commit -m "refactor(engine): introduce empty XpbdEngine base under MuscleSim"
```

### Task 3: Move shared primitives to `XpbdEngine`

Move these **verbatim** from `MuscleSim` into `XpbdEngine` (cut from subclass, paste into base — no edits to bodies): `_dispatch_constraints`, `solve_constraints`, `solve_extras`, `post_smooth`, `apply_dP`, `clear`, `integrate`, `update_velocities`, `apply_pintotarget`, `freeze_near_inverted`, `repair_inverted_tets`. Leave loaders, `build_constraints`, `_precompute_rest`, fiber/Millard, renderer, and `step()` on `MuscleSim`.

- [ ] **Step 1: Move the methods**

Cut each listed method body from `MuscleSim` and paste into `XpbdEngine`, preserving indentation and references to `self.*` exactly.

- [ ] **Step 2: Run MuscleSim golden**

Run: `PYTHONPATH=src python -m pytest tests/test_engine_parity.py::test_musclesim_humanv2_bitwise -v`
Expected: PASS. If FAIL → a moved method referenced a `MuscleSim`-only attribute that should also move; move it and retry. **Do not relax the test.**

- [ ] **Step 3: Commit**

```bash
git add src/muscle/muscle_warp.py
git commit -m "refactor(engine): move shared XPBD primitives to XpbdEngine"
```

### Task 4: Add the program runner; `step()` runs a default program

**Files:**
- Modify: `src/muscle/muscle_warp.py`

- [ ] **Step 1: Add `run_program` to `XpbdEngine`**

```python
    def run_program(self):
        """Execute self.program (list of zero-arg bound-method phases) once
        per substep for cfg.num_substeps. Subclasses set self.program in
        build_program()."""
        for _sub in range(self.cfg.num_substeps):
            for phase in self.program:
                phase()
```

- [ ] **Step 2: Build MuscleSim's default program mirroring current `step()`**

Replace `MuscleSim.step()` body with a program built to reproduce the **exact** current order (per `muscle_warp.py:3617-3643`). Phases are bound methods / tiny lambdas closing over current calls:

```python
    def build_program(self):
        sc = self  # noqa
        def integrate_phase():
            sc.clear_forces(); sc.accumulate_active_fiber_force(); sc.integrate()
            if hasattr(sc, "apply_pintotarget"): sc.apply_pintotarget()
            sc.clear()
        def solve_phase():
            sc.solve_constraints(run_extras=False)
            if sc.use_jacobi: sc.apply_dP()
            sc.solve_extras()
        def finalize_phase():
            sc.post_smooth(); sc.freeze_near_inverted(); sc.repair_inverted_tets()
            if hasattr(sc, "apply_pintotarget"): sc.apply_pintotarget()
            sc.update_velocities()
        self.program = [integrate_phase, solve_phase, finalize_phase]

    def step(self):
        self.update_attach_targets()
        if not hasattr(self, "program"):
            self.build_program()
        self.run_program()
```

> The debug-per-substep hook (`_debug_check_substep`) is dropped from the hot path; if `cfg.debug_per_substep` matters, append a `debug_phase` closing over `sc._debug_check_substep(sub_i, n_sub)` — but it does not affect `pos`, so parity holds either way. For bit-wise parity it can be omitted.

- [ ] **Step 3: Run MuscleSim golden**

Run: `PYTHONPATH=src python -m pytest tests/test_engine_parity.py::test_musclesim_humanv2_bitwise -v`
Expected: PASS. The phase decomposition issues the identical kernel sequence.

- [ ] **Step 4: Run the broader MuscleSim smoke suite**

Run: `PYTHONPATH=src python -m pytest tests/ -k "muscle or humanv2 or lion" -q`
Expected: no regressions (PASS/skip as before).

- [ ] **Step 5: Commit**

```bash
git add src/muscle/muscle_warp.py
git commit -m "refactor(engine): MuscleSim.step runs a bound-method program"
```

---

## Stage 2 — `OneWaySim` onto `XpbdEngine`

Goal: `OneWaySim` reuses the engine's program runner + collision-extras hook, keeping its **own** ARAP/volume/glue/multigrid kernels (redundant by design — not merged with MuscleSim's). Behavior byte-identical to current `substep()`.

### Task 5: OneWaySim subclasses XpbdEngine, program == current substep

**Files:**
- Modify: `src/muscle/coupleOneWayGmg/solver.py`

- [ ] **Step 1: Import the base and inherit**

```python
from muscle.muscle_warp import ssvd_fn, XpbdEngine

class OneWaySim(XpbdEngine):
    ...
```

> `OneWaySim` does not use `self.cfg.num_substeps`; it uses `self.nsubsteps`. Override `run_program` use by giving `OneWaySim` a `step_frame` that loops `self.nsubsteps` over the program (below). Do **not** route OneWay through `XpbdEngine.run_program` (different substep count source).

- [ ] **Step 2: Express `substep()` as a program, drive from `step_frame`**

Replace `OneWaySim.substep()` with a program built once, reproducing the **exact** order of the current `substep()` (`solver.py:1167-1239`): multigrid V-cycle → fascia anchor → `L.zero_` → `fine_iters` × (snapshot `pos_mid`; ARAP; optional volume; `collision_solvers_iter`) → fascia glue → `collision_solvers`. Keep every existing `wp.launch` call unchanged; only wrap them in phase closures.

```python
    def build_program(self):
        s = self
        def pre_phase():
            if s.use_multigrid: s._run_multigrid()      # extract current MG block verbatim
            if getattr(s, "merged", None) is not None:
                from muscle.coupleOneWayGmg.solver import apply_anchor
                apply_anchor(s.merged, s._device)
            s.L.zero_()
            if s.use_volume: s.vol_L.zero_()
        def iter_phase():
            for it in range(s.fine_iters):
                wp.copy(s.pos_mid, s.pos)
                wp.launch(arap_svd_kernel, dim=s.NT,
                          inputs=[s.pos_mid, s.tet_idx, s.B, s.inv_mass, s.alpha,
                                  s.L, s.dt, s.omega, s.pos], device=s._device)
                if s.use_volume:
                    wp.launch(solve_volume_kernel, dim=s.NT,
                              inputs=[s.pos_mid, s.tet_idx, s.B, s.inv_mass,
                                      s.vol_alpha, s.vol_L, s.dt, s.omega, s.pos],
                              device=s._device)
                for fn in getattr(s, "collision_solvers_iter", None) or []:
                    fn(it)
        def post_phase():
            if getattr(s, "merged", None) is not None:
                from muscle.coupleOneWayGmg.solver import solve_glue
                solve_glue(s.merged, s.dt, s.omega, s.fascia_glue_iters, s._device)
            for fn in getattr(s, "collision_solvers", None) or []:
                fn()
        self.program = [pre_phase, iter_phase, post_phase]

    def substep(self):
        if not hasattr(self, "program"):
            self.build_program()
        for phase in self.program:
            phase()
```

Extract the current multigrid block (`solver.py:1171-1200`) into `_run_multigrid()` verbatim so `pre_phase` stays small.

- [ ] **Step 3: Run OneWay lion parity**

Run: `PYTHONPATH=src python -m pytest tests/test_engine_parity.py::test_oneway_lion_bitwise -v`
Expected: PASS (same kernels, same order).

- [ ] **Step 4: Run lion + fascia smoke suites**

Run: `PYTHONPATH=src python -m pytest tests/test_lion_sim_smoke.py tests/test_fascia_sim_smoke.py -q`
Expected: no regressions.

- [ ] **Step 5: Capture + verify humanV2 and fascia OneWay goldens**

Add `capture_oneway_humanv2()` and `capture_oneway_fascia()` to the capture script (same shape as `capture_oneway`, with the humanV2-GMG and fascia configs), add matching parity tests, capture, and run:
Run: `PYTHONPATH=src python tests/golden/capture_engine_goldens.py && PYTHONPATH=src python -m pytest tests/test_engine_parity.py -k oneway -v`
Expected: all OneWay parity tests PASS.

> Capture goldens for these on a **clean checkout of the pre-Stage-2 commit** if Stage 2 is already applied — goldens must reflect pre-refactor behavior. If captured at Stage 0/1 (before touching `solver.py`), they are already valid.

- [ ] **Step 6: Commit**

```bash
git add src/muscle/coupleOneWayGmg/solver.py tests/
git commit -m "refactor(engine): OneWaySim runs on XpbdEngine program runner"
```

---

## Stage 3 — `MassSpringSim` onto `XpbdEngine`; rewire U-drop

Goal: the Gaia U-drop runs as a first-class engine subclass. Bit-identical to the bespoke driver golden (`massspring_u.npz`).

### Task 6: MassSpringSim class

**Files:**
- Create: `src/muscle/collision/udcf/massspring_sim.py`

- [ ] **Step 1: Write `MassSpringSim` wrapping `MassSpringTet` as a program**

```python
"""MassSpringSim — Gaia S01 dynamic MassSpring as an XpbdEngine program.

Composes the validated MassSpringTet material (collision/udcf/massspring.py)
with boundary + DCD collision, reproducing Gaia runStepCPU substep order:
  integrate -> spring -> volume(hard) -> boundary -> inversion -> DCD -> update_vel
Output is bit-identical to the pre-merge bespoke driver (see
tests/golden/capture_engine_goldens.py::capture_massspring)."""
from __future__ import annotations

import numpy as np
import warp as wp

from muscle.muscle_warp import XpbdEngine
from muscle.collision.udcf.massspring import MassSpringTet, MassSpringParams
from muscle.collision.udcf.solver import UdcfDcdCollider, UdcfParams
from muscle.collision.udcf.kernel import box_boundary_kernel
from muscle.util.geo_io import GeoFile
from muscle.util.mesh_io import build_surface_tris


class MassSpringSim(XpbdEngine):
    def __init__(self, ms, dcd, n_substeps, bounds_lo, bounds_hi,
                 mu_s, mu_d, no_inversion=False, device="cuda:0"):
        self.ms = ms
        self.dcd = dcd
        self.n_substeps = int(n_substeps)
        self.bounds_lo = bounds_lo
        self.bounds_hi = bounds_hi
        self.mu_s = float(mu_s)
        self.mu_d = float(mu_d)
        self.no_inversion = bool(no_inversion)
        self.device = device
        self.build_program()

    # expose engine-standard handles
    @property
    def pos(self):
        return self.ms.pos

    @property
    def vel(self):
        return self.ms.vel

    def build_program(self):
        s = self
        def integrate_phase():
            s.ms.integrate()
            s.ms.solve_material()                       # spring -> volume(hard)
            wp.launch(box_boundary_kernel, dim=s.ms.n_verts,
                      inputs=[s.ms.pos, s.ms.pprev, s.bounds_lo, s.bounds_hi,
                              wp.float32(s.mu_s), wp.float32(s.mu_d)],
                      device=s.device)
        def collide_phase():
            if not s.no_inversion:
                s.ms.solve_inversion()
            if s.dcd is not None:
                s.dcd.solve(0)
        def finalize_phase():
            s.ms.update_velocities()
        self.program = [integrate_phase, collide_phase, finalize_phase]

    def step(self):
        for _ in range(self.n_substeps):
            for phase in self.program:
                phase()

    @classmethod
    def u_drop(cls, geo_path, *, dt=1.0 / 60.0, n_substeps=20, density=3.0,
               gravity=-9.8, init_vel_y=-5.0, translate_y=0.2,
               spring_compliance=1e-4, spring_damping=0.01,
               mu_s=0.5, mu_d=0.4, thickness=1e-3, no_dcd=False,
               device="cuda:0"):
        """Build the S01 U-drop exactly as the bespoke driver did."""
        g = GeoFile(geo_path)
        verts = np.asarray(g.positions, np.float32).copy()
        verts[:, 1] += translate_y
        tets = np.asarray(g.tet_indices, np.int32)
        n_v = verts.shape[0]
        msp = MassSpringParams(spring_compliance=spring_compliance,
                               spring_damping=spring_damping,
                               density=density, gravity=gravity)
        ms = MassSpringTet(verts, tets, dt / n_substeps, params=msp, device=device)
        if init_vel_y != 0.0:
            v0 = np.zeros((n_v, 3), np.float32); v0[:, 1] = init_vel_y
            ms.vel.assign(v0)
        surf = build_surface_tris(tets, verts).astype(np.int32)
        dcd = None
        if not no_dcd:
            mp = max(65536, surf.shape[0] * (surf.shape[0] - 1) // 2)
            dv = np.array([[1e4, 1e4, 1e4], [1e4 + 1, 1e4, 1e4],
                           [1e4, 1e4, 1e4 + 1]], np.float32)
            dcd = UdcfDcdCollider(ms.pos, ms.inv_mass, surface_tris=surf,
                                  static_verts=dv, static_tris=np.array([[0, 1, 2]], np.int32),
                                  params=UdcfParams(allow_self=True, max_pairs=mp,
                                                    thickness=thickness, jacobi_avg=False,
                                                    gauss_seidel=True, max_push=0.0),
                                  device=device)
        lo = wp.vec3(-2.0, 0.0, -2.0); hi = wp.vec3(2.0, 5.0e4, 2.0)
        sim = cls(ms, dcd, n_substeps, lo, hi, mu_s, mu_d, device=device)
        sim.surf = surf
        return sim
```

- [ ] **Step 2: Run the MassSpring parity test**

Run: `PYTHONPATH=src python -m pytest tests/test_engine_parity.py::test_massspring_u_bitwise -v`
Expected: PASS — `MassSpringSim.u_drop(...).step()` issues the identical kernel sequence as the bespoke driver, so pos+vel match `massspring_u.npz` bit-for-bit.

> If it diverges: the phase order or a default (thickness/init_vel/translate) differs from the capture. Align the classmethod defaults to `capture_massspring` exactly. Do not relax the test.

- [ ] **Step 3: Commit**

```bash
git add src/muscle/collision/udcf/massspring_sim.py
git commit -m "feat(engine): MassSpringSim — Gaia U-drop on XpbdEngine program"
```

### Task 7: Rewire the U-drop example onto MassSpringSim

**Files:**
- Modify: `src/muscle/collision/udcf/example/u_drop.py`

- [ ] **Step 1: Replace the hand-wired sim with MassSpringSim**

Swap the manual `MassSpringTet` + `UdcfDcdCollider` + `substep()` construction (`u_drop.py:99-178`) for:

```python
    from muscle.collision.udcf.massspring_sim import MassSpringSim
    sim = MassSpringSim.u_drop(
        args.geo, dt=args.dt, n_substeps=args.num_substeps, density=args.density,
        gravity=args.gravity, init_vel_y=args.init_vel_y, translate_y=args.translate_y,
        spring_compliance=args.spring_compliance, spring_damping=args.spring_damping,
        mu_s=args.mu_s, mu_d=args.mu_d, thickness=args.thickness,
        no_dcd=args.no_dcd, device=args.device)
    ms = sim.ms
    surf = sim.surf
```

Keep the `--init_state` branch as a pre-`step` override (`ms.vel.assign(...)`, `ms.pos.assign(...)`) after construction. Keep the renderer/PLY/ref-compare loop, but replace the inner `for _ in range(num_substeps): substep()` with `sim.step()`.

- [ ] **Step 2: Run the example end-to-end**

Run: `PYTHONPATH=src python -m muscle.collision.udcf.example.u_drop --steps 10 --init_vel_y -5`
Expected: prints bounded `min_y`/`max_y`, no `NaN!`, ends `done -> output/udcf_u_drop.usd`.

- [ ] **Step 3: Confirm USD written**

Run: `ls -la output/udcf_u_drop.usd`
Expected: file exists, non-zero size.

- [ ] **Step 4: Commit**

```bash
git add src/muscle/collision/udcf/example/u_drop.py
git commit -m "refactor(collision): U-drop example drives MassSpringSim"
```

---

## Stage 4 — Constraint-registry unification (optional dedup)

Goal: route the still-bespoke collision (`box_boundary`, DCD) through the engine's `constraint_solvers[]` hook so all three sims share one collision-extras mechanism, and register MassSpring's edge-spring / hard-volume / inversion in `_dispatch_constraints` **only if** a bit-identical kernel path exists. Skip any dedup that perturbs a single ULP — redundant kernels stay.

### Task 8: Boundary + DCD as collision extras

**Files:**
- Modify: `src/muscle/collision/udcf/massspring_sim.py`

- [ ] **Step 1: Move boundary + DCD into `constraint_solvers`**

Register the boundary launch and `dcd.solve` as entries in `self.constraint_solvers` (a list the engine already understands), and have `collide_phase` iterate it instead of calling them directly. The launch arguments and order are unchanged.

```python
        self.constraint_solvers = []
        self.constraint_solvers.append(lambda: (
            None if s.no_inversion else s.ms.solve_inversion()))
        if s.dcd is not None:
            self.constraint_solvers.append(lambda: s.dcd.solve(0))
        def collide_phase():
            for fn in s.constraint_solvers:
                fn()
```

> Boundary stays in `integrate_phase` (it runs **before** inversion in Gaia order); only inversion + DCD are extras. Preserve exact order.

- [ ] **Step 2: Run MassSpring parity**

Run: `PYTHONPATH=src python -m pytest tests/test_engine_parity.py::test_massspring_u_bitwise -v`
Expected: PASS (same calls, same order, now via the list).

- [ ] **Step 3: Run the full parity suite**

Run: `PYTHONPATH=src python -m pytest tests/test_engine_parity.py -v`
Expected: every captured golden PASS; missing-data cases skip.

- [ ] **Step 4: Commit**

```bash
git add src/muscle/collision/udcf/massspring_sim.py
git commit -m "refactor(engine): MassSpringSim collision via constraint_solvers hook"
```

### Task 9: Final regression sweep + README note

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the whole test suite**

Run: `PYTHONPATH=src python -m pytest tests/ -q`
Expected: no failures; data-dependent tests skip if assets absent.

- [ ] **Step 2: Note the unified engine in README**

Add one line under the DCD example: that the U-drop now runs on the shared `XpbdEngine` (`MassSpringSim`), same engine as MuscleSim/OneWaySim. Keep the existing reproduce command (it still works via the rewired example).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(engine): note unified XpbdEngine in examples"
```

---

## Acceptance

- `tests/test_engine_parity.py` passes bit-wise for every captured golden (MuscleSim/humanV2, OneWaySim/lion+humanV2+fascia, MassSpring/U-drop) on `cuda:0`.
- `MuscleSim`, `OneWaySim`, `MassSpringSim` all subclass `XpbdEngine` and drive their substep through a bound-method program.
- U-drop example runs via `MassSpringSim`, USD written, no NaN.
- No kernel merged unless proven FP-identical; redundant ARAP/volume kept by design.
