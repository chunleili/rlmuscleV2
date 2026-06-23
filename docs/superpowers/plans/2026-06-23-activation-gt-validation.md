# Muscle-Activation GT Validation Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline harness that compares rl_mass's learned muscle activations against OpenSim Static-Optimization ground truth on the Arm26 model (2 DOF, 6 muscles).

**Architecture:** A small `.osim→MJCF` parser and `.mot→npz` loader produce an Arm26 MuJoCo model + reference motion; the existing `rl_mass.finger.BimanualEnv` trains a policy on it unchanged; a GT-source-agnostic comparison core (`GTLoader` + name-map + resample + metrics) compares rollout activations against the SO `.sto` and emits metrics + plots.

**Tech Stack:** Python, MuJoCo 3.8 (`uv run`), NumPy, Matplotlib, PyTorch (existing PPO), `xml.etree.ElementTree` for OpenSim XML.

## Global Constraints

- All commands run from repo root `/mnt/storage_disk_1/chunleli/Dev/rlmuscleV2-finger`, prefixed with `uv run`.
- Tests are flat runnable scripts under `tests/rl_mass/`, each with a `main()` that asserts and prints `OK`; run via `uv run python tests/rl_mass/<file>.py`. No pytest config in this repo.
- `rl_mass` is an installed (editable) package, but `tools/` is **not** on `sys.path` when a test script runs (Python puts the script's own dir first, not the repo root). Any test that does `from tools.X import ...` MUST start with this repo-root bootstrap so the plain `uv run python tests/rl_mass/<file>.py` command resolves the import:
  ```python
  import os, sys
  sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
  ```
  Likewise, any `tools/` script that imports a sibling `tools` module (e.g. `from tools._rollout import rollout`) is run as `PYTHONPATH=. uv run python tools/<script>.py` from repo root.
- Behavior/options go in JSON config; the only CLI arg for example/tool behavior is `--config`/`-d`/explicit `-i/-o` paths. No argparse flags that encode behavior choices.
- Commit messages end with the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Work stays on branch `finger`.
- Reuse `rl_mass.finger.BimanualEnv` unchanged for Arm26 (fixed base, no equality constraints).
- The metrics/align/plot core must contain **zero** OpenSim- or EMG-specific imports; source-specific code lives only in a `GTLoader` subclass.

**Source paths (read-only inputs):**
- Model: `/home/chunleli/Dev/OpenSimExample/Models/Arm26/InverseKinematics/arm26.osim`
- Motion: `/home/chunleli/Dev/OpenSimExample/Models/Arm26/OutputReference/InverseKinematics/arm26_InverseKinematics.mot`
- GT: `/home/chunleli/Dev/OpenSimExample/Models/Arm26/OutputReference/StaticOptimization/PhysiologicalResults/arm26_StaticOptimization_activation.sto`

**Generated artifacts (committed):**
- `data/rl_mass_arm26/arm26.xml` (MJCF), `data/rl_mass_arm26/arm26_motion.npz`, `data/rl_mass_arm26/config.json`

**Muscle facts (verified):** 6 muscles `TRIlong, TRIlat, TRImed, BIClong, BICshort, BRA`; flexors = `BIClong, BICshort, BRA`, extensors = `TRIlong, TRIlat, TRImed`. Coordinates in qpos order: `r_shoulder_elev, r_elbow_flex`.

---

## File Structure

```
tools/osim2mjcf.py            # T2-T4  .osim -> MJCF (Arm26 subset; errors loudly on unknown constructs)
tools/mot2npz.py              # T1     .mot -> qpos npz (TrajectoryRef format)
tools/_rollout.py             # T6     shared deterministic rollout (factored from bake tool)
tools/validate_vs_gt.py       # T10    harness: rollout activations vs GT -> metrics.json + compare.png + compare.csv
src/rl_mass/finger/gt/
  __init__.py                 # T7     re-exports
  loaders.py                  # T7     GTLoader ABC + OpenSimStoLoader
  align.py                    # T8     MuscleNameMap + resample_to_common_grid
  metrics.py                  # T9     pearson_r, rmse, r2, timing_lag, co_contraction_index
data/rl_mass_arm26/
  config.json                 # T5     BimanualEnv config for Arm26 (+ muscle_groups)
  arm26.xml                   # T2-T4  generated MJCF
  arm26_motion.npz            # T1     generated trajectory
tests/rl_mass/
  test_mot2npz.py             # T1
  test_osim2mjcf_skeleton.py  # T2
  test_osim2mjcf_muscles.py   # T3
  test_osim2mjcf_wrap.py      # T4
  test_arm26_env.py           # T5
  test_rollout_helper.py      # T6
  test_gt_loaders.py          # T7
  test_gt_align.py            # T8
  test_gt_metrics.py          # T9
  test_validate_vs_gt.py      # T10
```

---

### Task 1: `.mot → qpos npz` loader

**Files:**
- Create: `tools/mot2npz.py`
- Test: `tests/rl_mass/test_mot2npz.py`
- Output artifact: `data/rl_mass_arm26/arm26_motion.npz`

**Interfaces:**
- Produces: `read_mot(path) -> (col_names: list[str], time: np.ndarray (T,), data: np.ndarray (T, ncols), in_degrees: bool)` and `mot_to_npz(mot_path, out_path, coords: list[str], rotational: set[str]) -> None`. The npz has keys `n_episodes=1`, `dt: float`, `episode_0_traj_qpos: (T, len(coords))` — exactly the format `rl_mass.finger.TrajectoryRef` consumes.

- [ ] **Step 1: Write the failing test**

```python
# tests/rl_mass/test_mot2npz.py
"""mot2npz: parse a Storage .mot, select+reorder coords, deg->rad, write npz."""
import os, tempfile
import numpy as np
from tools.mot2npz import read_mot, mot_to_npz
from rl_mass.finger.trajectory_ref import TrajectoryRef

SAMPLE = """Coordinates
version=1
nRows=3
nColumns=3
inDegrees=yes

endheader
time\tr_shoulder_elev\tr_elbow_flex
0.0\t0.0\t90.0
0.5\t10.0\t90.0
1.0\t20.0\t90.0
"""

def main():
    d = tempfile.mkdtemp()
    mot = os.path.join(d, "x.mot")
    with open(mot, "w") as f:
        f.write(SAMPLE)
    cols, t, data, deg = read_mot(mot)
    assert cols == ["r_shoulder_elev", "r_elbow_flex"], cols
    assert deg is True
    assert t.shape == (3,) and data.shape == (3, 2), (t.shape, data.shape)
    assert abs(t[1] - 0.5) < 1e-12

    out = os.path.join(d, "x.npz")
    mot_to_npz(mot, out, coords=["r_shoulder_elev", "r_elbow_flex"],
               rotational={"r_shoulder_elev", "r_elbow_flex"})
    z = np.load(out)
    assert int(z["n_episodes"]) == 1
    assert abs(float(z["dt"]) - 0.5) < 1e-12
    q = z["episode_0_traj_qpos"]
    assert q.shape == (3, 2), q.shape
    # 90 deg -> pi/2 on the elbow column
    assert abs(q[0, 1] - np.pi / 2) < 1e-9, q[0, 1]
    # TrajectoryRef loads it
    ref = TrajectoryRef(out, nq=2)
    assert ref.n_frames == 3 and ref.nq == 2
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tests/rl_mass/test_mot2npz.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.mot2npz'` (or ImportError).

- [ ] **Step 3: Write minimal implementation**

```python
# tools/mot2npz.py
"""Convert an OpenSim Storage `.mot` motion to a TrajectoryRef `.npz`.

Selects the requested coordinate columns (in the given order = MJCF qpos
order), converts rotational coordinates degrees->radians when the file is
inDegrees=yes, and writes a single-episode npz in the format consumed by
rl_mass.finger.TrajectoryRef: keys `n_episodes`, `dt`, `episode_0_traj_qpos`.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def read_mot(path):
    """Parse a Storage .mot. Returns (col_names, time(T,), data(T,ncols), in_degrees)."""
    with open(path) as f:
        lines = f.read().splitlines()
    in_degrees = False
    i = 0
    for i, ln in enumerate(lines):
        s = ln.strip().lower()
        if s.startswith("indegrees"):
            in_degrees = s.split("=")[1].strip() == "yes"
        if s == "endheader":
            break
    header = lines[i + 1].split("\t")
    rows = [r for r in lines[i + 2:] if r.strip()]
    mat = np.array([[float(x) for x in r.split("\t")] for r in rows], dtype=np.float64)
    time = mat[:, 0]
    col_names = header[1:]
    data = mat[:, 1:]
    return col_names, time, data, in_degrees


def mot_to_npz(mot_path, out_path, coords, rotational):
    col_names, time, data, in_degrees = read_mot(mot_path)
    idx = [col_names.index(c) for c in coords]  # raises ValueError if missing
    q = data[:, idx].copy()
    if in_degrees:
        for j, c in enumerate(coords):
            if c in rotational:
                q[:, j] = np.deg2rad(q[:, j])
    dt = float(time[1] - time[0]) if len(time) > 1 else 1.0
    np.savez(
        out_path,
        n_episodes=np.int64(1),
        dt=np.float64(dt),
        episode_0_traj_qpos=q.astype(np.float64),
    )


def main():
    p = argparse.ArgumentParser(description="OpenSim .mot -> TrajectoryRef .npz")
    p.add_argument("-i", "--mot", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--coords", required=True,
                   help="Comma-separated coordinate names in qpos order")
    p.add_argument("--rotational", default="",
                   help="Comma-separated subset of --coords that are rotational")
    a = p.parse_args()
    coords = [c.strip() for c in a.coords.split(",") if c.strip()]
    rot = {c.strip() for c in a.rotational.split(",") if c.strip()}
    mot_to_npz(a.mot, a.out, coords, rot)
    print(json.dumps({"out": a.out, "coords": coords, "rotational": sorted(rot)}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_mot2npz.py`
Expected: `OK`

- [ ] **Step 5: Generate the real Arm26 motion artifact**

Run:
```bash
mkdir -p data/rl_mass_arm26
uv run python tools/mot2npz.py \
  -i /home/chunleli/Dev/OpenSimExample/Models/Arm26/OutputReference/InverseKinematics/arm26_InverseKinematics.mot \
  -o data/rl_mass_arm26/arm26_motion.npz \
  --coords r_shoulder_elev,r_elbow_flex \
  --rotational r_shoulder_elev,r_elbow_flex
```
Expected: prints JSON `{"out": "data/rl_mass_arm26/arm26_motion.npz", ...}` and the file exists. Verify:
```bash
uv run python -c "import numpy as np; z=np.load('data/rl_mass_arm26/arm26_motion.npz'); print(z['episode_0_traj_qpos'].shape, float(z['dt']))"
```
Expected: `(121, 2) 0.00833...`

- [ ] **Step 6: Commit**

```bash
git add tools/mot2npz.py tests/rl_mass/test_mot2npz.py data/rl_mass_arm26/arm26_motion.npz
git commit -m "feat(gt): .mot -> TrajectoryRef .npz loader + Arm26 motion artifact

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `.osim → MJCF` parser — bodies + joints (skeleton)

**Files:**
- Create: `tools/osim2mjcf.py`
- Test: `tests/rl_mass/test_osim2mjcf_skeleton.py`

**Interfaces:**
- Produces: `parse_bodies(root) -> dict[name, {mass, com(3), inertia6, meshes}]`; `parse_joints(root) -> list[{type, name, parent, child, pos(3), orient(3), axis?, coord?, range?}]`; `build_mjcf(osim_path) -> str` (full MJCF XML string). For Task 2, `build_mjcf` emits worldbody + nested bodies + hinge joints only (no muscles yet). `parse_muscles` / wrap come in later tasks; `build_mjcf` calls them but they may return `[]` until implemented.

- [ ] **Step 1: Write the failing test**

```python
# tests/rl_mass/test_osim2mjcf_skeleton.py
"""osim2mjcf skeleton: bodies + hinge joints compile in MuJoCo with nq=2."""
import mujoco
from tools.osim2mjcf import parse_bodies, parse_joints, build_mjcf
import xml.etree.ElementTree as ET

OSIM = "/home/chunleli/Dev/OpenSimExample/Models/Arm26/InverseKinematics/arm26.osim"

def main():
    root = ET.parse(OSIM).getroot()
    bodies = parse_bodies(root)
    assert set(bodies) == {"base", "r_humerus", "r_ulna_radius_hand"}, set(bodies)
    assert abs(bodies["r_humerus"]["mass"] - 1.864572) < 1e-4

    joints = parse_joints(root)
    customs = [j for j in joints if j["type"] == "CustomJoint"]
    assert {j["coord"] for j in customs} == {"r_shoulder_elev", "r_elbow_flex"}
    sh = next(j for j in customs if j["coord"] == "r_shoulder_elev")
    assert sh["parent"] == "base" and sh["child"] == "r_humerus"

    xml = build_mjcf(OSIM)
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.nq == 2, m.nq
    assert m.njnt == 2, m.njnt
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)}
    assert names == {"r_shoulder_elev", "r_elbow_flex"}, names
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tests/rl_mass/test_osim2mjcf_skeleton.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.osim2mjcf'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/osim2mjcf.py
"""Convert the Arm26-subset of an OpenSim `.osim` to a MuJoCo MJCF.

Scope (Arm26): BodySet/Body (mass, mass_center, diagonal inertia), JointSet
WeldJoint + single-axis CustomJoint -> MuJoCo hinge, ForceSet/Thelen2003Muscle
-> spatial tendons through PathPoint sites + MuJoCo <muscle> actuators (Task 3),
WrapCylinder/WrapEllipsoid -> tendon wrap geoms (Task 4).

The parser errors loudly (raises) on any construct it does not handle, so that
larger models (e.g. Gait2354: coordinate-coupler constraints, multi-axis joints,
free root) surface their gaps explicitly instead of being silently dropped.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET

import numpy as np


def _vec(text, n=None):
    v = [float(x) for x in (text or "").split()]
    if n is not None and len(v) != n:
        raise ValueError(f"expected {n} floats, got {v!r}")
    return v


def parse_bodies(root):
    bodies = {}
    bset = root.find(".//BodySet/objects")
    for b in bset.findall("Body"):
        meshes = [m.findtext("mesh_file") for m in b.findall(".//Mesh")]
        bodies[b.get("name")] = dict(
            mass=float(b.findtext("mass", "0")),
            com=_vec(b.findtext("mass_center", "0 0 0"), 3),
            inertia=_vec(b.findtext("inertia", "0 0 0 0 0 0"), 6),
            meshes=[m for m in meshes if m],
        )
    return bodies


def _offset_frames(joint):
    frames = {}
    for pof in joint.findall(".//PhysicalOffsetFrame"):
        frames[pof.get("name")] = dict(
            parent=pof.findtext("socket_parent").split("/")[-1],
            translation=_vec(pof.findtext("translation", "0 0 0"), 3),
            orientation=_vec(pof.findtext("orientation", "0 0 0"), 3),
        )
    return frames


def parse_joints(root):
    out = []
    jset = root.find(".//JointSet/objects")
    for j in jset:
        jtype = j.tag
        if jtype not in ("WeldJoint", "CustomJoint"):
            raise NotImplementedError(f"joint type {jtype!r} not handled")
        frames = _offset_frames(j)
        pf = frames[j.findtext("socket_parent_frame")]
        cf = frames[j.findtext("socket_child_frame")]
        if any(abs(x) > 1e-9 for x in cf["translation"]):
            raise NotImplementedError(
                f"non-zero child offset on joint {j.get('name')!r} not handled")
        if any(abs(x) > 1e-9 for x in pf["orientation"] + cf["orientation"]):
            raise NotImplementedError(
                f"non-zero joint frame orientation on {j.get('name')!r} not handled")
        info = dict(type=jtype, name=j.get("name"),
                    parent=pf["parent"], child=cf["parent"],
                    pos=pf["translation"], orient=pf["orientation"])
        if jtype == "CustomJoint":
            axis = coord = None
            st = j.find("SpatialTransform")
            for ta in st.findall("TransformAxis"):
                c = (ta.findtext("coordinates") or "").strip()
                if c and ta.find("LinearFunction") is not None:
                    if axis is not None:
                        raise NotImplementedError(
                            f"multi-DOF CustomJoint {j.get('name')!r} not handled")
                    axis = _vec(ta.findtext("axis"), 3)
                    coord = c
            if axis is None:
                raise NotImplementedError(
                    f"CustomJoint {j.get('name')!r} has no linear rotation axis")
            cobj = j.find(".//Coordinate")
            rng = _vec(cobj.findtext("range"), 2) if cobj is not None else [-3.14159, 3.14159]
            info.update(axis=axis, coord=coord, range=rng)
        out.append(info)
    return out


def parse_muscles(root):  # implemented in Task 3
    return []


def _muscle_xml(muscles):  # implemented in Task 3
    return "", ""


def _wrap_xml(root, muscles):  # implemented in Task 4
    return ""


def _fmt(v):
    return " ".join(f"{x:.9g}" for x in v)


def build_mjcf(osim_path):
    root = ET.parse(osim_path).getroot()
    bodies = parse_bodies(root)
    joints = parse_joints(root)
    muscles = parse_muscles(root)

    # Root bodies hang off ground (WeldJoint parent 'ground' or unparented).
    by_parent = {}
    weld_pos = {}
    for j in joints:
        by_parent.setdefault(j["parent"], []).append(j)
        if j["type"] == "WeldJoint":
            weld_pos[j["child"]] = j["pos"]

    def emit(body_name, pos, indent):
        b = bodies[body_name]
        pad = "  " * indent
        s = f'{pad}<body name="{body_name}" pos="{_fmt(pos)}">\n'
        # hinge joint into this body (skip for welded/root bodies)
        hinge = next((j for j in joints
                      if j["type"] == "CustomJoint" and j["child"] == body_name), None)
        if hinge is not None:
            s += (f'{pad}  <joint name="{hinge["coord"]}" type="hinge" '
                  f'axis="{_fmt(hinge["axis"])}" '
                  f'range="{_fmt(hinge["range"])}" limited="false"/>\n')
        if b["mass"] > 0:
            ixx, iyy, izz = b["inertia"][0], b["inertia"][1], b["inertia"][2]
            if any(abs(x) > 1e-12 for x in b["inertia"][3:]):
                raise NotImplementedError(f"off-diagonal inertia on {body_name!r}")
            s += (f'{pad}  <inertial pos="{_fmt(b["com"])}" mass="{b["mass"]:.9g}" '
                  f'diaginertia="{ixx:.9g} {iyy:.9g} {izz:.9g}"/>\n')
        # sites for muscle path points anchored on this body (Task 3 fills _site_map)
        for sname, sloc in _SITE_MAP.get(body_name, []):
            s += f'{pad}  <site name="{sname}" pos="{_fmt(sloc)}" size="0.005"/>\n'
        for j in by_parent.get(body_name, []):
            if j["type"] == "CustomJoint":
                s += emit(j["child"], j["pos"], indent + 1)
        s += f"{pad}</body>\n"
        return s

    global _SITE_MAP
    _SITE_MAP = {}
    tendon_xml, actuator_xml = _muscle_xml(muscles)  # also fills _SITE_MAP (Task 3)

    body_blocks = ""
    for child, pos in weld_pos.items():
        body_blocks += emit(child, pos, 2)

    wrap_assets = _wrap_xml(root, muscles)

    return f"""<mujoco model="arm26">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001"/>
  <worldbody>
{body_blocks}  </worldbody>
{wrap_assets}{tendon_xml}{actuator_xml}</mujoco>
"""


_SITE_MAP = {}


def main():
    p = argparse.ArgumentParser(description="OpenSim .osim -> MuJoCo MJCF (Arm26 subset)")
    p.add_argument("-i", "--osim", required=True)
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    xml = build_mjcf(a.osim)
    with open(a.out, "w") as f:
        f.write(xml)
    print(a.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_osim2mjcf_skeleton.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add tools/osim2mjcf.py tests/rl_mass/test_osim2mjcf_skeleton.py
git commit -m "feat(gt): osim2mjcf bodies+joints skeleton (Arm26)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `.osim → MJCF` parser — muscles (spatial tendons + actuators)

**Files:**
- Modify: `tools/osim2mjcf.py` (implement `parse_muscles`, `_muscle_xml`)
- Test: `tests/rl_mass/test_osim2mjcf_muscles.py`

**Interfaces:**
- Produces: `parse_muscles(root) -> list[{name, f0, ofl, tsl, pen, points: list[(site_name, body, loc3)]}]`. `_muscle_xml(muscles) -> (tendon_xml, actuator_xml)` and populates module-global `_SITE_MAP[body] = [(site_name, loc3), ...]`. Each muscle becomes one `<spatial>` tendon through its path-point sites and one `<muscle force="{f0}">` actuator named exactly as the muscle. After compile, `actuator_gainprm[i,2] == f0` (what `TendonMuscleSystem` reads).

- [ ] **Step 1: Write the failing test**

```python
# tests/rl_mass/test_osim2mjcf_muscles.py
"""osim2mjcf muscles: 6 spatial tendons + muscle actuators with Thelen force."""
import numpy as np
import mujoco
from tools.osim2mjcf import parse_muscles, build_mjcf
from rl_mass.engine import MuJoCoEngine
from rl_mass.finger.tendon_muscle import TendonMuscleSystem
import xml.etree.ElementTree as ET

OSIM = "/home/chunleli/Dev/OpenSimExample/Models/Arm26/InverseKinematics/arm26.osim"
NAMES = {"TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"}

def main():
    root = ET.parse(OSIM).getroot()
    mus = parse_muscles(root)
    assert {m["name"] for m in mus} == NAMES, {m["name"] for m in mus}
    bic = next(m for m in mus if m["name"] == "BIClong")
    assert abs(bic["f0"] - 624.3) < 1e-3, bic["f0"]
    assert len(bic["points"]) == 9, len(bic["points"])  # BIClong has 9 path points

    xml = build_mjcf(OSIM)
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.nu == 6, m.nu
    anames = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)}
    assert anames == NAMES, anames
    # f0 lives in gainprm[:,2] for MuJoCo muscle actuators (what TendonMuscleSystem reads)
    bic_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "BIClong")
    assert abs(m.actuator_gainprm[bic_id, 2] - 624.3) < 1e-2, m.actuator_gainprm[bic_id, 2]
    # MuJoCo auto-computed a valid length range
    assert np.all(m.actuator_lengthrange[:, 1] > m.actuator_lengthrange[:, 0])

    # TendonMuscleSystem builds against the generated model
    import tempfile, os
    d = tempfile.mkdtemp(); p = os.path.join(d, "arm26.xml")
    open(p, "w").write(xml)
    eng = MuJoCoEngine(p, timestep=1/600); eng.forward()
    tms = TendonMuscleSystem(eng, force_scale=1.0)
    assert tms.n_muscles == 6 and tms.num_total_related_dofs > 0
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tests/rl_mass/test_osim2mjcf_muscles.py`
Expected: FAIL — `parse_muscles` returns `[]`, so `{m["name"] for m in mus}` is empty → AssertionError.

- [ ] **Step 3: Write minimal implementation**

Replace the stub `parse_muscles` and `_muscle_xml` in `tools/osim2mjcf.py` with:

```python
def parse_muscles(root):
    out = []
    fset = root.find(".//ForceSet/objects")
    if fset is None:
        return out
    for mu in fset.findall("Thelen2003Muscle"):
        name = mu.get("name")
        pts = []
        for k, pp in enumerate(mu.findall(".//PathPoint")):
            body = pp.findtext("socket_parent_frame").split("/")[-1]
            loc = _vec(pp.findtext("location"), 3)
            pts.append((f"{name}_P{k}", body, loc))
        out.append(dict(
            name=name,
            f0=float(mu.findtext("max_isometric_force")),
            ofl=float(mu.findtext("optimal_fiber_length")),
            tsl=float(mu.findtext("tendon_slack_length")),
            pen=float(mu.findtext("pennation_angle_at_optimal", "0") or 0.0),
            points=pts,
        ))
    return out


def _muscle_xml(muscles):
    """Build tendon + actuator XML and populate _SITE_MAP. Returns (tendon, actuator)."""
    global _SITE_MAP
    _SITE_MAP = {}
    tendons = ["  <tendon>\n"]
    acts = ["  <actuator>\n"]
    for mu in muscles:
        tendons.append(f'    <spatial name="{mu["name"]}_t" width="0.002">\n')
        for sname, body, loc in mu["points"]:
            _SITE_MAP.setdefault(body, []).append((sname, loc))
            tendons.append(f'      <site site="{sname}"/>\n')
        tendons.append("    </spatial>\n")
        # MuJoCo muscle actuator: force = peak isometric force -> gainprm[2].
        # lengthrange auto-computed by the compiler; if that ever yields a
        # degenerate range, set lengthrange="{tsl+0.5*ofl} {tsl+1.5*ofl}".
        acts.append(
            f'    <muscle name="{mu["name"]}" tendon="{mu["name"]}_t" '
            f'force="{mu["f0"]:.9g}"/>\n')
    tendons.append("  </tendon>\n")
    acts.append("  </actuator>\n")
    return "".join(tendons), "".join(acts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_osim2mjcf_muscles.py`
Expected: `OK`

(If `actuator_lengthrange` comes back degenerate, add `lengthrange="{tsl+0.5*ofl:.6g} {tsl+1.5*ofl:.6g}"` to the `<muscle>` line as the code comment notes, then re-run.)

- [ ] **Step 5: Commit**

```bash
git add tools/osim2mjcf.py tests/rl_mass/test_osim2mjcf_muscles.py
git commit -m "feat(gt): osim2mjcf muscles as spatial tendons + muscle actuators

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `.osim → MJCF` parser — wrap objects

**Files:**
- Modify: `tools/osim2mjcf.py` (implement `_wrap_xml`, route active wraps into tendons)
- Test: `tests/rl_mass/test_osim2mjcf_wrap.py`

**Interfaces:**
- Produces: `parse_wraps(root) -> dict[body_name, list[{name, type, translation3, xyz_rot3, radius, length, dims3, quadrant}]]` and `_wrap_xml(root, muscles) -> str` emitting wrap geoms as MuJoCo `<geom>` of `type="cylinder"` (WrapCylinder) or `type="sphere"` (WrapEllipsoid, dominant-radius approximation) inside the owning body, plus modifies tendon emission to insert `<geom geom="..."/>` wrap entries. A muscle uses a wrap geom only when the OpenSim model lists that wrap in the muscle's `PathWrapSet`. For Arm26 v1 we attach each body's active wrap geoms to every tendon that has a site on that body (documented over-approximation; see note). The test asserts the model still compiles with `nu==6` and that at least one wrap geom is emitted.

- [ ] **Step 1: Write the failing test**

```python
# tests/rl_mass/test_osim2mjcf_wrap.py
"""osim2mjcf wrap: wrap geoms emitted; model still compiles with 6 muscles."""
import mujoco
from tools.osim2mjcf import parse_wraps, build_mjcf
import xml.etree.ElementTree as ET

OSIM = "/home/chunleli/Dev/OpenSimExample/Models/Arm26/InverseKinematics/arm26.osim"

def main():
    root = ET.parse(OSIM).getroot()
    wraps = parse_wraps(root)
    # base owns a WrapCylinder (TRIlongglen); r_humerus owns several.
    assert "base" in wraps and any(w["type"] == "WrapCylinder" for w in wraps["base"])
    assert "r_humerus" in wraps and len(wraps["r_humerus"]) >= 3

    xml = build_mjcf(OSIM)
    assert 'class="wrap"' in xml or "wrapgeom" in xml or "_wrap" in xml, "no wrap geom emitted"
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.nu == 6, m.nu
    # at least one tendon wrap geom present
    assert m.ngeom >= 1
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tests/rl_mass/test_osim2mjcf_wrap.py`
Expected: FAIL with `ImportError: cannot import name 'parse_wraps'`.

- [ ] **Step 3: Write minimal implementation**

Add to `tools/osim2mjcf.py`:

```python
def parse_wraps(root):
    """Body name -> list of wrap objects fixed to that body."""
    out = {}
    bset = root.find(".//BodySet/objects")
    for b in bset.findall("Body"):
        wl = []
        for w in b.findall(".//WrapObjectSet/objects/*"):
            if (w.findtext("active", "true") or "true").strip() != "true":
                continue
            wl.append(dict(
                name=w.get("name"),
                type=w.tag,
                translation=_vec(w.findtext("translation", "0 0 0"), 3),
                xyz_rot=_vec(w.findtext("xyz_body_rotation", "0 0 0"), 3),
                radius=float(w.findtext("radius", "0") or 0.0),
                length=float(w.findtext("length", "0") or 0.0),
                dims=_vec(w.findtext("dimensions", "0 0 0"), 3) if w.findtext("dimensions") else [0, 0, 0],
                quadrant=(w.findtext("quadrant", "all") or "all").strip(),
            ))
        if wl:
            out[b.get("name")] = wl
    return out


def _euler_xyz_to_quat(rx, ry, rz):
    cx, sx = np.cos(rx / 2), np.sin(rx / 2)
    cy, sy = np.cos(ry / 2), np.sin(ry / 2)
    cz, sz = np.cos(rz / 2), np.sin(rz / 2)
    # body-fixed x-y-z
    w = cx * cy * cz - sx * sy * sz
    x = sx * cy * cz + cx * sy * sz
    y = cx * sy * cz - sx * cy * sz
    z = cx * cy * sz + sx * sy * cz
    return [w, x, y, z]
```

Then extend `build_mjcf` so wrap geoms are emitted inside their owning body and registered for tendon insertion. Replace the `_SITE_MAP`/site-emit section of `emit()` and the wrap-return with:

```python
    # (inside build_mjcf, before defining emit)
    wraps = parse_wraps(root)
    global _WRAP_MAP
    _WRAP_MAP = {}  # body -> list of (geom_name)
    for body, wl in wraps.items():
        for w in wl:
            gname = f'{w["name"]}_wrap'
            _WRAP_MAP.setdefault(body, []).append((gname, w))
```

Add geom emission in `emit()` right after the `<inertial>` line:

```python
        for gname, w in _WRAP_MAP.get(body_name, []):
            quat = _euler_xyz_to_quat(*w["xyz_rot"])
            if w["type"] == "WrapCylinder":
                s += (f'{pad}  <geom name="{gname}" type="cylinder" '
                      f'pos="{_fmt(w["translation"])}" quat="{_fmt(quat)}" '
                      f'size="{w["radius"]:.9g} {max(w["length"], 0.01) / 2:.9g}" '
                      f'contype="0" conaffinity="0" group="3"/>\n')
            else:  # WrapEllipsoid -> sphere of dominant radius (documented approx)
                r = max(w["dims"]) if any(w["dims"]) else max(w["radius"], 0.005)
                s += (f'{pad}  <geom name="{gname}" type="sphere" '
                      f'pos="{_fmt(w["translation"])}" quat="{_fmt(quat)}" '
                      f'size="{r:.9g}" contype="0" conaffinity="0" group="3"/>\n')
```

Update `_muscle_xml` so each tendon inserts the wrap geoms of any body it passes through, placed between the two sites bracketing that body. Minimal over-approximation for Arm26: after each site whose body owns wrap geoms, insert that body's wrap geoms once:

```python
def _muscle_xml(muscles):
    global _SITE_MAP
    _SITE_MAP = {}
    tendons = ["  <tendon>\n"]
    acts = ["  <actuator>\n"]
    for mu in muscles:
        tendons.append(f'    <spatial name="{mu["name"]}_t" width="0.002">\n')
        seen_wrap = set()
        for sname, body, loc in mu["points"]:
            _SITE_MAP.setdefault(body, []).append((sname, loc))
            tendons.append(f'      <site site="{sname}"/>\n')
            for gname, _w in _WRAP_MAP.get(body, []):
                if gname not in seen_wrap:
                    tendons.append(f'      <geom geom="{gname}"/>\n')
                    seen_wrap.add(gname)
        tendons.append("    </spatial>\n")
        acts.append(
            f'    <muscle name="{mu["name"]}" tendon="{mu["name"]}_t" '
            f'force="{mu["f0"]:.9g}"/>\n')
    tendons.append("  </tendon>\n")
    acts.append("  </actuator>\n")
    return "".join(tendons), "".join(acts)
```

Set `_WRAP_MAP = {}` as a module global next to `_SITE_MAP`, and make `_wrap_xml` return `""` (wrap geoms now live inside bodies, not as a separate asset block):

```python
_WRAP_MAP = {}


def _wrap_xml(root, muscles):
    return ""
```

> **Note (documented confound):** MuJoCo requires a wrap geom to sit between exactly the two sites it wraps, and only wraps when the straight segment would penetrate. Inserting a body's wrap geoms after the first site on that body is an over-approximation of OpenSim's explicit `PathWrapSet`. It restores the dominant elbow/shoulder wrapping that drives moment arms, but is not geometrically identical. Faithful per-muscle `PathWrapSet` routing and true ellipsoid wraps are future refinements (see spec "Known confounds").

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_osim2mjcf_wrap.py`
Expected: `OK`

(If MuJoCo rejects a wrap geom because it does not sit strictly between two sites on a single-site body, drop that body's wrap insertion — guard with `if len(_SITE_MAP[body]) >= 1 and <next site exists>`; for Arm26 the wrapped bodies always carry ≥2 consecutive sites.)

- [ ] **Step 5: Generate + verify the real MJCF artifact**

Run:
```bash
uv run python tools/osim2mjcf.py \
  -i /home/chunleli/Dev/OpenSimExample/Models/Arm26/InverseKinematics/arm26.osim \
  -o data/rl_mass_arm26/arm26.xml
uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('data/rl_mass_arm26/arm26.xml'); print('nq',m.nq,'nu',m.nu,'ngeom',m.ngeom)"
```
Expected: prints `data/rl_mass_arm26/arm26.xml` then `nq 2 nu 6 ngeom <>=4>`.

- [ ] **Step 6: Commit**

```bash
git add tools/osim2mjcf.py tests/rl_mass/test_osim2mjcf_wrap.py data/rl_mass_arm26/arm26.xml
git commit -m "feat(gt): osim2mjcf wrap geoms + Arm26 MJCF artifact

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Arm26 training config + env smoke + short training

**Files:**
- Create: `data/rl_mass_arm26/config.json`
- Test: `tests/rl_mass/test_arm26_env.py`

**Interfaces:**
- Consumes: `data/rl_mass_arm26/arm26.xml` (T4), `data/rl_mass_arm26/arm26_motion.npz` (T1), `rl_mass.finger.BimanualEnv`.
- Produces: a config consumable by `rl_mass.env_factory.make_env` and `rl_mass.train`; a trained checkpoint under `output/arm26/nn/`. The config carries a `muscle_groups` block (`flexor`/`extensor` name lists) read later by the harness (T10).

- [ ] **Step 1: Write the failing test**

```python
# tests/rl_mass/test_arm26_env.py
"""Arm26 BimanualEnv builds from config: 2 DOF, 6 muscles, state/step run."""
import numpy as np
from rl_mass.env_factory import make_env

CFG = "data/rl_mass_arm26/config.json"

def main():
    env = make_env(CFG)
    assert env.num_action == 2, env.num_action       # no equality constraints
    assert env.num_muscles == 6, env.num_muscles
    s = env.reset(rsi=False)
    assert s.shape[0] == env.num_state
    # one control step with zero action + zero activation runs without NaN
    env.set_action(np.zeros(env.num_action))
    env.set_activation_levels(np.zeros(env.num_muscles))
    env.steps_at_once()
    assert not np.any(np.isnan(env.engine.get_qpos()))
    r = env.get_reward()
    assert 0.0 <= r <= 1.0, r
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tests/rl_mass/test_arm26_env.py`
Expected: FAIL with `FileNotFoundError` for `data/rl_mass_arm26/config.json`.

- [ ] **Step 3: Write the config**

```json
{
    "skeleton_format": "mujoco_native",
    "use_muscle": true,
    "control_hz": 60,
    "simulation_hz": 600,
    "skeleton_file": "/mnt/storage_disk_1/chunleli/Dev/rlmuscleV2-finger/data/rl_mass_arm26/arm26.xml",
    "trajectory_file": "/mnt/storage_disk_1/chunleli/Dev/rlmuscleV2-finger/data/rl_mass_arm26/arm26_motion.npz",
    "reward_param": {
        "w_q": 0.85,
        "w_v": 0.15
    },
    "pose_error_threshold": 50.0,
    "muscle_force_scale": 1.0,
    "spd": {
        "kp": 20.0
    },
    "muscle_groups": {
        "flexor": ["BIClong", "BICshort", "BRA"],
        "extensor": ["TRIlong", "TRIlat", "TRImed"]
    }
}
```

(`control_hz: 60` so the 120 Hz GT resamples cleanly; the motion is 120 Hz.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_arm26_env.py`
Expected: `OK`

- [ ] **Step 5: Smoke-train 3 iters (verify the PPO pipeline accepts Arm26)**

Run:
```bash
uv run python -m rl_mass.train -d data/rl_mass_arm26/config.json --outdir output/arm26 --num_envs 4 --max_iter 3
```
Expected: runs 3 iterations without error and writes `output/arm26/nn/current.pt` (+ `current_muscle.pt`). Confirm:
```bash
ls output/arm26/nn/
```
Expected: lists `current.pt`, `current_muscle.pt` (and `max.pt` if a max was hit).

- [ ] **Step 6: Commit (config only; checkpoints are not committed)**

```bash
git add data/rl_mass_arm26/config.json tests/rl_mass/test_arm26_env.py
git commit -m "feat(gt): Arm26 BimanualEnv training config + env smoke test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Launch full training out-of-band (not a test gate)**

Run (long; ~hundreds of iters, background it):
```bash
uv run python -m rl_mass.train -d data/rl_mass_arm26/config.json --outdir output/arm26 --num_envs 16 --max_iter 300
```
Expected: `output/arm26/train_rewards.csv` shows `avg_return` rising and `output/arm26/nn/max.pt` + `max_muscle.pt` exist. This checkpoint feeds Task 10. (Training quality is monitored via the CSV, not asserted by a unit test.)

---

### Task 6: Shared deterministic rollout helper

**Files:**
- Create: `tools/_rollout.py`
- Modify: `tools/bake_bimanual_usd.py` (import the shared helper instead of its local `rollout`)
- Test: `tests/rl_mass/test_rollout_helper.py`

**Interfaces:**
- Produces: `rollout(env, model, muscle_model, max_steps, deterministic) -> (geom_xpos (T+1,ngeom,3), geom_xmat (T+1,ngeom,3,3), activations (T+1,nu), qpos (T+1,nq), rewards: list[float])`. Identical signature/return to the function currently inside `tools/bake_bimanual_usd.py`. `tools/bake_bimanual_usd.py` imports it: `from tools._rollout import rollout`.

- [ ] **Step 1: Write the failing test**

```python
# tests/rl_mass/test_rollout_helper.py
"""Shared rollout helper: shapes line up on Arm26 with random-init nets."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from rl_mass.env_factory import make_env
from rl_mass.model import MuscleNN, SimulationNN
from tools._rollout import rollout

CFG = "data/rl_mass_arm26/config.json"

def main():
    env = make_env(CFG)
    model = SimulationNN(env.num_state, env.num_action)
    mm = MuscleNN(env.num_total_related_dofs, env.num_action, env.num_muscles)
    gx, gm, acts, qpos, rews = rollout(env, model, mm, max_steps=20, deterministic=True)
    T = gx.shape[0]
    assert acts.shape == (T, env.num_muscles), acts.shape
    assert qpos.shape[1] == env.engine.nq
    assert gm.shape[1:] == (env.model.ngeom, 3, 3)
    assert len(rews) <= T - 1  # one reward per non-terminal step
    assert np.all(acts >= 0.0)
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tests/rl_mass/test_rollout_helper.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools._rollout'`.

- [ ] **Step 3: Create the helper by moving the function out of the bake tool**

Create `tools/_rollout.py` with the exact `rollout` function (and its imports) currently in `tools/bake_bimanual_usd.py`:

```python
# tools/_rollout.py
"""Deterministic policy rollout shared by the USD bake and the GT harness.

Mirrors eval.py's inner muscle loop: per control step, the policy proposes an
action; then for `num_sim_per_control // 2` sub-iterations the muscle network
maps (muscle torques, desired torques) -> activations, which drive two sim
steps. Records per-control-step geom transforms, activations, qpos, rewards.
"""
from __future__ import annotations

import numpy as np

import torch
from torch import Tensor

_USE_CUDA = torch.cuda.is_available()


def rollout(env, model, muscle_model, max_steps, deterministic):
    import mujoco

    state = env.reset(rsi=False)
    mujoco.mj_forward(env.model, env.data)
    num_sim_per_control = env.simulation_hz // env.control_hz

    geom_xpos = [env.data.geom_xpos.copy()]
    geom_xmat = [env.data.geom_xmat.copy().reshape(-1, 3, 3)]
    activations = [np.zeros(env.num_muscles, dtype=np.float32)]
    qpos = [env.data.qpos.copy()]
    rewards = []

    for _ in range(max_steps):
        s = Tensor(state.reshape(1, -1).astype(np.float32))
        if _USE_CUDA:
            s = s.cuda()
        a_dist, _ = model(s)
        action = (a_dist.loc if deterministic else a_dist.sample())
        action = action.cpu().detach().numpy().squeeze()
        env.set_action(action)

        if env.use_muscle and muscle_model is not None:
            mt = Tensor(env.get_muscle_torques().reshape(1, -1).astype(np.float32))
            if _USE_CUDA:
                mt = mt.cuda()
            for _ in range(num_sim_per_control // 2):
                dt = Tensor(env.get_desired_torques().reshape(1, -1).astype(np.float32))
                if _USE_CUDA:
                    dt = dt.cuda()
                act = muscle_model(mt, dt).cpu().detach().numpy().squeeze()
                env.set_activation_levels(act)
                env.step()
                env.step()
        else:
            env.steps_at_once()

        geom_xpos.append(env.data.geom_xpos.copy())
        geom_xmat.append(env.data.geom_xmat.copy().reshape(-1, 3, 3))
        activations.append(env.get_activation_levels().astype(np.float32))
        qpos.append(env.data.qpos.copy())

        if env.is_end_of_episode():
            break
        rewards.append(env.get_reward())
        state = env.get_state()

    return (
        np.stack(geom_xpos).astype(np.float32),
        np.stack(geom_xmat).astype(np.float32),
        np.stack(activations).astype(np.float32),
        np.stack(qpos).astype(np.float32),
        rewards,
    )
```

Then in `tools/bake_bimanual_usd.py`: delete the local `rollout` function and the now-unused `torch`/`Tensor` import block if nothing else uses them, and add near the top:

```python
from tools._rollout import rollout
```

- [ ] **Step 4: Run test to verify it passes, and confirm the bake tool still imports**

Run:
```bash
uv run python tests/rl_mass/test_rollout_helper.py
PYTHONPATH=. uv run python -c "import tools.bake_bimanual_usd"
```
Expected: first prints `OK`; second prints nothing (clean import, no error).

- [ ] **Step 5: Commit**

```bash
git add tools/_rollout.py tools/bake_bimanual_usd.py tests/rl_mass/test_rollout_helper.py
git commit -m "refactor(gt): factor shared rollout helper from bake tool

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: GT loaders (`GTLoader` ABC + `OpenSimStoLoader`)

**Files:**
- Create: `src/rl_mass/finger/gt/__init__.py`, `src/rl_mass/finger/gt/loaders.py`
- Test: `tests/rl_mass/test_gt_loaders.py`

**Interfaces:**
- Produces: `class GTSeries` (NamedTuple: `names: list[str]`, `time: np.ndarray (T,)`, `activation: np.ndarray (T, M)`); `class GTLoader(ABC)` with `load(self) -> GTSeries`; `class OpenSimStoLoader(GTLoader)` constructed as `OpenSimStoLoader(sto_path, muscle_columns: list[str] | None = None)`. `load()` parses a Storage `.sto`, returns the named muscle columns (all non-`time` columns if `muscle_columns` is None), activation values as-is (already in `[0,1]`). `__init__.py` re-exports `GTSeries, GTLoader, OpenSimStoLoader`.

- [ ] **Step 1: Write the failing test**

```python
# tests/rl_mass/test_gt_loaders.py
"""OpenSimStoLoader parses the real Arm26 SO activation .sto."""
import numpy as np
from rl_mass.finger.gt import OpenSimStoLoader

STO = ("/home/chunleli/Dev/OpenSimExample/Models/Arm26/OutputReference/"
       "StaticOptimization/PhysiologicalResults/arm26_StaticOptimization_activation.sto")
NAMES = {"TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"}

def main():
    gt = OpenSimStoLoader(STO).load()
    assert set(gt.names) == NAMES, gt.names
    assert gt.time.shape[0] == 121, gt.time.shape
    assert gt.activation.shape == (121, 6), gt.activation.shape
    assert gt.activation.min() >= -1e-6 and gt.activation.max() <= 1.0 + 1e-6
    # column selection + order
    gt2 = OpenSimStoLoader(STO, muscle_columns=["BIClong", "TRIlong"]).load()
    assert gt2.names == ["BIClong", "TRIlong"]
    assert gt2.activation.shape == (121, 2)
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tests/rl_mass/test_gt_loaders.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl_mass.finger.gt'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/rl_mass/finger/gt/loaders.py
"""Ground-truth activation loaders. The comparison core depends only on the
GTSeries shape; each GT source (OpenSim now, EMG later) is one GTLoader."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import numpy as np


class GTSeries(NamedTuple):
    names: list[str]
    time: np.ndarray        # (T,)
    activation: np.ndarray  # (T, M)


class GTLoader(ABC):
    @abstractmethod
    def load(self) -> GTSeries:
        ...


def _read_storage(path):
    """Parse an OpenSim Storage (.sto/.mot). Returns (col_names, matrix(T,ncols))."""
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    for i, ln in enumerate(lines):
        if ln.strip().lower() == "endheader":
            break
    header = lines[i + 1].split("\t")
    rows = [r for r in lines[i + 2:] if r.strip()]
    mat = np.array([[float(x) for x in r.split("\t")] for r in rows], dtype=np.float64)
    return header, mat


class OpenSimStoLoader(GTLoader):
    def __init__(self, sto_path: str, muscle_columns: Optional[list[str]] = None):
        self.sto_path = sto_path
        self.muscle_columns = muscle_columns

    def load(self) -> GTSeries:
        header, mat = _read_storage(self.sto_path)
        time = mat[:, header.index("time")]
        cols = self.muscle_columns or [c for c in header if c != "time"]
        idx = [header.index(c) for c in cols]
        return GTSeries(names=list(cols), time=time, activation=mat[:, idx])
```

```python
# src/rl_mass/finger/gt/__init__.py
"""GT-source-agnostic activation validation: loaders, alignment, metrics."""
from rl_mass.finger.gt.loaders import GTLoader, GTSeries, OpenSimStoLoader

__all__ = ["GTLoader", "GTSeries", "OpenSimStoLoader"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_gt_loaders.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/rl_mass/finger/gt/__init__.py src/rl_mass/finger/gt/loaders.py tests/rl_mass/test_gt_loaders.py
git commit -m "feat(gt): GTLoader ABC + OpenSimStoLoader

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Alignment (name map + time resample)

**Files:**
- Create: `src/rl_mass/finger/gt/align.py`
- Modify: `src/rl_mass/finger/gt/__init__.py` (re-export)
- Test: `tests/rl_mass/test_gt_align.py`

**Interfaces:**
- Produces: `class MuscleNameMap` constructed `MuscleNameMap(mapping: dict[str, str] | None = None)` (None = identity); `reorder(self, src_names: list[str], dst_names: list[str], values: np.ndarray) -> np.ndarray` selects/reorders the columns of `values (T, len(src_names))` to match `dst_names` (mapping applied to `src_names` first). `resample_to_common_grid(t_a, a, t_b, b) -> (t_common, a_on_common, b_on_common)` linearly interpolates two time series onto the overlap of their time ranges sampled at the finer of the two rates. Both re-exported from `__init__`.

- [ ] **Step 1: Write the failing test**

```python
# tests/rl_mass/test_gt_align.py
"""MuscleNameMap reorders columns; resample lands both series on one grid."""
import numpy as np
from rl_mass.finger.gt import MuscleNameMap, resample_to_common_grid

def main():
    nm = MuscleNameMap()  # identity
    vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (T=2, 3)
    out = nm.reorder(["A", "B", "C"], ["C", "A"], vals)
    assert out.shape == (2, 2)
    assert np.allclose(out[:, 0], [3.0, 6.0]) and np.allclose(out[:, 1], [1.0, 4.0])

    nm2 = MuscleNameMap({"rl_bic": "BIClong"})
    out2 = nm2.reorder(["rl_bic"], ["BIClong"], np.array([[0.5], [0.7]]))
    assert np.allclose(out2[:, 0], [0.5, 0.7])

    # resample: a at 0..1 step .5 ; b at 0..1 step .25 -> common grid = finer (.25)
    ta = np.array([0.0, 0.5, 1.0]); a = np.array([[0.0], [1.0], [2.0]])
    tb = np.array([0.0, 0.25, 0.5, 0.75, 1.0]); b = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    tc, ac, bc = resample_to_common_grid(ta, a, tb, b)
    assert tc[0] == 0.0 and abs(tc[-1] - 1.0) < 1e-12
    assert ac.shape[0] == bc.shape[0] == tc.shape[0]
    # a is linear 0->2 over [0,1]; at t=0.25 expect 0.5
    j = np.argmin(np.abs(tc - 0.25))
    assert abs(ac[j, 0] - 0.5) < 1e-9, ac[j, 0]
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tests/rl_mass/test_gt_align.py`
Expected: FAIL with `ImportError: cannot import name 'MuscleNameMap'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/rl_mass/finger/gt/align.py
"""Align a learned-activation series to a GT series: column name-map + time
resample. No knowledge of the GT source — pure array/time operations."""
from __future__ import annotations

from typing import Optional

import numpy as np


class MuscleNameMap:
    def __init__(self, mapping: Optional[dict] = None):
        self.mapping = mapping or {}

    def reorder(self, src_names, dst_names, values):
        """Return values reordered so columns match dst_names.

        src_names label the columns of `values`; each is first translated
        through the mapping, then matched to dst_names.
        """
        translated = [self.mapping.get(n, n) for n in src_names]
        pos = {n: i for i, n in enumerate(translated)}
        idx = [pos[d] for d in dst_names]  # raises KeyError if a dst muscle is absent
        return values[:, idx]


def resample_to_common_grid(t_a, a, t_b, b):
    """Linearly interpolate (t_a, a) and (t_b, b) onto a shared time grid.

    Grid spans the overlap [max(start), min(end)] sampled at the finer of the
    two input rates. a, b are (T, M) with the SAME M and column order.
    """
    t0 = max(float(t_a[0]), float(t_b[0]))
    t1 = min(float(t_a[-1]), float(t_b[-1]))
    da = (t_a[-1] - t_a[0]) / max(len(t_a) - 1, 1)
    db = (t_b[-1] - t_b[0]) / max(len(t_b) - 1, 1)
    dt = min(da, db)
    n = int(np.floor((t1 - t0) / dt)) + 1
    tc = t0 + dt * np.arange(n)

    def interp(t, v):
        return np.column_stack([np.interp(tc, t, v[:, j]) for j in range(v.shape[1])])

    return tc, interp(t_a, a), interp(t_b, b)
```

Append to `src/rl_mass/finger/gt/__init__.py`:

```python
from rl_mass.finger.gt.align import MuscleNameMap, resample_to_common_grid

__all__ += ["MuscleNameMap", "resample_to_common_grid"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_gt_align.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/rl_mass/finger/gt/align.py src/rl_mass/finger/gt/__init__.py tests/rl_mass/test_gt_align.py
git commit -m "feat(gt): muscle name-map + time resample alignment

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Metrics

**Files:**
- Create: `src/rl_mass/finger/gt/metrics.py`
- Modify: `src/rl_mass/finger/gt/__init__.py` (re-export)
- Test: `tests/rl_mass/test_gt_metrics.py`

**Interfaces:**
- Produces, all operating on 1-D arrays unless noted:
  - `pearson_r(x, y) -> float`
  - `rmse(x, y) -> float`
  - `r2(y_true, y_pred) -> float`  (coefficient of determination)
  - `timing_lag(x, y, dt) -> float`  (seconds; argmax of normalized cross-correlation of `y` vs `x`; positive = `y` lags `x`)
  - `co_contraction_index(a, b) -> float`  (Falconer–Winter: `mean_t 2·min(a,b)/(a+b)`; inputs are summed-group activations, ≥0)

- [ ] **Step 1: Write the failing test**

```python
# tests/rl_mass/test_gt_metrics.py
"""Metrics on known inputs."""
import numpy as np
from rl_mass.finger.gt.metrics import pearson_r, rmse, r2, timing_lag, co_contraction_index

def main():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    assert abs(pearson_r(x, 2 * x + 1) - 1.0) < 1e-9
    assert abs(pearson_r(x, -x) + 1.0) < 1e-9
    assert abs(rmse(x, x)) < 1e-12
    assert abs(rmse(np.zeros(3), np.ones(3)) - 1.0) < 1e-12
    assert abs(r2(x, x) - 1.0) < 1e-9
    assert r2(x, np.full_like(x, x.mean())) < 1e-9 + 0.0  # predicting the mean -> R2 ~ 0

    # timing: y is x delayed by 2 samples, dt=0.1 -> lag ~ +0.2 s
    base = np.sin(np.linspace(0, 6.28, 50))
    y = np.roll(base, 2); y[:2] = 0.0
    lag = timing_lag(base, y, dt=0.1)
    assert abs(lag - 0.2) < 0.05, lag

    # co-contraction: identical groups -> 1.0 ; disjoint on/off -> 0.0
    a = np.array([0.2, 0.4, 0.6])
    assert abs(co_contraction_index(a, a) - 1.0) < 1e-9
    on = np.array([1.0, 0.0]); off = np.array([0.0, 1.0])
    assert abs(co_contraction_index(on, off)) < 1e-9
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tests/rl_mass/test_gt_metrics.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'rl_mass.finger.gt.metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/rl_mass/finger/gt/metrics.py
"""Per-muscle comparison metrics between two activation series.

Pure functions on numpy arrays; no GT-source knowledge. Used by the harness
to score learned activations against ground truth."""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def pearson_r(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean(); yc = y - y.mean()
    denom = np.sqrt((xc @ xc) * (yc @ yc))
    return float(xc @ yc / denom) if denom > _EPS else 0.0


def rmse(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > _EPS else 0.0


def timing_lag(x, y, dt) -> float:
    """Seconds by which y lags x (argmax of normalized cross-correlation)."""
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    nx = np.linalg.norm(x); ny = np.linalg.norm(y)
    if nx < _EPS or ny < _EPS:
        return 0.0
    corr = np.correlate(y / ny, x / nx, mode="full")
    lag_samples = np.argmax(corr) - (len(x) - 1)
    return float(lag_samples * dt)


def co_contraction_index(a, b) -> float:
    """Falconer-Winter co-contraction: mean_t 2*min(a,b)/(a+b). Inputs >= 0."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    s = a + b
    cci = np.where(s > _EPS, 2.0 * np.minimum(a, b) / np.maximum(s, _EPS), 0.0)
    return float(np.mean(cci))
```

Append to `src/rl_mass/finger/gt/__init__.py`:

```python
from rl_mass.finger.gt.metrics import (
    co_contraction_index, pearson_r, r2, rmse, timing_lag,
)

__all__ += ["pearson_r", "rmse", "r2", "timing_lag", "co_contraction_index"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_gt_metrics.py`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/rl_mass/finger/gt/metrics.py src/rl_mass/finger/gt/__init__.py tests/rl_mass/test_gt_metrics.py
git commit -m "feat(gt): activation comparison metrics

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Validation harness (`tools/validate_vs_gt.py`)

**Files:**
- Create: `tools/validate_vs_gt.py`
- Test: `tests/rl_mass/test_validate_vs_gt.py`

**Interfaces:**
- Consumes: `make_env`, `SimulationNN`/`MuscleNN`, `tools._rollout.rollout`, `rl_mass.finger.gt` (`OpenSimStoLoader`, `MuscleNameMap`, `resample_to_common_grid`, metrics). Driven by a JSON config.
- Produces: `compare(config_path) -> dict` writing `metrics.json`, `compare.png`, `compare.csv` to the config's `out_dir`; the returned dict has keys `per_muscle` (name -> {pearson, rmse, r2, timing_lag}), `aggregate` ({mean_pearson, mean_rmse, mean_r2, cci_rl, cci_gt}), and `out_dir`. Config keys (a small JSON, separate from the training config): `train_config`, `checkpoint` (prefix, e.g. `output/arm26/nn/max`), `gt_sto`, `out_dir`, optional `muscle_name_map`, optional `max_steps`.

- [ ] **Step 1: Write the failing test**

```python
# tests/rl_mass/test_validate_vs_gt.py
"""validate_vs_gt end-to-end on Arm26 with random-init nets (smoke).

Uses untrained nets (no checkpoint dependency) by pointing at a config whose
checkpoint is the empty string -> harness builds fresh nets. Asserts the three
output files exist and metrics dict has the right shape."""
import os, sys, json, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from tools.validate_vs_gt import compare

TRAIN_CFG = "data/rl_mass_arm26/config.json"
STO = ("/home/chunleli/Dev/OpenSimExample/Models/Arm26/OutputReference/"
       "StaticOptimization/PhysiologicalResults/arm26_StaticOptimization_activation.sto")

def main():
    d = tempfile.mkdtemp()
    cfg = os.path.join(d, "val.json")
    out = os.path.join(d, "out")
    json.dump({
        "train_config": TRAIN_CFG,
        "checkpoint": "",          # empty -> random-init nets (smoke)
        "gt_sto": STO,
        "out_dir": out,
        "max_steps": 30,
    }, open(cfg, "w"))

    res = compare(cfg)
    assert set(res["per_muscle"]) == {"TRIlong", "TRIlat", "TRImed", "BIClong", "BICshort", "BRA"}
    for m, d_ in res["per_muscle"].items():
        assert set(d_) == {"pearson", "rmse", "r2", "timing_lag"}, (m, d_)
    agg = res["aggregate"]
    assert {"mean_pearson", "mean_rmse", "mean_r2", "cci_rl", "cci_gt"} <= set(agg)
    assert os.path.exists(os.path.join(out, "metrics.json"))
    assert os.path.exists(os.path.join(out, "compare.png"))
    assert os.path.exists(os.path.join(out, "compare.csv"))
    print("OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tests/rl_mass/test_validate_vs_gt.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.validate_vs_gt'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/validate_vs_gt.py
"""Compare rl_mass learned muscle activations against a ground-truth source.

Rolls out a (trained) policy on the reference motion, loads GT activations,
aligns by muscle name + time, and writes per-muscle + aggregate metrics and an
overlay plot. The comparison core is GT-source-agnostic; only the GTLoader is
source-specific (OpenSim Static Optimization here).

Config JSON keys:
    train_config : path to the BimanualEnv training config (defines model+motion)
    checkpoint   : checkpoint prefix (e.g. output/arm26/nn/max); "" -> random nets
    gt_sto       : OpenSim Static-Optimization activation .sto
    out_dir      : directory for metrics.json / compare.png / compare.csv
    muscle_name_map : optional {rl_name: gt_name}
    max_steps    : optional rollout cap (default 10000)

Usage:
    uv run python tools/validate_vs_gt.py -c data/rl_mass_arm26/validate.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_mass.env_factory import make_env
from rl_mass.model import MuscleNN, SimulationNN
from rl_mass.finger.gt import (
    MuscleNameMap, OpenSimStoLoader, co_contraction_index,
    pearson_r, r2, resample_to_common_grid, rmse, timing_lag,
)
from tools._rollout import rollout


def _muscle_names(env):
    import mujoco
    return [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(env.num_muscles)]


def _control_times(n_frames, control_hz):
    return np.arange(n_frames) / float(control_hz)


def compare(config_path):
    cfg = json.load(open(config_path))
    env = make_env(cfg["train_config"])

    model = SimulationNN(env.num_state, env.num_action)
    mm = MuscleNN(env.num_total_related_dofs, env.num_action, env.num_muscles)
    if cfg.get("checkpoint"):
        model.load(cfg["checkpoint"] + ".pt")
        mm.load(cfg["checkpoint"] + "_muscle.pt")
    import torch
    if torch.cuda.is_available():
        model.cuda(); mm.cuda()
    model.eval(); mm.eval()

    _, _, acts, _, _ = rollout(
        env, model, mm, max_steps=int(cfg.get("max_steps", 10000)), deterministic=True)
    rl_names = _muscle_names(env)
    t_rl = _control_times(acts.shape[0], env.control_hz)

    gt = OpenSimStoLoader(cfg["gt_sto"]).load()
    name_map = MuscleNameMap(cfg.get("muscle_name_map"))
    # GT columns reordered to the rl muscle order (rl name -> GT name via map).
    gt_dst = [name_map.mapping.get(n, n) for n in rl_names]
    gt_cols = name_map.reorder(gt.names, gt_dst, gt.activation)  # (T_gt, nu) in rl order

    tc, a_rl, a_gt = resample_to_common_grid(t_rl, acts, gt.time, gt_cols)
    dt = float(tc[1] - tc[0]) if len(tc) > 1 else 1.0

    per = {}
    for j, name in enumerate(rl_names):
        per[name] = {
            "pearson": pearson_r(a_gt[:, j], a_rl[:, j]),
            "rmse": rmse(a_gt[:, j], a_rl[:, j]),
            "r2": r2(a_gt[:, j], a_rl[:, j]),
            "timing_lag": timing_lag(a_gt[:, j], a_rl[:, j], dt),
        }

    groups = json.load(open(cfg["train_config"])).get("muscle_groups", {})
    flex = groups.get("flexor", []); ext = groups.get("extensor", [])
    idx = {n: i for i, n in enumerate(rl_names)}

    def _grp_sum(mat, names):
        cols = [idx[n] for n in names if n in idx]
        return mat[:, cols].sum(axis=1) if cols else np.zeros(mat.shape[0])

    cci_rl = co_contraction_index(_grp_sum(a_rl, flex), _grp_sum(a_rl, ext)) if flex and ext else 0.0
    cci_gt = co_contraction_index(_grp_sum(a_gt, flex), _grp_sum(a_gt, ext)) if flex and ext else 0.0

    agg = {
        "mean_pearson": float(np.mean([per[n]["pearson"] for n in rl_names])),
        "mean_rmse": float(np.mean([per[n]["rmse"] for n in rl_names])),
        "mean_r2": float(np.mean([per[n]["r2"] for n in rl_names])),
        "cci_rl": cci_rl,
        "cci_gt": cci_gt,
    }

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"per_muscle": per, "aggregate": agg}, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)

    # overlay plot
    n = len(rl_names)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.6 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for j, name in enumerate(rl_names):
        axes[j].plot(tc, a_gt[:, j], label="GT (SO)", color="k", lw=1.2)
        axes[j].plot(tc, a_rl[:, j], label="rl_mass", color="tab:red", lw=1.0, alpha=0.8)
        axes[j].set_ylabel(name, fontsize=8)
        axes[j].set_ylim(-0.02, 1.02)
        if j == 0:
            axes[j].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"activation vs GT  mean_r={agg['mean_pearson']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare.png"), dpi=120)
    plt.close(fig)

    # aligned CSV
    header = ["time"] + [f"{n}_gt" for n in rl_names] + [f"{n}_rl" for n in rl_names]
    rows = np.column_stack([tc, a_gt, a_rl])
    np.savetxt(os.path.join(out_dir, "compare.csv"), rows,
               delimiter=",", header=",".join(header), comments="")

    return {"per_muscle": per, "aggregate": agg, "out_dir": out_dir}


def main():
    p = argparse.ArgumentParser(description="Validate rl_mass activations vs GT")
    p.add_argument("-c", "--config", required=True)
    a = p.parse_args()
    res = compare(a.config)
    print(json.dumps(res["aggregate"], indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_validate_vs_gt.py`
Expected: `OK` (metrics computed on random-init nets — values meaningless, shape correct).

- [ ] **Step 5: Run the real comparison against the trained checkpoint**

Pre-req: Task 5 Step 7 full training has produced `output/arm26/nn/max.pt`. Create `data/rl_mass_arm26/validate.json`:

```json
{
    "train_config": "data/rl_mass_arm26/config.json",
    "checkpoint": "output/arm26/nn/max",
    "gt_sto": "/home/chunleli/Dev/OpenSimExample/Models/Arm26/OutputReference/StaticOptimization/PhysiologicalResults/arm26_StaticOptimization_activation.sto",
    "out_dir": "output/arm26/gt_compare"
}
```

Run:
```bash
uv run python tools/validate_vs_gt.py -c data/rl_mass_arm26/validate.json
```
Expected: prints the aggregate metrics JSON; `output/arm26/gt_compare/{metrics.json,compare.png,compare.csv}` exist. Inspect `compare.png` for per-muscle GT-vs-rl_mass overlay.

- [ ] **Step 6: Commit**

```bash
git add tools/validate_vs_gt.py tests/rl_mass/test_validate_vs_gt.py data/rl_mass_arm26/validate.json
git commit -m "feat(gt): activation-vs-GT validation harness (Arm26 + OpenSim SO)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- "validation harness only, no training change" → T10 reads checkpoints, never writes loss. ✓
- "GT-source-agnostic core" → T7 GTLoader ABC, T8 align, T9 metrics have no OpenSim imports; only `OpenSimStoLoader` is source-specific. ✓
- ".osim→MJCF small parser, reusable, errors loudly" → T2–T4; `parse_joints`/`build_mjcf` raise `NotImplementedError` on unhandled constructs. ✓
- ".mot→qpos npz" → T1. ✓
- "Thelen params → MJCF muscle force (f0 in gainprm[2])" → T3, asserted in test. ✓
- "reuse BimanualEnv unchanged" → T5 uses `make_env`; no env edits. ✓
- "metrics: Pearson/RMSE/R²/timing/co-contraction" → T9. ✓
- "outputs metrics.json + compare.png + compare.csv" → T10. ✓
- "PhysiologicalResults .sto as GT" → T7/T10 paths point there. ✓
- "wrap approximation documented" → T4 note. ✓
- EMG / Gait2354 / L1 / vtp-mesh / GT-supervision → spec out-of-scope; no tasks. ✓

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Each code step is complete. The one acknowledged dead line in T10 (`gt_aligned`) is explicitly flagged with removal instructions rather than left ambiguous. ✓

**3. Type consistency:** `GTSeries(names, time, activation)` used consistently T7→T10. `rollout(...)` 5-tuple identical in `tools/_rollout.py` (T6) and consumers (T10). `MuscleNameMap.reorder(src_names, dst_names, values)` signature matches T8 def and T10 call. `resample_to_common_grid(t_a,a,t_b,b)->(tc,a,b)` matches T8/T10. Metric signatures (`pearson_r(x,y)`, `timing_lag(x,y,dt)`, `co_contraction_index(a,b)`) match T9/T10. `config.json` keys (`muscle_groups`, `skeleton_file`, `trajectory_file`) match BimanualEnv + T10 reads. ✓
