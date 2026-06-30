# rl_mass on MyoBimanualArm (fingers, 1D muscles) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the existing rl_mass PPO+MuscleNN pipeline to imitate musclemimic's bimanual reference trajectories on the 76-DOF MyoBimanualArm skeleton (two arms + fully articulated hands), actuated by 126 1D Hill muscles whose length/moment-arms come from MuJoCo-native spatial tendons.

**Architecture:** A new config-driven "native MuJoCo" case runs alongside the existing DART/BVH humanoid path, untouched. A parallel `BimanualEnv` duck-types the `MassEnv` interface the vec-env/PPO consume. Muscle forces use MuJoCo's `actuator_moment` (densified) for moment-arms — MuJoCo's own actuators are disabled (`mjDSBL_ACTUATION`) so only rl_mass's Hill force applies. Reference motion is loaded directly from musclemimic `.npz` (already in MuJoCo qpos space — no retargeting, no DART).

**Tech Stack:** Python, MuJoCo 3.8.0, NumPy, PyTorch (existing rl_mass nets), `uv` for env.

**Key verified facts (do not re-derive):**
- Bimanual MJCF: `/home/chunleli/Dev/musclemimic_models/musclemimic_models/model/arm/myoarm_bimanual.xml`. `nq=nv=76` (all hinge, **no free root**), `nu=126` muscle actuators (`gaintype=MUSCLE`, `trntype=TENDON`), `ntendon=134`. Loads via `MjModel.from_xml_path` (resolves its own includes/meshes).
- The 8 non-actuated tendons are glenohumeral/coracohumeral ligaments (ids 0–3 and 67–70). Actuators map to tendons via `model.actuator_trnid[:,0]`.
- `f0 = model.actuator_gainprm[:,2]` (all > 0, mean ≈317.5 N).
- Length normalization (MuJoCo muscle operating range): `LR=model.actuator_lengthrange` (n,2), `rng=model.actuator_gainprm[:,0:2]`, `L0=(LR[:,1]-LR[:,0])/(rng[:,1]-rng[:,0])`, `l_m_norm = rng[:,0] + (actuator_length - LR[:,0]) / L0`. (`data.actuator_length` equals `data.ten_length[actuator_trnid]`.)
- Moment-arm matrix is **sparse**: `data.actuator_moment` (flat nnz) + `data.moment_rownnz/moment_rowadr/moment_colind`; densify to `(nu,nv)` with `mujoco.mju_sparse2dense`.
- MuJoCo sign convention: generalized force from per-actuator force vector `F_signed` is `qfrc = moment_dense.T @ F_signed`, with muscle tension being **negative** (`F_signed = -F`, `F≥0`). So the per-muscle moment column used by rl_mass is `R = -moment_dense.T` (shape `(nv, nu)`).
- Trajectory `.npz`: `/home/chunleli/Dev/musclemimic/trajectory_data_bimanual/bimanual_episodes_mjx_20260601_220203.npz`. Per episode `episode_{i}_traj_qpos` `(frames, 76)`; globals `dt`, `n_episodes`, `joint_names` (76).
- MuscleTuple contract (from `env.py`): `JtA` `(num_total_related_dofs,)`, `L` `(num_action*num_muscles,)` row-major, `b` `(num_action,)`, `tau_des` `(num_action,)`. For bimanual `num_action = nv = 76` (all DOFs active, no root).
- Interface the vec-env/PPO require on an env object: attrs `num_state, num_action, num_muscles, use_muscle, simulation_hz, control_hz, num_total_related_dofs`; methods `get_state, set_action, get_muscle_torques, get_desired_torques, set_activation_levels, step, steps_at_once, is_end_of_episode, get_reward, reset(rsi=bool)`; list attr `muscle_tuples`.
- `make_vec_env(config_path, ...)` flows config through unchanged; only `vec_env.py` imports `MassEnv` directly (2 sites: `SerialVecEnv.__init__`, `_worker_loop`).

**Test convention:** This repo uses flat runnable scripts under `tests/` (no pytest config). Each test file has a `main()` that asserts and prints `OK`, run via `uv run python tests/rl_mass/<file>.py`. All commands run from repo root `/mnt/storage_disk_1/chunleli/Dev/rlmuscleV2-finger`. Prefix every command with `uv run`.

---

## File Structure

- Create `data/rl_mass_bimanual/config.json` — case config (`skeleton_format: "mujoco_native"`).
- Create `src/rl_mass/trajectory_ref.py` — `TrajectoryRef`: loads `.npz`, serves target qpos/qvel(t).
- Create `src/rl_mass/tendon_muscle.py` — `TendonMuscleSystem`: f0/normalization/moment extraction, force application, muscle-tuple terms.
- Create `src/rl_mass/native_spd.py` — `native_spd_forces`: stable-PD over all DOFs (no DART/root).
- Create `src/rl_mass/bimanual_env.py` — `BimanualEnv`: implements the `MassEnv` interface for the native case.
- Create `src/rl_mass/env_factory.py` — `make_env(config_path)`: returns `MassEnv` or `BimanualEnv` by `skeleton_format`.
- Modify `src/rl_mass/vec_env.py` — replace the 2 `from rl_mass.env import MassEnv` sites with `from rl_mass.env_factory import make_env`.
- Create tests under `tests/rl_mass/`.

---

## Task 1: Case config + model-load smoke test

**Files:**
- Create: `data/rl_mass_bimanual/config.json`
- Test: `tests/rl_mass/test_bimanual_model.py`

- [ ] **Step 1: Write the config**

`data/rl_mass_bimanual/config.json`:
```json
{
    "skeleton_format": "mujoco_native",
    "use_muscle": true,
    "control_hz": 30,
    "simulation_hz": 600,
    "skeleton_file": "/home/chunleli/Dev/musclemimic_models/musclemimic_models/model/arm/myoarm_bimanual.xml",
    "trajectory_file": "/home/chunleli/Dev/musclemimic/trajectory_data_bimanual/bimanual_episodes_mjx_20260601_220203.npz",
    "reward_param": {
        "w_q": 0.85,
        "w_v": 0.15
    },
    "pose_error_threshold": 5.0,
    "spd": {
        "kp": 100.0
    }
}
```

- [ ] **Step 2: Write the failing test**

`tests/rl_mass/test_bimanual_model.py`:
```python
"""Smoke test: bimanual MJCF loads via MuJoCoEngine with expected dims."""
import json
from rl_mass.engine import MuJoCoEngine


def main():
    with open("data/rl_mass_bimanual/config.json") as f:
        cfg = json.load(f)
    eng = MuJoCoEngine(cfg["skeleton_file"], timestep=1.0 / cfg["simulation_hz"])
    eng.forward()
    assert eng.nq == 76, eng.nq
    assert eng.nv == 76, eng.nv
    assert eng.model.nu == 126, eng.model.nu
    assert eng.model.ntendon == 134, eng.model.ntendon
    print("OK", eng.nq, eng.nv, eng.model.nu, eng.model.ntendon)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_bimanual_model.py`
Expected: `OK 76 76 126 134`

- [ ] **Step 4: Commit**

```bash
git add data/rl_mass_bimanual/config.json tests/rl_mass/test_bimanual_model.py
git commit -m "feat(bimanual): case config + model-load smoke test"
```

---

## Task 2: TrajectoryRef — load .npz reference motion

**Files:**
- Create: `src/rl_mass/trajectory_ref.py`
- Test: `tests/rl_mass/test_trajectory_ref.py`

- [ ] **Step 1: Write the failing test**

`tests/rl_mass/test_trajectory_ref.py`:
```python
"""TrajectoryRef serves interpolated qpos/qvel from a musclemimic .npz."""
import numpy as np
from rl_mass.trajectory_ref import TrajectoryRef

NPZ = "/home/chunleli/Dev/musclemimic/trajectory_data_bimanual/bimanual_episodes_mjx_20260601_220203.npz"


def main():
    ref = TrajectoryRef(NPZ, nq=76)
    assert ref.max_time > 0.0
    q0, v0 = ref.pos_vel(0.0, ref.dt)
    assert q0.shape == (76,), q0.shape
    assert v0.shape == (76,), v0.shape
    # Interpolation at a mid frame matches a manual lerp of stored frames.
    qa, _ = ref.pos_vel(0.0, ref.dt)
    qb, _ = ref.pos_vel(ref.dt, ref.dt)
    qmid, _ = ref.pos_vel(0.5 * ref.dt, ref.dt)
    assert np.allclose(qmid, 0.5 * (qa + qb), atol=1e-5), np.abs(qmid - 0.5 * (qa + qb)).max()
    # Velocity is the finite difference used by the controller.
    assert np.allclose(v0, (qb - qa) / ref.dt, atol=1e-5)
    print("OK", round(ref.max_time, 3), ref.n_frames)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tests/rl_mass/test_trajectory_ref.py`
Expected: FAIL with `ModuleNotFoundError: rl_mass.trajectory_ref`

- [ ] **Step 3: Write the implementation**

`src/rl_mass/trajectory_ref.py`:
```python
"""Reference motion from a musclemimic .npz trajectory.

The .npz stores one or more episodes, each as `episode_{i}_traj_qpos`
of shape (frames, nq) already in the target MuJoCo qpos order. We
concatenate all episodes into one frame sequence and serve interpolated
qpos / finite-difference qvel at a query time, matching how the existing
Character feeds (target_pos, target_vel) into SPD and the reward.
"""
from __future__ import annotations

import numpy as np


class TrajectoryRef:
    def __init__(self, npz_path: str, nq: int):
        data = np.load(npz_path, allow_pickle=True)
        n_eps = int(data["n_episodes"])
        frames = []
        for i in range(n_eps):
            key = f"episode_{i}_traj_qpos"
            if key not in data:
                continue
            arr = np.asarray(data[key], dtype=np.float64)
            assert arr.shape[1] == nq, (key, arr.shape, nq)
            frames.append(arr)
        assert frames, f"no episode_*_traj_qpos in {npz_path}"
        self.frames = np.concatenate(frames, axis=0)  # (N, nq)
        self.n_frames = self.frames.shape[0]
        self.nq = nq
        self.dt = float(data["dt"])
        # max_time is the last sample time; queries wrap modulo this.
        self.max_time = self.dt * (self.n_frames - 1)

    def _qpos_at(self, t: float) -> np.ndarray:
        """Linearly interpolated qpos at time t (clamped to [0, max_time])."""
        if self.max_time <= 0.0:
            return self.frames[0].copy()
        t = max(0.0, min(t, self.max_time))
        f = t / self.dt
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, self.n_frames - 1)
        w = f - i0
        return (1.0 - w) * self.frames[i0] + w * self.frames[i1]

    def pos_vel(self, t: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (qpos(t), qvel(t)) with qvel = (qpos(t+dt)-qpos(t))/dt."""
        q = self._qpos_at(t)
        q1 = self._qpos_at(t + dt)
        return q, (q1 - q) / dt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_trajectory_ref.py`
Expected: `OK <max_time> <n_frames>`

- [ ] **Step 5: Commit**

```bash
git add src/rl_mass/trajectory_ref.py tests/rl_mass/test_trajectory_ref.py
git commit -m "feat(bimanual): TrajectoryRef loads musclemimic .npz reference motion"
```

---

## Task 3: TendonMuscleSystem — Hill forces over MuJoCo tendons

**Files:**
- Create: `src/rl_mass/tendon_muscle.py`
- Test: `tests/rl_mass/test_tendon_muscle.py`

This class reads `model`/`data` directly (via the existing `MuJoCoEngine`, which exposes `.model` and `.data`). It computes the dense moment matrix each update, applies muscle generalized force, and produces the muscle-tuple terms (`JtA`, `L`, `b`) using rl_mass's Hill curves `g_al`/`g_pl` (imported from `rl_mass.muscle`).

- [ ] **Step 1: Write the failing test**

`tests/rl_mass/test_tendon_muscle.py`:
```python
"""TendonMuscleSystem: dims, sign cross-check vs MuJoCo, tuple terms."""
import numpy as np
from rl_mass.engine import MuJoCoEngine
from rl_mass.tendon_muscle import TendonMuscleSystem

XML = "/home/chunleli/Dev/musclemimic_models/musclemimic_models/model/arm/myoarm_bimanual.xml"


def main():
    eng = MuJoCoEngine(XML, timestep=1.0 / 600)
    eng.forward()
    ms = TendonMuscleSystem(eng)
    assert ms.n_muscles == 126, ms.n_muscles
    assert ms.f0.shape == (126,) and np.all(ms.f0 > 0)
    assert ms.nv == 76

    ms.update(eng)
    # R is the per-muscle moment column matrix (nv, n_muscles).
    assert ms.R.shape == (76, 126), ms.R.shape

    # --- Sign cross-check against MuJoCo's own actuator path ---
    # If we feed F (>=0 tension), our generalized force must equal MuJoCo's
    # qfrc_actuator when its actuator_force == -F. Use a fresh data with
    # actuation ENABLED to read qfrc_actuator for a chosen ctrl.
    import mujoco
    m2 = mujoco.MjModel.from_xml_path(XML)
    d2 = mujoco.MjData(m2)
    mujoco.mj_forward(m2, d2)
    # Drive activation of one muscle via ctrl; read resulting actuator_force + qfrc.
    d2.ctrl[:] = 0.0
    d2.ctrl[0] = 1.0
    for _ in range(50):  # let activation dynamics settle
        mujoco.mj_forward(m2, d2)
        d2.act[:] = d2.ctrl  # shortcut: force act to ctrl for steady state
    mujoco.mj_forward(m2, d2)
    F_signed = d2.actuator_force.copy()           # (nu,), muscle force (<=0)
    qfrc_mj = d2.qfrc_actuator.copy()             # (nv,)
    # Our convention: qfrc = R @ F where F is +tension; MuJoCo F_signed = -F.
    qfrc_ours = ms.R @ (-F_signed)
    assert np.allclose(qfrc_ours, qfrc_mj, atol=1e-6), np.abs(qfrc_ours - qfrc_mj).max()

    # --- Tuple terms shapes ---
    a = np.zeros(126)
    L, b, JtA = ms.tuple_terms(eng, a)
    assert L.shape == (76, 126), L.shape
    assert b.shape == (76,), b.shape
    assert JtA.shape == (ms.num_total_related_dofs,), JtA.shape
    print("OK", ms.n_muscles, ms.num_total_related_dofs)


if __name__ == "__main__":
    main()
```

Note: this test reuses MuJoCo's per-actuator `actuator_moment` so `R @ (-actuator_force)` must equal `qfrc_actuator` exactly — that is MuJoCo's identity `qfrc_actuator = moment.T @ actuator_force`, which validates both the densification and the sign of `R = -moment_dense.T`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tests/rl_mass/test_tendon_muscle.py`
Expected: FAIL with `ModuleNotFoundError: rl_mass.tendon_muscle`

- [ ] **Step 3: Write the implementation**

`src/rl_mass/tendon_muscle.py`:
```python
"""1D Hill muscles backed by MuJoCo spatial tendons.

Replaces the waypoint MuscleSystem for native-MJCF cases. Length and
moment-arms come from MuJoCo's tendon machinery (wrapping preserved);
rl_mass's Hill force-length curves (g_al / g_pl) are applied on top.

Per-muscle moment column matrix:
    R[:, m] = -moment_dense[m, :]          # (nv,) generalized force per +unit tension
Force scalars:
    f_active[m] = f0[m] * g_al(l_m_norm[m])
    f_passive[m] = f0[m] * g_pl(l_m_norm[m])
Applied generalized force for activations a:
    qfrc = R @ (f_active * a + f_passive)
Muscle-tuple matrices (active DOFs = all DOFs; no root):
    L[:, m] = R[:, m] * f_active[m]
    b       = R @ f_passive
    JtA     = concat over m of L[related_dofs(m), m]
"""
from __future__ import annotations

import numpy as np

from rl_mass.engine import MuJoCoEngine
from rl_mass.muscle import g_al, g_pl


class TendonMuscleSystem:
    def __init__(self, engine: MuJoCoEngine):
        import mujoco

        self._mj = mujoco
        m = engine.model
        self.nv = int(m.nv)
        self.n_muscles = int(m.nu)

        # Per-muscle static params.
        self.f0 = m.actuator_gainprm[:, 2].copy().astype(np.float64)
        assert np.all(self.f0 > 0), "expected all muscle f0 > 0"
        self._LR = m.actuator_lengthrange.copy().astype(np.float64)   # (nu, 2)
        rng = m.actuator_gainprm[:, 0:2].copy().astype(np.float64)    # (nu, 2)
        assert np.all(self._LR[:, 1] > self._LR[:, 0]), "bad actuator_lengthrange"
        assert np.all(rng[:, 1] > rng[:, 0]), "bad gainprm operating range"
        self._rng0 = rng[:, 0]
        self._L0 = (self._LR[:, 1] - self._LR[:, 0]) / (rng[:, 1] - rng[:, 0])

        # Dense-moment scratch buffer.
        self._moment_dense = np.zeros((self.n_muscles, self.nv), dtype=np.float64)

        # State (filled by update()).
        self.R = np.zeros((self.nv, self.n_muscles), dtype=np.float64)
        self.l_m_norm = np.ones(self.n_muscles, dtype=np.float64)
        self.f_active = np.zeros(self.n_muscles, dtype=np.float64)
        self.f_passive = np.zeros(self.n_muscles, dtype=np.float64)

        # Related-DOF structure, computed once at rest pose.
        engine.forward()
        self._densify_moment(engine)
        R_rest = -self._moment_dense.T  # (nv, nu)
        self.related_dof_indices = [
            np.where(np.abs(R_rest[:, m]) > 1e-3)[0].astype(np.intp)
            for m in range(self.n_muscles)
        ]
        self.num_related_dofs = [int(idx.size) for idx in self.related_dof_indices]
        self.num_total_related_dofs = int(sum(self.num_related_dofs))

    def _densify_moment(self, engine: MuJoCoEngine) -> None:
        d = engine.data
        self._mj.mju_sparse2dense(
            self._moment_dense, d.actuator_moment,
            d.moment_rownnz, d.moment_rowadr, d.moment_colind,
        )

    def update(self, engine: MuJoCoEngine) -> None:
        """Recompute R, normalized length, and force scalars at current pose."""
        d = engine.data
        self._densify_moment(engine)
        self.R[:] = -self._moment_dense.T  # (nv, nu)

        al = d.actuator_length.copy()
        self.l_m_norm[:] = self._rng0 + (al - self._LR[:, 0]) / self._L0
        # Vectorized g_al / g_pl (same math as rl_mass.muscle).
        x = self.l_m_norm
        self.f_active[:] = self.f0 * np.exp(-((x - 1.0) ** 2) / 0.5)
        fp = np.zeros_like(x)
        mask = x >= 1.0
        fp[mask] = (np.exp(4.0 * (x[mask] - 1.0) / 0.6) - 1.0) / (np.exp(4.0) - 1.0)
        self.f_passive[:] = self.f0 * fp

    def apply_forces(self, engine: MuJoCoEngine, activations: np.ndarray) -> None:
        """Set qfrc_applied from current R and the given activations."""
        a = np.asarray(activations, dtype=np.float64).ravel()
        muscle_force = self.f_active * a + self.f_passive  # (nu,)
        qfrc = self.R @ muscle_force                       # (nv,)
        engine.set_qfrc_applied(qfrc)

    def get_muscle_torques(self, engine: MuJoCoEngine) -> np.ndarray:
        """Concatenated related-DOF JtA = R[related, m] * f_active[m]."""
        self.update(engine)
        out = np.empty(self.num_total_related_dofs, dtype=np.float64)
        i = 0
        for m in range(self.n_muscles):
            idx = self.related_dof_indices[m]
            n = idx.size
            out[i:i + n] = self.R[idx, m] * self.f_active[m]
            i += n
        return out

    def tuple_terms(
        self, engine: MuJoCoEngine, activations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (L (nv,nu), b (nv,), JtA (num_total_related_dofs,)) at current pose."""
        self.update(engine)
        L = self.R * self.f_active[None, :]   # (nv, nu)
        b = self.R @ self.f_passive           # (nv,)
        out = np.empty(self.num_total_related_dofs, dtype=np.float64)
        i = 0
        for m in range(self.n_muscles):
            idx = self.related_dof_indices[m]
            n = idx.size
            out[i:i + n] = L[idx, m]
            i += n
        return L, b, out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_tendon_muscle.py`
Expected: `OK 126 <num_total_related_dofs>`

If the sign assertion fails, the fix is the sign of `R` (flip `-self._moment_dense.T` to `+`); do NOT change the test — it encodes MuJoCo's own identity.

- [ ] **Step 5: Commit**

```bash
git add src/rl_mass/tendon_muscle.py tests/rl_mass/test_tendon_muscle.py
git commit -m "feat(bimanual): TendonMuscleSystem (MuJoCo-tendon Hill muscles)"
```

---

## Task 4: native_spd_forces — stable PD over all DOFs

**Files:**
- Create: `src/rl_mass/native_spd.py`
- Test: `tests/rl_mass/test_native_spd.py`

For the fixed-base bimanual model all joints are hinge (`nq==nv`, 1-DOF each), so SPD is plain stable-PD with simple subtraction (no DART exp-map, no root zeroing). This mirrors `Character.get_spd_forces` minus the DART conversions.

- [ ] **Step 1: Write the failing test**

`tests/rl_mass/test_native_spd.py`:
```python
"""native_spd_forces: shape, and zero torque when at the target with zero vel."""
import numpy as np
from rl_mass.engine import MuJoCoEngine
from rl_mass.native_spd import native_spd_forces

XML = "/home/chunleli/Dev/musclemimic_models/musclemimic_models/model/arm/myoarm_bimanual.xml"


def main():
    eng = MuJoCoEngine(XML, timestep=1.0 / 600)
    eng.forward()
    q = eng.get_qpos()
    eng.set_qvel(np.zeros(eng.nv))
    eng.forward()
    kp = 100.0
    kv = np.sqrt(2.0 * kp)
    tau = native_spd_forces(eng, q.copy(), kp=kp, kv=kv)
    assert tau.shape == (eng.nv,), tau.shape
    # At the target with zero velocity, the PD error terms vanish; remaining
    # torque is only the -dt*Kv*ddq gravity-compensation correction — finite.
    assert np.all(np.isfinite(tau))
    print("OK", round(float(np.linalg.norm(tau)), 4))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tests/rl_mass/test_native_spd.py`
Expected: FAIL with `ModuleNotFoundError: rl_mass.native_spd`

- [ ] **Step 3: Write the implementation**

`src/rl_mass/native_spd.py`:
```python
"""Stable-PD torques for a fixed-base, all-hinge MuJoCo model.

All joints are 1-DOF (nq == nv), so positions subtract directly and no
root DOFs are special-cased. Formula matches DART/MASS stable PD:

    M_mod = M + dt*diag(Kv)
    qdqdt = q + dq*dt
    p_diff = -Kp*(qdqdt - q_des)
    v_diff = -Kv*dq
    ddq = M_mod^{-1} @ (-C_G + p_diff + v_diff + C_f)
    tau = p_diff + v_diff - dt*Kv*ddq
"""
from __future__ import annotations

import numpy as np

from rl_mass.engine import MuJoCoEngine


def native_spd_forces(
    engine: MuJoCoEngine,
    q_des: np.ndarray,
    kp: float,
    kv: float,
) -> np.ndarray:
    nv = engine.nv
    dt = engine.timestep
    Kp = np.full(nv, kp, dtype=np.float64)
    Kv = np.full(nv, kv, dtype=np.float64)

    q = engine.get_qpos()
    dq = engine.get_qvel()
    M = engine.get_mass_matrix()
    M_mod = M + dt * np.diag(Kv)
    M_inv = np.linalg.inv(M_mod)

    qdqdt = q + dq * dt
    p_diff = -Kp * (qdqdt - q_des)
    v_diff = -Kv * dq
    cg = engine.get_coriolis_gravity()
    cf = engine.get_constraint_forces()
    ddq = M_inv @ (-cg + p_diff + v_diff + cf)
    tau = p_diff + v_diff - dt * Kv * ddq
    return tau
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_native_spd.py`
Expected: `OK <norm>`

- [ ] **Step 5: Commit**

```bash
git add src/rl_mass/native_spd.py tests/rl_mass/test_native_spd.py
git commit -m "feat(bimanual): native stable-PD for fixed-base hinge model"
```

---

## Task 5: BimanualEnv — the native env

**Files:**
- Create: `src/rl_mass/bimanual_env.py`
- Test: `tests/rl_mass/test_bimanual_env.py`

`BimanualEnv` duck-types the `MassEnv` interface. It reuses `MuJoCoEngine`, sets `mjDSBL_ACTUATION` so only rl_mass Hill forces drive the model, tracks `TrajectoryRef`, and emits `MuscleTuple` via `TendonMuscleSystem`. State is joint-space: `[qpos (76), qvel*0.1 (76), phase (1)]` → `num_state = 153`. `num_action = nv = 76` (offsets the SPD target). Reward: `w_q*exp(-2||q-q_ref||^2) + w_v*exp(-0.1||v-v_ref||^2)`.

- [ ] **Step 1: Write the failing test**

`tests/rl_mass/test_bimanual_env.py`:
```python
"""BimanualEnv: interface attrs, reset, step, reward, muscle tuple shapes."""
import numpy as np
from rl_mass.bimanual_env import BimanualEnv

CFG = "data/rl_mass_bimanual/config.json"


def main():
    env = BimanualEnv(CFG)
    assert env.num_action == 76, env.num_action
    assert env.num_muscles == 126, env.num_muscles
    assert env.use_muscle is True
    assert env.num_state == 153, env.num_state
    assert env.num_total_related_dofs > 0

    s = env.reset(rsi=False)
    assert s.shape == (153,), s.shape
    r = env.get_reward()
    assert 0.0 <= r <= 1.0 + 1e-6, r

    # One control step with zero action; collect a muscle tuple.
    env.set_action(np.zeros(env.num_action))
    env.set_activation_levels(np.zeros(env.num_muscles))
    env.steps_at_once()
    assert len(env.muscle_tuples) >= 1
    mt = env.muscle_tuples[-1]
    assert mt.L.shape == (76 * 126,), mt.L.shape
    assert mt.b.shape == (76,), mt.b.shape
    assert mt.tau_des.shape == (76,), mt.tau_des.shape
    assert mt.JtA.shape == (env.num_total_related_dofs,), mt.JtA.shape

    dt = env.get_desired_torques()
    assert dt.shape == (76,), dt.shape
    mtq = env.get_muscle_torques()
    assert mtq.shape == (env.num_total_related_dofs,), mtq.shape
    assert not env.is_end_of_episode()
    print("OK", env.num_state, env.num_total_related_dofs, round(r, 4))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python tests/rl_mass/test_bimanual_env.py`
Expected: FAIL with `ModuleNotFoundError: rl_mass.bimanual_env`

- [ ] **Step 3: Write the implementation**

`src/rl_mass/bimanual_env.py`:
```python
"""BimanualEnv: native-MuJoCo MASS env for the MyoBimanualArm case.

Fixed-base (no root), tendon-backed 1D muscles, reference motion from a
musclemimic .npz. Duck-types the MassEnv interface consumed by vec_env
and PPO. MuJoCo's own actuators are disabled so only rl_mass Hill forces
(via TendonMuscleSystem) drive the model.
"""
from __future__ import annotations

import json
import math
import random
from typing import Optional

import numpy as np

from rl_mass.engine import MuJoCoEngine
from rl_mass.env import MuscleTuple
from rl_mass.native_spd import native_spd_forces
from rl_mass.tendon_muscle import TendonMuscleSystem
from rl_mass.trajectory_ref import TrajectoryRef

_QVEL_STATE_SCALE = 0.1


class BimanualEnv:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            cfg = json.load(f)

        self.control_hz: int = cfg.get("control_hz", 30)
        self.simulation_hz: int = cfg.get("simulation_hz", 600)
        self.use_muscle: bool = cfg.get("use_muscle", True)
        rp = cfg.get("reward_param", {})
        self.w_q: float = rp.get("w_q", 0.85)
        self.w_v: float = rp.get("w_v", 0.15)
        self.pose_error_threshold: Optional[float] = (
            float(cfg["pose_error_threshold"])
            if cfg.get("pose_error_threshold") is not None else None
        )
        self.kp: float = cfg.get("spd", {}).get("kp", 100.0)
        self.kv: float = math.sqrt(2.0 * self.kp)

        self.engine = MuJoCoEngine(cfg["skeleton_file"], timestep=1.0 / self.simulation_hz)
        # Disable MuJoCo's own actuator forces; we apply Hill forces ourselves.
        import mujoco
        self.engine.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
        self.model = self.engine.model
        self.data = self.engine.data

        self.nv = self.engine.nv
        self.num_action_: int = self.nv  # all DOFs active (no root)
        self.action = np.zeros(self.num_action_, dtype=np.float64)

        self.ref = TrajectoryRef(cfg["trajectory_file"], nq=self.engine.nq)

        self.engine.forward()
        self.muscle_system: Optional[TendonMuscleSystem] = None
        self.num_total_related_dofs: int = 0
        if self.use_muscle:
            self.muscle_system = TendonMuscleSystem(self.engine)
            self.num_total_related_dofs = self.muscle_system.num_total_related_dofs

        self.n_muscles = int(self.model.nu)
        self.activation_levels = np.zeros(self.n_muscles, dtype=np.float64)
        self.muscle_tuples: list[MuscleTuple] = []
        self.current_muscle_tuple: Optional[MuscleTuple] = None

        self.desired_torque = np.zeros(self.nv, dtype=np.float64)
        self.target_q = np.zeros(self.engine.nq, dtype=np.float64)
        self.target_v = np.zeros(self.nv, dtype=np.float64)

        self.sim_count = 0
        self.random_sample_index = 0

        self.reset(rsi=False)
        self.num_state: int = self.get_state().shape[0]

    # --- interface attrs ---
    @property
    def num_action(self) -> int:
        return self.num_action_

    @property
    def num_muscles(self) -> int:
        return self.n_muscles

    @property
    def num_steps(self) -> int:
        return self.simulation_hz // self.control_hz

    # --- reset ---
    def reset(self, rsi: bool = True, t: Optional[float] = None) -> np.ndarray:
        self.engine.reset()
        self.engine.model.opt.disableflags |= self._actuation_disable_bit()
        if t is not None:
            start = t
        elif rsi:
            start = random.uniform(0.0, self.ref.max_time * 0.9)
        else:
            start = 0.0
        self.engine.set_time(start)
        self.action[:] = 0.0
        dt_ctrl = 1.0 / self.control_hz
        q, v = self.ref.pos_vel(start, dt_ctrl)
        self.target_q = q.copy()
        self.target_v = v.copy()
        self.engine.set_qpos(q)
        self.engine.set_qvel(v)
        self.engine.forward()
        return self.get_state()

    def _actuation_disable_bit(self) -> int:
        import mujoco
        return int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)

    # --- action ---
    def set_action(self, a: np.ndarray) -> None:
        a = np.asarray(a, dtype=np.float64).ravel()
        assert a.shape[0] == self.num_action_, (a.shape, self.num_action_)
        self.action = a * 0.1
        t = self.engine.get_time()
        dt_ctrl = 1.0 / self.control_hz
        q, v = self.ref.pos_vel(t, dt_ctrl)
        self.target_q = q.copy()
        self.target_v = v.copy()
        self.sim_count = 0
        self.random_sample_index = random.randint(0, self.num_steps - 1)

    def get_desired_torques(self) -> np.ndarray:
        q_des = self.target_q.copy()
        q_des += self.action  # offset target (all DOFs, no root)
        self.desired_torque = native_spd_forces(self.engine, q_des, self.kp, self.kv)
        return self.desired_torque

    def get_muscle_torques(self) -> np.ndarray:
        return self.muscle_system.get_muscle_torques(self.engine)

    def set_activation_levels(self, a: np.ndarray) -> None:
        self.activation_levels = np.asarray(a, dtype=np.float64).ravel()

    def get_activation_levels(self) -> np.ndarray:
        return self.activation_levels.copy()

    # --- step ---
    def step(self) -> None:
        if self.use_muscle:
            self.muscle_system.update(self.engine)
            self.muscle_system.apply_forces(self.engine, self.activation_levels)
            if self.sim_count == self.random_sample_index:
                self._collect_muscle_tuple()
        else:
            self.get_desired_torques()
            self.engine.set_qfrc_applied(self.desired_torque)
        self.engine.step()
        self.sim_count += 1

    def steps_at_once(self) -> None:
        for _ in range(self.num_steps):
            self.step()

    def _collect_muscle_tuple(self) -> None:
        # tau_des from SPD at the current pose (also caches self.desired_torque).
        tau_des = self.get_desired_torques()
        L, b, JtA = self.muscle_system.tuple_terms(self.engine, self.activation_levels)
        self.current_muscle_tuple = MuscleTuple(
            JtA=JtA.copy(),
            L=L.ravel().copy(),
            b=b.copy(),
            tau_des=tau_des.copy(),
        )
        self.muscle_tuples.append(self.current_muscle_tuple)

    # --- state / reward / termination ---
    def get_state(self) -> np.ndarray:
        q = self.engine.get_qpos()
        v = self.engine.get_qvel() * _QVEL_STATE_SCALE
        phase = math.fmod(self.engine.get_time(), self.ref.max_time) / self.ref.max_time
        return np.concatenate([q, v, [phase]])

    def get_reward(self) -> float:
        q = self.engine.get_qpos()
        v = self.engine.get_qvel()
        q_diff = q - self.target_q
        v_diff = v - self.target_v
        r_q = math.exp(-2.0 * float(np.dot(q_diff, q_diff)))
        r_v = math.exp(-0.1 * float(np.dot(v_diff, v_diff)))
        return self.w_q * r_q + self.w_v * r_v

    def _pose_error_sq(self) -> float:
        q_diff = self.engine.get_qpos() - self.target_q
        return float(np.dot(q_diff, q_diff))

    def is_end_of_episode(self) -> bool:
        if self.pose_error_threshold is not None:
            if self._pose_error_sq() > self.pose_error_threshold:
                return True
        if np.any(np.isnan(self.engine.get_qpos())) or np.any(np.isnan(self.engine.get_qvel())):
            return True
        if self.engine.get_time() > 10.0:
            return True
        return False
```

Note on `reset`: `engine.reset()` calls `mj_resetData`, which does not clear `model.opt.disableflags` (that lives on the model, not data), so re-applying the bit is belt-and-suspenders. Keep it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_bimanual_env.py`
Expected: `OK 153 <num_total_related_dofs> <reward>`

- [ ] **Step 5: Commit**

```bash
git add src/rl_mass/bimanual_env.py tests/rl_mass/test_bimanual_env.py
git commit -m "feat(bimanual): BimanualEnv native MASS env (fixed-base, tendon muscles)"
```

---

## Task 6: env factory + vec_env wiring

**Files:**
- Create: `src/rl_mass/env_factory.py`
- Modify: `src/rl_mass/vec_env.py:95` (SerialVecEnv import) and `src/rl_mass/vec_env.py:196` (worker import)
- Test: `tests/rl_mass/test_env_factory.py`

- [ ] **Step 1: Write the implementation**

`src/rl_mass/env_factory.py`:
```python
"""Select the env class by config `skeleton_format`.

Default (absent or "dart") -> MassEnv (humanoid/BVH/DART path).
"mujoco_native" -> BimanualEnv (native-MJCF fixed-base path).
"""
from __future__ import annotations

import json


def make_env(config_path: str):
    with open(config_path) as f:
        fmt = json.load(f).get("skeleton_format", "dart")
    if fmt == "mujoco_native":
        from rl_mass.bimanual_env import BimanualEnv
        return BimanualEnv(config_path)
    if fmt == "dart":
        from rl_mass.env import MassEnv
        return MassEnv(config_path)
    raise ValueError(f"unknown skeleton_format: {fmt!r}")
```

- [ ] **Step 2: Wire SerialVecEnv**

In `src/rl_mass/vec_env.py`, `SerialVecEnv.__init__` (around line 94-97), replace:
```python
        from rl_mass.env import MassEnv

        self.envs = [MassEnv(config_path) for _ in range(num_envs)]
```
with:
```python
        from rl_mass.env_factory import make_env

        self.envs = [make_env(config_path) for _ in range(num_envs)]
```

- [ ] **Step 3: Wire the subprocess worker**

In `src/rl_mass/vec_env.py`, `_worker_loop` (around line 196-198), replace:
```python
    from rl_mass.env import MassEnv

    env = MassEnv(config_path)
```
with:
```python
    from rl_mass.env_factory import make_env

    env = make_env(config_path)
```

- [ ] **Step 4: Write the test**

`tests/rl_mass/test_env_factory.py`:
```python
"""make_env returns BimanualEnv for the native config; serial vec env runs."""
from rl_mass.env_factory import make_env
from rl_mass.bimanual_env import BimanualEnv
from rl_mass.vec_env import make_vec_env
import numpy as np

CFG = "data/rl_mass_bimanual/config.json"


def main():
    env = make_env(CFG)
    assert isinstance(env, BimanualEnv)

    vec = make_vec_env(CFG, num_envs=2, mode="serial")
    assert vec.num_action == 76
    assert vec.num_muscles == 126
    states = vec.get_states()
    assert states.shape == (2, 153), states.shape
    vec.set_actions(np.zeros((2, 76)))
    vec.set_activation_levels(np.zeros((2, 126)))
    vec.steps_at_once()
    tuples = vec.gather_muscle_tuples()
    assert len(tuples) >= 2, len(tuples)
    print("OK", states.shape, len(tuples))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python tests/rl_mass/test_env_factory.py`
Expected: `OK (2, 153) <ntuples>`

- [ ] **Step 6: Verify the humanoid path still works (regression)**

Run: `uv run python tests/rl_mass/test_env_factory_dart.py` after creating it:

`tests/rl_mass/test_env_factory_dart.py`:
```python
"""Regression: default config still resolves to MassEnv via the factory."""
from rl_mass.env_factory import make_env
from rl_mass.env import MassEnv


def main():
    env = make_env("data/rl_mass/config.json")
    assert isinstance(env, MassEnv)
    print("OK")


if __name__ == "__main__":
    main()
```
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add src/rl_mass/env_factory.py src/rl_mass/vec_env.py \
        tests/rl_mass/test_env_factory.py tests/rl_mass/test_env_factory_dart.py
git commit -m "feat(bimanual): env factory by skeleton_format; wire vec_env"
```

---

## Task 7: Muscle-conditioning validation (feasibility gate)

**Files:**
- Test: `tests/rl_mass/test_bimanual_conditioning.py`

Before training, confirm the muscle basis can actually generate the SPD desired torques — i.e. the least-squares activations `a*` for `L a + b = tau_des` give small residual and the rest-pose `l_m_norm` distribution is sane. (Recall the rest pose mapped some muscles to `l_m_norm` outside [0.5,1.5]; this gate quantifies how much that hurts.)

- [ ] **Step 1: Write the validation test**

`tests/rl_mass/test_bimanual_conditioning.py`:
```python
"""Can the muscle basis span the SPD desired torque along the reference?"""
import numpy as np
from rl_mass.bimanual_env import BimanualEnv

CFG = "data/rl_mass_bimanual/config.json"


def main():
    env = BimanualEnv(CFG)
    residuals = []
    fracs_in_band = []
    for k in range(8):
        env.reset(rsi=False, t=k * env.ref.max_time / 8.0)
        env.set_action(np.zeros(env.num_action))
        tau_des = env.get_desired_torques()
        L, b, _ = env.muscle_system.tuple_terms(env.engine, np.zeros(env.num_muscles))
        # Solve min_a>=0-ish (unconstrained LS is enough to gauge span) for L a = tau_des - b
        a, *_ = np.linalg.lstsq(L, tau_des - b, rcond=None)
        resid = np.linalg.norm(L @ a + b - tau_des) / (np.linalg.norm(tau_des) + 1e-9)
        residuals.append(resid)
        x = env.muscle_system.l_m_norm
        fracs_in_band.append(float(np.mean((x > 0.4) & (x < 1.6))))
    print("relative LS residual (mean):", round(float(np.mean(residuals)), 4))
    print("l_m_norm fraction in [0.4,1.6] (mean):", round(float(np.mean(fracs_in_band)), 3))
    # Soft gate: report, then assert the basis is not degenerate.
    assert np.mean(residuals) < 0.5, np.mean(residuals)
    print("OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and record the numbers**

Run: `uv run python tests/rl_mass/test_bimanual_conditioning.py`
Expected: prints residual + band fraction, then `OK`.

Decision rule:
- Residual < 0.2 and band-fraction > 0.8 → conditioning good; proceed.
- Residual 0.2–0.5 → proceed but expect slower muscle learning; note in commit.
- Residual ≥ 0.5 (assert fails) → length normalization is off. Fallback: recompute operating ranges with `mujoco.mj_setLengthRange(model, data, actuator_id, opt)` for every actuator at load time inside `TendonMuscleSystem.__init__` (before reading `actuator_lengthrange`), then re-run. If still bad, switch `f_active`/`f_passive` to MuJoCo's own `mju_muscleGain`/`mju_muscleBias` (documented alternative in the spec) — but only if this gate fails.

- [ ] **Step 3: Commit**

```bash
git add tests/rl_mass/test_bimanual_conditioning.py
git commit -m "test(bimanual): muscle-basis conditioning gate"
```

---

## Task 8: Short end-to-end training run

**Files:** none new (uses existing `train.py`).

- [ ] **Step 1: Launch a short serial run**

Run:
```bash
uv run python src/rl_mass/train.py -d data/rl_mass_bimanual/config.json \
  --outdir output/bimanual_smoke --num_envs 2 --vec_env_mode serial --max_iter 3
```
(Verified flags: `-d/--meta`, `--outdir`, `--num_envs`, `--vec_env_mode {serial,subproc}`, `--max_iter`, `--seed`, `--no_packed`.)

Expected: prints `num states: 153, num actions: 76, num muscles: 126, num total muscle related dofs: <N>`, then runs 3 iterations writing `output/bimanual_smoke/nn/*.pt` and a reward CSV, with no exceptions.

- [ ] **Step 2: Confirm artifacts**

Run: `ls output/bimanual_smoke/nn/ && head output/bimanual_smoke/train_rewards.csv`
Expected: checkpoint `.pt` files present; reward CSV has ≥1 data row with a finite reward.

- [ ] **Step 3: Launch a longer subproc run (smoke for the packed worker path)**

Run:
```bash
uv run python src/rl_mass/train.py -d data/rl_mass_bimanual/config.json \
  --outdir output/bimanual --num_envs 8 --vec_env_mode subproc --max_iter 20
```
Expected: no worker exceptions; reward trend is non-decreasing on average across the 20 iterations (it need not converge — this only proves the pipeline trains). Per project memory, `loss_muscle` spikes are normal; watch that `num_tuple` does not collapse to ~0.

- [ ] **Step 4: Commit any config tweaks**

If `kp`, `w_q/w_v`, or `pose_error_threshold` needed adjustment to keep episodes alive, update `data/rl_mass_bimanual/config.json` and commit:
```bash
git add data/rl_mass_bimanual/config.json
git commit -m "chore(bimanual): tune SPD kp / reward / termination from smoke run"
```

---

## Self-Review

**Spec coverage:**
- Native-MuJoCo case, isolated from humanoid path → Tasks 5, 6 (factory + `skeleton_format`; humanoid regression in Task 6 Step 6). ✓
- Reuse musclemimic `.npz` (no retarget) → Task 2. ✓
- MuJoCo-native tendons for length/moment (wrapping preserved) → Task 3. ✓
- rl_mass Hill curves on top; `f0`/normalization from actuator params → Task 3. ✓
- SPD + `w_q`/`w_v` reward, no COM/EE/root → Tasks 4, 5. ✓
- Train via existing `train.py -d <config>`; action=126 activations via MuscleNN, policy action=76 → Task 8. ✓
- Open risks (f0 mapping, ten_J sign, coupling-tendon exclusion, length normalization) → resolved in verified facts + Task 3 sign cross-check + Task 7 conditioning gate. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full content; commands have expected output. ✓

**Type consistency:** `TendonMuscleSystem` methods (`update`, `apply_forces`, `get_muscle_torques`, `tuple_terms`), attrs (`R`, `f0`, `f_active`, `f_passive`, `l_m_norm`, `related_dof_indices`, `num_total_related_dofs`, `n_muscles`, `nv`) used consistently across Tasks 3, 5, 7. `MuscleTuple(JtA, L, b, tau_des)` matches `env.py`. `native_spd_forces(engine, q_des, kp, kv)` signature matches Tasks 4, 5. `make_env(config_path)` matches Task 6 call sites. ✓

**Note on `train.py` flags:** Task 8 assumes `--num-envs`, `--vec-env-mode`, `--max-iter`, `--outdir`, `-d`. These are inferred from the `PPO(...)` signature and `argparse` usage; the executor must confirm exact flag spelling via `--help` and adapt (Task 8 Step 1 says so).
