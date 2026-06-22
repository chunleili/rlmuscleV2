"""Coupled muscle-bone solver: DGF activation dynamics, controllability shaping, and the Newton-MuscleSim bridge."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo

from muscle.musclesim import MuscleSim

log = logging.getLogger("couple")


# ===========================================================================
# DGF activation dynamics (moved from activation.py)
# ===========================================================================

@wp.func
def dgf_activation_dynamics(
    excitation: float,
    activation: float,
    dt: float,
    tau_a: float,
    tau_d: float,
    b: float,
    min_activation: float,
) -> float:
    """One step of DGF activation dynamics (implicit Euler).

    da/dt = f(e, a) * (e - a)
    where f = (0.5+0.5*tanh(b*(e-a)))/(tau_a*(0.5+1.5*a))
            + (0.5-0.5*tanh(b*(e-a)))*(0.5+1.5*a)/tau_d
    """
    t = wp.tanh(b * (excitation - activation))
    f_act = (0.5 + 0.5 * t) / (tau_a * (0.5 + 1.5 * activation))
    f_deact = (0.5 - 0.5 * t) * (0.5 + 1.5 * activation) / tau_d
    da_dt = (f_act + f_deact) * (excitation - activation)

    a_new = activation + dt * da_dt
    return wp.clamp(a_new, min_activation, 1.0)



def activation_dynamics_step_scalar(
    excitation: float,
    activation: float,
    dt: float,
    tau_act: float = 0.015,
    tau_deact: float = 0.060,
    b: float = 10.0,
    min_activation: float = 0.0,
) -> float:
    """Scalar version of first-order activation dynamics (no numpy overhead)."""
    t = math.tanh(b * (excitation - activation))
    f_act = (0.5 + 0.5 * t) / (tau_act * (0.5 + 1.5 * activation))
    f_deact = (0.5 - 0.5 * t) * (0.5 + 1.5 * activation) / tau_deact
    da_dt = (f_act + f_deact) * (excitation - activation)
    a_new = activation + dt * da_dt
    return max(min_activation, min(1.0, a_new))


def activation_dynamics_step_np(
    excitation: np.ndarray,
    activation: np.ndarray,
    dt: float,
    tau_act: float = 0.015,
    tau_deact: float = 0.060,
    b: float = 10.0,
    min_activation: float = 0.0,
) -> np.ndarray:
    """NumPy version of first-order activation dynamics."""
    t = np.tanh(b * (excitation - activation))
    f_act = (0.5 + 0.5 * t) / (tau_act * (0.5 + 1.5 * activation))
    f_deact = (0.5 - 0.5 * t) * (0.5 + 1.5 * activation) / tau_deact
    da_dt = (f_act + f_deact) * (excitation - activation)
    a_new = activation + dt * da_dt
    return np.clip(a_new, min_activation, 1.0)


# ===========================================================================
# Controllability shaping (moved from controllability.py)
# ===========================================================================

DEFAULT_SWEEP_LEVELS = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)


@dataclass(slots=True)
class CouplingControlConfig:
    """Configuration for activation-to-torque controllability shaping."""

    preset: str = "smooth_nonlinear"
    mode: str = "smooth_nonlinear"
    k_coupling: float = 30000.0
    max_torque: float = 8.0
    passive_scale: float = 0.0
    torque_ema: float = 0.35
    torque_slew_rate: float = 45.0
    nonlinear_gamma: float = 0.6
    axis_project_torque: bool = True
    use_activation_dynamics: bool = True
    activation_tau_act: float = 0.02
    activation_tau_deact: float = 0.08
    activation_floor: float = 0.0
    activation_release_threshold: float = 1e-4


PRESET_CONFIGS: dict[str, CouplingControlConfig] = {
    "legacy": CouplingControlConfig(
        preset="legacy",
        mode="linear",
        k_coupling=150000.0,
        max_torque=25.0,
        passive_scale=0.05,
        torque_ema=1.0,
        torque_slew_rate=0.0,
        nonlinear_gamma=1.0,
        axis_project_torque=False,
        use_activation_dynamics=False,
        activation_tau_act=0.015,
        activation_tau_deact=0.06,
    ),
    "linear_tuned": CouplingControlConfig(
        preset="linear_tuned",
        mode="linear",
        k_coupling=22000.0,
        max_torque=6.0,
        passive_scale=0.0,
        torque_ema=0.25,
        torque_slew_rate=28.0,
        nonlinear_gamma=1.0,
        axis_project_torque=True,
        use_activation_dynamics=True,
        activation_tau_act=0.025,
        activation_tau_deact=0.10,
    ),
    "smooth_nonlinear": CouplingControlConfig(
        preset="smooth_nonlinear",
        mode="smooth_nonlinear",
        k_coupling=24000.0,
        max_torque=6.5,
        passive_scale=0.0,
        torque_ema=0.30,
        torque_slew_rate=24.0,
        nonlinear_gamma=0.6,
        axis_project_torque=True,
        use_activation_dynamics=True,
        activation_tau_act=0.03,
        activation_tau_deact=0.12,
    ),
    "rate_limited": CouplingControlConfig(
        preset="rate_limited",
        mode="rate_limited",
        k_coupling=20000.0,
        max_torque=5.5,
        passive_scale=0.0,
        torque_ema=0.45,
        torque_slew_rate=16.0,
        nonlinear_gamma=1.0,
        axis_project_torque=True,
        use_activation_dynamics=True,
        activation_tau_act=0.03,
        activation_tau_deact=0.14,
    ),
}


def list_presets() -> tuple[str, ...]:
    """Return available coupling presets."""
    return tuple(PRESET_CONFIGS.keys())


def build_coupling_config(
    preset: str = "smooth_nonlinear",
    **overrides,
) -> CouplingControlConfig:
    """Create a coupling config from a named preset plus optional overrides."""
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown controllability preset '{preset}'. Choose from {sorted(PRESET_CONFIGS)}")
    cfg = replace(PRESET_CONFIGS[preset])
    for key, value in overrides.items():
        if value is None or not hasattr(cfg, key):
            continue
        setattr(cfg, key, value)
    cfg.preset = preset
    return cfg


class ActivationController:
    """Stateful activation filter that converts excitation to effective activation."""

    def __init__(self, cfg: CouplingControlConfig, initial_activation: float = 0.0):
        self.cfg = cfg
        self.activation = float(np.clip(initial_activation, 0.0, 1.0))

    def reset(self, activation: float = 0.0):
        self.activation = float(np.clip(activation, 0.0, 1.0))

    def step(self, excitation: float, dt: float) -> float:
        excitation = float(np.clip(excitation, 0.0, 1.0))
        if not self.cfg.use_activation_dynamics:
            self.activation = excitation
            return self.activation

        updated = activation_dynamics_step_np(
            np.array([excitation], dtype=np.float32),
            np.array([self.activation], dtype=np.float32),
            float(dt),
            tau_act=self.cfg.activation_tau_act,
            tau_deact=self.cfg.activation_tau_deact,
            min_activation=self.cfg.activation_floor,
        )
        self.activation = float(updated[0])
        if (
            excitation <= self.cfg.activation_release_threshold
            and self.activation <= max(self.cfg.activation_floor, 5.0 * self.cfg.activation_release_threshold)
        ):
            self.activation = 0.0
        return self.activation


def activation_to_gain(activation: float, cfg: CouplingControlConfig) -> float:
    """Map normalized activation to normalized torque gain."""
    activation = float(np.clip(activation, 0.0, 1.0))
    if cfg.mode == "linear" or cfg.mode == "rate_limited":
        shaped = activation
    elif cfg.mode == "smooth_nonlinear":
        shaped = 1.0 - math.pow(1.0 - activation, cfg.nonlinear_gamma)
    else:
        raise ValueError(f"Unknown coupling mode '{cfg.mode}'")
    gain = cfg.passive_scale + (1.0 - cfg.passive_scale) * shaped
    return float(np.clip(gain, 0.0, 1.0))


def shape_torque_target(
    raw_torque: np.ndarray,
    effective_activation: float,
    previous_torque: np.ndarray,
    dt: float,
    cfg: CouplingControlConfig,
) -> np.ndarray:
    """Apply activation mapping, rate limiting, smoothing, and clipping."""
    raw_torque = np.asarray(raw_torque, dtype=np.float32)
    previous_torque = np.asarray(previous_torque, dtype=np.float32)
    target = raw_torque * activation_to_gain(effective_activation, cfg)

    if cfg.torque_slew_rate > 0.0:
        delta = target - previous_torque
        delta_mag = float(np.linalg.norm(delta))
        max_delta = cfg.torque_slew_rate * float(dt)
        if delta_mag > max_delta > 0.0:
            target = previous_torque + delta * (max_delta / delta_mag)

    alpha = float(np.clip(cfg.torque_ema, 0.0, 1.0))
    if alpha < 1.0:
        target = previous_torque + alpha * (target - previous_torque)

    mag = float(np.linalg.norm(target))
    if cfg.max_torque > 0.0 and mag > cfg.max_torque:
        target = target * (cfg.max_torque / mag)
    return target.astype(np.float32)


def solver_sample(solver, state) -> dict[str, float]:
    """Extract scalar observables needed for controllability evaluation."""
    joint_q = state.joint_q.numpy()
    joint_qd = state.joint_qd.numpy()
    dof = getattr(solver, "_joint_dof_index", 0)
    axis_torque = float(np.dot(solver._muscle_torque, solver._joint_axis)) if solver._joint_axis is not None else 0.0
    return {
        "effective_activation": float(getattr(solver, "_effective_activation", 0.0)),
        "axis_torque": axis_torque,
        "torque_norm": float(np.linalg.norm(solver._muscle_torque)),
        "joint_angle": float(joint_q[dof]) if len(joint_q) > dof else 0.0,
        "joint_velocity": float(joint_qd[dof]) if len(joint_qd) > dof else 0.0,
    }


def _steady_value(values: np.ndarray, window: int) -> float:
    if values.size == 0:
        return 0.0
    window = max(1, min(window, values.size))
    return float(np.mean(values[-window:]))


def _settle_step(
    torque_values: np.ndarray,
    velocity_values: np.ndarray,
    torque_eps: float,
    velocity_eps: float,
    window: int,
) -> int | None:
    if torque_values.size == 0:
        return 0
    for idx in range(0, max(1, torque_values.size - window + 1)):
        torque_ok = np.all(np.abs(torque_values[idx:idx + window]) <= torque_eps)
        vel_ok = np.all(np.abs(velocity_values[idx:idx + window]) <= velocity_eps)
        if torque_ok and vel_ok:
            return idx + 1
    return None


def summarize_episode(
    level: float,
    hold_samples: list[dict[str, float]],
    release_samples: list[dict[str, float]],
    dt: float,
    steady_window: int = 20,
    settle_window: int = 10,
    settle_torque_eps: float = 0.05,
    settle_velocity_eps: float = 0.02,
) -> dict[str, float | int | None]:
    """Summarize a single hold-release activation episode."""
    hold_axis = np.array([sample["axis_torque"] for sample in hold_samples], dtype=np.float32)
    hold_angle = np.array([sample["joint_angle"] for sample in hold_samples], dtype=np.float32)
    release_axis = np.array([sample["axis_torque"] for sample in release_samples], dtype=np.float32)
    release_vel = np.array([sample["joint_velocity"] for sample in release_samples], dtype=np.float32)

    steady_axis = _steady_value(hold_axis, steady_window)
    steady_angle = _steady_value(hold_angle, steady_window)
    peak_axis = float(np.max(np.abs(hold_axis))) if hold_axis.size else 0.0
    peak_angle = float(np.max(np.abs(hold_angle))) if hold_angle.size else 0.0
    settle_step = _settle_step(release_axis, release_vel, settle_torque_eps, settle_velocity_eps, settle_window)

    return {
        "activation": float(level),
        "steady_axis_torque": steady_axis,
        "steady_axis_torque_abs": float(abs(steady_axis)),
        "peak_axis_torque": peak_axis,
        "steady_joint_angle": steady_angle,
        "steady_joint_angle_abs": float(abs(steady_angle)),
        "peak_joint_angle": peak_angle,
        "overshoot_axis_torque": float(max(0.0, peak_axis - abs(steady_axis))),
        "overshoot_joint_angle": float(max(0.0, peak_angle - abs(steady_angle))),
        "settle_steps_after_release": settle_step,
        "settle_time_after_release": None if settle_step is None else float(settle_step * dt),
    }


def run_activation_sweep(
    label: str,
    dt: float,
    step_fn: Callable[[], None],
    reset_fn: Callable[[], None],
    set_excitation_fn: Callable[[float], None],
    sample_fn: Callable[[], dict[str, float]],
    levels: tuple[float, ...] = DEFAULT_SWEEP_LEVELS,
    hold_steps: int = 90,
    release_steps: int = 90,
    warmup_steps: int = 20,
) -> dict[str, object]:
    """Run repeated hold-release episodes over a fixed activation sweep."""
    episodes: list[dict[str, float | int | None]] = []

    for level in levels:
        reset_fn()
        set_excitation_fn(0.0)
        for _ in range(warmup_steps):
            step_fn()

        hold_samples: list[dict[str, float]] = []
        set_excitation_fn(level)
        for _ in range(hold_steps):
            step_fn()
            hold_samples.append(sample_fn())

        release_samples: list[dict[str, float]] = []
        set_excitation_fn(0.0)
        for _ in range(release_steps):
            step_fn()
            release_samples.append(sample_fn())

        episodes.append(summarize_episode(level, hold_samples, release_samples, dt=dt))

    steady_torques = [float(item["steady_axis_torque_abs"]) for item in episodes]
    steady_angles = [float(item["steady_joint_angle_abs"]) for item in episodes]
    monotonic_torque = all(b >= a - 1e-4 for a, b in zip(steady_torques, steady_torques[1:]))
    monotonic_angle = all(b >= a - 1e-4 for a, b in zip(steady_angles, steady_angles[1:]))

    return {
        "label": label,
        "levels": list(levels),
        "dt": float(dt),
        "hold_steps": int(hold_steps),
        "release_steps": int(release_steps),
        "episodes": episodes,
        "monotonic_steady_torque": monotonic_torque,
        "monotonic_steady_angle": monotonic_angle,
    }


def write_sweep_report(output_path: str | Path, payload: dict[str, object]):
    """Write a JSON sweep report to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def config_to_dict(cfg: CouplingControlConfig) -> dict[str, float | str | bool]:
    """Serialize a coupling config for logging."""
    return asdict(cfg)


def parse_levels(text: str) -> tuple[float, ...]:
    """Parse comma-separated activation levels, clipping to [0, 1]."""
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    return tuple(float(np.clip(v, 0.0, 1.0)) for v in values) if values else DEFAULT_SWEEP_LEVELS


def run_eval_sweep(
    solver,
    sim,
    state,
    cfg,
    dt: float,
    label: str,
    levels: str = "0,0.1,0.3,0.5,0.7,1.0",
    hold_steps: int = 90,
    release_steps: int = 90,
    warmup_steps: int = 20,
    output_dir: str | Path = "output",
) -> dict:
    """Run activation sweep evaluation and write JSON report.

    High-level wrapper around ``run_activation_sweep`` used by couple examples.
    """
    import logging as _logging

    _log = _logging.getLogger("couple")

    def reset_state():
        sim.reset()
        solver.reset_bone(state)

    def step_once():
        solver.step(state, state, dt=dt)

    def set_excitation(value: float):
        cfg.activation = float(value)

    preset = solver.control_config.preset

    report = run_activation_sweep(
        label=label,
        dt=dt,
        step_fn=step_once,
        reset_fn=reset_state,
        set_excitation_fn=set_excitation,
        sample_fn=lambda: solver_sample(solver, state),
        levels=parse_levels(levels),
        hold_steps=hold_steps,
        release_steps=release_steps,
        warmup_steps=warmup_steps,
    )
    report["control_config"] = config_to_dict(solver.control_config)

    report_path = Path(output_dir) / f"{label.replace(':', '_')}_eval_{preset}.json"
    write_sweep_report(report_path, report)
    _log.info("Evaluation sweep saved: %s", report_path)
    for episode in report["episodes"]:
        _log.info(
            "eval act=%.2f steady_tau=%.4f steady_q=%.4f overshoot_tau=%.4f settle=%s",
            episode["activation"],
            episode["steady_axis_torque"],
            episode["steady_joint_angle"],
            episode["overshoot_axis_torque"],
            episode["settle_steps_after_release"],
        )
    _log.info(
        "monotonic torque=%s angle=%s",
        report["monotonic_steady_torque"],
        report["monotonic_steady_angle"],
    )
    return report


# ===========================================================================
# SolverMuscleBoneCoupled (moved from solver_muscle_bone_coupled_warp.py)
# ===========================================================================

log = logging.getLogger("couple")


class SolverMuscleBoneCoupled:
    """Couples Newton rigid-body bone dynamics with Warp MuscleSim PBD.

    Both systems operate in Y-up coordinate space — no conversion needed.

    Per-step coupling flow:
      1. Apply previous frame's muscle torque to Newton joint_f
      2. Newton bone substeps (rigid-body dynamics)
      3. Sync bone positions to MuscleSim bone_pos_field
      4. Muscle PBD substeps (bilateral attach accumulates C*n)
      5. Extract muscle torque for next frame (EMA smoothed)
    """

    def __init__(self, model, core: MuscleSim,
                 bone_substeps: int = 5,
                 k_coupling: float = 5000.0,
                 max_torque: float = 50.0,
                 control_config: CouplingControlConfig | None = None):
        self.model = model
        self.core = core
        self.bone_substeps = bone_substeps
        self.control_config = control_config or build_coupling_config(
            "legacy",
            k_coupling=k_coupling,
            max_torque=max_torque,
        )
        self.k_coupling = self.control_config.k_coupling
        self.max_torque = self.control_config.max_torque
        self.bone_solver = SolverMuJoCo(model, solver="cg", use_mujoco_cpu=True)
        self._coupling_configured = False
        self._muscle_torque = np.zeros(3, dtype=np.float32)  # last-substep torque, for external inspection
        self._raw_muscle_torque = np.zeros(3, dtype=np.float32)
        self._axis_torque = 0.0
        self._step_count = 0
        self._effective_activation = float(max(getattr(self.core.cfg, "activation", 0.0), 0.0))
        self._activation_controller = ActivationController(self.control_config, self._effective_activation)
        self._muscle_graph = None  # CUDA graph for muscle substep

    # -- CUDA Graph Capture -------------------------------------------------

    def capture_muscle_graph(self):
        """Capture the muscle PBD substep as a CUDA graph for replay.

        Must be called AFTER at least one normal step (to JIT-compile all
        kernels and populate arrays). Only works on CUDA with mempool enabled.
        """
        device = self.core.pos.device
        if not device.is_cuda or not wp.is_mempool_enabled(device):
            log.warning("CUDA graph capture unavailable (device=%s, mempool=%s)",
                        device, wp.is_mempool_enabled(device) if device.is_cuda else "N/A")
            self._muscle_graph = None
            return
        log.info("Capturing CUDA graph for muscle substep on %s ...", device)
        with wp.ScopedCapture(device) as capture:
            self.core._capturable_substep()
        self._muscle_graph = capture.graph
        log.info("CUDA graph captured successfully")

    # -- Configuration ------------------------------------------------------

    def configure_coupling(
        self,
        bone_body_id: int,
        bone_rest_verts: np.ndarray,
        bone_vertex_indices: np.ndarray,
        joint_index: int,
        joint_pivot: np.ndarray,
        joint_axis: np.ndarray = None,
    ):
        """Set up bidirectional bone-muscle coupling.

        Args:
            bone_body_id: Newton body index for the dynamic bone.
            bone_rest_verts: Rest-pose vertices, shape (N, 3).
            bone_vertex_indices: Indices into MuscleSim bone_pos_field.
            joint_index: Newton joint index for the driven joint.
            joint_pivot: Joint pivot position in world space, shape (3,).
            joint_axis: Joint rotation axis (for revolute projection).
        """
        self._bone_body_id = bone_body_id

        # Numpy arrays for bone sync
        n_bone = len(bone_vertex_indices)
        self._bone_rest_verts = bone_rest_verts.astype(np.float32)  # (N, 3)
        self._bone_idx = bone_vertex_indices.astype(np.int32)       # (N,)
        self._n_bone_verts = n_bone

        # Joint DOF info
        qd_starts = self.model.joint_qd_start.numpy()
        self._joint_dof_index = int(qd_starts[joint_index])
        if joint_index + 1 < len(qd_starts):
            self._joint_n_dofs = int(qd_starts[joint_index + 1]) - self._joint_dof_index
        else:
            self._joint_n_dofs = int(self.model.joint_dof_count) - self._joint_dof_index
        self._joint_pivot = np.array(joint_pivot, dtype=np.float32)
        self._joint_axis = None
        if joint_axis is not None:
            self._joint_axis = np.array(joint_axis, dtype=np.float32)
            self._joint_axis /= np.linalg.norm(self._joint_axis)

        # Collect attach constraints targeting this bone
        radius_set = set(bone_vertex_indices.tolist())
        attach_cidx = []
        for c in self.core.attach_constraints:
            if int(c['pts'][2]) in radius_set:
                attach_cidx.append(int(c['cidx']))

        self._n_attach = len(attach_cidx)
        self._attach_cidx = np.array(attach_cidx, dtype=np.int32) if attach_cidx else np.empty(0, dtype=np.int32)
        self._coupling_configured = True
        log.info(f"Coupling: {self._n_attach} constraints, "
                 f"DOF={self._joint_dof_index}, n_dofs={self._joint_n_dofs}, "
                 f"k={self.k_coupling}, control={config_to_dict(self.control_config)}")

    # -- Bone sync ----------------------------------------------------------

    def _sync_bone_positions(self, state):
        """Read body_q from Newton, transform rest verts, write to bone_pos_field via numpy."""
        body_q = state.body_q.numpy()
        xf = body_q[self._bone_body_id].astype(np.float32)
        pos = xf[:3]                       # translation
        q_xyz, qw = xf[3:6], xf[6]        # quaternion (x,y,z,w)

        # Vectorised quaternion rotation: v' = v + 2w(q x v) + 2(q x (q x v))
        v = self._bone_rest_verts           # (N, 3)
        t = 2.0 * np.cross(q_xyz, v)       # (N, 3)
        rotated = v + qw * t + np.cross(q_xyz, t) + pos  # (N, 3)

        # Write back to warp bone_pos_field in-place (preserves array pointer
        # for CUDA graph capture compatibility)
        bone_np = self.core.bone_pos_field.numpy()
        bone_np[self._bone_idx] = rotated
        wp.copy(self.core.bone_pos_field,
                wp.from_numpy(bone_np, dtype=wp.vec3, device=self.core.bone_pos_field.device))

    # -- Torque extraction --------------------------------------------------

    def _compute_raw_muscle_torque(self, inv_N) -> np.ndarray:
        """Extract raw 3D joint torque from bilateral attach reactions via numpy."""
        if self._n_attach == 0:
            return np.zeros(3, dtype=np.float32)

        cons_np = self.core.cons.numpy()
        restvec_np = cons_np['restvector']                   # (n_cons, 4)
        reaction_np = self.core.reaction_accum.numpy()       # (n_cons, 3)

        cidxs = self._attach_cidx
        targets = restvec_np[cidxs, :3]                     # (n_attach, 3)
        reactions = reaction_np[cidxs]                       # (n_attach, 3)

        forces = -self.k_coupling * reactions * inv_N        # (n_attach, 3)
        arms = targets - self._joint_pivot                   # (n_attach, 3)
        torque = np.cross(arms, forces).sum(axis=0).astype(np.float32)
        return torque

    # -- Main step ----------------------------------------------------------

    def step(self, state_in, state_out, control=None, contacts=None, dt=None):
        """One coupled simulation step — interleaved substeps (no one-frame delay).

        Each substep:
          1. muscle.integrate + solve_constraints  -> reaction_accum (this substep)
          2. reaction -> torque -> control.joint_f (applied immediately)
          3. bone_solver.step                      (uses current muscle torque)
          4. sync new bone position -> muscle       (for next substep's attach targets)
        """
        if dt is None:
            dt = 1.0 / 60.0
        if control is None:
            control = self.model.control(clone_variables=False)

        N = self.core.cfg.num_substeps
        dt_sub = dt / N
        self.core.dt = dt_sub

        # Sync initial bone position to muscle before first substep
        if self._coupling_configured:
            self._sync_bone_positions(state_in)

        joint_f_device = control.joint_f.device
        joint_f_np = np.zeros(control.joint_f.shape[0], dtype=np.float32)
        excitation = float(np.clip(self.core.cfg.activation, 0.0, 1.0))

        for _ in range(N):
            self._effective_activation = self._activation_controller.step(excitation, dt_sub)
            self.core.activation.fill_(self._effective_activation)

            # Muscle PBD substep: graph replay or full dispatch
            if self._muscle_graph is not None:
                wp.capture_launch(self._muscle_graph)
            else:
                self.core._capturable_substep()

            # 3. Apply torque to bone
            if self._coupling_configured:
                self._raw_muscle_torque = self._compute_raw_muscle_torque(inv_N=1.0)
                if (
                    self.control_config.axis_project_torque
                    and self._joint_n_dofs == 1
                    and self._joint_axis is not None
                ):
                    raw_axis = float(np.dot(self._raw_muscle_torque, self._joint_axis))
                    self._raw_muscle_torque = (self._joint_axis * raw_axis).astype(np.float32)
                torque = shape_torque_target(
                    self._raw_muscle_torque,
                    self._effective_activation,
                    self._muscle_torque,
                    dt_sub,
                    self.control_config,
                )
                self._muscle_torque = torque  # expose for external inspection
                dof = self._joint_dof_index
                n_dofs = self._joint_n_dofs
                joint_f_np[:] = 0.0
                if n_dofs == 1 and self._joint_axis is not None:
                    self._axis_torque = float(np.dot(torque, self._joint_axis))
                    joint_f_np[dof] = self._axis_torque
                else:
                    m = min(n_dofs, 3)
                    joint_f_np[dof:dof + m] = torque[:m]
                    self._axis_torque = float(joint_f_np[dof]) if n_dofs > 0 else 0.0
                control.joint_f.assign(
                    wp.from_numpy(joint_f_np, dtype=wp.float32, device=joint_f_device))

            # 4. Bone substep
            self.bone_solver.step(state_in, state_out, control, None, dt_sub)

            # 5. Sync new bone position to muscle for next substep
            if self._coupling_configured:
                self._sync_bone_positions(state_out)

        self._step_count += 1
        if self._coupling_configured and (self._step_count % 5 == 1):
            mag = float(np.linalg.norm(self._muscle_torque))
            axis_info = ""
            if self._joint_axis is not None:
                axis_info = f" axis_tau={self._axis_torque:.4f}"
            
            # Joint angle and bone orientation
            joint_q = state_out.joint_q.numpy()
            joint_angle = float(joint_q[self._joint_dof_index]) if len(joint_q) > self._joint_dof_index else 0.0
            body_q = state_out.body_q.numpy()[self._bone_body_id]
            qx, qy, qz, qw = float(body_q[3]), float(body_q[4]), float(body_q[5]), float(body_q[6])
            
            log.info(
                f"step={self._step_count} exc={self.core.cfg.activation:.2f} "
                f"act={self._effective_activation:.2f} |tau|={mag:.4f}{axis_info}"
            )
            log.info(f"  q_joint={joint_angle:.4f} q_bone=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f})")

    # -- Reset ---------------------------------------------------------------

    def reset_bone(self, state):
        """Reset bone rigid-body state to initial pose and sync to muscle."""
        # Restore initial joint angles and zero velocities
        state.joint_q = wp.clone(self.model.joint_q)
        state.joint_qd.zero_()
        # Recompute body transforms from joint state
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        # Sync bone vertex positions to muscle PBD
        if self._coupling_configured:
            self._sync_bone_positions(state)
        self._muscle_torque = np.zeros(3, dtype=np.float32)
        self._raw_muscle_torque = np.zeros(3, dtype=np.float32)
        self._axis_torque = 0.0
        self._step_count = 0
        self._effective_activation = 0.0
        self._activation_controller.reset()
        log.info("Bone reset to initial pose")

