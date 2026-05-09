"""Shared controllability helpers for coupled muscle-bone examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from muscle.couple.activation import activation_dynamics_step_np

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
