"""DGF activation dynamics for volumetric muscle.

Provides both Warp kernel and NumPy implementations of the first-order
activation dynamics ODE from DeGroote-Fregly 2016.
"""

import math

import numpy as np
import warp as wp


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
