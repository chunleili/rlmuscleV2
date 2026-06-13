"""DeGroote-Fregly (2016) muscle force-length / force-velocity curves (GPU @wp.func).

Extracted from engine.py. Constants mirror dgf_curves.py. Pure Warp device
functions with no engine dependencies; imported back into engine.py for the
fiber-constraint kernels.
"""
from __future__ import annotations

import warp as wp


@wp.func
def dgf_active_force_length_wp(lm_tilde: float) -> float:
    """Active force-length curve: 3 Gaussian-like terms."""
    b11 = 0.815;  b21 = 1.055;  b31 = 0.162;  b41 = 0.063
    b12 = 0.433;  b22 = 0.717;  b32 = -0.030; b42 = 0.200
    b13 = 0.1;    b23 = 1.0;    b33 = 0.354
    EPS = 1.0e-6

    d1 = wp.max(wp.abs(b31 + b41 * lm_tilde), EPS)
    g1 = b11 * wp.exp(-0.5 * ((lm_tilde - b21) / d1) * ((lm_tilde - b21) / d1))

    d2 = wp.max(wp.abs(b32 + b42 * lm_tilde), EPS)
    g2 = b12 * wp.exp(-0.5 * ((lm_tilde - b22) / d2) * ((lm_tilde - b22) / d2))

    g3 = b13 * wp.exp(-0.5 * ((lm_tilde - b23) / b33) * ((lm_tilde - b23) / b33))
    return g1 + g2 + g3


@wp.func
def dgf_passive_force_length_wp(lm_tilde: float) -> float:
    """Passive force-length curve: exponential."""
    KPE = 4.0
    E0 = 0.6
    LM_MIN = 0.2

    offset = wp.exp(KPE * (LM_MIN - 1.0) / E0)
    denom = wp.exp(KPE) - offset
    arg = wp.clamp(KPE * (lm_tilde - 1.0) / E0, -50.0, 50.0)
    result = (wp.exp(arg) - offset) / denom
    return wp.max(result, 0.0)


@wp.func
def dgf_force_velocity_wp(v_norm: float) -> float:
    """Force-velocity curve: hyperbolic (asinh-based)."""
    D1 = -0.3211346127989808
    D2 = -8.149
    D3 = -0.374
    D4 = 0.8825327733249912

    x = D2 * v_norm + D3
    return D1 * wp.log(x + wp.sqrt(x * x + 1.0)) + D4
