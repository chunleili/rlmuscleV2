"""Millard 2012 muscle curves: CPU Bezier compilation + GPU @wp.func evaluation.

Extracted from engine.py (was muscle_warp.py). Self-contained: CPU side builds
quintic-Bezier active/passive force-length curves and packs them for GPU upload;
the @wp.func side evaluates them inside the fiber-constraint kernels (which stay
in engine.py and import these funcs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import warp as wp


# ===========================================================================
# Millard 2012 curves (moved from millard_curves.py)
# ===========================================================================

def scale_curviness(curviness: float) -> float:
    """Map user curviness [0,1] to internal [0.1,0.9].

    Ports: SmoothSegmentedFunctionFactory::scaleCurviness
    """
    return 0.1 + 0.8 * curviness


def quintic_bezier_corner_control_points(
    x0: float, y0: float, dydx0: float,
    x1: float, y1: float, dydx1: float,
    curviness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 6 quintic Bezier control points for a C-shaped corner.

    Ports: SegmentedQuinticBezierToolkit::calcQuinticBezierCornerControlPoints

    Returns:
        (xPts[6], yPts[6]) control point arrays.
    """
    root_eps = math.sqrt(2.220446049250313e-16)  # sqrt(machine eps)

    # Tangent line intersection
    if abs(dydx0 - dydx1) > root_eps:
        xC = (y1 - y0 - x1 * dydx1 + x0 * dydx0) / (dydx0 - dydx1)
    else:
        xC = (x1 + x0) / 2.0
    yC = (xC - x1) * dydx1 + y1

    # C-shape check (triangle inequality)
    a = (xC - x0) ** 2 + (yC - y0) ** 2
    b = (xC - x1) ** 2 + (yC - y1) ** 2
    c = (x1 - x0) ** 2 + (y1 - y0) ** 2
    assert c > a and c > b, (
        f"Tangent intersection inconsistent with C-shaped corner: "
        f"a={a:.6f}, b={b:.6f}, c={c:.6f}"
    )

    # Intermediate control points (doubled for C1 continuity)
    x0_mid = x0 + curviness * (xC - x0)
    y0_mid = y0 + curviness * (yC - y0)
    x1_mid = x1 + curviness * (xC - x1)
    y1_mid = y1 + curviness * (yC - y1)

    xPts = np.array([x0, x0_mid, x0_mid, x1_mid, x1_mid, x1], dtype=np.float64)
    yPts = np.array([y0, y0_mid, y0_mid, y1_mid, y1_mid, y1], dtype=np.float64)
    return xPts, yPts


# ---------------------------------------------------------------------------
# 2. Millard curve factories (matching OpenSim defaults)
# ---------------------------------------------------------------------------

def build_active_fl_bezier(
    x0: float = 0.4441,
    x1: float = 0.73,
    x2: float = 1.0,
    x3: float = 1.8123,
    ylow: float = 0.1,
    dydx: float = 0.8616,
    curviness: float = 1.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build active force-length curve: 5 quintic Bezier segments.

    Ports: SmoothSegmentedFunctionFactory::createFiberActiveForceLengthCurve

    Returns:
        List of 5 (xPts[6], yPts[6]) control point pairs.
    """
    c = scale_curviness(curviness)
    xDelta = 0.05 * x2
    xs = x2 - xDelta

    # Ascending limb
    y0_val = 0.0
    dydx0_val = 0.0
    y1_val = 1.0 - dydx * (xs - x1)
    dydx01 = 1.25 * (y1_val - y0_val) / (x1 - x0)
    x01 = x0 + 0.5 * (x1 - x0)
    y01 = y0_val + 0.5 * (y1_val - y0_val)

    # Shallow ascending plateau
    x1s = x1 + 0.5 * (xs - x1)
    y1s = y1_val + 0.5 * (1.0 - y1_val)
    dydx1s = dydx

    # Peak
    y2 = 1.0
    dydx2 = 0.0

    # Descending limb
    y3 = 0.0
    dydx3 = 0.0
    x23 = (x2 + xDelta) + 0.5 * (x3 - (x2 + xDelta))
    y23 = y2 + 0.5 * (y3 - y2)
    dydx23 = (y3 - y2) / ((x3 - xDelta) - (x2 + xDelta))

    # 5 segments
    segments = [
        quintic_bezier_corner_control_points(x0, ylow, dydx0_val, x01, y01, dydx01, c),
        quintic_bezier_corner_control_points(x01, y01, dydx01, x1s, y1s, dydx1s, c),
        quintic_bezier_corner_control_points(x1s, y1s, dydx1s, x2, y2, dydx2, c),
        quintic_bezier_corner_control_points(x2, y2, dydx2, x23, y23, dydx23, c),
        quintic_bezier_corner_control_points(x23, y23, dydx23, x3, ylow, dydx3, c),
    ]
    return segments


def build_passive_fpe_bezier(
    e_zero: float = 0.0,
    e_iso: float = 0.7,
    k_low: float = 0.2,
    k_iso: float = 2.0 / 0.7,  # ~2.857
    curviness: float = 0.75,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build passive fiber force-length curve: 2 quintic Bezier segments.

    Ports: SmoothSegmentedFunctionFactory::createFiberForceLengthCurve

    Note: x-axis is normalized fiber length (not strain). xZero = 1+eZero, xIso = 1+eIso.

    Returns:
        List of 2 (xPts[6], yPts[6]) control point pairs.
    """
    c = scale_curviness(curviness)
    xZero = 1.0 + e_zero
    yZero = 0.0
    xIso = 1.0 + e_iso
    yIso = 1.0

    deltaX = min(0.1 / k_iso, 0.1 * (xIso - xZero))
    xLow = xZero + deltaX
    xfoot = xZero + 0.5 * (xLow - xZero)
    yfoot = 0.0
    yLow = yfoot + k_low * (xLow - xfoot)

    segments = [
        quintic_bezier_corner_control_points(xZero, yZero, 0.0, xLow, yLow, k_low, c),
        quintic_bezier_corner_control_points(xLow, yLow, k_low, xIso, yIso, k_iso, c),
    ]
    return segments


# ---------------------------------------------------------------------------
# 3. Symbolic derivation: Bernstein → power basis → integral
# ---------------------------------------------------------------------------

def _bernstein_to_power_basis(ctrl_pts: np.ndarray) -> np.ndarray:
    """Convert 6 quintic Bernstein control points to power-basis coefficients.

    Given control points P[0..5], the quintic Bezier is:
        B(u) = Σ P[j] * C(5,j) * u^j * (1-u)^(5-j)

    Returns coefficients c[0..5] such that B(u) = c[0] + c[1]*u + ... + c[5]*u^5.

    Uses the exact conversion matrix (Bernstein-to-power) for degree 5.
    """
    P = ctrl_pts.astype(np.float64)

    # Bernstein-to-power conversion for degree 5:
    # B_j^5(u) = C(5,j) * u^j * (1-u)^(5-j) = Σ_k M[k,j] * u^k
    # where M[k,j] = C(5,j) * C(5-j, k-j) * (-1)^(k-j)  for k >= j, else 0
    #
    # Power basis: B(u) = Σ_j P[j] * Σ_k M[k,j] * u^k = Σ_k (Σ_j P[j]*M[k,j]) * u^k
    #
    # Precomputed conversion matrix M (row=power degree k, col=control point j):
    M = np.array([
        [  1,   0,   0,   0,   0,  0],  # u^0
        [ -5,   5,   0,   0,   0,  0],  # u^1
        [ 10, -20,  10,   0,   0,  0],  # u^2
        [-10,  30, -30,  10,   0,  0],  # u^3
        [  5, -20,  30, -20,   5,  0],  # u^4
        [ -1,   5, -10,  10,  -5,  1],  # u^5
    ], dtype=np.float64)

    coeffs = M @ P  # c[k] = Σ_j M[k,j] * P[j]
    return coeffs  # [c0, c1, c2, c3, c4, c5] where B(u) = c0 + c1*u + ... + c5*u^5


def _compute_integral_coeffs(x_coeffs: np.ndarray, y_coeffs: np.ndarray) -> np.ndarray:
    """Compute F(u) = ∫₀ᵘ y(t)·x'(t) dt as a degree-10 polynomial.

    Given x(u) = Σ x_c[k]*u^k (degree 5) and y(u) = Σ y_c[k]*u^k (degree 5),
    the integrand y(u)·x'(u) is degree 9. Its antiderivative F(u) is degree 10.

    Returns coefficients F[0..10] such that F(u) = F[1]*u + F[2]*u^2 + ... + F[10]*u^10.
    (F[0] = 0 since F(0) = 0.)
    """
    # x'(u) = Σ_{k=1}^{5} k * x_c[k] * u^{k-1}  (degree 4)
    xp_coeffs = np.array([k * x_coeffs[k] for k in range(1, 6)], dtype=np.float64)
    # xp_coeffs[j] is coefficient of u^j for j=0..4

    # y(u) · x'(u) = polynomial multiplication (degree 5 * degree 4 = degree 9)
    product = np.convolve(y_coeffs, xp_coeffs)  # length = 6+5-1 = 10, indices 0..9

    # F(u) = ∫₀ᵘ product(t) dt: antiderivative with F(0)=0
    # If product = Σ p[k]*u^k, then F(u) = Σ p[k]/(k+1) * u^{k+1}
    F_coeffs = np.zeros(11, dtype=np.float64)  # indices 0..10
    for k in range(len(product)):
        F_coeffs[k + 1] = product[k] / (k + 1)

    return F_coeffs


# ---------------------------------------------------------------------------
# 4. Data classes for compiled curves
# ---------------------------------------------------------------------------

@dataclass
class MillardSegment:
    """One quintic Bezier segment with exact polynomial representation."""
    # Bezier control points (for reference / validation)
    ctrl_x: np.ndarray  # (6,) Bezier control points in x
    ctrl_y: np.ndarray  # (6,) Bezier control points in y

    # Power-basis coefficients: f(u) = c[0] + c[1]*u + ... + c[5]*u^5
    x_coeffs: np.ndarray  # (6,) x(u) polynomial
    y_coeffs: np.ndarray  # (6,) y(u) polynomial

    # Energy integral: F(u) = F[1]*u + F[2]*u^2 + ... + F[10]*u^10
    F_coeffs: np.ndarray  # (11,) ∫₀ᵘ y(t)·x'(t) dt

    # Domain in x-space: [x(0), x(1)]
    x_start: float = 0.0
    x_end: float = 0.0


@dataclass
class MillardCurve:
    """Complete Millard curve = multiple quintic Bezier segments."""
    segments: list[MillardSegment] = field(default_factory=list)

    # Domain boundaries (for extrapolation)
    x_lo: float = 0.0
    x_hi: float = 0.0
    y_lo: float = 0.0
    y_hi: float = 0.0
    dydx_lo: float = 0.0
    dydx_hi: float = 0.0

    def eval(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate y(x) on CPU (vectorized)."""
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        y = np.empty_like(x)
        for i, xi in enumerate(x):
            y[i] = self._eval_scalar(xi)
        return float(y[0]) if scalar else y

    def eval_scalar(self, x: float) -> float:
        """Evaluate y(x) for a single scalar (no numpy overhead)."""
        return self._eval_scalar(float(x))

    def eval_integral(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate ∫_{x_lo}^{x} y(s) ds on CPU."""
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        result = np.empty_like(x)
        for i, xi in enumerate(x):
            result[i] = self._eval_integral_scalar(xi)
        return float(result[0]) if scalar else result

    def _eval_scalar(self, x: float) -> float:
        """Evaluate y at a single x value."""
        # Linear extrapolation outside domain
        if x < self.x_lo:
            return self.y_lo + self.dydx_lo * (x - self.x_lo)
        if x > self.x_hi:
            return self.y_hi + self.dydx_hi * (x - self.x_hi)

        # Find segment
        seg = self.segments[0]
        for s in self.segments:
            if x <= s.x_end + 1e-14:
                seg = s
                break
            seg = s  # use last if x is at boundary

        u = _newton_find_u(x, seg.x_coeffs)
        return _horner_eval(u, seg.y_coeffs)

    def _eval_integral_scalar(self, x: float) -> float:
        """Evaluate cumulative integral from x_lo to x."""
        if x <= self.x_lo:
            # Quadratic extrapolation: y_lo*(x-x_lo) + 0.5*dydx_lo*(x-x_lo)^2
            dx = x - self.x_lo
            return self.y_lo * dx + 0.5 * self.dydx_lo * dx * dx

        total = 0.0
        for seg in self.segments:
            if x <= seg.x_start:
                break
            if x >= seg.x_end:
                # Full segment contribution: F(1)
                total += _horner_eval(1.0, seg.F_coeffs)
            else:
                # Partial segment
                u = _newton_find_u(x, seg.x_coeffs)
                total += _horner_eval(u, seg.F_coeffs)
                break

        # Linear extrapolation beyond domain
        if x > self.x_hi:
            dx = x - self.x_hi
            total += self.y_hi * dx + 0.5 * self.dydx_hi * dx * dx

        return total


# ---------------------------------------------------------------------------
# 5. Numerical helpers
# ---------------------------------------------------------------------------

def _horner_eval(u: float, coeffs: np.ndarray) -> float:
    """Evaluate polynomial using Horner's method: c[0] + c[1]*u + ... + c[n]*u^n."""
    result = 0.0
    for k in range(len(coeffs) - 1, -1, -1):
        result = result * u + coeffs[k]
    return result


def _horner_eval_deriv(u: float, coeffs: np.ndarray) -> tuple[float, float]:
    """Evaluate polynomial and its derivative simultaneously."""
    n = len(coeffs) - 1
    val = coeffs[n]
    dval = 0.0
    for k in range(n - 1, -1, -1):
        dval = dval * u + val
        val = val * u + coeffs[k]
    return val, dval


def _newton_find_u(x_target: float, x_coeffs: np.ndarray,
                   tol: float = 1e-12, max_iter: int = 20) -> float:
    """Newton iteration to find u such that x(u) = x_target.

    x(u) is monotonic for well-formed Bezier, so convergence is guaranteed.
    Initial guess: linear interpolation.
    """
    x0 = x_coeffs[0]  # x(0) = c[0]
    x1 = sum(x_coeffs)  # x(1) = c[0]+c[1]+...+c[5]
    if abs(x1 - x0) < 1e-20:
        return 0.5

    # Linear initial guess
    u = (x_target - x0) / (x1 - x0)
    u = max(0.0, min(1.0, u))

    for _ in range(max_iter):
        xu, dxu = _horner_eval_deriv(u, x_coeffs)
        err = xu - x_target
        if abs(err) < tol:
            break
        if abs(dxu) < 1e-20:
            break
        u -= err / dxu
        u = max(-0.05, min(1.05, u))  # allow slight overshoot for robustness

    return u


# ---------------------------------------------------------------------------
# 6. Curve compilation: Bezier control points → MillardCurve
# ---------------------------------------------------------------------------

def _compile_curve(
    bezier_segments: list[tuple[np.ndarray, np.ndarray]],
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    dydx_lo: float, dydx_hi: float,
) -> MillardCurve:
    """Compile Bezier control points into MillardCurve with exact polynomial coefficients."""
    segments = []
    for ctrl_x, ctrl_y in bezier_segments:
        x_coeffs = _bernstein_to_power_basis(ctrl_x)
        y_coeffs = _bernstein_to_power_basis(ctrl_y)
        F_coeffs = _compute_integral_coeffs(x_coeffs, y_coeffs)

        x_start = float(x_coeffs[0])  # x(0)
        x_end = float(sum(x_coeffs))  # x(1)

        segments.append(MillardSegment(
            ctrl_x=ctrl_x,
            ctrl_y=ctrl_y,
            x_coeffs=x_coeffs,
            y_coeffs=y_coeffs,
            F_coeffs=F_coeffs,
            x_start=x_start,
            x_end=x_end,
        ))

    return MillardCurve(
        segments=segments,
        x_lo=x_lo, x_hi=x_hi,
        y_lo=y_lo, y_hi=y_hi,
        dydx_lo=dydx_lo, dydx_hi=dydx_hi,
    )


# ---------------------------------------------------------------------------
# 7. Public API: build default Millard curves
# ---------------------------------------------------------------------------

class MillardCurves:
    """Container for default Millard 2012 muscle curves.

    Usage:
        mc = MillardCurves()
        fl_val = mc.fl.eval(0.8)        # active force-length at lambda=0.8
        fpe_val = mc.fpe.eval(1.3)      # passive force-length at lambda=1.3
        psi_pe = mc.fpe.eval_integral(1.3)  # passive energy integral
    """

    def __init__(self, **kwargs):
        # Active force-length
        fl_bezier = build_active_fl_bezier(**{
            k: v for k, v in kwargs.items()
            if k in ('x0', 'x1', 'x2', 'x3', 'ylow', 'dydx', 'curviness')
        })
        fl_x0 = kwargs.get('x0', 0.4441)
        fl_x3 = kwargs.get('x3', 1.8123)
        fl_ylow = kwargs.get('ylow', 0.1)
        self.fl = _compile_curve(
            fl_bezier,
            x_lo=fl_x0, x_hi=fl_x3,
            y_lo=fl_ylow, y_hi=fl_ylow,
            dydx_lo=0.0, dydx_hi=0.0,
        )

        # Passive fiber force-length
        fpe_bezier = build_passive_fpe_bezier(**{
            k: v for k, v in kwargs.items()
            if k in ('e_zero', 'e_iso', 'k_low', 'k_iso', 'curviness')
        })
        e_zero = kwargs.get('e_zero', 0.0)
        e_iso = kwargs.get('e_iso', 0.7)
        k_iso = kwargs.get('k_iso', 2.0 / 0.7)
        self.fpe = _compile_curve(
            fpe_bezier,
            x_lo=1.0 + e_zero, x_hi=1.0 + e_iso,
            y_lo=0.0, y_hi=1.0,
            dydx_lo=0.0, dydx_hi=k_iso,
        )

    def get_gpu_arrays(self) -> dict:
        """Package curve coefficients for GPU upload.

        Returns dict with flat numpy arrays ready for wp.from_numpy().
        """
        return {
            'fl': _curve_to_gpu_arrays(self.fl),
            'fpe': _curve_to_gpu_arrays(self.fpe),
        }


def _curve_to_gpu_arrays(curve: MillardCurve) -> dict:
    """Flatten curve data for GPU consumption."""
    n_seg = len(curve.segments)
    x_coeffs = np.zeros((n_seg, 6), dtype=np.float32)
    y_coeffs = np.zeros((n_seg, 6), dtype=np.float32)
    F_coeffs = np.zeros((n_seg, 11), dtype=np.float32)

    for i, seg in enumerate(curve.segments):
        x_coeffs[i] = seg.x_coeffs.astype(np.float32)
        y_coeffs[i] = seg.y_coeffs.astype(np.float32)
        F_coeffs[i] = seg.F_coeffs.astype(np.float32)

    return {
        'n_segments': n_seg,
        'x_coeffs': x_coeffs.flatten(),  # (n_seg*6,)
        'y_coeffs': y_coeffs.flatten(),  # (n_seg*6,)
        'F_coeffs': F_coeffs.flatten(),  # (n_seg*11,)
        'x_lo': np.float32(curve.x_lo),
        'x_hi': np.float32(curve.x_hi),
        'y_lo': np.float32(curve.y_lo),
        'y_hi': np.float32(curve.y_hi),
        'dydx_lo': np.float32(curve.dydx_lo),
        'dydx_hi': np.float32(curve.dydx_hi),
    }


# ---------------------------------------------------------------------------
# 8. Equilibrium solver (CPU-side, per-step)
# ---------------------------------------------------------------------------

def millard_equilibrium_fiber_length(
    activation: float,
    normalized_load: float,
    fl_curve: MillardCurve,
    search_range: tuple[float, float] | None = None,
) -> float:
    """Find fiber length where a*f_L(lm) + f_PE(lm) = normalized_load.

    Searches ascending limb for intersection. Uses bisection for robustness.

    Args:
        activation: Muscle activation [0, 1].
        normalized_load: F_external / F_max (normalized).
        fl_curve: Active force-length MillardCurve.
        search_range: Optional (lm_min, lm_max) to search. Default: ascending limb.

    Returns:
        Equilibrium normalized fiber length.
    """
    if search_range is None:
        search_range = (fl_curve.x_lo, 1.0)  # ascending limb

    lo, hi = search_range
    target = normalized_load / max(activation, 1e-8)

    # Bisection on ascending limb: f_L is monotonically increasing from lo to 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = fl_curve.eval(mid)
        if val < target:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < 1e-12:
            break

    lm_eq = 0.5 * (lo + hi)
    return lm_eq


# ---------------------------------------------------------------------------
# 9. Quick validation / demo
# ---------------------------------------------------------------------------

def _quick_validate():
    """Quick sanity check: print curve values at key points."""
    mc = MillardCurves()

    print("=== Active force-length (f_L) ===")
    print(f"  Segments: {len(mc.fl.segments)}")
    for lm in [0.4, 0.4441, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8123, 2.0]:
        print(f"  f_L({lm:.4f}) = {mc.fl.eval(lm):.6f}")

    print("\n=== Passive force-length (f_PE) ===")
    print(f"  Segments: {len(mc.fpe.segments)}")
    for lm in [0.8, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9]:
        print(f"  f_PE({lm:.4f}) = {mc.fpe.eval(lm):.6f}")

    print("\n=== Energy integrals ===")
    for lm in [0.6, 0.8, 1.0, 1.2, 1.5]:
        psi_l = mc.fl.eval_integral(lm)
        print(f"  Psi_L({lm:.1f}) = {psi_l:.6f}")
    for lm in [1.0, 1.2, 1.5, 1.7]:
        psi_pe = mc.fpe.eval_integral(lm)
        print(f"  Psi_PE({lm:.1f}) = {psi_pe:.6f}")

    # Verify integral consistency: F'(x) ≈ f(x)
    print("\n=== Integral derivative check (should match f_L) ===")
    eps = 1e-7
    for lm in [0.6, 0.8, 1.0, 1.2, 1.5]:
        F_plus = mc.fl.eval_integral(lm + eps)
        F_minus = mc.fl.eval_integral(lm - eps)
        numerical_deriv = (F_plus - F_minus) / (2 * eps)
        analytical = mc.fl.eval(lm)
        print(f"  lm={lm:.1f}: F'={numerical_deriv:.6f}, f_L={analytical:.6f}, "
              f"err={abs(numerical_deriv - analytical):.2e}")


if __name__ == "__main__":
    _quick_validate()

# ---------------------------------------------------------------------------
# Millard 2012 quintic Bezier curve evaluation (GPU)
# Coefficients in power basis: f(u) = c0 + c1*u + c2*u^2 + ... + c5*u^5
# Energy integral F(u) = ∫₀ᵘ y(t)·x'(t)dt is degree-10 polynomial.
# ---------------------------------------------------------------------------

@wp.func
def horner10_wp(u: float, c0: float, c1: float, c2: float, c3: float,
                c4: float, c5: float, c6: float, c7: float, c8: float,
                c9: float, c10: float) -> float:
    """Evaluate degree-10 polynomial via Horner's method."""
    return c0 + u * (c1 + u * (c2 + u * (c3 + u * (c4 + u * (
           c5 + u * (c6 + u * (c7 + u * (c8 + u * (c9 + u * c10)))))))))


@wp.func
def horner5_wp(u: float, c0: float, c1: float, c2: float,
               c3: float, c4: float, c5: float) -> float:
    """Evaluate degree-5 polynomial via Horner's method."""
    return c0 + u * (c1 + u * (c2 + u * (c3 + u * (c4 + u * c5))))


@wp.func
def horner5_deriv_wp(u: float, c1: float, c2: float,
                     c3: float, c4: float, c5: float) -> float:
    """Evaluate derivative of degree-5 polynomial: c1 + 2c2*u + ... + 5c5*u^4."""
    return c1 + u * (2.0 * c2 + u * (3.0 * c3 + u * (4.0 * c4 + u * 5.0 * c5)))


@wp.func
def millard_eval_wp(
    x_target: float,
    x_coeffs: wp.array(dtype=wp.float32),
    y_coeffs: wp.array(dtype=wp.float32),
    seg_bounds: wp.array(dtype=wp.float32),
    n_segments: int,
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    dydx_lo: float, dydx_hi: float,
) -> float:
    """Evaluate Millard piecewise quintic Bezier curve y(x) on GPU.

    Args:
        x_target: Input value (normalized fiber length).
        x_coeffs: Flattened power-basis coefficients for x(u), shape (n_seg*6,).
        y_coeffs: Flattened power-basis coefficients for y(u), shape (n_seg*6,).
        seg_bounds: Segment x-boundaries, shape (n_seg+1,): [x_start_0, x_end_0, ...].
        n_segments: Number of Bezier segments.
        x_lo, x_hi: Domain boundaries.
        y_lo, y_hi, dydx_lo, dydx_hi: Boundary values for linear extrapolation.
    """
    # Linear extrapolation outside domain
    if x_target < x_lo:
        return y_lo + dydx_lo * (x_target - x_lo)
    if x_target > x_hi:
        return y_hi + dydx_hi * (x_target - x_hi)

    # Find segment via linear scan (no break — Warp limitation)
    seg = int(n_segments - 1)
    found = int(0)
    for i in range(n_segments):
        if found == 0 and x_target <= seg_bounds[i + 1]:
            seg = i
            found = 1

    # Read x(u) coefficients for this segment
    off = seg * 6
    xc0 = float(x_coeffs[off])
    xc1 = float(x_coeffs[off + 1])
    xc2 = float(x_coeffs[off + 2])
    xc3 = float(x_coeffs[off + 3])
    xc4 = float(x_coeffs[off + 4])
    xc5 = float(x_coeffs[off + 5])

    # Newton iteration: x(u) = x_target → solve for u
    x_at_1 = xc0 + xc1 + xc2 + xc3 + xc4 + xc5
    denom = x_at_1 - xc0
    u = float(0.5)
    if wp.abs(denom) > 1.0e-20:
        u = (x_target - xc0) / denom
    u = wp.clamp(u, 0.0, 1.0)

    for _iter in range(5):
        xu = horner5_wp(u, xc0, xc1, xc2, xc3, xc4, xc5)
        dxu = horner5_deriv_wp(u, xc1, xc2, xc3, xc4, xc5)
        if wp.abs(dxu) > 1.0e-20:
            u = u - (xu - x_target) / dxu
            u = wp.clamp(u, -0.05, 1.05)

    # Read y(u) coefficients and evaluate
    yc0 = float(y_coeffs[off])
    yc1 = float(y_coeffs[off + 1])
    yc2 = float(y_coeffs[off + 2])
    yc3 = float(y_coeffs[off + 3])
    yc4 = float(y_coeffs[off + 4])
    yc5 = float(y_coeffs[off + 5])

    return horner5_wp(u, yc0, yc1, yc2, yc3, yc4, yc5)


@wp.func
def millard_energy_eval_wp(
    x_target: float,
    x_coeffs: wp.array(dtype=wp.float32),
    F_coeffs: wp.array(dtype=wp.float32),
    seg_bounds: wp.array(dtype=wp.float32),
    n_segments: int,
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    dydx_lo: float, dydx_hi: float,
) -> float:
    """Evaluate cumulative energy integral Psi(x) = integral_{x_lo}^{x} y(s) ds.

    Uses precomputed degree-10 polynomial F_coeffs per segment.
    F_coeffs layout: (n_seg*11,) flat, F_coeffs[seg*11 : seg*11+11].
    """
    # Left extrapolation: quadratic (integral of linear extrapolation)
    if x_target <= x_lo:
        dx = x_target - x_lo
        return y_lo * dx + 0.5 * dydx_lo * dx * dx

    # Accumulate complete segments + partial segment
    total = float(0.0)

    # Warp has no break — use done flag
    done = int(0)
    for i in range(n_segments):
        if done == 0:
            if x_target >= seg_bounds[i + 1]:
                # Complete segment: add F(1)
                off = i * 11
                total += horner10_wp(
                    1.0,
                    float(F_coeffs[off]), float(F_coeffs[off + 1]),
                    float(F_coeffs[off + 2]), float(F_coeffs[off + 3]),
                    float(F_coeffs[off + 4]), float(F_coeffs[off + 5]),
                    float(F_coeffs[off + 6]), float(F_coeffs[off + 7]),
                    float(F_coeffs[off + 8]), float(F_coeffs[off + 9]),
                    float(F_coeffs[off + 10]),
                )
            else:
                # Partial segment: Newton x→u, then F(u)
                xoff = i * 6
                xc0 = float(x_coeffs[xoff])
                xc1 = float(x_coeffs[xoff + 1])
                xc2 = float(x_coeffs[xoff + 2])
                xc3 = float(x_coeffs[xoff + 3])
                xc4 = float(x_coeffs[xoff + 4])
                xc5 = float(x_coeffs[xoff + 5])

                x_at_1 = xc0 + xc1 + xc2 + xc3 + xc4 + xc5
                denom = x_at_1 - xc0
                u = float(0.5)
                if wp.abs(denom) > 1.0e-20:
                    u = (x_target - xc0) / denom
                u = wp.clamp(u, 0.0, 1.0)

                for _iter in range(5):
                    xu = horner5_wp(u, xc0, xc1, xc2, xc3, xc4, xc5)
                    dxu = horner5_deriv_wp(u, xc1, xc2, xc3, xc4, xc5)
                    if wp.abs(dxu) > 1.0e-20:
                        u = u - (xu - x_target) / dxu
                        u = wp.clamp(u, -0.05, 1.05)

                off = i * 11
                total += horner10_wp(
                    u,
                    float(F_coeffs[off]), float(F_coeffs[off + 1]),
                    float(F_coeffs[off + 2]), float(F_coeffs[off + 3]),
                    float(F_coeffs[off + 4]), float(F_coeffs[off + 5]),
                    float(F_coeffs[off + 6]), float(F_coeffs[off + 7]),
                    float(F_coeffs[off + 8]), float(F_coeffs[off + 9]),
                    float(F_coeffs[off + 10]),
                )
                done = 1

    # Right extrapolation: quadratic
    if x_target > x_hi:
        dx = x_target - x_hi
        total += y_hi * dx + 0.5 * dydx_hi * dx * dx

    return total
