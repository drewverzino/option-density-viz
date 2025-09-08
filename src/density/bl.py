"""
BL finite-difference density extraction.

Breeden–Litzenberger (1978): under mild conditions, the risk-neutral
probability density f_Q(K, T) is the second derivative of the call
price with respect to strike, discounted back to today:

    f_Q(K, T) = exp(r T) * d^2 C(K, T) / dK^2

This module provides two practical, numerically robust pathways:

1) bl_pdf_from_calls_nonuniform:
   operate directly on the original, non-uniform strike grid using a
   three-point second-derivative stencil that is valid for uneven
   spacing.

2) bl_pdf_from_calls:
   smooth the call curve C(K) with a PCHIP monotone cubic spline,
   sample it on a dense uniform grid, then apply central (uniform)
   second differences. This reduces noise and grid artifacts while
   preserving shape (PCHIP is shape-preserving and monotone).

We also provide helpers to compute diagnostics and to integrate a pdf
with the trapezoid rule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

try:
    # SciPy PCHIP for shape-preserving interpolation
    from scipy.interpolate import PchipInterpolator  # type: ignore

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    PchipInterpolator = None
    _HAVE_SCIPY = False

# Set up module logger
logger = logging.getLogger("density.bl")


@dataclass
class BLDiagnostics:
    """Summary statistics and sanity checks for a recovered PDF."""

    integral: float  # ∫ pdf dK on the computed grid
    neg_frac: float  # fraction of grid points with pdf < 0 (pre-clip)
    rn_mean: float  # ∫ K * pdf dK
    rn_var: float  # variance around rn_mean
    note: str = ""  # freeform notes

    def __post_init__(self):
        """Log diagnostic creation and validate values."""
        logger.debug(
            f"BL diagnostics: integral={self.integral:.6f}, "
            f"neg_frac={self.neg_frac:.4f}, mean={self.rn_mean:.2f}, "
            f"var={self.rn_var:.6f}"
        )

        # Quality checks
        if abs(self.integral - 1.0) > 0.1:
            logger.warning(f"PDF integral far from 1.0: {self.integral:.6f}")

        if self.neg_frac > 0.05:
            logger.warning(
                f"High negative density fraction: {self.neg_frac:.4f} "
                "(may indicate arbitrage)"
            )
        elif self.neg_frac > 0.01:
            logger.info(f"Moderate negative density: {self.neg_frac:.4f}")

        if not np.isfinite(self.rn_mean):
            logger.warning("Risk-neutral mean is not finite")

        if self.rn_var < 0:
            logger.warning(f"Negative variance computed: {self.rn_var:.6f}")


def _trapz(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoid rule with float return."""
    result = float(np.trapezoid(y, x))
    logger.debug(
        f"Trapezoid integration: ∫f dx = {result:.6f} over [{x.min():.2f}, {x.max():.2f}]"
    )
    return result


def _central_second_uniform(y: np.ndarray, h: float) -> np.ndarray:
    """
    Second derivative for uniform spacing using the classic central
    three-point stencil:

        y''_i ≈ (y_{i+1} - 2 y_i + y_{i-1}) / h^2

    Endpoints are set to 0 (no info there, and avoids NaNs).
    """
    logger.debug(
        f"Computing central second derivative: {len(y)} points, h={h:.6f}"
    )

    out = np.zeros_like(y, dtype=float)
    if y.size >= 3:
        out[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (h * h)

        # Log derivative statistics
        finite_derivs = out[np.isfinite(out)]
        if len(finite_derivs) > 0:
            logger.debug(
                f"Second derivative range: [{finite_derivs.min():.8f}, {finite_derivs.max():.8f}]"
            )
    else:
        logger.warning(
            f"Insufficient points for second derivative: {y.size} < 3"
        )

    return out


def _second_nonuniform(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Three-point second derivative on a non-uniform grid.
    For interior i, let h0 = x_i - x_{i-1}, h1 = x_{i+1} - x_i:

        y''(x_i) ≈ 2 * [ y_{i-1}/(h0*(h0+h1))
                         - y_i/(h0*h1)
                         + y_{i+1}/(h1*(h0+h1)) ]

    Endpoints are set to 0.
    """
    n = x.size
    logger.debug(f"Computing non-uniform second derivative: {n} points")

    if n < 3:
        logger.warning(f"Insufficient points for second derivative: {n} < 3")
        return np.zeros(n, dtype=float)

    # Check grid spacing uniformity
    spacings = np.diff(x)
    spacing_cv = (
        np.std(spacings) / np.mean(spacings) if np.mean(spacings) > 0 else 0
    )
    logger.debug(
        f"Grid spacing: mean={np.mean(spacings):.4f}, CV={spacing_cv:.4f}"
    )

    if spacing_cv > 0.5:
        logger.info(f"Highly non-uniform grid (CV={spacing_cv:.4f})")

    out = np.zeros(n, dtype=float)
    h0 = x[1:-1] - x[:-2]
    h1 = x[2:] - x[1:-1]

    # Check for zero spacings
    zero_h0 = (h0 == 0).sum()
    zero_h1 = (h1 == 0).sum()
    if zero_h0 > 0 or zero_h1 > 0:
        logger.error(f"Zero spacings detected: h0={zero_h0}, h1={zero_h1}")

    a = 2.0 / (h0 * (h0 + h1))
    b = -2.0 / (h0 * h1)
    c = 2.0 / (h1 * (h0 + h1))
    out[1:-1] = a * y[:-2] + b * y[1:-1] + c * y[2:]

    # Log derivative statistics
    finite_derivs = out[np.isfinite(out)]
    if len(finite_derivs) > 0:
        logger.debug(
            f"Second derivative range: [{finite_derivs.min():.8f}, {finite_derivs.max():.8f}]"
        )

    return out


def bl_pdf_from_calls_nonuniform(
    strikes: ArrayLike,
    calls: ArrayLike,
    r: float,
    T: float,
    *,
    clip_negative: bool = True,
    renormalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, BLDiagnostics]:
    """
    Recover pdf directly on the original (possibly uneven) strike grid.

    Parameters
    ----------
    strikes, calls : array-like
        Strike levels and corresponding call prices C(K). Must be the
        same length; unsorted is fine (we sort).
    r : float
        Continuously-compounded risk-free rate.
    T : float
        Year fraction to expiry.
    clip_negative : bool
        If True, replace negative pdf values with 0 before computing
        diagnostics/renormalization.
    renormalize : bool
        If True, scale the pdf so that ∫ pdf dK = 1 (if integral > 0).

    Returns
    -------
    (K, pdf, diag)
    """
    logger.info("Starting BL density extraction on non-uniform grid")
    logger.debug(
        f"Parameters: r={r:.6f}, T={T:.4f}, clip_negative={clip_negative}, "
        f"renormalize={renormalize}"
    )

    K = np.asarray(strikes, dtype=float).copy()
    C = np.asarray(calls, dtype=float).copy()

    initial_count = len(K)
    logger.debug(f"Input: {initial_count} strike-call pairs")
    logger.debug(f"Strike range: [{K.min():.2f}, {K.max():.2f}]")
    logger.debug(f"Call price range: [{C.min():.6f}, {C.max():.6f}]")

    # Filter finite values
    m = np.isfinite(K) & np.isfinite(C)
    K, C = K[m], C[m]
    filtered_count = len(K)

    if filtered_count < initial_count:
        logger.warning(
            f"Filtered out {initial_count - filtered_count} non-finite values"
        )

    if K.size < 3:
        logger.error(f"Insufficient data after filtering: {K.size} < 3")
        raise ValueError("Need at least 3 call points for second derivative.")

    # Sort by strikes
    order = np.argsort(K)
    K, C = K[order], C[order]
    logger.debug("Sorted by strike price")

    # Check for monotonicity violations in call prices
    price_violations = (np.diff(C) > 0).sum()
    total_intervals = len(C) - 1
    if price_violations > total_intervals * 0.1:  # More than 10% violations
        logger.warning(
            f"Call prices not monotonically decreasing: "
            f"{price_violations}/{total_intervals} increases"
        )

    # Compute second derivative
    logger.debug("Computing second derivative of call prices")
    Cpp = _second_nonuniform(K, C)

    # Apply BL formula
    discount_factor = np.exp(r * T)
    pdf = discount_factor * Cpp
    logger.debug(
        f"Applied BL formula with discount factor {discount_factor:.6f}"
    )

    # Analyze negativity before clipping
    neg_mask = pdf < 0.0
    neg_frac = float(np.mean(neg_mask)) if pdf.size else 0.0
    logger.info(f"Negative density fraction (pre-clip): {neg_frac:.4f}")

    if neg_frac > 0:
        neg_values = pdf[neg_mask]
        logger.debug(
            f"Negative values: count={len(neg_values)}, "
            f"range=[{neg_values.min():.8f}, {neg_values.max():.8f}]"
        )

    # Handle negative values
    if clip_negative:
        pdf = np.where(np.isfinite(pdf) & (pdf > 0.0), pdf, 0.0)
        logger.debug("Clipped negative values to zero")
    else:
        pdf = np.where(np.isfinite(pdf), pdf, 0.0)
        logger.debug("Set non-finite values to zero (kept negatives)")

    # Compute integral
    integral = _trapz(K, pdf)
    logger.info(f"PDF integral before normalization: {integral:.6f}")

    # Renormalize if requested
    if renormalize and integral > 0.0:
        pdf = pdf / integral
        logger.debug(f"Renormalized PDF by factor {1/integral:.6f}")
        integral = 1.0

    # Compute moments
    logger.debug("Computing risk-neutral moments")
    mean = _trapz(K, K * pdf)
    var = max(_trapz(K, (K - mean) ** 2 * pdf), 0.0)

    logger.info(f"BL density results: mean={mean:.2f}, var={var:.6f}")

    diag = BLDiagnostics(
        integral=integral,
        neg_frac=neg_frac,
        rn_mean=mean,
        rn_var=var,
        note="nonuniform grid",
    )

    logger.info("BL density extraction completed successfully")
    return K, pdf, diag


def bl_pdf_from_calls(
    strikes: ArrayLike,
    calls: ArrayLike,
    r: float,
    T: float,
    *,
    grid_n: int = 401,
    clip_negative: bool = True,
    renormalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, BLDiagnostics]:
    """
    Smoothed-PCHIP + uniform-grid BL density (recommended).

    Steps
    -----
    1) Sort and drop NaNs in (K, C).
    2) Fit a shape-preserving PCHIP interpolator for C(K).
    3) Sample uniformly on [K_min, K_max] with `grid_n` points.
    4) Apply central uniform second differences to get C''(K).
    5) f(K) = exp(rT) * C''(K); clip negatives and (optionally) renormalize.

    Returns
    -------
    (K_u, pdf_u, diag)
    """
    logger.info("Starting BL density extraction with PCHIP smoothing")
    logger.debug(
        f"Parameters: r={r:.6f}, T={T:.4f}, grid_n={grid_n}, "
        f"clip_negative={clip_negative}, renormalize={renormalize}"
    )

    K = np.asarray(strikes, dtype=float).copy()
    C = np.asarray(calls, dtype=float).copy()

    initial_count = len(K)
    logger.debug(f"Input: {initial_count} strike-call pairs")
    logger.debug(f"Strike range: [{K.min():.2f}, {K.max():.2f}]")
    logger.debug(f"Call price range: [{C.min():.6f}, {C.max():.6f}]")

    # Filter and validate
    m = np.isfinite(K) & np.isfinite(C)
    K, C = K[m], C[m]
    filtered_count = len(K)

    if filtered_count < initial_count:
        logger.warning(
            f"Filtered out {initial_count - filtered_count} non-finite values"
        )

    if K.size < 3:
        logger.error(f"Insufficient data after filtering: {K.size} < 3")
        raise ValueError("Need at least 3 call points for second derivative.")

    # Sort by strikes
    order = np.argsort(K)
    K, C = K[order], C[order]
    logger.debug("Sorted by strike price")

    # Check SciPy availability
    if not _HAVE_SCIPY:
        logger.warning(
            "SciPy not available, falling back to non-uniform method"
        )
        return bl_pdf_from_calls_nonuniform(
            K, C, r, T, clip_negative=clip_negative, renormalize=renormalize
        )

    # Fit PCHIP interpolator
    logger.debug("Fitting PCHIP interpolator")
    try:
        pchip = PchipInterpolator(K, C, extrapolate=False)
        logger.debug("PCHIP interpolator created successfully")
    except Exception as e:
        logger.error(f"PCHIP interpolation failed: {e}")
        logger.info("Falling back to non-uniform method")
        return bl_pdf_from_calls_nonuniform(
            K, C, r, T, clip_negative=clip_negative, renormalize=renormalize
        )

    # Create uniform grid
    K_u = np.linspace(K.min(), K.max(), int(grid_n))
    logger.debug(
        f"Created uniform grid: {len(K_u)} points from {K_u[0]:.2f} to {K_u[-1]:.2f}"
    )

    # Evaluate interpolator
    C_u = pchip(K_u)

    # Handle potential NaNs from extrapolation
    nan_count = np.isnan(C_u).sum()
    if nan_count > 0:
        logger.warning(
            f"PCHIP produced {nan_count} NaN values, using linear interpolation"
        )
        C_u = np.interp(K_u, K, C)

    logger.debug(
        f"Interpolated call prices range: [{C_u.min():.6f}, {C_u.max():.6f}]"
    )

    # Compute second derivative on uniform grid
    h = float((K_u[-1] - K_u[0]) / max(K_u.size - 1, 1))
    logger.debug(f"Uniform grid spacing: {h:.6f}")

    Cpp_u = _central_second_uniform(C_u, h)

    # Apply BL formula
    discount_factor = np.exp(r * T)
    pdf = discount_factor * Cpp_u
    logger.debug(
        f"Applied BL formula with discount factor {discount_factor:.6f}"
    )

    # Analyze and handle negative values
    neg_frac = float(np.mean(pdf < 0.0)) if pdf.size else 0.0
    logger.info(f"Negative density fraction (pre-clip): {neg_frac:.4f}")

    if clip_negative:
        pdf = np.where(pdf > 0.0, pdf, 0.0)
        logger.debug("Clipped negative values to zero")

    # Compute integral and normalize
    integral = _trapz(K_u, pdf)
    logger.info(f"PDF integral before normalization: {integral:.6f}")

    if renormalize and integral > 0.0:
        pdf = pdf / integral
        logger.debug(f"Renormalized PDF by factor {1/integral:.6f}")
        integral = 1.0

    # Compute moments
    logger.debug("Computing risk-neutral moments")
    mean = _trapz(K_u, K_u * pdf)
    var = max(_trapz(K_u, (K_u - mean) ** 2 * pdf), 0.0)

    logger.info(f"BL density results: mean={mean:.2f}, var={var:.6f}")

    diag = BLDiagnostics(
        integral=integral,
        neg_frac=neg_frac,
        rn_mean=mean,
        rn_var=var,
        note="pchipped uniform grid",
    )

    logger.info("BL density extraction with PCHIP completed successfully")
    return K_u, pdf, diag
