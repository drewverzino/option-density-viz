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

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator


@dataclass
class BLDiagnostics:
    """Summary statistics and sanity checks for a recovered PDF."""

    integral: float  # ∫ pdf dK on the computed grid
    neg_frac: float  # fraction of grid points with pdf < 0 (pre-clip)
    rn_mean: float  # ∫ K * pdf dK
    rn_var: float  # variance around rn_mean
    note: str = ""  # freeform notes


def _trapz(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoid rule with float return."""
    return float(np.trapezoid(y, x))


def _central_second_uniform(y: np.ndarray, h: float) -> np.ndarray:
    """
    Second derivative for uniform spacing using the classic central
    three-point stencil:

        y''_i ≈ (y_{i+1} - 2 y_i + y_{i-1}) / h^2

    Endpoints are set to 0 (no info there, and avoids NaNs).
    """
    out = np.zeros_like(y, dtype=float)
    if y.size >= 3:
        out[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (h * h)
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
    out = np.zeros(n, dtype=float)
    if n < 3:
        return out
    h0 = x[1:-1] - x[:-2]
    h1 = x[2:] - x[1:-1]
    a = 2.0 / (h0 * (h0 + h1))
    b = -2.0 / (h0 * h1)
    c = 2.0 / (h1 * (h0 + h1))
    out[1:-1] = a * y[:-2] + b * y[1:-1] + c * y[2:]
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
    K, pdf, diag
    """
    K = np.asarray(strikes, dtype=float).copy()
    C = np.asarray(calls, dtype=float).copy()

    m = np.isfinite(K) & np.isfinite(C)
    K, C = K[m], C[m]
    if K.size < 3:
        raise ValueError("Need at least 3 call points for second derivative.")

    order = np.argsort(K)
    K, C = K[order], C[order]

    Cpp = _second_nonuniform(K, C)
    pdf = np.exp(r * T) * Cpp

    # pre-clip negativity fraction (treat NaNs as non-negative)
    neg_frac = float(np.mean(pdf < 0.0)) if pdf.size else 0.0

    if clip_negative:
        pdf = np.where(np.isfinite(pdf) & (pdf > 0.0), pdf, 0.0)
    else:
        pdf = np.where(np.isfinite(pdf), pdf, 0.0)

    integral = _trapz(K, pdf)
    if renormalize and integral > 0.0:
        pdf = pdf / integral
        integral = 1.0

    mean = _trapz(K, K * pdf)
    var = max(_trapz(K, (K - mean) ** 2 * pdf), 0.0)

    diag = BLDiagnostics(
        integral=integral,
        neg_frac=neg_frac,
        rn_mean=mean,
        rn_var=var,
        note="nonuniform grid",
    )
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
    K_u, pdf_u, diag
    """
    K = np.asarray(strikes, dtype=float).copy()
    C = np.asarray(calls, dtype=float).copy()

    m = np.isfinite(K) & np.isfinite(C)
    K, C = K[m], C[m]
    if K.size < 3:
        raise ValueError("Need at least 3 call points for second derivative.")

    order = np.argsort(K)
    K, C = K[order], C[order]

    pchip = PchipInterpolator(K, C, extrapolate=False)

    K_u = np.linspace(K.min(), K.max(), int(grid_n))
    C_u = pchip(K_u)

    # If PCHIP produced NaNs at ends, fill by linear interp on (K, C)
    if np.isnan(C_u).any():
        C_u = np.interp(K_u, K, C)

    h = float((K_u[-1] - K_u[0]) / max(K_u.size - 1, 1))
    Cpp_u = _central_second_uniform(C_u, h)

    pdf = np.exp(r * T) * Cpp_u

    neg_frac = float(np.mean(pdf < 0.0)) if pdf.size else 0.0
    if clip_negative:
        pdf = np.where(pdf > 0.0, pdf, 0.0)

    integral = _trapz(K_u, pdf)
    if renormalize and integral > 0.0:
        pdf = pdf / integral
        integral = 1.0

    mean = _trapz(K_u, K_u * pdf)
    var = max(_trapz(K_u, (K_u - mean) ** 2 * pdf), 0.0)

    diag = BLDiagnostics(
        integral=integral,
        neg_frac=neg_frac,
        rn_mean=mean,
        rn_var=var,
        note="pchipped uniform grid",
    )
    return K_u, pdf, diag
