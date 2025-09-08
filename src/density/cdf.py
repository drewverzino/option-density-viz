"""
CDF and quantile helpers for densities defined on a strike grid.

Given a (K, pdf) pair that approximates a risk-neutral density on a
finite interval [K_min, K_max], we provide:

- build_cdf(K, pdf) -> (K, cdf): trapezoid-integrated CDF, re-normalized.
- ppf_from_cdf(K, cdf, qs) -> quantiles for probabilities 0..1.
- moments_from_pdf(K, pdf) -> dict of mean/var/skew/kurtosis (excess).

These are generic utilities; they do not assume the pdf came from BL.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def build_cdf(K: np.ndarray, pdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trapezoid-accumulate a CDF from a non-negative pdf sample.
    Renormalizes the final CDF to end at 1.0 (if integral > 0).
    """
    K = np.asarray(K, dtype=float)
    f = np.asarray(pdf, dtype=float)
    f = np.where(np.isfinite(f) & (f >= 0.0), f, 0.0)

    c = np.zeros_like(f, dtype=float)
    # cumulative trapezoid (manual to avoid SciPy dep)
    for i in range(1, f.size):
        h = K[i] - K[i - 1]
        c[i] = c[i - 1] + 0.5 * h * (f[i] + f[i - 1])

    if c[-1] > 0.0:
        c = c / c[-1]  # normalize to 1
    return K, c


def ppf_from_cdf(
    K: np.ndarray, cdf: np.ndarray, qs: Iterable[float]
) -> np.ndarray:
    """
    Invert the CDF via linear interpolation. Values outside [0,1]
    are clipped.
    """
    K = np.asarray(K, dtype=float)
    c = np.asarray(cdf, dtype=float)
    q = np.clip(np.asarray(list(qs), dtype=float), 0.0, 1.0)
    # Ensure CDF is non-decreasing
    c = np.maximum.accumulate(c)
    return np.interp(q, c, K)


def moments_from_pdf(K: np.ndarray, pdf: np.ndarray) -> Dict[str, float]:
    """
    Compute mean, variance, skewness, and excess kurtosis for a pdf.
    Assumes `pdf` is non-negative and ∫pdf dK ≈ 1 (we renormalize if
    needed). Uses trapezoid integration.
    """
    K = np.asarray(K, dtype=float)
    f = np.asarray(pdf, dtype=float)
    f = np.where(np.isfinite(f) & (f >= 0.0), f, 0.0)

    # Renormalize if necessary
    Z = float(np.trapezoid(f, K))
    if Z <= 0.0:
        return {
            "mean": np.nan,
            "var": np.nan,
            "skew": np.nan,
            "exkurt": np.nan,
        }
    f = f / Z

    mean = float(np.trapezoid(K * f, K))
    var = float(np.trapezoid((K - mean) ** 2 * f, K))
    if var <= 0.0:
        return {"mean": mean, "var": var, "skew": np.nan, "exkurt": np.nan}

    std = np.sqrt(var)
    m3 = float(np.trapezoid(((K - mean) / std) ** 3 * f, K))
    m4 = float(np.trapezoid(((K - mean) / std) ** 4 * f, K))
    return {"mean": mean, "var": var, "skew": m3, "exkurt": m4 - 3.0}
