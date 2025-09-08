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

import logging
from typing import Dict, Iterable, Tuple

import numpy as np

# Set up module logger
logger = logging.getLogger(__name__)


def build_cdf(K: np.ndarray, pdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trapezoid-accumulate a CDF from a non-negative pdf sample.
    Renormalizes the final CDF to end at 1.0 (if integral > 0).
    """
    logger.debug(f"Building CDF from PDF: {len(K)} points")

    K = np.asarray(K, dtype=float)
    f = np.asarray(pdf, dtype=float)

    # Input validation
    if len(K) != len(f):
        logger.error(
            f"Array length mismatch: K has {len(K)} points, pdf has {len(f)} points"
        )
        raise ValueError("K and pdf must have the same length")

    logger.debug(f"K range: [{K.min():.4f}, {K.max():.4f}]")
    logger.debug(f"PDF range: [{f.min():.8f}, {f.max():.8f}]")

    # Clean up PDF values
    original_f = f.copy()
    f = np.where(np.isfinite(f) & (f >= 0.0), f, 0.0)

    # Log cleaning statistics
    nan_count = (~np.isfinite(original_f)).sum()
    neg_count = (original_f < 0).sum()
    if nan_count > 0:
        logger.warning(f"Set {nan_count} non-finite PDF values to zero")
    if neg_count > 0:
        logger.warning(f"Set {neg_count} negative PDF values to zero")

    # Manual cumulative trapezoid to avoid SciPy dependency
    c = np.zeros_like(f, dtype=float)
    logger.debug("Computing cumulative trapezoid integration")

    for i in range(1, f.size):
        h = K[i] - K[i - 1]
        if h <= 0:
            logger.warning(f"Non-positive spacing at index {i}: h={h:.8f}")
        c[i] = c[i - 1] + 0.5 * h * (f[i] + f[i - 1])

    total_integral = c[-1] if c.size > 0 else 0.0
    logger.debug(f"Total PDF integral: {total_integral:.8f}")

    # Normalize CDF to end at 1.0
    if total_integral > 0.0:
        c = c / total_integral
        logger.debug(f"Normalized CDF by factor {1/total_integral:.8f}")
    else:
        logger.warning("Zero integral - CDF will be all zeros")

    # Validate CDF properties
    if c.size > 0:
        logger.debug(f"Final CDF range: [{c.min():.8f}, {c.max():.8f}]")

        # Check monotonicity
        cdf_decreases = (np.diff(c) < -1e-10).sum()
        if cdf_decreases > 0:
            logger.warning(
                f"CDF not monotonic: {cdf_decreases} decreases detected"
            )

    logger.debug("CDF construction completed")
    return K, c


def ppf_from_cdf(
    K: np.ndarray, cdf: np.ndarray, qs: Iterable[float]
) -> np.ndarray:
    """
    Invert the CDF via linear interpolation. Values outside [0,1]
    are clipped. Ensures monotone CDF.
    """
    K = np.asarray(K, dtype=float)
    c = np.asarray(cdf, dtype=float)
    q = np.clip(np.asarray(list(qs), dtype=float), 0.0, 1.0)

    logger.debug(
        f"Computing quantiles: {len(q)} probabilities for {len(K)} CDF points"
    )
    logger.debug(f"Quantile range: [{q.min():.4f}, {q.max():.4f}]")

    # Input validation
    if len(K) != len(c):
        logger.error(
            f"Array length mismatch: K has {len(K)} points, CDF has {len(c)} points"
        )
        raise ValueError("K and cdf must have the same length")

    # Enforce monotonicity in CDF
    original_c = c.copy()
    c = np.maximum.accumulate(c)

    # Log monotonicity enforcement
    violations = (c != original_c).sum()
    if violations > 0:
        logger.info(
            f"Enforced monotonicity: corrected {violations} CDF values"
        )

    logger.debug(
        f"CDF range after monotonicity: [{c.min():.8f}, {c.max():.8f}]"
    )

    # Check CDF coverage
    if c.max() < 0.99:
        logger.warning(f"CDF maximum is only {c.max():.6f} (should be ≈1.0)")
    if c.min() > 0.01:
        logger.warning(f"CDF minimum is {c.min():.6f} (should be ≈0.0)")

    # Interpolate to find quantiles
    quantiles = np.interp(q, c, K)

    logger.debug(
        f"Computed quantiles range: [{quantiles.min():.4f}, {quantiles.max():.4f}]"
    )

    # Log some key quantiles for validation
    if len(q) >= 3:
        median_idx = len(q) // 2
        logger.debug(
            f"Sample quantiles: p={q[0]:.2f}→{quantiles[0]:.2f}, "
            f"p={q[median_idx]:.2f}→{quantiles[median_idx]:.2f}, "
            f"p={q[-1]:.2f}→{quantiles[-1]:.2f}"
        )

    return quantiles


def moments_from_pdf(K: np.ndarray, pdf: np.ndarray) -> Dict[str, float]:
    """
    Compute mean, variance, skewness, and excess kurtosis for a pdf.
    Assumes `pdf` is non-negative and ∫pdf dK ≈ 1 (renormalizes if needed).
    Uses trapezoid integration.
    """
    logger.debug(f"Computing moments from PDF: {len(K)} points")

    K = np.asarray(K, dtype=float)
    f = np.asarray(pdf, dtype=float)

    # Input validation
    if len(K) != len(f):
        logger.error(
            f"Array length mismatch: K has {len(K)} points, pdf has {len(f)} points"
        )
        raise ValueError("K and pdf must have the same length")

    logger.debug(f"K range: [{K.min():.4f}, {K.max():.4f}]")
    logger.debug(f"PDF range: [{f.min():.8f}, {f.max():.8f}]")

    # Clean PDF
    original_f = f.copy()
    f = np.where(np.isfinite(f) & (f >= 0.0), f, 0.0)

    # Log cleaning
    cleaned_count = (f != original_f).sum()
    if cleaned_count > 0:
        logger.warning(f"Cleaned {cleaned_count} invalid PDF values")

    # Check normalization
    Z = float(np.trapezoid(f, K))
    logger.debug(f"PDF integral: {Z:.8f}")

    if Z <= 0.0:
        logger.error(
            "PDF integral is zero or negative - cannot compute moments"
        )
        return {
            "mean": np.nan,
            "var": np.nan,
            "skew": np.nan,
            "exkurt": np.nan,
        }

    # Renormalize if needed
    if abs(Z - 1.0) > 0.001:
        logger.info(f"Renormalizing PDF by factor {1/Z:.8f}")
        f = f / Z

    # Compute mean
    mean = float(np.trapezoid(K * f, K))
    logger.debug(f"Mean: {mean:.6f}")

    # Compute variance
    var = float(np.trapezoid((K - mean) ** 2 * f, K))
    logger.debug(f"Variance: {var:.8f}")

    if var <= 0.0:
        logger.warning(f"Non-positive variance: {var:.8f}")
        return {"mean": mean, "var": var, "skew": np.nan, "exkurt": np.nan}

    # Compute higher moments
    std = np.sqrt(var)
    logger.debug(f"Standard deviation: {std:.6f}")

    # Standardized moments
    z_scores = (K - mean) / std
    m3 = float(np.trapezoid(z_scores**3 * f, K))  # skewness
    m4 = float(np.trapezoid(z_scores**4 * f, K))  # kurtosis
    exkurt = m4 - 3.0  # excess kurtosis

    logger.info(
        f"Moments computed: mean={mean:.4f}, std={std:.4f}, "
        f"skew={m3:.4f}, exkurt={exkurt:.4f}"
    )

    # Validate moment ranges
    if abs(m3) > 10:
        logger.warning(f"Extreme skewness: {m3:.4f}")
    if abs(exkurt) > 20:
        logger.warning(f"Extreme excess kurtosis: {exkurt:.4f}")

    return {"mean": mean, "var": var, "skew": m3, "exkurt": exkurt}
