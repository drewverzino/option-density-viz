"""
No-arbitrage diagnostics for smiles and surfaces.

What these checks mean
----------------------
- Butterfly (static arbitrage across strikes, single expiry):
  Call price C(K) must be a convex function of K. Negative second
  derivative indicates a butterfly arbitrage opportunity.

- Calendar (monotonicity across maturities):
  Total variance w(k, T) should be non-decreasing in T on a shared
  k-grid. If w decreases with maturity, there is calendar arbitrage.

What this module provides
-------------------------
- butterfly_violations(K, C, tol):
    Fraction and indices of K where convexity fails (finite differences).
- calendar_violations(k, w_T1, w_T2, tol):
    Fraction and gaps where w(k, T2) < w(k, T1) by more than tol.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

# Set up module logger
logger = logging.getLogger("vol.no_arb")


def _second_derivative_nonuniform(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Three-point second derivative on a non-uniform grid.

    Let x_{i-1} < x_i < x_{i+1}, with spacings
      h0 = x_i - x_{i-1},   h1 = x_{i+1} - x_i.
    A consistent finite-difference approximation is:
      y''(x_i) ~= 2 * [ h0*y_{i+1} - (h0+h1)*y_i + h1*y_{i-1} ]
                       / ( h0*h1*(h0+h1) )

    We return an array of length n with NaNs at the endpoints because
    curvature is undefined there for a 3-point stencil.
    """
    logger.debug(
        f"Computing second derivative on non-uniform grid: {len(x)} points"
    )

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if len(y) != n:
        logger.error(
            f"Array length mismatch: x has {n} points, y has {len(y)} points"
        )
        raise ValueError("x and y must have the same length")

    logger.debug(
        f"Input ranges: x ∈ [{x.min():.6f}, {x.max():.6f}], y ∈ [{y.min():.6f}, {y.max():.6f}]"
    )

    out = np.full(n, np.nan, dtype=float)

    if n < 3:
        logger.warning(f"Insufficient points for second derivative: {n} < 3")
        return out

    # Check for monotonicity in x (required for meaningful derivatives)
    x_diffs = np.diff(x)
    if np.any(x_diffs <= 0):
        non_monotonic = (x_diffs <= 0).sum()
        logger.warning(
            f"Grid not strictly increasing: {non_monotonic} non-positive differences"
        )
        # Sort if needed
        if non_monotonic > 0:
            sort_idx = np.argsort(x)
            x = x[sort_idx]
            y = y[sort_idx]
            logger.debug("Sorted arrays by x values")

    # Interior spacings and numerator/denominator of the formula
    h0 = x[1:-1] - x[:-2]  # spacing to left
    h1 = x[2:] - x[1:-1]  # spacing to right

    # Check for zero spacings (would cause division by zero)
    zero_h0 = (h0 == 0).sum()
    zero_h1 = (h1 == 0).sum()
    if zero_h0 > 0 or zero_h1 > 0:
        logger.warning(f"Zero spacings detected: h0={zero_h0}, h1={zero_h1}")

    num = h0 * y[2:] - (h0 + h1) * y[1:-1] + h1 * y[:-2]
    den = h0 * h1 * (h0 + h1)

    # Avoid division by zero
    valid_den = den != 0
    n_valid = valid_den.sum()
    n_interior = len(den)

    if n_valid < n_interior:
        logger.warning(
            f"Zero denominators: {n_interior - n_valid}/{n_interior} interior points unusable"
        )

    # Multiply by 2 as per the derivation above
    out[1:-1] = np.where(valid_den, 2.0 * num / den, np.nan)

    # Log statistics
    finite_curvature = np.isfinite(out[1:-1])
    if finite_curvature.any():
        curv_values = out[1:-1][finite_curvature]
        logger.debug(
            f"Second derivative: {finite_curvature.sum()}/{n_interior} finite interior values"
        )
        logger.debug(
            f"Curvature range: [{curv_values.min():.8f}, {curv_values.max():.8f}]"
        )

        negative_curv = (curv_values < 0).sum()
        if negative_curv > 0:
            logger.debug(
                f"Negative curvature at {negative_curv}/{len(curv_values)} interior points"
            )
    else:
        logger.warning("No finite curvature values computed")

    return out


def butterfly_violations(
    K: np.ndarray, C: np.ndarray, *, tol: float = 1e-10
) -> Dict[str, Any]:
    """
    Diagnose butterfly arbitrage on a single expiry using convexity.

    Parameters
    ----------
    K : array
        Strike grid. Will be sorted ascending internally.
    C : array
        Call prices on the same strikes.
    tol : float
        Negative curvature below -tol counts as a violation. Use a
        small positive tol to ignore harmless numerical noise.

    Returns
    -------
    dict with:
      fraction : float    # violations / interior points
      count : int         # number of interior violations
      n_interior : int    # number of points with curvature defined
      indices : ndarray   # indices (in the sorted grid) of violations
      curvature : ndarray # full curvature array with NaNs at edges

    Notes
    -----
    - If the input K is unsorted, we sort K and reorder C to match.
    - The result indices correspond to this sorted order.
    """
    logger.info("=" * 50)
    logger.info("BUTTERFLY ARBITRAGE ANALYSIS")
    logger.info("=" * 50)

    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)

    n_input = len(K)
    logger.info(f"Input: {n_input} strike-price pairs")
    logger.info(f"Tolerance: {tol:.2e} (negative curvature threshold)")

    if len(C) != n_input:
        logger.error(
            f"Array length mismatch: K has {n_input} points, C has {len(C)} points"
        )
        raise ValueError("K and C must have the same length")

    logger.debug(f"Strike range: [{K.min():.2f}, {K.max():.2f}]")
    logger.debug(f"Price range: [{C.min():.6f}, {C.max():.6f}]")

    # Check if sorting is needed
    order = np.argsort(K)
    if not np.array_equal(order, np.arange(len(K))):
        logger.info("Sorting by strike price")
        K = K[order]
        C = C[order]
        logger.debug(
            f"Sorted ranges: K ∈ [{K.min():.2f}, {K.max():.2f}], C ∈ [{C.min():.6f}, {C.max():.6f}]"
        )
    else:
        logger.debug("Data already sorted by strike")

    # Validate call price monotonicity (rough sanity check)
    price_increases = (np.diff(C) > 0).sum()
    price_decreases = (np.diff(C) < 0).sum()
    price_flat = (np.diff(C) == 0).sum()

    logger.debug(
        f"Price monotonicity: {price_increases} increases, {price_decreases} decreases, {price_flat} flat"
    )

    if price_increases < price_decreases:
        logger.warning(
            "Call prices mostly decreasing with strike - this is unusual"
        )

    # Compute second derivative (curvature)
    logger.info("Computing call price curvature")
    curv = _second_derivative_nonuniform(K, C)

    # Find interior points where curvature is defined
    mask = np.isfinite(curv)
    interior = np.where(mask)[0]
    n_interior = interior.size

    logger.info(f"Interior points with defined curvature: {n_interior}")

    if n_interior == 0:
        logger.warning("No interior points with defined curvature")
        return {
            "fraction": 0.0,
            "count": 0,
            "n_interior": 0,
            "indices": np.array([], dtype=int),
            "curvature": curv,
        }

    # Find violations (negative curvature beyond tolerance)
    curv_interior = curv[interior]
    violation_mask = curv_interior < -abs(tol)
    bad_indices = interior[violation_mask]

    n_violations = len(bad_indices)
    violation_fraction = float(n_violations) / float(n_interior)

    # Detailed logging
    logger.info(f"Curvature statistics:")
    logger.info(
        f"  Range: [{curv_interior.min():.8f}, {curv_interior.max():.8f}]"
    )
    logger.info(f"  Mean: {curv_interior.mean():.8f}")
    logger.info(f"  Negative values: {(curv_interior < 0).sum()}/{n_interior}")

    logger.info(f"Arbitrage violations:")
    logger.info(f"  Count: {n_violations}/{n_interior} interior points")
    logger.info(f"  Fraction: {violation_fraction:.1%}")

    if n_violations > 0:
        worst_violation = curv_interior[violation_mask].min()
        worst_strike = K[bad_indices[np.argmin(curv_interior[violation_mask])]]
        logger.warning(
            f"Worst violation: curvature={worst_violation:.8f} at K={worst_strike:.2f}"
        )

        # Log all violations if not too many
        if n_violations <= 10:
            logger.warning("Violation details:")
            for i, idx in enumerate(bad_indices):
                logger.warning(
                    f"  #{i+1}: K={K[idx]:.2f}, C={C[idx]:.6f}, curvature={curv[idx]:.8f}"
                )
    else:
        logger.info("No butterfly arbitrage violations detected")

    result = {
        "fraction": violation_fraction,
        "count": n_violations,
        "n_interior": n_interior,
        "indices": bad_indices,
        "curvature": curv,
    }

    logger.info("Butterfly analysis complete")
    return result


def calendar_violations(
    k: np.ndarray, w_T1: np.ndarray, w_T2: np.ndarray, *, tol: float = 1e-10
) -> Dict[str, float]:
    """
    Diagnose calendar arbitrage on a shared k-grid (two maturities).

    We check that total variance is non-decreasing with maturity:
      w(k, T2) >= w(k, T1) - tol

    Parameters
    ----------
    k : array
        Shared log-moneyness grid.
    w_T1 : array
        Total variance at the near maturity T1.
    w_T2 : array
        Total variance at the far maturity T2.
    tol : float
        Tolerance to ignore tiny negative gaps from numerical noise.

    Returns
    -------
    dict with:
      fraction : float  # share of k where w_T2 < w_T1 - tol
      min_gap : float   # min(w_T2 - w_T1)
      max_gap : float   # max(w_T2 - w_T1)
    """
    logger.info("=" * 50)
    logger.info("CALENDAR ARBITRAGE ANALYSIS")
    logger.info("=" * 50)

    k = np.asarray(k, dtype=float)
    w1 = np.asarray(w_T1, dtype=float)
    w2 = np.asarray(w_T2, dtype=float)

    n_input = len(k)
    logger.info(f"Input: {n_input} log-moneyness points")
    logger.info(f"Tolerance: {tol:.2e} (calendar violation threshold)")

    # Validate input lengths
    if len(w1) != n_input:
        logger.error(
            f"Array length mismatch: k has {n_input} points, w_T1 has {len(w1)} points"
        )
        raise ValueError("k and w_T1 must have the same length")

    if len(w2) != n_input:
        logger.error(
            f"Array length mismatch: k has {n_input} points, w_T2 has {len(w2)} points"
        )
        raise ValueError("k and w_T2 must have the same length")

    logger.debug(f"Log-moneyness range: [{k.min():.3f}, {k.max():.3f}]")
    logger.debug(f"T1 total variance range: [{w1.min():.6f}, {w1.max():.6f}]")
    logger.debug(f"T2 total variance range: [{w2.min():.6f}, {w2.max():.6f}]")

    # Keep only finite aligned entries
    logger.debug("Filtering for finite values across all arrays")
    mask = np.isfinite(k) & np.isfinite(w1) & np.isfinite(w2)
    n_finite = mask.sum()

    if n_finite < n_input:
        logger.info(
            f"Finite value filtering: kept {n_finite}/{n_input} points ({n_finite/n_input:.1%})"
        )

    if n_finite == 0:
        logger.warning("No finite values found across all arrays")
        return {"fraction": 0.0, "min_gap": 0.0, "max_gap": 0.0}

    k, w1, w2 = k[mask], w1[mask], w2[mask]

    logger.debug(f"Filtered ranges:")
    logger.debug(f"  k: [{k.min():.3f}, {k.max():.3f}]")
    logger.debug(f"  w_T1: [{w1.min():.6f}, {w1.max():.6f}]")
    logger.debug(f"  w_T2: [{w2.min():.6f}, {w2.max():.6f}]")

    # Compute variance differences (gap = w_T2 - w_T1)
    dif = w2 - w1
    logger.debug(
        f"Variance gaps (w_T2 - w_T1): [{dif.min():.8f}, {dif.max():.8f}]"
    )

    # Check for violations: w_T2 < w_T1 - tol
    violation_threshold = -abs(tol)
    violation_mask = dif < violation_threshold
    n_violations = violation_mask.sum()
    violation_fraction = float(n_violations) / float(len(k))

    # Summary statistics
    min_gap = float(dif.min())
    max_gap = float(dif.max())
    mean_gap = float(dif.mean())

    logger.info(f"Gap statistics:")
    logger.info(f"  Min gap (most negative): {min_gap:.8f}")
    logger.info(f"  Max gap (most positive): {max_gap:.8f}")
    logger.info(f"  Mean gap: {mean_gap:.8f}")

    logger.info(f"Calendar violations:")
    logger.info(f"  Count: {n_violations}/{len(k)} points")
    logger.info(f"  Fraction: {violation_fraction:.1%}")

    if n_violations > 0:
        worst_gap = dif[violation_mask].min()
        worst_k = k[violation_mask][np.argmin(dif[violation_mask])]
        logger.warning(
            f"Worst violation: gap={worst_gap:.8f} at k={worst_k:.3f}"
        )

        # Additional violation statistics
        violation_gaps = dif[violation_mask]
        logger.warning(
            f"Violation gap range: [{violation_gaps.min():.8f}, {violation_gaps.max():.8f}]"
        )
        logger.warning(f"Mean violation gap: {violation_gaps.mean():.8f}")

        # Log individual violations if not too many
        if n_violations <= 10:
            logger.warning("Violation details:")
            violation_indices = np.where(violation_mask)[0]
            for i, idx in enumerate(violation_indices):
                logger.warning(
                    f"  #{i+1}: k={k[idx]:.3f}, w_T1={w1[idx]:.6f}, w_T2={w2[idx]:.6f}, gap={dif[idx]:.8f}"
                )
        elif n_violations <= 50:
            logger.warning(
                f"Large number of violations ({n_violations}), showing worst 5:"
            )
            worst_indices = np.argsort(dif[violation_mask])[:5]
            violation_indices = np.where(violation_mask)[0]
            for i, worst_idx in enumerate(worst_indices):
                idx = violation_indices[worst_idx]
                logger.warning(
                    f"  #{i+1}: k={k[idx]:.3f}, w_T1={w1[idx]:.6f}, w_T2={w2[idx]:.6f}, gap={dif[idx]:.8f}"
                )
    else:
        logger.info("No calendar arbitrage violations detected")

    # Additional insights
    positive_gaps = (dif > 0).sum()
    zero_gaps = (dif == 0).sum()
    negative_gaps = (dif < 0).sum()

    logger.debug(
        f"Gap distribution: {positive_gaps} positive, {zero_gaps} zero, {negative_gaps} negative"
    )

    if negative_gaps > 0 and n_violations == 0:
        logger.info(
            f"Note: {negative_gaps} negative gaps exist but all within tolerance"
        )

    result = {
        "fraction": violation_fraction,
        "min_gap": min_gap,
        "max_gap": max_gap,
    }

    logger.info("Calendar analysis complete")
    return result
