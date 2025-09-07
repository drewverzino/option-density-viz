# src/vol/no_arb.py
from __future__ import annotations

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

from typing import Any, Dict

import numpy as np


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
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if n < 3:
        return out

    # Interior spacings and numerator/denominator of the formula.
    h0 = x[1:-1] - x[:-2]
    h1 = x[2:] - x[1:-1]
    num = h0 * y[2:] - (h0 + h1) * y[1:-1] + h1 * y[:-2]
    den = h0 * h1 * (h0 + h1)

    # Multiply by 2 as per the derivation above.
    out[1:-1] = 2.0 * num / den
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
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)
    order = np.argsort(K)
    K = K[order]
    C = C[order]

    curv = _second_derivative_nonuniform(K, C)
    mask = np.isfinite(curv)
    interior = np.where(mask)[0]
    if interior.size == 0:
        return {
            "fraction": 0.0,
            "count": 0,
            "n_interior": 0,
            "indices": np.array([], dtype=int),
            "curvature": curv,
        }

    bad = np.where(curv[interior] < -abs(tol))[0]
    count = int(bad.size)
    frac = float(count) / float(interior.size)
    return {
        "fraction": frac,
        "count": count,
        "n_interior": int(interior.size),
        "indices": interior[bad],
        "curvature": curv,
    }


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
    k = np.asarray(k, dtype=float)
    w1 = np.asarray(w_T1, dtype=float)
    w2 = np.asarray(w_T2, dtype=float)

    # Keep only finite aligned entries.
    mask = np.isfinite(k) & np.isfinite(w1) & np.isfinite(w2)
    k, w1, w2 = k[mask], w1[mask], w2[mask]
    if len(k) == 0:
        return {"fraction": 0.0, "min_gap": 0.0, "max_gap": 0.0}

    dif = w2 - w1
    viol = np.where(dif < -abs(tol))[0]
    frac = float(viol.size) / float(len(k))
    return {
        "fraction": frac,
        "min_gap": float(dif.min()),
        "max_gap": float(dif.max()),
    }
