"""
SVI (Stochastic Volatility Inspired) calibration utilities.

This module fits *per-expiry* SVI parameters to observed implied
volatility smiles expressed in *log-moneyness* k and *total variance*
w = sigma^2 * T.

Key ideas:
- We fit in *total variance* space (not IV) because SVI is linear in T
  and many no-arbitrage properties are easier to reason about in w.
- To approximate *price-space* least squares while still fitting w,
  we optionally weight residuals by the Black-76 *vega* and the chain
  rule factor d sigma / d w = 1 / (2 T sigma). See `_price_weights`.
- We expose one public entry point:
    `calibrate_svi_from_quotes(k, w=..., iv=..., T=..., ...) -> SVIFit`
  which returns a small dataclass with parameters and diagnostics.

Notation:
- k = log-moneyness = ln(K/F) (we adopt the market convention K/F, so
  negative k means ITM calls / OTM puts).
- w(k) = a + b * { rho * (k - m) + sqrt((k - m)^2 + sigma^2) }
  with constraints:
    b > 0, sigma > 0, |rho| < 1.  (We also clamp rho to (-0.999, 0.999))
- T is year-fraction to expiry (ACT/365.25 in the rest of the repo).

References:
- Gatheral, "The Volatility Surface: A Practitioner's Guide".
- SVI parameterization used industry-wide for robust smile fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

# ------------------------------ Data Types ------------------------------ #


@dataclass
class SVIFit:
    """
    Container for a single-expiry SVI fit.

    Attributes
    ----------
    params : tuple[float, float, float, float, float]
        (a, b, rho, m, sigma) SVI parameters.
    loss : float
        Root-mean-squared residual in the objective space (after weights).
    n_used : int
        Number of quotes actually used in the fit after filtering.
    method : str
        Name of the optimizer (e.g., 'L-BFGS-B').
    notes : str
        Human-readable comments about bounds, seeds, or any fallback.
    """

    params: Tuple[float, float, float, float, float]
    loss: float
    n_used: int
    method: str = "L-BFGS-B"
    notes: str = ""


# ------------------------------ Math Utils ------------------------------ #

SQRT2PI = np.sqrt(2.0 * np.pi)


def _phi(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal PDF φ(x)."""
    return np.exp(-0.5 * np.square(x)) / SQRT2PI


def _ensure_1d(a: ArrayLike, name: str) -> np.ndarray:
    """Convert to 1D float array and sanity-check finite-ness."""
    x = np.asarray(a, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(x).any():
        raise ValueError(f"{name} has no finite values")
    return x


def _filter_finite(*arrs: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Keep positions where *all* arrays are finite."""
    m = np.logical_and.reduce([np.isfinite(a) for a in arrs])
    return tuple(a[m] for a in arrs)


# ------------------------------ SVI Model ------------------------------- #


def svi_total_variance(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """
    SVI total variance function:

        w(k) = a + b * { rho * (k - m) + sqrt((k - m)^2 + sigma^2) }.

    Parameters must satisfy: b > 0, sigma > 0, |rho| < 1 for sane shapes.
    """
    x = k - m
    return a + b * (rho * x + np.sqrt(np.square(x) + sigma * sigma))


def _price_weights(
    k: np.ndarray,
    w: np.ndarray,
    *,
    F: Optional[float],
    T: Optional[float],
    r: float = 0.0,
) -> np.ndarray:
    """
    Compute weights that approximate *price-space* LS while fitting
    in total-variance space.

    Heuristic:
      price residual ≈ vega * Δsigma,
      and Δsigma = (d sigma / d w) * Δw  with  d sigma / d w = 1/(2 T sigma).
      => price residual ≈ vega * Δw / (2 T sigma)

    So we weight the w-residual by  vega / (2 T sigma).
    If F or T is None, or w has very small/negative values, we fall back
    to unit weights.

    Black-76 ingredients:
      K = F * exp(k),
      Df = exp(-r T),
      sigma = sqrt(w / T),
      d1 = [ln(F/K) + 0.5 sigma^2 T] / (sigma sqrt(T)) = (-k + 0.5 w)/sqrt(w),
      vega = Df * F * sqrt(T) * φ(d1).
    """
    if F is None or T is None or T <= 0.0:
        return np.ones_like(k, dtype=float)

    w_pos = np.maximum(w, 1e-12)
    sigma = np.sqrt(w_pos / T)
    rootw = np.sqrt(w_pos)
    # Avoid division by zero for tiny w
    rootw = np.maximum(rootw, 1e-9)

    d1 = (-k + 0.5 * w_pos) / rootw
    vega = np.exp(-r * T) * F * np.sqrt(T) * _phi(d1)

    denom = 2.0 * T * sigma
    denom = np.where(denom <= 1e-12, 1e-12, denom)

    wts = vega / denom
    # Normalize weights to have median ~ 1 for numerical stability.
    med = np.median(wts[wts > 0])
    if np.isfinite(med) and med > 0.0:
        wts = wts / med
    else:
        wts = np.ones_like(k, dtype=float)
    return wts


# ------------------------------ Calibration ---------------------------- #


def _initial_bounds_from_data(
    k: np.ndarray,
    w: np.ndarray,
) -> Tuple[Tuple[float, float], ...]:
    """
    Construct conservative bounds using only data ranges.

    We pick:
      a     in [1e-8, max(w)]
      b     in [1e-6, 10.0]             (slope scale)
      rho   in [-0.999, 0.999]          (skew)
      m     in [k_min - 1.0, k_max + 1.0] (horizontal shift)
      sigma in [1e-6, 5.0]              (wing curvature scale)
    """
    kmin, kmax = float(np.min(k)), float(np.max(k))
    wmax = float(np.max(np.maximum(w, 1e-12)))
    bounds = (
        (1e-8, max(1e-6, wmax)),  # a
        (1e-6, 10.0),  # b
        (-0.999, 0.999),  # rho
        (kmin - 1.0, kmax + 1.0),  # m
        (1e-6, 5.0),  # sigma
    )
    return bounds


def _seed_grid_from_data(
    k: np.ndarray,
    w: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    grid_size: int = 5,
) -> Iterable[Tuple[float, float, float, float, float]]:
    """
    Generate a small set of diverse seeds informed by data.

    Heuristics:
      - a: start near min(w) and ~median(w)
      - b: small to moderate slopes in [0.1, 2.0]
      - rho: {-0.75, -0.25, 0.0, 0.25, 0.75} truncated to bounds
      - m: around median(k) ± a few quantiles
      - sigma: {0.05, 0.15, 0.3, 0.6} clipped to bounds
    """
    k_med = float(np.median(k))
    a_low = float(np.percentile(w, 10))
    a_med = float(np.percentile(w, 50))
    a_hi = float(np.percentile(w, 80))

    a_vals = [a_low, a_med, a_hi]
    b_vals = [0.1, 0.3, 0.7, 1.5]
    rho_vals = [-0.75, -0.25, 0.0, 0.25, 0.75]
    m_vals = [k_med - 0.5, k_med, k_med + 0.5]
    sig_vals = [0.05, 0.15, 0.3, 0.6]

    # Clip to bounds to avoid invalid seeds
    (a_lo, a_hi_b), (b_lo, b_hi), (r_lo, r_hi), (m_lo, m_hi), (s_lo, s_hi) = (
        bounds
    )
    a_vals = [float(np.clip(x, a_lo, a_hi_b)) for x in a_vals]
    b_vals = [float(np.clip(x, b_lo, b_hi)) for x in b_vals]
    rho_vals = [float(np.clip(x, r_lo, r_hi)) for x in rho_vals]
    m_vals = [float(np.clip(x, m_lo, m_hi)) for x in m_vals]
    sig_vals = [float(np.clip(x, s_lo, s_hi)) for x in sig_vals]

    seeds = []
    for a0 in a_vals:
        for b0 in b_vals:
            for r0 in rho_vals:
                for m0 in m_vals:
                    for s0 in sig_vals:
                        seeds.append((a0, b0, r0, m0, s0))
                        if len(seeds) >= max(grid_size, 1) * 25:
                            # Keep the grid from exploding; still diverse
                            return seeds
    return seeds


def calibrate_svi_from_quotes(
    k: ArrayLike,
    w: Optional[ArrayLike] = None,
    *,
    iv: Optional[ArrayLike] = None,
    T: Optional[float] = None,
    weights: Optional[ArrayLike] = None,
    use_price_weighting: bool = True,
    F: Optional[float] = None,
    r: float = 0.0,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    seed: Optional[Tuple[float, float, float, float, float]] = None,
    grid_seeds: bool = True,
    grid_size: int = 5,
    reg_lambda: float = 0.0,
) -> SVIFit:
    """
    Fit SVI to a single-expiry smile.

    Provide either:
      - w (total variance) directly, or
      - iv (implied vol) and T, so we compute w = iv^2 * T.

    Parameters
    ----------
    k : array-like
        Log-moneyness values ln(K/F). (1D)
    w : array-like, optional
        Total variance at the same points. (1D)
    iv : array-like, optional
        Implied vol at the same points. Used if w is None.
    T : float, optional
        Year fraction to expiry. Required if iv is provided and w is None.
    weights : array-like, optional
        Custom weights per point (1D). Overrides price-weighting if given.
    use_price_weighting : bool
        If True and weights is None, compute vega-based weights using F, T.
    F : float, optional
        Forward level for the expiry (used for price-weighting).
    r : float
        Continuously-compounded rate (only for vega/discount).
    bounds : sequence of (lo, hi), optional
        Bounds for (a,b,rho,m,sigma). If None, derived from data.
    seed : tuple[5], optional
        Initial guess (a,b,rho,m,sigma). If None, we use a small seed grid.
    grid_seeds : bool
        Try a handful of diverse seeds (robust) before local refine.
    grid_size : int
        Controls the size of the seed grid (roughly).
    reg_lambda : float
        Optional L2 regularization strength applied to parameters
        (weak shrinkage toward 0; helps pathological wings).

    Returns
    -------
    SVIFit
        Fitted parameters and diagnostics.
    """
    k = _ensure_1d(k, "k")

    if w is None:
        if iv is None or T is None or T <= 0.0:
            raise ValueError("Provide w, or (iv and positive T).")
        iv = _ensure_1d(iv, "iv")
        if iv.size != k.size:
            raise ValueError("iv and k must have the same length.")
        w = np.square(iv) * float(T)
    else:
        w = _ensure_1d(w, "w")
        if w.size != k.size:
            raise ValueError("w and k must have the same length.")

    # Filter non-finite and non-positive total variance entries
    k, w = _filter_finite(k, w)
    w = np.where(w > 0.0, w, np.nan)
    m = np.isfinite(w)
    k, w = k[m], w[m]
    n_used = int(k.size)
    if n_used < 5:
        raise ValueError("Need at least 5 valid points to fit SVI.")

    # Weights: custom > price-weighting > ones
    if weights is not None:
        wt = _ensure_1d(weights, "weights")
        if wt.size != n_used:
            raise ValueError("weights must match k/w length after filtering.")
        wt = np.where(np.isfinite(wt) & (wt > 0), wt, 1.0)
    elif use_price_weighting:
        wt = _price_weights(k, w, F=F, T=T, r=r)
    else:
        wt = np.ones_like(k, dtype=float)

    # Bounds and seeds
    if bounds is None:
        bounds = _initial_bounds_from_data(k, w)

    def objective(theta: np.ndarray) -> float:
        """
        Weighted L2 loss in total-variance space with optional L2-regularization.
        """
        a, b, rho, m0, sig = theta
        # Enforce soft validity (bounds handled by optimizer but guard anyway)
        if b <= 0 or sig <= 0 or not (-0.999 < rho < 0.999):
            return 1e12
        w_model = svi_total_variance(k, a, b, rho, m0, sig)
        # Protect from NaNs; huge penalty if exploded
        if not np.isfinite(w_model).all():
            return 1e12
        res = (w_model - w) * wt
        sse = float(np.dot(res, res))
        if reg_lambda > 0.0:
            sse += reg_lambda * float(np.dot(theta, theta))
        return sse

    # Try seed grid (robust) then refine best with L-BFGS-B
    tried: list[Tuple[float, np.ndarray]] = []

    def _local_refine(theta0: np.ndarray) -> Tuple[float, np.ndarray]:
        out = minimize(
            objective,
            x0=theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        val = float(out.fun)
        params = np.asarray(out.x, dtype=float)
        return val, params

    # Candidate seeds
    cand_seeds: list[Tuple[float, float, float, float, float]] = []
    if seed is not None:
        cand_seeds.append(tuple(float(x) for x in seed))
    if grid_seeds or not cand_seeds:
        cand_seeds.extend(
            _seed_grid_from_data(k, w, bounds, grid_size=grid_size)
        )

    best_val = np.inf
    best_params = None

    for s in cand_seeds:
        val0 = objective(np.asarray(s, dtype=float))
        # quick reject absurd seeds
        if not np.isfinite(val0) or val0 > 1e16:
            continue
        val, params = _local_refine(np.asarray(s, dtype=float))
        tried.append((val, params))
        if val < best_val:
            best_val = val
            best_params = params

    if best_params is None:
        # As a last resort, fall back to mid-bounds
        mid = np.array([(lo + hi) / 2.0 for (lo, hi) in bounds], dtype=float)
        best_val, best_params = _local_refine(mid)
        note = "fallback: mid-bounds seed"
    else:
        note = "grid-seeded + L-BFGS-B"

    # RMSE in the weighted space (informational)
    rmse = float(np.sqrt(best_val / max(n_used, 1)))

    a, b, rho, m0, sig = (float(x) for x in best_params)
    # Hard clip rho slightly inside (-1, 1) for downstream stability.
    rho = float(np.clip(rho, -0.999, 0.999))
    fit = SVIFit(
        params=(a, b, rho, m0, sig),
        loss=rmse,
        n_used=n_used,
        method="L-BFGS-B",
        notes=note,
    )
    return fit
