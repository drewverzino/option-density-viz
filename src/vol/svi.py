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

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

# Set up module logger
logger = logging.getLogger(__name__)

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

    def __post_init__(self):
        """Log fit creation and validate parameters."""
        a, b, rho, m, sigma = self.params
        logger.debug(
            f"Created SVIFit: a={a:.6f}, b={b:.6f}, rho={rho:.6f}, m={m:.6f}, σ={sigma:.6f}"
        )
        logger.debug(
            f"Fit diagnostics: loss={self.loss:.6f}, n_used={self.n_used}, method={self.method}"
        )

        # Validate parameter constraints
        if b <= 0:
            logger.warning(f"SVI parameter b={b:.6f} ≤ 0, may cause issues")
        if sigma <= 0:
            logger.warning(
                f"SVI parameter σ={sigma:.6f} ≤ 0, may cause issues"
            )
        if abs(rho) >= 1:
            logger.warning(
                f"SVI parameter |ρ|={abs(rho):.6f} ≥ 1, may cause issues"
            )


# ------------------------------ Math Utils ------------------------------ #

SQRT2PI = np.sqrt(2.0 * np.pi)


def _phi(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal PDF φ(x)."""
    return np.exp(-0.5 * np.square(x)) / SQRT2PI


def _ensure_1d(a: ArrayLike, name: str) -> np.ndarray:
    """Convert to 1D float array and sanity-check finite-ness."""
    logger.debug(f"Converting {name} to 1D array")

    x = np.asarray(a, dtype=float).reshape(-1)
    if x.size == 0:
        logger.error(f"{name} array is empty")
        raise ValueError(f"{name} must be non-empty")

    n_finite = np.isfinite(x).sum()
    n_total = len(x)

    if n_finite == 0:
        logger.error(f"{name} has no finite values")
        raise ValueError(f"{name} has no finite values")

    if n_finite < n_total:
        logger.warning(f"{name}: only {n_finite}/{n_total} finite values")
    else:
        logger.debug(f"{name}: all {n_total} values finite")

    logger.debug(
        f"{name} range: [{x[np.isfinite(x)].min():.6f}, {x[np.isfinite(x)].max():.6f}]"
    )
    return x


def _filter_finite(*arrs: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Keep positions where *all* arrays are finite."""
    logger.debug(f"Filtering finite values across {len(arrs)} arrays")

    m = np.logical_and.reduce([np.isfinite(a) for a in arrs])
    n_kept = m.sum()
    n_total = len(m)

    if n_kept < n_total:
        logger.info(
            f"Filtering finite values: kept {n_kept}/{n_total} ({n_kept/n_total:.1%})"
        )
    else:
        logger.debug(f"All {n_total} values finite, no filtering needed")

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
    logger.debug(
        f"SVI evaluation: a={a:.6f}, b={b:.6f}, rho={rho:.6f}, "
        f"m={m:.6f}, sigma={sigma:.6f} on {len(k)} points"
    )

    # Validate parameters with more detailed logging
    param_issues = []
    if b <= 0:
        param_issues.append(f"b={b:.6f} ≤ 0")
    if sigma <= 0:
        param_issues.append(f"σ={sigma:.6f} ≤ 0")
    if abs(rho) >= 1:
        param_issues.append(f"|ρ|={abs(rho):.6f} ≥ 1")

    if param_issues:
        logger.warning(f"Invalid SVI parameters: {', '.join(param_issues)}")

    x = k - m
    sqrt_term = np.sqrt(np.square(x) + sigma * sigma)
    result = a + b * (rho * x + sqrt_term)

    # Detailed result validation
    n_negative = (result < 0).sum()
    n_nonfinite = (~np.isfinite(result)).sum()

    if n_negative > 0:
        logger.warning(
            f"SVI produced {n_negative}/{len(result)} negative total variance values"
        )
        logger.debug(
            f"Negative variance range: [{result[result < 0].min():.6f}, {result[result < 0].max():.6f}]"
        )

    if n_nonfinite > 0:
        logger.error(
            f"SVI produced {n_nonfinite}/{len(result)} non-finite values"
        )

    if n_negative == 0 and n_nonfinite == 0:
        logger.debug(
            f"SVI evaluation successful: range [{result.min():.6f}, {result.max():.6f}]"
        )

    return result


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
      price residual ≈ vega * Δσ,
      and Δσ = (d σ / d w) * Δw  with  d σ / d w = 1/(2 T σ).
      => price residual ≈ vega * Δw / (2 T σ)

    So we weight the w-residual by  vega / (2 T σ).
    If F or T is None, or w has very small/negative values, we fall back
    to unit weights.

    Black-76 ingredients:
      K = F * exp(k),
      Df = exp(-r T),
      σ = sqrt(w / T),
      d1 = [ln(F/K) + 0.5 σ^2 T] / (σ sqrt(T)) = (-k + 0.5 w)/sqrt(w),
      vega = Df * F * sqrt(T) * φ(d1).
    """
    logger.debug(f"Computing price weights: F={F}, T={T}, r={r:.6f}")
    logger.debug(
        f"Input: {len(k)} points, k ∈ [{k.min():.3f}, {k.max():.3f}], w ∈ [{w.min():.6f}, {w.max():.6f}]"
    )

    if F is None or T is None or T <= 0.0:
        reason = []
        if F is None:
            reason.append("F=None")
        if T is None:
            reason.append("T=None")
        if T is not None and T <= 0.0:
            reason.append(f"T={T}≤0")

        logger.info(f"Using unit weights: {', '.join(reason)}")
        return np.ones_like(k, dtype=float)

    # Protect against degenerate total variance
    w_pos = np.maximum(w, 1e-12)
    n_floored = (w != w_pos).sum()
    if n_floored > 0:
        logger.debug(
            f"Floored {n_floored}/{len(w)} total variance values to avoid sqrt issues"
        )

    sigma = np.sqrt(w_pos / T)
    rootw = np.sqrt(w_pos)
    rootw = np.maximum(rootw, 1e-9)  # Avoid division by zero

    # Black-76 d1 and vega calculation
    d1 = (-k + 0.5 * w_pos) / rootw
    Df = np.exp(-r * T)
    vega = Df * F * np.sqrt(T) * _phi(d1)

    # Weight formula
    denom = 2.0 * T * sigma
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    wts = vega / denom

    # Check for issues
    n_nonfinite = (~np.isfinite(wts)).sum()
    n_nonpositive = (wts <= 0).sum()

    if n_nonfinite > 0:
        logger.warning(
            f"Price weights: {n_nonfinite}/{len(wts)} non-finite values"
        )
        wts = np.where(np.isfinite(wts), wts, 1.0)

    if n_nonpositive > 0:
        logger.warning(
            f"Price weights: {n_nonpositive}/{len(wts)} non-positive values"
        )
        wts = np.maximum(wts, 1e-6)

    # Normalize weights to have median ~ 1 for numerical stability
    positive_weights = wts[wts > 0]
    if len(positive_weights) > 0:
        med = np.median(positive_weights)
        if np.isfinite(med) and med > 0.0:
            wts = wts / med
            logger.debug(f"Price weights normalized by median {med:.6f}")
            logger.debug(
                f"Final weight range: [{wts.min():.6f}, {wts.max():.6f}]"
            )
        else:
            logger.warning(
                "Price weight normalization failed: non-finite/zero median"
            )
            wts = np.ones_like(k, dtype=float)
    else:
        logger.warning("No positive weights found, using unit weights")
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
    logger.debug("Constructing SVI parameter bounds from data")

    kmin, kmax = float(np.min(k)), float(np.max(k))
    wmin, wmax = float(np.min(w)), float(np.max(w))
    wmax_safe = max(1e-6, wmax)

    bounds = (
        (1e-8, wmax_safe),  # a
        (1e-6, 10.0),  # b
        (-0.999, 0.999),  # rho
        (kmin - 1.0, kmax + 1.0),  # m
        (1e-6, 5.0),  # sigma
    )

    logger.info(
        f"Data-driven bounds from k∈[{kmin:.3f}, {kmax:.3f}], w∈[{wmin:.6f}, {wmax:.6f}]:"
    )
    logger.info(f"  a ∈ [{bounds[0][0]:.2e}, {bounds[0][1]:.6f}]")
    logger.info(f"  b ∈ [{bounds[1][0]:.2e}, {bounds[1][1]:.1f}]")
    logger.info(f"  ρ ∈ [{bounds[2][0]:.3f}, {bounds[2][1]:.3f}]")
    logger.info(f"  m ∈ [{bounds[3][0]:.3f}, {bounds[3][1]:.3f}]")
    logger.info(f"  σ ∈ [{bounds[4][0]:.2e}, {bounds[4][1]:.1f}]")

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
    logger.debug(f"Generating seed grid with target size ~{grid_size}")

    k_med = float(np.median(k))
    k_q25 = float(np.percentile(k, 25))
    k_q75 = float(np.percentile(k, 75))

    w_10 = float(np.percentile(w, 10))
    w_50 = float(np.percentile(w, 50))
    w_80 = float(np.percentile(w, 80))

    logger.debug(f"Data-informed seed values:")
    logger.debug(
        f"  k percentiles: 25%={k_q25:.3f}, 50%={k_med:.3f}, 75%={k_q75:.3f}"
    )
    logger.debug(
        f"  w percentiles: 10%={w_10:.6f}, 50%={w_50:.6f}, 80%={w_80:.6f}"
    )

    # Base parameter grids
    a_vals = [w_10, w_50, w_80]
    b_vals = [0.1, 0.3, 0.7, 1.5]
    rho_vals = [-0.75, -0.25, 0.0, 0.25, 0.75]
    m_vals = [k_q25, k_med, k_q75]
    sig_vals = [0.05, 0.15, 0.3, 0.6]

    # Clip to bounds to avoid invalid seeds
    (a_lo, a_hi), (b_lo, b_hi), (r_lo, r_hi), (m_lo, m_hi), (s_lo, s_hi) = (
        bounds
    )

    a_vals = [float(np.clip(x, a_lo, a_hi)) for x in a_vals]
    b_vals = [float(np.clip(x, b_lo, b_hi)) for x in b_vals]
    rho_vals = [float(np.clip(x, r_lo, r_hi)) for x in rho_vals]
    m_vals = [float(np.clip(x, m_lo, m_hi)) for x in m_vals]
    sig_vals = [float(np.clip(x, s_lo, s_hi)) for x in sig_vals]

    logger.debug(f"Clipped seed values to bounds:")
    logger.debug(f"  a: {[f'{x:.6f}' for x in a_vals]}")
    logger.debug(f"  b: {[f'{x:.2f}' for x in b_vals]}")
    logger.debug(f"  ρ: {[f'{x:.3f}' for x in rho_vals]}")
    logger.debug(f"  m: {[f'{x:.3f}' for x in m_vals]}")
    logger.debug(f"  σ: {[f'{x:.3f}' for x in sig_vals]}")

    # Generate Cartesian product up to reasonable limit
    seeds = []
    max_seeds = max(grid_size, 1) * 25  # Reasonable upper bound

    for a0 in a_vals:
        for b0 in b_vals:
            for r0 in rho_vals:
                for m0 in m_vals:
                    for s0 in sig_vals:
                        seeds.append((a0, b0, r0, m0, s0))
                        if len(seeds) >= max_seeds:
                            logger.debug(
                                f"Reached seed limit: generated {len(seeds)} seeds"
                            )
                            return seeds

    logger.debug(f"Generated {len(seeds)} total seeds from parameter grid")
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
    logger.info("=" * 60)
    logger.info("STARTING SVI CALIBRATION")
    logger.info("=" * 60)

    logger.info(f"Input parameters: T={T}, F={F}, r={r:.6f}")
    logger.info(
        f"Options: use_price_weighting={use_price_weighting}, grid_seeds={grid_seeds}, grid_size={grid_size}"
    )
    logger.info(f"Regularization: λ={reg_lambda}")

    # Input validation and preparation
    k = _ensure_1d(k, "k")
    n_input = len(k)
    logger.info(
        f"Input log-moneyness: {n_input} points, range [{k.min():.3f}, {k.max():.3f}]"
    )

    if w is None:
        if iv is None or T is None or T <= 0.0:
            logger.error(
                "Missing required inputs for total variance computation"
            )
            logger.error(
                "Need either: (1) w directly, or (2) iv and positive T"
            )
            raise ValueError("Provide w, or (iv and positive T).")

        logger.info("Computing total variance from implied vol")
        iv = _ensure_1d(iv, "iv")
        if iv.size != k.size:
            logger.error(f"Size mismatch: iv.size={iv.size}, k.size={k.size}")
            raise ValueError("iv and k must have the same length.")

        w = np.square(iv) * float(T)
        logger.info(
            f"Computed w = iv² × T: range [{w.min():.6f}, {w.max():.6f}]"
        )
    else:
        logger.info("Using provided total variance")
        w = _ensure_1d(w, "w")
        if w.size != k.size:
            logger.error(f"Size mismatch: w.size={w.size}, k.size={k.size}")
            raise ValueError("w and k must have the same length.")
        logger.info(f"Total variance range: [{w.min():.6f}, {w.max():.6f}]")

    # Filter non-finite and non-positive total variance entries
    logger.debug("Filtering data for finite and positive values")
    k, w = _filter_finite(k, w)

    pre_positive_filter = len(w)
    w = np.where(w > 0.0, w, np.nan)
    m = np.isfinite(w)
    k, w = k[m], w[m]
    n_used = int(k.size)

    post_positive_filter = len(w)
    if post_positive_filter < pre_positive_filter:
        logger.warning(
            f"Removed {pre_positive_filter - post_positive_filter} non-positive total variance entries"
        )

    logger.info(
        f"Final dataset: {n_used} valid points from {n_input} input ({n_used/n_input:.1%})"
    )
    logger.info(
        f"Final ranges: k ∈ [{k.min():.3f}, {k.max():.3f}], w ∈ [{w.min():.6f}, {w.max():.6f}]"
    )

    if n_used < 5:
        logger.error(
            f"Insufficient data: only {n_used} valid points (need ≥5)"
        )
        raise ValueError("Need at least 5 valid points to fit SVI.")

    # Handle weights
    logger.info("Setting up weights")
    if weights is not None:
        logger.info("Using custom weights")
        wt = _ensure_1d(weights, "weights")
        if wt.size != n_used:
            logger.error(
                f"Weight size mismatch after filtering: {wt.size} vs {n_used}"
            )
            raise ValueError("weights must match k/w length after filtering.")
        wt = np.where(np.isfinite(wt) & (wt > 0), wt, 1.0)
        logger.debug(f"Custom weights range: [{wt.min():.6f}, {wt.max():.6f}]")
    elif use_price_weighting:
        logger.info("Computing price-based weights")
        wt = _price_weights(k, w, F=F, T=T, r=r)
    else:
        logger.info("Using unit weights")
        wt = np.ones_like(k, dtype=float)

    # Set up bounds and objective function
    if bounds is None:
        logger.info("Deriving parameter bounds from data")
        bounds = _initial_bounds_from_data(k, w)
    else:
        logger.info("Using provided parameter bounds")
        for i, (param, (lo, hi)) in enumerate(
            zip(['a', 'b', 'ρ', 'm', 'σ'], bounds)
        ):
            logger.debug(f"  {param} ∈ [{lo:.6f}, {hi:.6f}]")

    def objective(theta: np.ndarray) -> float:
        """
        Weighted L2 loss in total-variance space with optional L2-regularization.
        """
        a, b, rho, m0, sig = theta

        # Enforce soft validity (bounds handled by optimizer but guard anyway)
        if b <= 0 or sig <= 0 or not (-0.999 < rho < 0.999):
            logger.debug(
                f"Objective: invalid params a={a:.4f}, b={b:.4f}, ρ={rho:.4f}, m={m0:.4f}, σ={sig:.4f}"
            )
            return 1e12

        try:
            w_model = svi_total_variance(k, a, b, rho, m0, sig)

            # Protect from NaNs; huge penalty if exploded
            if not np.isfinite(w_model).all():
                logger.debug(f"Objective: non-finite SVI output")
                return 1e12

            res = (w_model - w) * wt
            sse = float(np.dot(res, res))

            if reg_lambda > 0.0:
                reg_term = reg_lambda * float(np.dot(theta, theta))
                sse += reg_term
                logger.debug(
                    f"Objective: SSE={sse-reg_term:.6f}, reg={reg_term:.6f}, total={sse:.6f}"
                )

            return sse
        except Exception as e:
            logger.debug(f"Objective evaluation failed: {e}")
            return 1e12

    # Set up optimization: seed generation and refinement
    logger.info("Setting up optimization")

    def _local_refine(theta0: np.ndarray) -> Tuple[float, np.ndarray]:
        """Run L-BFGS-B from given starting point."""
        logger.debug(f"Local refinement from: {[f'{x:.6f}' for x in theta0]}")

        out = minimize(
            objective,
            x0=theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        val = float(out.fun)
        params = np.asarray(out.x, dtype=float)

        logger.debug(
            f"Local refinement result: loss={val:.8f}, success={out.success}"
        )
        logger.debug(f"Final params: {[f'{x:.6f}' for x in params]}")

        return val, params

    # Generate candidate seeds
    cand_seeds: list[Tuple[float, float, float, float, float]] = []

    if seed is not None:
        logger.info(f"Using provided seed: {[f'{x:.6f}' for x in seed]}")
        cand_seeds.append(tuple(float(x) for x in seed))

    if grid_seeds or not cand_seeds:
        logger.info("Generating grid-based seeds")
        grid_generated = list(
            _seed_grid_from_data(k, w, bounds, grid_size=grid_size)
        )
        cand_seeds.extend(grid_generated)
        logger.info(f"Generated {len(grid_generated)} grid seeds")

    logger.info(f"Total candidate seeds: {len(cand_seeds)}")

    # Optimization: try all seeds and refine best
    best_val = np.inf
    best_params = None
    valid_seeds = 0
    seed_results = []

    logger.info("Starting multi-seed optimization")

    for i, s in enumerate(cand_seeds):
        logger.debug(
            f"Trying seed {i+1}/{len(cand_seeds)}: {[f'{x:.4f}' for x in s]}"
        )

        # Quick evaluation to reject obviously bad seeds
        val0 = objective(np.asarray(s, dtype=float))
        if not np.isfinite(val0) or val0 > 1e16:
            logger.debug(f"Seed {i+1} rejected: initial objective {val0}")
            continue

        valid_seeds += 1
        logger.debug(f"Seed {i+1} accepted: initial objective {val0:.6f}")

        # Local refinement
        try:
            val, params = _local_refine(np.asarray(s, dtype=float))
            seed_results.append((i, val, params))

            if val < best_val:
                best_val = val
                best_params = params
                logger.info(f"New best result from seed {i+1}: loss={val:.8f}")
                logger.debug(f"Best params: {[f'{x:.6f}' for x in params]}")

        except Exception as e:
            logger.warning(f"Refinement failed for seed {i+1}: {e}")
            continue

    # Results summary
    logger.info("Optimization complete")
    logger.info(f"Seeds evaluated: {valid_seeds}/{len(cand_seeds)} valid")
    logger.info(f"Successful optimizations: {len(seed_results)}")

    if best_params is None:
        logger.warning("All seeds failed, attempting fallback mid-bounds seed")
        # As a last resort, fall back to mid-bounds
        mid = np.array([(lo + hi) / 2.0 for (lo, hi) in bounds], dtype=float)
        logger.info(f"Fallback seed: {[f'{x:.6f}' for x in mid]}")

        try:
            best_val, best_params = _local_refine(mid)
            note = "fallback: mid-bounds seed"
            logger.warning(f"Fallback successful: loss={best_val:.8f}")
        except Exception as e:
            logger.error(f"Fallback optimization also failed: {e}")
            raise RuntimeError("All optimization attempts failed")
    else:
        note = f"grid-seeded + L-BFGS-B ({len(seed_results)} successful seeds)"

    # Final parameter validation and RMSE computation
    rmse = float(np.sqrt(best_val / max(n_used, 1)))
    a, b, rho, m0, sig = (float(x) for x in best_params)

    # Hard clip rho slightly inside (-1, 1) for downstream stability
    rho_original = rho
    rho = float(np.clip(rho, -0.999, 0.999))
    if rho != rho_original:
        logger.debug(f"Clipped ρ from {rho_original:.6f} to {rho:.6f}")

    # Final parameter validation
    param_warnings = []
    if b <= 0:
        param_warnings.append(f"b={b:.6f} ≤ 0")
    if sig <= 0:
        param_warnings.append(f"σ={sig:.6f} ≤ 0")
    if abs(rho) >= 0.999:
        param_warnings.append(f"|ρ|={abs(rho):.6f} ≥ 0.999")

    if param_warnings:
        logger.warning(f"Final parameter issues: {', '.join(param_warnings)}")

    # Success logging
    logger.info("=" * 60)
    logger.info("SVI CALIBRATION COMPLETE")
    logger.info("=" * 60)
    logger.info(
        f"Final parameters: a={a:.6f}, b={b:.6f}, ρ={rho:.6f}, m={m0:.6f}, σ={sig:.6f}"
    )
    logger.info(f"Final loss (RMSE): {rmse:.8f}")
    logger.info(f"Points used: {n_used}")
    logger.info(f"Method: {note}")

    # Test final SVI function
    try:
        w_final = svi_total_variance(k, a, b, rho, m0, sig)
        residuals = w_final - w
        max_abs_residual = np.abs(residuals).max()
        logger.info(
            f"Final fit quality: max |residual| = {max_abs_residual:.8f}"
        )

        negative_w = (w_final < 0).sum()
        if negative_w > 0:
            logger.warning(
                f"Final SVI produces {negative_w} negative total variance values"
            )
    except Exception as e:
        logger.error(f"Final SVI evaluation failed: {e}")

    fit = SVIFit(
        params=(a, b, rho, m0, sig),
        loss=rmse,
        n_used=n_used,
        method="L-BFGS-B",
        notes=note,
    )

    logger.info("SVI fit object created successfully")
    return fit
