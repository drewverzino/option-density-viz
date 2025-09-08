"""
SVI surface across maturities (time dimension) with price→IV bootstrapping.

What this provides
------------------
- Fit per-expiry SVI using `calibrate_svi_from_quotes` from `svi.py`.
- Build a light `SVISurface` container to evaluate w(k,T) & iv(K,T).
- Calendar no-arb spot checks in total variance space.
- Optional smoothing of parameters across T.
- When an IV column is missing, solve **implied vols from call prices**
  via **Black-76** on forwards (robust bisection with sanity clamps).

Alignment with svi.py
---------------------
- We import: SVIFit, calibrate_svi_from_quotes, svi_total_variance.
- We always unpack parameters from `fit.params` (a,b,rho,m,sigma).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

# Optional SciPy for smoothing; code works without it.
try:
    from scipy.interpolate import UnivariateSpline  # type: ignore

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

from .svi import SVIFit, calibrate_svi_from_quotes, svi_total_variance

# Set up module logger
logger = logging.getLogger("vol.surface")

# --------------------------------------------------------------------- #
# Black-76 helpers (for price→IV bootstrapping)
# --------------------------------------------------------------------- #

SQRT2PI = np.sqrt(2.0 * np.pi)


def _phi(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal pdf φ(x)."""
    return np.exp(-0.5 * np.square(x)) / SQRT2PI


def _Phi(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal cdf Φ(x) via erf."""
    # Use numpy.erf if present; fall back to math.erf scalar-wise.
    try:
        from numpy import erf  # type: ignore

        return 0.5 * (1.0 + erf(np.asarray(x) / np.sqrt(2.0)))
    except Exception:  # pragma: no cover
        from math import erf as m_erf

        x_arr = np.asarray(x, dtype=float)
        return 0.5 * (
            1.0 + np.vectorize(lambda t: m_erf(t / np.sqrt(2.0)))(x_arr)
        )


def _black76_call_price(
    F: float, K: float, T: float, sigma: float, Df: float
) -> float:
    """
    Black-76 call on forward:
        C = Df * [ F Φ(d1) - K Φ(d2) ],
      d1 = [ln(F/K) + 0.5 σ^2 T] / (σ sqrt(T)),
      d2 = d1 - σ sqrt(T).
    """
    logger.debug(
        f"Black-76 pricing: F={F:.4f}, K={K:.4f}, T={T:.4f}, σ={sigma:.4f}, Df={Df:.6f}"
    )

    if sigma <= 0.0 or T <= 0.0:
        # Limit: price is just discounted intrinsic
        intrinsic = Df * max(F - K, 0.0)
        logger.debug(
            f"Zero vol/time case: returning intrinsic value {intrinsic:.6f}"
        )
        return float(intrinsic)

    sqrtT = np.sqrt(T)
    s = max(sigma, 1e-12)
    lnFK = np.log(max(F, 1e-300) / max(K, 1e-300))
    d1 = (lnFK + 0.5 * s * s * T) / (s * sqrtT)
    d2 = d1 - s * sqrtT

    price = float(Df * (F * _Phi(d1) - K * _Phi(d2)))
    logger.debug(
        f"Black-76 result: d1={d1:.4f}, d2={d2:.4f}, price={price:.6f}"
    )

    return price


def _black76_vega(
    F: float, K: float, T: float, sigma: float, Df: float
) -> float:
    """Black-76 vega = Df * F * φ(d1) * sqrt(T)."""
    if sigma <= 0.0 or T <= 0.0:
        logger.debug("Zero vega due to zero vol or time")
        return 0.0

    sqrtT = np.sqrt(T)
    s = max(sigma, 1e-12)
    lnFK = np.log(max(F, 1e-300) / max(K, 1e-300))
    d1 = (lnFK + 0.5 * s * s * T) / (s * sqrtT)
    vega = float(Df * F * _phi(d1) * sqrtT)

    logger.debug(f"Black-76 vega: {vega:.6f}")
    return vega


def _implied_vol_black76_call(
    C: float,
    F: float,
    K: float,
    T: float,
    Df: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Robust scalar implied vol (Black-76) via bracketed bisection
    with sanity clamps:

      intrinsic_low = Df * max(F - K, 0)
      upper_bound   = Df * F    (as σ→∞, call → Df * F)

    If C is outside [intrinsic_low, upper_bound], we clamp it into
    that range (minus a tiny epsilon on the upper side) and proceed.
    """
    logger.debug(f"Solving IV: C={C:.6f}, F={F:.4f}, K={K:.4f}, T={T:.4f}")

    # Sanity clamps on price
    intrinsic = Df * max(F - K, 0.0)
    upper = Df * F

    if not np.isfinite(C):
        logger.warning(f"Non-finite call price {C}, returning zero vol")
        return 0.0

    C_original = C
    C = float(np.clip(C, intrinsic, max(upper - 1e-12, intrinsic)))

    if C != C_original:
        logger.debug(
            f"Clamped price from {C_original:.6f} to {C:.6f} (bounds: [{intrinsic:.6f}, {upper:.6f}])"
        )

    # Quick exits
    if C <= intrinsic + 1e-14:
        logger.debug("Price at intrinsic, returning zero vol")
        return 0.0

    # Bracket: low ~ near-zero vol, high grow until price >= C
    lo, hi = 1e-8, 1.0
    price_hi = _black76_call_price(F, K, T, hi, Df)
    iterations = 0

    while price_hi < C and hi < 10.0:  # 10 is a generous cap
        hi *= 2.0
        price_hi = _black76_call_price(F, K, T, hi, Df)
        iterations += 1

    logger.debug(
        f"Initial bracketing: lo={lo:.6f}, hi={hi:.6f} after {iterations} iterations"
    )

    # Bisection
    for i in range(max_iter):
        mid = 0.5 * (lo + hi)
        price_mid = _black76_call_price(F, K, T, mid, Df)

        if abs(price_mid - C) < tol:
            logger.debug(f"IV converged in {i+1} iterations: σ={mid:.6f}")
            return float(mid)

        if price_mid < C:
            lo = mid
        else:
            hi = mid

    final_vol = float(0.5 * (lo + hi))
    logger.debug(
        f"IV bisection completed {max_iter} iterations: σ={final_vol:.6f}"
    )
    return final_vol


def _solve_iv_from_calls(
    df_calls: pd.DataFrame,
    F: float,
    T: float,
    r: float,
    price_col: str,
    strike_col: str,
) -> np.ndarray:
    """
    Vectorized implied vol solver for call quotes in a DataFrame.
    Returns an array of IVs aligned with df_calls rows.
    """
    logger.info(
        f"Solving IVs from {len(df_calls)} call prices: F={F:.4f}, T={T:.4f}, r={r:.4f}"
    )

    if df_calls.empty:
        logger.warning("Empty DataFrame provided to IV solver")
        return np.array([])

    Df = float(np.exp(-r * T))
    K = df_calls[strike_col].to_numpy(float)
    C = df_calls[price_col].to_numpy(float)

    logger.debug(f"Strike range: [{K.min():.2f}, {K.max():.2f}]")
    logger.debug(f"Price range: [{C.min():.6f}, {C.max():.6f}]")

    out = np.empty_like(C, dtype=float)
    failed_count = 0

    for i in range(C.size):
        try:
            out[i] = _implied_vol_black76_call(
                C=float(C[i]), F=float(F), K=float(K[i]), T=float(T), Df=Df
            )
        except Exception as e:
            logger.warning(
                f"IV solve failed for point {i} (K={K[i]:.2f}, C={C[i]:.6f}): {e}"
            )
            out[i] = 0.0
            failed_count += 1

    if failed_count > 0:
        logger.warning(f"IV solving failed for {failed_count}/{C.size} points")

    valid_ivs = np.isfinite(out) & (out > 0)
    logger.info(
        f"Successfully solved {valid_ivs.sum()}/{C.size} IVs, range: [{out[valid_ivs].min():.4f}, {out[valid_ivs].max():.4f}]"
    )

    return out


# --------------------------------------------------------------------- #
# Data container for a stitched surface
# --------------------------------------------------------------------- #


@dataclass
class SVISurface:
    """
    Per-expiry SVI fits + metadata.

    fits : dict[expiry, SVIFit]       → params in fit.params (a,b,rho,m,sigma)
    T, F, r : dict[expiry, float]     → year-fraction, forward, rate
    asof_utc : optional timestamp     → provenance for display/logging
    """

    fits: Dict[datetime, SVIFit]
    T: Dict[datetime, float]
    F: Dict[datetime, float]
    r: Dict[datetime, float]
    asof_utc: Optional[datetime] = None

    def __post_init__(self):
        """Log surface creation and validate consistency."""
        logger.info(f"Created SVISurface with {len(self.fits)} expiries")

        if self.asof_utc:
            logger.info(f"Surface as-of time: {self.asof_utc}")

        # Validate consistency
        expiries = set(self.fits.keys())
        if expiries != set(self.T.keys()):
            logger.error("Mismatch between fits and T dictionaries")
        if expiries != set(self.F.keys()):
            logger.error("Mismatch between fits and F dictionaries")
        if expiries != set(self.r.keys()):
            logger.error("Mismatch between fits and r dictionaries")

        # Log summary statistics
        T_values = list(self.T.values())
        F_values = list(self.F.values())
        r_values = list(self.r.values())

        logger.info(
            f"Time to expiry range: [{min(T_values):.4f}, {max(T_values):.4f}] years"
        )
        logger.info(
            f"Forward price range: [{min(F_values):.2f}, {max(F_values):.2f}]"
        )
        logger.info(
            f"Interest rate range: [{min(r_values):.4f}, {max(r_values):.4f}]"
        )

    def expiries(self) -> List[datetime]:
        return sorted(self.fits.keys())

    def params_tuple(
        self, expiry: datetime
    ) -> Tuple[float, float, float, float, float]:
        if expiry not in self.fits:
            logger.error(f"Expiry {expiry} not found in surface")
            raise KeyError(f"Expiry {expiry} not in surface")

        a, b, rho, m, sigma = self.fits[expiry].params
        logger.debug(
            f"Retrieved params for {expiry}: a={a:.6f}, b={b:.6f}, rho={rho:.6f}, m={m:.6f}, σ={sigma:.6f}"
        )
        return float(a), float(b), float(rho), float(m), float(sigma)

    def w(self, k: np.ndarray | float, expiry: datetime) -> np.ndarray:
        """Total variance w(k,T) at given expiry."""
        logger.debug(
            f"Computing total variance for expiry {expiry}, k range: {np.array(k).min():.3f} to {np.array(k).max():.3f}"
        )

        a, b, rho, m, sigma = self.params_tuple(expiry)
        kk = np.asarray(k, dtype=float)
        w_result = svi_total_variance(kk, a, b, rho, m, sigma)

        # Check for negative variance
        negative_count = (w_result < 0).sum()
        if negative_count > 0:
            logger.warning(
                f"SVI produced {negative_count} negative total variance values for expiry {expiry}"
            )

        return w_result

    def iv(self, K: np.ndarray | float, expiry: datetime) -> np.ndarray:
        """Implied vol σ_imp(K,T) from total variance."""
        logger.debug(
            f"Computing IV for expiry {expiry}, K range: {np.array(K).min():.2f} to {np.array(K).max():.2f}"
        )

        F = float(self.F[expiry])
        T = float(self.T[expiry])
        K_arr = np.asarray(K, dtype=float)
        k = np.log(K_arr / F)
        wT = self.w(k, expiry)
        iv_result = np.sqrt(np.maximum(wT, 0.0) / max(T, 1e-12))

        logger.debug(
            f"IV computation: F={F:.2f}, T={T:.4f}, IV range: [{iv_result.min():.4f}, {iv_result.max():.4f}]"
        )
        return iv_result

    def calendar_violations(
        self, k_grid: np.ndarray, tol: float = 0.0
    ) -> Dict[str, object]:
        """
        Calendar check: for T2 > T1 require w(k,T2) >= w(k,T1) ∀k.
        Returns counts and per-adjacent-maturity fractions on k_grid.
        """
        logger.info(
            f"Checking calendar arbitrage on {len(k_grid)} points with tolerance {tol}"
        )

        exps = self.expiries()
        if len(exps) < 2:
            logger.info("Less than 2 expiries, no calendar checks possible")
            return {"count": 0, "n_checks": 0, "fraction": 0.0, "by_pair": []}

        k = np.asarray(k_grid, dtype=float)
        total = 0
        bad = 0
        by_pair: List[Tuple[float, float, float]] = []

        for e1, e2 in zip(exps[:-1], exps[1:]):
            logger.debug(f"Checking calendar pair: {e1} vs {e2}")

            w1 = self.w(k, e1)
            w2 = self.w(k, e2)
            vio = (w2 + tol) < w1
            n = int(k.size)
            c = int(np.count_nonzero(vio))
            total += n
            bad += c
            t1, t2 = float(self.T[e1]), float(self.T[e2])

            violation_fraction = (c / n) if n else 0.0
            by_pair.append((t1, t2, violation_fraction))

            if c > 0:
                logger.warning(
                    f"Calendar violations between T={t1:.4f} and T={t2:.4f}: {c}/{n} points ({violation_fraction:.1%})"
                )

        overall_fraction = (bad / total) if total else 0.0
        logger.info(
            f"Calendar check complete: {bad}/{total} violations ({overall_fraction:.1%})"
        )

        return {
            "count": bad,
            "n_checks": total,
            "fraction": overall_fraction,
            "by_pair": by_pair,
        }


# --------------------------------------------------------------------- #
# Prepare smile data (k, w, weights) from a per-expiry frame
# --------------------------------------------------------------------- #


def _prepare_smile_data_from_frame(
    df: pd.DataFrame,
    *,
    T: float,
    F: float,
    r: float,
    price_col: str = "mid",
    type_col: str = "type",
    strike_col: str = "strike",
    iv_col: str = "iv",
    use_flags: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Build (k, w, weights) from a *clean* per-expiry DataFrame.

    Strategy
    --------
    - Use **calls only** to avoid put IV convention ambiguity.
    - If `iv_col` exists and has finite values → use it.
    - Else, **solve implied vol** from call mid prices with Black-76.
    - Optionally filter out crossed/wide quotes if flags exist.

    Returns
    -------
    (k, w, weights)
      k       : ln(K / F)
      w       : total variance = iv^2 * T
      weights : optional array (e.g., 1/rel_spread); or None
    """
    logger.info(
        f"Preparing smile data: input {len(df)} rows, T={T:.4f}, F={F:.2f}"
    )

    d = df.copy()
    initial_count = len(d)

    # Optional hygiene flags from preprocess.midprice
    if use_flags:
        pre_flag_count = len(d)
        if "crossed" in d.columns:
            crossed_count = d["crossed"].astype(bool).sum()
            d = d[~d["crossed"].astype(bool)]
            logger.debug(f"Filtered {crossed_count} crossed quotes")
        if "wide" in d.columns:
            wide_count = d["wide"].astype(bool).sum()
            d = d[~d["wide"].astype(bool)]
            logger.debug(f"Filtered {wide_count} wide quotes")

        post_flag_count = len(d)
        if post_flag_count < pre_flag_count:
            logger.info(
                f"Quote filtering removed {pre_flag_count - post_flag_count} rows"
            )

    # Restrict to calls (C)
    pre_call_count = len(d)
    t = d[type_col].astype(str).str.upper().str.strip()
    d = d[t.str.startswith("C")]
    post_call_count = len(d)

    logger.info(f"Call filtering: {pre_call_count} → {post_call_count} rows")

    d = d.dropna(subset=[strike_col, price_col])
    final_count = len(d)

    if final_count < pre_call_count:
        logger.info(
            f"Dropped {pre_call_count - final_count} rows with missing strike/price data"
        )

    if d.empty:
        logger.warning("No valid call data remaining after filtering")
        return np.array([]), np.array([]), None

    # If IV column is present and useful, use it; otherwise solve IVs.
    iv = None
    if iv_col in d.columns:
        logger.debug(f"Checking IV column '{iv_col}'")
        iv_raw = d[iv_col].to_numpy(dtype=float)
        iv = np.where(np.isfinite(iv_raw) & (iv_raw > 0.0), iv_raw, np.nan)
        valid_iv_count = np.count_nonzero(np.isfinite(iv))

        logger.info(f"IV column has {valid_iv_count}/{len(iv)} valid values")

        if valid_iv_count < 5:
            logger.info(
                "Insufficient valid IVs in column, will solve from prices"
            )
            iv = None
        else:
            logger.info("Using IV column data")

    if iv is None:
        # Solve IVs from call prices (Black-76 on forwards)
        logger.info("Solving implied volatilities from call prices")
        iv = _solve_iv_from_calls(
            df_calls=d,
            F=float(F),
            T=float(T),
            r=float(r),
            price_col=price_col,
            strike_col=strike_col,
        )

    # Build w and k
    K = d[strike_col].to_numpy(dtype=float)
    k = np.log(K / float(F))
    w = (np.abs(iv) ** 2) * float(T)

    logger.debug(
        f"Computed data: k ∈ [{k.min():.3f}, {k.max():.3f}], w ∈ [{w.min():.6f}, {w.max():.6f}]"
    )

    # Optional weights: inverse relative spread if present
    weights = None
    if "rel_spread" in d.columns:
        logger.debug("Computing weights from relative spreads")
        rel_spreads = d["rel_spread"].to_numpy(dtype=float)
        wts = 1.0 / np.clip(rel_spreads, 1e-6, np.inf)

        if np.any(np.isfinite(wts)):
            weights = wts
            logger.info(
                f"Using spread-based weights: range [{wts.min():.2f}, {wts.max():.2f}]"
            )
        else:
            logger.warning("All spread-based weights non-finite, ignoring")

    # Cull any remaining non-finite entries
    pre_cull_count = len(k)
    mask = np.isfinite(k) & np.isfinite(w)
    if weights is not None:
        mask = mask & np.isfinite(weights)
        weights = weights[mask]

    k_final, w_final = k[mask], w[mask]
    post_cull_count = len(k_final)

    if post_cull_count < pre_cull_count:
        logger.info(
            f"Final filtering removed {pre_cull_count - post_cull_count} non-finite entries"
        )

    logger.info(
        f"Smile data preparation complete: {post_cull_count} final points from {initial_count} input rows"
    )

    return k_final, w_final, weights


# --------------------------------------------------------------------- #
# Fit per-expiry SVI and build a surface
# --------------------------------------------------------------------- #


def fit_surface_from_frames(
    frames_by_expiry: Mapping[datetime, pd.DataFrame],
    *,
    T_by_expiry: Mapping[datetime, float],
    F_by_expiry: Mapping[datetime, float],
    r_by_expiry: Mapping[datetime, float],
    price_col: str = "mid",
    type_col: str = "type",
    strike_col: str = "strike",
    iv_col: str = "iv",
    use_flags: bool = True,
) -> SVISurface:
    """
    Fit SVI per expiry from *clean* frames and return an SVISurface.

    For each expiry present in all mappings:
      1) Prepare (k, w, weights) using IV column or price→IV bootstrapping.
      2) Fit params with `calibrate_svi_from_quotes` (vega-weighted loss
         if F,T are provided and no custom weights).
      3) Skip expiries with too few valid points (< 7).
    """
    logger.info(
        f"Starting surface fitting from {len(frames_by_expiry)} expiry frames"
    )

    fits: Dict[datetime, SVIFit] = {}

    # Only expiries that exist in all inputs
    common = (
        set(frames_by_expiry)
        & set(T_by_expiry)
        & set(F_by_expiry)
        & set(r_by_expiry)
    )

    logger.info(f"Found {len(common)} expiries with complete data")

    if not common:
        logger.error("No overlapping expiries across inputs")
        raise ValueError("No overlapping expiries across inputs.")

    skipped_expiries = []
    failed_expiries = []

    for exp in sorted(common):
        logger.info(f"Processing expiry {exp}")

        try:
            df = frames_by_expiry[exp]
            T = float(T_by_expiry[exp])
            F = float(F_by_expiry[exp])
            r = float(r_by_expiry[exp])

            k, w, wts = _prepare_smile_data_from_frame(
                df,
                T=T,
                F=F,
                r=r,
                price_col=price_col,
                type_col=type_col,
                strike_col=strike_col,
                iv_col=iv_col,
                use_flags=use_flags,
            )

            if k.size < 7:
                logger.warning(
                    f"Skipping expiry {exp}: only {k.size} points (need ≥7)"
                )
                skipped_expiries.append((exp, k.size))
                continue

            logger.info(f"Fitting SVI for expiry {exp} with {k.size} points")

            fit = calibrate_svi_from_quotes(
                k,
                w=w,
                F=F,
                T=T,
                r=r,
                weights=wts,  # if None, fitter will price-weight
                use_price_weighting=(wts is None),
                grid_size=5,
            )

            fits[exp] = fit
            logger.info(
                f"Successfully fit expiry {exp}: loss={fit.loss:.6f}, {fit.notes}"
            )

        except Exception as e:
            logger.error(f"Failed to fit expiry {exp}: {e}")
            failed_expiries.append((exp, str(e)))
            continue

    # Summary logging
    logger.info(f"Surface fitting complete: {len(fits)} successful fits")

    if skipped_expiries:
        logger.warning(
            f"Skipped {len(skipped_expiries)} expiries due to insufficient data:"
        )
        for exp, count in skipped_expiries:
            logger.warning(f"  {exp}: {count} points")

    if failed_expiries:
        logger.error(f"Failed to fit {len(failed_expiries)} expiries:")
        for exp, error in failed_expiries:
            logger.error(f"  {exp}: {error}")

    if not fits:
        logger.error("No expiries produced a valid SVI fit")
        raise ValueError("No expiries produced a valid SVI fit.")

    T_map = {e: float(T_by_expiry[e]) for e in fits}
    F_map = {e: float(F_by_expiry[e]) for e in fits}
    r_map = {e: float(r_by_expiry[e]) for e in fits}

    surface = SVISurface(fits=fits, T=T_map, F=F_map, r=r_map)
    logger.info("Surface construction complete")

    return surface


# --------------------------------------------------------------------- #
# Smoothing (across maturities)
# --------------------------------------------------------------------- #


def _param_series(
    surface: SVISurface,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extract arrays for T and each SVI parameter series.
    Returns T and dict with keys: 'a','b','rho','m','sigma'.
    """
    logger.debug("Extracting parameter series from surface")

    exps = surface.expiries()
    T = np.array([surface.T[e] for e in exps], dtype=float)
    a = np.array([surface.fits[e].params[0] for e in exps], dtype=float)
    b = np.array([surface.fits[e].params[1] for e in exps], dtype=float)
    rho = np.array([surface.fits[e].params[2] for e in exps], dtype=float)
    m = np.array([surface.fits[e].params[3] for e in exps], dtype=float)
    sig = np.array([surface.fits[e].params[4] for e in exps], dtype=float)

    logger.debug(f"Parameter ranges across {len(exps)} expiries:")
    logger.debug(f"  T: [{T.min():.4f}, {T.max():.4f}]")
    logger.debug(f"  a: [{a.min():.6f}, {a.max():.6f}]")
    logger.debug(f"  b: [{b.min():.6f}, {b.max():.6f}]")
    logger.debug(f"  rho: [{rho.min():.6f}, {rho.max():.6f}]")
    logger.debug(f"  m: [{m.min():.6f}, {m.max():.6f}]")
    logger.debug(f"  sigma: [{sig.min():.6f}, {sig.max():.6f}]")

    return T, {"a": a, "b": b, "rho": rho, "m": m, "sigma": sig}


def _smooth_1d(
    T: np.ndarray, y: np.ndarray, method: str = "cubic_spline"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth y(T) across maturities.

    method:
      - 'cubic_spline' (default): requires SciPy & >=4 points.
      - 'poly3': cubic polynomial fit (works without SciPy).
      - 'linear': identity interpolation on given T-grid.
    """
    logger.debug(f"Smoothing with method '{method}': {len(T)} points")

    idx = np.argsort(T)
    Ts, ys = T[idx], y[idx]

    logger.debug(
        f"Sorted input: T ∈ [{Ts.min():.4f}, {Ts.max():.4f}], y ∈ [{ys.min():.6f}, {ys.max():.6f}]"
    )

    if method == "poly3" or not _HAVE_SCIPY:
        if not _HAVE_SCIPY and method == "cubic_spline":
            logger.warning("SciPy not available, falling back to poly3 method")

        logger.debug("Using polynomial degree-3 smoothing")
        coeff = np.polyfit(Ts, ys, deg=3)
        result = np.polyval(coeff, Ts)
        logger.debug(
            f"Poly3 result range: [{result.min():.6f}, {result.max():.6f}]"
        )
        return Ts, result

    if method == "linear":
        logger.debug("Using linear interpolation (identity)")
        return Ts, np.interp(Ts, Ts, ys)

    if _HAVE_SCIPY and len(Ts) >= 4:
        logger.debug("Using SciPy UnivariateSpline")
        try:
            spl = UnivariateSpline(Ts, ys, s=0.0, k=3)
            result = spl(Ts)
            logger.debug(
                f"Spline result range: [{result.min():.6f}, {result.max():.6f}]"
            )
            return Ts, result
        except Exception as e:
            logger.warning(
                f"Spline smoothing failed: {e}, falling back to linear"
            )

    # Fallback: simple linear interpolation
    logger.debug("Using fallback linear interpolation")
    return Ts, np.interp(Ts, Ts, ys)


def smooth_params(
    surface: SVISurface,
    *,
    method: str = "cubic_spline",
    rho_clip: float = 0.999,
    sigma_floor: float = 1e-6,
) -> SVISurface:
    """
    Smooth SVI params (a,b,rho,m,sigma) across maturities and
    return a **new** surface with smoothed parameters. T/F/r copied.
    """
    logger.info(f"Smoothing surface parameters with method '{method}'")
    logger.debug(f"Constraints: |rho| ≤ {rho_clip}, σ ≥ {sigma_floor}")

    T, series = _param_series(surface)
    exps = surface.expiries()

    logger.debug("Smoothing individual parameters...")

    Ts, a_hat = _smooth_1d(T, series["a"], method)
    _, b_hat = _smooth_1d(T, series["b"], method)
    _, rho_hat = _smooth_1d(T, series["rho"], method)
    _, m_hat = _smooth_1d(T, series["m"], method)
    _, s_hat = _smooth_1d(T, series["sigma"], method)

    # Apply constraints
    rho_original_range = [rho_hat.min(), rho_hat.max()]
    sigma_original_range = [s_hat.min(), s_hat.max()]

    rho_hat = np.clip(rho_hat, -rho_clip, rho_clip)
    s_hat = np.maximum(s_hat, sigma_floor)

    logger.debug(
        f"Applied rho clipping: [{rho_original_range[0]:.6f}, {rho_original_range[1]:.6f}] → [{rho_hat.min():.6f}, {rho_hat.max():.6f}]"
    )
    logger.debug(
        f"Applied sigma floor: [{sigma_original_range[0]:.6f}, {sigma_original_range[1]:.6f}] → [{s_hat.min():.6f}, {s_hat.max():.6f}]"
    )

    # Build new surface with smoothed parameters
    fits: Dict[datetime, SVIFit] = {}
    for i, exp in enumerate(exps):
        old = surface.fits[exp]
        params = (
            float(a_hat[i]),
            float(b_hat[i]),
            float(rho_hat[i]),
            float(m_hat[i]),
            float(s_hat[i]),
        )

        logger.debug(
            f"Smoothed params for {exp}: a={params[0]:.6f}, b={params[1]:.6f}, rho={params[2]:.6f}, m={params[3]:.6f}, σ={params[4]:.6f}"
        )

        fits[exp] = SVIFit(
            params=params,
            loss=getattr(old, "loss", math.nan),
            n_used=getattr(old, "n_used", 0),
            method=getattr(old, "method", "L-BFGS-B"),
            notes="smoothed",
        )

    logger.info("Parameter smoothing complete")

    return SVISurface(
        fits=fits,
        T=surface.T.copy(),
        F=surface.F.copy(),
        r=surface.r.copy(),
        asof_utc=surface.asof_utc,
    )


# --------------------------------------------------------------------- #
# Sampling helper for plotting / exports
# --------------------------------------------------------------------- #


def sample_grid(
    surface: SVISurface,
    *,
    K_list: Iterable[float],
    expiry_list: Iterable[datetime],
) -> pd.DataFrame:
    """
    Build a tidy table of (expiry, T, K, k, w, iv) from the surface.

    Columns: expiry (datetime64), T, K, k, w, iv
    """
    K_array = list(K_list)
    exp_array = list(expiry_list)

    logger.info(
        f"Sampling surface grid: {len(K_array)} strikes × {len(exp_array)} expiries"
    )

    rows: List[Dict[str, object]] = []

    for exp in exp_array:
        if exp not in surface.fits:
            logger.warning(f"Expiry {exp} not found in surface, skipping")
            continue

        logger.debug(f"Sampling expiry {exp}")

        F = float(surface.F[exp])
        T = float(surface.T[exp])
        K = np.asarray(K_array, dtype=float)
        k = np.log(K / F)
        wT = surface.w(k, exp)
        iv = np.sqrt(np.maximum(wT, 0.0) / max(T, 1e-12))

        for Ki, ki, wTi, ivi in zip(K, k, wT, iv):
            rows.append(
                {
                    "expiry": exp,
                    "T": T,
                    "K": float(Ki),
                    "k": float(ki),
                    "w": float(wTi),
                    "iv": float(ivi),
                }
            )

    logger.info(f"Generated {len(rows)} grid points")
    return pd.DataFrame(rows)
