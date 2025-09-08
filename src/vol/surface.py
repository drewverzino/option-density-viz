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
    if sigma <= 0.0 or T <= 0.0:
        # Limit: price is just discounted intrinsic
        return float(Df * max(F - K, 0.0))
    sqrtT = np.sqrt(T)
    s = max(sigma, 1e-12)
    lnFK = np.log(max(F, 1e-300) / max(K, 1e-300))
    d1 = (lnFK + 0.5 * s * s * T) / (s * sqrtT)
    d2 = d1 - s * sqrtT
    return float(Df * (F * _Phi(d1) - K * _Phi(d2)))


def _black76_vega(
    F: float, K: float, T: float, sigma: float, Df: float
) -> float:
    """Black-76 vega = Df * F * φ(d1) * sqrt(T)."""
    if sigma <= 0.0 or T <= 0.0:
        return 0.0
    sqrtT = np.sqrt(T)
    s = max(sigma, 1e-12)
    lnFK = np.log(max(F, 1e-300) / max(K, 1e-300))
    d1 = (lnFK + 0.5 * s * s * T) / (s * sqrtT)
    return float(Df * F * _phi(d1) * sqrtT)


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
    # Sanity clamps on price
    intrinsic = Df * max(F - K, 0.0)
    upper = Df * F
    if not np.isfinite(C):
        return 0.0
    C = float(np.clip(C, intrinsic, max(upper - 1e-12, intrinsic)))

    # Quick exits
    if C <= intrinsic + 1e-14:
        return 0.0

    # Bracket: low ~ near-zero vol, high grow until price >= C
    lo, hi = 1e-8, 1.0
    price_hi = _black76_call_price(F, K, T, hi, Df)
    while price_hi < C and hi < 10.0:  # 10 is a generous cap
        hi *= 2.0
        price_hi = _black76_call_price(F, K, T, hi, Df)

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price_mid = _black76_call_price(F, K, T, mid, Df)
        if abs(price_mid - C) < tol:
            return float(mid)
        if price_mid < C:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


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
    Df = float(np.exp(-r * T))
    K = df_calls[strike_col].to_numpy(float)
    C = df_calls[price_col].to_numpy(float)
    out = np.empty_like(C, dtype=float)
    for i in range(C.size):
        out[i] = _implied_vol_black76_call(
            C=float(C[i]), F=float(F), K=float(K[i]), T=float(T), Df=Df
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

    def expiries(self) -> List[datetime]:
        return sorted(self.fits.keys())

    def params_tuple(
        self, expiry: datetime
    ) -> Tuple[float, float, float, float, float]:
        a, b, rho, m, sigma = self.fits[expiry].params
        return float(a), float(b), float(rho), float(m), float(sigma)

    def w(self, k: np.ndarray | float, expiry: datetime) -> np.ndarray:
        """Total variance w(k,T) at given expiry."""
        a, b, rho, m, sigma = self.params_tuple(expiry)
        kk = np.asarray(k, dtype=float)
        return svi_total_variance(kk, a, b, rho, m, sigma)

    def iv(self, K: np.ndarray | float, expiry: datetime) -> np.ndarray:
        """Implied vol σ_imp(K,T) from total variance."""
        F = float(self.F[expiry])
        T = float(self.T[expiry])
        K_arr = np.asarray(K, dtype=float)
        k = np.log(K_arr / F)
        wT = self.w(k, expiry)
        return np.sqrt(np.maximum(wT, 0.0) / max(T, 1e-12))

    def calendar_violations(
        self, k_grid: np.ndarray, tol: float = 0.0
    ) -> Dict[str, object]:
        """
        Calendar check: for T2 > T1 require w(k,T2) >= w(k,T1) ∀k.
        Returns counts and per-adjacent-maturity fractions on k_grid.
        """
        exps = self.expiries()
        if len(exps) < 2:
            return {"count": 0, "n_checks": 0, "fraction": 0.0, "by_pair": []}

        k = np.asarray(k_grid, dtype=float)
        total = 0
        bad = 0
        by_pair: List[Tuple[float, float, float]] = []

        for e1, e2 in zip(exps[:-1], exps[1:]):
            w1 = self.w(k, e1)
            w2 = self.w(k, e2)
            vio = (w2 + tol) < w1
            n = int(k.size)
            c = int(np.count_nonzero(vio))
            total += n
            bad += c
            t1, t2 = float(self.T[e1]), float(self.T[e2])
            by_pair.append((t1, t2, (c / n) if n else 0.0))

        return {
            "count": bad,
            "n_checks": total,
            "fraction": (bad / total) if total else 0.0,
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
    d = df.copy()

    # Optional hygiene flags from preprocess.midprice
    if use_flags:
        if "crossed" in d.columns:
            d = d[~d["crossed"].astype(bool)]
        if "wide" in d.columns:
            d = d[~d["wide"].astype(bool)]

    # Restrict to calls (C)
    t = d[type_col].astype(str).str.upper().str.strip()
    d = d[t.str.startswith("C")]
    d = d.dropna(subset=[strike_col, price_col])
    if d.empty:
        return np.array([]), np.array([]), None

    # If IV column is present and useful, use it; otherwise solve IVs.
    iv = None
    if iv_col in d.columns:
        iv = d[iv_col].to_numpy(dtype=float)
        iv = np.where(np.isfinite(iv) & (iv > 0.0), iv, np.nan)
        if np.count_nonzero(np.isfinite(iv)) < 5:
            iv = None  # not enough reliable IVs

    if iv is None:
        # Solve IVs from call prices (Black-76 on forwards)
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

    # Optional weights: inverse relative spread if present
    weights = None
    if "rel_spread" in d.columns:
        wts = 1.0 / np.clip(
            d["rel_spread"].to_numpy(dtype=float), 1e-6, np.inf
        )
        if np.any(np.isfinite(wts)):
            weights = wts

    # Cull any remaining non-finite entries
    mask = np.isfinite(k) & np.isfinite(w)
    if weights is not None:
        mask = mask & np.isfinite(weights)
        weights = weights[mask]
    return k[mask], w[mask], weights


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
    fits: Dict[datetime, SVIFit] = {}

    # Only expiries that exist in all inputs
    common = (
        set(frames_by_expiry)
        & set(T_by_expiry)
        & set(F_by_expiry)
        & set(r_by_expiry)
    )
    if not common:
        raise ValueError("No overlapping expiries across inputs.")

    for exp in sorted(common):
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
            continue  # not enough points for a stable smile

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

    if not fits:
        raise ValueError("No expiries produced a valid SVI fit.")

    T_map = {e: float(T_by_expiry[e]) for e in fits}
    F_map = {e: float(F_by_expiry[e]) for e in fits}
    r_map = {e: float(r_by_expiry[e]) for e in fits}
    return SVISurface(fits=fits, T=T_map, F=F_map, r=r_map)


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
    exps = surface.expiries()
    T = np.array([surface.T[e] for e in exps], dtype=float)
    a = np.array([surface.fits[e].params[0] for e in exps], dtype=float)
    b = np.array([surface.fits[e].params[1] for e in exps], dtype=float)
    rho = np.array([surface.fits[e].params[2] for e in exps], dtype=float)
    m = np.array([surface.fits[e].params[3] for e in exps], dtype=float)
    sig = np.array([surface.fits[e].params[4] for e in exps], dtype=float)
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
    idx = np.argsort(T)
    Ts, ys = T[idx], y[idx]

    if method == "poly3" or not _HAVE_SCIPY:
        coeff = np.polyfit(Ts, ys, deg=3)
        return Ts, np.polyval(coeff, Ts)

    if method == "linear":
        return Ts, np.interp(Ts, Ts, ys)

    if _HAVE_SCIPY and len(Ts) >= 4:
        spl = UnivariateSpline(Ts, ys, s=0.0, k=3)
        return Ts, spl(Ts)

    # Fallback: simple linear interpolation
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
    T, series = _param_series(surface)
    exps = surface.expiries()

    Ts, a_hat = _smooth_1d(T, series["a"], method)
    _, b_hat = _smooth_1d(T, series["b"], method)
    _, rho_hat = _smooth_1d(T, series["rho"], method)
    _, m_hat = _smooth_1d(T, series["m"], method)
    _, s_hat = _smooth_1d(T, series["sigma"], method)

    rho_hat = np.clip(rho_hat, -rho_clip, rho_clip)
    s_hat = np.maximum(s_hat, sigma_floor)

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
        fits[exp] = SVIFit(
            params=params,
            loss=getattr(old, "loss", math.nan),
            n_used=getattr(old, "n_used", 0),
            method=getattr(old, "method", "L-BFGS-B"),
            notes="smoothed",
        )

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
    rows: List[Dict[str, object]] = []
    for exp in expiry_list:
        F = float(surface.F[exp])
        T = float(surface.T[exp])
        K = np.asarray(list(K_list), dtype=float)
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
    return pd.DataFrame(rows)
