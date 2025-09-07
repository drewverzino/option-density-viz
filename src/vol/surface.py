# src/vol/surface.py
from __future__ import annotations

"""
SVI surface across maturities (time dimension).

Why this module exists
----------------------
You already have a *per-expiry* SVI fit (one smile per maturity). Most
real analyses need the *full surface* w(k, T) or σ_imp(K, T), plus
calendar-arbitrage checks. This file:

1) Fits SVI per expiry from your preprocessed DataFrames
   (using vol.svi.prepare_smile_data + vol.svi.fit_svi).
2) Stitches those fits into an SVISurface object that can:
   - evaluate total variance w(k, T) and implied vols σ_imp(K, T)
   - smooth parameters across T (spline/poly) for nicer surfaces
   - run calendar no-arbitrage diagnostics
   - sample a tidy grid for plotting/exports

Inputs (what you must supply)
-----------------------------
For each expiry date `exp` you need:
- a pandas DataFrame (already "clean") with columns:
    strike (float), type ('C'/'P'), *either* iv (implied vol)
    or a price column (default 'mid'). Flags from preprocess are used
    if `use_flags=True`.
- scalars per expiry:
    T_by_expiry[exp]  -> time to maturity in years
    F_by_expiry[exp]  -> forward price (same units as strike)
    r_by_expiry[exp]  -> risk-free rate (continuously compounded)

Conventions
-----------
- k = log(K / F) is log-moneyness. Surface internally works in k.
- w(k, T) is *total variance* (sigma^2 * T).
- σ_imp = sqrt(w / T). We guard against T≈0 and w<0 numerically.

Minimal example
---------------
>>> surf = fit_surface_from_frames(frames, T_map, F_map, r_map)
>>> surf = smooth_params(surf, method="cubic_spline")
>>> iv = surf.iv([90, 100, 110], expiry=list(surf.fits)[0])
>>> diag = surf.calendar_violations(k_grid=np.linspace(-1, 1, 101))

Implementation notes
--------------------
- Smoothing helps when per-expiry fits are noisy. We clamp rho to
  (-rho_clip, +rho_clip) and floor sigma to a small positive value to
  avoid pathological wings / negative square-roots.
- SciPy's UnivariateSpline is used if available; otherwise we fall back
  to a cubic polynomial fit. For tiny maturity sets (<4), splines are
  automatically skipped.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple
from datetime import datetime

import math
import numpy as np
import pandas as pd

# SciPy is optional: if missing, we still provide a working fallback.
try:
    from scipy.interpolate import UnivariateSpline  # type: ignore

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# Per-expiry SVI fitting utilities.
from .svi import SVIFit, prepare_smile_data, fit_svi, svi_w


# --------------------------------------------------------------------------- #
# Data container
# --------------------------------------------------------------------------- #


@dataclass
class SVISurface:
    """
    A light container holding per-expiry SVI fits and meta.

    Attributes
    ----------
    fits : dict[datetime, SVIFit]
        SVI parameters (a, b, rho, m, sigma) per expiry.
    T, F, r : dict[datetime, float]
        Time to maturity, forward, and risk-free per expiry.
    asof_utc : datetime | None
        Optional "as of" timestamp for bookkeeping.

    Methods
    -------
    w(k, expiry)  -> total variance at log-moneyness k
    iv(K, expiry) -> implied volatility at strike K
    calendar_violations(k_grid) -> no-arb diagnostics across maturities
    """

    fits: Dict[datetime, SVIFit]
    T: Dict[datetime, float]
    F: Dict[datetime, float]
    r: Dict[datetime, float]
    asof_utc: Optional[datetime] = None

    # ---- convenience ----

    def expiries(self) -> List[datetime]:
        """Return expiries in ascending chronological order."""
        return sorted(self.fits.keys())

    def as_tuple(
        self, expiry: datetime
    ) -> Tuple[float, float, float, float, float]:
        """Return (a, b, rho, m, sigma) for a given expiry."""
        fit = self.fits[expiry]
        return fit.a, fit.b, fit.rho, fit.m, fit.sigma

    # ---- evaluations ----

    def w(self, k: np.ndarray, expiry: datetime) -> np.ndarray:
        """
        Evaluate total variance w(k, T) at a given expiry.

        Parameters
        ----------
        k : array_like
            Log-moneyness values.
        expiry : datetime
            Maturity to evaluate.

        Returns
        -------
        np.ndarray
            Total variance values, same shape as `k`.
        """
        a, b, rho, m, sigma = self.as_tuple(expiry)
        return svi_w(np.asarray(k, dtype=float), a, b, rho, m, sigma)

    def iv(self, K: np.ndarray | float, expiry: datetime) -> np.ndarray:
        """
        Evaluate implied volatility σ_imp(K, T) at a given expiry.

        Notes
        -----
        Converts K to k = log(K/F), then σ = sqrt(w / T). We floor T
        to avoid division by ~0, and clip w at 0 to stay real.
        """
        F = float(self.F[expiry])
        T = float(self.T[expiry])
        K_arr = np.asarray(K, dtype=float)
        k = np.log(K_arr / F)
        wT = self.w(k, expiry)
        iv = np.sqrt(np.maximum(wT, 0.0) / max(T, 1e-12))
        return iv

    # ---- diagnostics ----

    def calendar_violations(
        self,
        k_grid: np.ndarray,
        tol: float = 0.0,
    ) -> Dict[str, object]:
        """
        Calendar no-arbitrage check on total variance.

        Rule checked
        ------------
        For T2 > T1 we should have w(k, T2) >= w(k, T1) for all k.
        We count violations on a supplied k_grid.

        Parameters
        ----------
        k_grid : array_like
            Log-moneyness points to check on.
        tol : float
            Soft tolerance. Using a small negative tol lets you ignore
            tiny numerical noise (e.g. tol = -1e-8).

        Returns
        -------
        dict with keys:
            count     : number of violated (pair, k) checks
            n_checks  : total checks (pairs * len(k_grid))
            fraction  : count / n_checks
            by_pair   : list of (T1, T2, fraction_for_that_pair)
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
            # A violation is when later maturity has *less* total variance.
            violate = (w2 + tol) < w1
            n = int(k.size)
            c = int(np.count_nonzero(violate))
            total += n
            bad += c
            t1 = float(self.T[e1])
            t2 = float(self.T[e2])
            frac = c / n if n else 0.0
            by_pair.append((t1, t2, frac))

        frac_all = bad / total if total else 0.0
        return {
            "count": bad,
            "n_checks": total,
            "fraction": frac_all,
            "by_pair": by_pair,
        }


# --------------------------------------------------------------------------- #
# Fit per-expiry SVI and build a surface
# --------------------------------------------------------------------------- #


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

    What this does
    --------------
    For each expiry present in all mappings:
    - Build (k, w, weights) via prepare_smile_data.
    - Fit SVI parameters with fit_svi (vega-weighted LS by default).
    - Skip expiries with too few valid points (<7).

    Parameters (columns)
    --------------------
    price_col : column name for prices when no IV is present (default 'mid')
    type_col  : column with 'C'/'P'
    strike_col: strikes in same units as the forward
    iv_col    : optional; if present we can skip IV solve from prices

    Returns
    -------
    SVISurface
        Surface with per-expiry fits and the provided meta (T, F, r).
    """
    fits: Dict[datetime, SVIFit] = {}

    # Only keep expiries that exist in all meta dictionaries.
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

        k, w, wts = prepare_smile_data(
            df,
            T=T,
            r=r,
            F=F,
            price_col=price_col,
            type_col=type_col,
            strike_col=strike_col,
            iv_col=iv_col,
            use_flags=use_flags,
        )
        # Heuristic: need a handful of points across wings to be stable.
        if len(k) < 7:
            continue

        fits[exp] = fit_svi(k, w, weights=wts)

    if not fits:
        raise ValueError("No expiries produced a valid SVI fit.")

    T_map = {e: float(T_by_expiry[e]) for e in fits}
    F_map = {e: float(F_by_expiry[e]) for e in fits}
    r_map = {e: float(r_by_expiry[e]) for e in fits}
    return SVISurface(fits=fits, T=T_map, F=F_map, r=r_map)


# --------------------------------------------------------------------------- #
# Smoothing utilities
# --------------------------------------------------------------------------- #


def _param_series(
    surface: SVISurface,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extract arrays for T and each SVI parameter series.

    Returns
    -------
    T : np.ndarray shape (n_expiries,)
    series : dict[str, np.ndarray]
        Keys: 'a', 'b', 'rho', 'm', 'sigma'
    """
    exps = surface.expiries()
    T = np.array([surface.T[e] for e in exps], dtype=float)
    a = np.array([surface.fits[e].a for e in exps], dtype=float)
    b = np.array([surface.fits[e].b for e in exps], dtype=float)
    rho = np.array([surface.fits[e].rho for e in exps], dtype=float)
    m = np.array([surface.fits[e].m for e in exps], dtype=float)
    sig = np.array([surface.fits[e].sigma for e in exps], dtype=float)
    return T, {"a": a, "b": b, "rho": rho, "m": m, "sigma": sig}


def _smooth_1d(
    T: np.ndarray,
    y: np.ndarray,
    method: str = "cubic_spline",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth 1-D series y(T).

    method:
      - 'cubic_spline' (default): C^2 spline with s=0 (interpolant).
        Requires SciPy and at least 4 maturities.
      - 'poly3': cubic polynomial least-squares fit (no SciPy needed).
      - 'linear': pass-through / identity interpolation.

    Returns (T_sorted, y_smooth(T_sorted)).
    """
    idx = np.argsort(T)
    Ts = T[idx]
    ys = y[idx]

    if method == "poly3" or not _HAVE_SCIPY:
        coeff = np.polyfit(Ts, ys, deg=3)
        yhat = np.polyval(coeff, Ts)
        return Ts, yhat

    if method == "linear":
        # identity on the known grid
        return Ts, np.interp(Ts, Ts, ys)

    # cubic spline if available and we have enough points
    if _HAVE_SCIPY and len(Ts) >= 4:
        spl = UnivariateSpline(Ts, ys, s=0.0, k=3)
        return Ts, spl(Ts)

    # fallback if not enough points
    return Ts, np.interp(Ts, Ts, ys)


def smooth_params(
    surface: SVISurface,
    *,
    method: str = "cubic_spline",
    rho_clip: float = 0.999,
    sigma_floor: float = 1e-6,
) -> SVISurface:
    """
    Smooth SVI params (a,b,rho,m,sigma) across maturities.

    Why clamp/floor?
    ----------------
    - rho is a correlation-like skew in (-1, +1). Values near ±1 can
      make wings explode; we clip to a safe interior.
    - sigma is like a "horizontal" width. Non-positive sigma would
      break the square-root in SVI; we floor to a tiny positive value.

    Returns a NEW SVISurface with smoothed parameters. F, r, T are
    copied over unchanged.
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

    # Rebuild SVIFit objects in the original expiry order.
    fits: Dict[datetime, SVIFit] = {}
    for i, exp in enumerate(exps):
        old = surface.fits[exp]
        fits[exp] = SVIFit(
            a=float(a_hat[i]),
            b=float(b_hat[i]),
            rho=float(rho_hat[i]),
            m=float(m_hat[i]),
            sigma=float(s_hat[i]),
            # carry diagnostics/metadata through
            loss=getattr(old, "loss", math.nan),
            n_used=getattr(old, "n_used", 0),
            notes="smoothed",
        )

    return SVISurface(
        fits=fits,
        T=surface.T.copy(),
        F=surface.F.copy(),
        r=surface.r.copy(),
        asof_utc=surface.asof_utc,
    )


# --------------------------------------------------------------------------- #
# Sampling helper for plotting / exports
# --------------------------------------------------------------------------- #


def sample_grid(
    surface: SVISurface,
    *,
    K_list: Iterable[float],
    expiry_list: Iterable[datetime],
) -> pd.DataFrame:
    """
    Build a tidy table of (expiry, T, K, k, w, iv) from the surface.

    Why this is useful
    ------------------
    Many plotting libs (wireframes/contours) are easier to drive from a
    rectangular-ish DataFrame. This helper does the IV/w lookups and
    returns a table you can feed directly to seaborn/plotly/mpl.

    Returns
    -------
    DataFrame columns:
      expiry (datetime64[ns]), T, K, k, w, iv
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
