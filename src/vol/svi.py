# src/vol/svi.py
from __future__ import annotations

"""
SVI calibration (single-expiry smile) with vega-weighted least squares.

What this module does
---------------------
1) Converts a preprocessed quote DataFrame into:
   - k  : log-moneyness, k = ln(K / F)
   - w  : total variance, w = sigma^2 * T
   - wts: per-point weights (Black-76 vega), so ATM has more influence
2) Fits SVI total-variance function
      w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )
   using L-BFGS-B with safe bounds and multiple seeds.

Why vega weights?
-----------------
Vega is largest near ATM and small deep ITM/OTM. Weighting by vega targets
the fit where the market is most informative, and de-emphasizes noisy
wings.

Inputs you need
---------------
- Forward F for this expiry (estimate via preprocess.forward).
- Year fraction T and continuously compounded rate r.
- A DataFrame with at least: strike, type ('C' or 'P'), mid price.
  Optional 'iv' column (precomputed implied vol); if absent we solve IV.

Key entry points
----------------
- prepare_smile_data(...): build (k, w, weights)
- fit_svi(...): return calibrated SVI parameters (SVIFit)
- svi_w(...): evaluate the SVI total-variance function
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    # SciPy is required for optimization and root solving
    from scipy.optimize import brentq, minimize
except Exception as e:  # pragma: no cover
    raise ImportError("scipy is required for SVI calibration") from e


# ------------------------- Black-76 helpers -------------------------


SQRT2 = math.sqrt(2.0)


def _phi(x: np.ndarray) -> np.ndarray:
    """Standard normal pdf."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * x * x)


def _Phi(x: np.ndarray) -> np.ndarray:
    """Standard normal cdf via erf (vectorized)."""
    return 0.5 * (1.0 + np.erf(x / SQRT2))


def black_d1_d2(
    F: float, K: float, sigma: float, T: float
) -> Tuple[float, float]:
    """
    Compute d1, d2 under Black-76. Returns NaNs if inputs are degenerate
    (e.g., T<=0, sigma<=0, or non-positive F/K).
    """
    if sigma <= 0.0 or T <= 0.0 or F <= 0.0 or K <= 0.0:
        return float("nan"), float("nan")
    vol_sqrt_T = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return d1, d2


def black_price(
    F: float, K: float, sigma: float, T: float, r: float, opt_type: str
) -> float:
    """
    Black-76 forward price discounted by exp(-rT).
    Falls back to intrinsic value when sigma≈0 to avoid NaNs.
    """
    Df = math.exp(-r * T)
    d1, d2 = black_d1_d2(F, K, sigma, T)
    if not np.isfinite(d1) or not np.isfinite(d2):
        # sigma ~ 0 fallback: intrinsic value
        if str(opt_type).upper().startswith("C"):
            return Df * max(F - K, 0.0)
        return Df * max(K - F, 0.0)

    if str(opt_type).upper().startswith("C"):
        return Df * (F * _Phi(d1) - K * _Phi(d2))
    return Df * (K * _Phi(-d2) - F * _Phi(-d1))


def black_vega(F: float, K: float, sigma: float, T: float, r: float) -> float:
    """
    Forward vega under Black-76.
    If sigma or T are non-positive, or d1 is NaN, returns 0 (no weight).
    """
    if sigma <= 0.0 or T <= 0.0:
        return 0.0
    Df = math.exp(-r * T)
    d1, _ = black_d1_d2(F, K, sigma, T)
    if not np.isfinite(d1):
        return 0.0
    return Df * F * math.sqrt(T) * float(_phi(np.array([d1]))[0])


def implied_vol_black(
    price: float, F: float, K: float, T: float, r: float, opt_type: str
) -> float:
    """
    Compute Black-76 implied vol by bracketing on [1e-8, 5] and using Brent.

    Returns NaN if:
    - inputs are invalid (e.g., price<=0, T<=0);
    - the target price is not in the achievable price range;
    - the bracket fails to sign-change (no root).
    """
    if price <= 0.0 or F <= 0.0 or K <= 0.0 or T <= 0.0:
        return float("nan")

    Df = math.exp(-r * T)

    # Tight theoretical bounds on call/put values under Black-76.
    if str(opt_type).upper().startswith("C"):
        lo_price = Df * max(F - K, 0.0)  # intrinsic, sigma -> 0
        hi_price = Df * F  # sigma -> inf (call cap)
    else:
        lo_price = Df * max(K - F, 0.0)
        hi_price = Df * K

    if not (lo_price <= price <= hi_price):
        return float("nan")

    def f(sig: float) -> float:
        return black_price(F, K, sig, T, r, opt_type) - price

    a, b = 1e-8, 5.0
    try:
        fa, fb = f(a), f(b)
        # If the bracket does not straddle zero, give up cleanly.
        if math.isnan(fa) or math.isnan(fb) or fa * fb > 0.0:
            return float("nan")
        return float(brentq(f, a, b, maxiter=100, xtol=1e-8))
    except Exception:
        return float("nan")


# ----------------------------- SVI core -----------------------------


def svi_w(
    k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float
) -> np.ndarray:
    """
    SVI total variance w(k) in the "raw" (not-J-W) parameterization.

    Parameters
    ----------
    k : array
        Log-moneyness grid ln(K/F).
    a, b, rho, m, sigma : floats
        SVI parameters (bounded during calibration).

    Returns
    -------
    w : array
        Total variance on the same grid.
    """
    x = k - m
    return a + b * (rho * x + np.sqrt(x * x + sigma * sigma))


@dataclass
class SVIFit:
    """
    Container for SVI fit results.
    - (a, b, rho, m, sigma): fitted parameters
    - loss: objective value (weighted MSE in w)
    - n_used: number of points used in the fit
    - notes: small status message ("ok" or solver details)
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    loss: float
    n_used: int
    notes: str = ""

    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        """Return parameters as a 5-tuple in the usual order."""
        return (self.a, self.b, self.rho, self.m, self.sigma)


def prepare_smile_data(
    df,
    *,
    T: float,
    r: float,
    F: float,
    price_col: str = "mid",
    type_col: str = "type",
    strike_col: str = "strike",
    iv_col: Optional[str] = None,
    use_flags: bool = True,
    crossed_col: str = "crossed",
    wide_col: str = "wide",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (k, w, weights) for SVI from a quotes DataFrame.

    Steps
    -----
    1) Drop rows missing price/strike. Optionally drop rows flagged as
       crossed or wide (produced by preprocess.midprice).
    2) Use column `iv` if present and finite; otherwise compute Black-76
       implied vol by solving price(mid) -> sigma.
    3) Compute log-moneyness k = ln(K/F) and total variance w = sigma^2*T.
    4) Weight each point by Black-76 vega. If vega=0 or invalid, use 1.

    Returns
    -------
    (k, w, wts) as numpy arrays. Empty arrays if nothing survives filters.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):  # pragma: no cover
        raise TypeError("df must be a pandas DataFrame")

    d = df.copy()
    d = d.dropna(subset=[price_col, strike_col])

    # Remove obviously bad quotes (if flags exist).
    if use_flags:
        if crossed_col in d.columns:
            d = d[~d[crossed_col].astype(bool)]
        if wide_col in d.columns:
            d = d[~d[wide_col].astype(bool)]
    if len(d) == 0:
        return np.array([]), np.array([]), np.array([])

    # Normalize type to 'C'/'P' single letters.
    if type_col in d.columns:
        t = d[type_col].astype(str).str.upper().str[0]
    else:
        t = pd.Series(["C"] * len(d), index=d.index)

    K = d[strike_col].astype(float).to_numpy()
    P = d[price_col].astype(float).to_numpy()

    # If an IV column exists, prefer it; else solve.
    iv_series = (
        d[iv_col].astype(float) if (iv_col and iv_col in d.columns) else None
    )

    sigmas = np.empty_like(P, dtype=float)
    sigmas.fill(np.nan)

    for i in range(len(P)):
        # Use supplied IV if good; otherwise, solve for IV.
        iv_guess = (
            float(iv_series.iloc[i]) if iv_series is not None else float("nan")
        )
        if np.isfinite(iv_guess) and iv_guess > 0.0:
            sigmas[i] = iv_guess
        else:
            sigmas[i] = implied_vol_black(P[i], F, K[i], T, r, t.iloc[i])

    # Keep only valid (positive) vols and strikes.
    mask = np.isfinite(sigmas) & (sigmas > 0.0) & np.isfinite(K) & (K > 0.0)
    K = K[mask]
    sigmas = sigmas[mask]
    if len(sigmas) == 0:
        return np.array([]), np.array([]), np.array([])

    k = np.log(K / float(F))
    w = (sigmas * sigmas) * float(T)

    # Vega weights: if vega is invalid or 0 -> fallback to 1.
    wts = np.empty_like(sigmas, dtype=float)
    for i in range(len(sigmas)):
        v = black_vega(F, K[i], float(sigmas[i]), T, r)
        wts[i] = v if np.isfinite(v) and v > 0.0 else 1.0
    return k, w, wts


def _default_bounds() -> Dict[str, Tuple[float, float]]:
    """
    Conservative box constraints. They keep the optimizer away from
    degenerate values but are wide enough for crypto/equity smiles.
    """
    return {
        "a": (1e-8, 5.0),
        "b": (1e-8, 10.0),
        "rho": (-0.999, 0.999),
        "m": (-2.5, 2.5),
        "sigma": (1e-8, 5.0),
    }


def _unpack(x: np.ndarray) -> Dict[str, float]:
    """Turn a parameter vector into a dict with named fields."""
    return {
        "a": float(x[0]),
        "b": float(x[1]),
        "rho": float(x[2]),
        "m": float(x[3]),
        "sigma": float(x[4]),
    }


def _seed_grid(k: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Build a small seed grid for (a, b, rho, m, sigma).

    Intuition:
    - a ~ min total variance (kept small, but >0)
    - b ~ curvature level; use std(w) as a rough scale
    - m, rho shape the skew; we try a handful of plausible pairs
    - sigma is the "kink" softness; start moderate
    """
    a0 = max(1e-4, float(np.nanmin(w)) * 0.7)
    b0 = max(1e-3, float(np.nanstd(w)))
    sigma0 = 0.2
    m_vals = np.array([-0.2, 0.0, 0.2])
    rho_vals = np.array([-0.5, -0.2, 0.0, 0.2])
    seeds = []
    for m in m_vals:
        for rho in rho_vals:
            seeds.append(np.array([a0, b0, rho, m, sigma0], dtype=float))
    return np.vstack(seeds)


def fit_svi(
    k: np.ndarray,
    w: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    seeds: Optional[np.ndarray] = None,
) -> SVIFit:
    """
    Fit SVI by minimizing vega-weighted MSE in total variance.

    Objective
    ---------
      minimize  sum_i weights_i * ( w_i - w_svi(k_i) )^2  / n_used

    Notes
    -----
    - We prefilter non-finite values and require at least 6 points.
    - If no weights are provided we use ones.
    - We search from a small grid of seeds and keep the best result.

    Returns
    -------
    SVIFit with parameters, final loss, number of points used, and notes.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)

    # Filter out bad points and tiny/negative variances.
    mask = np.isfinite(k) & np.isfinite(w) & (w > 0.0)
    k, w = k[mask], w[mask]
    n_used = int(mask.sum())
    if n_used < 6:
        raise ValueError("Need at least 6 valid points to fit SVI")

    # Align weights to filtered points and sanitize.
    if weights is None:
        weights = np.ones_like(w)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights[mask]
        weights = np.where(
            np.isfinite(weights) & (weights > 0.0), weights, 1.0
        )

    # Bounds keep the optimizer well-behaved.
    if bounds is None:
        bounds = _default_bounds()
    lb = np.array(
        [
            bounds["a"][0],
            bounds["b"][0],
            bounds["rho"][0],
            bounds["m"][0],
            bounds["sigma"][0],
        ],
        dtype=float,
    )
    ub = np.array(
        [
            bounds["a"][1],
            bounds["b"][1],
            bounds["rho"][1],
            bounds["m"][1],
            bounds["sigma"][1],
        ],
        dtype=float,
    )

    def obj(x: np.ndarray) -> float:
        """Weighted mean square error in total variance."""
        p = _unpack(x)
        wp = svi_w(k, p["a"], p["b"], p["rho"], p["m"], p["sigma"])
        res = w - wp
        return float(np.sum(weights * res * res) / n_used)

    # Try several seeds and keep the best solution.
    if seeds is None:
        seeds = _seed_grid(k, w)

    best = None
    best_fun = float("inf")
    best_notes = ""

    for s in seeds:
        # Clamp seed to bounds to avoid immediate failures.
        x0 = np.minimum(np.maximum(s, lb), ub)
        res = minimize(
            obj,
            x0,
            method="L-BFGS-B",
            bounds=list(zip(lb, ub)),
            options={"maxiter": 200, "ftol": 1e-10},
        )
        if res.success and res.fun < best_fun:
            best = res.x
            best_fun = float(res.fun)
            best_notes = "ok"
        elif best is None:
            # Keep the first attempt even if not "success".
            best = res.x
            best_fun = float(res.fun)
            best_notes = f"{res.message}"

    p = _unpack(np.asarray(best, dtype=float))
    return SVIFit(
        a=p["a"],
        b=p["b"],
        rho=p["rho"],
        m=p["m"],
        sigma=p["sigma"],
        loss=best_fun,
        n_used=n_used,
        notes=best_notes,
    )
