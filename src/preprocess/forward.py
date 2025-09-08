"""
Forward price and log-moneyness utilities.

We provide:
- forward_price(S, r, T, q): F = S * exp((r - q) * T)
- log_moneyness(K, F): k = ln(K / F)
- estimate_forward_from_chain(...): robust forward estimator from PCP across strikes

Forward from parity:
    C - P = S e^{-qT} - K e^{-rT}
 => (C - P) e^{rT} = F - K
 => F = K + (C - P) e^{rT}
This holds regardless of q once F is defined as S e^{(r - q)T}.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


# ----------------------------- basic helpers ---------------------------- #
def forward_price(S: float, r: float, T: float, q: float = 0.0) -> float:
    """F = S * exp((r - q) * T)"""
    return float(S * np.exp((r - q) * T))


def yearfrac(
    asof: pd.Timestamp | np.datetime64 | "datetime",
    expiry: pd.Timestamp | np.datetime64 | "datetime",
) -> float:
    """ACT/365.25 approximation; clamp to a tiny positive value."""
    # 365.25 * 24 * 3600 = 31_557_600 seconds in a mean tropical year
    sec = (pd.Timestamp(expiry) - pd.Timestamp(asof)).total_seconds()
    return max(float(sec) / 31_557_600.0, 1e-12)


def forward_from_carry(spot: float, r: float, T: float) -> float:
    """No-dividend forward: F = S * exp(r T)."""
    return float(spot) * float(np.exp(r * T))


def log_moneyness(K: np.ndarray | float, F: float) -> np.ndarray | float:
    """k = ln(K / F). Handles scalar or array K."""
    return np.log(np.asarray(K, dtype=float) / float(F))


# ------------------------- forward estimators --------------------------- #


def estimate_forward_from_chain(
    df_quotes: pd.DataFrame,
    *,
    r: float,
    T: float,
    price_col: str = "mid",
    type_col: str = "type",
    strike_col: str = "strike",
    spot_hint: Optional[float] = None,
    top_n: int = 7,
) -> float:
    """
    Estimate the forward F from call/put pairs via PCP:
        F_i = K_i + (C_i - P_i) * e^{rT}

    Strategy:
    - Build a strike-indexed table with C and P using the given 'price_col' (e.g., mids).
    - Select the 'top_n' strikes closest to an ATM guess (spot_hint or median strike).
    - Weight each F_i inversely by relative spread if available; otherwise use equal weights.

    Returns
    -------
    float
        Estimated forward price F.
    """
    from .pcp import pivot_calls_puts_by_strike  # local import to avoid cycles

    wide = pivot_calls_puts_by_strike(
        df_quotes,
        price_col=price_col,
        type_col=type_col,
        strike_col=strike_col,
    )
    if wide.empty:
        raise ValueError(
            "Not enough quotes to estimate forward (no C/P pairs)."
        )

    # Base F_i per strike
    disc_r = np.exp(r * T)
    Ks = wide.index.to_numpy(dtype=float)
    Ci = wide["C"].to_numpy(dtype=float)
    Pi = wide["P"].to_numpy(dtype=float)
    Fi = Ks + (Ci - Pi) * disc_r

    # Choose ATM neighborhood
    if spot_hint is not None and np.isfinite(spot_hint) and spot_hint > 0:
        guess = float(spot_hint)  # a first approximation of F
    else:
        # fallback: median strike as rough ATM
        guess = float(np.median(Ks))

    # Rank by |K - guess| and keep top_n
    order = np.argsort(np.abs(Ks - guess))
    keep = order[: max(3, min(top_n, len(order)))]
    Ks_sel, Fi_sel = Ks[keep], Fi[keep]

    # Optional weights from relative spread if present
    w = None
    rel_spread_col = "rel_spread"
    if rel_spread_col in df_quotes.columns:
        # compute per-strike average relative spread (C & P rows)
        tmp = df_quotes[[strike_col, rel_spread_col]].dropna()
        rel_by_K = tmp.groupby(strike_col)[rel_spread_col].mean()
        rel = rel_by_K.reindex(Ks_sel, fill_value=np.nan).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / np.clip(rel, 1e-6, np.inf)
        # fallback to equal weights if all NaN/inf
        if not np.any(np.isfinite(w)):
            w = None

    if w is None:
        return float(np.mean(Fi_sel))
    else:
        w = np.where(np.isfinite(w), w, 0.0)
        if w.sum() <= 0:
            return float(np.mean(Fi_sel))
        return float(np.average(Fi_sel, weights=w))


def estimate_forward_from_pcp(
    calls: np.ndarray | pd.Series,
    puts: np.ndarray | pd.Series,
    strikes: np.ndarray | pd.Series,
    *,
    r: float,
    T: float,
    weights: np.ndarray | pd.Series | None = None,
) -> float:
    """
    Direct PCP-based forward estimator from parallel arrays/Series.

    Given arrays (or pandas Series) of call mids `calls`, put mids `puts`, and
    strikes `strikes` for the *same* set of strikes (rows), compute per-strike
    PCP forwards:
        F_i = K_i + (C_i - P_i) * exp(rT)

    Then return an equal- or weight-averaged estimate.

    Parameters
    ----------
    calls, puts, strikes : array-like (same length)
        Call/put mids and strikes. Any row with a NaN in (C, P, K) is ignored.
    r : float
        Continuously-compounded risk-free (or carry) rate.
    T : float
        Time to maturity in years.
    weights : array-like, optional
        Optional non-negative weights (e.g., inverse relative spread). If not
        provided, all valid rows are equally weighted.

    Returns
    -------
    float
        Estimated forward price F.

    Raises
    ------
    ValueError
        If fewer than one valid (C,P,K) triple is available.
    """
    C = np.asarray(calls, dtype=float)
    P = np.asarray(puts, dtype=float)
    K = np.asarray(strikes, dtype=float)

    mask = np.isfinite(C) & np.isfinite(P) & np.isfinite(K)
    if not np.any(mask):
        raise ValueError("No valid (call, put, strike) triples for PCP.")

    C, P, K = C[mask], P[mask], K[mask]
    disc_r = np.exp(r * T)
    F_i = K + (C - P) * disc_r

    if weights is None:
        return float(np.mean(F_i))

    w = np.asarray(weights, dtype=float)
    w = w[mask] if w.shape[0] == mask.shape[0] else w
    w = np.where(np.isfinite(w) & (w >= 0.0), w, 0.0)
    if w.sum() <= 0:
        return float(np.mean(F_i))
    return float(np.average(F_i, weights=w))
