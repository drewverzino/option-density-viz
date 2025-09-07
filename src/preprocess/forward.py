from __future__ import annotations

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

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def forward_price(S: float, r: float, T: float, q: float = 0.0) -> float:
    """F = S * exp((r - q) * T)"""
    return float(S * np.exp((r - q) * T))


def log_moneyness(K: np.ndarray | float, F: float) -> np.ndarray | float:
    """k = ln(K / F)"""
    return np.log(np.asarray(K, dtype=float) / float(F))


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
        df_quotes, price_col=price_col, type_col=type_col, strike_col=strike_col
    )
    if wide.empty:
        raise ValueError("Not enough quotes to estimate forward (no C/P pairs).")

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
