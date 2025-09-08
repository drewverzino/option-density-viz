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

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------- basic helpers ---------------------------- #
def forward_price(S: float, r: float, T: float, q: float = 0.0) -> float:
    """F = S * exp((r - q) * T)"""
    result = float(S * np.exp((r - q) * T))
    logger.debug(
        f"Forward price: S={S:.4f}, r={r:.4f}, T={T:.4f}, q={q:.4f} -> F={result:.4f}"
    )
    return result


def yearfrac(
    asof: pd.Timestamp | np.datetime64 | "datetime",
    expiry: pd.Timestamp | np.datetime64 | "datetime",
) -> float:
    """ACT/365.25 approximation; clamp to a tiny positive value."""
    # 365.25 * 24 * 3600 = 31_557_600 seconds in a mean tropical year
    sec = (pd.Timestamp(expiry) - pd.Timestamp(asof)).total_seconds()
    result = max(float(sec) / 31_557_600.0, 1e-12)

    logger.debug(
        f"Year fraction: {asof} to {expiry} = {result:.6f} years ({result*365:.1f} days)"
    )
    return result


def forward_from_carry(spot: float, r: float, T: float) -> float:
    """No-dividend forward: F = S * exp(r T)."""
    result = float(spot) * float(np.exp(r * T))
    logger.debug(
        f"Forward from carry: spot={spot:.4f} * exp({r:.4f} * {T:.4f}) = {result:.4f}"
    )
    return result


def log_moneyness(K: np.ndarray | float, F: float) -> np.ndarray | float:
    """k = ln(K / F). Handles scalar or array K."""
    K_arr = np.asarray(K, dtype=float)
    result = np.log(K_arr / float(F))

    if np.isscalar(K):
        logger.debug(f"Log moneyness: ln({K:.4f}/{F:.4f}) = {result:.4f}")
    else:
        logger.debug(
            f"Log moneyness array: {len(K_arr)} strikes, range [{result.min():.3f}, {result.max():.3f}]"
        )

    return result


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
    logger.info(f"Estimating forward from {len(df_quotes)} quotes using PCP")
    logger.debug(
        f"Parameters: r={r:.4f}, T={T:.4f}, price_col='{price_col}', top_n={top_n}"
    )

    from .pcp import pivot_calls_puts_by_strike  # local import to avoid cycles

    wide = pivot_calls_puts_by_strike(
        df_quotes,
        price_col=price_col,
        type_col=type_col,
        strike_col=strike_col,
    )

    if wide.empty:
        logger.error("No call/put pairs found for forward estimation")
        raise ValueError(
            "Not enough quotes to estimate forward (no C/P pairs)."
        )

    logger.info(f"Found {len(wide)} call/put pairs for forward estimation")

    # Base F_i per strike
    disc_r = np.exp(r * T)
    Ks = wide.index.to_numpy(dtype=float)
    Ci = wide["C"].to_numpy(dtype=float)
    Pi = wide["P"].to_numpy(dtype=float)
    Fi = Ks + (Ci - Pi) * disc_r

    logger.debug(f"Strike range: {Ks.min():.2f} to {Ks.max():.2f}")
    logger.debug(
        f"Per-strike forwards range: {Fi.min():.4f} to {Fi.max():.4f}"
    )

    # Choose ATM neighborhood
    if spot_hint is not None and np.isfinite(spot_hint) and spot_hint > 0:
        guess = float(spot_hint)
        logger.debug(f"Using spot hint as ATM guess: {guess:.4f}")
    else:
        guess = float(np.median(Ks))
        logger.debug(f"Using median strike as ATM guess: {guess:.4f}")

    # Rank by |K - guess| and keep top_n
    distances = np.abs(Ks - guess)
    order = np.argsort(distances)
    keep = order[: max(3, min(top_n, len(order)))]
    Ks_sel, Fi_sel = Ks[keep], Fi[keep]

    logger.debug(f"Selected {len(keep)} strikes near ATM: {Ks_sel}")
    logger.debug(f"Corresponding forwards: {Fi_sel}")

    # Optional weights from relative spread if present
    w = None
    rel_spread_col = "rel_spread"
    if rel_spread_col in df_quotes.columns:
        logger.debug("Using relative spread for weighting")
        # compute per-strike average relative spread (C & P rows)
        tmp = df_quotes[[strike_col, rel_spread_col]].dropna()
        rel_by_K = tmp.groupby(strike_col)[rel_spread_col].mean()
        rel = rel_by_K.reindex(Ks_sel, fill_value=np.nan).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            w = 1.0 / np.clip(rel, 1e-6, np.inf)
        # fallback to equal weights if all NaN/inf
        if not np.any(np.isfinite(w)):
            logger.debug(
                "Relative spread weights invalid, using equal weights"
            )
            w = None
        else:
            logger.debug(f"Relative spread weights: {w[np.isfinite(w)]}")

    if w is None:
        result = float(np.mean(Fi_sel))
        logger.info(f"Forward estimate (equal weights): {result:.4f}")
    else:
        w = np.where(np.isfinite(w), w, 0.0)
        if w.sum() <= 0:
            result = float(np.mean(Fi_sel))
            logger.info(
                f"Forward estimate (fallback to equal weights): {result:.4f}"
            )
        else:
            result = float(np.average(Fi_sel, weights=w))
            logger.info(f"Forward estimate (weighted): {result:.4f}")

    # Sanity check
    if result <= 0:
        logger.error(f"Invalid forward estimate: {result}")
        raise ValueError(f"Invalid forward estimate: {result}")

    return result


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
    logger.info(
        f"Estimating forward from parallel arrays: {len(calls)} entries"
    )
    logger.debug(f"Parameters: r={r:.4f}, T={T:.4f}")

    C = np.asarray(calls, dtype=float)
    P = np.asarray(puts, dtype=float)
    K = np.asarray(strikes, dtype=float)

    # Validate input lengths
    if not (len(C) == len(P) == len(K)):
        logger.error(
            f"Array length mismatch: calls={len(C)}, puts={len(P)}, strikes={len(K)}"
        )
        raise ValueError("calls, puts, and strikes must have the same length")

    mask = np.isfinite(C) & np.isfinite(P) & np.isfinite(K)
    n_valid = mask.sum()

    logger.debug(f"Valid entries: {n_valid}/{len(mask)}")

    if not np.any(mask):
        logger.error("No valid (call, put, strike) triples found")
        raise ValueError("No valid (call, put, strike) triples for PCP.")

    C, P, K = C[mask], P[mask], K[mask]
    disc_r = np.exp(r * T)
    F_i = K + (C - P) * disc_r

    logger.debug(
        f"Per-strike forwards: min={F_i.min():.4f}, max={F_i.max():.4f}, "
        f"mean={F_i.mean():.4f}, std={F_i.std():.4f}"
    )

    if weights is None:
        result = float(np.mean(F_i))
        logger.info(f"Forward estimate (equal weights): {result:.4f}")
        return result

    logger.debug("Applying weights to forward estimation")
    w = np.asarray(weights, dtype=float)
    w = w[mask] if w.shape[0] == mask.shape[0] else w
    w = np.where(np.isfinite(w) & (w >= 0.0), w, 0.0)

    if w.sum() <= 0:
        logger.warning("All weights are zero or invalid, using equal weights")
        result = float(np.mean(F_i))
    else:
        result = float(np.average(F_i, weights=w))
        logger.debug(
            f"Effective weights: min={w[w>0].min():.4f}, max={w.max():.4f}"
        )

    logger.info(f"Forward estimate (weighted): {result:.4f}")
    return result
