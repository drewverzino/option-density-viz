from __future__ import annotations

"""
Midprice and quote-quality utilities.

This module provides vectorized helpers to compute robust mids and diagnostic flags:
- 'mid': midpoint using both sides when possible; falls back to the available side.
- 'spread': ask - bid (NaN-safe).
- 'rel_spread': (ask - bid) / mid, only when both sides exist and mid>0.
- 'crossed': bid > ask by more than 'eps'.
- 'wide': relative spread exceeds a configurable threshold.
- 'side_used': {"both","bid_only","ask_only","none"} for quick auditing.

Design choices:
- We treat non-positive bids/asks as missing.
- We leave original columns untouched and return a new DataFrame.
"""

from typing import Literal, Optional
import math
import pandas as pd


def _clean_side(x: float | None) -> float | None:
    """Return float or None; coerce negatives/zeros/NaN to None."""
    try:
        if x is None:
            return None
        xf = float(x)
        if not math.isfinite(xf) or xf <= 0.0:
            return None
        return xf
    except Exception:
        return None


def add_midprice_columns(
    df: pd.DataFrame,
    bid_col: str = "bid",
    ask_col: str = "ask",
    *,
    out_prefix: str = "",
    wide_rel_threshold: float = 0.10,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Compute midprice and quality flags and append them as new columns.

    Parameters
    ----------
    df : DataFrame
        Must contain 'bid_col' and 'ask_col'.
    bid_col, ask_col : str
        Column names for bid and ask.
    out_prefix : str
        Optional prefix for all output columns (e.g., 'quote_' -> 'quote_mid').
    wide_rel_threshold : float
        Mark quotes 'wide' when (ask - bid)/mid > threshold (only when both sides present).
    eps : float
        Tolerance for detecting crossed markets (bid > ask + eps).

    Returns
    -------
    DataFrame
        A copy of df with columns:
            f"{out_prefix}mid"
            f"{out_prefix}spread"
            f"{out_prefix}rel_spread"
            f"{out_prefix}crossed"
            f"{out_prefix}wide"
            f"{out_prefix}side_used"  # {"both","bid_only","ask_only","none"}
    """
    d = df.copy()

    bid = d[bid_col].map(_clean_side)
    ask = d[ask_col].map(_clean_side)

    # Determine side availability
    has_bid = bid.notna()
    has_ask = ask.notna()
    both = has_bid & has_ask
    bid_only = has_bid & (~has_ask)
    ask_only = has_ask & (~has_bid)

    # Spread and crossed flag (only meaningful if both sides exist)
    spread = (ask - bid).where(both)
    crossed = (bid > ask + eps).where(both, False)

    # Mid computation
    mid = ((bid + ask) / 2).where(both)
    mid = mid.where(mid > 0)  # non-positive mids -> NaN
    mid = mid.where(~mid.isna(), bid)   # fallback to bid
    mid = mid.where(~mid.isna(), ask)   # fallback to ask

    # Relative spread (use mid only when both sides exist and mid>0)
    rel_spread = (spread / mid).where(both & mid.gt(0))

    # Wide flag
    wide = (rel_spread > wide_rel_threshold).fillna(False)

    # Side used tag
    side_used = pd.Series(index=d.index, dtype=object)
    side_used.loc[both] = "both"
    side_used.loc[bid_only] = "bid_only"
    side_used.loc[ask_only] = "ask_only"
    side_used.loc[~(both | bid_only | ask_only)] = "none"

    # Attach results
    d[f"{out_prefix}mid"] = mid
    d[f"{out_prefix}spread"] = spread
    d[f"{out_prefix}rel_spread"] = rel_spread
    d[f"{out_prefix}crossed"] = crossed.fillna(False)
    d[f"{out_prefix}wide"] = wide
    d[f"{out_prefix}side_used"] = side_used

    return d