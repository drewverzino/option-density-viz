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

from __future__ import annotations

import logging
import math

import pandas as pd

logger = logging.getLogger("preprocess.midprice")


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
    logger.info(
        f"Computing midprices for {len(df)} quotes (wide_threshold={wide_rel_threshold:.1%})"
    )

    if bid_col not in df.columns or ask_col not in df.columns:
        raise ValueError(
            f"DataFrame missing required columns: {bid_col}, {ask_col}"
        )

    d = df.copy()

    # Clean and validate bid/ask data
    logger.debug(f"Cleaning bid/ask columns: {bid_col}, {ask_col}")
    bid = d[bid_col].map(_clean_side)
    ask = d[ask_col].map(_clean_side)

    # Determine side availability
    has_bid = bid.notna()
    has_ask = ask.notna()
    both = has_bid & has_ask
    bid_only = has_bid & (~has_ask)
    ask_only = has_ask & (~has_bid)
    neither = ~(has_bid | has_ask)

    # Log data quality summary
    n_both = both.sum()
    n_bid_only = bid_only.sum()
    n_ask_only = ask_only.sum()
    n_neither = neither.sum()

    logger.info(
        f"Quote availability: {n_both} both sides, {n_bid_only} bid-only, "
        f"{n_ask_only} ask-only, {n_neither} neither"
    )

    # Spread and crossed flag (only meaningful if both sides exist)
    spread = (ask - bid).where(both)
    crossed = (bid > ask + eps).where(both, False)

    # Count crossed markets
    n_crossed = crossed.sum()
    if n_crossed > 0:
        logger.warning(
            f"Found {n_crossed} crossed markets (bid > ask + {eps})"
        )
        if logger.isEnabledFor(logging.DEBUG):
            crossed_strikes = d[crossed].get('strike', range(len(d[crossed])))
            logger.debug(f"Crossed strikes: {list(crossed_strikes)[:5]}...")

    # Mid computation
    mid = ((bid + ask) / 2).where(both)
    mid = mid.where(mid > 0)  # non-positive mids -> NaN
    mid = mid.where(~mid.isna(), bid)  # fallback to bid
    mid = mid.where(~mid.isna(), ask)  # fallback to ask

    # Count successful mid calculations
    n_mids = mid.notna().sum()
    logger.debug(f"Computed {n_mids}/{len(df)} valid midprices")

    # Relative spread (use mid only when both sides exist and mid>0)
    rel_spread = (spread / mid).where(both & mid.gt(0))

    # Wide flag
    wide = (rel_spread > wide_rel_threshold).fillna(False)
    n_wide = wide.sum()

    if n_wide > 0:
        logger.info(f"Found {n_wide} wide spreads (>{wide_rel_threshold:.1%})")
        if logger.isEnabledFor(logging.DEBUG):
            avg_wide_spread = rel_spread[wide].mean()
            max_wide_spread = rel_spread[wide].max()
            logger.debug(
                f"Wide spread stats: avg={avg_wide_spread:.1%}, max={max_wide_spread:.1%}"
            )

    # Side used tag
    side_used = pd.Series(index=d.index, dtype=object)
    side_used.loc[both] = "both"
    side_used.loc[bid_only] = "bid_only"
    side_used.loc[ask_only] = "ask_only"
    side_used.loc[neither] = "none"

    # Attach results
    d[f"{out_prefix}mid"] = mid
    d[f"{out_prefix}spread"] = spread
    d[f"{out_prefix}rel_spread"] = rel_spread
    d[f"{out_prefix}crossed"] = crossed.fillna(False)
    d[f"{out_prefix}wide"] = wide
    d[f"{out_prefix}side_used"] = side_used

    logger.info(
        f"Added midprice columns with prefix '{out_prefix}': "
        f"{n_mids} mids, {n_crossed} crossed, {n_wide} wide"
    )

    return d
