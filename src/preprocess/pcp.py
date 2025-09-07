from __future__ import annotations

"""
Put–Call Parity (PCP) utilities.

We provide:
- synthesis functions to create the missing leg given (S, K, r, T) and the other leg,
- residual checks (how far a given pair deviates from parity),
- DataFrame helpers to pivot calls/puts by strike and compute residuals and synthetic prices.

Assumptions:
- European options, continuous compounding.
- For equities with dividends, use forward S*exp(-qT) or pass the forward directly.
- PCP with dividend yield q: C - P = S*e^{-qT} - K*e^{-rT}. The forward is F = S*e^{(r-q)T}.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def synth_put_from_call(
    S: float, K: float, r: float, T: float, call: float, q: float = 0.0
) -> float:
    """P = C + K e^{-rT} - S e^{-qT}"""
    return float(call + K * np.exp(-r * T) - S * np.exp(-q * T))


def synth_call_from_put(
    S: float, K: float, r: float, T: float, put: float, q: float = 0.0
) -> float:
    """C = P + S e^{-qT} - K e^{-rT}"""
    return float(put + S * np.exp(-q * T) - K * np.exp(-r * T))


def pcp_residual(
    S: float,
    K: float,
    r: float,
    T: float,
    call: float,
    put: float,
    q: float = 0.0,
) -> float:
    """
    Residual of parity: C + K e^{-rT} - (P + S e^{-qT})
    - Zero means exact parity.
    - Positive residual suggests call side rich vs put side (given S, r, q).
    """
    return float(call + K * np.exp(-r * T) - (put + S * np.exp(-q * T)))


def pivot_calls_puts_by_strike(
    df: pd.DataFrame,
    *,
    price_col: str = "mid",
    type_col: str = "type",
    strike_col: str = "strike",
) -> pd.DataFrame:
    """
    Build a wide table indexed by strike with columns ['C','P'] from a long quote table.
    Rows require both legs to compute residuals.
    """
    if type_col not in df or strike_col not in df or price_col not in df:
        raise ValueError(
            "DataFrame must contain type, strike, and price columns"
        )

    d = df[[strike_col, type_col, price_col]].copy()
    d = d.dropna(subset=[price_col])
    wide = d.pivot_table(
        index=strike_col, columns=type_col, values=price_col, aggfunc="mean"
    )
    # Normalize column labels to 'C'/'P' if present
    cols = {c: c for c in wide.columns}
    if "call" in cols:
        cols["call"] = "C"
    if "put" in cols:
        cols["put"] = "P"
    wide = wide.rename(columns=cols)
    return (
        wide.dropna(subset=["C", "P"], how="any")
        if set(["C", "P"]).issubset(wide.columns)
        else pd.DataFrame(columns=["C", "P"])
    )


def add_pcp_diagnostics(
    df_quotes: pd.DataFrame,
    *,
    spot: float,
    r: float,
    T: float,
    q: float = 0.0,
    price_col: str = "mid",
    type_col: str = "type",
    strike_col: str = "strike",
) -> pd.DataFrame:
    """
    Return a small table indexed by strike with columns:
      ['C','P','residual','residual_abs','put_synth','call_synth']

    Useful to visualize violations and synthesize missing legs.
    """
    wide = pivot_calls_puts_by_strike(
        df_quotes,
        price_col=price_col,
        type_col=type_col,
        strike_col=strike_col,
    )
    if wide.empty:
        return wide

    disc_K = np.exp(-r * T)
    disc_S = np.exp(-q * T)

    residual = wide["C"] + wide.index * disc_K - (wide["P"] + spot * disc_S)
    put_synth = wide["C"] + wide.index * disc_K - spot * disc_S
    call_synth = wide["P"] + spot * disc_S - wide.index * disc_K

    out = pd.DataFrame(
        {
            "C": wide["C"],
            "P": wide["P"],
            "residual": residual,
            "residual_abs": residual.abs(),
            "put_synth": put_synth,
            "call_synth": call_synth,
        },
        index=wide.index,
    )
    return out.sort_index()
