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

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# -------------------------- synthesis formulas -------------------------- #


def synth_put_from_call(
    S: float, K: float, r: float, T: float, call: float, q: float = 0.0
) -> float:
    """P = C + K e^{-rT} - S e^{-qT}"""
    result = float(call + K * np.exp(-r * T) - S * np.exp(-q * T))
    logger.debug(
        f"Synthetic put: C={call:.4f} + K={K:.2f}*e^(-{r:.4f}*{T:.4f}) "
        f"- S={S:.4f}*e^(-{q:.4f}*{T:.4f}) = {result:.4f}"
    )
    return result


def synth_call_from_put(
    S: float, K: float, r: float, T: float, put: float, q: float = 0.0
) -> float:
    """C = P + S e^{-qT} - K e^{-rT}"""
    result = float(put + S * np.exp(-q * T) - K * np.exp(-r * T))
    logger.debug(
        f"Synthetic call: P={put:.4f} + S={S:.4f}*e^(-{q:.4f}*{T:.4f}) "
        f"- K={K:.2f}*e^(-{r:.4f}*{T:.4f}) = {result:.4f}"
    )
    return result


# ---------------------------- residual checks --------------------------- #


def residuals_from_parity(
    S: float,
    K: float,
    r: float,
    T: float,
    call: float,
    put: float,
    q: float = 0.0,
) -> float:
    """
    PCP residual:
        res = C + K e^{-rT} - (P + S e^{-qT})

    - res = 0 : exact parity.
    - res > 0 : call side rich vs put side (given S, r, q).
    """
    result = float(call + K * np.exp(-r * T) - (put + S * np.exp(-q * T)))
    logger.debug(
        f"PCP residual: K={K:.2f}, C={call:.4f}, P={put:.4f} -> residual={result:.6f}"
    )
    return result


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
    return residuals_from_parity(S, K, r, T, call, put, q)


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
    logger.debug(
        f"Pivoting {len(df)} quotes by strike using columns: "
        f"{price_col}, {type_col}, {strike_col}"
    )

    if type_col not in df or strike_col not in df or price_col not in df:
        logger.error(
            f"Missing required columns. Available: {list(df.columns)}"
        )
        raise ValueError(
            "DataFrame must contain type, strike, and price columns"
        )

    d = df[[strike_col, type_col, price_col]].copy()
    d = d.dropna(subset=[price_col])

    n_before = len(d)
    logger.debug(f"After dropping NaN prices: {len(d)}/{n_before} quotes")

    wide = d.pivot_table(
        index=strike_col, columns=type_col, values=price_col, aggfunc="mean"
    )

    logger.debug(f"Pivot result columns: {list(wide.columns)}")

    # Normalize column labels to 'C'/'P' if present
    cols = {c: c for c in wide.columns}
    if "call" in cols:
        cols["call"] = "C"
    if "put" in cols:
        cols["put"] = "P"
    wide = wide.rename(columns=cols)

    # Filter to strikes with both C and P
    if set(["C", "P"]).issubset(wide.columns):
        before_filter = len(wide)
        wide = wide.dropna(subset=["C", "P"], how="any")
        logger.info(
            f"Found {len(wide)} strikes with both calls and puts "
            f"(filtered from {before_filter} total strikes)"
        )
    else:
        logger.warning(
            f"Missing C or P columns after pivot. Available: {list(wide.columns)}"
        )
        wide = pd.DataFrame(columns=["C", "P"])

    return wide


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
    logger.info(f"Computing PCP diagnostics for {len(df_quotes)} quotes")
    logger.debug(
        f"Parameters: spot={spot:.4f}, r={r:.4f}, T={T:.4f}, q={q:.4f}"
    )

    wide = pivot_calls_puts_by_strike(
        df_quotes,
        price_col=price_col,
        type_col=type_col,
        strike_col=strike_col,
    )

    if wide.empty:
        logger.warning("No call/put pairs found for PCP diagnostics")
        return wide

    disc_K = np.exp(-r * T)
    disc_S = np.exp(-q * T)

    logger.debug(f"Discount factors: K={disc_K:.6f}, S={disc_S:.6f}")

    residual = wide["C"] + wide.index * disc_K - (wide["P"] + spot * disc_S)
    put_synth = wide["C"] + wide.index * disc_K - spot * disc_S
    call_synth = wide["P"] + spot * disc_S - wide.index * disc_K

    # Log residual statistics
    abs_residuals = residual.abs()
    logger.info(
        f"PCP residuals: mean={residual.mean():.6f}, "
        f"std={residual.std():.6f}, max_abs={abs_residuals.max():.6f}"
    )

    # Count significant violations
    large_residuals = abs_residuals > 0.01  # 1 cent threshold
    if large_residuals.any():
        n_violations = large_residuals.sum()
        logger.warning(
            f"Found {n_violations} strikes with large PCP violations (>$0.01)"
        )
        if logger.isEnabledFor(logging.DEBUG):
            violation_strikes = wide.index[large_residuals]
            logger.debug(
                f"Violation strikes: {list(violation_strikes)[:5]}..."
            )

    out = pd.DataFrame(
        {
            "C": wide["C"],
            "P": wide["P"],
            "residual": residual,
            "residual_abs": abs_residuals,
            "put_synth": put_synth,
            "call_synth": call_synth,
        },
        index=wide.index,
    )

    logger.debug(f"PCP diagnostics table created: {len(out)} strikes")
    return out.sort_index()


# ------------------------ synth missing leg (API) ----------------------- #


def synthesize_missing_leg(
    S: float,
    K: float,
    r: float,
    T: float,
    *,
    call: float | None = None,
    put: float | None = None,
    q: float = 0.0,
) -> tuple[str, float]:
    """
    Synthesize the *missing* option leg via put–call parity.

    Exactly one of `call` or `put` must be provided:

      - If `call` is provided and `put` is None → compute and return ('P', P_syn).
      - If `put` is provided and `call` is None → compute and return ('C', C_syn).

    Parameters
    ----------
    S, K, r, T, q : floats
        Spot, strike, risk-free rate, year fraction, and dividend (or carry) yield.
    call, put : float or None
        Provided leg mid. Set the other to None.

    Returns
    -------
    (leg, value) : (str, float)
        'C' or 'P' and the synthesized mid price.

    Raises
    ------
    ValueError
        If both `call` and `put` are provided, or both are None.
    """
    logger.debug(
        f"Synthesizing missing leg: K={K:.2f}, call={call}, put={put}"
    )

    have_call = call is not None and np.isfinite(call)
    have_put = put is not None and np.isfinite(put)

    if have_call and have_put:
        logger.error("Both call and put provided - need exactly one")
        raise ValueError("Provide exactly one leg: call OR put (not both).")
    if not have_call and not have_put:
        logger.error("Neither call nor put provided - need exactly one")
        raise ValueError(
            "Provide exactly one leg: call OR put (neither given)."
        )

    if have_call:
        synth_value = synth_put_from_call(S, K, r, T, float(call), q=q)
        logger.info(
            f"Synthesized put from call: K={K:.2f}, C={call:.4f} -> P={synth_value:.4f}"
        )
        return "P", synth_value
    else:
        synth_value = synth_call_from_put(S, K, r, T, float(put), q=q)
        logger.info(
            f"Synthesized call from put: K={K:.2f}, P={put:.4f} -> C={synth_value:.4f}"
        )
        return "C", synth_value
