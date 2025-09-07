# src/preprocess/__init__.py
"""
Preprocessing utilities (quote cleaning, parity, forward/log-moneyness).

Public API
----------
- add_midprice_columns(df, ...): compute robust mids, rel. spreads, and flags
- synth_put_from_call(...), synth_call_from_put(...): PCP synthetic legs
- pcp_residual(...): parity residual (diagnostics)
- pivot_calls_puts_by_strike(df, ...): C/P wide table by strike
- add_pcp_diagnostics(df, ...): per-strike residuals + synthetic legs
- forward_price(...), log_moneyness(...)
- estimate_forward_from_chain(df, ...): forward from PCP pairs (robust)
"""

from .forward import (
    estimate_forward_from_chain,
    forward_price,
    log_moneyness,
)
from .midprice import add_midprice_columns
from .pcp import (
    add_pcp_diagnostics,
    pcp_residual,
    pivot_calls_puts_by_strike,
    synth_call_from_put,
    synth_put_from_call,
)

__all__ = [
    "add_midprice_columns",
    "synth_put_from_call",
    "synth_call_from_put",
    "pcp_residual",
    "pivot_calls_puts_by_strike",
    "add_pcp_diagnostics",
    "forward_price",
    "log_moneyness",
    "estimate_forward_from_chain",
]
