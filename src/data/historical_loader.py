# src/data/historical_loader.py
from __future__ import annotations

"""
Helpers to save and load OptionChain objects to/from CSV/Parquet.

Why this file:
- Reproducibility: you can freeze a chain you fetched live and reload it later.
- CI/tests: use the same loader to build chains from synthetic CSVs.

Notes:
- CSV is universally available. Parquet is optional and needs 'pyarrow' or 'fastparquet'.
- We store a thin, human-readable schema that mirrors OptionQuote + some chain fields.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .base import OptionChain, OptionQuote


# ----------------- DataFrame conversion -----------------


def chain_to_dataframe(chain: OptionChain) -> pd.DataFrame:
    """
    Flatten an OptionChain into a tidy DataFrame.

    The DataFrame includes:
      - One row per OptionQuote
      - 'asof_utc' and 'spot' broadcast to all rows for convenience
    """
    rows = []
    for q in chain.quotes:
        rows.append(
            {
                "symbol": q.symbol,
                "underlying": q.underlying,
                "asset_class": q.asset_class,
                "expiry": q.expiry,  # keep tz info if present
                "strike": q.strike,
                "type": q.opt_type,
                "bid": q.bid,
                "ask": q.ask,
                "last": q.last,
                "mark": q.mark,
                "volume": q.volume,
                "open_interest": q.open_interest,
                "contract_size": q.contract_size,
                "underlying_ccy": q.underlying_ccy,
                "quote_ccy": q.quote_ccy,
                "is_inverse": q.is_inverse,
                "iv": (q.extra or {}).get(
                    "iv", None
                ),  # IV is optional and backend-dependent
            }
        )
    df = pd.DataFrame(rows)
    df["asof_utc"] = chain.asof_utc
    df["spot"] = chain.spot
    return df


def dataframe_to_chain(df: pd.DataFrame) -> OptionChain:
    """
    Rebuild an OptionChain from a DataFrame produced by chain_to_dataframe().

    Robustness:
    - Accepts 'expiry' either as timestamps or strings (we coerce to UTC).
    - Missing 'asof_utc' falls back to "now" so downstream code still works.
    """
    if df.empty:
        raise ValueError("Empty DataFrame cannot build a chain")

    # Ensure 'expiry' is a tz-aware datetime64 column
    if not pd.api.types.is_datetime64_any_dtype(df["expiry"]):
        df = df.copy()
        df["expiry"] = pd.to_datetime(df["expiry"], utc=True)

    first = df.iloc[0]
    quotes: List[OptionQuote] = []

    for _, r in df.iterrows():
        quotes.append(
            OptionQuote(
                symbol=str(r["symbol"]),
                underlying=str(r["underlying"]),
                asset_class=r["asset_class"],
                expiry=r["expiry"].to_pydatetime(),
                strike=float(r["strike"]),
                opt_type=str(r["type"]),
                bid=_nan_to_none(r.get("bid")),
                ask=_nan_to_none(r.get("ask")),
                last=_nan_to_none(r.get("last")),
                mark=_nan_to_none(r.get("mark")),
                volume=_nan_to_none(r.get("volume")),
                open_interest=_nan_to_none(r.get("open_interest")),
                contract_size=float(r["contract_size"]),
                underlying_ccy=str(r["underlying_ccy"]),
                quote_ccy=str(r["quote_ccy"]),
                is_inverse=bool(r["is_inverse"]),
                extra={"iv": _nan_to_none(r.get("iv"))},
            )
        )

    # asof_utc: allow missing then default to "now" to keep the chain usable
    asof_utc = first.get("asof_utc")
    if pd.isna(asof_utc):
        asof_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    else:
        asof_utc = pd.to_datetime(asof_utc, utc=True).to_pydatetime()

    return OptionChain(
        underlying=str(first["underlying"]),
        asset_class=first["asset_class"],
        spot=_nan_to_none(first.get("spot")),
        asof_utc=asof_utc,
        quotes=quotes,
    )


def _nan_to_none(x):
    """Coerce NaN/None to None; keep numeric types as float when possible."""
    import math

    try:
        return (
            None
            if x is None or (isinstance(x, float) and math.isnan(x))
            else (float(x) if isinstance(x, (int, float)) else x)
        )
    except Exception:
        return None


# ----------------- CSV / Parquet I/O -----------------


def save_chain_csv(path: str | Path, chain: OptionChain) -> None:
    """Write a chain to CSV; parents are created if missing."""
    df = chain_to_dataframe(chain)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_chain_csv(path: str | Path) -> OptionChain:
    """Load a chain from a CSV produced by save_chain_csv()."""
    df = pd.read_csv(path)
    return dataframe_to_chain(df)


def save_chain_parquet(path: str | Path, chain: OptionChain) -> None:
    """
    Write a chain to Parquet (columnar, compressed).
    Requires 'pyarrow' or 'fastparquet'.
    """
    df = chain_to_dataframe(chain)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        raise RuntimeError(
            "Parquet support requires 'pyarrow' or 'fastparquet'. Install one of them."
        ) from e


def load_chain_parquet(path: str | Path) -> OptionChain:
    """Load a chain from a Parquet file created by save_chain_parquet()."""
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(
            "Parquet support requires 'pyarrow' or 'fastparquet'. Install one of them."
        ) from e
    return dataframe_to_chain(df)


# ----------------- Batch helpers -----------------


def load_chains_csv(paths: Iterable[str | Path]) -> List[OptionChain]:
    """Load multiple CSV chains into memory."""
    return [load_chain_csv(p) for p in paths]


def load_chains_parquet(paths: Iterable[str | Path]) -> List[OptionChain]:
    """Load multiple Parquet chains into memory."""
    return [load_chain_parquet(p) for p in paths]
