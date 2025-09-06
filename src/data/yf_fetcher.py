# src/data/yf_fetcher.py
from __future__ import annotations

"""
Equity backend using yfinance.

Key quirks handled:
- yfinance.Ticker.option_chain(expiry) returns a single object with .calls/.puts,
  not a 2-tuple. We access attributes explicitly.
- Spot retrieval: 'fast_info' (cheap) -> 'info' -> 'history' fallback.

Threading:
- yfinance is sync. We push its calls into a thread via asyncio.to_thread so
  the rest of our code can be async.
"""

import asyncio
import math
from datetime import datetime, timezone
from typing import List, Optional

import yfinance as yf
from .base import OptionFetcher, OptionChain, OptionQuote


def _nan_to_none(x):
    """Coerce NaN to None; keep floats as floats."""
    try:
        return (
            None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
        )
    except Exception:
        return None


class YFinanceFetcher(OptionFetcher):
    """Simple equity fetcher that returns normalized OptionChain objects."""

    def __init__(self, contract_size: float = 100.0):
        # U.S. equity options typically have a 100x multiplier
        self.contract_size = contract_size

    async def list_expiries(self, underlying: str) -> List[datetime]:
        """Return tz-aware expiry datetimes for the given underlying."""

        def _get():
            return yf.Ticker(underlying).options  # list[str] like "2025-12-19"

        dates = await asyncio.to_thread(_get)
        return [datetime.fromisoformat(d).replace(tzinfo=timezone.utc) for d in dates]

    async def fetch_chain(self, underlying: str, expiry: datetime) -> OptionChain:
        """
        Fetch calls/puts for one expiry and return a normalized OptionChain.

        IV handling:
          - yfinance provides impliedVolatility per row; we pass it via quote.extra["iv"].
          - 'mark' is not guaranteed; downstream can compute mids from bid/ask.
        """
        exp_str = expiry.date().isoformat()

        def _get():
            t = yf.Ticker(underlying)

            # Spot: fast_info -> info -> history fallback
            spot = None
            try:
                fi = getattr(t, "fast_info", None)
                # In newer yfinance versions fast_info behaves like a mapping
                if fi and "last_price" in fi and fi["last_price"] is not None:
                    spot = float(fi["last_price"])
            except Exception:
                pass
            if spot is None:
                try:
                    spot = t.info.get("regularMarketPrice")
                except Exception:
                    spot = None
            if spot is None:
                hist = t.history(period="1d")
                if not hist.empty:
                    spot = float(hist["Close"].iloc[-1])

            # IMPORTANT: option_chain returns an object with .calls and .puts
            chain_obj = t.option_chain(exp_str)
            calls_df = chain_obj.calls
            puts_df = chain_obj.puts
            return float(spot) if spot is not None else None, calls_df, puts_df

        spot, calls, puts = await asyncio.to_thread(_get)
        quotes: list[OptionQuote] = []

        # Build normalized quotes from calls
        for _, r in calls.iterrows():
            quotes.append(
                OptionQuote(
                    symbol=str(r.get("contractSymbol")),
                    underlying=underlying,
                    asset_class="equity",
                    expiry=expiry,
                    strike=float(r["strike"]),
                    opt_type="C",
                    bid=_nan_to_none(r.get("bid")),
                    ask=_nan_to_none(r.get("ask")),
                    last=_nan_to_none(r.get("lastPrice")),
                    mark=None,
                    volume=_nan_to_none(r.get("volume")),
                    open_interest=_nan_to_none(r.get("openInterest")),
                    contract_size=self.contract_size,
                    underlying_ccy="USD",
                    quote_ccy="USD",
                    is_inverse=False,
                    extra={"iv": _nan_to_none(r.get("impliedVolatility"))},
                )
            )

        # ...and from puts
        for _, r in puts.iterrows():
            quotes.append(
                OptionQuote(
                    symbol=str(r.get("contractSymbol")),
                    underlying=underlying,
                    asset_class="equity",
                    expiry=expiry,
                    strike=float(r["strike"]),
                    opt_type="P",
                    bid=_nan_to_none(r.get("bid")),
                    ask=_nan_to_none(r.get("ask")),
                    last=_nan_to_none(r.get("lastPrice")),
                    mark=None,
                    volume=_nan_to_none(r.get("volume")),
                    open_interest=_nan_to_none(r.get("openInterest")),
                    contract_size=self.contract_size,
                    underlying_ccy="USD",
                    quote_ccy="USD",
                    is_inverse=False,
                    extra={"iv": _nan_to_none(r.get("impliedVolatility"))},
                )
            )

        return OptionChain(
            underlying=underlying,
            asset_class="equity",
            spot=spot,
            asof_utc=datetime.utcnow().replace(tzinfo=timezone.utc),
            quotes=quotes,
        )
