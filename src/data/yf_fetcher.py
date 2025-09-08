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

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import List

import yfinance as yf

from .base import OptionChain, OptionFetcher, OptionQuote

logger = logging.getLogger("data.yf_fetcher")


def _nan_to_none(x):
    """Coerce NaN to None; keep floats as floats."""
    try:
        return (
            None
            if x is None or (isinstance(x, float) and math.isnan(x))
            else float(x)
        )
    except Exception:
        return None


class YFinanceFetcher(OptionFetcher):
    """Simple equity fetcher that returns normalized OptionChain objects."""

    def __init__(self, contract_size: float = 100.0):
        # U.S. equity options typically have a 100x multiplier
        self.contract_size = contract_size
        logger.info(
            f"YFinanceFetcher initialized with contract_size={contract_size}"
        )

    async def list_expiries(self, underlying: str) -> List[datetime]:
        """Return tz-aware expiry datetimes for the given underlying."""
        logger.info(f"Listing expiries for {underlying}")

        def _get():
            ticker = yf.Ticker(underlying)
            return ticker.options  # list[str] like "2025-12-19"

        try:
            dates = await asyncio.to_thread(_get)
            logger.debug(f"yfinance returned {len(dates)} expiry strings")

            parsed_dates = []
            for d in dates:
                try:
                    dt = datetime.fromisoformat(d).replace(tzinfo=timezone.utc)
                    parsed_dates.append(dt)
                except Exception as e:
                    logger.warning(f"Failed to parse expiry date '{d}': {e}")

            logger.info(
                f"Found {len(parsed_dates)} valid expiries for {underlying}: "
                f"{[d.date() for d in parsed_dates[:5]]}"
                f"{'...' if len(parsed_dates) > 5 else ''}"
            )
            return parsed_dates

        except Exception as e:
            logger.error(f"Failed to fetch expiries for {underlying}: {e}")
            raise

    async def fetch_chain(
        self, underlying: str, expiry: datetime
    ) -> OptionChain:
        """
        Fetch calls/puts for one expiry and return a normalized OptionChain.

        IV handling:
          - yfinance provides impliedVolatility per row; we pass it via quote.extra["iv"].
          - 'mark' is not guaranteed; downstream can compute mids from bid/ask.
        """
        logger.info(
            f"Fetching option chain for {underlying} expiry {expiry.date()}"
        )
        exp_str = expiry.date().isoformat()

        def _get():
            t = yf.Ticker(underlying)
            logger.debug(f"Created yfinance Ticker for {underlying}")

            # Spot: fast_info -> info -> history fallback
            spot = None
            try:
                fi = getattr(t, "fast_info", None)
                # In newer yfinance versions fast_info behaves like a mapping
                if fi and "last_price" in fi and fi["last_price"] is not None:
                    spot = float(fi["last_price"])
                    logger.debug(f"Got spot from fast_info: {spot}")
            except Exception as e:
                logger.debug(f"fast_info failed: {e}")

            if spot is None:
                try:
                    info_data = t.info
                    spot = info_data.get("regularMarketPrice")
                    if spot:
                        logger.debug(f"Got spot from info: {spot}")
                except Exception as e:
                    logger.debug(f"info lookup failed: {e}")

            if spot is None:
                try:
                    logger.debug("Falling back to history for spot price")
                    hist = t.history(period="1d")
                    if not hist.empty:
                        spot = float(hist["Close"].iloc[-1])
                        logger.debug(f"Got spot from history: {spot}")
                except Exception as e:
                    logger.warning(f"History fallback failed: {e}")

            # IMPORTANT: option_chain returns an object with .calls and .puts
            logger.debug(f"Fetching option chain for expiry {exp_str}")
            chain_obj = t.option_chain(exp_str)
            calls_df = chain_obj.calls
            puts_df = chain_obj.puts

            logger.debug(
                f"Retrieved {len(calls_df)} calls, {len(puts_df)} puts"
            )
            return float(spot) if spot is not None else None, calls_df, puts_df

        try:
            spot, calls, puts = await asyncio.to_thread(_get)
            logger.info(
                f"Retrieved data: spot={spot}, {len(calls)} calls, {len(puts)} puts"
            )

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

            logger.debug(f"Created {len(quotes)} total option quotes")

            chain = OptionChain(
                underlying=underlying,
                asset_class="equity",
                spot=spot,
                asof_utc=datetime.utcnow().replace(tzinfo=timezone.utc),
                quotes=quotes,
            )

            logger.info(
                f"Created OptionChain for {underlying}: {len(quotes)} quotes, spot={spot}"
            )
            return chain

        except Exception as e:
            logger.error(
                f"Failed to fetch option chain for {underlying} {expiry.date()}: {e}"
            )
            raise
