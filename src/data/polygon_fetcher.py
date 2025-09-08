# src/data/polygon_fetcher.py
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
import pandas as pd

from .base import (  # normalized schema/protocol
    OptionChain,
    OptionFetcher,
    OptionQuote,
)

logger = logging.getLogger(__name__)

# -------------------------------------------
# Configuration / auth
# -------------------------------------------
POLYGON_BASE = os.getenv("POLYGON_BASE_URL", "https://api.polygon.io")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()

if not POLYGON_API_KEY:
    logger.warning(
        "POLYGON_API_KEY is not set. Set it in your environment to enable PolygonFetcher."
    )


def _to_float(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None


class PolygonFetcher(OptionFetcher):
    """
    U.S. equity options backend via Polygon.io REST v3.

    Key endpoints we use:
      - List expiries:   GET /v3/reference/options/contracts?underlying_ticker={SYM}&limit=1000
                         (collect distinct 'expiration_date'; handle pagination via 'next_url')
      - Chain snapshot:  GET /v3/snapshot/options/{underlying}
                         (contains per-contract snapshots incl. latest quote, trade, greeks, IV, OI)
                         We filter client-side for the requested expiry.

    Notes:
      - Auth: add ?apiKey=... to each request.
      - Spot: the option chain snapshot response contains the underlying price.
      - Contract ticker field is usually like AAPL250920C00190000; we preserve it.

    Docs:
      - Option Chain Snapshot: https://polygon.io/docs/rest/options/snapshots/option-chain-snapshot
      - All Contracts (reference): https://polygon.io/docs/rest/options/contracts/all-contracts
    """

    def __init__(self, timeout: float = 20.0):
        self.client = httpx.AsyncClient(timeout=timeout)
        logger.info(
            f"PolygonFetcher initialized (base: {POLYGON_BASE}, timeout: {timeout}s)"
        )

    # --------------- public API ---------------

    async def list_expiries(self, underlying: str) -> List[datetime]:
        """
        Return sorted, tz-aware expiry datetimes for the given underlying.

        Implementation:
          - Walk /v3/reference/options/contracts?underlying_ticker=SYM
            and collect distinct 'expiration_date' (YYYY-MM-DD).
        """
        if not POLYGON_API_KEY:
            raise RuntimeError("POLYGON_API_KEY is not set")

        url = f"{POLYGON_BASE}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "limit": 1000,
            "apiKey": POLYGON_API_KEY,
        }

        expiries: set[datetime] = set()

        next_url: Optional[str] = url
        next_params: Optional[Dict] = params

        while next_url:
            try:
                r = await self.client.get(next_url, params=next_params)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                logger.error(f"Polygon list_expiries failed: {e}")
                raise

            results = data.get("results", []) or []
            for c in results:
                exp = c.get("expiration_date")  # 'YYYY-MM-DD'
                if not exp:
                    continue
                try:
                    dt = datetime.fromisoformat(exp).replace(
                        tzinfo=timezone.utc
                    )
                    expiries.add(dt)
                except Exception as pe:
                    logger.debug(f"Failed to parse expiration '{exp}': {pe}")

            # pagination: next_url is an absolute URL with apiKey included on Polygon
            next_url = data.get("next_url")
            # If next_url is present, Polygon expects apiKey as query param too
            next_params = {"apiKey": POLYGON_API_KEY} if next_url else None

            # be polite
            await asyncio.sleep(0.02)

        exps = sorted(expiries)
        logger.info(
            f"Polygon list_expiries: found {len(exps)} expiries for {underlying}: "
            f"{[e.date() for e in exps[:5]]}{'...' if len(exps) > 5 else ''}"
        )
        return exps

    async def fetch_chain(
        self, underlying: str, expiry: datetime
    ) -> OptionChain:
        """
        Fetch calls/puts for one expiry and return a normalized OptionChain.

        Implementation:
          - GET /v3/snapshot/options/{underlying}?apiKey=...
          - Filter per-contract snapshots on 'expiration_date' == expiry.date()
          - Map bid/ask/last/iv/oi into OptionQuote
          - Use 'underlying_asset' section for spot if provided
        """
        if not POLYGON_API_KEY:
            raise RuntimeError("POLYGON_API_KEY is not set")

        expiry_date = expiry.date().isoformat()
        url = f"{POLYGON_BASE}/v3/snapshot/options/{underlying}"
        params = {"apiKey": POLYGON_API_KEY}

        try:
            r = await self.client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.error(
                f"Polygon chain snapshot failed for {underlying}: {e}"
            )
            raise

        # Underlying price (if available)
        spot = None
        try:
            # Snapshot often includes an 'underlying_asset' (with 'price' or 'last_trade')
            ua = data.get("underlying_asset") or {}
            # Prefer a direct price if present; otherwise try nested location
            spot = _to_float(ua.get("price")) or _to_float(
                (ua.get("last_trade") or {}).get("p")
            )
        except Exception:
            pass

        # Per-contract snapshots
        contracts = data.get("results", []) or []

        quotes: List[OptionQuote] = []
        kept = 0

        for c in contracts:
            try:
                # Defensive field extraction; polygon payloads can evolve
                details = c.get("details") or {}
                o_ticker = (
                    details.get("contract")
                    or details.get("ticker")
                    or c.get("ticker")
                )
                opt_type_raw = details.get("type") or details.get(
                    "contract_type"
                )  # "call"/"put"
                exp_str = (
                    details.get("expiration_date") or ""
                ).strip()  # "YYYY-MM-DD"

                if not exp_str or exp_str != expiry_date:
                    continue  # skip other expirations

                # strike can appear in details or summary
                strike = details.get("strike_price")
                if strike is None:
                    strike = (c.get("break_even_price") or {}).get(
                        "strike"
                    )  # rare, fallback

                # last quote/trade blocks
                last_q = c.get("last_quote") or {}
                last_t = c.get("last_trade") or {}

                bid = _to_float(last_q.get("bid"))
                ask = _to_float(last_q.get("ask"))
                last = _to_float(last_t.get("price") or last_t.get("p"))
                mark = None  # polygon snapshot doesn't always carry a mark; leave None

                oi = _to_float((c.get("open_interest") or {}).get("oi"))
                iv = _to_float((c.get("implied_volatility") or {}).get("iv"))

                # Standardize type to "C"/"P"
                typ = str(opt_type_raw or "").strip().upper()
                opt_type = "C" if typ.startswith("C") else "P"

                # Construct OptionQuote
                quotes.append(
                    OptionQuote(
                        symbol=str(o_ticker or ""),
                        underlying=underlying,
                        asset_class="equity",
                        expiry=expiry.replace(tzinfo=timezone.utc),
                        strike=(
                            float(strike)
                            if strike is not None
                            else float("nan")
                        ),
                        opt_type=opt_type,
                        bid=bid,
                        ask=ask,
                        last=last,
                        mark=mark,
                        volume=None,  # available via other endpoints if needed
                        open_interest=oi,
                        contract_size=100.0,  # U.S. equity options multiplier
                        underlying_ccy="USD",
                        quote_ccy="USD",
                        is_inverse=False,
                        extra={"iv": iv} if iv is not None else {},
                    )
                )
                kept += 1
            except Exception as e:
                logger.debug(
                    f"Skipping one contract snapshot due to parse error: {e}"
                )

        logger.info(
            f"Polygon fetch_chain({underlying}, {expiry_date}): kept {kept} contracts; "
            f"spot={spot}"
        )

        chain = OptionChain(
            underlying=underlying,
            asset_class="equity",
            spot=spot,
            asof_utc=datetime.utcnow().replace(tzinfo=timezone.utc),
            quotes=quotes,
        )
        return chain

    async def aclose(self):
        """Close the underlying HTTP client (good hygiene for long tests)."""
        try:
            await self.client.aclose()
        except Exception:
            pass
