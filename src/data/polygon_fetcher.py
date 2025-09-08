# src/data/polygon_fetcher.py
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

import httpx  # used only as a fallback for specific endpoints if SDK surface differs
from polygon import RESTClient  # <- official Polygon SDK

from .base import (  # normalized schema/protocol
    OptionChain,
    OptionFetcher,
    OptionQuote,
)

logger = logging.getLogger("data.polygon_fetcher")

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


def _to_utc_date(dt_str: str) -> datetime:
    # Polygon returns 'YYYY-MM-DD'
    return datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)


class PolygonFetcher(OptionFetcher):
    """
    U.S. equity options backend using the official Polygon Python SDK (RESTClient).

    Primary operations:
      - list_expiries(): via client.list_options_contracts(...) (SDK iterator)
      - fetch_chain():   via chain snapshot endpoint (SDK call if available),
                         else fallback to raw GET /v3/snapshot/options/{underlying}

    Notes:
      - Methods on the SDK can differ slightly by version. We try several method names
        for the chain snapshot and fallback to HTTP if none are present.
      - All public methods are async to match the OptionFetcher protocol; SDK calls
        are executed in a worker thread via asyncio.to_thread.
    """

    def __init__(self, timeout: float = 20.0):
        if not POLYGON_API_KEY:
            raise RuntimeError("POLYGON_API_KEY is not set")
        # RESTClient is sync; we’ll wrap calls using asyncio.to_thread for async API.
        self.client = RESTClient(
            POLYGON_API_KEY, connect_timeout=timeout, read_timeout=timeout
        )
        self._timeout = timeout
        logger.info(
            f"PolygonFetcher initialized with RESTClient (base: {POLYGON_BASE}, timeout: {timeout}s)"
        )
        # Fallback async HTTP client only if we must hit an endpoint not exposed by current SDK
        self._http_fallback = httpx.AsyncClient(timeout=timeout)

    # --------------- internal helpers ---------------

    async def _sdk_iter(
        self, fn: Callable[..., Iterable[Any]], **kwargs
    ) -> List[Any]:
        """Call an SDK iterator method (like list_options_contracts) in a thread; collect results."""

        def _collect():
            return list(fn(**kwargs))

        return await asyncio.to_thread(_collect)

    async def _sdk_call(self, fn: Callable[..., Any], **kwargs) -> Any:
        """Call an SDK function in a thread and return result."""
        return await asyncio.to_thread(fn, **kwargs)

    async def _fallback_get_json(
        self, url: str, params: Optional[Dict] = None
    ) -> Dict:
        """Raw GET against Polygon REST as a last resort (keeps behavior compatible across SDK versions)."""
        p = dict(params or {})
        if "apiKey" not in p:
            p["apiKey"] = POLYGON_API_KEY
        r = await self._http_fallback.get(url, params=p)
        r.raise_for_status()
        return r.json()

    # --------------- public API ---------------

    async def list_expiries(self, underlying: str) -> List[datetime]:
        """
        Return sorted, tz-aware expiry datetimes for the given underlying.

        Preferred: SDK iterator list_options_contracts(underlying_ticker=..., limit=...)
        Fallback:  raw GET /v3/reference/options/contracts
        """
        logger.info(f"[Polygon] list_expiries for {underlying}")

        # Try SDK path first
        try:
            items = await self._sdk_iter(
                self.client.list_options_contracts,
                underlying_ticker=underlying,
                limit=1000,
            )
            expiries: set[datetime] = set()
            for c in items:
                # SDK models generally expose 'expiration_date'
                exp = getattr(c, "expiration_date", None) or getattr(
                    c, "expirationDate", None
                )
                if not exp:
                    continue
                try:
                    expiries.add(_to_utc_date(str(exp)))
                except Exception as pe:
                    logger.debug(f"Failed to parse expiration '{exp}': {pe}")

            out = sorted(expiries)
            logger.info(
                f"[Polygon SDK] list_expiries: found {len(out)} expiries for {underlying}: "
                f"{[e.date() for e in out[:5]]}{'...' if len(out) > 5 else ''}"
            )
            if out:
                return out

        except Exception as e:
            logger.warning(
                f"[Polygon SDK] list_expiries failed, falling back to REST: {e}"
            )

        # Fallback: raw REST (mirrors your previous implementation)
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
                data = await self._fallback_get_json(next_url, next_params)
            except Exception as e:
                logger.error(f"[Polygon REST] list_expiries failed: {e}")
                raise

            results = data.get("results", []) or []
            for c in results:
                exp = c.get("expiration_date")
                if not exp:
                    continue
                try:
                    expiries.add(_to_utc_date(exp))
                except Exception as pe:
                    logger.debug(f"Failed to parse expiration '{exp}': {pe}")

            next_url = data.get("next_url")
            next_params = {"apiKey": POLYGON_API_KEY} if next_url else None
            await asyncio.sleep(0.02)  # be polite

        exps = sorted(expiries)
        logger.info(
            f"[Polygon REST] list_expiries: found {len(exps)} expiries for {underlying}: "
            f"{[e.date() for e in exps[:5]]}{'...' if len(exps) > 5 else ''}"
        )
        return exps

    async def fetch_chain(
        self, underlying: str, expiry: datetime
    ) -> OptionChain:
        """
        Fetch calls/puts for one expiry and return a normalized OptionChain.

        Preferred: SDK chain snapshot (method name can vary by SDK version).
                   Try, in order:
                     - get_options_chain
                     - get_options_chain_snapshot
                     - get_snapshot_options_chain
        Fallback:  raw GET /v3/snapshot/options/{underlying}
        """
        if not isinstance(expiry, datetime):
            raise TypeError("expiry must be a datetime with tzinfo=UTC")

        expiry_date = expiry.date().isoformat()
        logger.info(f"[Polygon] fetch_chain({underlying}, {expiry_date})")

        # ---------- Try SDK snapshot first ----------
        sdk_methods = [
            "get_options_chain",
            "get_options_chain_snapshot",
            "get_snapshot_options_chain",
        ]
        snapshot = None
        sdk_used = None
        for name in sdk_methods:
            fn = getattr(self.client, name, None)
            if fn is None:
                continue
            try:
                snapshot = await self._sdk_call(fn, underlying=underlying)
                sdk_used = name
                break
            except Exception as e:
                logger.debug(f"[Polygon SDK] {name} failed: {e}")
                snapshot = None

        quotes: List[OptionQuote] = []
        spot = None

        if snapshot is not None:
            # SDK object structure may vary slightly by version — access defensively
            try:
                ua = (
                    getattr(snapshot, "underlying_asset", None)
                    or getattr(snapshot, "underlyingAsset", None)
                    or {}
                )
                # try direct 'price', else nested last_trade.p
                spot = _to_float(getattr(ua, "price", None))
                if spot is None:
                    last_trade = getattr(ua, "last_trade", None) or getattr(
                        ua, "lastTrade", None
                    )
                    if last_trade is not None:
                        spot = _to_float(
                            getattr(last_trade, "p", None)
                            or getattr(last_trade, "price", None)
                        )
            except Exception:
                pass

            results = getattr(snapshot, "results", None) or []
            kept = 0
            for c in results:
                try:
                    details = getattr(c, "details", None) or {}
                    # Most SDKs model 'details' as an object; fallback to dict-like access
                    o_ticker = (
                        getattr(details, "contract", None)
                        or getattr(details, "ticker", None)
                        or getattr(c, "ticker", None)
                        or (
                            details.get("contract")
                            if isinstance(details, dict)
                            else None
                        )
                        or (
                            details.get("ticker")
                            if isinstance(details, dict)
                            else None
                        )
                    )
                    opt_type_raw = (
                        getattr(details, "contract_type", None)
                        or getattr(details, "type", None)
                        or (
                            details.get("contract_type")
                            if isinstance(details, dict)
                            else None
                        )
                        or (
                            details.get("type")
                            if isinstance(details, dict)
                            else None
                        )
                    )
                    exp_str = (
                        getattr(details, "expiration_date", None)
                        or (
                            details.get("expiration_date")
                            if isinstance(details, dict)
                            else None
                        )
                        or ""
                    )

                    if not exp_str or str(exp_str).strip() != expiry_date:
                        continue  # filter other expirations

                    strike = getattr(details, "strike_price", None) or (
                        details.get("strike_price")
                        if isinstance(details, dict)
                        else None
                    )

                    last_q = (
                        getattr(c, "last_quote", None)
                        or getattr(c, "lastQuote", None)
                        or {}
                    )
                    last_t = (
                        getattr(c, "last_trade", None)
                        or getattr(c, "lastTrade", None)
                        or {}
                    )

                    bid = _to_float(
                        getattr(last_q, "bid", None)
                        or (
                            last_q.get("bid")
                            if isinstance(last_q, dict)
                            else None
                        )
                    )
                    ask = _to_float(
                        getattr(last_q, "ask", None)
                        or (
                            last_q.get("ask")
                            if isinstance(last_q, dict)
                            else None
                        )
                    )
                    last = _to_float(
                        getattr(last_t, "price", None)
                        or getattr(last_t, "p", None)
                        or (
                            last_t.get("price")
                            if isinstance(last_t, dict)
                            else None
                        )
                        or (
                            last_t.get("p")
                            if isinstance(last_t, dict)
                            else None
                        )
                    )

                    oi_obj = (
                        getattr(c, "open_interest", None)
                        or getattr(c, "openInterest", None)
                        or {}
                    )
                    iv_obj = (
                        getattr(c, "implied_volatility", None)
                        or getattr(c, "impliedVolatility", None)
                        or {}
                    )

                    oi = _to_float(
                        getattr(oi_obj, "oi", None)
                        or (
                            oi_obj.get("oi")
                            if isinstance(oi_obj, dict)
                            else None
                        )
                    )
                    iv = _to_float(
                        getattr(iv_obj, "iv", None)
                        or (
                            iv_obj.get("iv")
                            if isinstance(iv_obj, dict)
                            else None
                        )
                    )

                    typ = str(opt_type_raw or "").strip().upper()
                    opt_type = "C" if typ.startswith("C") else "P"

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
                            mark=None,  # Polygon snapshot doesn't guarantee mark
                            volume=None,  # could be retrieved via a different endpoint
                            open_interest=oi,
                            contract_size=100.0,  # U.S. equity options
                            underlying_ccy="USD",
                            quote_ccy="USD",
                            is_inverse=False,
                            extra={"iv": iv} if iv is not None else {},
                        )
                    )
                    kept += 1
                except Exception as e:
                    logger.debug(
                        f"[Polygon SDK] skipping one contract due to parse error: {e}"
                    )

            logger.info(
                f"[Polygon SDK:{sdk_used}] fetch_chain({underlying}, {expiry_date}): kept {kept} contracts; spot={spot}"
            )

            return OptionChain(
                underlying=underlying,
                asset_class="equity",
                spot=spot,
                asof_utc=datetime.utcnow().replace(tzinfo=timezone.utc),
                quotes=quotes,
            )

        # ---------- Fallback to raw REST ----------
        url = f"{POLYGON_BASE}/v3/snapshot/options/{underlying}"
        try:
            data = await self._fallback_get_json(
                url, {"apiKey": POLYGON_API_KEY}
            )
        except Exception as e:
            logger.error(
                f"[Polygon REST] chain snapshot failed for {underlying}: {e}"
            )
            raise

        try:
            ua = data.get("underlying_asset") or {}
            spot = _to_float(ua.get("price")) or _to_float(
                (ua.get("last_trade") or {}).get("p")
            )
        except Exception:
            spot = None

        contracts = data.get("results", []) or []
        kept = 0
        for c in contracts:
            try:
                details = c.get("details") or {}
                o_ticker = (
                    details.get("contract")
                    or details.get("ticker")
                    or c.get("ticker")
                )
                opt_type_raw = details.get("type") or details.get(
                    "contract_type"
                )
                exp_str = (details.get("expiration_date") or "").strip()
                if not exp_str or exp_str != expiry_date:
                    continue
                strike = details.get("strike_price")
                last_q = c.get("last_quote") or {}
                last_t = c.get("last_trade") or {}

                bid = _to_float(last_q.get("bid"))
                ask = _to_float(last_q.get("ask"))
                last = _to_float(last_t.get("price") or last_t.get("p"))
                oi = _to_float((c.get("open_interest") or {}).get("oi"))
                iv = _to_float((c.get("implied_volatility") or {}).get("iv"))

                typ = str(opt_type_raw or "").strip().upper()
                opt_type = "C" if typ.startswith("C") else "P"

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
                        mark=None,
                        volume=None,
                        open_interest=oi,
                        contract_size=100.0,
                        underlying_ccy="USD",
                        quote_ccy="USD",
                        is_inverse=False,
                        extra={"iv": iv} if iv is not None else {},
                    )
                )
                kept += 1
            except Exception as e:
                logger.debug(
                    f"[Polygon REST] skipping one contract due to parse error: {e}"
                )

        logger.info(
            f"[Polygon REST] fetch_chain({underlying}, {expiry_date}): kept {kept} contracts; spot={spot}"
        )

        return OptionChain(
            underlying=underlying,
            asset_class="equity",
            spot=spot,
            asof_utc=datetime.utcnow().replace(tzinfo=timezone.utc),
            quotes=quotes,
        )

    async def aclose(self):
        """Close the fallback HTTP client (RESTClient is sync; no close needed)."""
        try:
            await self._http_fallback.aclose()
        except Exception:
            pass
