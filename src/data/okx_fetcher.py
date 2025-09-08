"""
Crypto options backend using OKX public API, with optional private (signed) calls.

Highlights:
- **Public** endpoints (no credentials needed): instruments, per-instrument tickers, index price.
- **Private** (optional): GET /api/v5/account/balance with HMAC signing.
- Handles OKX's expiry token formats (YYMMDD and YYYYMMDD).
- Uses server time for signatures to avoid clock skew.
- Adds 'x-simulated-trading: 1' header if OKX_SIMULATED=true is set (demo accounts).

Important:
- Keep your API key/secret/passphrase in a .env (never commit secrets).
- For research, a read-only key is sufficient.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

from .base import OptionChain, OptionFetcher, OptionQuote

# Load .env for optional private credentials (OKX_API_KEY/SECRET/PASSPHRASE)
load_dotenv()

# Demo/simulated trading: some accounts require this header for private endpoints
SIM_ENV = (
    os.getenv("OKX_SIMULATED") or os.getenv("OKX_USE_TESTNET") or "false"
).lower() in ("1", "true", "yes")


# ------------------------ small helpers ------------------------


def _to_float(x):
    """Convert to float or None when unavailable/bad."""
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def _expiry_token_from_instid(inst_id: str) -> str:
    """
    Extract the expiry token from an OKX instrument id.

    Example OKX instId: 'BTC-USD-250906-30000-C'  -> token '250906'
    We accept YYYYMMDD in some corners as well, so parsing is robust downstream.
    """
    parts = inst_id.split("-")
    if len(parts) < 5:
        raise ValueError(f"Unexpected instId format: {inst_id}")
    return parts[2].strip()


def _parse_expiry_token(token: str) -> datetime:
    """
    Parse 'YYMMDD' or 'YYYYMMDD' and return a tz-aware UTC datetime at midnight.
    """
    token = token.strip()
    if len(token) == 6:
        dt = datetime.strptime(token, "%y%m%d")
    elif len(token) == 8:
        dt = datetime.strptime(token, "%Y%m%d")
    else:
        raise ValueError(f"Unrecognized expiry token: {token}")
    return dt.replace(tzinfo=timezone.utc)


def _okx_token_for(expiry_dt: datetime, prefer_short: bool = True) -> str:
    """Generate an OKX-style expiry token; OKX commonly uses YYMMDD."""
    return (
        expiry_dt.strftime("%y%m%d")
        if prefer_short
        else expiry_dt.strftime("%Y%m%d")
    )


# ------------------------ fetcher ------------------------


class OKXFetcher(OptionFetcher):
    """
    Public endpoints work without credentials.
    If API key/secret/passphrase are supplied (env or args), private endpoints like
    /api/v5/account/balance are available.

    Env vars (optional):
      OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE
      OKX_SIMULATED=true/false  (adds x-simulated-trading: 1 for demo accounts)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        underlying_ccy: str = "USD",
        quote_ccy: str = "USDT",
        timeout: float = 15.0,
    ):
        self.BASE = "https://www.okx.com"  # OKX uses one base URL; demo is an account flag

        # Credentials (optional; required only for private endpoints)
        self.api_key = api_key or os.getenv("OKX_API_KEY")
        self.api_secret_bytes = (
            api_secret or os.getenv("OKX_API_SECRET") or ""
        ).encode()
        self.passphrase = passphrase or os.getenv("OKX_API_PASSPHRASE")

        # Cosmetic labels for the normalized quotes
        self.underlying_ccy = underlying_ccy
        self.quote_ccy = quote_ccy

        # One async client for the fetcher lifetime
        self.client = httpx.AsyncClient(timeout=timeout)

    # -------- public: instruments / expiries / tickers --------

    async def list_expiries(self, underlying: str) -> List[datetime]:
        """
        Return a sorted list of expiry dates available for the given underlying (BTC, ETH, ...).
        """
        insts = await self._get_instruments(underlying)
        exps = sorted(
            {
                _parse_expiry_token(_expiry_token_from_instid(i["instId"]))
                for i in insts
            }
        )
        return exps

    async def fetch_chain(
        self, underlying: str, expiry: datetime
    ) -> OptionChain:
        """
        Fetch all options for a specific expiry and return a normalized OptionChain.

        Implementation note:
        - OKX tickers are per-instrument. We fetch each one sequentially with a small delay
          to keep usage polite. For higher throughput, wire in data/rate_limit.py helpers.
        """
        insts = await self._get_instruments(underlying)

        # Filter instruments by expiry token (OKX typically uses YYMMDD)
        target_token = _okx_token_for(expiry, prefer_short=True)
        insts = [
            i
            for i in insts
            if _expiry_token_from_instid(i["instId"]) == target_token
        ]

        # Fetch per-instrument tickers and assemble a map
        symbols = [i["instId"] for i in insts]
        tick_map = await self._get_tickers_map(symbols)

        quotes: List[OptionQuote] = []
        for i in insts:
            sym = i["instId"]
            strike, opt_type = self._parse_strike_type(sym)
            tk = tick_map.get(sym, {})
            quotes.append(
                OptionQuote(
                    symbol=sym,
                    underlying=underlying,
                    asset_class="crypto",
                    expiry=expiry.replace(tzinfo=timezone.utc),
                    strike=strike,
                    opt_type=opt_type,
                    bid=_to_float(tk.get("bidPx")),
                    ask=_to_float(tk.get("askPx")),
                    last=_to_float(tk.get("last")),
                    mark=_to_float(tk.get("markPx")),
                    volume=_to_float(tk.get("vol24h")),
                    open_interest=None,  # available via separate endpoints if needed
                    contract_size=1.0,  # OKX options multiplier is effectively 1 underlying unit
                    underlying_ccy=self.underlying_ccy,
                    quote_ccy=self.quote_ccy,
                    is_inverse=False,
                    extra={},
                )
            )

        spot = await self._get_spot(underlying)
        return OptionChain(
            underlying=underlying,
            asset_class="crypto",
            spot=spot,
            asof_utc=datetime.utcnow().replace(tzinfo=timezone.utc),
            quotes=quotes,
        )

    async def _get_instruments(self, underlying: str):
        """
        OKX instruments endpoint:
            GET /api/v5/public/instruments?instType=OPTION&uly=BTC-USD
        Response includes 'instId' strings like BTC-USD-250906-30000-C.
        """
        uly = f"{underlying}-USD"
        url = f"{self.BASE}/api/v5/public/instruments"
        r = await self.client.get(
            url, params={"instType": "OPTION", "uly": uly}
        )
        r.raise_for_status()
        return r.json().get("data", [])

    async def _get_tickers_map(self, inst_ids: List[str]) -> Dict[str, Dict]:
        """
        Build dict: instId -> ticker payload (bidPx, askPx, last, markPx, ...).

        Rate limit:
        - Intentionally slow down with a small sleep to avoid bursty behavior.
          For production, use AsyncRateLimiter + retry_with_backoff.
        """
        out: Dict[str, Dict] = {}
        for sym in inst_ids:
            r = await self.client.get(
                f"{self.BASE}/api/v5/market/ticker", params={"instId": sym}
            )
            if r.status_code == 200:
                d = r.json().get("data", [])
                if d:
                    out[sym] = d[0]
            await asyncio.sleep(0.02)  # be polite
        return out

    async def _get_spot(self, underlying: str) -> Optional[float]:
        """
        Use OKX index price (e.g., BTC-USD) as a spot proxy:
            GET /api/v5/market/index-tickers?instId=BTC-USD
        """
        r = await self.client.get(
            f"{self.BASE}/api/v5/market/index-tickers",
            params={"instId": f"{underlying}-USD"},
        )
        if r.status_code == 200 and r.json().get("data"):
            return _to_float(r.json()["data"][0].get("idxPx"))
        return None

    def _parse_strike_type(self, inst_id: str) -> Tuple[float, str]:
        """BTC-USD-250906-30000-C -> (30000.0, 'C')"""
        parts = inst_id.split("-")
        if len(parts) < 5:
            raise ValueError(f"Unexpected instId format: {inst_id}")
        strike = float(parts[3])
        typ = parts[4].upper()
        return strike, ("C" if typ == "C" else "P")

    # ---------------- private: signing + server-time ----------------

    def _has_creds(self) -> bool:
        """True if all required credentials are present."""
        return bool(self.api_key and self.api_secret_bytes and self.passphrase)

    async def _get_server_time_iso(self) -> str:
        """
        Query server time for signatures to avoid clock-skew errors.
        Fallback to local UTC if the time endpoint fails.
        """
        try:
            r = await self.client.get(
                f"{self.BASE}/api/v5/public/time", timeout=5.0
            )
            r.raise_for_status()
            # Payload example: {"code":"0","data":[{"ts":"1725600000000"}],"msg":""}
            ts_ms = int(r.json()["data"][0]["ts"])
            dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        except Exception:
            return (
                datetime.utcnow()
                .replace(tzinfo=timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )

    def _sign(self, ts: str, method: str, path: str, body: str = "") -> str:
        """
        OKX signature:
            sign = base64( HMAC_SHA256( secret, ts + method + path + body ) )
        - 'path' must include the query string when present.
        - 'method' is uppercased.
        """
        msg = (ts + method.upper() + path + body).encode()
        sig = hmac.new(self.api_secret_bytes, msg, sha256).digest()
        return base64.b64encode(sig).decode()

    def _auth_headers(self, ts: str, sign: str) -> Dict[str, str]:
        """Construct OKX auth headers, adding simulated flag if requested."""
        headers = {
            "OK-ACCESS-KEY": self.api_key or "",
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self.passphrase or "",
            "Content-Type": "application/json",
        }
        if SIM_ENV:
            headers["x-simulated-trading"] = "1"
        return headers

    async def _private_get(
        self, path: str, params: Optional[Dict] = None
    ) -> Dict:
        """
        Signed GET request. Raises RuntimeError if credentials are missing and prints
        the error body for actionable debugging on HTTP errors.
        """
        if not self._has_creds():
            raise RuntimeError(
                "OKX credentials not set. Provide env vars or pass into OKXFetcher()."
            )
        ts = await self._get_server_time_iso()
        query = ""
        if params:
            qs = httpx.QueryParams(params)
            query = f"?{qs}"
        sign = self._sign(ts, "GET", path + query, "")
        headers = self._auth_headers(ts, sign)
        url = f"{self.BASE}{path}{query}"
        r = await self.client.get(url, headers=headers)
        if r.status_code >= 400:
            print("OKX error body:", r.text)
        r.raise_for_status()
        return r.json()

    async def _private_post(self, path: str, payload: Dict) -> Dict:
        """Signed POST request (not currently used in this project)."""
        if not self._has_creds():
            raise RuntimeError(
                "OKX credentials not set. Provide env vars or pass into OKXFetcher()."
            )
        ts = await self._get_server_time_iso()
        body = json.dumps(payload)
        sign = self._sign(ts, "POST", path, body)
        headers = self._auth_headers(ts, sign)
        url = f"{self.BASE}{path}"
        r = await self.client.post(url, headers=headers, content=body)
        if r.status_code >= 400:
            print("OKX error body:", r.text)
        r.raise_for_status()
        return r.json()

    # Example private read-only call (requires Account:Read permission)
    async def get_balance(self, ccy: Optional[str] = None) -> Dict:
        """GET /api/v5/account/balance (optionally filter by 'ccy')."""
        params = {"ccy": ccy} if ccy else None
        return await self._private_get(
            "/api/v5/account/balance", params=params
        )

    async def aclose(self):
        """Close the underlying HTTP client (good hygiene for long tests)."""
        await self.client.aclose()
