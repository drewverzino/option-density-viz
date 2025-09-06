"""
Option Viz — data package

Backends and types for fetching normalized option chains from
- Equities (yfinance)
- Crypto (OKX; public by default, optional private endpoints via .env)

Quick start:
    from data import get_fetcher
    f = get_fetcher("equity")   # or "crypto"
    exps = await f.list_expiries("AAPL")
    chain = await f.fetch_chain("AAPL", exps[0])

Env (for OKX private endpoints — optional):
    OKX_API_KEY=...
    OKX_API_SECRET=...
    OKX_API_PASSPHRASE=...
    OKX_SIMULATED=true  # if using demo/sim trading
"""

from .base import OptionQuote, OptionChain, OptionFetcher
from .registry import get_fetcher

# Optional, only if these files exist in your repo
try:
    from .yf_fetcher import YFinanceFetcher
except Exception:  # keep imports resilient while you iterate
    YFinanceFetcher = None  # type: ignore

try:
    from .okx_fetcher import OKXFetcher
except Exception:
    OKXFetcher = None  # type: ignore

__all__ = [
    "OptionQuote",
    "OptionChain",
    "OptionFetcher",
    "get_fetcher",
    "YFinanceFetcher",
    "OKXFetcher",
]

__version__ = "0.1.0"
