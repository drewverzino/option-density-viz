# src/data/__init__.py
"""
Option Viz — data package

Convenience imports and documentation so end users can write:

    from data import get_fetcher, OKXFetcher, YFinanceFetcher, OptionChain, OptionQuote

Backends:
- Equities (yfinance)
- Crypto (OKX public; optional private endpoints via .env)

Optional env (for OKX private):
    OKX_API_KEY=...
    OKX_API_SECRET=...
    OKX_API_PASSPHRASE=...
    OKX_SIMULATED=true
"""

from .base import OptionChain, OptionFetcher, OptionQuote
from .registry import get_fetcher

# Import backends defensively so partial checkouts/tests don't break imports
try:
    from .yf_fetcher import YFinanceFetcher
except Exception:
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
