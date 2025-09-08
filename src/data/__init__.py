"""
Data-layer exports (types, backends, caching, I/O, rates, throttling).
"""

from __future__ import annotations

from .base import OptionChain, OptionFetcher, OptionQuote
from .cache import KVCache
from .historical_loader import (
    chain_to_dataframe,
    dataframe_to_chain,
    load_chain_csv,
    load_chain_parquet,
    save_chain_csv,
    save_chain_parquet,
)
from .okx_fetcher import OKXFetcher  # crypto (public endpoints)
from .polygon_fetcher import PolygonFetcher  # equity/crypto via Polygon.io
from .rate_limit import AsyncRateLimiter, retry_with_backoff
from .registry import get_fetcher
from .risk_free import RiskFreeConfig, RiskFreeProvider
from .yf_fetcher import YFinanceFetcher  # equity

__all__ = [
    "OptionQuote",
    "OptionChain",
    "OptionFetcher",
    "get_fetcher",
    "YFinanceFetcher",
    "OKXFetcher",
    "KVCache",
    "chain_to_dataframe",
    "dataframe_to_chain",
    "save_chain_csv",
    "save_chain_parquet",
    "load_chain_csv",
    "load_chain_parquet",
    "RiskFreeProvider",
    "RiskFreeConfig",
    "AsyncRateLimiter",
    "retry_with_backoff",
    "PolygonFetcher",
]
