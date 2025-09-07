# src/data/registry.py
from __future__ import annotations

"""
Factory for returning a concrete data backend behind the OptionFetcher protocol.

Why a registry:
- Keeps app code simple: 'get_fetcher("equity")' or 'get_fetcher("crypto")'
  instead of importing backend modules everywhere.
- Encapsulates backend selection and any construction defaults in one place.
"""

from typing import Literal

from .base import OptionFetcher
from .okx_fetcher import OKXFetcher
from .yf_fetcher import YFinanceFetcher


def get_fetcher(
    asset_class: Literal["equity", "crypto"], **kwargs
) -> OptionFetcher:
    """
    Return a concrete fetcher.

    kwargs are forwarded to the underlying fetcher constructors so callers can
    override defaults (timeouts, currency labels, etc.).
    """
    if asset_class == "equity":
        return YFinanceFetcher(**kwargs)
    if asset_class == "crypto":
        return OKXFetcher(**kwargs)
    raise ValueError(f"Unsupported asset_class: {asset_class}")
