# src/data/registry.py
from __future__ import annotations

from typing import Literal

from .base import OptionFetcher
from .yf_fetcher import YFinanceFetcher
from .okx_fetcher import OKXFetcher


def get_fetcher(asset_class: Literal["equity", "crypto"], **kwargs) -> OptionFetcher:
    """
    Factory that returns a concrete fetcher behind the OptionFetcher protocol.
    kwargs are forwarded to the underlying fetcher constructors.
    """
    if asset_class == "equity":
        return YFinanceFetcher(**kwargs)
    if asset_class == "crypto":
        return OKXFetcher(**kwargs)
    raise ValueError(f"Unsupported asset_class: {asset_class}")
