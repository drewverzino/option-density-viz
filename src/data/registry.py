"""
Factory for returning a concrete data backend behind the OptionFetcher protocol.

Why a registry:
- Keeps app code simple: 'get_fetcher("equity")' or 'get_fetcher("crypto")'
  instead of importing backend modules everywhere.
- Encapsulates backend selection and any construction defaults in one place.
"""

from __future__ import annotations

import logging
from typing import Literal

from .base import OptionFetcher
from .okx_fetcher import OKXFetcher
from .polygon_fetcher import PolygonFetcher
from .yf_fetcher import YFinanceFetcher

logger = logging.getLogger(__name__)


def get_fetcher(
    asset_class: Literal["equity", "crypto"], **kwargs
) -> OptionFetcher:
    """
    Return a concrete fetcher.

    kwargs are forwarded to the underlying fetcher constructors so callers can
    override defaults (timeouts, currency labels, etc.).
    """
    logger.debug(f"Creating {asset_class} fetcher with kwargs: {kwargs}")

    if asset_class == "equity":
        fetcher = YFinanceFetcher(**kwargs)
        logger.info(f"Created YFinanceFetcher")
        return fetcher
    elif asset_class == "crypto":
        fetcher = OKXFetcher(**kwargs)
        logger.info(f"Created OKXFetcher")
        return fetcher
    elif asset_class == "equity_polygon":
        logger.info(f"Created PolygonFetcher")
        return PolygonFetcher(**kwargs)
    elif asset_class == "crypto_polygon":
        logger.info(f"Created PolygonFetcher")
        return PolygonFetcher(**kwargs)
    else:
        logger.error(f"Unsupported asset_class: {asset_class}")
        raise ValueError(f"Unsupported asset_class: {asset_class}")
