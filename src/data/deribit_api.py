"""Deribit API client for option-density-viz.

This module provides a minimal interface to the Deribit public REST API
for retrieving option instruments, ticker prices, and order book summaries.

Note: The functions in this module rely on the ``requests`` library to
perform HTTP requests. Network errors or invalid responses will raise
exceptions. Users should handle these exceptions appropriately in their
application code.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import requests

# Base URL for the Deribit API
DERIBIT_BASE_URL: str = "https://www.deribit.com/api/v2"

logger = logging.getLogger(__name__)


def _get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Internal helper to perform a GET request to the Deribit API.

    This stub function defines the signature and intended behavior of the
    network call but leaves the actual implementation to the user. When
    implementing, use the ``requests`` library to perform an HTTP GET
    request to the Deribit REST API and return the parsed JSON result.

    Args:
        endpoint: API endpoint path (e.g. ``/public/get_instruments``).
        params: Optional query parameters to include in the request.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError(
        "_get has not been implemented. Use requests.get to call the Deribit API and parse the JSON response."
    )


def fetch_instruments(currency: str, kind: str = "option") -> List[Dict[str, Any]]:
    """Fetch a list of instruments for a given currency and kind.

    This function should call the Deribit API endpoint ``/public/get_instruments``
    with the specified currency and kind parameters. It is defined here as
    a placeholder and must be implemented by the user.

    Args:
        currency: Currency symbol, e.g. ``"BTC"`` or ``"ETH"``.
        kind: Instrument kind. Defaults to ``"option"``.

    Returns:
        A list of instrument dictionaries as returned by the API.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError(
        "fetch_instruments has not been implemented. It should query the Deribit API for option instruments."
    )


def fetch_ticker(instrument_name: str) -> Dict[str, Any]:
    """Fetch ticker information for a specific instrument.

    The implementation should call the Deribit endpoint ``/public/ticker``
    with the instrument name and return the JSON result. This stub
    function must be filled in by the user.

    Args:
        instrument_name: The full name of the instrument, e.g. ``"BTC-30SEP22-30000-C"``.

    Returns:
        A dictionary containing ticker data such as last price, bid/ask, and implied volatility.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError(
        "fetch_ticker has not been implemented. Use _get to call /public/ticker and return the result."
    )


def fetch_orderbook(instrument_name: str) -> Dict[str, Any]:
    """Fetch order book summary for a specific instrument.

    The implementation should call the Deribit endpoint ``/public/get_book_summary_by_instrument``
    and return the first entry from the resulting list. This function is
    currently a stub and must be implemented.

    Args:
        instrument_name: The full name of the instrument.

    Returns:
        A dictionary containing order book summary data such as bid/ask sizes and last price.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError(
        "fetch_orderbook has not been implemented. Use _get to call /public/get_book_summary_by_instrument and return the result."
    )
