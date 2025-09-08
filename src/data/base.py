"""
Core types and the fetcher protocol used throughout the project.

Why this file exists:
- We want all backends (equity via yfinance, crypto via OKX, etc.) to return data
  in a **single, normalized schema** so the rest of the pipeline never cares
  where the data came from.
- We also define a **minimal protocol** that any backend must implement.

Tip:
- Keep these types tiny and stable; downstream code (preprocess/vol/density/viz)
  depends on them heavily.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Literal, Optional, Protocol

# A couple of small, expressive type aliases for clarity
OptType = Literal["C", "P"]  # option type: Call or Put
AssetClass = Literal["equity", "crypto"]

logger = logging.getLogger(__name__)


@dataclass
class OptionQuote:
    """
    A single, normalized option quote.

    Notes:
    - 'mark' can be None; many venues do not provide mark prices consistently.
      Downstream code should compute a mid if needed (see preprocess/midprice).
    - 'extra' exists to carry venue-specific goodies (e.g., implied vol from yfinance).
    - 'contract_size' is 100 for U.S. equity options; 1 for OKX options.
    """

    symbol: str
    underlying: str
    asset_class: AssetClass
    expiry: datetime  # timezone-aware UTC recommended
    strike: float
    opt_type: OptType  # "C" or "P"
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    mark: Optional[float]
    volume: Optional[float]
    open_interest: Optional[float]
    contract_size: float
    underlying_ccy: str  # e.g., "USD"
    quote_ccy: str  # e.g., "USD", "USDT"
    is_inverse: bool  # False for yfinance and OKX options
    extra: Dict[str, float]  # optional fields like "iv"


@dataclass
class OptionChain:
    """
    A set of quotes for a single underlying at one point in time.

    'spot' and 'asof_utc' travel alongside the quotes for reproducibility.
    """

    underlying: str
    asset_class: AssetClass
    spot: Optional[float]
    asof_utc: datetime
    quotes: List[OptionQuote]

    def __post_init__(self):
        """Log chain summary after creation."""
        n_calls = sum(1 for q in self.quotes if q.opt_type == "C")
        n_puts = sum(1 for q in self.quotes if q.opt_type == "P")
        logger.debug(
            f"Created OptionChain: {self.underlying} ({self.asset_class}) "
            f"with {len(self.quotes)} quotes ({n_calls}C, {n_puts}P), "
            f"spot={self.spot}"
        )


class OptionFetcher(Protocol):
    """
    Minimal interface that every backend must implement.

    Implementations:
      - data/yf_fetcher.py (equity via yfinance)
      - data/okx_fetcher.py (crypto via OKX)

    Expected behavior:
      - 'list_expiries' returns tz-aware datetimes
      - 'fetch_chain' returns an OptionChain with normalized OptionQuotes
    """

    async def list_expiries(self, underlying: str) -> List[datetime]: ...

    async def fetch_chain(
        self, underlying: str, expiry: datetime
    ) -> OptionChain: ...
