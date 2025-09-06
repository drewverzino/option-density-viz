# data/base.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, List, Protocol, Optional, Dict

OptType = Literal["C", "P"]
AssetClass = Literal["equity", "crypto"]

@dataclass
class OptionQuote:
    symbol: str                 # venue-specific option symbol
    underlying: str             # e.g., "AAPL", "SPY", "BTC"
    asset_class: AssetClass     # "equity" | "crypto"
    expiry: datetime            # UTC
    strike: float
    opt_type: OptType           # "C" or "P"
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    mark: Optional[float]       # mid or venue-provided mark
    volume: Optional[float]
    open_interest: Optional[float]
    contract_size: float        # shares/contracts-per-option (100 for US eq, 1 for OKX)
    underlying_ccy: str         # "USD", "USDT"
    quote_ccy: str              # price currency (usually same as underlying ccy)
    is_inverse: bool            # False for yfinance & OKX
    extra: Dict[str, float]     # room for greeks/implied_vol if present

@dataclass
class OptionChain:
    underlying: str
    asset_class: AssetClass
    spot: Optional[float]
    asof_utc: datetime
    quotes: List[OptionQuote]

class OptionFetcher(Protocol):
    async def list_expiries(self, underlying: str) -> List[datetime]: ...
    async def fetch_chain(self, underlying: str, expiry: datetime) -> OptionChain: ...
