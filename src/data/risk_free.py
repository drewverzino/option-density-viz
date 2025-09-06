# src/data/risk_free.py
from __future__ import annotations

"""
Simple risk-free rate provider.

Design goals:
- Pull a daily SOFR rate from a local CSV (date, rate) when available.
- Fall back to a constant otherwise (configurable).
- Optionally forward-fill weekends/holidays so every date has a value.

CSV format:
    date,rate
    2025-01-02,0.0525
    2025-01-03,0.0526
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class RiskFreeConfig:
    sofr_csv_path: Optional[Path] = None  # path to local SOFR CSV (optional)
    default_rate: float = 0.05            # constant fallback when CSV missing/out-of-range
    forward_fill: bool = True             # fill non-business days using prior value
    cache: bool = True                    # reserved for future (e.g., memoization)


class RiskFreeProvider:
    """
    Provide 'get_rate(datetime|date) -> float'.

    Behavior:
    - If a CSV is provided and readable, we query it (with optional ffill).
    - Otherwise return the constant 'default_rate'.
    """

    def __init__(self, config: RiskFreeConfig | None = None):
        self.config = config or RiskFreeConfig()
        self._df: Optional[pd.DataFrame] = None

        if self.config.sofr_csv_path and Path(self.config.sofr_csv_path).exists():
            df = pd.read_csv(self.config.sofr_csv_path)
            if not {"date", "rate"}.issubset(df.columns):
                raise ValueError("SOFR CSV must contain columns: date, rate")

            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.sort_values("date").drop_duplicates("date")

            if self.config.forward_fill:
                # Build a continuous daily index and forward-fill missing dates
                idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D").date
                df = df.set_index("date").reindex(idx).ffill().rename_axis("date").reset_index()

            self._df = df

    def get_rate(self, when: datetime | date) -> float:
        """Return a daily rate for 'when' (or a reasonable fallback)."""
        d = when.date() if isinstance(when, datetime) else when

        if self._df is None:
            return float(self.config.default_rate)

        df = self._df
        # Exact date match
        row = df[df["date"] == d]
        if not row.empty:
            return float(row["rate"].iloc[0])

        # Outside known range -> use boundary values if available
        if d < df["date"].min():
            return float(df["rate"].iloc[0])
        if d > df["date"].max():
            return float(df["rate"].iloc[-1])

        # Between known business days and forward_fill=False -> default
        return float(self.config.default_rate)
