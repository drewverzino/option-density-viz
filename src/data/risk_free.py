# src/data/risk_free.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class RiskFreeConfig:
    sofr_csv_path: Optional[Path] = None  # CSV with columns: date, rate (decimal, e.g., 0.0525)
    default_rate: float = 0.05             # fallback if CSV missing or date not found
    forward_fill: bool = True              # forward-fill weekends/holidays
    cache: bool = True


class RiskFreeProvider:
    """
    Supplies a risk-free rate r(t). By default reads a local SOFR CSV; else returns a constant.
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
                # make a daily index; ffill missing
                idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D").date
                df = df.set_index("date").reindex(idx).ffill().rename_axis("date").reset_index()
            self._df = df

    def get_rate(self, when: datetime | date) -> float:
        if isinstance(when, datetime):
            d = when.date()
        else:
            d = when

        if self._df is None:
            return float(self.config.default_rate)

        df = self._df
        # exact match fast-path
        row = df[df["date"] == d]
        if not row.empty:
            return float(row["rate"].iloc[0])

        # outside known range → closest available (start or end), or default
        if d < df["date"].min():
            return float(df["rate"].iloc[0])
        if d > df["date"].max():
            return float(df["rate"].iloc[-1])

        # between known days and forward_fill=False → default
        return float(self.config.default_rate)
