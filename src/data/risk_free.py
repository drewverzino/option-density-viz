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

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskFreeConfig:
    sofr_csv_path: Optional[Path] = None  # path to local SOFR CSV (optional)
    default_rate: float = (
        0.05  # constant fallback when CSV missing/out-of-range
    )
    forward_fill: bool = True  # fill non-business days using prior value
    cache: bool = True  # reserved for future (e.g., memoization)


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

        logger.debug(
            f"Initializing RiskFreeProvider with config: "
            f"csv_path={self.config.sofr_csv_path}, "
            f"default_rate={self.config.default_rate}, "
            f"forward_fill={self.config.forward_fill}"
        )

        if (
            self.config.sofr_csv_path
            and Path(self.config.sofr_csv_path).exists()
        ):
            logger.info(f"Loading SOFR CSV from: {self.config.sofr_csv_path}")
            try:
                df = pd.read_csv(self.config.sofr_csv_path)
                if not {"date", "rate"}.issubset(df.columns):
                    raise ValueError(
                        "SOFR CSV must contain columns: date, rate"
                    )

                df["date"] = pd.to_datetime(df["date"]).dt.date
                df = df.sort_values("date").drop_duplicates("date")

                logger.debug(
                    f"Loaded {len(df)} SOFR rates from {df['date'].min()} to {df['date'].max()}"
                )

                if self.config.forward_fill:
                    # Build a continuous daily index and forward-fill missing dates
                    idx = pd.date_range(
                        df["date"].min(), df["date"].max(), freq="D"
                    ).date
                    df = (
                        df.set_index("date")
                        .reindex(idx)
                        .ffill()
                        .rename_axis("date")
                        .reset_index()
                    )
                    logger.debug(f"Forward-filled to {len(df)} daily rates")

                self._df = df
                logger.info(f"Successfully loaded SOFR data: {len(df)} rates")

            except Exception as e:
                logger.error(f"Failed to load SOFR CSV: {e}")
                logger.warning(
                    f"Will use default rate: {self.config.default_rate}"
                )
                self._df = None
        else:
            if self.config.sofr_csv_path:
                logger.warning(
                    f"SOFR CSV not found: {self.config.sofr_csv_path}"
                )
            logger.info(
                f"Using constant risk-free rate: {self.config.default_rate}"
            )

    def get_rate(self, when: datetime | date) -> float:
        """Return a daily rate for 'when' (or a reasonable fallback)."""
        d = when.date() if isinstance(when, datetime) else when
        logger.debug(f"Getting risk-free rate for {d}")

        if self._df is None:
            logger.debug(
                f"No CSV data, returning default rate: {self.config.default_rate}"
            )
            return float(self.config.default_rate)

        df = self._df
        # Exact date match
        row = df[df["date"] == d]
        if not row.empty:
            rate = float(row["rate"].iloc[0])
            logger.debug(f"Found exact rate for {d}: {rate}")
            return rate

        # Outside known range -> use boundary values if available
        if d < df["date"].min():
            rate = float(df["rate"].iloc[0])
            logger.debug(f"Date {d} before range, using first rate: {rate}")
            return rate
        if d > df["date"].max():
            rate = float(df["rate"].iloc[-1])
            logger.debug(f"Date {d} after range, using last rate: {rate}")
            return rate

        # Between known business days and forward_fill=False -> default
        logger.debug(
            f"Date {d} in gap, using default rate: {self.config.default_rate}"
        )
        return float(self.config.default_rate)
