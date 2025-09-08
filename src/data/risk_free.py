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

import csv
import logging
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("data.risk_free")

# ---------------------------
# Helpers & conventions
# ---------------------------


def _to_date(x) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    return datetime.fromisoformat(str(x)).date()


def _dcf(start: date, end: date, *, basis: str) -> float:
    """Year fraction between two dates."""
    days = (end - start).days
    if basis.lower() in ("act/365", "act365", "act/365f", "act365f"):
        return max(0.0, days) / 365.0
    if basis.lower() in ("act/360", "act360"):
        return max(0.0, days) / 360.0
    # default: ACT/365F
    return max(0.0, days) / 365.0


def simple_to_cont(r_simple: float, T: float) -> float:
    """Convert simple/linear annualized rate to continuous for maturity T (years)."""
    # Price with simple: (1 + r_simple * T) ; set = exp(r_cont * T)
    return 0.0 if T <= 0 else math.log(max(1e-12, 1.0 + r_simple * T)) / T


def cont_to_simple(r_cont: float, T: float) -> float:
    """Convert continuous to simple for maturity T (years)."""
    return 0.0 if T <= 0 else (math.exp(r_cont * T) - 1.0) / T


def apy_to_cont(apy: float) -> float:
    """Nominal APY (annual compounding) -> continuous."""
    # (1+apy) = exp(r_cont) -> r_cont = ln(1+apy)
    return math.log(max(1e-12, 1.0 + apy))


def cont_to_apy(r_cont: float) -> float:
    return math.expm1(r_cont)


def clamp(x: float, lo: float = -0.5, hi: float = 1.5) -> float:
    try:
        return min(max(float(x), lo), hi)
    except Exception:
        return float("nan")


# ---------------------------
# Config
# ---------------------------


@dataclass
class RiskFreeConfig:
    # Storage
    db_path: Optional[Path] = (
        None  # if None, use .cache/rates.db next to process cwd
    )
    ensure_dirs: bool = True

    # Ingest
    sofr_csv_path: Optional[Path] = None  # optional daily SOFR CSV
    sofr_csv_has_header: bool = True
    sofr_csv_date_col: str = "Date"
    sofr_csv_rate_col: str = "SOFR"  # accept "Rate" or "SOFR" etc.

    # Defaults / behavior
    default_rate: float = 0.02  # used if no data available
    basis: str = "ACT/365"  # ACT/365 or ACT/360
    forward_fill: bool = True  # allow prior date carry-forward
    linear_interp_tenor: bool = True  # linearly interpolate across tenor
    max_ff_days: int = 30  # max lookback days for forward-fill
    cache: bool = True  # memoize lookups in memory


# ---------------------------
# SQLite store
# ---------------------------


class RiskFreeStore:
    """
    Tiny SQLite-backed storage for zero rates.
    Each row stores a *continuous* zero rate for a given date and tenor (in days).
    We also keep the source and a normalized simple rate for convenience.

    Schema:
      rates(
        dt TEXT NOT NULL,           -- 'YYYY-MM-DD'
        tenor_days INTEGER NOT NULL,
        r_cont REAL NOT NULL,       -- continuous annualized
        r_simple REAL NOT NULL,     -- linear annualized for that tenor
        source TEXT NOT NULL,       -- 'SOFR', 'CSV', 'DEFAULT', 'MANUAL'
        PRIMARY KEY (dt, tenor_days, source)
      )
      CREATE INDEX IF NOT EXISTS idx_rates_dt ON rates(dt);
      CREATE INDEX IF NOT EXISTS idx_rates_dt_tenor ON rates(dt, tenor_days);
    """

    def __init__(self, path: Path):
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(str(self.path))

    def _init_db(self):
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS rates (
                  dt TEXT NOT NULL,
                  tenor_days INTEGER NOT NULL,
                  r_cont REAL NOT NULL,
                  r_simple REAL NOT NULL,
                  source TEXT NOT NULL,
                  PRIMARY KEY (dt, tenor_days, source)
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_rates_dt ON rates(dt)")
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_rates_dt_tenor ON rates(dt, tenor_days)"
            )
        logger.debug(f"RiskFreeStore initialized at {self.path}")

    def upsert_many(
        self, rows: Iterable[Tuple[str, int, float, float, str]]
    ) -> int:
        """
        Insert or replace many rows.
        rows: iterable of (dt_iso, tenor_days, r_cont, r_simple, source)
        """
        with self._connect() as con:
            cur = con.executemany(
                """
                INSERT OR REPLACE INTO rates (dt, tenor_days, r_cont, r_simple, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                list(rows),
            )
            return cur.rowcount or 0

    def get_curve(self, dt: date) -> List[Tuple[int, float]]:
        """Return [(tenor_days, r_cont)] for a specific date (all sources collapsed, prefer non-DEFAULT)."""
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT tenor_days, r_cont, source
                FROM rates WHERE dt = ?
                ORDER BY CASE WHEN source='DEFAULT' THEN 1 ELSE 0 END, tenor_days
                """,
                (dt.isoformat(),),
            )
            rows = cur.fetchall()
        # collapse by tenor: prefer first occurrence (non-DEFAULT comes first)
        best: Dict[int, float] = {}
        for tenor_days, r_cont, _src in rows:
            if tenor_days not in best:
                best[int(tenor_days)] = float(r_cont)
        curve = sorted(best.items())
        return curve

    def get_nearest_date(
        self, dt: date, *, max_back_days: int
    ) -> Optional[date]:
        """Find latest available date <= dt within max_back_days."""
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT dt FROM rates
                WHERE dt <= ?
                GROUP BY dt
                ORDER BY dt DESC
                LIMIT 1
                """,
                (dt.isoformat(),),
            )
            row = cur.fetchone()
        if not row:
            return None
        found = datetime.fromisoformat(row[0]).date()
        if (dt - found).days > max_back_days:
            return None
        return found

    def clear_source(self, source: str) -> int:
        with self._connect() as con:
            cur = con.execute("DELETE FROM rates WHERE source=?", (source,))
            return cur.rowcount or 0


# ---------------------------
# Provider
# ---------------------------


class RiskFreeProvider:
    """
    High-level access:
      - ingest SOFR CSV (overnight) and build a simple zero curve for standard tenors
      - query get_rate(asof_dt, tenor_days=365) -> continuous annualized zero rate
      - interpolation in tenor + optional forward-fill in date
    """

    # Default tenors to construct from an overnight input. You can expand if desired.
    DEFAULT_TENORS = (1, 7, 30, 90, 180, 365)  # days

    def __init__(self, cfg: RiskFreeConfig):
        self.cfg = cfg
        db_path = cfg.db_path or Path(".cache") / "rates.db"
        if cfg.ensure_dirs and db_path.parent:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.store = RiskFreeStore(db_path)
        self._memo: Dict[Tuple[date, int], float] = {}

        if cfg.sofr_csv_path and Path(cfg.sofr_csv_path).exists():
            try:
                self._ingest_sofr_csv(Path(cfg.sofr_csv_path))
                logger.info(f"Ingested SOFR CSV from {cfg.sofr_csv_path}")
            except Exception as e:
                logger.warning(f"SOFR CSV ingest failed: {e}")

        # Always ensure at least a DEFAULT zero curve for recent days using default_rate
        # This provides a deterministic fallback.
        self._ensure_default_curve()

    # ---------- ingest ----------

    def _ingest_sofr_csv(self, path: Path) -> None:
        """
        Expect a CSV with at least [Date, SOFR] or [Date, Rate] columns (APY or percent).
        We will treat the daily value as *overnight* APY and extend to a coarse curve by
        mapping to DEFAULT_TENORS using flat extrapolation in continuous space.
        """
        rows: List[Tuple[str, int, float, float, str]] = []
        with path.open("r", newline="") as f:
            reader = (
                csv.DictReader(f)
                if self.cfg.sofr_csv_has_header
                else csv.reader(f)
            )
            for rec in reader:
                try:
                    if isinstance(rec, dict):
                        dt = _to_date(rec[self.cfg.sofr_csv_date_col])
                        raw = float(
                            str(
                                rec.get(
                                    self.cfg.sofr_csv_rate_col,
                                    rec.get("Rate", ""),
                                )
                            ).strip()
                        )
                    else:
                        # no header: assume [Date, Rate, ...]
                        dt = _to_date(rec[0])
                        raw = float(rec[1])
                except Exception:
                    continue

                # Accept both percent (e.g., 5.30) or decimal (0.053)
                apy = raw / 100.0 if raw > 1.0 else raw
                r_cont_ov = apy_to_cont(apy)  # overnight continuous annualized

                # Build a coarse curve by flat-in-time continuous rate (simple heuristic)
                for tenor in self.DEFAULT_TENORS:
                    T = max(1e-10, tenor / 365.0)
                    r_cont = r_cont_ov  # flat assumption; replace with more elaborate transform if needed
                    r_simple = cont_to_simple(r_cont, T)
                    rows.append(
                        (
                            dt.isoformat(),
                            int(tenor),
                            float(r_cont),
                            float(r_simple),
                            "SOFR",
                        )
                    )

        if rows:
            # Remove previous SOFR to avoid stale duplicates
            self.store.clear_source("SOFR")
            up = self.store.upsert_many(rows)
            logger.info(f"Inserted {up} SOFR-derived zero points")

    def _ensure_default_curve(self):
        """Create/refresh a default curve for today using config.default_rate across default tenors."""
        today = datetime.now(timezone.utc).date()
        r_cont = clamp(self.cfg.default_rate)
        rows = []
        for tenor in self.DEFAULT_TENORS:
            T = max(1e-10, tenor / 365.0)
            r_simple = cont_to_simple(r_cont, T)
            rows.append(
                (
                    today.isoformat(),
                    int(tenor),
                    float(r_cont),
                    float(r_simple),
                    "DEFAULT",
                )
            )
        # Keep last two weeks of DEFAULT curves (avoid unbounded growth)
        self.store.upsert_many(rows)

    # ---------- query ----------

    def get_rate(self, when: datetime | date, tenor_days: int = 365) -> float:
        """
        Return a continuous zero rate for 'when' (datetime or date) and tenor_days.

        Resolution:
          1) exact date curve if available
          2) if forward_fill: nearest past date within max_ff_days
          3) default curve (injected with config.default_rate)
        Tenor interpolation:
          - piecewise linear in continuous space across available tenors.
        """
        if isinstance(when, datetime):
            asof = when.date()
        else:
            asof = when

        tenor_days = max(1, int(tenor_days))
        key = (asof, tenor_days)
        if self.cfg.cache and key in self._memo:
            return self._memo[key]

        # Step 1: try exact date
        curve = self.store.get_curve(asof)
        if not curve and self.cfg.forward_fill:
            # Step 2: look back up to max_ff_days
            nearest = self.store.get_nearest_date(
                asof, max_back_days=self.cfg.max_ff_days
            )
            if nearest:
                curve = self.store.get_curve(nearest)
                logger.debug(
                    f"Forward-fill from {nearest} to {asof} (curve size={len(curve)})"
                )

        if not curve:
            # Step 3: use today's default curve
            self._ensure_default_curve()
            curve = self.store.get_curve(datetime.now(timezone.utc).date())
            logger.debug("Falling back to DEFAULT curve")

        # If still empty (shouldn't happen), return config.default_rate
        if not curve:
            logger.warning("No curve available; returning config.default_rate")
            r = clamp(self.cfg.default_rate)
            self._memo[key] = r
            return r

        # If the tenor matches exactly:
        for t, r_cont in curve:
            if t == tenor_days:
                self._memo[key] = float(r_cont)
                return float(r_cont)

        # Interpolate in tenor (continuous space)
        tenors = [t for t, _ in curve]
        rates = [r for _, r in curve]

        # Before first tenor -> flat to first
        if tenor_days <= tenors[0]:
            self._memo[key] = float(rates[0])
            return float(rates[0])

        # After last tenor -> flat extrapolation
        if tenor_days >= tenors[-1]:
            self._memo[key] = float(rates[-1])
            return float(rates[-1])

        # Find bracketing tenors
        for i in range(1, len(tenors)):
            t0, t1 = tenors[i - 1], tenors[i]
            if t0 <= tenor_days <= t1:
                r0, r1 = rates[i - 1], rates[i]
                if self.cfg.linear_interp_tenor and t1 > t0:
                    w = (tenor_days - t0) / (t1 - t0)
                    r = (1.0 - w) * r0 + w * r1
                else:
                    r = r0
                self._memo[key] = float(r)
                return float(r)

        # Fallback (should not reach here)
        self._memo[key] = float(rates[-1])
        return float(rates[-1])

    # ---------- utility ----------

    def insert_manual_curve(
        self,
        dt: date,
        pairs: Iterable[Tuple[int, float]],
        *,
        rate_kind: str = "cont",
        source: str = "MANUAL",
    ) -> int:
        """
        Insert a manual curve for a date:
          pairs: [(tenor_days, rate_value)], rate_value either continuous or simple per 'rate_kind'.
        """
        rows = []
        for tenor, rv in pairs:
            T = max(1e-10, tenor / 365.0)
            if rate_kind == "cont":
                r_cont = float(rv)
                r_simple = cont_to_simple(r_cont, T)
            elif rate_kind in ("simple", "linear"):
                r_simple = float(rv)
                r_cont = simple_to_cont(r_simple, T)
            elif rate_kind in ("apy", "annual"):
                r_cont = apy_to_cont(float(rv))
                r_simple = cont_to_simple(r_cont, T)
            else:
                raise ValueError(f"Unknown rate_kind: {rate_kind}")
            rows.append((dt.isoformat(), int(tenor), r_cont, r_simple, source))
        return self.store.upsert_many(rows)

    def describe_curve(self, dt: date) -> pd.DataFrame:
        """Return a pandas DataFrame of the stored curve for a date (if pandas is available)."""
        try:
            import pandas as _pd  # type: ignore
        except Exception:
            raise RuntimeError("pandas not available")
        curve = self.store.get_curve(dt)
        if not curve:
            return _pd.DataFrame(columns=["tenor_days", "r_cont"])
        return _pd.DataFrame(curve, columns=["tenor_days", "r_cont"])
