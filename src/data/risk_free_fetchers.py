from __future__ import annotations

import logging
import math
from datetime import date, datetime, timezone
from typing import Iterable, Tuple

import httpx

from .risk_free import RiskFreeProvider, apy_to_cont, clamp, cont_to_simple

log = logging.getLogger("data.risk_free_fetch")

FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series/observations"
TREASURY_BASE = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
)


def _iso(x) -> str:
    return x if isinstance(x, str) else x.isoformat()


async def ingest_fred_series(
    provider: RiskFreeProvider,
    *,
    series_id: str = "SOFR",
    start: date | str,
    end: date | str,
    api_key: str,
) -> int:
    """
    Ingest a daily FRED series (percent units) and store a simple curve at DEFAULT tenors
    by assuming a flat continuous rate across tenor (good for overnight series like SOFR/EFFR).
    """
    start_s, end_s = _iso(start), _iso(end)
    params = {
        "series_id": series_id,
        "observation_start": start_s,
        "observation_end": end_s,
        "file_type": "json",
        "api_key": api_key,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(FRED_SERIES_URL, params=params)
        r.raise_for_status()
        js = r.json()

    # FRED returns "value" as string, '.' for missing. Units are percent.
    rows = []
    for obs in js.get("observations", []):
        dt = obs.get("date")
        val = obs.get("value")
        if not dt or not val or val == ".":
            continue
        pct = float(val)  # e.g. 4.42
        apy = pct / 100.0
        r_cont = apy_to_cont(apy)  # convert APY→continuous annualized
        dt_iso = dt[:10]
        # build a coarse curve across provider.DEFAULT_TENORS
        for tenor in provider.DEFAULT_TENORS:
            T = max(1e-10, tenor / 365.0)
            r_simple = math.expm1(r_cont * T) / T
            rows.append(
                (
                    dt_iso,
                    int(tenor),
                    float(r_cont),
                    float(r_simple),
                    f"FRED_{series_id}",
                )
            )

    if not rows:
        return 0
    # clear previous source then insert
    provider.store.clear_source(f"FRED_{series_id}")
    up = provider.store.upsert_many(rows)
    log.info(f"FRED {series_id}: inserted {up} rows from {start_s} to {end_s}")
    return up


async def ingest_treasury_par_curve(
    provider: RiskFreeProvider,
    *,
    start: date | str,
    end: date | str,
) -> int:
    """
    Ingest Daily Treasury Par Yield Curve (percent), convert to continuous, store per tenor-day.
    """
    start_s, end_s = _iso(start), _iso(end)
    # API docs allow $filter and $format; we select needed fields only.
    url = f"{TREASURY_BASE}/v1/accounting/od/avg_interest_rates"
    # Alternatively, the “Daily Treasury Par Yield Curve Rates” dataset path may vary;
    # The official site also offers a 'Download CSV' per month/year.
    params = {
        "$filter": f"record_date ge {start_s} and record_date le {end_s}",
        "$format": "json",
        "$select": ",".join(
            [
                "record_date",
                "avg_1_yr",
                "avg_2_yr",
                "avg_3_yr",
                "avg_5_yr",
                "avg_7_yr",
                "avg_10_yr",
                "avg_20_yr",
                "avg_30_yr",
            ]
        ),
        "$page": 1,
        "$page_size": 10000,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        js = r.json()

    # Map tenor years→days; convert percent→continuous and store exact tenor buckets
    year_to_days = {
        1: 365,
        2: 730,
        3: 1095,
        5: 1825,
        7: 2555,
        10: 3650,
        20: 7300,
        30: 10950,
    }
    inserted = 0
    rows = []
    for row in js.get("data", []):
        dt_iso = row["record_date"][:10]
        for yrs, col in [
            (1, "avg_1_yr"),
            (2, "avg_2_yr"),
            (3, "avg_3_yr"),
            (5, "avg_5_yr"),
            (7, "avg_7_yr"),
            (10, "avg_10_yr"),
            (20, "avg_20_yr"),
            (30, "avg_30_yr"),
        ]:
            val = row.get(col)
            if val is None:
                continue
            pct = float(val)  # e.g. 4.18
            apy = pct / 100.0
            r_cont = apy_to_cont(apy)  # APY→continuous annualized
            tenor_days = year_to_days[yrs]
            T = yrs  # years
            r_simple = math.expm1(r_cont * T) / T
            rows.append(
                (
                    dt_iso,
                    int(tenor_days),
                    float(r_cont),
                    float(r_simple),
                    "TREASURY_PAR",
                )
            )
    if rows:
        provider.store.clear_source("TREASURY_PAR")
        inserted = provider.store.upsert_many(rows)
        log.info(
            f"Treasury PAR: inserted {inserted} rows from {start_s} to {end_s}"
        )
    return inserted
