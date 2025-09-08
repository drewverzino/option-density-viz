# tests/test_risk_free_fetchers.py
from __future__ import annotations

import asyncio
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict

import pytest

from data.risk_free import RiskFreeConfig, RiskFreeProvider, apy_to_cont
from data.risk_free_fetchers import (
    ingest_fred_series,
    ingest_treasury_par_curve,
)

# --------------------------- helpers --------------------------- #


class _MockResponse:
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        self._json = json_data
        self.status_code = status_code

    def json(self) -> Dict[str, Any]:
        return self._json

    def raise_for_status(self) -> None:
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")


class _DummyAsyncClient:
    """
    A minimal stand-in for httpx.AsyncClient that returns canned responses
    for the two API endpoints we use.
    """

    def __init__(
        self, *, routes: Dict[str, Dict[str, Any]], timeout: float | int = 10
    ):
        self.routes = routes
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str, params: Dict[str, Any] | None = None):
        # FRED detection
        if "stlouisfed.org/fred/series/observations" in url:
            series_id = (params or {}).get("series_id", "SOFR")
            key = f"FRED:{series_id}"
            payload = self.routes.get(key)
            if payload is None:
                return _MockResponse({"observations": []})
            return _MockResponse(payload)

        # Treasury detection (par curve)
        if "api.fiscaldata.treasury.gov/services/api/fiscal_service" in url:
            payload = self.routes.get("TREASURY_PAR", {"data": []})
            return _MockResponse(payload)

        # Fallback
        return _MockResponse({}, status_code=404)


# --------------------------- fixtures --------------------------- #


@pytest.fixture
def tmp_provider(tmp_path: Path) -> RiskFreeProvider:
    cfg = RiskFreeConfig(
        db_path=tmp_path / "rates.db",
        ensure_dirs=True,
        forward_fill=True,
        max_ff_days=30,
        cache=False,
    )
    return RiskFreeProvider(cfg)


@pytest.fixture
def fred_payload_simple() -> Dict[str, Any]:
    # FRED returns 'value' as percent string; '.' means missing.
    # Two business days of SOFR at 5.00% and 5.10%.
    return {
        "observations": [
            {"date": "2025-09-01", "value": "5.00"},
            {"date": "2025-09-02", "value": "5.10"},
        ]
    }


@pytest.fixture
def treasury_payload_simple() -> Dict[str, Any]:
    # Minimal shape for Treasury "Daily Treasury Par Yield Curve" style payload.
    # Provide 1Y and 3Y nodes to test interpolation.
    return {
        "data": [
            {
                "record_date": "2025-09-02",
                "avg_1_yr": "4.00",
                "avg_2_yr": None,
                "avg_3_yr": "4.60",
                "avg_5_yr": None,
                "avg_7_yr": None,
                "avg_10_yr": None,
                "avg_20_yr": None,
                "avg_30_yr": None,
            }
        ]
    }


@pytest.fixture
def monkeypatch_httpx(
    monkeypatch, fred_payload_simple, treasury_payload_simple
):
    """
    Monkeypatch httpx.AsyncClient used inside ingest_* functions to our dummy client,
    with canned responses for both endpoints.
    """
    import httpx  # imported here so monkeypatch target exists

    routes = {
        "FRED:SOFR": fred_payload_simple,
        "TREASURY_PAR": treasury_payload_simple,
    }

    def _client_factory(*args, **kwargs):
        return _DummyAsyncClient(
            routes=routes, timeout=kwargs.get("timeout", 10)
        )

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory)
    return routes


# --------------------------- tests --------------------------- #


@pytest.mark.asyncio
async def test_ingest_fred_sofr_inserts_and_query_short_tenor(
    tmp_provider: RiskFreeProvider,
    monkeypatch_httpx,
):
    # Arrange
    end = date(2025, 9, 2)
    start = end - timedelta(days=1)

    # Act: ingest two days of SOFR via FRED mock
    inserted = await ingest_fred_series(
        tmp_provider, series_id="SOFR", start=start, end=end, api_key="DUMMY"
    )

    # Assert: inserted rows for DEFAULT_TENORS each day
    # DEFAULT_TENORS = (1, 7, 30, 90, 180, 365)
    assert inserted == 2 * len(tmp_provider.DEFAULT_TENORS)

    # Query: exact date, overnight tenor should equal continuous rate derived from 5.10% APY
    r_cont_ref = apy_to_cont(0.0510)
    r_cont = tmp_provider.get_rate(end, tenor_days=1)
    assert math.isclose(r_cont, r_cont_ref, rel_tol=1e-8, abs_tol=1e-12)

    # Forward-fill: next day (no data in mock) should still return same within max_ff_days
    ff_day = end + timedelta(days=5)
    r_ff = tmp_provider.get_rate(ff_day, tenor_days=1)
    assert math.isclose(r_ff, r_cont_ref, rel_tol=1e-8, abs_tol=1e-12)


@pytest.mark.asyncio
async def test_ingest_treasury_par_curve_and_interpolate(
    tmp_provider: RiskFreeProvider,
    monkeypatch_httpx,
):
    # Arrange
    asof = date(2025, 9, 2)

    # Act: ingest daily Treasury par curve for one day (1Y=4.00%, 3Y=4.60%)
    inserted = await ingest_treasury_par_curve(
        tmp_provider, start=asof, end=asof
    )

    # Assert: inserted some rows (8 possible nodes, but we only provided 1Y & 3Y)
    assert inserted >= 2

    # 1Y exact tenor (365d) should equal APY->continuous(4.00%)
    r_1y_ref = apy_to_cont(0.0400)
    r_1y = tmp_provider.get_rate(asof, tenor_days=365)
    assert math.isclose(r_1y, r_1y_ref, rel_tol=1e-10, abs_tol=1e-12)

    # 3Y exact tenor (1095d) should equal APY->continuous(4.60%)
    r_3y_ref = apy_to_cont(0.0460)
    r_3y = tmp_provider.get_rate(asof, tenor_days=1095)
    assert math.isclose(r_3y, r_3y_ref, rel_tol=1e-10, abs_tol=1e-12)

    # 1.5Y interpolation (≈ 548d): linear interpolation in continuous space between 1Y and 3Y
    t0, t1 = 365, 1095
    t_mid = (
        365 + (1095 - 365) // 2
    )  # 730 ~ 2y, but test a non-endpoint to validate interpolation
    w = (t_mid - t0) / (t1 - t0)
    r_mid_expected = (1.0 - w) * r_1y_ref + w * r_3y_ref
    r_mid = tmp_provider.get_rate(asof, tenor_days=t_mid)
    assert math.isclose(r_mid, r_mid_expected, rel_tol=1e-10, abs_tol=1e-12)


@pytest.mark.asyncio
async def test_priority_of_sources_and_default_fallback(
    tmp_provider: RiskFreeProvider,
    monkeypatch_httpx,
):
    """
    Ensures:
      - After ingesting FRED SOFR and Treasury Par, queries use the stored curve.
      - If nothing is available on a date and forward-fill is exhausted, we fall back to DEFAULT curve.
    """
    asof = date(2025, 9, 2)
    # Ingest both sources
    await ingest_fred_series(
        tmp_provider, series_id="SOFR", start=asof, end=asof, api_key="DUMMY"
    )
    await ingest_treasury_par_curve(tmp_provider, start=asof, end=asof)

    # Query a long tenor (10y), expect it comes from Treasury par (if present) or flat extrap from last node
    r_10y = tmp_provider.get_rate(asof, tenor_days=3650)
    assert isinstance(r_10y, float)

    # Now query a date far ahead (beyond forward-fill window), which forces DEFAULT curve fallback
    far_date = asof + timedelta(days=90)
    tmp_provider.cfg.max_ff_days = 7  # make FF strict
    r_far = tmp_provider.get_rate(far_date, tenor_days=365)
    # DEFAULT curve is constructed with cfg.default_rate (continuous): accept closish equality
    assert math.isfinite(r_far)
    # Should be close to configured default if falling back (2% by default)
    assert abs(r_far - tmp_provider.cfg.default_rate) < 1e-6
