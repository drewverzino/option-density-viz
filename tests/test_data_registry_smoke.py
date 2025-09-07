import os

import pytest

from data.okx_fetcher import OKXFetcher
from data.registry import get_fetcher
from data.yf_fetcher import YFinanceFetcher


def test_get_fetcher_types():
    f_eq = get_fetcher("equity")
    f_cr = get_fetcher("crypto")
    assert isinstance(f_eq, YFinanceFetcher.__class__) or isinstance(
        f_eq, YFinanceFetcher
    )
    assert isinstance(f_cr, OKXFetcher.__class__) or isinstance(
        f_cr, OKXFetcher
    )


@pytest.mark.skipif(
    os.getenv("ENABLE_NETWORK_TESTS", "0") != "1",
    reason="network tests disabled by default",
)
@pytest.mark.asyncio
async def test_equity_expiries_network():
    f = get_fetcher("equity")
    exps = await f.list_expiries("AAPL")
    assert isinstance(exps, list)


@pytest.mark.skipif(
    os.getenv("ENABLE_NETWORK_TESTS", "0") != "1",
    reason="network tests disabled by default",
)
@pytest.mark.asyncio
async def test_crypto_expiries_network():
    f = get_fetcher("crypto")
    exps = await f.list_expiries("BTC")
    assert isinstance(exps, list)
