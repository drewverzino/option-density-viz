from datetime import datetime, timezone
from data.base import OptionChain, OptionQuote
from data.historical_loader import save_chain_csv, load_chain_csv

def test_csv_roundtrip(tmp_path):
    # Build a tiny chain with one quote
    q = OptionQuote(
        symbol="SYM-20250101-100-C",
        underlying="AAPL",
        asset_class="equity",
        expiry=datetime(2025,1,1,tzinfo=timezone.utc),
        strike=100.0,
        opt_type="C",
        bid=1.0,
        ask=2.0,
        last=1.5,
        mark=None,
        volume=10,
        open_interest=20,
        contract_size=100.0,
        underlying_ccy="USD",
        quote_ccy="USD",
        is_inverse=False,
        extra={"iv": 0.25},
    )
    ch = OptionChain(
        underlying="AAPL",
        asset_class="equity",
        spot=200.0,
        asof_utc=datetime.now(tz=timezone.utc),
        quotes=[q],
    )

    path = tmp_path / "chain.csv"
    save_chain_csv(path, ch)
    ch2 = load_chain_csv(path)

    assert ch2.underlying == "AAPL"
    assert ch2.asset_class == "equity"
    assert ch2.spot == 200.0
    assert len(ch2.quotes) == 1
    q2 = ch2.quotes[0]
    assert q2.strike == 100.0 and q2.opt_type == "C"
    assert q2.contract_size == 100.0