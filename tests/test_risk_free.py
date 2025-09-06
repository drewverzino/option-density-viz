from datetime import datetime
import pandas as pd

from data.risk_free import RiskFreeProvider, RiskFreeConfig


def test_risk_free_with_csv(tmp_path):
    # Make a tiny SOFR CSV
    p = tmp_path / "sofr.csv"
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "rate": [0.05, 0.052]}
    )
    df.to_csv(p, index=False)

    rf = RiskFreeProvider(
        RiskFreeConfig(sofr_csv_path=p, default_rate=0.04, forward_fill=True)
    )

    assert abs(rf.get_rate(datetime(2025, 1, 1)) - 0.05) < 1e-12
    # Forward-filled weekend/holes should return last known
    assert abs(rf.get_rate(datetime(2025, 1, 5)) - 0.052) < 1e-12
    # Before range -> boundary
    assert abs(rf.get_rate(datetime(2024, 12, 31)) - 0.05) < 1e-12
    # After range -> boundary
    assert abs(rf.get_rate(datetime(2025, 12, 31)) - 0.052) < 1e-12
