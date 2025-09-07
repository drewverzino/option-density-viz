import pandas as pd

from preprocess.midprice import add_midprice_columns


def test_midprice_basic():
    df = pd.DataFrame(
        {
            "bid": [1.0, 1.0, None, 2.0, 3.0],
            "ask": [1.2, None, 1.5, 1.0, 3.5],
        }
    )
    out = add_midprice_columns(df, wide_rel_threshold=0.15)
    # Row 0: both sides -> mid = 1.1, rel_spread ~ 0.1818 -> wide True
    assert abs(out["mid"][0] - 1.1) < 1e-9
    assert out["crossed"][0] is False
    assert out["wide"][0] is True

    # Row 1: bid only -> mid = bid
    assert out["side_used"][1] == "bid_only"
    assert abs(out["mid"][1] - 1.0) < 1e-9

    # Row 2: ask only -> mid = ask
    assert out["side_used"][2] == "ask_only"
    assert abs(out["mid"][2] - 1.5) < 1e-9

    # Row 3: crossed (bid 2.0 > ask 1.0)
    assert out["crossed"][3] is True

    # Row 4: both sides OK
    assert out["side_used"][4] == "both"
