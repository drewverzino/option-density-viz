import numpy as np
import pandas as pd
from preprocess.pcp import (
    synth_put_from_call,
    synth_call_from_put,
    pcp_residual,
    add_pcp_diagnostics,
)


def test_pcp_scalar_identities():
    S, K, r, T, q = 100.0, 100.0, 0.05, 1.0, 0.0
    call = 12.0
    put = synth_put_from_call(S, K, r, T, call, q)
    # Reconstruct call
    call2 = synth_call_from_put(S, K, r, T, put, q)
    assert abs(call - call2) < 1e-9
    # Residual is zero by construction
    assert abs(pcp_residual(S, K, r, T, call, put, q)) < 1e-12


def test_pcp_dataframe_helpers():
    # Build a tiny long table: strikes K=[90,100,110], type in {"C","P"}, price=mid
    rows = []
    S, r, T = 100.0, 0.02, 0.5
    for K in [90.0, 100.0, 110.0]:
        C = 10.0  # arbitrary
        P = C + K * np.exp(-r * T) - S  # parity with q=0
        rows.append({"strike": K, "type": "C", "mid": C})
        rows.append({"strike": K, "type": "P", "mid": P})
    df = pd.DataFrame(rows)
    diag = add_pcp_diagnostics(
        df,
        spot=S,
        r=r,
        T=T,
        q=0.0,
        price_col="mid",
        type_col="type",
        strike_col="strike",
    )
    assert set(["C", "P", "residual", "put_synth", "call_synth"]).issubset(diag.columns)
    # Residuals should be ~0 (numerical noise)
    assert np.allclose(diag["residual"].to_numpy(), 0.0, atol=1e-12)
