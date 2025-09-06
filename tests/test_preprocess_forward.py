import numpy as np
import pandas as pd
from preprocess.forward import forward_price, log_moneyness, estimate_forward_from_chain

def test_forward_price_and_k():
    S, r, T, q = 100.0, 0.03, 1.0, 0.01
    F = forward_price(S, r, T, q)
    assert abs(F - S*np.exp((r-q)*T)) < 1e-12
    k = log_moneyness(np.array([80.0, 100.0, 120.0]), F)
    assert np.isfinite(k).all()

def test_estimate_forward_from_chain():
    # Build consistent C/P pairs satisfying parity for a constant forward
    S, r, T = 100.0, 0.05, 1.0
    Ks = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    # Choose arbitrary C(K) and derive P(K) from parity (q=0)
    rows = []
    for K in Ks:
        C = max(0.0, 20.0 - 0.1*abs(K-100.0))  # arbitrary profile
        P = C + K*np.exp(-r*T) - S
        rows.append({"strike": K, "type": "C", "mid": C, "rel_spread": 0.05})
        rows.append({"strike": K, "type": "P", "mid": P, "rel_spread": 0.05})
    df = pd.DataFrame(rows)
    F_est = estimate_forward_from_chain(df, r=r, T=T, price_col="mid", strike_col="strike", type_col="type", spot_hint=S)
    # True forward
    F_true = S*np.exp(r*T)
    assert abs(F_est - F_true) < 1e-6