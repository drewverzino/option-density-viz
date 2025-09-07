# tests/test_vol_noarb.py
import numpy as np

from vol.no_arb import butterfly_violations, calendar_violations


def test_butterfly_convex_calls():
    # Build convex call curve: C(K) = max(F-K,0) smoothed
    F = 100.0
    K = np.linspace(60.0, 140.0, 81)
    # Smooth intrinsic with a tiny quadratic near ATM
    C = np.maximum(F - K, 0.0) + 1e-4 * (K - 100.0) ** 2
    res = butterfly_violations(K, C, tol=1e-9)
    assert res["fraction"] < 0.05


def test_calendar():
    k = np.linspace(-1.0, 1.0, 51)
    w1 = 0.04 + 0.1 * (k**2)
    w2 = w1 + 0.02  # strictly larger
    diag = calendar_violations(k, w1, w2, tol=1e-12)
    assert diag["fraction"] == 0.0
