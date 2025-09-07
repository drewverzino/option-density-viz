# tests/test_vol_svi.py
import numpy as np

from vol.svi import fit_svi, svi_w


def test_svi_fit_synthetic():
    rng = np.random.default_rng(42)
    k = np.linspace(-1.0, 1.0, 41)
    # True params
    a, b, rho, m, s = 0.02, 0.8, -0.3, 0.0, 0.25
    w_true = svi_w(k, a, b, rho, m, s)
    # Add small noise
    w_obs = w_true + rng.normal(0.0, 0.002, size=k.shape)
    fit = fit_svi(k, w_obs)
    # Reconstruct
    w_fit = svi_w(k, fit.a, fit.b, fit.rho, fit.m, fit.sigma)
    # Weighted RMSE should be small
    rmse = np.sqrt(np.mean((w_fit - w_true) ** 2))
    assert rmse < 0.02
