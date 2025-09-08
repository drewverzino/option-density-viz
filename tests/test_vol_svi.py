# tests/test_vol_svi.py
import numpy as np

from vol.svi import calibrate_svi_from_quotes, svi_total_variance


def test_svi_fit_synthetic():
    rng = np.random.default_rng(42)
    k = np.linspace(-1.0, 1.0, 41)

    # Ground-truth SVI parameters
    a_true, b_true, rho_true, m_true, s_true = 0.02, 0.8, -0.3, 0.0, 0.25
    w_true = svi_total_variance(k, a_true, b_true, rho_true, m_true, s_true)

    # Add small observation noise in total-variance space
    w_obs = w_true + rng.normal(0.0, 0.002, size=k.shape)

    # Fit SVI (use_price_weighting False so we don't need F)
    fit = calibrate_svi_from_quotes(
        k, w=w_obs, use_price_weighting=False, grid_size=5
    )
    a, b, rho, m, sigma = fit.params

    # Reconstruct and measure RMSE vs. the true curve (not the noisy obs)
    w_fit = svi_total_variance(k, a, b, rho, m, sigma)
    rmse = float(np.sqrt(np.mean((w_fit - w_true) ** 2)))

    # Basic accuracy check on the fit
    assert rmse < 0.02

    # Sanity checks on parameters (loose bounds)
    assert 0.0 <= a <= 0.2
    assert 0.0 < b <= 5.0
    assert -0.999 < rho < 0.999
    assert -1.0 <= m <= 1.0
    assert 0.0 < sigma <= 5.0
