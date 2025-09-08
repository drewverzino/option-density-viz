import numpy as np
import pytest
from scipy.special import erf

from density.bl import (
    bl_pdf_from_calls,
    bl_pdf_from_calls_nonuniform,
)
from density.cdf import moments_from_pdf


def _make_lognormal_pdf(F=100.0, vol=0.2, T=0.5, n=2000, xmax_mult=3.0):
    """
    Construct a lognormal RN density on a grid for testing.
    Choose mu so that E[S_T] = F.
    """
    sigma = vol * np.sqrt(T)
    mu = np.log(F) - 0.5 * sigma**2
    # Grid for S_T
    x_max = F * np.exp(xmax_mult * sigma)
    x = np.linspace(max(1e-6, F * np.exp(-xmax_mult * sigma)), x_max, n)
    pdf = (1.0 / (x * sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -((np.log(x) - mu) ** 2) / (2.0 * sigma**2)
    )
    # normalize numerically
    pdf = pdf / np.trapezoid(pdf, x)
    return x, pdf


def _call_from_pdf(Ks, x, pdf, r=0.0, T=0.5):
    """
    Compute C(K) = e^{-rT} ∫_{K}^{∞} (s-K) f(s) ds using cumulative sums.
    """
    s = x
    h = s[1] - s[0]
    # tail integrals from the right
    tail_prob = np.cumsum(pdf[::-1])[::-1] * h
    tail_first = np.cumsum((s * pdf)[::-1])[::-1] * h
    # For each K, locate index in s-grid
    C = np.empty_like(Ks, dtype=float)
    disc = np.exp(-r * T)
    for i, K in enumerate(Ks):
        j = np.searchsorted(s, K, side="left")
        if j >= s.size:
            C[i] = 0.0
            continue
        # ∫_K^∞ (s - K) f(s) ds = ∫_K^∞ s f(s) ds - K ∫_K^∞ f(s) ds
        C[i] = disc * (tail_first[j] - K * tail_prob[j])
    return C


@pytest.mark.parametrize("vol,T", [(0.2, 0.25), (0.6, 0.05)])
def test_bl_recovery_basic(vol, T):
    F = 100.0
    r = 0.0  # simplify synthetic test
    # Ground-truth RN density
    x, pdf_true = _make_lognormal_pdf(F=F, vol=vol, T=T, n=4000, xmax_mult=4.0)
    # Build a moderately coarse call curve
    Ks = np.linspace(x.min(), x.max(), 161)
    C = _call_from_pdf(Ks, x, pdf_true, r=r, T=T)

    # Recover pdf via BL (smoothed uniform)
    Ku, pdf_est, diag = bl_pdf_from_calls(Ks, C, r=r, T=T, grid_n=801)

    # Basic diagnostics
    assert 0.98 <= diag.integral <= 1.02
    assert diag.neg_frac <= 0.10

    # Check the mean vs. F (RN mean should be F when r=0)
    m = moments_from_pdf(Ku, pdf_est)["mean"]
    assert abs(m - F) / F < 0.02  # within 2%

    # Compare distributions roughly via L1 distance
    pdf_true_u = np.interp(Ku, x, pdf_true)
    l1 = np.trapezoid(np.abs(pdf_true_u - pdf_est), Ku)
    assert l1 < 0.25  # tolerant due to smoothing


def test_bl_handles_small_input():
    Ks = np.array([90.0, 100.0, 110.0])
    C = np.array([12.0, 7.0, 3.0])
    Ku, pdf, diag = bl_pdf_from_calls(Ks, C, r=0.0, T=0.25, grid_n=51)
    assert np.isfinite(pdf).all()
    assert 0.9 <= diag.integral <= 1.1


def _black76_call(F, K, T, r, sigma):
    """Black-76 call option pricing (vectorized)."""
    # Use numpy functions for vectorized operations
    Df = np.exp(-r * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Use scipy.special.erf for vectorized operations
    Phi = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

    return Df * (F * Phi(d1) - K * Phi(d2))


def test_bl_uniform_grid_synthetic():
    F, r, T, sigma = 100.0, 0.02, 0.25, 0.4
    K = np.linspace(50.0, 170.0, 241)
    C = _black76_call(F, K, T, r, sigma)

    Ku, pdf, diag = bl_pdf_from_calls(K, C, r=r, T=T, grid_n=401)

    # Normalization and mean sanity
    assert 0.98 <= diag.integral <= 1.02
    assert abs(diag.rn_mean - F) <= 2.0  # within ~2 bucks on synthetic curve
    # Limited negative mass fraction before clipping is okay on numeric grids
    assert 0.0 <= diag.neg_frac <= 0.25
    # pdf non-negativity after clipping
    assert np.all(pdf >= 0.0)
    # Grid alignment
    assert Ku.size == pdf.size


def test_bl_nonuniform_grid_synthetic():
    F, r, T, sigma = 100.0, 0.01, 0.5, 0.35
    K = np.array(
        [60, 70, 80, 85, 90, 92, 95, 97, 100, 103, 106, 110, 120, 135, 150],
        dtype=float,
    )
    C = _black76_call(F, K, T, r, sigma)

    Kn, pdf, diag = bl_pdf_from_calls_nonuniform(K, C, r=r, T=T)

    assert 0.98 <= diag.integral <= 1.02
    assert abs(diag.rn_mean - F) <= 3.0
    assert np.all(pdf >= 0.0)
    assert Kn.size == pdf.size
