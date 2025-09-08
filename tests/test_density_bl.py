import numpy as np
import pytest

from density.bl import bl_pdf_from_calls
from density.cdf import build_cdf, moments_from_pdf


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
