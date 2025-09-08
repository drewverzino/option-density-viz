import numpy as np
from scipy.special import erf

from density.bl import bl_pdf_from_calls
from density.cdf import build_cdf, moments_from_pdf, ppf_from_cdf

# def test_cdf_and_ppf_roundtrip():
#     # Triangle pdf on [0, 2]
#     K = np.linspace(0.0, 2.0, 201)
#     pdf = np.where(K <= 1.0, K, 2.0 - K)
#     pdf = np.where(pdf > 0.0, pdf, 0.0)
#     pdf = pdf / np.trapezoid(pdf, K)

#     Kc, cdf = build_cdf(K, pdf)
#     qs = np.array([0.05, 0.5, 0.95])
#     qsK = ppf_from_cdf(Kc, cdf, qs)

#     assert qsK[0] < 0.5 < qsK[1] < 1.5 < qsK[2]
#     m = moments_from_pdf(K, pdf)
#     assert abs(m["mean"] - 1.0) < 1e-2
#     assert m["var"] > 0.0


def _black76_call(F, K, T, r, sigma):
    """Black-76 call option pricing (vectorized)."""
    # Use numpy functions for vectorized operations
    Df = np.exp(-r * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Use scipy.special.erf for vectorized operations
    Phi = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

    return Df * (F * Phi(d1) - K * Phi(d2))


def test_cdf_and_ppf_roundtrip():
    F, r, T, sigma = 150.0, 0.03, 0.4, 0.5
    K = np.linspace(50.0, 350.0, 301)
    C = _black76_call(F, K, T, r, sigma)

    Ku, pdf, diag = bl_pdf_from_calls(K, C, r=r, T=T, grid_n=601)
    Kc, cdf = build_cdf(Ku, pdf)

    # CDF is monotone and ends at 1
    assert np.all(np.diff(cdf) >= -1e-12)
    assert 0.999 <= cdf[-1] <= 1.001

    qs = np.array([0.05, 0.5, 0.95])
    xs = ppf_from_cdf(Kc, cdf, qs)

    # Quantiles ordered and lie within range
    assert xs[0] < xs[1] < xs[2]
    assert Kc.min() - 1e-9 <= xs[0] <= Kc.max() + 1e-9

    # Moments sanity: mean close to forward, variance positive
    m = moments_from_pdf(Ku, pdf)
    assert abs(m["mean"] - F) <= 3.0
    assert m["var"] > 0.0
    # skew/exkurt may vary, but should be finite
    assert np.isfinite(m["skew"])
    assert np.isfinite(m["exkurt"])
