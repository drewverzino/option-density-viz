import numpy as np

from density.cdf import build_cdf, moments_from_pdf, ppf_from_cdf


def test_cdf_and_ppf_roundtrip():
    # Triangle pdf on [0, 2]
    K = np.linspace(0.0, 2.0, 201)
    pdf = np.where(K <= 1.0, K, 2.0 - K)
    pdf = np.where(pdf > 0.0, pdf, 0.0)
    pdf = pdf / np.trapezoid(pdf, K)

    Kc, cdf = build_cdf(K, pdf)
    qs = np.array([0.05, 0.5, 0.95])
    qsK = ppf_from_cdf(Kc, cdf, qs)

    assert qsK[0] < 0.5 < qsK[1] < 1.5 < qsK[2]
    m = moments_from_pdf(K, pdf)
    assert abs(m["mean"] - 1.0) < 1e-2
    assert m["var"] > 0.0
