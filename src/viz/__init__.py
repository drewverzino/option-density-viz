# src/viz/__init__.py
"""
Visualization helpers for Option Viz.

Public API:
- plot_smile:        IV vs log-moneyness (k)
- plot_pdf_cdf:      Risk-neutral PDF (left) + CDF (right, twin axis)
- plot_svi_vs_market: Compare market IV points vs SVI model IV curve

All functions return a `matplotlib.figure.Figure` and accept:
- `theme`: "light" or "dark"
- `save_path`: optional path to save a PNG/SVG/PDF of the figure
"""

from .plots import plot_pdf_cdf, plot_smile, plot_svi_vs_market

__all__ = ["plot_smile", "plot_pdf_cdf", "plot_svi_vs_market"]
