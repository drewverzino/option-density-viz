"""Plotting helpers and Streamlit dashboard."""

try:
    from .plots import plot_density, plot_smile
except Exception:
    pass

__all__ = [n for n in ["plot_smile", "plot_density"] if n in globals()]
