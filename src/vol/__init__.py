"""Volatility surface fitting for option-density-viz.

The ``vol`` subpackage contains functions and classes for fitting
implied volatility smiles and surfaces. Currently, it provides
an implementation of the Stochastic Volatility Inspired (SVI) model
following the parameterization introduced by Gatheral (2011).
"""

from .svi import svi_total_variance, calibrate_svi, SVIParameters

__all__ = [
    "SVIParameters",
    "svi_total_variance",
    "calibrate_svi",
]
