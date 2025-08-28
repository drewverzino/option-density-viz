"""Risk‑neutral density estimators for option-density-viz.

This subpackage provides functions to extract the implied risk‑neutral
probability density from option price data. Two primary approaches are
implemented: the Breeden–Litzenberger finite difference method and the
COS Fourier expansion method.
"""

from .rnd import breeden_litzenberger, cos_density

__all__ = [
    "breeden_litzenberger",
    "cos_density",
]