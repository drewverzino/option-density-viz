"""SVI volatility smile model for option-density-viz.

This module provides a minimal implementation of the Stochastic Volatility
Inspired (SVI) model for fitting implied volatility smiles. The SVI
parameterization is widely used in practice because it can enforce
arbitrage‑free conditions and capture typical smile shapes with only a
handful of parameters.

Reference:
    Jim Gatheral and Antoine Jacquier (2011), ``Arbitrage‑Free SVI Volatility
    Surfaces``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class SVIParameters:
    """Data structure for the five SVI parameters.

    Attributes:
        a: Represents the minimum total variance.
        b: Controls the overall slope of the smile.
        rho: Determines the skewness (correlation) of the smile.
        m: Horizontal shift parameter (location of minimum).
        sigma: Controls the curvature (width) of the smile.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_total_variance(k: np.ndarray, params: SVIParameters) -> np.ndarray:
    """Compute the SVI total variance for log strikes ``k``.

    This placeholder function defines the interface for evaluating the SVI
    total variance curve. Users should implement the computation
    according to the Gatheral-Jacquier formulation:

    .. code-block:: python

        w(k) = a + b * (rho * (k - m) + sqrt((k - m)**2 + sigma**2))

    Args:
        k: An array of log‑moneyness values (log(K/F)).
        params: Fitted SVI parameters.

    Returns:
        An array of total variances ``w(k)`` for each log strike.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError(
        "svi_total_variance has not been implemented. Implement the SVI total variance formula here."
    )


def calibrate_svi(k: np.ndarray, w: np.ndarray) -> SVIParameters:
    """Calibrate the SVI parameters to observed total variance data.

    This function should take arrays of log‑moneyness values and total variances
    and return a set of SVI parameters that best fit the observed smile.
    While a complete implementation typically involves optimization with
    constraints to enforce no arbitrage, this stub provides only the
    function signature and documentation.

    Args:
        k: Log‑moneyness array (log(K/F)).
        w: Observed total variance array (``sigma_imp**2 * T``).

    Returns:
        SVIParameters: The fitted parameters.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError(
        "calibrate_svi has not been implemented. Implement an optimizer to fit SVI parameters to the data."
    )
