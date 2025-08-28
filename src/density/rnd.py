"""Risk‑neutral density computation functions.

This module implements methods to recover the risk‑neutral probability density
from a set of option prices. The primary methods included are the
Breeden–Litzenberger finite difference approach and the COS Fourier
expansion method. These functions take as input arrays of strikes and
option prices and return approximations to the underlying density.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple


def breeden_litzenberger(k: np.ndarray, call_prices: np.ndarray, r: float, T: float) -> np.ndarray:
    """Compute the risk‑neutral density via the Breeden–Litzenberger formula.

    This stub defines the interface for deriving the risk‑neutral probability
    density from option prices using finite differences. Users should
    implement the calculation of the second derivative of the call price
    with respect to strike and apply the appropriate discount factor.

    Args:
        k: Array of strike prices (must be sorted ascending).
        call_prices: Corresponding call option prices (same length as ``k``).
        r: Continuously compounded risk‑free rate.
        T: Time to maturity in years.

    Returns:
        Array of estimated density values at the midpoints of the strike grid.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError(
        "breeden_litzenberger has not been implemented. Implement the finite difference approximation of the second derivative here."
    )


def cos_density(characteristic_func, x: np.ndarray, N: int, a: float, b: float) -> np.ndarray:
    """Compute the density using the COS method.

    This stub outlines the COS method for inverting a characteristic
    function to obtain a probability density. To complete the
    implementation, follow these steps:

    1. Compute the frequencies ``u_k = k * pi / (b - a)`` for ``k = 0, …, N-1``.
    2. Evaluate the characteristic function at ``u_k``.
    3. Precompute cosine terms ``cos((x - a) * u_k)``.
    4. Assemble the density via the cosine series expansion.

    Args:
        characteristic_func: Callable returning the characteristic function \phi(u).
        x: Points at which to evaluate the density.
        N: Number of terms in the COS expansion.
        a, b: Integration range bounds for the truncated domain.

    Returns:
        Array of density values at ``x``.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError(
        "cos_density has not been implemented. Implement the COS method for density recovery here."
    )
