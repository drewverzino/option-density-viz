"""Plotting functions for option-density-viz.

This module provides functions to visualize risk‑neutral densities
obtained from option prices. Both Matplotlib and Plotly are supported.
"""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def plot_density(
    strikes: Iterable[float],
    density: Iterable[float],
    *,
    title: str = "Risk‑Neutral Density",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Create a static plot of the risk‑neutral density using Matplotlib.

    This stub defines the API for producing a Matplotlib plot of a
    risk‑neutral density. Users should implement the plotting logic,
    including axis labels and titles.

    Args:
        strikes: Sequence of strike prices at which the density is evaluated.
        density: Corresponding density values.
        title: Plot title.
        ax: Optional Matplotlib axes object to draw on. If None, a new figure
            and axes should be created.

    Returns:
        The Matplotlib axes containing the plot.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError(
        "plot_density has not been implemented. Use matplotlib to plot the density curve."
    )


def plot_density_interactive(
    strikes: Iterable[float],
    density: Iterable[float],
    *,
    title: str = "Risk‑Neutral Density",
) -> go.Figure:
    """Create an interactive Plotly figure of the risk‑neutral density.

    This stub defines the interface for generating an interactive Plotly
    figure. Implementers should instantiate a ``go.Figure``, add a line
    trace, and set axis labels and titles as appropriate.

    Args:
        strikes: Sequence of strike prices.
        density: Sequence of density values.
        title: Title of the plot.

    Returns:
        A Plotly Figure object that can be shown in a Jupyter notebook or exported.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError(
        "plot_density_interactive has not been implemented. Use plotly.graph_objects to build an interactive plot."
    )
