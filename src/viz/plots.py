"""
Matplotlib plotting helpers for Option Viz.

This module intentionally has NO heavy dependencies beyond NumPy and Matplotlib.
It focuses on two core visuals:
  1) Implied-volatility smiles (IV vs log-moneyness k)
  2) Risk-neutral PDF/CDF from the BL pipeline

Design goals
------------
- Minimal inputs: plain NumPy arrays or simple Python lists.
- Friendly defaults: readable labels, grid, and optional titles.
- Light/dark themes via Matplotlib style contexts.
- Optional file saving with `save_path` (PNG/SVG/PDF inferred by extension).
- Return the `matplotlib.figure.Figure` so callers can further customize.

Notes
-----
- We do not force colors/styles; we rely on Matplotlib defaults so your global
  style or notebook theme can drive the look.
- For PDF/CDF, we use a twin y-axis (left=PDF, right=CDF).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Set up module logger
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


@contextmanager
def _theme_context(theme: str = "light"):
    """
    Apply a temporary Matplotlib style for light/dark themes.

    - "light": default Matplotlib style (no override)
    - "dark" : use 'dark_background' for better contrast

    Example:
        with _theme_context("dark"):
            ... draw plots ...
    """
    logger.debug(f"Applying theme: {theme}")

    if str(theme).lower() == "dark":
        logger.debug("Using dark_background style")
        with plt.style.context("dark_background"):
            yield
    else:
        logger.debug("Using default matplotlib style")
        # Keep current style (respect user's global rcParams)
        yield


def _maybe_save(
    fig: "plt.Figure",
    save_path: Optional[str | Path],
    *,
    dpi: int = 160,
    tight: bool = True,
) -> None:
    """
    Save the figure to disk if `save_path` is provided.

    The image format is inferred from the file extension (e.g., .png, .svg, .pdf).
    """
    if not save_path:
        logger.debug("No save path provided, skipping file save")
        return

    p = Path(save_path)
    logger.debug(f"Saving figure to: {p}")

    # Create directory if needed
    p.parent.mkdir(parents=True, exist_ok=True)

    # Log format information
    format_ext = p.suffix.lower()
    logger.debug(f"Detected format from extension: {format_ext}")

    try:
        if tight:
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(p, dpi=dpi)

        # Log file size if possible
        try:
            file_size = p.stat().st_size
            logger.debug(f"Saved figure: {file_size} bytes at {dpi} DPI")
        except Exception:
            logger.debug(f"Figure saved successfully")

    except Exception as e:
        logger.error(f"Failed to save figure to {p}: {e}")
        raise


def _as_float_array(x) -> np.ndarray:
    """Convert to a 1D float NumPy array and drop NaNs."""
    arr = np.asarray(x, dtype=float).ravel()
    original_size = arr.size

    m = np.isfinite(arr)
    arr = arr[m]

    filtered_size = arr.size
    if filtered_size < original_size:
        logger.debug(
            f"Filtered array: {original_size} → {filtered_size} points "
            f"({original_size - filtered_size} non-finite values removed)"
        )

    if filtered_size == 0:
        logger.warning("Array contains no finite values after filtering")
    else:
        logger.debug(f"Array range: [{arr.min():.6f}, {arr.max():.6f}]")

    return arr


def _sort_by_first(x: np.ndarray, *ys: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Sort arrays by the first array's ascending order, applying the same permutation
    to all `ys`. Returns a tuple (x_sorted, *ys_sorted).
    """
    if x.size == 0:
        logger.debug("Empty array provided to sort function")
        return (x, *ys)

    logger.debug(f"Sorting {len(ys) + 1} arrays by first array")

    # Check if already sorted
    is_sorted = np.all(np.diff(x) >= 0)
    if is_sorted:
        logger.debug("Arrays already sorted")
        return (x, *ys)

    order = np.argsort(x)
    logger.debug(f"Applied sorting permutation to {len(ys)} additional arrays")

    return (x[order],) + tuple(y[order] if y.size == x.size else y for y in ys)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_smile(
    k: Iterable[float],
    *,
    iv: Optional[Iterable[float]] = None,
    w: Optional[Iterable[float]] = None,
    T: Optional[float] = None,
    title: Optional[str] = None,
    theme: str = "light",
    save_path: Optional[str | Path] = None,
) -> "plt.Figure":
    """
    Plot an implied-volatility smile vs log-moneyness k.

    Parameters
    ----------
    k : array-like
        Log-moneyness points (k = ln(K/F)). Will be sorted ascending.
    iv : array-like, optional
        Implied volatilities corresponding to k (annualized). Provide either `iv` OR `w`.
    w : array-like, optional
        Total variance points w(k) = sigma(k)^2 * T. Provide either `iv` OR `w`.
    T : float, optional
        Time-to-maturity in years. Only required if `w` is provided so we can compute IV = sqrt(w / T).
    title : str, optional
    theme : {"light","dark"}
    save_path : str or Path, optional
        If provided, save the figure to this path (format inferred from extension).

    Returns
    -------
    matplotlib.figure.Figure
    """
    logger.info("Creating IV smile plot")
    logger.debug(f"Parameters: T={T}, theme={theme}, title='{title}'")

    k_arr = _as_float_array(k)
    logger.debug(f"Log-moneyness: {len(k_arr)} points")

    # Input validation
    if iv is not None and w is not None:
        logger.error("Both iv and w provided - ambiguous input")
        raise ValueError("Provide either `iv` or `w`, not both.")
    if iv is None and w is None:
        logger.error("Neither iv nor w provided")
        raise ValueError("Provide `iv` or `w` to plot a smile.")

    # Determine IV vector
    if w is not None:
        logger.debug("Converting total variance to implied volatility")
        if T is None or T <= 0.0:
            logger.error(f"Invalid time to maturity: T={T}")
            raise ValueError(
                "T must be provided and > 0 when plotting from total variance `w`."
            )
        w_arr = _as_float_array(w)
        if w_arr.size != k_arr.size:
            logger.error(
                f"Size mismatch: k has {k_arr.size} points, w has {w_arr.size} points"
            )
            raise ValueError("`k` and `w` must have the same length.")

        # Check for negative total variance
        negative_w = (w_arr < 0).sum()
        if negative_w > 0:
            logger.warning(
                f"Found {negative_w} negative total variance values"
            )

        iv_arr = np.sqrt(np.maximum(w_arr, 0.0) / float(T))
        logger.debug(
            f"Computed IV from total variance: range [{iv_arr.min():.4f}, {iv_arr.max():.4f}]"
        )
    else:
        logger.debug("Using provided implied volatility values")
        iv_arr = _as_float_array(iv)
        if iv_arr.size != k_arr.size:
            logger.error(
                f"Size mismatch: k has {k_arr.size} points, iv has {iv_arr.size} points"
            )
            raise ValueError("`k` and `iv` must have the same length.")

        logger.debug(f"IV range: [{iv_arr.min():.4f}, {iv_arr.max():.4f}]")

        # IV quality checks
        zero_iv = (iv_arr == 0).sum()
        high_iv = (iv_arr > 5.0).sum()  # >500% vol
        if zero_iv > 0:
            logger.warning(f"Found {zero_iv} zero implied volatility values")
        if high_iv > 0:
            logger.warning(f"Found {high_iv} extremely high IV values (>500%)")

    # Sort data
    k_arr, iv_arr = _sort_by_first(k_arr, iv_arr)
    logger.debug(
        f"Final data ranges: k ∈ [{k_arr.min():.3f}, {k_arr.max():.3f}], "
        f"IV ∈ [{iv_arr.min():.4f}, {iv_arr.max():.4f}]"
    )

    # Create plot
    logger.debug("Creating matplotlib figure")
    with _theme_context(theme):
        fig, ax = plt.subplots(figsize=(6.8, 4.0))

        line = ax.plot(k_arr, iv_arr, marker="o", linestyle="-")
        logger.debug(f"Plotted {len(k_arr)} points with markers and line")

        ax.set_xlabel("log-moneyness  k = ln(K/F)")
        ax.set_ylabel("implied volatility (annualized)")
        ax.grid(True, alpha=0.3)

        if title:
            ax.set_title(title)
            logger.debug(f"Set plot title: '{title}'")

        # Log axis ranges
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        logger.debug(
            f"Plot ranges: x=[{xlim[0]:.3f}, {xlim[1]:.3f}], y=[{ylim[0]:.4f}, {ylim[1]:.4f}]"
        )

        _maybe_save(fig, save_path)

    logger.info("IV smile plot created successfully")
    return fig


def plot_pdf_cdf(
    K: Iterable[float],
    pdf: Iterable[float],
    *,
    cdf: Optional[Iterable[float]] = None,
    marks: Optional[Dict[str, float]] = None,
    title: Optional[str] = None,
    theme: str = "light",
    save_path: Optional[str | Path] = None,
) -> "plt.Figure":
    """
    Plot risk-neutral PDF (left axis) and CDF (right axis) on the same figure.

    Parameters
    ----------
    K : array-like
        Strike grid.
    pdf : array-like
        Risk-neutral density values on K (should be non-negative; renormalized upstream).
    cdf : array-like, optional
        If not provided, we'll compute it via `density.build_cdf(K, pdf)`.
    marks : dict, optional
        Optional vertical markers. Example: {"mean": 430.2, "VaR(5%)": 380.1}
    title : str, optional
    theme : {"light","dark"}
    save_path : str or Path, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    logger.info("Creating PDF/CDF plot")
    logger.debug(f"Parameters: theme={theme}, title='{title}', marks={marks}")

    K_arr = _as_float_array(K)
    pdf_arr = _as_float_array(pdf)

    logger.debug(
        f"Input arrays: K has {len(K_arr)} points, PDF has {len(pdf_arr)} points"
    )

    if K_arr.size != pdf_arr.size:
        logger.error(
            f"Size mismatch: K has {K_arr.size} points, PDF has {pdf_arr.size} points"
        )
        raise ValueError("`K` and `pdf` must have the same length.")

    # Sort and clean PDF
    K_arr, pdf_arr = _sort_by_first(K_arr, pdf_arr)

    negative_pdf = (pdf_arr < 0).sum()
    if negative_pdf > 0:
        logger.warning(
            f"Found {negative_pdf} negative PDF values, clipping to zero"
        )

    pdf_arr = np.maximum(pdf_arr, 0.0)

    logger.debug(f"Strike range: [{K_arr.min():.2f}, {K_arr.max():.2f}]")
    logger.debug(f"PDF range: [{pdf_arr.min():.8f}, {pdf_arr.max():.8f}]")

    # Validate PDF properties
    pdf_integral = np.trapezoid(pdf_arr, K_arr)
    logger.debug(f"PDF integral: {pdf_integral:.6f}")
    if abs(pdf_integral - 1.0) > 0.1:
        logger.warning(f"PDF integral far from 1.0: {pdf_integral:.6f}")

    # Handle CDF
    if cdf is None:
        logger.debug("Computing CDF from PDF")
        try:
            from density import (  # late import to prevent circular deps
                build_cdf,
            )

            _, cdf_arr = build_cdf(K_arr, pdf_arr)
            logger.debug("CDF computed successfully")
        except Exception as e:
            logger.error(f"Failed to compute CDF: {e}")
            raise
    else:
        logger.debug("Using provided CDF")
        cdf_arr = _as_float_array(cdf)
        if cdf_arr.size != K_arr.size:
            logger.error(f"CDF size mismatch: {cdf_arr.size} vs {K_arr.size}")
            raise ValueError("`K` and `cdf` must have the same length.")

        K_arr, cdf_arr = _sort_by_first(K_arr, cdf_arr)
        cdf_arr = np.clip(np.maximum.accumulate(cdf_arr), 0.0, 1.0)
        logger.debug("CDF cleaned and enforced monotonicity")

    logger.debug(f"CDF range: [{cdf_arr.min():.6f}, {cdf_arr.max():.6f}]")

    # Create plot
    logger.debug("Creating dual-axis matplotlib figure")
    with _theme_context(theme):
        fig, ax_pdf = plt.subplots(figsize=(6.8, 4.0))

        # Plot PDF
        pdf_line = ax_pdf.plot(
            K_arr, pdf_arr, linestyle="-", marker=None, label="PDF"
        )
        ax_pdf.set_xlabel("strike  K")
        ax_pdf.set_ylabel("risk-neutral pdf")
        ax_pdf.grid(True, alpha=0.3)

        logger.debug("PDF plotted on left axis")

        # Plot CDF on right axis
        ax_cdf = ax_pdf.twinx()
        cdf_line = ax_cdf.plot(
            K_arr, cdf_arr, linestyle="--", marker=None, label="CDF"
        )
        ax_cdf.set_ylabel("cdf  (0..1)")

        logger.debug("CDF plotted on right axis")

        # Add vertical markers
        if marks:
            logger.debug(f"Adding {len(marks)} vertical markers")
            for label, x in marks.items():
                try:
                    xv = float(x)
                    if not (K_arr.min() <= xv <= K_arr.max()):
                        logger.warning(
                            f"Marker '{label}' at {xv:.2f} outside data range"
                        )

                    ax_pdf.axvline(xv, linestyle=":", linewidth=1.2, alpha=0.8)
                    ax_pdf.text(
                        xv,
                        ax_pdf.get_ylim()[1] * 0.95,
                        label,
                        rotation=90,
                        ha="right",
                        va="top",
                        fontsize=9,
                        alpha=0.9,
                    )
                    logger.debug(f"Added marker: {label} at {xv:.2f}")
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Invalid marker value for '{label}': {x} ({e})"
                    )
                    continue

        if title:
            ax_pdf.set_title(title)
            logger.debug(f"Set plot title: '{title}'")

        # Log axis ranges
        pdf_ylim = ax_pdf.get_ylim()
        cdf_ylim = ax_cdf.get_ylim()
        logger.debug(
            f"Plot ranges: K=[{K_arr.min():.2f}, {K_arr.max():.2f}], "
            f"PDF=[{pdf_ylim[0]:.6f}, {pdf_ylim[1]:.6f}], "
            f"CDF=[{cdf_ylim[0]:.6f}, {cdf_ylim[1]:.6f}]"
        )

        _maybe_save(fig, save_path)

    logger.info("PDF/CDF plot created successfully")
    return fig


def plot_svi_vs_market(
    k: Iterable[float],
    *,
    iv_mkt: Optional[Iterable[float]] = None,
    iv_model: Optional[Iterable[float]] = None,
    w_mkt: Optional[Iterable[float]] = None,
    w_model: Optional[Iterable[float]] = None,
    T: Optional[float] = None,
    title: Optional[str] = None,
    theme: str = "light",
    save_path: Optional[str | Path] = None,
) -> "plt.Figure":
    """
    Compare market vs SVI model on the same (k, IV) axes.

    Provide either IV directly OR total variance `w`; if you pass `w`, T must be > 0.

    Parameters
    ----------
    k : array-like
        Log-moneyness points.
    iv_mkt, iv_model : array-like, optional
        Market and model implied vol arrays.
    w_mkt, w_model : array-like, optional
        Market and model total variance arrays (require T to convert to IV).
    T : float, optional
        Time-to-maturity in years (required if passing w_*).
    title, theme, save_path : see other functions.

    Returns
    -------
    matplotlib.figure.Figure
    """
    logger.info("Creating SVI vs market comparison plot")
    logger.debug(f"Parameters: T={T}, theme={theme}, title='{title}'")

    k_arr = _as_float_array(k)
    logger.debug(f"Log-moneyness: {len(k_arr)} points")

    # Input validation
    has_market = (iv_mkt is not None) or (w_mkt is not None)
    has_model = (iv_model is not None) or (w_model is not None)

    if not has_market and not has_model:
        logger.error("No market or model data provided")
        raise ValueError(
            "Provide market and/or model data via IV OR total variance."
        )

    logger.debug(f"Data availability: market={has_market}, model={has_model}")

    def _to_iv(x, kind: str, label: str) -> Optional[np.ndarray]:
        if x is None:
            return None

        logger.debug(f"Converting {label} {kind} to IV")
        arr = _as_float_array(x)

        if kind == "w":
            if T is None or T <= 0.0:
                logger.error(f"Invalid time to maturity for {label}: T={T}")
                raise ValueError(
                    "T must be provided and > 0 when converting total variance to IV."
                )

            # Check for negative total variance
            negative_w = (arr < 0).sum()
            if negative_w > 0:
                logger.warning(
                    f"{label}: {negative_w} negative total variance values"
                )

            arr = np.sqrt(np.maximum(arr, 0.0) / float(T))
            logger.debug(f"Converted {label} total variance to IV")

        if arr.size != k_arr.size:
            logger.error(
                f"Size mismatch: k has {k_arr.size} points, {label} has {arr.size} points"
            )
            raise ValueError("`k` and series length mismatch.")

        logger.debug(f"{label} IV range: [{arr.min():.4f}, {arr.max():.4f}]")
        return arr

    # Convert to IV
    m_iv = (
        _to_iv(iv_mkt, "iv", "market")
        if iv_mkt is not None
        else _to_iv(w_mkt, "w", "market")
    )
    f_iv = (
        _to_iv(iv_model, "iv", "model")
        if iv_model is not None
        else _to_iv(w_model, "w", "model")
    )

    # Sort data consistently
    if m_iv is not None:
        k_arr, m_iv = _sort_by_first(k_arr, m_iv)
        logger.debug("Sorted market data by log-moneyness")

    if f_iv is not None:
        # Apply same sorting to model data
        order = np.argsort(k_arr)
        if f_iv.size == k_arr.size:
            f_iv = f_iv[order]
            logger.debug("Applied same sorting to model data")

    # Create comparison plot
    logger.debug("Creating comparison matplotlib figure")
    with _theme_context(theme):
        fig, ax = plt.subplots(figsize=(6.8, 4.0))

        plots_added = 0

        # Plot market data as points
        if m_iv is not None:
            market_plot = ax.plot(
                k_arr,
                m_iv,
                marker="o",
                linestyle="None",
                label="market",
                markersize=4,
                alpha=0.8,
            )
            plots_added += 1
            logger.debug(f"Plotted {len(k_arr)} market IV points")

        # Plot model as line
        if f_iv is not None:
            model_plot = ax.plot(
                k_arr,
                f_iv,
                linestyle="-",
                linewidth=2,
                label="SVI model",
                alpha=0.9,
            )
            plots_added += 1
            logger.debug(f"Plotted SVI model line with {len(k_arr)} points")

        ax.set_xlabel("log-moneyness  k = ln(K/F)")
        ax.set_ylabel("implied volatility (annualized)")
        ax.grid(True, alpha=0.3)

        if plots_added > 1:
            ax.legend(loc="best")
            logger.debug("Added legend for multiple data series")

        if title:
            ax.set_title(title)
            logger.debug(f"Set plot title: '{title}'")

        # Log fit quality if both series available
        if m_iv is not None and f_iv is not None:
            mse = np.mean((m_iv - f_iv) ** 2)
            mae = np.mean(np.abs(m_iv - f_iv))
            logger.debug(f"Model fit quality: MSE={mse:.8f}, MAE={mae:.6f}")

            if mae > 0.1:  # 10% average error
                logger.warning(
                    f"Large model-market IV differences: MAE={mae:.4f}"
                )

        # Log axis ranges
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        logger.debug(
            f"Plot ranges: k=[{xlim[0]:.3f}, {xlim[1]:.3f}], IV=[{ylim[0]:.4f}, {ylim[1]:.4f}]"
        )

        _maybe_save(fig, save_path)

    logger.info("SVI vs market comparison plot created successfully")
    return fig
