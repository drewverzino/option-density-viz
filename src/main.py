"""Command-line interface for option-density-viz.

This script provides a simple entry point for running the various steps
involved in estimating and visualizing risk‑neutral densities from option
data. It parses command-line arguments to determine the underlying asset,
option expiry, and density estimation method, then orchestrates data
retrieval, volatility surface fitting, density computation, and plotting.

Example usage:

    python src/main.py --asset BTC --expiry 2025-09-30 --method bl

This will fetch BTC option data for the September 30, 2025 expiry, fit an
SVI smile, compute the risk‑neutral density via the Breeden–Litzenberger
method, and display a plot.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from .data import fetch_instruments, fetch_ticker
from .vol import calibrate_svi, svi_total_variance, SVIParameters
from .density import breeden_litzenberger
from .viz import plot_density


def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: List of argument strings (typically ``sys.argv[1:]``).

    Returns:
        Namespace of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Compute and plot risk‑neutral densities from Deribit option data.")
    parser.add_argument("--asset", required=True, help="Underlying asset symbol (e.g. BTC, ETH)")
    parser.add_argument("--expiry", required=True, help="Option expiry date (YYYY-MM-DD)")
    parser.add_argument("--method", default="bl", choices=["bl", "cos"], help="Density estimation method: 'bl' for Breeden-Litzenberger, 'cos' for COS method")
    parser.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    return parser.parse_args(args)


def main(argv: List[str] | None = None) -> int:
    """Main entry point for the command-line interface stub.

    This stub outlines the CLI workflow but does not implement the
    end-to-end logic. Populate this function with data fetching,
    SVI calibration, density computation, and visualization to
    create a complete application.

    Args:
        argv: Optional list of argument strings. If None, uses ``sys.argv[1:]``.

    Returns:
        Exit status code.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))
    logger = logging.getLogger(__name__)
    logger.info(
        "CLI called with asset %s, expiry %s, method %s",
        args.asset,
        args.expiry,
        args.method,
    )
    raise NotImplementedError(
        "main has not been implemented. Add data retrieval, SVI calibration, density computation, and plotting here."
    )


if __name__ == "__main__":
    sys.exit(main())