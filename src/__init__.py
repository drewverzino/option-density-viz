"""
Top-level package exports for option-density-viz.

These re-exports make common imports concise, e.g.:

    from option_density_viz import (
        OptionQuote, OptionChain, get_fetcher,
        add_midprice_columns, estimate_forward_from_chain,
        calibrate_svi_from_quotes, fit_surface_from_frames,
        bl_pdf_from_calls, build_cdf,
    )

Keep lines < 79 chars to satisfy linters.
"""

from __future__ import annotations

# Data layer
from .data.base import OptionChain, OptionFetcher, OptionQuote
from .data.cache import KVCache
from .data.historical_loader import (
    chain_to_dataframe,
    dataframe_to_chain,
    load_chain_csv,
    load_chain_parquet,
    save_chain_csv,
    save_chain_parquet,
)
from .data.okx_fetcher import OKXFetcher  # crypto (public endpoints)
from .data.polygon_fetcher import (  # equity/crypto via Polygon.io
    PolygonFetcher,
)
from .data.registry import get_fetcher
from .data.risk_free import RiskFreeConfig, RiskFreeProvider
from .data.yf_fetcher import YFinanceFetcher  # equity

# Density & CDF helpers
from .density import (
    BLDiagnostics,
    bl_pdf_from_calls,
    bl_pdf_from_calls_nonuniform,
    build_cdf,
    moments_from_pdf,
    ppf_from_cdf,
)
from .preprocess.forward import (
    estimate_forward_from_chain,
    estimate_forward_from_pcp,
    forward_from_carry,
    log_moneyness,
    yearfrac,
)

# Preprocess
from .preprocess.midprice import add_midprice_columns
from .preprocess.pcp import (
    pivot_calls_puts_by_strike,
    residuals_from_parity,
    synth_call_from_put,
    synth_put_from_call,
    synthesize_missing_leg,
)
from .vol.no_arb import butterfly_violations, calendar_violations
from .vol.surface import fit_surface_from_frames, smooth_params

# Volatility modelling
from .vol.svi import SVIFit, calibrate_svi_from_quotes

__all__ = [
    # data
    "OKXFetcher",
    "YFinanceFetcher",
    "PolygonFetcher",
    "OptionQuote",
    "OptionChain",
    "OptionFetcher",
    "get_fetcher",
    "chain_to_dataframe",
    "dataframe_to_chain",
    "save_chain_csv",
    "save_chain_parquet",
    "load_chain_csv",
    "load_chain_parquet",
    "KVCache",
    "RiskFreeProvider",
    "RiskFreeConfig",
    # preprocess
    "add_midprice_columns",
    "pivot_calls_puts_by_strike",
    "residuals_from_parity",
    "synthesize_missing_leg",
    "synth_put_from_call",
    "synth_call_from_put",
    "yearfrac",
    "log_moneyness",
    "forward_from_carry",
    "estimate_forward_from_chain",
    "estimate_forward_from_pcp",
    # vol
    "SVIFit",
    "calibrate_svi_from_quotes",
    "butterfly_violations",
    "calendar_violations",
    "fit_surface_from_frames",
    "smooth_params",
    # density
    "bl_pdf_from_calls",
    "bl_pdf_from_calls_nonuniform",
    "BLDiagnostics",
    "build_cdf",
    "ppf_from_cdf",
    "moments_from_pdf",
]
