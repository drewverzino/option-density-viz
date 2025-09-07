# src/__init__.py
"""
Option Viz (src-layout package root).

This file is optional in a src-layout repo. We keep it lean and provide
convenience re-exports so you *may* do:

    import src as ov
    chain = await ov.get_fetcher("equity").fetch_chain("AAPL", expiry)

But the recommended import style in this repo remains:

    from data.registry import get_fetcher
    from preprocess.midprice import add_midprice_columns
    ...

Notes
-----
- The project uses a `src/` layout. Ensure the `src` directory is on `PYTHONPATH`
  (the notebooks do this automatically).
- Subpackages like `data`, `preprocess`, `vol`, `density`, `viz` are meant to be
  imported directly (e.g., `from data.base import OptionChain`). This file does
  not change that.
"""

__all__ = [
    # Top-level types
    "OptionQuote", "OptionChain", "OptionFetcher",
    # Factory / providers
    "get_fetcher", "RiskFreeProvider", "RiskFreeConfig",
    # Preprocess convenience
    "preprocess",
]

__version__ = "0.1.0"

# Re-export core data types & factory
try:
    from data.base import OptionQuote, OptionChain, OptionFetcher
except Exception:  # pragma: no cover - guard for partial installs
    OptionQuote = OptionChain = OptionFetcher = None  # type: ignore

try:
    from data.registry import get_fetcher
except Exception:  # pragma: no cover
    def get_fetcher(*args, **kwargs):  # type: ignore
        raise ImportError("data.registry.get_fetcher is not available on PYTHONPATH")

# Risk-free provider
try:
    from data.risk_free import RiskFreeProvider, RiskFreeConfig
except Exception:  # pragma: no cover
    RiskFreeProvider = RiskFreeConfig = None  # type: ignore

# Namespace import for preprocess (so users can do `src.preprocess.add_midprice_columns` if they want)
try:
    from . import preprocess  # type: ignore
except Exception:  # pragma: no cover
    preprocess = None  # type: ignore
