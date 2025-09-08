# src/cli/main.py
"""
Option Viz CLI
==============

End-to-end pipeline:
    data → preprocess → SVI fit → BL density → plots + artifacts

Artifacts written to --out:
    - chain.csv                : normalized quotes for the selected expiry
    - results.json             : diagnostics (SVI, BL, counts)
    - smile_market.png         : market smile (IV vs log-moneyness)
    - smile_model.png          : SVI model vs market IV curve
    - density_pdf_cdf.png      : RN PDF & CDF with markers

Notes
-----
- Uses OKX (public) for crypto and yfinance for equities.
- Run with:  python -m src.cli.main --asset-class crypto --underlying BTC --expiry 2025-09-26 --out docs/run_btc
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np

# ---------- Logging setup ----------------


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels."""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        # Only colorize if we're outputting to a terminal
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, '')
            levelname_colored = f"{color}{record.levelname}{self.RESET}"
            # Temporarily replace levelname for formatting
            original_levelname = record.levelname
            record.levelname = levelname_colored
            formatted = super().format(record)
            record.levelname = original_levelname  # Restore original
            return formatted
        else:
            return super().format(record)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup structured logging with timestamps and colors."""
    logger = logging.getLogger("oviz")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler with colored formatting
    handler = logging.StreamHandler(sys.stdout)
    formatter = ColoredFormatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S',
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Quiet external libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    return logger


logger = setup_logging()

# ---------- Flexible imports (package style first, then flat) ----------------


def _import_modules():
    """
    Import project modules whether we run as `python -m src.cli.main`
    (package-style) or via a flat PYTHONPATH pointing at ./src.

    Forward estimator note:
      - It lives in preprocess.forward (not preprocess.pcp).
      - Name may be `estimate_forward_from_pcp` or `estimate_forward_from_chain`.
        We try both and alias to `estimate_forward`.
    """
    logger.debug("Importing project modules...")
    try:
        # Package-style imports (preferred)
        from ..data.historical_loader import (  # type: ignore
            chain_to_dataframe,
            save_chain_csv,
        )
        from ..data.registry import get_fetcher  # type: ignore
        from ..data.risk_free import (  # type: ignore
            RiskFreeConfig,
            RiskFreeProvider,
        )
        from ..preprocess.forward import yearfrac  # type: ignore
        from ..preprocess.midprice import add_midprice_columns  # type: ignore

        try:
            from ..preprocess.forward import (
                estimate_forward_from_pcp,  # type: ignore
            )

            estimate_forward = estimate_forward_from_pcp
            logger.debug("Using estimate_forward_from_pcp")
        except Exception:
            from ..preprocess.forward import (
                estimate_forward_from_chain,  # type: ignore
            )

            estimate_forward = estimate_forward_from_chain
            logger.debug("Using estimate_forward_from_chain")

        from ..density import (  # type: ignore
            bl_pdf_from_calls,
            build_cdf,
            moments_from_pdf,
        )
        from ..viz import (  # type: ignore
            plot_pdf_cdf,
            plot_smile,
            plot_svi_vs_market,
        )
        from ..vol.svi import (  # type: ignore
            SVIFit,
            calibrate_svi_from_quotes,
            svi_total_variance,
        )

        logger.debug("Successfully imported modules (package-style)")
    except Exception as e:
        logger.debug(
            f"Package-style import failed: {e}, trying flat imports..."
        )
        # Flat-style imports (PYTHONPATH includes ./src)
        from data.historical_loader import (  # type: ignore
            chain_to_dataframe,
            save_chain_csv,
        )
        from data.registry import get_fetcher  # type: ignore
        from data.risk_free import (  # type: ignore
            RiskFreeConfig,
            RiskFreeProvider,
        )
        from preprocess.forward import yearfrac  # type: ignore
        from preprocess.midprice import add_midprice_columns  # type: ignore

        try:
            from preprocess.forward import (
                estimate_forward_from_pcp,  # type: ignore
            )

            estimate_forward = estimate_forward_from_pcp
            logger.debug("Using estimate_forward_from_pcp")
        except Exception:
            from preprocess.forward import (
                estimate_forward_from_chain,  # type: ignore
            )

            estimate_forward = estimate_forward_from_chain
            logger.debug("Using estimate_forward_from_chain")

        from density import (  # type: ignore
            bl_pdf_from_calls,
            build_cdf,
            moments_from_pdf,
        )
        from viz import (  # type: ignore
            plot_pdf_cdf,
            plot_smile,
            plot_svi_vs_market,
        )
        from vol.svi import (  # type: ignore
            SVIFit,
            calibrate_svi_from_quotes,
            svi_total_variance,
        )

        logger.debug("Successfully imported modules (flat-style)")

    return {
        "get_fetcher": get_fetcher,
        "chain_to_dataframe": chain_to_dataframe,
        "save_chain_csv": save_chain_csv,
        "RiskFreeProvider": RiskFreeProvider,
        "RiskFreeConfig": RiskFreeConfig,
        "add_midprice_columns": add_midprice_columns,
        "yearfrac": yearfrac,
        "estimate_forward": estimate_forward,
        "calibrate_svi_from_quotes": calibrate_svi_from_quotes,
        "svi_total_variance": svi_total_variance,
        "SVIFit": SVIFit,
        "bl_pdf_from_calls": bl_pdf_from_calls,
        "build_cdf": build_cdf,
        "moments_from_pdf": moments_from_pdf,
        "plot_smile": plot_smile,
        "plot_pdf_cdf": plot_pdf_cdf,
        "plot_svi_vs_market": plot_svi_vs_market,
    }


M = _import_modules()

# ------------------------------ CLI args ------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="oviz",
        description="Option Viz: data → preprocess → SVI → BL density → plots",
    )
    p.add_argument(
        "--asset-class",
        choices=["equity", "crypto"],
        required=True,
        help="Data backend to use: equity (yfinance) or crypto (OKX).",
    )
    p.add_argument(
        "--underlying", required=True, help="Ticker/symbol, e.g., AAPL or BTC."
    )
    p.add_argument(
        "--expiry",
        required=True,
        help="Target expiry date (YYYY-MM-DD). We'll match the exact date.",
    )
    p.add_argument(
        "--out", default="docs/run", help="Output directory for artifacts."
    )
    p.add_argument(
        "--theme",
        choices=["light", "dark"],
        default="light",
        help="Plot theme.",
    )
    p.add_argument(
        "--min-points", type=int, default=7, help="Min IV points for SVI fit."
    )
    p.add_argument(
        "--grid-n", type=int, default=401, help="Grid size for BL density."
    )
    p.add_argument(
        "--skip-density", action="store_true", help="Skip BL density step."
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level.",
    )
    return p.parse_args()


# ------------------------------ Helpers -------------------------------------


def _iso_to_date(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception as e:
        raise ValueError(f"Invalid --expiry {s!r}; expected YYYY-MM-DD") from e


def _match_expiry(expiries: list[datetime], want: datetime) -> datetime:
    logger.debug(
        f"Matching expiry {want.date()} from {len(expiries)} available expiries"
    )
    for e in expiries:
        if e.date() == want.date():
            logger.debug(f"Found exact match: {e}")
            return e

    available_str = [x.date().isoformat() for x in expiries[:10]]
    if len(expiries) > 10:
        available_str.append("...")

    raise ValueError(
        f"Expiry {want.date().isoformat()} not found; "
        f"available: {available_str}"
    )


def _log_moneyness(K: np.ndarray, F: float) -> np.ndarray:
    F = float(F)
    if F <= 0.0:
        raise ValueError(
            "Forward F must be positive to compute log-moneyness."
        )
    return np.log(np.maximum(K, 1e-12) / F)


def _save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.debug(f"Saved JSON: {path}")


async def _maybe_aclose(fetcher) -> None:
    """
    Best-effort cleanup for async clients (e.g., OKXFetcher with httpx.AsyncClient).
    Works whether `aclose()` is coroutine or `close()` is sync.
    """
    fn = getattr(fetcher, "aclose", None) or getattr(fetcher, "close", None)
    if not fn:
        return
    res = fn()
    if inspect.iscoroutine(res):
        await res
    logger.debug("Closed async fetcher")


# ------------------------------ Main (async) --------------------------------


async def main() -> None:
    args = _parse_args()

    # Update logging level
    logger.setLevel(getattr(logging, args.log_level))

    logger.info(f"Starting Option Viz CLI")
    logger.info(
        f"Target: {args.underlying} ({args.asset_class}) expiry {args.expiry}"
    )
    logger.info(f"Output directory: {args.out}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data fetch (single event loop for entire run)
    logger.info("Step 1/9: Fetching option chain data...")
    fetcher = M["get_fetcher"](args.asset_class)
    try:
        logger.debug(f"Listing expiries for {args.underlying}")
        expiries = sorted(await fetcher.list_expiries(args.underlying))
        logger.info(f"Found {len(expiries)} available expiries")

        target = _match_expiry(expiries, _iso_to_date(args.expiry))
        logger.info(f"Target expiry: {target}")

        logger.debug("Fetching option chain...")
        chain = await fetcher.fetch_chain(args.underlying, target)
        asof = chain.asof_utc
        logger.info(
            f"Fetched chain as of {asof} with {len(chain.quotes)} quotes"
        )

        df = M["chain_to_dataframe"](chain)
        logger.debug(
            f"Converted to DataFrame: {df.shape[0]} rows, {df.shape[1]} columns"
        )

        # Save normalized chain early (raw snapshot)
        logger.debug("Saving raw chain to CSV...")
        M["save_chain_csv"](out_dir / "chain.csv", chain)

        # 2) Preprocess (mids & flags)
        logger.info("Step 2/9: Preprocessing quotes (midprices, flags)...")
        df = M["add_midprice_columns"](df, wide_rel_threshold=0.15)

        n_crossed = df["crossed"].sum() if "crossed" in df.columns else 0
        n_wide = df["wide"].sum() if "wide" in df.columns else 0
        logger.info(
            f"Quote quality: {n_crossed} crossed, {n_wide} wide spreads"
        )

        # 3) Risk-free rate and time to maturity
        logger.info(
            "Step 3/9: Computing risk-free rate and time to maturity..."
        )
        rf = M["RiskFreeProvider"](M["RiskFreeConfig"]())
        try:
            r = float(rf.get_rate(asof.date()))
            logger.info(f"Risk-free rate: {r:.4f} ({r*100:.2f}%)")
        except Exception as e:
            r = 0.0
            logger.warning(f"Failed to get risk-free rate, using 0.0: {e}")

        T = float(M["yearfrac"](asof, target))
        logger.info(f"Time to maturity: {T:.4f} years ({T*365:.1f} days)")

        # 4) Forward estimate (prefer PCP-based estimator)
        logger.info("Step 4/9: Estimating forward price...")
        spot = chain.spot
        logger.debug(f"Spot price: {spot}")

        try:
            F = float(M["estimate_forward"](df, r=r, T=T, min_pairs=3))
            logger.info(f"Forward price (from PCP): {F:.4f}")
        except Exception as e:
            F = float(spot * math.exp(r * T))
            logger.warning(
                f"PCP forward estimation failed, using spot*exp(rT): {F:.4f} | Error: {e}"
            )

        # 5) Build K, k and collect market IVs (calls only)
        logger.info(
            "Step 5/9: Filtering calls and extracting implied volatilities..."
        )
        calls_df = df.copy()
        if "type" in calls_df.columns:
            t = calls_df["type"].astype(str).str.upper().str.strip()
            calls_df = calls_df[t.str.startswith("C")]
        calls_df = calls_df.dropna(subset=["strike", "mid"])

        logger.info(
            f"Found {len(calls_df)} call options with valid strikes and midprices"
        )

        K = calls_df["strike"].to_numpy(float)
        C_mid = calls_df["mid"].to_numpy(float)
        k = _log_moneyness(K, F)

        logger.debug(f"Strike range: {K.min():.2f} to {K.max():.2f}")
        logger.debug(f"Log-moneyness range: {k.min():.3f} to {k.max():.3f}")

        iv = None
        if "iv" in calls_df.columns:
            iv = calls_df["iv"].to_numpy(float)
            logger.debug("Using 'iv' column for implied volatilities")
        elif "extra" in calls_df.columns:
            try:
                iv = np.array(
                    [
                        float((x or {}).get("iv", np.nan))
                        for x in calls_df["extra"]
                    ]
                )
                logger.debug("Extracted IVs from 'extra' column")
            except Exception as e:
                iv = None
                logger.debug(f"Failed to extract IVs from 'extra': {e}")

        # Finite IV points for SVI
        if iv is not None:
            mask = np.isfinite(iv) & np.isfinite(k) & (iv > 0)
            k_mkt = k[mask]
            iv_mkt = iv[mask]
            logger.info(f"Market IV data: {len(iv_mkt)} valid points")
            if len(iv_mkt) > 0:
                logger.info(
                    f"IV range: {iv_mkt.min():.3f} to {iv_mkt.max():.3f}"
                )
        else:
            k_mkt = np.array([])
            iv_mkt = np.array([])
            logger.warning("No implied volatility data available")

        # 6) SVI fit
        logger.info("Step 6/9: Fitting SVI volatility smile...")
        w_model = None
        notes = ""
        min_points = max(3, args.min_points)

        if k_mkt.size >= min_points:
            logger.debug(
                f"Attempting SVI fit with {k_mkt.size} points (min required: {min_points})"
            )
            try:
                fit = M["calibrate_svi_from_quotes"](
                    k=k_mkt, iv=iv_mkt, T=T, weights=None
                )
                a, b, rho, m0, sig = fit.params  # type: ignore[attr-defined]
                w_model = M["svi_total_variance"](k_mkt, a, b, rho, m0, sig)

                svi_dict = {
                    "params": list(map(float, fit.params)),
                    "loss": float(getattr(fit, "loss", float("nan"))),
                    "n_used": int(getattr(fit, "n_used", k_mkt.size)),
                    "method": str(getattr(fit, "method", "L-BFGS-B")),
                }
                logger.info(f"SVI fit successful: loss={svi_dict['loss']:.6f}")
                logger.debug(
                    f"SVI params: {[f'{p:.4f}' for p in svi_dict['params']]}"
                )
            except Exception as e:
                svi_dict = {"error": repr(e)}
                notes = f"SVI fit failed: {e!r}"
                logger.error(f"SVI fit failed: {e}")
        else:
            svi_dict = {
                "note": f"Insufficient IV points: {k_mkt.size} < {min_points}"
            }
            logger.warning(
                f"Skipping SVI fit: insufficient data points ({k_mkt.size} < {min_points})"
            )

        # 7) BL density
        logger.info(
            "Step 7/9: Computing Breeden-Litzenberger risk-neutral density..."
        )
        bl_dict = None
        K_pdf = None
        pdf = None
        cdf = None

        if (not args.skip_density) and (K.size >= 3):
            logger.debug(
                f"Computing BL density with {K.size} call prices, grid size {args.grid_n}"
            )
            try:
                K_pdf, pdf, pdf_diag = M["bl_pdf_from_calls"](
                    strikes=K,
                    calls=C_mid,
                    r=float(r),
                    T=float(T),
                    grid_n=args.grid_n,
                    clip_negative=True,
                    renormalize=True,
                )
                _, cdf = M["build_cdf"](K_pdf, pdf)

                bl_dict = {
                    "integral": float(
                        getattr(pdf_diag, "integral", float("nan"))
                    ),
                    "neg_frac": float(
                        getattr(pdf_diag, "neg_frac", float("nan"))
                    ),
                    "rn_mean": float(
                        getattr(pdf_diag, "rn_mean", float("nan"))
                    ),
                    "rn_var": float(getattr(pdf_diag, "rn_var", float("nan"))),
                    "note": str(getattr(pdf_diag, "note", "")),
                }

                logger.info(
                    f"BL density computed: integral={bl_dict['integral']:.4f}, "
                    f"RN mean={bl_dict['rn_mean']:.2f}"
                )

                if bl_dict['neg_frac'] > 0.01:
                    logger.warning(
                        f"High negative density fraction: {bl_dict['neg_frac']:.4f}"
                    )

            except Exception as e:
                bl_dict = {"error": repr(e)}
                notes += f" | BL density failed: {e!r}"
                logger.error(f"BL density computation failed: {e}")
        elif args.skip_density:
            logger.info("Skipping BL density computation (--skip-density)")
        else:
            logger.warning(
                f"Insufficient call data for BL density: {K.size} < 3"
            )

        # 8) Plots
        logger.info("Step 8/9: Generating plots...")
        plots_created = 0

        if k_mkt.size:
            logger.debug("Creating market IV smile plot...")
            M["plot_smile"](
                k=k_mkt,
                iv=iv_mkt,
                title=f"{args.underlying} {args.expiry} — market IV",
                theme=args.theme,
                save_path=out_dir / "smile_market.png",
            )
            plots_created += 1

        if k_mkt.size and w_model is not None and T > 0.0:
            logger.debug("Creating SVI vs market comparison plot...")
            iv_model = np.sqrt(np.maximum(w_model, 0.0) / T)
            M["plot_svi_vs_market"](
                k=k_mkt,
                iv_mkt=iv_mkt,
                iv_model=iv_model,
                title=f"{args.underlying} {args.expiry} — SVI vs market",
                theme=args.theme,
                save_path=out_dir / "smile_model.png",
            )
            plots_created += 1

        if K_pdf is not None and pdf is not None:
            logger.debug("Creating PDF/CDF plot...")
            marks = {}
            stats = M["moments_from_pdf"](K_pdf, pdf)
            if np.isfinite(stats.get("mean", np.nan)):
                marks["mean"] = float(stats["mean"])
            M["plot_pdf_cdf"](
                K=K_pdf,
                pdf=pdf,
                cdf=cdf,
                marks=marks,
                title=f"{args.underlying} {args.expiry} — RN PDF/CDF",
                theme=args.theme,
                save_path=out_dir / "density_pdf_cdf.png",
            )
            plots_created += 1

        logger.info(f"Created {plots_created} plots")

        # 9) Save results.json
        logger.info("Step 9/9: Saving results summary...")
        results: Dict[str, object] = {
            "asset_class": args.asset_class,
            "underlying": args.underlying,
            "expiry": args.expiry,
            "asof_utc": asof.isoformat(),
            "n_quotes": int(df.shape[0]),
            "n_calls": int(len(calls_df)),
            "n_iv_points": int(k_mkt.size),
            "r": float(r),
            "T": float(T),
            "F": float(F),
            "spot": float(spot),
            "svi": svi_dict,
            "bl": bl_dict,
            "plots_created": plots_created,
            "notes": notes.strip(),
        }
        _save_json(out_dir / "results.json", results)

        logger.info(f"Pipeline completed successfully!")
        logger.info(f"Artifacts written to: {out_dir.resolve()}")

        # Summary statistics
        logger.info("=" * 50)
        logger.info("SUMMARY:")
        logger.info(f"  Data: {len(calls_df)} calls, {k_mkt.size} with IV")
        logger.info(f"  SVI:  {'✓' if 'params' in svi_dict else '✗'}")
        logger.info(
            f"  BL:   {'✓' if bl_dict and 'integral' in bl_dict else '✗'}"
        )
        logger.info(f"  Plots: {plots_created} created")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        # Ensure async resources (e.g., httpx.AsyncClient) are closed before loop ends.
        await _maybe_aclose(fetcher)


if __name__ == "__main__":
    asyncio.run(main())
