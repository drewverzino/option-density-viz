# src/cli/main.py
"""
Option Viz CLI
==============
Two ways to use:
1) Pipeline (default): data → preprocess → SVI → BL density → plots + artifacts
2) Dashboard: launch the Streamlit UI (interactive)
Artifacts: chain.csv, results.json, smile_market.png, smile_model.png, density_pdf_cdf.png
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------- Logging setup ----------
class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record):
        if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, "")
            original = record.levelname
            record.levelname = f"{color}{original}{self.RESET}"
            out = super().format(record)
            record.levelname = original
            return out
        return super().format(record)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("main")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter(
            "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return logger


logger = setup_logging()


# ---------- Flexible imports ----------
def _import_modules():
    logger.debug("Importing project modules...")
    try:
        # Package-style
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

    except Exception as e:
        logger.debug(
            f"Package-style import failed: {e}, trying flat imports..."
        )
        # Flat
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
        description="Option Viz: pipeline runner and Streamlit dashboard launcher",
    )
    p.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Streamlit dashboard UI and exit.",
    )
    p.add_argument(
        "--dashboard-path",
        default=str(
            Path(__file__).resolve().parents[2]
            / "src"
            / "viz"
            / "dashboard.py"
        ),
        help="Path to the Streamlit dashboard script (advanced).",
    )
    p.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="Try to open the dashboard in your browser (Streamlit default).",
    )

    p.add_argument(
        "--asset-class",
        choices=["equity", "crypto"],
        help="Data backend to use: equity (Polygon→YF) or crypto (OKX).",
    )
    p.add_argument(
        "--underlying",
        "--asset",
        dest="underlying",
        help="Ticker/symbol, e.g., AAPL or BTC. (--asset alias supported)",
    )
    p.add_argument(
        "--expiry",
        help="Target expiry date: YYYY-MM-DD (new) or YYMMDD (legacy).",
    )
    p.add_argument(
        "--out",
        default="docs/run",
        help="Output directory for artifacts (pipeline mode).",
    )
    p.add_argument(
        "--theme",
        choices=["light", "dark"],
        default="light",
        help="Plot theme (pipeline mode).",
    )
    p.add_argument(
        "--min-points",
        type=int,
        default=7,
        help="Min IV points for SVI fit (pipeline mode).",
    )
    p.add_argument(
        "--grid-n",
        type=int,
        default=401,
        help="Grid size for BL density (pipeline mode).",
    )
    p.add_argument(
        "--skip-density",
        action="store_true",
        help="Skip BL density step (pipeline mode).",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level.",
    )
    return p.parse_args()


# ------------------------------ Utilities -----------------------------------
def _parse_expiry_any(s: str) -> datetime:
    if re.fullmatch(r"\d{6}", s):
        dt = datetime.strptime(s, "%y%m%d")
        return dt.replace(tzinfo=timezone.utc)
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception as e:
        raise ValueError(
            f"Invalid --expiry {s!r}; expected YYYY-MM-DD or YYMMDD"
        ) from e


def _match_expiry(expiries: List[datetime], want: datetime) -> datetime:
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
        f"Expiry {want.date().isoformat()} not found; available: {available_str}"
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
    fn = getattr(fetcher, "aclose", None) or getattr(fetcher, "close", None)
    if not fn:
        return
    res = fn()
    if inspect.iscoroutine(res):
        await res
    logger.debug("Closed async fetcher")


def _launch_streamlit_dashboard(
    script_path: Path, open_browser: bool = True
) -> int:
    if not script_path.exists():
        raise FileNotFoundError(f"Dashboard script not found: {script_path}")
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{pp}" if pp else str(repo_root)
    )
    args = [sys.executable, "-m", "streamlit", "run", str(script_path)]
    if not open_browser:
        args += ["--server.headless=true"]
    logger.info(f"Launching Streamlit dashboard: {script_path}")
    logger.debug(f"PYTHONPATH={env['PYTHONPATH']}")
    try:
        return subprocess.call(args, env=env)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Failed to launch Streamlit. Make sure 'streamlit' is installed:\n    pip install streamlit plotly\n"
        ) from e


# ---- Equity fetcher selection: Polygon → YFinance (fallback) ----
async def _choose_equity_fetchers():
    candidates = []
    try:
        from ..data.polygon_fetcher import (  # prefer polygon if available
            PolygonFetcher,
        )

        candidates.append(("polygon", PolygonFetcher()))
    except Exception as e:
        logger.info(
            f"Polygon backend unavailable ({e}); will fall back to Yahoo Finance."
        )
    # Always include Yahoo fallback (yfinance)
    try:
        from ..data.yf_fetcher import YFinanceFetcher
    except Exception:
        from data.yf_fetcher import YFinanceFetcher
    candidates.append(("yfinance", YFinanceFetcher()))
    return candidates


def _estimate_forward_compat(df: pd.DataFrame, r: float, T: float):
    func = M["estimate_forward"]
    func_name = getattr(func, '__name__', str(func))

    # Check if this is estimate_forward_from_chain (takes DataFrame)
    if func_name == 'estimate_forward_from_chain':
        sig = inspect.signature(func)
        params = set(sig.parameters.keys())
        kwargs = {"r": r, "T": T}

        # Add optional parameters if supported
        if "price_col" in params:
            kwargs["price_col"] = "mid"
        if "type_col" in params:
            kwargs["type_col"] = "type"
        if "strike_col" in params:
            kwargs["strike_col"] = "strike"
        if "top_n" in params:
            kwargs["top_n"] = 7

        return float(func(df, **kwargs))

    # Check if this is estimate_forward_from_pcp (takes arrays)
    elif func_name == 'estimate_forward_from_pcp':
        # Need to extract call/put/strike arrays from DataFrame
        try:
            from ..preprocess.pcp import pivot_calls_puts_by_strike
        except ImportError:
            try:
                from preprocess.pcp import pivot_calls_puts_by_strike
            except ImportError:
                raise ValueError(
                    "Cannot import pivot_calls_puts_by_strike for PCP forward estimation"
                )

        # Get call/put pairs
        wide = pivot_calls_puts_by_strike(
            df, price_col="mid", type_col="type", strike_col="strike"
        )

        if wide.empty or "C" not in wide.columns or "P" not in wide.columns:
            raise ValueError(
                "No call/put pairs found for PCP forward estimation"
            )

        calls = wide["C"].to_numpy(dtype=float)
        puts = wide["P"].to_numpy(dtype=float)
        strikes = wide.index.to_numpy(dtype=float)

        return float(func(calls, puts, strikes, r=r, T=T))

    else:
        # Try generic approach
        try:
            return float(func(df, r=r, T=T))
        except TypeError:
            return float(func(df, r, T))


# ------------------------------ Pipeline (async) -----------------------------
async def _run_pipeline(args) -> None:
    logger.setLevel(getattr(logging, args.log_level))
    from math import exp

    logger.info("Starting Option Viz CLI")
    logger.info(
        f"Target: {args.underlying} ({args.asset_class}) expiry {args.expiry}"
    )
    logger.info(f"Output directory: {args.out}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data fetch
    logger.info("Step 1/9: Fetching option chain data...")
    if args.asset_class == "equity":
        candidates = await _choose_equity_fetchers()
    else:
        candidates = [(args.asset_class, M["get_fetcher"](args.asset_class))]

    chain = None
    chosen_backend = None
    last_err: Optional[Exception] = None

    for name, fetcher in candidates:
        try:
            logger.debug(f"Listing expiries for {args.underlying}")
            expiries = sorted(await fetcher.list_expiries(args.underlying))
            logger.info(f"Found {len(expiries)} available expiries")
            target = _match_expiry(expiries, _parse_expiry_any(args.expiry))
            logger.info(f"Target expiry: {target}")
            logger.debug("Fetching option chain...")
            candidate_chain = await fetcher.fetch_chain(
                args.underlying, target
            )
            await _maybe_aclose(fetcher)

            # --- NEW: auto-fallback if 0 quotes ---
            if not candidate_chain.quotes:
                logger.warning(
                    f"{name} returned 0 quotes for {args.underlying} {target.date()} — trying next backend..."
                )
                continue

            chain = candidate_chain
            chosen_backend = name
            break

        except Exception as e:
            last_err = e
            logger.warning(
                f"{name} backend failed: {e}. Trying next candidate..."
            )
            await _maybe_aclose(fetcher)

    if chain is None:
        raise RuntimeError(f"All backends failed. Last error: {last_err}")

    asof = chain.asof_utc
    logger.info(f"Fetched chain as of {asof} with {len(chain.quotes)} quotes")
    logger.info(
        f"Backend: {chosen_backend}{' (fallback)' if chosen_backend == 'yfinance' else ' (preferred)'}"
    )

    df = M["chain_to_dataframe"](chain)
    logger.debug(
        f"Converted to DataFrame: {df.shape[0]} rows, {df.shape[1]} cols"
    )

    # Save normalized chain early (raw snapshot)
    logger.debug("Saving raw chain to CSV...")
    M["save_chain_csv"](out_dir / "chain.csv", chain)

    # --- NEW: Hard guard against empty/malformed frames ---
    if df.empty or "strike" not in df.columns:
        logger.error(
            "No usable quotes (empty DataFrame or missing 'strike'). "
            f"Backend={chosen_backend}. Exiting early."
        )
        _save_json(
            out_dir / "results.json",
            {
                "asset_class": args.asset_class,
                "underlying": args.underlying,
                "expiry": args.expiry,
                "backend": chosen_backend,
                "asof_utc": asof.isoformat(),
                "n_quotes": 0,
                "notes": "No usable quotes; skipped preprocessing and modeling.",
            },
        )
        logger.info("Pipeline completed (early exit due to empty data).")
        return

    # 2) Preprocess
    logger.info("Step 2/9: Preprocessing quotes (midprices, flags)...")

    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.upper().str.strip().str[0]
        df = df[df["type"].isin(["C", "P"])]

    for c in ("strike", "bid", "ask", "mid", "iv"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "mid" not in df.columns and {"bid", "ask"}.issubset(df.columns):
        df["mid"] = (df["bid"] + df["ask"]) / 2.0

    df = df.dropna(subset=["strike"])
    df = df[df["strike"] > 0]
    if "mid" in df.columns:
        df = df.dropna(subset=["mid"])
        df = df[df["mid"] > 0]

    if "iv" in df.columns:
        df.loc[
            (df["iv"] <= 0) | (~np.isfinite(df["iv"])) | (df["iv"] > 3.0), "iv"
        ] = np.nan

    df = M["add_midprice_columns"](df, wide_rel_threshold=0.15)
    n_crossed = (
        int(df.get("crossed", pd.Series([], dtype=bool)).sum())
        if "crossed" in df.columns
        else 0
    )  # type: ignore
    n_wide = (
        int(df.get("wide", pd.Series([], dtype=bool)).sum())
        if "wide" in df.columns
        else 0
    )  # type: ignore
    logger.info(f"Quote quality: {n_crossed} crossed, {n_wide} wide")

    # 3) Risk-free and T
    logger.info("Step 3/9: Risk-free rate and time to maturity...")
    rf = M["RiskFreeProvider"](M["RiskFreeConfig"]())
    try:
        r = float(rf.get_rate(asof.date()))
        logger.info(f"Risk-free rate: {r:.4f} ({r*100:.2f}%)")
    except Exception as e:
        r = 0.0
        logger.warning(f"Failed to get risk-free rate, using 0.0: {e}")

    target = _parse_expiry_any(args.expiry)
    T = float(M["yearfrac"](asof, target))
    logger.info(f"T = {T:.4f} years (~{T*365:.1f} days)")

    # 4) Forward estimate — signature-compatible
    logger.info("Step 4/9: Estimating forward price...")
    spot = chain.spot if np.isfinite(chain.spot or np.nan) else np.nan
    try:
        F = _estimate_forward_compat(df, r, T)
        logger.info(f"Forward (PCP): {F:.4f}")
    except Exception as e:
        F = (
            float((spot or 0.0) * np.exp(r * T))
            if np.isfinite(spot)
            else float("nan")
        )
        logger.warning(
            f"PCP forward failed; using spot*exp(rT): {F:.4f} | {e}"
        )

    # 5) Calls and IVs
    logger.info("Step 5/9: Filtering calls & extracting IVs...")
    calls_df = df.copy()
    if "type" in calls_df.columns:
        t = calls_df["type"].astype(str).str.upper().str.strip()
        calls_df = calls_df[t.str.startswith("C")]
    calls_df = calls_df.dropna(subset=["strike", "mid"])
    logger.info(f"Found {len(calls_df)} call options with valid strike/mid")

    K = calls_df["strike"].to_numpy(float)
    C_mid = calls_df["mid"].to_numpy(float)
    k = _log_moneyness(K, F) if np.isfinite(F) and F > 0 else np.array([])

    iv = None
    if "iv" in calls_df.columns:
        iv = calls_df["iv"].to_numpy(float)
        logger.debug("Using 'iv' column")
    elif "extra" in calls_df.columns:
        try:
            iv = np.array(
                [float((x or {}).get("iv", np.nan)) for x in calls_df["extra"]]
            )
            logger.debug("Extracted IVs from 'extra'")
        except Exception:
            iv = None

    if iv is not None and k.size:
        mask = np.isfinite(iv) & np.isfinite(k) & (iv > 0)
        k_mkt = k[mask]
        iv_mkt = iv[mask]
        logger.info(f"Market IV points: {len(iv_mkt)}")
    else:
        k_mkt = np.array([])
        iv_mkt = np.array([])
        logger.warning("No implied volatilities present or invalid forward")

    # 6) SVI fit
    logger.info("Step 6/9: Fitting SVI...")
    w_model = None
    notes = ""
    min_points = max(3, args.min_points)

    if k_mkt.size >= min_points:
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
            logger.info(f"SVI fit OK: loss={svi_dict['loss']:.6f}")
        except Exception as e:
            svi_dict = {"error": repr(e)}
            notes = f"SVI failed: {e!r}"
            logger.error(f"SVI fit failed: {e}")
    else:
        svi_dict = {
            "note": f"Insufficient IV points: {k_mkt.size} < {min_points}"
        }
        logger.warning("Skipping SVI (insufficient data)")

    # 7) BL density
    logger.info("Step 7/9: Breeden–Litzenberger density...")
    bl_dict = None
    K_pdf = None
    pdf = None
    cdf = None
    if (not args.skip_density) and (K.size >= 3):
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
                "integral": float(getattr(pdf_diag, "integral", float("nan"))),
                "neg_frac": float(getattr(pdf_diag, "neg_frac", float("nan"))),
                "rn_mean": float(getattr(pdf_diag, "rn_mean", float("nan"))),
                "rn_var": float(getattr(pdf_diag, "rn_var", float("nan"))),
                "note": str(getattr(pdf_diag, "note", "")),
            }
            logger.info(
                f"BL: ∫pdf={bl_dict['integral']:.4f}, RN mean={bl_dict['rn_mean']:.2f}"
            )
        except Exception as e:
            bl_dict = {"error": repr(e)}
            notes += f" | BL failed: {e!r}"
            logger.error(f"BL density failed: {e}")
    elif args.skip_density:
        logger.info("Skipping BL density (--skip-density)")
    else:
        logger.warning("Insufficient call data for BL density")

    # 8) Plots
    logger.info("Step 8/9: Generating plots...")
    plots_created = 0
    if k_mkt.size:
        M["plot_smile"](
            k=k_mkt,
            iv=iv_mkt,
            title=f"{args.underlying} {args.expiry} — market IV",
            theme=args.theme,
            save_path=Path(args.out) / "smile_market.png",
        )
        plots_created += 1
    if k_mkt.size and w_model is not None and T > 0.0:
        iv_model = np.sqrt(np.maximum(w_model, 0.0) / T)
        M["plot_svi_vs_market"](
            k=k_mkt,
            iv_mkt=iv_mkt,
            iv_model=iv_model,
            title=f"{args.underlying} {args.expiry} — SVI vs market",
            theme=args.theme,
            save_path=Path(args.out) / "smile_model.png",
        )
        plots_created += 1
    if K_pdf is not None and pdf is not None:
        stats = M["moments_from_pdf"](K_pdf, pdf)
        marks = {}
        if np.isfinite(stats.get("mean", np.nan)):
            marks["mean"] = float(stats["mean"])
        M["plot_pdf_cdf"](
            K=K_pdf,
            pdf=pdf,
            cdf=cdf,
            marks=marks,
            title=f"{args.underlying} {args.expiry} — RN PDF/CDF",
            theme=args.theme,
            save_path=Path(args.out) / "density_pdf_cdf.png",
        )
        plots_created += 1
    logger.info(f"Created {plots_created} plots")

    # 9) Save results.json
    logger.info("Step 9/9: Saving results summary...")
    results: Dict[str, object] = {
        "asset_class": args.asset_class,
        "underlying": args.underlying,
        "expiry": args.expiry,
        "backend": chosen_backend,  # record which backend produced data
        "asof_utc": asof.isoformat(),
        "n_quotes": int(df.shape[0]),
        "n_calls": int(len(calls_df)),
        "n_iv_points": int(k_mkt.size),
        "r": float(r),
        "T": float(T),
        "F": float(F),
        "spot": float(spot) if np.isfinite(spot) else None,
        "svi": svi_dict,
        "bl": bl_dict,
        "plots_created": plots_created,
        "notes": notes.strip(),
    }
    _save_json(Path(args.out) / "results.json", results)

    logger.info("Pipeline completed successfully!")
    logger.info(f"Artifacts → {Path(args.out).resolve()}")


# ------------------------------ Main entry ----------------------------------
async def main() -> None:
    args = _parse_args()
    if args.dashboard:
        script_path = Path(args.dashboard_path).resolve()
        rc = _launch_streamlit_dashboard(
            script_path, open_browser=args.open_browser
        )
        if rc != 0:
            logger.error(f"Streamlit exited with code {rc}")
            raise SystemExit(rc)
        return

    missing = [
        flag
        for flag, val in {
            "--asset-class": args.asset_class,
            "--underlying": args.underlying,
            "--expiry": args.expiry,
        }.items()
        if not val
    ]
    if missing:
        raise SystemExit(
            f"Missing required arguments for pipeline mode: {' '.join(missing)}\nTip: use --dashboard to launch the interactive app without these."
        )
    await _run_pipeline(args)


if __name__ == "__main__":
    asyncio.run(main())
