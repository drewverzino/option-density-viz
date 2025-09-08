# src/viz/dashboard.py
from __future__ import annotations
import plotly.graph_objects as go
import plotly.io as pio
from src.vol.surface import SVISurface, fit_surface_from_frames, smooth_params
from src.preprocess.pcp import add_pcp_diagnostics
from src.preprocess.midprice import add_midprice_columns
from src.preprocess.forward import estimate_forward_from_chain, yearfrac
from src.density.cdf import build_cdf, moments_from_pdf
from src.density.bl import bl_pdf_from_calls
from src.data.yf_fetcher import YFinanceFetcher
from src.data.risk_free import RiskFreeConfig, RiskFreeProvider
from src.data.registry import get_fetcher
from src.data.polygon_fetcher import PolygonFetcher
from src.data.historical_loader import chain_to_dataframe
from src.data.cache import CacheConfig, KVCache

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Debug visibility (styling-only; no structural changes)
DEBUG_UI = st.sidebar.toggle(
    "Show debug details", value=False, help="Toggle verbose UI messages")


def ui_dbg(msg):
    if DEBUG_UI:
        st.write(msg)


# --- Global Plotly template & palette (styling only) ---
OVIZ_COLORWAY = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
]

pio.templates["oviz"] = go.layout.Template(
    layout=go.Layout(
        template="plotly_white",
        colorway=OVIZ_COLORWAY,
        font=dict(
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto", size=13),
        title=dict(x=0.02, xanchor="left"),
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="left", x=0.0),
        xaxis=dict(showgrid=True, zeroline=False,
                   gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(showgrid=True, zeroline=False,
                   gridcolor="rgba(0,0,0,0.08)"),
        scene=dict(
            xaxis=dict(showgrid=True, backgroundcolor="#ffffff",
                       gridcolor="rgba(0,0,0,0.08)"),
            yaxis=dict(showgrid=True, backgroundcolor="#ffffff",
                       gridcolor="rgba(0,0,0,0.08)"),
            zaxis=dict(showgrid=True, backgroundcolor="#ffffff",
                       gridcolor="rgba(0,0,0,0.08)"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
    )
)
pio.templates.default = "oviz"


# Logging
logger = logging.getLogger("oviz.dashboard")
logging.basicConfig(level=logging.INFO)

# Cache configuration
CACHE_TTL_SECONDS = 300
DB_PATH = ".cache/kv.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Streamlit config

st.markdown(
    """
    <style>
    .stApp { background-color: #fafafa; }
    section > div.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-family: ui-sans-serif, system-ui, -apple-system; }
    .metric { padding: 10px 12px; border-radius: 12px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.set_page_config(
    page_title="Option Viz — SVI Surface & Density",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Option Viz — SVI Surface & Density Dashboard")

# Initialize cache
cache_cfg = CacheConfig(path=Path(DB_PATH), ensure_dirs=False)
_kv = KVCache(cache_cfg)

# Helper functions


@st.cache_data(show_spinner=False)
def _fig_html_bytes(fig: go.Figure) -> bytes:
    """Small helper to export Plotly fig to HTML bytes (for download_button)."""
    return fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")


def _surface3d(
    x_k: np.ndarray, y_T: np.ndarray, z: np.ndarray, *, title: str, ztitle: str
) -> go.Figure:
    """Create 3D surface plot with proper dimension handling"""

    # Ensure inputs are numpy arrays
    x_k = np.asarray(x_k, dtype=float)
    y_T = np.asarray(y_T, dtype=float)
    z = np.asarray(z, dtype=float)

    # Handle NaN/inf values
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure z is 2D
    if z.ndim == 1:
        expected_size = len(y_T) * len(x_k)
        if len(z) == expected_size:
            z = z.reshape(len(y_T), len(x_k))
        else:
            st.error(
                f"Cannot reshape z: got {len(z)} elements, expected {expected_size}"
            )
            return go.Figure()

    # Check dimensions match
    if z.shape != (len(y_T), len(x_k)):
        st.error(
            f"Dimension mismatch: z.shape={z.shape}, expected ({len(y_T)}, {len(x_k)})"
        )
        min_rows = min(z.shape[0], len(y_T))
        min_cols = min(z.shape[1], len(x_k))

        z = z[:min_rows, :min_cols]
        y_T = y_T[:min_rows]
        x_k = x_k[:min_cols]

        st.warning(f"Truncated to dimensions: {z.shape}")

    # Create meshgrid for proper surface plotting
    X, Y = np.meshgrid(x_k, y_T)

    # Verify final dimensions
    if X.shape != z.shape or Y.shape != z.shape:
        st.error(
            f"Final dimension mismatch: X:{X.shape}, Y:{Y.shape}, Z:{z.shape}"
        )
        return go.Figure()

    try:
        fig = go.Figure(
            data=go.Surface(
                x=X,
                y=Y,
                z=z,
                colorbar=dict(title=ztitle),
                contours={
                    "z": {
                        "show": True,
                        "usecolormap": True,
                        "start": z.min(),
                        "end": z.max(),
                        "size": (z.max() - z.min()) / 10,
                    }
                },
                showscale=True,
            )
        )

        fig.update_layout(
            template="plotly_white",
            title=title,
            scene=dict(
                xaxis_title="log-moneyness k",
                yaxis_title="T (years)",
                zaxis_title=ztitle,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                aspectmode='cube',
            ),
            width=800,
            height=600,
        )

        return fig

    except Exception as e:
        st.error(f"Failed to create 3D surface: {e}")
        return go.Figure()


def create_fallback_surface_plot(k_grid, T_vec, data_matrix, title, z_label):
    """Create a fallback plot when 3D surface fails"""

    try:
        fig = go.Figure(
            data=go.Heatmap(
                x=k_grid,
                y=T_vec,
                z=data_matrix,
                colorbar=dict(title=z_label),
                hoverongaps=False,
            )
        )

        fig.update_layout(
            template="plotly_white",
            title=f"{title} (Heatmap)",
            xaxis_title="log-moneyness k",
            yaxis_title="T (years)",
        )

        return fig

    except Exception as e:
        st.error(f"Even heatmap failed: {e}")
        return None


def _cache_key(label: str, *parts) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str)
    h = hashlib.blake2s(payload.encode(), digest_size=16).hexdigest()
    return f"{label}:{h}"


def _normalize_type_col(df: pd.DataFrame) -> pd.DataFrame:
    if "type" not in df.columns:
        return df
    d = df.copy()
    d["type"] = d["type"].astype(str).str.upper().str.strip().str[0]
    return d[d["type"].isin(["C", "P"])]


def _clean_for_svi_fixed(df: pd.DataFrame, F: float) -> pd.DataFrame:
    """Clean data for SVI with proper IV handling - NO UNIT CONVERSION HERE"""
    if df.empty or "strike" not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    d = d.dropna(subset=["strike"])
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    d = d.dropna(subset=["strike"])
    d = d[d["strike"] > 0]

    if d.empty:
        return d

    # Handle mid prices
    if "mid" not in d.columns and {"bid", "ask"}.issubset(d.columns):
        d["bid"] = pd.to_numeric(d["bid"], errors="coerce")
        d["ask"] = pd.to_numeric(d["ask"], errors="coerce")
        mask = (
            pd.notna(d["bid"])
            & pd.notna(d["ask"])
            & (d["bid"] > 0)
            & (d["ask"] > 0)
        )
        d.loc[mask, "mid"] = (d.loc[mask, "bid"] + d.loc[mask, "ask"]) / 2.0

    if "mid" in d.columns:
        d["mid"] = pd.to_numeric(d["mid"], errors="coerce")
        d = d.dropna(subset=["mid"])
        d = d[d["mid"] > 0]

    # Clean IV - but DON'T convert units here (should already be done)
    if "iv" in d.columns:
        d["iv"] = pd.to_numeric(d["iv"], errors="coerce")
        d.loc[(d["iv"] <= 0) | (~np.isfinite(d["iv"])), "iv"] = np.nan

        # Sanity check - warn if still looks like percentages
        iv_mean = d["iv"].mean()
        if pd.notna(iv_mean) and iv_mean > 1:
            st.error(
                f"IV values still in percentage format (mean={iv_mean:.1f}%) - check early conversion!"
            )

        d["iv"] = d["iv"].clip(lower=0.01, upper=3.0)

    # Validate forward
    if not F or not np.isfinite(F) or F <= 0:
        F = float(np.nanmedian(d["strike"]))

    # Log-moneyness
    d["k"] = np.log(d["strike"] / float(F))
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["k"])

    return d


def _expiry_quality_metrics(d: pd.DataFrame) -> dict:
    if d.empty:
        return {
            "n_rows": 0,
            "uniq_strikes": 0,
            "n_calls": 0,
            "n_puts": 0,
            "left_k": 0,
            "right_k": 0,
            "span_k": 0.0,
        }

    uniq_strikes = int(d["strike"].nunique()) if "strike" in d.columns else 0
    has_c = int((d["type"] == "C").sum()) if "type" in d.columns else 0
    has_p = int((d["type"] == "P").sum()) if "type" in d.columns else 0

    if "k" in d.columns and not d["k"].isna().all():
        left = int((d["k"] < 0).sum())
        right = int((d["k"] > 0).sum())
        span_k = float(d["k"].max() - d["k"].min()) if len(d) else 0.0
    else:
        left = right = 0
        span_k = 0.0

    return {
        "n_rows": int(len(d)),
        "uniq_strikes": uniq_strikes,
        "n_calls": has_c,
        "n_puts": has_p,
        "left_k": left,
        "right_k": right,
        "span_k": span_k,
    }


def check_input_data_units(usable_frames):
    """Check if input IV data has unit issues"""

    st.write("### Input Data Unit Check")

    all_ivs = []
    all_prices = []

    for exp, df in usable_frames.items():
        if 'iv' in df.columns:
            iv_vals = pd.to_numeric(df['iv'], errors='coerce').dropna()
            all_ivs.extend(iv_vals.tolist())

        if 'mid' in df.columns:
            price_vals = pd.to_numeric(df['mid'], errors='coerce').dropna()
            all_prices.extend(price_vals.tolist())

    if all_ivs:
        iv_array = np.array(all_ivs)
        st.write(
            f"**Input IV range:** [{iv_array.min():.6f}, {iv_array.max():.6f}]"
        )
        st.write(f"**Input IV mean:** {iv_array.mean():.6f}")

        if iv_array.mean() > 1:
            st.warning(
                "Input IVs look like percentages (>100%) - may need conversion"
            )
        elif iv_array.mean() < 0.01:
            st.warning("Input IVs very small - may need scaling")
        else:
            st.info("Input IVs look reasonable (decimal format)")

    if all_prices:
        price_array = np.array(all_prices)
        st.write(
            f"**Input price range:** [{price_array.min():.2f}, {price_array.max():.2f}]"
        )


def debug_svi_params(surface, sample_exp=None):
    """Debug SVI parameter values to identify unit issues"""

    st.write("### SVI Parameter Debug")

    exps = surface.expiries()
    if not exps:
        st.error("No expiries in surface")
        return

    if sample_exp is None:
        sample_exp = sorted(exps)[0]

    try:
        # Get SVI parameters
        a, b, rho, m, sigma = surface.params_tuple(sample_exp)
        T = surface.T[sample_exp]
        F = surface.F[sample_exp]

        st.write(
            f"**Expiry:** {pd.to_datetime(sample_exp).strftime('%Y-%m-%d')}"
        )
        st.write(f"**T:** {T:.4f} years, **F:** {F:.2f}")
        st.write(
            f"**SVI params:** a={a:.6f}, b={b:.6f}, ρ={rho:.6f}, m={m:.6f}, σ={sigma:.6f}"
        )

        # Test SVI formula at a few points
        test_k = np.array([-0.2, 0.0, 0.2])
        test_w = surface.w(test_k, sample_exp)
        test_iv = np.sqrt(test_w / T)

        st.write(f"**Test evaluations:**")
        for i, k in enumerate(test_k):
            st.write(f"  k={k:.1f}: w={test_w[i]:.6f}, iv={test_iv[i]:.6f}")

        # Check if parameters seem reasonable
        warnings = []
        if a < 0:
            warnings.append("Negative 'a' parameter - unusual")
        if b <= 0:
            warnings.append("Non-positive 'b' parameter - invalid")
        if abs(rho) >= 1:
            warnings.append("Invalid rho parameter (should be < 1)")
        if sigma <= 0:
            warnings.append("Non-positive sigma parameter - invalid")

        # Check total variance scale
        atm_w = surface.w(np.array([0.0]), sample_exp)[0]
        atm_iv = np.sqrt(atm_w / T)

        if atm_w > 1:
            warnings.append(f"High ATM total variance: {atm_w:.6f}")
        if atm_iv > 2:
            warnings.append(f"High ATM IV: {atm_iv:.6f} (>200%)")

        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("SVI parameters look reasonable")

    except Exception as e:
        st.error(f"SVI debug failed: {e}")


# Multi-expiry data fetching
def fetch_multi_expiry_quotes_sync(
    market: str, symbol: str, max_expiries: int = 10
) -> pd.DataFrame:
    """Fetch quotes for multiple expiries WITH EARLY IV UNIT CONVERSION"""

    async def _fetch_multi_async():
        candidates = []

        # Try Polygon first
        try:
            candidates.append(("polygon", PolygonFetcher()))
        except Exception as e:
            st.info(f"Polygon backend unavailable: {e}")

        # Add fallbacks
        if market == "equity":
            candidates.append(("yfinance", YFinanceFetcher()))
        elif market == "crypto":
            try:
                candidates.append(("crypto_registry", get_fetcher(market)))
            except Exception:
                pass

        all_dfs = []

        for name, fetcher in candidates:
            try:
                st.caption(f"{name}: listing expiries...")
                expiries = await fetcher.list_expiries(symbol)

                if not expiries:
                    continue

                # Filter future expiries and sort
                future_expiries = [
                    e for e in expiries if e.date() > datetime.now().date()
                ]
                if not future_expiries:
                    st.warning(f"No future expiries available from {name}")
                    continue

                # Take the next N expiries
                selected_expiries = sorted(future_expiries)[:max_expiries]

                st.caption(
                    f"{name}: found {len(future_expiries)} future expiries, fetching {len(selected_expiries)}"
                )

                # Fetch data for each expiry
                for i, exp in enumerate(selected_expiries):
                    try:
                        st.caption(
                            f"{name}: fetching expiry {i+1}/{len(selected_expiries)}: {exp.date()}"
                        )
                        chain = await fetcher.fetch_chain(symbol, expiry=exp)

                        if not chain.quotes:
                            continue

                        # Convert to DataFrame
                        df = chain_to_dataframe(chain)

                        if df.empty or "strike" not in df.columns:
                            continue

                        # Clean up data
                        df["asof_utc"] = pd.to_datetime(
                            datetime.now(timezone.utc), utc=True
                        )
                        df["expiry"] = pd.to_datetime(exp, utc=True)

                        # Normalize types and prices
                        for col in ("strike", "bid", "ask", "mid", "spot"):
                            if col in df.columns:
                                df[col] = pd.to_numeric(
                                    df[col], errors="coerce"
                                )

                        # CRITICAL: Fix IV units EARLY, before any processing
                        if "iv" in df.columns:
                            df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
                            # Check if values are in percentage format
                            iv_mean = df["iv"].mean()
                            if pd.notna(iv_mean) and iv_mean > 1:
                                st.info(
                                    f"Converting IV from percentage format for {exp.date()} (mean={iv_mean:.1f}%)"
                                )
                                df["iv"] = df["iv"] / 100.0

                        # Handle mid prices
                        if "mid" not in df.columns and {"bid", "ask"}.issubset(
                            df.columns
                        ):
                            mask = pd.notna(df["bid"]) & pd.notna(df["ask"])
                            df.loc[mask, "mid"] = (
                                df.loc[mask, "bid"] + df.loc[mask, "ask"]
                            ) / 2.0

                        df = _normalize_type_col(df)

                        # Clean up and sort
                        base_cols = [
                            c
                            for c in ["expiry", "type", "strike"]
                            if c in df.columns
                        ]
                        if base_cols:
                            df = df.sort_values(base_cols)
                            df = df.drop_duplicates(
                                subset=base_cols, keep="last"
                            )

                        df = df.reset_index(drop=True)

                        if not df.empty:
                            all_dfs.append(df)
                            st.success(
                                f"✓ Fetched {len(df)} quotes for {exp.date()}"
                            )

                    except Exception as e:
                        st.warning(f"Failed to fetch expiry {exp.date()}: {e}")
                        continue

                if all_dfs:
                    # Combine all expiries into one DataFrame
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    st.success(
                        f"Backend: {name} - successfully fetched {len(all_dfs)} expiries with {len(combined_df)} total quotes"
                    )
                    return combined_df

            except Exception as e:
                logger.error(f"Backend {name} failed: {e}")
                if hasattr(fetcher, 'aclose'):
                    try:
                        await fetcher.aclose()
                    except:
                        pass
                continue

        raise RuntimeError("All backends failed")

    return asyncio.run(_fetch_multi_async())


# Single expiry data fetching (corrected)
def fetch_quotes_sync_fixed(
    market: str, symbol: str, asof_utc: Optional[datetime] = None
) -> pd.DataFrame:
    """Synchronous wrapper for async quote fetching WITH EARLY IV UNIT CONVERSION"""

    async def _fetch_async():
        candidates = []

        try:
            candidates.append(("polygon", PolygonFetcher()))
        except Exception as e:
            st.info(f"Polygon backend unavailable: {e}")

        if market == "equity":
            candidates.append(("yfinance", YFinanceFetcher()))
        elif market == "crypto":
            try:
                candidates.append(("crypto_registry", get_fetcher(market)))
            except Exception:
                pass

        for name, fetcher in candidates:
            try:
                st.caption(f"{name}: listing expiries...")
                expiries = await fetcher.list_expiries(symbol)

                if not expiries:
                    continue

                # Choose expiry - but filter out past expiries
                future_expiries = [
                    e for e in expiries if e.date() > datetime.now().date()
                ]
                if not future_expiries:
                    st.warning(f"No future expiries available from {name}")
                    continue

                if asof_utc:
                    target_d = asof_utc.date()
                    chosen_exp = min(
                        future_expiries,
                        default=future_expiries[0],
                        key=lambda e: abs((e.date() - target_d).days),
                    )
                else:
                    chosen_exp = future_expiries[0]

                st.caption(
                    f"{name}: fetching chain for {chosen_exp.date()}..."
                )
                chain = await fetcher.fetch_chain(symbol, expiry=chosen_exp)

                if not chain.quotes:
                    continue

                df = chain_to_dataframe(chain)

                if df.empty or "strike" not in df.columns:
                    continue

                df["asof_utc"] = pd.to_datetime(
                    datetime.now(timezone.utc), utc=True
                )
                df["expiry"] = pd.to_datetime(chosen_exp, utc=True)

                for col in ("strike", "bid", "ask", "mid", "spot"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                # CRITICAL: Fix IV units EARLY
                if "iv" in df.columns:
                    df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
                    iv_mean = df["iv"].mean()
                    if pd.notna(iv_mean) and iv_mean > 1:
                        st.info(
                            f"Converting IV from percentage format (mean={iv_mean:.1f}%)"
                        )
                        df["iv"] = df["iv"] / 100.0
                        st.info(
                            f"After conversion: IV mean = {df['iv'].mean():.3f}"
                        )

                if "mid" not in df.columns and {"bid", "ask"}.issubset(
                    df.columns
                ):
                    mask = pd.notna(df["bid"]) & pd.notna(df["ask"])
                    df.loc[mask, "mid"] = (
                        df.loc[mask, "bid"] + df.loc[mask, "ask"]
                    ) / 2.0

                df = _normalize_type_col(df)

                base_cols = [
                    c for c in ["expiry", "type", "strike"] if c in df.columns
                ]
                if base_cols:
                    df = df.sort_values(base_cols)
                    df = df.drop_duplicates(subset=base_cols, keep="last")

                df = df.reset_index(drop=True)

                if df.empty:
                    continue

                st.caption(f"Backend: {name}")
                return df

            except Exception as e:
                logger.error(f"Backend {name} failed: {e}")
                if hasattr(fetcher, 'aclose'):
                    try:
                        await fetcher.aclose()
                    except:
                        pass
                continue

        raise RuntimeError("All backends failed")

    return asyncio.run(_fetch_async())


@st.cache_data(show_spinner="Fetching multi-expiry quotes...")
def _fetch_multi_expiry_quotes(
    market: str, symbol: str, max_expiries: int = 10
) -> pd.DataFrame:
    try:
        return fetch_multi_expiry_quotes_sync(market, symbol, max_expiries)
    except Exception as e:
        st.error(f"Failed to fetch multi-expiry data: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Fetching single expiry quotes...")
def _fetch_quotes_fixed(
    market: str, symbol: str, asof_utc: Optional[datetime] = None
) -> pd.DataFrame:
    try:
        return fetch_quotes_sync_fixed(market, symbol, asof_utc)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()


def _group_frames_by_expiry(
    quotes: pd.DataFrame, expiry_col="expiry"
) -> Dict[pd.Timestamp, pd.DataFrame]:
    out = {}
    if quotes.empty or expiry_col not in quotes.columns:
        return out

    try:
        quotes = quotes.dropna(subset=[expiry_col])
        for exp, df in quotes.groupby(expiry_col):
            if pd.notna(exp):
                out[pd.to_datetime(exp)] = df.sort_values("strike")
    except Exception as e:
        logger.error(f"Error grouping by expiry: {e}")
        return {}

    return out


def _maps_from_frames_fixed(
    frames_by_expiry: Dict[pd.Timestamp, pd.DataFrame],
    asof_utc: datetime,
    rf_provider: RiskFreeProvider,
    mid_wide_threshold: float,
) -> Tuple[
    Dict[pd.Timestamp, float],
    Dict[pd.Timestamp, float],
    Dict[pd.Timestamp, float],
    Dict[pd.Timestamp, pd.DataFrame],
]:

    T_map = {}
    r_map = {}
    F_map = {}
    clean_frames = {}

    for exp, df in frames_by_expiry.items():
        try:
            df = _normalize_type_col(df)
            exp_dt = pd.to_datetime(exp)

            # CRITICAL: Fix the time calculation
            time_diff = exp_dt - pd.to_datetime(asof_utc)
            T = max(time_diff.total_seconds() / (365.25 * 24 * 3600), 1e-6)

            if T <= 0:
                st.warning(
                    f"Skipping past expiry: {exp_dt.strftime('%Y-%m-%d')}"
                )
                continue

            try:
                r = float(rf_provider.get_rate(asof_utc))
            except Exception:
                r = 0.02

            d = add_midprice_columns(df, wide_rel_threshold=mid_wide_threshold)
            d["T"] = T
            d["r"] = r

            # Estimate forward
            try:
                has_c = (
                    int((d["type"] == "C").sum()) if "type" in d.columns else 0
                )
                has_p = (
                    int((d["type"] == "P").sum()) if "type" in d.columns else 0
                )

                if has_c > 0 and has_p > 0:
                    F_est = estimate_forward_from_chain(
                        d,
                        r=r,
                        T=T,
                        price_col="mid",
                        type_col="type",
                        strike_col="strike",
                        top_n=7,
                    )
                else:
                    if "spot" in d.columns and pd.notna(d["spot"]).any():
                        spot_vals = pd.to_numeric(
                            d["spot"], errors="coerce"
                        ).dropna()
                        F_est = (
                            float(spot_vals.iloc[0])
                            if not spot_vals.empty
                            else float(np.nanmedian(d["strike"]))
                        )
                    else:
                        F_est = float(np.nanmedian(d["strike"]))
            except Exception:
                F_est = float(np.nanmedian(d["strike"]))

            if not np.isfinite(F_est) or F_est <= 0:
                continue

            d["F"] = F_est
            T_map[exp] = T
            r_map[exp] = r
            F_map[exp] = F_est
            clean_frames[exp] = d

        except Exception as e:
            logger.error(f"Error processing expiry {exp}: {e}")
            continue

    return T_map, r_map, F_map, clean_frames


async def _get_surface_kv(
    frames_by_expiry, T_map, F_map, r_map, smoothing
) -> SVISurface:
    # Convert pd.Timestamp keys to datetime
    datetime_frames = {}
    datetime_T_map = {}
    datetime_F_map = {}
    datetime_r_map = {}

    for exp in frames_by_expiry:
        exp_dt = pd.to_datetime(exp).to_pydatetime()
        datetime_frames[exp_dt] = frames_by_expiry[exp]
        datetime_T_map[exp_dt] = T_map[exp]
        datetime_F_map[exp_dt] = F_map[exp]
        datetime_r_map[exp_dt] = r_map[exp]

    surface = fit_surface_from_frames(
        datetime_frames,
        T_by_expiry=datetime_T_map,
        F_by_expiry=datetime_F_map,
        r_by_expiry=datetime_r_map,
    )

    if smoothing != "None":
        method = "cubic_spline" if smoothing == "Cubic spline" else "poly3"
        surface = smooth_params(surface, method=method)

    return surface


async def _get_sample_kv_fixed(
    usable_frames, k_span, k_points, smoothing, surface
):
    """Generate surface samples with proper handling"""

    exps = surface.expiries()
    if not exps:
        raise ValueError("No expiries in surface")

    # Sort expiries by time to maturity
    exps_sorted = sorted(exps, key=lambda e: surface.T[e])

    T_vec = np.array([surface.T[e] for e in exps_sorted], dtype=float)
    k_grid = np.linspace(k_span[0], k_span[1], int(k_points))

    # Initialize matrices
    n_exp = len(exps_sorted)
    n_k = len(k_grid)

    w_mat = np.full((n_exp, n_k), np.nan)
    iv_mat = np.full((n_exp, n_k), np.nan)

    # Fill matrices row by row
    for i, exp in enumerate(exps_sorted):
        try:
            T_val = surface.T[exp]
            if T_val <= 0:
                continue

            # Get total variance for this expiry
            w_slice = surface.w(k_grid, exp)
            w_slice = np.asarray(w_slice, dtype=float)

            # Handle negative or invalid w values
            w_slice = np.where(
                np.isfinite(w_slice) & (w_slice >= 0), w_slice, 0.0
            )

            # Convert to IV: iv = sqrt(w / T)
            iv_slice = np.sqrt(w_slice / T_val)

            # Final validation and clipping
            iv_slice = np.where(np.isfinite(iv_slice), iv_slice, 0.0)
            iv_slice = np.clip(iv_slice, 0.0, 5.0)  # Reasonable IV bounds

            # Store in matrices
            w_mat[i, :] = w_slice
            iv_mat[i, :] = iv_slice

        except Exception as e:
            st.error(f"Failed to process expiry {exp}: {e}")
            continue

    # Final cleanup
    w_mat = np.nan_to_num(w_mat, nan=0.0, posinf=0.0, neginf=0.0)
    iv_mat = np.nan_to_num(iv_mat, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "k_grid": k_grid,
        "T_vec": T_vec,
        "w_mat": w_mat,
        "iv_mat": iv_mat,
        "expiries": exps_sorted,
    }


def check_time_to_expiry(
    quotes_df: pd.DataFrame, asof_utc: datetime
) -> pd.DataFrame:
    """Check and fix time to expiry calculations"""

    ui_dbg("### Time to Expiry Debug")

    if "expiry" not in quotes_df.columns:
        st.error("No expiry column found")
        return quotes_df

    # Ensure expiry is datetime
    quotes_df["expiry"] = pd.to_datetime(quotes_df["expiry"], utc=True)

    # Get unique expiries
    unique_expiries = quotes_df["expiry"].unique()

    ui_dbg("**Expiry Analysis:**")
    for exp in unique_expiries:
        exp_dt = pd.to_datetime(exp)
        time_diff = exp_dt - pd.to_datetime(asof_utc)
        days_to_exp = time_diff.total_seconds() / (24 * 3600)
        years_to_exp = days_to_exp / 365.25

        ui_dbg(f"- Expiry: {exp_dt.strftime('%Y-%m-%d')}")
        st.write(f"  - Days to expiry: {days_to_exp:.1f}")
        st.write(f"  - Years to expiry: {years_to_exp:.4f}")

        if days_to_exp <= 0:
            st.error("⚠️ Expiry is in the past!")
        elif days_to_exp < 1:
            st.warning("⚠️ Expiry is very soon (< 1 day)")

    return quotes_df


# Sidebar
with st.sidebar:
    st.header("Inputs")

    market = st.selectbox("Market", ["equity", "crypto"], index=0)
    symbol = st.text_input(
        "Symbol", value=("AAPL" if market == "equity" else "BTC")
    )

    st.markdown("---")
    st.caption("Expiry Mode")
    fetch_mode = st.selectbox(
        "Data fetching mode",
        ["Single Expiry", "Multiple Expiries (3D Surface)"],
        index=1,  # Default to multi-expiry
    )

    if fetch_mode == "Multiple Expiries (3D Surface)":
        max_expiries = st.slider("Max expiries to fetch", 2, 15, 10)
        st.info(
            "This will fetch multiple expiries to enable 3D surface plotting"
        )
    else:
        st.info("Single expiry mode - will show 2D smile plot")
        asof_date = st.date_input(
            "As-of (local date)", value=datetime.now().date()
        )
        asof_time = st.time_input("As-of time", value=datetime.now().time())
        try:
            asof_utc = datetime.combine(asof_date, asof_time).replace(
                tzinfo=timezone.utc
            )
        except Exception:
            asof_utc = datetime.now(timezone.utc)

    st.markdown("---")
    st.caption("Risk-free configuration")
    default_rate_val = st.number_input(
        "Default rate", value=0.02, step=0.005, format="%.4f"
    )

    rf_cfg = RiskFreeConfig(
        default_rate=default_rate_val, forward_fill=True, cache=True
    )
    rf_provider = RiskFreeProvider(rf_cfg)

    st.markdown("---")
    st.caption("Preprocessing")
    mid_wide_threshold = st.slider(
        "Wide spread threshold", 0.01, 0.50, 0.10, 0.01
    )
    min_quotes_per_expiry = st.slider("Min quotes per expiry", 5, 50, 10, 1)

    st.markdown("---")
    st.caption("Surface fitting")
    smoothing = st.selectbox(
        "Parameter smoothing", ["None", "Cubic spline", "Poly3"], index=1
    )
    k_span = st.slider("k-range (log-moneyness)", -2.0, 2.0, (-1.2, 1.2), 0.05)
    k_points = st.select_slider(
        "Grid density", options=[60, 120, 240, 360], value=120
    )

# Main content
try:
    # Fetch data based on mode
    if fetch_mode == "Multiple Expiries (3D Surface)":
        quotes_all = _fetch_multi_expiry_quotes(market, symbol, max_expiries)
        # Use current time for multi-expiry
        asof_utc = datetime.now(timezone.utc)
    else:
        quotes_all = _fetch_quotes_fixed(market, symbol, asof_utc=asof_utc)

    if quotes_all.empty:
        st.warning("No quotes returned. Check symbol/market settings.")
        st.stop()

    # Time to expiry check
    quotes_all = check_time_to_expiry(quotes_all, asof_utc)

    st.subheader("Raw Quotes Summary")

    # Show expiry breakdown
    if "expiry" in quotes_all.columns:
        expiry_summary = (
            quotes_all.groupby('expiry')
            .agg(
                {
                    'strike': 'count',
                    'type': lambda x: f"{(x == 'C').sum()}C/{(x == 'P').sum()}P",
                }
            )
            .rename(columns={'strike': 'total_quotes', 'type': 'C/P_split'})
        )
        st.dataframe(expiry_summary, width='stretch')

    st.dataframe(quotes_all.head(20), width='stretch')

    frames = _group_frames_by_expiry(quotes_all)
    if not frames:
        st.warning("No expiries found in quotes.")
        st.stop()

    T_map, r_map, F_map, clean_frames = _maps_from_frames_fixed(
        frames, asof_utc, rf_provider, mid_wide_threshold
    )

    # Build usable frames
    quality_rows = []
    usable_frames = {}

    REQ_UNIQ_STRIKES = 8
    REQ_LEFT_RIGHT = 2
    REQ_SPAN_K = 0.15

    for exp, df in clean_frames.items():
        try:
            df = _normalize_type_col(df)
            F = float(F_map.get(exp, np.nan))
            d = _clean_for_svi_fixed(df, F)

            if len(d) < min_quotes_per_expiry:
                quality_rows.append(
                    {
                        "expiry": pd.to_datetime(exp),
                        **_expiry_quality_metrics(d),
                        "kept": False,
                        "reason": "too_few_quotes",
                    }
                )
                continue

            qm = _expiry_quality_metrics(d)
            reason = None

            if qm["uniq_strikes"] < REQ_UNIQ_STRIKES:
                reason = "few_strikes"
            elif (
                qm["left_k"] < REQ_LEFT_RIGHT or qm["right_k"] < REQ_LEFT_RIGHT
            ):
                reason = "weak_OTM_balance"
            elif qm["span_k"] < REQ_SPAN_K:
                reason = "short_k_span"

            if reason is None:
                usable_frames[exp] = d
                quality_rows.append(
                    {
                        "expiry": pd.to_datetime(exp),
                        **qm,
                        "kept": True,
                        "reason": "",
                    }
                )
            else:
                quality_rows.append(
                    {
                        "expiry": pd.to_datetime(exp),
                        **qm,
                        "kept": False,
                        "reason": reason,
                    }
                )
        except Exception as e:
            quality_rows.append(
                {
                    "expiry": pd.to_datetime(exp),
                    "n_rows": 0,
                    "uniq_strikes": 0,
                    "n_calls": 0,
                    "n_puts": 0,
                    "left_k": 0,
                    "right_k": 0,
                    "span_k": 0.0,
                    "kept": False,
                    "reason": "processing_error",
                }
            )

    if not usable_frames:
        st.error("No expiry passed SVI prerequisites.")
        qdf = pd.DataFrame(quality_rows).sort_values("expiry")
        if not qdf.empty:
            st.dataframe(qdf, width='stretch')
        st.stop()

    qdf = pd.DataFrame(quality_rows).sort_values(
        ["kept", "expiry"], ascending=[False, True]
    )

    with st.expander("SVI input quality by expiry"):
        st.dataframe(qdf, width='stretch')

    st.success(f"Prepared {len(usable_frames)} expiries for analysis.")

    # Debug Information
    with st.expander("Debug Information"):
        # Check input data units
        check_input_data_units(usable_frames)

        # Check how many expiries you actually have
        st.write(f"**Usable frames:** {len(usable_frames)} expiries")
        for exp, df in usable_frames.items():
            st.write(
                f"- {pd.to_datetime(exp).strftime('%Y-%m-%d')}: {len(df)} quotes"
            )

        # Quick check of T_map, F_map values
        st.write("**Time/Forward/Rate values:**")
        for exp in sorted(usable_frames.keys()):
            T = T_map.get(exp, "N/A")
            F = F_map.get(exp, "N/A")
            r = r_map.get(exp, "N/A")
            st.write(
                f"- {pd.to_datetime(exp).strftime('%Y-%m-%d')}: T={T:.4f}, F={F:.2f}, r={r:.4f}"
            )

    # Single-expiry diagnostics
    if len(usable_frames) == 1:
        with st.expander("Single-Expiry Diagnostics"):
            exp = list(usable_frames.keys())[0]
            df_exp = usable_frames[exp]
            T = T_map[exp]
            r = r_map[exp]
            F = F_map[exp]

            st.write(f"**T**={T:.4f} yrs, **r**={r:.4f}, **F**≈{F:.4f}")

            # PCP
            has_c = int((df_exp["type"] == "C").sum())
            has_p = int((df_exp["type"] == "P").sum())

            if has_c > 0 and has_p > 0:
                try:
                    pcp_tbl = add_pcp_diagnostics(df_exp, spot=F, r=r, T=T)
                    if pcp_tbl is not None and not pcp_tbl.empty:
                        st.dataframe(
                            pcp_tbl.head(10), width='stretch'
                        )
                except Exception as e:
                    st.error(f"PCP failed: {e}")

            # BL density
            try:
                df_calls = df_exp[df_exp["type"] == "C"].dropna(
                    subset=["strike", "mid"]
                )
                if len(df_calls) >= 5:
                    K = np.sort(df_calls["strike"].to_numpy(dtype=float))
                    C_mid = df_calls.sort_values("strike")["mid"].to_numpy(
                        dtype=float
                    )

                    KK, pdf, _ = bl_pdf_from_calls(K, C_mid, r=r, T=T)
                    mom = moments_from_pdf(KK, pdf)

                    st.write(
                        f"**BL moments**: mean={mom['mean']:.4f}, var={mom['var']:.6f}"
                    )

                    fig_pdf = go.Figure()
                    fig_pdf.add_scatter(x=KK, y=pdf, mode="lines", name="pdf")
                    fig_pdf.update_layout(
                        template="plotly_white",
                        title="Breeden–Litzenberger PDF",
                        xaxis_title="Strike K",
                        yaxis_title="Density",
                    )
                    st.plotly_chart(fig_pdf, width='stretch')
            except Exception as e:
                st.error(f"BL density failed: {e}")

    # SVI Surface section
    st.markdown("## SVI Surface Analysis")

    try:
        surface = asyncio.run(
            _get_surface_kv(usable_frames, T_map, F_map, r_map, smoothing)
        )

        # Check number of expiries
        n_expiries = len(surface.expiries())
        st.write(f"**Number of expiries fitted:** {n_expiries}")

        if n_expiries < 2:
            st.warning(
                "⚠️ Only 1 expiry available - showing 2D smile instead of 3D surface."
            )

            # For single expiry, show 2D smile
            exp = surface.expiries()[0]
            T_val = surface.T[exp]
            F_val = surface.F[exp]

            ui_dbg(f"**Expiry:** {pd.to_datetime(exp).strftime('%Y-%m-%d')}")
            st.write(f"**T:** {T_val:.4f} years, **F:** {F_val:.2f}")

            # Debug SVI parameters
            debug_svi_params(surface, exp)

            # Create fine k grid for smooth smile
            k_fine = np.linspace(k_span[0], k_span[1], 200)

            try:
                w_smile = surface.w(k_fine, exp)
                iv_smile = np.sqrt(np.maximum(w_smile, 0.0) / T_val)

                # Create 2D smile plot
                fig_smile = go.Figure()

                # Add fitted SVI curve
                fig_smile.add_scatter(
                    x=k_fine,
                    y=iv_smile,
                    mode='lines',
                    name='SVI fit',
                    line=dict(color='blue', width=3),
                )

                # Add original data points if available
                exp_df = usable_frames.get(pd.to_datetime(exp))
                if (
                    exp_df is not None
                    and 'k' in exp_df.columns
                    and 'iv' in exp_df.columns
                ):
                    for opt_type, color in [('C', 'red'), ('P', 'green')]:
                        type_data = exp_df[exp_df['type'] == opt_type]
                        if not type_data.empty:
                            obs_k = type_data['k'].values
                            obs_iv = pd.to_numeric(
                                type_data['iv'], errors='coerce'
                            ).values

                            valid_obs = np.isfinite(obs_k) & np.isfinite(
                                obs_iv
                            )

                            fig_smile.add_scatter(
                                x=obs_k[valid_obs],
                                y=obs_iv[valid_obs],
                                mode='markers',
                                name=f'{opt_type} Market Data',
                                marker=dict(color=color, size=8, opacity=0.7),
                            )

                # Add ATM line
                fig_smile.add_vline(
                    x=0,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text="ATM",
                )

                fig_smile.update_layout(
                    template="plotly_white",
                    title=f"IV Smile - {symbol} {pd.to_datetime(exp).strftime('%Y-%m-%d')} (T={T_val:.3f}y)",
                    xaxis_title="Log-moneyness k = ln(K/F)",
                    yaxis_title="Implied Volatility",
                    height=600,
                )

                st.plotly_chart(fig_smile, width='stretch')

                # Download option
                st.download_button(
                    "Download IV smile plot (HTML)",
                    data=_fig_html_bytes(fig_smile),
                    file_name=f"iv_smile_{symbol}_{pd.to_datetime(exp).strftime('%Y%m%d')}.html",
                    mime="text/html",
                )

            except Exception as e:
                st.error(f"Failed to create smile plot: {e}")
                import traceback

                st.code(traceback.format_exc())

        else:
            # Multiple expiries - proceed with 3D surface
            st.success(
                f"✅ {n_expiries} expiries available - creating 3D surface!"
            )

            sample = asyncio.run(
                _get_sample_kv_fixed(
                    usable_frames, k_span, k_points, smoothing, surface
                )
            )

            k_grid = sample["k_grid"]
            T_vec = sample["T_vec"]
            w_mat = sample["w_mat"]
            iv_mat = sample["iv_mat"]

            # Enhanced diagnostics
            st.write(f"**Surface Diagnostics:**")
            ui_dbg(f"- Expiries: {len(surface.expiries())}")
            st.write(
                f"- k grid: {len(k_grid)} points from {k_grid.min():.3f} to {k_grid.max():.3f}"
            )
            st.write(
                f"- T vector: {len(T_vec)} points from {T_vec.min():.4f} to {T_vec.max():.4f} years"
            )
            st.write(f"- w matrix shape: {w_mat.shape}")
            st.write(f"- IV matrix shape: {iv_mat.shape}")
            st.write(f"- w range: [{w_mat.min():.6f}, {w_mat.max():.6f}]")
            st.write(f"- IV range: [{iv_mat.min():.6f}, {iv_mat.max():.6f}]")

            # Check for issues
            if iv_mat.max() > 5:
                st.error(
                    f"IV values still too high (max={iv_mat.max():.1f}) - check SVI fitting"
                )
            elif iv_mat.max() < 0.01:
                st.error(
                    f"IV values too low (max={iv_mat.max():.6f}) - check SVI fitting"
                )
            else:
                st.success("✅ IV values look reasonable")

            # Show SVI parameter summary
            with st.expander("SVI Parameter Summary"):
                param_rows = []
                for exp in sorted(
                    surface.expiries(), key=lambda e: surface.T[e]
                ):
                    a, b, rho, m, sigma = surface.params_tuple(exp)
                    param_rows.append(
                        {
                            'expiry': pd.to_datetime(exp).strftime('%Y-%m-%d'),
                            'T': surface.T[exp],
                            'F': surface.F[exp],
                            'a': a,
                            'b': b,
                            'rho': rho,
                            'm': m,
                            'sigma': sigma,
                        }
                    )

                param_df = pd.DataFrame(param_rows)
                st.dataframe(param_df, width='stretch')

            # 3D plots
            col1, col2 = st.columns(2)

            with col1:
                try:
                    st.write("**3D IV Surface**")
                    fig_iv = _surface3d(
                        k_grid,
                        T_vec,
                        iv_mat,
                        title="3D IV Surface",
                        ztitle="IV",
                    )
                    if fig_iv.data:
                        st.plotly_chart(fig_iv, width='stretch')
                        st.download_button(
                            "Download 3D IV surface (HTML)",
                            data=_fig_html_bytes(fig_iv),
                            file_name=f"surface3d_iv_{symbol}.html",
                            mime="text/html",
                        )
                    else:
                        # Fallback to heatmap
                        st.warning("3D surface failed, showing heatmap...")
                        fig_fallback = create_fallback_surface_plot(
                            k_grid, T_vec, iv_mat, "IV Surface", "IV"
                        )
                        if fig_fallback:
                            st.plotly_chart(
                                fig_fallback, width='stretch'
                            )

                except Exception as e:
                    st.error(f"3D IV plot failed: {e}")

            with col2:
                try:
                    st.write("**3D Total Variance Surface**")
                    fig_w = _surface3d(
                        k_grid,
                        T_vec,
                        w_mat,
                        title="3D Total Variance",
                        ztitle="w",
                    )
                    if fig_w.data:
                        st.plotly_chart(fig_w, width='stretch')
                    else:
                        # Fallback to heatmap
                        st.warning("3D surface failed, showing heatmap...")
                        fig_fallback = create_fallback_surface_plot(
                            k_grid, T_vec, w_mat, "Total Variance Surface", "w"
                        )
                        if fig_fallback:
                            st.plotly_chart(
                                fig_fallback, width='stretch'
                            )

                except Exception as e:
                    st.error(f"3D w plot failed: {e}")

            # 2D views for multiple expiries
            st.markdown("## 2D Views")

            with st.expander("IV vs k slices", expanded=True):
                try:
                    exps = surface.expiries()
                    exps_sorted = sorted(
                        exps, key=lambda e: float(surface.T[e])
                    )
                    max_lines = st.slider(
                        "Max expiries to show",
                        3,
                        min(15, len(exps_sorted)),
                        min(8, len(exps_sorted)),
                    )
                    step = max(1, len(exps_sorted) // max_lines)
                    plot_exps = exps_sorted[::step][:max_lines]

                    fig2d_iv = go.Figure()
                    for e in plot_exps:
                        try:
                            iv_slice = np.sqrt(
                                np.maximum(surface.w(k_grid, e), 0.0)
                                / max(surface.T[e], 1e-12)
                            )

                            fig2d_iv.add_scatter(
                                x=k_grid,
                                y=iv_slice,
                                mode="lines",
                                name=f"{pd.to_datetime(e).strftime('%Y-%m-%d')} (T={surface.T[e]:.3f})",
                                line=dict(width=2),
                            )
                        except Exception:
                            continue

                    fig2d_iv.update_layout(
                        template="plotly_white",
                        title="IV(k) slices across expiries",
                        xaxis_title="log-moneyness k",
                        yaxis_title="Implied Volatility",
                        height=500,
                    )
                    st.plotly_chart(fig2d_iv, width='stretch')
                except Exception as e:
                    st.error(f"2D slices failed: {e}")

            # Calendar arbitrage check
            st.markdown("### Calendar Arbitrage Check")
            try:
                tol = st.number_input("Tolerance", value=1e-10, format="%.1e")
                cal = surface.calendar_violations(k_grid, tol=float(tol))

                st.write(
                    f"**Violations:** {cal['count']} / {cal['n_checks']} ({cal['fraction']:.1%})"
                )

                if cal.get("by_pair"):
                    cal_df = pd.DataFrame(
                        cal["by_pair"],
                        columns=["T1", "T2", "violation_fraction"],
                    )
                    st.dataframe(cal_df, width='stretch')

                    # Plot calendar violations
                    if len(cal_df) > 1:
                        fig_cal = go.Figure()
                        fig_cal.add_scatter(
                            x=cal_df["T2"],
                            y=cal_df["violation_fraction"],
                            mode="markers+lines",
                            name="Violation %",
                        )
                        fig_cal.update_layout(
                            template="plotly_white",
                            title="Calendar Arbitrage Violations by Maturity",
                            xaxis_title="T2 (years)",
                            yaxis_title="Violation Fraction",
                        )
                        st.plotly_chart(fig_cal, width='stretch')
            except Exception as e:
                st.error(f"Calendar check failed: {e}")

                # --------------------------------------------------------------------------------
            # NEW: BL Risk-Neutral Density Surface (3D)
            # --------------------------------------------------------------------------------
            st.markdown("## BL Risk-Neutral Density Surface")

            try:
                # Expect these to already exist in your current app state:
                # usable_frames: Dict[pd.Timestamp, pd.DataFrame] of clean, per-expiry quotes
                # F_map: Dict[pd.Timestamp, float] forward per expiry
                # T_map: Dict[pd.Timestamp, float] year fractions per expiry
                # r_map: Dict[pd.Timestamp, float] rates per expiry
                # symbol: current underlying ticker string
                # bl_pdf_from_calls: function to compute BL PDF from (K, C(K), r, T)

                exp_keys = sorted(list(usable_frames.keys()),
                                  key=lambda e: T_map[pd.to_datetime(e)])
                if len(exp_keys) < 2:
                    st.info(
                        "Need at least 2 expiries to build a surface. Select more expiries above.")
                else:
                    # Common k-grid around ATM
                    k_min, k_max, k_points = -1.2, 1.2, 161
                    k_grid = np.linspace(k_min, k_max, k_points)
                    T_vec = np.array([T_map[pd.to_datetime(e)]
                                      for e in exp_keys], dtype=float)
                    Z = np.zeros((len(T_vec), k_points), dtype=float)

                    used_rows = 0
                    for i, e in enumerate(exp_keys):
                        df_e = usable_frames.get(pd.to_datetime(e))
                        r = float(r_map.get(pd.to_datetime(e), 0.0))
                        F = float(F_map.get(pd.to_datetime(e), np.nan))
                        T = float(T_map.get(pd.to_datetime(e), np.nan))
                        if df_e is None or not np.isfinite(F) or not np.isfinite(T):
                            continue
                        calls = df_e[df_e.get("type", "").astype(
                            str).str.upper().str[0] == "C"].dropna(subset=["strike", "mid"])
                        if len(calls) < 5:
                            continue

                        K = np.sort(pd.to_numeric(
                            calls["strike"], errors="coerce").to_numpy(dtype=float))
                        C_mid = calls.sort_values(
                            "strike")["mid"].to_numpy(dtype=float)

                        try:
                            K_grid, pdf_K, _ = bl_pdf_from_calls(
                                K, C_mid, r=r, T=T)
                            # Convert to k-space via Jacobian: k = ln(K/F), pdf_k = pdf_K * dK/dk = pdf_K * K
                            k_e = np.log(np.maximum(K_grid, 1e-12)/F)
                            pdf_k = pdf_K * K_grid
                            # Interpolate onto common k_grid
                            pdf_interp = np.interp(
                                k_grid, k_e, pdf_k, left=0.0, right=0.0)
                            Z[i, :] = np.maximum(pdf_interp, 0.0)
                            used_rows += 1
                        except Exception:
                            continue

                    if used_rows >= 2:
                        # Slice-wise renormalization in K-space to ~unit mass
                        for i, e in enumerate(exp_keys):
                            F = float(F_map.get(pd.to_datetime(e), np.nan))
                            if not np.isfinite(F):
                                continue
                            K_row = F*np.exp(k_grid)
                            with np.errstate(divide="ignore", invalid="ignore"):
                                pdf_K_row = np.where(
                                    K_row > 0, Z[i, :] / K_row, 0.0)
                            mass = np.trapz(pdf_K_row, K_row)
                            if np.isfinite(mass) and mass > 0:
                                Z[i, :] /= mass

                        fig_bl = _surface3d(
                            k_grid, T_vec, Z,
                            title="3D Risk-Neutral Density (BL)",
                            ztitle="pdf(k)"
                        )
                        st.plotly_chart(fig_bl, use_container_width=True)
                        st.caption(
                            "Density surface in log-moneyness k; each slice is a BL PDF for a given maturity.")

                        # Optional download (HTML)
                        try:
                            st.download_button(
                                "Download BL surface (HTML)",
                                data=_fig_html_bytes(fig_bl),
                                file_name=f"surface3d_bl_{symbol}.html",
                                mime="text/html",
                            )
                        except Exception:
                            pass
                    else:
                        st.warning(
                            "Insufficient clean expiries to construct BL surface.")
            except Exception as e:
                st.error(f"BL surface failed: {e}")
            # Export section
            st.markdown("### Export sampled grid")
            try:
                grid_rows = []
                exps = surface.expiries()
                for i, e in enumerate(exps):
                    for j, k in enumerate(k_grid):
                        grid_rows.append(
                            {
                                "expiry": pd.to_datetime(e),
                                "T": float(T_vec[i]),
                                "k": float(k),
                                "K_over_F": float(np.exp(k)),
                                "w": float(w_mat[i, j]),
                                "iv": float(iv_mat[i, j]),
                            }
                        )

                grid_df = pd.DataFrame(grid_rows)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download sampled surface (CSV)",
                        data=grid_df.to_csv(index=False).encode(),
                        file_name=f"svi_surface_sample_{symbol}.csv",
                        mime="text/csv",
                    )
                with col2:
                    # Also export the parameters
                    params_df = pd.DataFrame(param_rows)
                    st.download_button(
                        "Download SVI parameters (CSV)",
                        data=params_df.to_csv(index=False).encode(),
                        file_name=f"svi_parameters_{symbol}.csv",
                        mime="text/csv",
                    )

                st.caption(
                    "Tip: K/F = exp(k). Multiply by per-expiry forward F(T) for absolute strikes."
                )
            except Exception as e:
                st.error(f"Export failed: {e}")

    except Exception as e:
        st.error(f"SVI surface fitting failed: {e}")
        import traceback

        st.code(traceback.format_exc())
        st.info(
            "Check the debug information and quality table above for diagnostics."
        )

except Exception as e:
    st.error(f"Dashboard pipeline failed: {e}")
    import traceback

    st.code(traceback.format_exc())
    st.info("Please check your inputs and try again.")

# Footer
st.markdown("---")
st.caption(
    f"Dashboard running in {fetch_mode} mode | Data as of: {asof_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"
)
