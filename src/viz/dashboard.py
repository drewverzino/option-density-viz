# src/viz/dashboard.py
from __future__ import annotations

# ----------------------------- stdlib ----------------------------- #
import asyncio
import hashlib
import inspect
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------- third-party ------------------------ #
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.cache import CacheConfig, KVCache
from src.data.historical_loader import chain_to_dataframe
from src.data.polygon_fetcher import PolygonFetcher  # type: ignore
from src.data.registry import get_fetcher
from src.data.risk_free import RiskFreeConfig, RiskFreeProvider

# ----------------------------- project modules -------------------- #
from src.data.yf_fetcher import YFinanceFetcher

# density utilities (single-expiry peek)
from src.density.bl import bl_pdf_from_calls
from src.density.cdf import build_cdf, moments_from_pdf
from src.preprocess.forward import estimate_forward_from_chain, yearfrac
from src.preprocess.midprice import add_midprice_columns
from src.preprocess.pcp import add_pcp_diagnostics

# vol/ modules (surface builds on svi; dashboard calls surface)
from src.vol.surface import (
    SVISurface,
    fit_surface_from_frames,
    smooth_params,
)

# ------------------------------ logging ------------------------------ #
logger = logging.getLogger("oviz.dashboard")
logger.info("Dashboard module imported")

# ------------------------------
# Global cache configuration
# ------------------------------
CACHE_TTL_SECONDS = 300  # 5 minutes by default

# ================================================================== #
#                          Streamlit Config                          #
# ================================================================== #
st.set_page_config(
    page_title="Option Viz — SVI Surface & Density",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Option Viz — SVI Surface & Density Dashboard")
logger.info("Streamlit dashboard initialized")

# ================================================================== #
#                         Cache DB Initialization                    #
# ================================================================== #
DB_PATH = ".cache/kv.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
with sqlite3.connect(DB_PATH) as _con:
    _con.execute("PRAGMA journal_mode=WAL;")
    _con.execute("PRAGMA synchronous=NORMAL;")
logger.debug("SQLite cache DB configured (WAL, NORMAL)")

cache_cfg = CacheConfig(path=DB_PATH, ensure_dirs=False)
_kv = KVCache(cache_cfg)
logger.debug("KVCache ready")


# ================================================================== #
#                              Helpers                               #
# ================================================================== #
@st.cache_data(show_spinner=False)
def _fig_html_bytes(fig: go.Figure) -> bytes:
    logger.debug("Serializing Plotly figure to HTML bytes")
    return fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")


def _surface3d(
    x_k: np.ndarray, y_T: np.ndarray, z: np.ndarray, *, title: str, ztitle: str
) -> go.Figure:
    logger.debug(
        f"Building 3D surface: title={title}, ztitle={ztitle}, "
        f"x_len={len(x_k)}, y_len={len(y_T)}, z_shape={getattr(z, 'shape', None)}"
    )
    fig = go.Figure(
        data=go.Surface(
            x=x_k,
            y=y_T,
            z=z,
            colorbar=dict(title=ztitle),
            contours={
                "z": {"show": True, "usecolormap": True, "highlightwidth": 1}
            },
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=title,
        scene=dict(
            xaxis_title="log-moneyness k",
            yaxis_title="T (years)",
            zaxis_title=ztitle,
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            zaxis=dict(showgrid=True),
        ),
        uirevision="keep-3d",
    )
    fig.update_scenes(camera=dict(eye=dict(x=1.35, y=1.35, z=0.9)))
    return fig


def _hash_frames(
    frames_by_expiry: Dict[pd.Timestamp, pd.DataFrame],
    cols=("type", "strike", "mid", "bid", "ask", "iv"),
) -> str:
    logger.debug("Hashing frames_by_expiry for cache key")
    h = hashlib.blake2s(digest_size=16)
    for exp in sorted(frames_by_expiry):
        df = frames_by_expiry[exp]
        keep = [c for c in cols if c in df.columns]
        d = df[keep].copy()
        for c in d.select_dtypes(include="number").columns:
            d[c] = np.round(d[c].astype(float), 10)
        h.update(str(pd.to_datetime(exp).value).encode())
        h.update(d.to_csv(index=False).encode())
    digest = h.hexdigest()
    logger.debug(f"frames_by_expiry hash={digest}")
    return digest


# ------------------------------ Cache key helpers ------------------------------ #
def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return sorted(list(obj))
    return str(obj)


def _cache_key(label: str, *parts) -> str:
    payload = json.dumps(parts, sort_keys=True, default=_json_default)
    h = hashlib.blake2s(payload.encode(), digest_size=16).hexdigest()
    key = f"{label}:{h}"
    logger.debug(f"Built cache key for {label}: {key}")
    return key


# ---------- Normalization / SVI cleaning helpers ---------- #
def _normalize_type_col(df: pd.DataFrame) -> pd.DataFrame:
    if "type" not in df.columns:
        return df
    d = df.copy()
    d["type"] = d["type"].astype(str).str.upper().str.strip().str[0]
    d = d[d["type"].isin(["C", "P"])]
    return d


def _clean_for_svi(df: pd.DataFrame, F: float) -> pd.DataFrame:
    logger.debug(f"Cleaning frame for SVI: rows={len(df)}, F={F}")
    d = df.copy()
    # strikes
    d = d.dropna(subset=["strike"]).copy()
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    d = d.dropna(subset=["strike"])
    d = d[d["strike"] > 0]
    # mid price
    if "mid" not in d.columns and {"bid", "ask"}.issubset(d.columns):
        d["bid"] = pd.to_numeric(d["bid"], errors="coerce")
        d["ask"] = pd.to_numeric(d["ask"], errors="coerce")
        d["mid"] = (d["bid"] + d["ask"]) / 2.0
    if "mid" in d.columns:
        d["mid"] = pd.to_numeric(d["mid"], errors="coerce")
        d = d.dropna(subset=["mid"])
        d = d[d["mid"] > 0]
    # iv cleanup (optional)
    if "iv" in d.columns:
        d["iv"] = pd.to_numeric(d["iv"], errors="coerce")
        d.loc[(d["iv"] <= 0) | (~np.isfinite(d["iv"])), "iv"] = np.nan
        d["iv"] = d["iv"].clip(lower=0.01, upper=3.0)
    # log-moneyness
    d["k"] = (
        np.log(d["strike"] / float(F))
        if F and np.isfinite(F) and F > 0
        else np.nan
    )
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["k"])
    logger.debug(
        f"Cleaned frame: rows={len(d)}, k_span={float(d['k'].max()-d['k'].min()) if len(d) else 0.0}"
    )
    return d


def _expiry_quality_metrics(d: pd.DataFrame) -> dict:
    uniq_strikes = int(d["strike"].nunique()) if "strike" in d.columns else 0
    has_c = int((d["type"] == "C").sum()) if "type" in d.columns else 0
    has_p = int((d["type"] == "P").sum()) if "type" in d.columns else 0
    left = int((d["k"] < 0).sum())
    right = int((d["k"] > 0).sum())
    span_k = float(d["k"].max() - d["k"].min()) if len(d) else 0.0
    return {
        "n_rows": int(len(d)),
        "uniq_strikes": uniq_strikes,
        "n_calls": has_c,
        "n_puts": has_p,
        "left_k": left,
        "right_k": right,
        "span_k": span_k,
    }


# ================================================================== #
#                            Data Access                             #
# ================================================================== #
@st.cache_data(show_spinner="Fetching quotes…")
def _fetch_quotes(
    market: str,
    symbol: str,
    *,
    asof_utc: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Return a normalized long quotes DataFrame with columns like:
      ['expiry','type','strike','bid','ask','mid','iv', ...]
    Uses KV cache and prefers Polygon for equity, falling back to yfinance.
    """
    t0 = time.perf_counter()

    # Build a stable cache key (minute-level granularity for as-of)
    asof_key = None
    if asof_utc is not None:
        asof_key = asof_utc.replace(second=0, microsecond=0).isoformat()

    key = _cache_key("quotes_long", market, symbol.upper().strip(), asof_key)
    logger.info(
        f"[fetch] market={market} symbol={symbol} asof={asof_key} key={key}"
    )

    async def _compute():
        # --- choose fetchers: Polygon first for BOTH equity and crypto ---
        candidates = []
        try:
            candidates.append(("polygon", PolygonFetcher()))
            logger.debug("[fetch] Using PolygonFetcher first")
        except Exception as e:
            st.info(f"Polygon backend unavailable ({e}); trying fallbacks.")
            logger.warning(f"[fetch] Polygon unavailable: {e}")

        if market == "equity":
            # Always have Yahoo as a safety net for equities
            candidates.append(("yfinance", YFinanceFetcher()))
            logger.debug("[fetch] Added yfinance fallback (equity)")
        elif market == "crypto":
            # Crypto fallback via registry (e.g., OKX)
            candidates.append(("crypto_registry", get_fetcher(market)))
            logger.debug("[fetch] Added crypto registry fallback")
        else:
            logger.error(f"Unsupported market: {market}")
            raise ValueError(f"Unsupported market: {market}")

        last_err: Optional[Exception] = None

        for name, fetcher in candidates:
            try:
                t_list = time.perf_counter()
                logger.info(f"[fetch:{name}] listing expiries for {symbol}")
                st.caption(f"{name}: listing expiries…")
                expiries = await fetcher.list_expiries(symbol)
                dt_list = time.perf_counter() - t_list
                logger.info(
                    f"[fetch:{name}] {len(expiries)} expiries in {dt_list:.3f}s"
                )
                if not expiries:
                    raise ValueError(f"{name}: no expiries for {symbol}")

                # 2) Choose an expiry near the as-of date
                if asof_utc is not None:
                    target_d = asof_utc.date()
                    future = [e for e in expiries if e.date() >= target_d]
                    chosen_exp = min(
                        future,
                        default=min(
                            expiries, key=lambda e: abs(e.date() - target_d)
                        ),
                    )
                else:
                    chosen_exp = expiries[0]
                logger.info(f"[fetch:{name}] chosen expiry {chosen_exp}")

                # 3) Fetch chain
                st.caption(f"{name}: fetching chain for {chosen_exp.date()} …")
                t_chain = time.perf_counter()
                chain = await fetcher.fetch_chain(symbol, expiry=chosen_exp)
                dt_chain = time.perf_counter() - t_chain
                logger.info(
                    f"[fetch:{name}] chain with {len(chain.quotes)} quotes in {dt_chain:.3f}s "
                    f"(spot={chain.spot}, asof={chain.asof_utc})"
                )

                # 4) Normalize to DataFrame
                st.caption(f"{name}: normalizing quotes…")
                t_norm = time.perf_counter()
                df = chain_to_dataframe(chain)
                if asof_utc is not None:
                    df["asof_utc"] = pd.to_datetime(asof_utc, utc=True)

                if "expiry" in df.columns:
                    df["expiry"] = pd.to_datetime(
                        df["expiry"], utc=True, errors="coerce"
                    )

                if "type" in df.columns:
                    df["type"] = (
                        df["type"].astype(str).str.upper().str.strip().str[0]
                    )  # C/P

                for col in ("strike", "bid", "ask", "mid", "iv", "spot"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                if "mid" not in df.columns and {"bid", "ask"}.issubset(
                    df.columns
                ):
                    df["mid"] = (df["bid"] + df["ask"]) / 2.0

                df = _normalize_type_col(df)

                base_cols = [
                    c for c in ["expiry", "type", "strike"] if c in df.columns
                ]
                if base_cols:
                    df = df.sort_values(
                        base_cols
                        + (["asof_utc"] if "asof_utc" in df.columns else [])
                    )
                    df = df.drop_duplicates(subset=base_cols, keep="last")

                want = [
                    "expiry",
                    "type",
                    "strike",
                    "bid",
                    "ask",
                    "mid",
                    "iv",
                    "asof_utc",
                    "spot",
                ]
                cols = [c for c in want if c in df.columns] + [
                    c for c in df.columns if c not in want
                ]
                df = df[cols].reset_index(drop=True)
                dt_norm = time.perf_counter() - t_norm
                logger.info(
                    f"[fetch:{name}] normalized {len(df)} rows in {dt_norm:.3f}s"
                )

                # Success with this backend
                st.caption(
                    f"Backend: {name}{' (preferred)' if name != 'yfinance' else ' (fallback)'}"
                )
                return df

            except Exception as e:
                last_err = e
                logger.error(f"[fetch:{name}] failed: {e}")
                # try the next candidate
                continue

        # Out of candidates — surface the last error
        logger.error("[fetch] all candidates failed")
        raise last_err or RuntimeError("No fetcher succeeded.")

    result = asyncio.run(
        _kv.get_cached(key, _compute, ttl_seconds=CACHE_TTL_SECONDS)
    )
    dt_total = time.perf_counter() - t0
    logger.info(f"[fetch] completed in {dt_total:.3f}s (rows={len(result)})")
    return result


def _group_frames_by_expiry(
    quotes: pd.DataFrame, expiry_col="expiry"
) -> Dict[pd.Timestamp, pd.DataFrame]:
    logger.debug(
        f"Grouping quotes by expiry on column='{expiry_col}', rows={len(quotes)}"
    )
    out: Dict[pd.Timestamp, pd.DataFrame] = {}
    if expiry_col not in quotes.columns:
        logger.warning("Quotes missing 'expiry' column")
        return out
    for exp, df in quotes.groupby(expiry_col):
        out[pd.to_datetime(exp)] = df.sort_values("strike")
    logger.debug(f"Grouped into {len(out)} expiries")
    return out


def _maps_from_frames(
    frames_by_expiry: Dict[pd.Timestamp, pd.DataFrame],
    *,
    asof_utc: datetime,
    rf_provider: RiskFreeProvider,
    mid_wide_threshold: float,
) -> Tuple[
    Dict[pd.Timestamp, float],
    Dict[pd.Timestamp, float],
    Dict[pd.Timestamp, float],
    Dict[pd.Timestamp, pd.DataFrame],
]:
    """
    Per-expiry maps (T, r, F) and cleaned frames with mid/flags.
    """
    logger.info("Building T/r/F maps and cleaned frames")
    T_map: Dict[pd.Timestamp, float] = {}
    r_map: Dict[pd.Timestamp, float] = {}
    F_map: Dict[pd.Timestamp, float] = {}
    clean_frames: Dict[pd.Timestamp, pd.DataFrame] = {}

    for exp, df in frames_by_expiry.items():
        logger.debug(f"Processing expiry={exp} rows={len(df)}")
        df = _normalize_type_col(df)

        T = yearfrac(asof_utc, exp)
        # Risk-free from provider (date-based API)
        r = float(rf_provider.get_rate(asof_utc))

        d = add_midprice_columns(df, wide_rel_threshold=mid_wide_threshold)
        d["T"] = T
        d["r"] = r

        # Estimate forward via PCP only if we have both calls & puts
        try:
            has_c = int((d["type"] == "C").sum()) if "type" in d.columns else 0
            has_p = int((d["type"] == "P").sum()) if "type" in d.columns else 0
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
                logger.debug(f"Estimated F via PCP: {F_est}")
            else:
                # fallback: use spot if present, else median strike
                if "spot" in d.columns and pd.notna(d["spot"]).any():
                    F_est = float(
                        pd.to_numeric(d["spot"], errors="coerce")
                        .dropna()
                        .iloc[0]
                    )
                    logger.debug(f"Estimated F via spot: {F_est}")
                else:
                    F_est = float(
                        np.nanmedian(
                            pd.to_numeric(d["strike"], errors="coerce")
                        )
                    )
                    logger.debug(f"Estimated F via median strike: {F_est}")
        except Exception as ex_f:
            logger.warning(
                f"Forward estimate failed ({ex_f}); using median strike"
            )
            F_est = float(
                np.nanmedian(pd.to_numeric(d["strike"], errors="coerce"))
            )

        d["F"] = F_est

        T_map[exp] = T
        r_map[exp] = r
        F_map[exp] = F_est
        clean_frames[exp] = d

    logger.info(f"Built maps for {len(clean_frames)} expiries")
    return T_map, r_map, F_map, clean_frames


# ================================================================== #
#                      Async KV-backed compute helpers               #
# ================================================================== #
async def _get_surface_kv(
    frames_by_expiry: Dict[pd.Timestamp, pd.DataFrame],
    T_map: Dict[pd.Timestamp, float],
    F_map: Dict[pd.Timestamp, float],
    r_map: Dict[pd.Timestamp, float],
    smoothing: str,
) -> SVISurface:
    key = _cache_key(
        "svi_surface",
        _hash_frames(frames_by_expiry),
        sorted((str(e), float(T_map[e])) for e in T_map),
        smoothing,
    )
    logger.info(f"[svi] fitting surface (key={key}, smoothing={smoothing})")

    async def _compute():
        t0 = time.perf_counter()
        surface = fit_surface_from_frames(
            frames_by_expiry,
            T_by_expiry=T_map,
            F_by_expiry=F_map,
            r_by_expiry=r_map,
        )
        logger.info(
            f"[svi] raw fit completed in {time.perf_counter()-t0:.3f}s"
        )
        if smoothing != "None":
            method = "cubic_spline" if smoothing == "Cubic spline" else "poly3"
            t1 = time.perf_counter()
            surface = smooth_params(surface, method=method)
            logger.info(
                f"[svi] smoothing '{method}' completed in {time.perf_counter()-t1:.3f}s"
            )
        return surface

    surface: SVISurface = await _kv.get_cached(
        key, _compute, ttl_seconds=CACHE_TTL_SECONDS
    )
    logger.debug(
        f"[svi] surface cached/ready with {len(surface.expiries())} expiries"
    )
    return surface


async def _get_sample_kv(
    usable_frames: Dict[pd.Timestamp, pd.DataFrame],
    k_span: Tuple[float, float],
    k_points: int,
    smoothing: str,
    surface: SVISurface,
) -> Dict[str, np.ndarray]:
    key = _cache_key(
        "svi_surface_sample",
        _hash_frames(usable_frames),
        (float(k_span[0]), float(k_span[1])),
        int(k_points),
        smoothing,
    )
    logger.info(
        f"[svi] sampling grid (key={key}, k_span={k_span}, k_points={k_points})"
    )

    async def _compute():
        t0 = time.perf_counter()
        exps = surface.expiries()
        T_vec = np.array([surface.T[e] for e in exps], dtype=float)
        k_grid = np.linspace(k_span[0], k_span[1], int(k_points))
        w_mat = np.vstack([surface.w(k_grid, e) for e in exps])  # E x K
        iv_mat = np.sqrt(
            np.maximum(w_mat, 0.0) / np.maximum(T_vec[:, None], 1e-12)
        )
        dt = time.perf_counter() - t0
        logger.info(
            f"[svi] sampling done in {dt:.3f}s (E={len(exps)}, K={len(k_grid)})"
        )
        return {
            "k_grid": k_grid,
            "T_vec": T_vec,
            "w_mat": w_mat,
            "iv_mat": iv_mat,
        }

    sample: Dict[str, np.ndarray] = await _kv.get_cached(
        key, _compute, ttl_seconds=CACHE_TTL_SECONDS
    )
    return sample


# ================================================================== #
#                              Sidebar                               #
# ================================================================== #
with st.sidebar:
    st.header("Inputs")

    market = st.selectbox("Market", ["equity", "crypto"], index=0)
    symbol = st.text_input(
        "Symbol / Underlying", value=("AAPL" if market == "equity" else "BTC")
    )
    asof_date = st.date_input(
        "As-of (local date)", value=datetime.now().date()
    )
    asof_time = st.time_input(
        "As-of time (local)", value=datetime.now().time()
    )
    asof_utc = datetime.combine(asof_date, asof_time).astimezone(timezone.utc)
    logger.debug(
        f"Sidebar inputs: market={market}, symbol={symbol}, asof_utc={asof_utc}"
    )

    st.markdown("---")
    st.caption("Risk-free configuration")
    default_rate_val = st.number_input(
        "Default rate (cont. comp.)", value=0.02, step=0.005, format="%.4f"
    )
    sofr_csv_input = st.text_input("SOFR CSV path (optional)", value="")

    rf_cfg = RiskFreeConfig(
        sofr_csv_path=(
            Path(sofr_csv_input).resolve() if sofr_csv_input.strip() else None
        ),
        default_rate=default_rate_val,
        forward_fill=True,
        cache=True,
    )
    rf_provider = RiskFreeProvider(rf_cfg)
    logger.debug("Risk-free provider initialized")

    st.markdown("---")
    st.caption("Preprocessing")
    mid_wide_threshold = st.slider(
        "Wide spread threshold (rel)", 0.01, 0.50, 0.10, 0.01
    )
    min_quotes_per_expiry = st.slider(
        "Minimum quotes per expiry (after cleaning)", 5, 50, 12, 1
    )
    logger.debug(
        f"Preprocess params: mid_wide_threshold={mid_wide_threshold}, "
        f"min_quotes_per_expiry={min_quotes_per_expiry}"
    )

    st.markdown("---")
    st.caption("Surface fitting")
    smoothing = st.selectbox(
        "Parameter smoothing", ["None", "Cubic spline", "Poly3"], index=1
    )
    k_span = st.slider("k-range (log-moneyness)", -2.0, 2.0, (-1.2, 1.2), 0.05)
    k_points = st.select_slider(
        "Grid density", options=[60, 120, 240, 360], value=120
    )
    logger.debug(
        f"SVI params: smoothing={smoothing}, k_span={k_span}, k_points={k_points}"
    )

# ================================================================== #
#                          Fetch & Prepare                           #
# ================================================================== #
t_fetch = time.perf_counter()
quotes_all = _fetch_quotes(market, symbol, asof_utc=asof_utc)
logger.info(
    f"Fetched {len(quotes_all)} quotes in {time.perf_counter()-t_fetch:.3f}s"
)

if quotes_all.empty:
    logger.warning("No quotes returned. Stopping.")
    st.warning("No quotes returned. Check symbol/market/as-of time.")
    st.stop()

st.subheader("Raw Quotes (head)")
logger.debug("Displaying head(20) of quotes")
st.dataframe(quotes_all.head(20), width='stretch')

t_group = time.perf_counter()
frames = _group_frames_by_expiry(quotes_all, expiry_col="expiry")
logger.info(
    f"Grouped into {len(frames)} expiry frames in {time.perf_counter()-t_group:.3f}s"
)
if not frames:
    logger.warning("Grouping failed or 'expiry' column missing")
    st.warning("Quotes missing 'expiry' column or grouping failed.")
    st.stop()

t_maps = time.perf_counter()
T_map, r_map, F_map, clean_frames = _maps_from_frames(
    frames,
    asof_utc=asof_utc,
    rf_provider=rf_provider,
    mid_wide_threshold=mid_wide_threshold,
)
logger.info(
    f"Built maps & cleaned frames for {len(clean_frames)} expiries in {time.perf_counter()-t_maps:.3f}s"
)

# ================================================================== #
#           Build 'usable_frames' with SVI-suitable structure        #
# ================================================================== #
quality_rows = []
usable_frames: Dict[pd.Timestamp, pd.DataFrame] = {}

REQ_UNIQ_STRIKES = 8  # >= distinct strikes
REQ_LEFT_RIGHT = 2  # >= 2 on each side of ATM
REQ_SPAN_K = 0.15  # min span in log-moneyness

logger.info("Applying SVI prerequisites per expiry")
for exp, df in clean_frames.items():
    df = _normalize_type_col(df)
    F = float(F_map.get(exp, np.nan))
    d = _clean_for_svi(df, F)

    # Minimal quote count gate
    if len(d) < min_quotes_per_expiry:
        quality_rows.append(
            {
                "expiry": pd.to_datetime(exp),
                **_expiry_quality_metrics(d),
                "kept": False,
                "reason": "too_few_quotes",
            }
        )
        logger.debug(f"Expiry {exp}: rejected (too few quotes: {len(d)})")
        continue

    # Structural gates
    qm = _expiry_quality_metrics(d)
    reason = None
    if qm["uniq_strikes"] < REQ_UNIQ_STRIKES:
        reason = "few_strikes"
    elif qm["left_k"] < REQ_LEFT_RIGHT or qm["right_k"] < REQ_LEFT_RIGHT:
        reason = "weak_OTM_balance"
    elif qm["span_k"] < REQ_SPAN_K:
        reason = "short_k_span"

    if reason is None:
        usable_frames[exp] = d
        quality_rows.append(
            {"expiry": pd.to_datetime(exp), **qm, "kept": True, "reason": ""}
        )
        logger.debug(
            f"Expiry {exp}: kept (rows={len(d)}, uniq_strikes={qm['uniq_strikes']}, span_k={qm['span_k']:.3f})"
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
        logger.debug(f"Expiry {exp}: rejected ({reason})")

if not usable_frames:
    logger.error("No expiry passed SVI prerequisites")
    st.error("No expiry passed SVI prerequisites after cleaning.")
    qdf = pd.DataFrame(quality_rows).sort_values("expiry")
    if not qdf.empty:
        st.dataframe(qdf, width='stretch')
        st.caption(
            "Tip: Try a different symbol/market, lower the 'Minimum quotes per expiry', or broaden k-range."
        )
    st.stop()
else:
    qdf = pd.DataFrame(quality_rows).sort_values(
        ["kept", "expiry"], ascending=[False, True]
    )
    with st.expander("SVI input quality by expiry"):
        logger.debug("Displaying per-expiry quality DataFrame")
        st.dataframe(qdf, width='stretch')

st.success(
    f"Prepared {len(usable_frames)} expiries (≥{min_quotes_per_expiry} cleaned quotes each)."
)
logger.info(f"Prepared {len(usable_frames)} usable expiries")

# ================================================================== #
#                      Single-Expiry Quick Peek (PCP/BL)             #
# ================================================================== #
with st.expander("Single-Expiry Diagnostics (PCP & BL Density)"):
    pick_exp = st.selectbox(
        "Choose expiry",
        sorted(usable_frames.keys()),
        format_func=lambda t: pd.to_datetime(t).strftime("%Y-%m-%d %H:%M"),
    )
    logger.debug(f"PCP/BL selected expiry: {pick_exp}")
    df_exp = usable_frames[pick_exp]
    df_exp = _normalize_type_col(df_exp)

    T = T_map[pick_exp]
    r = r_map[pick_exp]
    F = F_map[pick_exp]

    st.write(f"**T**={T:.4f} yrs, **r**={r:.4f}, **F**≈{F:.4f}")
    logger.debug(f"PCP/BL params: T={T}, r={r}, F={F}")

    # PCP residuals only if we have both C & P
    has_c = (
        int((df_exp["type"] == "C").sum()) if "type" in df_exp.columns else 0
    )
    has_p = (
        int((df_exp["type"] == "P").sum()) if "type" in df_exp.columns else 0
    )
    if has_c > 0 and has_p > 0:
        sample_spot = F
        t_pcp = time.perf_counter()
        pcp_tbl = add_pcp_diagnostics(
            df_exp,
            spot=sample_spot,
            r=r,
            T=T,
            price_col="mid",
            type_col="type",
            strike_col="strike",
        )
        logger.info(
            f"PCP diagnostics computed in {time.perf_counter()-t_pcp:.3f}s "
            f"(rows={len(pcp_tbl) if pcp_tbl is not None else 0})"
        )
        if not pcp_tbl.empty:
            st.dataframe(
                pcp_tbl.sort_values("residual_abs", ascending=False).head(20),
                width='stretch',
            )
        else:
            st.info("Not enough C/P pairs for PCP diagnostics on this expiry.")
    else:
        logger.debug("PCP skipped (calls or puts missing)")
        st.caption("PCP: skipped — this expiry has only calls or only puts.")

    # BL density (calls only)
    df_calls = df_exp[df_exp["type"] == "C"].dropna(subset=["strike", "mid"])
    if len(df_calls) >= 5:
        K = np.sort(df_calls["strike"].to_numpy(dtype=float))
        C_mid = df_calls.sort_values("strike")["mid"].to_numpy(dtype=float)
        try:
            t_bl = time.perf_counter()
            KK, pdf, _ = bl_pdf_from_calls(K, C_mid, r=r, T=T)
            _, cdf = build_cdf(KK, pdf)
            mom = moments_from_pdf(KK, pdf)
            logger.info(
                f"BL density OK in {time.perf_counter()-t_bl:.3f}s; "
                f"moments: mean={mom['mean']:.4f}, var={mom['var']:.6f}, "
                f"skew={mom['skew']:.3f}, exkurt={mom['exkurt']:.3f}"
            )
            st.write(
                f"**BL moments**: mean={mom['mean']:.4f}, var={mom['var']:.6f}, "
                f"skew={mom['skew']:.3f}, exkurt={mom['exkurt']:.3f}"
            )
            fig_pdf = go.Figure()
            fig_pdf.add_scatter(x=KK, y=pdf, mode="lines", name="pdf")
            fig_pdf.update_layout(
                template="plotly_white",
                title="Breeden–Litzenberger PDF (single expiry)",
                xaxis_title="Strike K",
                yaxis_title="Density",
            )
            st.plotly_chart(fig_pdf, width='stretch')
        except Exception as e:
            logger.warning(f"BL density failed: {e}")
            st.info(f"BL density failed: {e}")
    else:
        logger.debug("BL density skipped (not enough calls)")
        st.info("Not enough call quotes to build BL density.")

# ================================================================== #
#                           SVI Surface Block                        #
# ================================================================== #
st.markdown("## SVI Surface — Multi-Expiry")
logger.info("Fitting SVI surface")

# Fit the surface via async KV cache
try:
    t_fit = time.perf_counter()
    surface: SVISurface = asyncio.run(
        _get_surface_kv(usable_frames, T_map, F_map, r_map, smoothing)
    )
    logger.info(
        f"SVI fit pipeline finished in {time.perf_counter()-t_fit:.3f}s"
    )
except Exception as e:
    logger.error(f"SVI fit failed: {e}")
    st.error(f"SVI fit failed: {e}")
    st.info("Showing per-expiry quality to help you tune filters.")
    st.stop()

# Sample grid (also cached via async KV)
logger.info("Sampling SVI surface on grid")
t_sample = time.perf_counter()
sample = asyncio.run(
    _get_sample_kv(usable_frames, k_span, int(k_points), smoothing, surface)
)
k_grid = sample["k_grid"]
T_vec = sample["T_vec"]
w_mat = sample["w_mat"]
iv_mat = sample["iv_mat"]
logger.info(
    f"Sampling completed in {time.perf_counter()-t_sample:.3f}s "
    f"(k_points={len(k_grid)}, E={len(T_vec)})"
)

# -------------------- Guard against empty / NaN matrices -------------------- #
valid_mask = np.isfinite(iv_mat)
num_valid = int(valid_mask.sum()) if iv_mat.size else 0
logger.debug(f"IV matrix size={iv_mat.size}, valid_points={num_valid}")
if iv_mat.size == 0 or num_valid < max(10, iv_mat.size // 20):
    logger.warning("Not enough valid IV points to render a surface")
    st.warning(
        "Not enough valid IV points to render a surface. "
        "Try widening k-range, increasing grid density, loosening SVI gates, or selecting a more liquid symbol."
    )

# --------------------------- 3D IV Surface --------------------------- #
t_plot_iv = time.perf_counter()
fig_iv = _surface3d(
    k_grid,
    T_vec,
    np.nan_to_num(iv_mat, nan=np.nan),
    title="3D IV Surface — IV(k, T)",
    ztitle="IV",
)
st.plotly_chart(fig_iv, width='stretch')
logger.debug(f"Rendered 3D IV surface in {time.perf_counter()-t_plot_iv:.3f}s")

st.download_button(
    "Download 3D IV surface (HTML)",
    data=_fig_html_bytes(fig_iv),
    file_name=f"surface3d_iv_{symbol}.html",
    mime="text/html",
)
logger.debug("Exposed download for 3D IV surface")

# --------------------------- 3D w(k, T) Surface --------------------------- #
t_plot_w = time.perf_counter()
fig_w = _surface3d(
    k_grid,
    T_vec,
    np.nan_to_num(w_mat, nan=np.nan),
    title="3D Total Variance — w(k, T)=IV²·T",
    ztitle="w",
)
st.plotly_chart(fig_w, width='stretch')
logger.debug(f"Rendered 3D w surface in {time.perf_counter()-t_plot_w:.3f}s")

st.download_button(
    "Download 3D w(k,T) surface (HTML)",
    data=_fig_html_bytes(fig_w),
    file_name=f"surface3d_w_{symbol}.html",
    mime="text/html",
)
logger.debug("Exposed download for 3D w surface")

# ================================================================== #
#                          2D Visualization                          #
# ================================================================== #
st.markdown("## 2D Views")
logger.info("Rendering 2D views")

# --------------------------- 2D: IV(k) slices --------------------------- #
with st.expander("IV vs k (per expiry) — 2D"):
    exps = surface.expiries()
    exps_sorted = sorted(exps, key=lambda e: float(surface.T[e]))
    max_lines = st.slider(
        "Max expiries to overlay",
        3,
        min(10, len(exps_sorted)),
        min(5, len(exps_sorted)),
    )
    step = max(1, len(exps_sorted) // max_lines)
    plot_exps = exps_sorted[::step][:max_lines]
    logger.debug(
        f"2D IV(k) slices: plotting {len(plot_exps)} / {len(exps_sorted)} expiries"
    )

    fig2d_iv = go.Figure()
    for e in plot_exps:
        iv_slice = np.sqrt(
            np.maximum(surface.w(k_grid, e), 0.0) / max(surface.T[e], 1e-12)
        )
        fig2d_iv.add_scatter(
            x=k_grid,
            y=iv_slice,
            mode="lines",
            name=pd.to_datetime(e).strftime("%Y-%m-%d"),
        )
    fig2d_iv.update_layout(
        template="plotly_white",
        title="IV(k) slices across expiries",
        xaxis_title="log-moneyness k",
        yaxis_title="Implied Volatility",
        legend_title="Expiry",
    )
    st.plotly_chart(fig2d_iv, width='stretch')
    logger.debug("Rendered 2D IV(k) overlay")

# --------------------------- 2D: Term structure at k≈0 --------------------------- #
with st.expander("Term structure (ATM IV) — 2D"):
    atm_k = 0.0
    T_vals = []
    iv_atm = []
    labels = []
    for e in exps_sorted:
        T_e = float(surface.T[e])
        w_slice = surface.w(k_grid, e)
        if not np.all(np.isnan(w_slice)):
            iv_slice = np.sqrt(np.maximum(w_slice, 0.0) / max(T_e, 1e-12))
            try:
                iv0 = np.interp(atm_k, k_grid, iv_slice)
                if np.isfinite(iv0):
                    T_vals.append(T_e)
                    iv_atm.append(iv0)
                    labels.append(pd.to_datetime(e).strftime("%Y-%m-%d"))
            except Exception as ex_interp:
                logger.debug(
                    f"ATM interpolation failed for expiry {e}: {ex_interp}"
                )

    if len(T_vals) >= 2:
        fig_ts = go.Figure()
        fig_ts.add_scatter(
            x=T_vals, y=iv_atm, mode="lines+markers", name="ATM IV"
        )
        fig_ts.update_layout(
            template="plotly_white",
            title="Term structure at k ≈ 0",
            xaxis_title="T (years)",
            yaxis_title="Implied Volatility (ATM)",
        )
        st.plotly_chart(fig_ts, width='stretch')
        ts_df = pd.DataFrame(
            {"expiry": labels, "T_years": T_vals, "ATM_IV": iv_atm}
        )
        st.dataframe(ts_df, width='stretch')
        logger.debug(f"Rendered ATM term structure with {len(T_vals)} points")
    else:
        logger.info("Not enough ATM points to draw a term structure")
        st.info("Not enough valid ATM points to draw a term structure.")

# ================================================================== #
#                       Calendar No-Arb Diagnostics                  #
# ================================================================== #
st.markdown("### Calendar Arbitrage Check")
tol = st.number_input(
    "Tolerance on w(k,T2) − w(k,T1) (neg allowed up to –tol)",
    value=1e-10,
    format="%.1e",
)
logger.debug(f"Calendar arb tolerance set to {tol:.1e}")
cal = surface.calendar_violations(k_grid, tol=float(tol))

st.write(
    f"**Overall violations:** {cal['count']} / {cal['n_checks']} points ({cal['fraction']:.1%}) across adjacent maturities"
)
logger.info(
    f"Calendar violations: {cal['count']} of {cal['n_checks']} ({cal['fraction']:.3%})"
)

if cal.get("by_pair"):
    cal_df = pd.DataFrame(
        cal["by_pair"], columns=["T1", "T2", "violation_fraction"]
    )
    st.dataframe(
        cal_df.style.format(
            {"T1": "{:.4f}", "T2": "{:.4f}", "violation_fraction": "{:.1%}"}
        ),
        width='stretch',
    )
    logger.debug(
        f"Rendered by-pair calendar violations table with {len(cal_df)} rows"
    )

# ================================================================== #
#                               Export                               #
# ================================================================== #
st.markdown("### Export sampled grid")
logger.info("Preparing sampled grid for export")
grid_rows: List[Dict[str, object]] = []
exps = surface.expiries()
for i, e in enumerate(exps):
    for j, k in enumerate(k_grid):
        grid_rows.append(
            {
                "expiry": pd.to_datetime(e),
                "T": float(T_vec[i]),
                "k": float(k),
                "K_over_F": float(np.exp(k)),  # consumers can multiply by F(T)
                "w": float(w_mat[i, j]),
                "iv": float(iv_mat[i, j]),
            }
        )
grid_df = pd.DataFrame(grid_rows)
logger.info(
    f"Sampled grid prepared: rows={len(grid_df)}, cols={list(grid_df.columns)}"
)
st.download_button(
    "Download sampled surface (CSV)",
    data=grid_df.to_csv(index=False).encode(),
    file_name=f"svi_surface_sample_{symbol}.csv",
    mime="text/csv",
)
st.caption(
    "Tip: K/F = exp(k). Multiply by per-expiry forward F(T) if you need absolute strikes."
)
logger.debug("Exposed CSV download for sampled surface")
