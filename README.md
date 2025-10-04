# option-density-viz

_Risk-neutral probability density visualization from options on **crypto (BTC/ETH)** and **equities (AAPL/SPY)** using **OKX (public)** and **yfinance**._

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

---

## Overview

`option-density-viz` is a research and visualization tool that extracts **risk-neutral probability densities (RNDs)** from options market data.  
It normalizes live **BTC/ETH** options from the **OKX public API** and **equity** options via **yfinance**, fits **implied volatility smiles** using an arbitrage‑aware **SVI** workflow, stitches a **volatility surface** across maturities, and (next) applies the **Breeden–Litzenberger relation** and **COS method** to compute and plot the probability density of future prices.

This project is designed to help **quantitative researchers, students, and traders** understand how option markets are pricing future outcomes and uncertainty.

> Current status: **Data layer, preprocessing, SVI calibration, surface smoothing, and no‑arbitrage diagnostics are implemented.**  
> The **BL** (finite‑difference) and **COS** (spectral) density pipelines are the next deliverables.

---

## Features

- **Live API ingestion** — pull BTC/ETH option chains from **OKX (public endpoints, no keys required)**, plus equity chains from **yfinance**.
- **Unified schema** — a single `OptionQuote`/`OptionChain` model for crypto and equities.
- **Preprocessing helpers** — robust mids/flags, **PCP diagnostics**, **forward estimation from PCP**.
- **Reproducibility** — save/load chains to **CSV/Parquet**; build offline datasets for experiments.
- **Polite data fetching** — **TTL cache** (in‑mem + SQLite), **async** calls, **rate‑limit** gate + **exponential backoff** retries.
- **SVI calibration** — vega‑weighted least squares (Black‑76), grid‑seeded **L‑BFGS‑B** with safe bounds.
- **No‑arbitrage checks** — butterfly convexity and calendar monotonicity diagnostics.
- 🗺 **Surface fitting** — per‑expiry SVI → smoothed parameters across maturities with evaluators `iv(k,T)` / `w(k,T)`.
- **Visualization** — Matplotlib plots for smiles/surfaces now; Streamlit dashboard planned.
- **RND extraction (roadmap)** — **Breeden–Litzenberger** (finite differences) and **COS** (Fourier) methods.

---

## 🏗 Project Structure

```
option-density-viz/
│── src/
│   ├── data/                     # Backends & plumbing (OKX, yfinance, caching, I/O)
│   │   ├── base.py               # OptionQuote, OptionChain, OptionFetcher protocol
│   │   ├── registry.py           # get_fetcher("equity"|"crypto")
│   │   ├── yf_fetcher.py         # Equity via yfinance (async via to_thread)
│   │   ├── okx_fetcher.py        # Crypto via OKX (public endpoints)
│   │   ├── cache.py              # KVCache (in-mem + SQLite TTL)
│   │   ├── historical_loader.py  # CSV/Parquet save/load
│   │   ├── risk_free.py          # SOFR CSV loader + constant fallback
│   │   └── rate_limit.py         # AsyncRateLimiter + retry_with_backoff
│   ├── preprocess/               # Mids/flags, PCP, forwards & log-moneyness
│   │   ├── midprice.py           # add_midprice_columns(...)
│   │   ├── pcp.py                # residuals, synth legs, strike pivot
│   │   └── forward.py            # forward_price, log_moneyness, PCP forward estimators
│   ├── vol/                      # SVI fitting + no-arb + surfaces
│   │   ├── svi.py                # SVI (Black-76, vega-weighted LS; L-BFGS-B)
│   │   ├── no_arb.py             # butterfly/calendar diagnostics
│   │   └── surface.py            # per-expiry fits → smoothed surface
│   ├── density/                  # BL density, CDF/moments, (planned) COS method
│   ├── viz/                      # Matplotlib plots and (planned) Streamlit dashboard
│   └── cli/                      # Plots and (planned) Streamlit dashboard
│── notebooks/      
│   ├── data_test.ipynb           # End-to-end data validation & demos
│   ├── SVI_Test.ipynb            # SVI per-expiry + surface + diagnostics
│   └── Suite_Test.ipynb          # Full suite: data → preprocess → SVI → surface
│── docs/                         # Images, examples, extended documentation
│── requirements.txt              # Python dependencies
│── README.md                     # Project overview
│── LICENSE                       # License file
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/option-density-viz.git
cd option-density-viz
pip install -r requirements.txt
```

**Dependencies (core):**

- Python 3.10+
- `numpy`, `scipy`, `pandas`
- `matplotlib`
- **Backends:** `httpx`, `yfinance`
- **Optional:** `pyarrow` or `fastparquet` for Parquet, `python-dotenv` for local env loading

> We no longer use Deribit or `websockets`. Crypto data comes from **OKX public REST**.

---

## Usage

### Quickstart (data only)

```python
# examples/quickstart_data.py
import asyncio
from data.registry import get_fetcher

async def main():
    # Equity example (AAPL)
    yf = get_fetcher("equity")
    exps = await yf.list_expiries("AAPL")
    chain_eq = await yf.fetch_chain("AAPL", exps[0])
    print("AAPL spot:", chain_eq.spot, "| quotes:", len(chain_eq.quotes))

    # Crypto example (BTC via OKX public)
    okx = get_fetcher("crypto")
    exps_c = await okx.list_expiries("BTC")
    chain_cr = await okx.fetch_chain("BTC", exps_c[0])
    print("BTC spot:", chain_cr.spot, "| quotes:", len(chain_cr.quotes))

if __name__ == "__main__":
    asyncio.run(main())
```

### SVI & surface quickstart

```python
# examples/quickstart_svi.py
import numpy as np
from preprocess.midprice import add_midprice_columns
from preprocess.forward import yearfrac, estimate_forward_from_chain, log_moneyness
from data.historical_loader import chain_to_dataframe
from vol.svi import calibrate_svi_from_quotes  # fit per-expiry SVI
from vol.surface import fit_surface_from_frames  # build surface across expiries

# assume you already fetched a few expiries into {expiry: chain}
frames = {}
T_map, F_map, r_map = {}, {}, {}
for expiry, chain in frames.items():
    df = add_midprice_columns(chain_to_dataframe(chain))
    T = yearfrac(chain.asof_utc, expiry)
    r = 0.0  # or RiskFreeProvider(...).get_rate(chain.asof_utc.date())
    F = estimate_forward_from_chain(df, r=r, T=T, spot_hint=chain.spot)
    k = log_moneyness(df["strike"], F)

    # Build (k, w, weight) arrays for SVI
    # ... (see notebooks/SVI_Test.ipynb for detailed preparation)

# Then build & smooth a surface
surface = fit_surface_from_frames(frames, T_map, F_map, r_map)
iv_atm = surface.iv(k=0.0, T=list(T_map.values())[0])
print("ATM IV at first expiry:", iv_atm)
```

Or just run the notebooks:

- `notebooks/SVI_Test.ipynb` — SVI per‑expiry, surface smoothing, diagnostics  
- `notebooks/Suite_Test.ipynb` — data → preprocess → SVI → surface

### CLI (recommended)

```bash
# Crypto via OKX public
python -m src.cli.main --asset-class crypto --underlying BTC --expiry 2025-09-26 --out docs/run_btc

# Equities via yfinance
python -m src.cli.main --asset-class equity --underlying AAPL --expiry 2025-12-19 --out docs/run_aapl
```

Artifacts created in `--out`:

- `chain.csv` — normalized chain for the selected expiry  
- `results.json` — r, T, F, SVI fit diagnostics, BL density stats  
- `smile_market.png`, `smile_model.png` — market IV and SVI vs market  
- `density_pdf_cdf.png` — RN PDF and CDF

---

## Results

This section highlights example artifacts and the **metrics** we track to sanity‑check calibrations and densities. Commit a small set of figures to `docs/` so reviewers can skim results without running anything.

### Artifacts (examples)

- `docs/example_smile_aapl.png` — AAPL IV smile at a recent expiry
- `docs/example_surface_btc.png` — BTC IV surface wireframe
- `docs/example_density_btc.png` — BTC risk‑neutral PDF at a recent expiry (coming with BL)
- `docs/example_cdf_btc.png` — BTC CDF with key quantiles (coming with BL/COS)

> Tip: Export plots from the notebooks and drop the files into `docs/` with short, dated filenames.

### Metrics we report

We summarize each snapshot (per expiry) with:

| Metric | Meaning | Target / Check |
|---|---|---|
| Quotes used | Cleaned quotes after filters | Higher is better |
| VW-RMSE (total variance) | Vega‑weighted fit error in **total variance** | ↓ vs. ATM‑only baseline |
| Butterfly violations | Fraction of strikes with negative second derivative | < 1% |
| Calendar violations | Any decrease of total variance with maturity | 0 |
| ∫pdf − 1 | Density normalization error | ≤ 1e‑2 |
| RN mean − Forward (bps) | Risk‑neutral mean vs forward | ≈ 0 |
| Negative pdf rate | Share of grid with pdf < 0 | ≈ 0 |
| Runtime (s) | Seconds per expiry | Informational |

> Until the density modules land, focus results on **data completeness**, **SVI fit quality**, **no‑arb diagnostics**, and **reproducibility** (CSV/Parquet round‑trips).

## Theory Background

- **Breeden–Litzenberger (1978)**: Risk-neutral density can be obtained as the **second derivative** of option prices w.r.t. strike.
- **SVI (Stochastic Volatility Inspired)**: Robust parametrization of implied volatility smiles that enforces arbitrage-aware conditions.
- **COS method**: Fourier expansion method that recovers probability densities from characteristic functions, offering numerical stability.

---

## Theory Deep Dive

This project sits at the intersection of **derivatives pricing** and **numerical analysis**. Below are the core concepts and how we use them.

### 1) Risk‑neutral measure, forwards, and log‑moneyness

- Under the **risk‑neutral (Q) measure**, discounted asset prices are martingales. Pricing expectations are taken under Q with the risk‑free discount.
- **Put–Call Parity (PCP)** for European options (no divs): $C - P = S - K e^{-rT}$. With dividends/carry you use $S_0 e^{-qT}$ or the **forward** $F = S_0 e^{(r-q)T}$.
- We prefer to work in **log‑moneyness**: $k = \ln(K / F)$. This centers smiles across assets and maturities and is the natural coordinate for SVI.

### 2) From implied volatility to total variance

- Market quotes give implied volatilities $\sigma_{imp}(K, T)$ (or prices). We convert to **total variance** $w(k, T) = \sigma_{imp}^2(T) \cdot T$.
- SVI and many no‑arbitrage conditions are easier to express in terms of $w$ (linear in time and convex in strike).

### 3) SVI smile (per expiry) and calibration

- **SVI (Stochastic Volatility Inspired)** parameterizes total variance for a fixed $T$:

  $$w(k) = a + b \left\lbrace \rho\,(k - m) + \sqrt{(k - m)^2 + \sigma^2} \right\rbrace$$

  Parameters: $a$ (level), $b$ (slope), $\rho$ (skew, $|\rho|<1$), $m$ (shift), $\sigma$ (wing curvature, $>0$).
- **Why SVI?** It fits empirical smiles well and admits known **sufficient conditions** for (approximate) no‑arbitrage.
- **Calibration tips used**:
  - **Seeds** from coarse grid / heuristics (ATM slope/curvature).
  - **Vega‑weighted loss** in **total variance** (not IV) to emphasize informative strikes.
  - **Bounds/regularization** on $(a,b,\rho,m,\sigma)$ to avoid pathological wings.
  - (Across maturities) smooth parameters over $T$ and check calendar monotonicity of $w(\cdot,T)$.

### 4) No‑arbitrage checks (sanity layer)

- **Butterfly (static) arbitrage:** Calls must be **convex in K** ($\partial^2C/\partial K^2 \geq 0$). We screen the fitted smile or the price curve for negativity.
- **Calendar arbitrage:** Total variance should be **non‑decreasing in T** for fixed $k$. We spot‑check adjacent maturities.
- **Data hygiene:** Filter **crossed** ($bid>ask$) / **wide** spreads and stale quotes; compute robust mids before fitting.

### 5) Breeden–Litzenberger (BL) density (model‑free; next milestone)

- BL states the **risk‑neutral PDF** is the **second derivative** of the call price with respect to strike (under mild conditions):

  $$f_Q(K,T) = \frac{\partial^2C(K,T)}{\partial K^2} \cdot e^{rT}$$

- Numerically fragile → we will stabilize by:
  - Smoothing the **call price curve** in $K$ (e.g., monotone splines / regularized fits).
  - Using **central or higher‑order finite differences** with **adaptive spacing**.
  - Enforcing **boundary behavior** (wing extrapolation consistent with forwards).
- Diagnostics: $f_Q \geq 0$, $\int f_Q \mathrm{d}K \approx 1$, RN mean ≈ forward.

### 6) COS method (spectral, model‑driven; next)

- **COS** recovers densities/prices from a **characteristic function** via a truncated cosine series on $[a,b]$:

  $$C(K) \approx e^{-rT} \sum_{n=0}^{N-1} \text{Re}\left( \phi( u_n ) \cdot F_n(K) \right)$$

  where $\phi$ is the CF of log‑price, $u_n = n\pi/(b-a)$, and $F_n$ are known coefficients.
- Key practical choices:
  - **Truncation bounds** $[a,b]$ from **cumulants** (match mean/variance/skew/kurtosis).
  - **Series length** $N$ from an error budget (trade speed vs accuracy).
- COS is extremely fast and stable once $\phi$ is known; we will use it to cross‑check BL.

### 7) Diagnostics & consistency checks

- **Normalization**: $\left| \int f_Q - 1 \right| \leq 1e-2$.
- **Forward consistency**: RN mean vs forward within a few **bps**.
- **Moment checks**: Compare variance/skew/kurtosis from the density to those implied by the smile.
- **Price‑from‑density**: Re‑integrate the density to recover call prices and measure error.

---

## 🛠 Roadmap

- [ ] **BL** derivative (high‑order finite differences) and **COS** pipeline
- [ ] Streamlit dashboard (interactive smiles/densities)
- [ ] Calibration comparisons (SVI vs SABR)
- [ ] Additional asset classes (index options like SPX/QQQ)

---

## Contributing

Contributions are welcome!

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/my-feature`)  
3. Commit changes (`git commit -m 'Add feature'`)  
4. Push branch (`git push origin feature/my-feature`)  
5. Open a Pull Request

---

## License

MIT License © 2025 `option-density-viz` contributors.  
See [LICENSE](LICENSE) for details.

---

## Authors

- [Drew Verzino](https://github.com/drewverzino)
- [Rahul Rajesh](https://github.com/RajeshGang)
- [Tinashe Dzemwa](https://github.com/tinashe13)

---

## Acknowledgments

- **OKX** for public crypto options data  
- **yfinance** for convenient equity options data  
- Jim Gatheral for the **SVI volatility smile framework**  
- Breeden & Litzenberger (1978) for the foundational density extraction m


## Troubleshooting
- **Empty chains**: If Polygon returns 0 quotes, the CLI falls back to yfinance automatically.
- **Missing strike column**: The CLI now exits gracefully, writing a minimal results.json instead of crashing.
- **Streamlit slider error (min==max)**: Fixed — slider disabled when only one expiry is available.
- **asyncio.run() in Jupyter**: Use the async-safe Polygon notebook (`oviz_polygon_live_tests_async.ipynb`).
