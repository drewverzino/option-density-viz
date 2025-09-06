# option-density-viz

_Risk-neutral probability density visualization from options on **crypto (BTC/ETH)** and **equities (AAPL/SPY)** using **OKX (public)** and **yfinance**._

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

---

## 📌 Overview

`option-density-viz` is a research and visualization tool that extracts **risk-neutral probability densities (RNDs)** from options market data.  
It normalizes live **BTC/ETH** options from the **OKX public API** and **equity** options via **yfinance**, fits (planned) **implied volatility smiles** using an arbitrage-aware **SVI** workflow, then applies the **Breeden–Litzenberger relation** and (planned) **COS method** to compute and plot the probability density of future prices.

This project is designed to help **quantitative researchers, students, and traders** understand how option markets are pricing future outcomes and uncertainty.

> Note: The data layer (OKX + yfinance), caching, rate-limiting/retries, historical I/O, and risk-free rates are implemented. SVI fitting, no-arb checks, and COS/BL pipelines are an active roadmap.

---

## ✨ Features

- 🔗 **Live API ingestion** — pull BTC/ETH option chains from **OKX (public endpoints, no keys required)**, plus equity chains from **yfinance**.
- 🧱 **Unified schema** — a single `OptionQuote`/`OptionChain` model for crypto and equities.
- 💾 **Reproducibility** — save/load chains to **CSV/Parquet**; build offline datasets for experiments.
- 🚦 **Polite data fetching** — **TTL cache** (in-mem + SQLite), **async** calls, **rate-limit** gate + **exponential backoff** retries.
- 📈 **Implied volatility surface fitting (SVI)** — scaffolding in place; no-arbitrage checks on the roadmap.
- 🧮 **Risk-neutral density extraction** — via **Breeden–Litzenberger** (second derivative) and **COS** (Fourier) methods (roadmap).
- 🎨 **Visualization** — Matplotlib plots for smiles/densities; Streamlit dashboard planned.
- 📊 **Analytics add-ons** — implied mean, skewness, and kurtosis of the distribution (roadmap).

---

## 🏗 Project Structure

```
option-density-viz/
│── src/
│   ├── data/               # Backends & plumbing (OKX, yfinance, caching, I/O)
│   │   ├── base.py         # OptionQuote, OptionChain, OptionFetcher protocol
│   │   ├── registry.py     # get_fetcher("equity"|"crypto")
│   │   ├── yf_fetcher.py   # Equity via yfinance (async via to_thread)
│   │   ├── okx_fetcher.py  # Crypto via OKX (public endpoints)
│   │   ├── cache.py        # KVCache (in-mem + SQLite TTL)
│   │   ├── historical_loader.py  # CSV/Parquet save/load
│   │   ├── risk_free.py    # SOFR CSV loader + constant fallback
│   │   └── rate_limit.py   # AsyncRateLimiter + retry_with_backoff
│   ├── vol/                # (Roadmap) SVI fitting, no-arb checks, surfaces
│   ├── density/            # (Roadmap) BL finite diffs, COS method
│   └── viz/                # Plots and (planned) Streamlit dashboard
│── notebooks/
│   └── data_test.ipynb     # End-to-end data validation & demos
│── docs/                   # Images, examples, extended documentation
│── requirements.txt        # Python dependencies
│── README.md               # Project overview
│── LICENSE                 # License file
```

---

## ⚡ Installation

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

## 🚀 Usage

Run the **validation notebook** to fetch data, inspect a chain, and export CSV/Parquet:

```text
notebooks/data_test.ipynb
```

Or try a minimal script:

```python
# examples/quickstart.py
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

<!-- Example output:

![Demo Plot](docs/example_density.png) -->

---

## 📊 Results

This section highlights example artifacts and the **metrics** we track to sanity‑check calibrations and densities. Commit a small set of figures to `docs/` so reviewers can skim results without running anything.

### Artifacts (examples)

- `docs/example_smile_aapl.png` — AAPL IV smile at a recent expiry
- `docs/example_density_btc.png` — BTC risk‑neutral PDF at a recent expiry
- `docs/example_cdf_btc.png` — BTC CDF with key quantiles

> Tip: Export plots from the validation notebook and drop the files into `docs/` with short, dated filenames.

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

> Until the modeling modules land, focus results on **data completeness**, **consistency** (PCP/forward checks), and **reproducibility** (CSV/Parquet round‑trips).

### Reproduce these results

1. Run `notebooks/data_test.ipynb` for **AAPL** and **BTC**.  
2. Export figures to `docs/` (e.g., `example_smile_aapl.png`, `example_density_btc.png`).  
3. (When modeling modules are added) run the “Report” cell to print/save a metrics table (CSV/JSON) to `docs/results/`.

## 📚 Theory Background

- **Breeden–Litzenberger (1978)**: Risk-neutral density can be obtained as the **second derivative** of option prices w.r.t. strike.
- **SVI (Stochastic Volatility Inspired)**: Robust parametrization of implied volatility smiles that enforces arbitrage-aware conditions.
- **COS method**: Fourier expansion method that recovers probability densities from characteristic functions, offering numerical stability.

---

## 🧠 Theory Deep Dive

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

  $$w(k) = a + b \left\{ \rho (k - m) + \sqrt{ (k - m)^2 + \sigma^2 } \right\}$$

  Parameters: $a$ (level), $b$ (slope), $\rho$ (skew, $|\rho|<1$), $m$ (shift), $\sigma$ (wing curvature, $>0$).
- **Why SVI?** It fits empirical smiles well and admits known **sufficient conditions** for (approximate) no‑arbitrage.
- **Calibration tips used/planned**:
  - **Seeds** from coarse grid / heuristics (ATM slope/curvature).
  - **Vega‑weighted loss** in **total variance** (not IV) to emphasize informative strikes.
  - **Bounds/regularization** on $(a,b,\rho,m,\sigma)$ to avoid pathological wings.
  - (Across maturities) smooth parameters over $T$ and check calendar monotonicity of $w(\cdot,T)$.

### 4) No‑arbitrage checks (sanity layer)

- **Butterfly (static) arbitrage:** Calls must be **convex in K** ($\partial^2C/\partial K^2 \geq 0$). We screen the fitted smile or the price curve for negativity.
- **Calendar arbitrage:** Total variance should be **non‑decreasing in T** for fixed $k$. We spot‑check adjacent maturities.
- **Data hygiene:** Filter **crossed** ($bid>ask$) / **wide** spreads and stale quotes; compute robust mids before fitting.

### 5) Breeden–Litzenberger (BL) density (model‑free)

- BL states the **risk‑neutral PDF** is the **second derivative** of the call price with respect to strike (under mild conditions):

  $$f_Q(K,T) = \frac{\partial^2C(K,T)}{\partial K^2} \cdot e^{rT}$$

- Numerically fragile → we stabilize by:
  - Smoothing the **call price curve** in $K$ (e.g., monotone splines / regularized fits).
  - Using **central or higher‑order finite differences** with **adaptive spacing**.
  - Enforcing **boundary behavior** (wing extrapolation consistent with forwards).
- Diagnostics: $f_Q \geq 0$, $\int f_Q dK \approx 1$, RN mean ≈ forward.

### 6) COS method (spectral, model‑driven)

- **COS** recovers densities/prices from a **characteristic function** via a truncated cosine series on $[a,b]$:

  $$C(K) \approx e^{-rT} \sum_{n=0}^{N-1} \text{Re}\left( \phi( u_n ) \cdot F_n(K) \right)$$

  where $\phi$ is the CF of log‑price, $u_n = n\pi/(b-a)$, and $F_n$ are known coefficients.
- Key practical choices:
  - **Truncation bounds** $[a,b]$ from **cumulants** (match mean/variance/skew/kurtosis).
  - **Series length** $N$ from an error budget (trade speed vs accuracy).
- COS is extremely fast and stable once $\phi$ is known; we use it to cross‑check BL.

### 7) Diagnostics & consistency checks

- **Normalization**: $\left| \int f_Q - 1 \right| \leq 1e-2$.
- **Forward consistency**: RN mean vs forward within a few **bps**.
- **Moment checks**: Compare variance/skew/kurtosis from the density to those implied by the smile.
- **Price‑from‑density**: Re‑integrate the density to recover call prices and measure error.

> In practice, a robust pipeline alternates between **modeling** (SVI/COS) and **model‑free** (BL) views, using diagnostics to decide when to trust which.

---

## 🛠 Roadmap

- [ ] SVI calibration with regularization + no-arbitrage checks
- [ ] BL derivative (high-order finite differences) and COS pipeline
- [ ] Streamlit dashboard (interactive smiles/densities)
- [ ] Calibration comparisons (SVI vs SABR)
- [ ] Additional asset classes (index options like SPX/QQQ)

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/my-feature`)  
3. Commit changes (`git commit -m 'Add feature'`)  
4. Push branch (`git push origin feature/my-feature`)  
5. Open a Pull Request

---

## 📜 License

MIT License © 2025 `option-density-viz` contributors.  
See [LICENSE](LICENSE) for details.

---

## 👥 Authors

- [Drew Verzino](https://github.com/drewverzino)
- [Rahul Rajesh](https://github.com/RajeshGang)
- [Tinashe Dwemza](https://github.com/tinashe13)

---

## 🙌 Acknowledgments

- **OKX** for public crypto options data  
- **yfinance** for convenient equity options data  
- Jim Gatheral for the **SVI volatility smile framework**  
- Breeden & Litzenberger (1978) for the foundational density extraction method
