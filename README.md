# option-density-viz

_Risk-neutral probability density visualization from options on **crypto (BTC/ETH)** and **equities (AAPL/SPY)** using **OKX (public)** and **yfinance**._

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

---

## 📌 Overview

`option-density-viz` extracts **risk-neutral probability densities (RNDs)** from options data.  
It normalizes live **BTC/ETH** options from the **OKX public API** and **equity** options via **yfinance**, fits **SVI** implied-vol smiles with basic **no-arbitrage** checks, computes a **Breeden–Litzenberger (BL)** density, and renders clean **Matplotlib** plots. A simple **CLI** ties the whole pipeline together.

> Status: data (OKX+yfinance), caching, risk-free rates, preprocessing, **SVI calibration**, **BL density**, **viz**, and **CLI** are implemented. COS method and term-structure smoothing are on the roadmap.

---

## ✨ Features

- 🔗 **Live ingestion** — BTC/ETH via **OKX (public REST)**; equities via **yfinance**.  
- 🧱 **Unified schema** — one `OptionQuote`/`OptionChain` for crypto & equities.  
- 💾 **Reproducibility** — save/load chains to **CSV/Parquet**.  
- 🚦 **Resilience** — async fetching, **TTL cache** (in-mem + SQLite), **rate limiter** + **exponential backoff**.  
- 📈 **SVI calibration** — price→IV bootstrapping, vega/price-weighted losses, bounds/regularization, diagnostics.  
- 🧮 **RND (BL)** — stabilized second-derivative with smoothing, clipping & renormalization.  
- 🎨 **Plots** — smiles, SVI vs market, PDF/CDF (light/dark).  
- 🔧 **CLI** — single-run command produces artifacts and a `results.json` summary.
- 🪵 **Structured logging** — console or JSON logs with request-level context and timings.

---

## 🏗 Project Structure

```
option-density-viz/
│── src/
│   ├── data/               # OKX, yfinance, cache, I/O, risk-free, rate limiting
│   ├── preprocess/         # mids, PCP synthesis, forward/log-moneyness
│   ├── vol/                # SVI, no-arb checks, (basic) surface helpers
│   ├── density/            # BL density, CDF/moments
│   ├── viz/                # Matplotlib plots
│   └── cli/                # CLI entrypoint
│── notebooks/              # data_test.ipynb, suite_test.ipynb
│── docs/                   # artifacts gallery (user-generated)
│── requirements.txt
│── README.md
│── LICENSE
```

---

## ⚡ Installation

```bash
git clone https://github.com/yourusername/option-density-viz.git
cd option-density-viz
python -m venv .venv && . .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Optional:** for Parquet I/O install `pyarrow` or `fastparquet`.

---

## 🚀 Usage

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

### Notebooks

- `notebooks/data_test.ipynb` — fetch, inspect, export  
- `notebooks/suite_test.ipynb` — **data → preprocess → vol → density** (end-to-end)

---

## 🪵 Logging

The project emits consistent, contextual logs across modules (data → preprocess → vol → density → viz → CLI).  
Choose **console** or **JSON** formatting via environment variables:

```bash
# Console logs (default)
OVIZ_LOG_LEVEL=INFO python -m src.cli.main ...

# JSON logs (great for piping into tools)
OVIZ_LOG_JSON=1 OVIZ_LOG_LEVEL=DEBUG python -m src.cli.main ...
```

Typical log fields include `asset`, `backend`, `expiry`, `request_id`, plus each step’s diagnostics (counts, parameters, timings).  
You can set request-level context in Python via `utils.logging.set_context(asset="BTC", backend="okx", expiry="2025-09-26")`.

---

## 📊 Results

Ship a small gallery in `docs/` to showcase outputs (commit your PNGs). We track:

| Metric | Meaning | Target / Check |
|---|---|---|
| Quotes used | Cleaned quotes after filters | Higher is better |
| VW-RMSE (total variance) | Vega/price-weighted SVI loss | ↓ vs ATM baseline |
| Butterfly violations | Discrete convexity negatives | < 1% |
| Calendar violations | Total variance vs T decreases | 0 |
| ∫pdf − 1 | Density normalization error | ≤ 1e-2 |
| RN mean − Forward (bps) | Consistency check | ≈ 0 |

---

## 📚 Theory (brief)

- **Put–Call Parity / Forward**: use PCP pairs to infer forward; work in log-moneyness \(k=\ln(K/F)\).  
- **SVI smile**: \( w(k)=a+b\{\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}\} \) fitted per expiry in total variance.  
- **BL density**: \( f_Q(K,T)=e^{rT}\partial^2 C/\partial K^2 \) stabilized via smoothing, central differences, clipping/renorm.

> COS method (spectral) and smoothed term-structure are planned for cross-checks and speed.

---

## 🤝 Contributing

PRs welcome: tidy functions, add tests, and propose diagnostics. See the Project Board for open items.

---

## 📜 License

MIT © 2025 option-density-viz contributors.
