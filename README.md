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

## 📚 Theory Background

- **Breeden–Litzenberger (1978)**: Risk-neutral density can be obtained as the **second derivative** of option prices w.r.t. strike.
- **SVI (Stochastic Volatility Inspired)**: Robust parametrization of implied volatility smiles that enforces arbitrage-aware conditions.
- **COS method**: Fourier expansion method that recovers probability densities from characteristic functions, offering numerical stability.

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
