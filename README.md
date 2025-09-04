# option-density-viz

_Risk-neutral probability density visualization from crypto options (BTC/ETH) using Deribit data._

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

---

## 📌 Overview

`option-density-viz` is a research and visualization tool that extracts **risk-neutral probability densities (RNDs)** from options market data.  
Using live BTC and ETH options data from the [Deribit API](https://docs.deribit.com/), the project fits **implied volatility smiles** with an arbitrage-free **SVI model**, then applies the **Breeden–Litzenberger relation** and **COS method** to compute and plot the probability density of future prices.

This project is designed to help **quantitative researchers, students, and traders** understand how option markets are pricing future outcomes and uncertainty.

---

## ✨ Features

- 🔗 **Live API ingestion** — pull BTC/ETH options chains directly from Deribit.
- 📈 **Implied volatility surface fitting** — SVI parameterization with no-arbitrage constraints.
- 🧮 **Risk-neutral density extraction** — via Breeden–Litzenberger (second derivative) and COS Fourier methods.
- 🎨 **Interactive visualization** — Plotly-based density plots and animations across maturities.
- 📊 **Analytics add-ons** — compute implied mean, skewness, and kurtosis of the distribution.
- 💾 **Export support** — save plots as PNG/GIF/MP4 for research or presentations.

---

## 🏗 Project Structure

```
option-density-viz/
│── src/
│   ├── data/             # Deribit API fetching & cleaning
│   ├── vol/              # Implied vol fitting (SVI, sanity checks)
│   ├── density/          # Risk-neutral density methods (BL, COS)
│   └── viz/              # Plotting and dashboards
│── notebooks/            # Prototyping & demo notebooks
│── docs/                 # Images, examples, extended documentation
│── requirements.txt      # Python dependencies
│── README.md             # Project overview
│── LICENSE               # License file
```

---

## ⚡ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/option-density-viz.git
cd option-density-viz
pip install -r requirements.txt
```

**Dependencies:**

- Python 3.10+
- `numpy`, `scipy`, `pandas`
- `plotly`, `matplotlib`
- `requests`, `websockets` (for Deribit API)

---

## 🚀 Usage

Run the main script to fetch data, fit the volatility surface, and plot the density:

```bash
python src/main.py --asset BTC --expiry 2025-09-30
```

**Arguments:**

- `--asset` : underlying asset (`BTC` or `ETH`)
- `--expiry` : option expiry date in `YYYY-MM-DD` format
- `--method` : density estimation method (`bl` or `cos`)

Example output:

![Demo Plot](docs/example_density.png)

---

## 📚 Theory Background

- **Breeden–Litzenberger (1978)**: Risk-neutral density can be obtained as the **second derivative** of option prices w.r.t. strike.
- **SVI (Stochastic Volatility Inspired)**: Robust parametrization of implied volatility smiles that enforces arbitrage-free conditions.
- **COS method**: Fourier expansion method that recovers probability densities from characteristic functions, offering numerical stability.

---

## 🛠 Roadmap

- [ ] Add support for equity index options (SPX, QQQ)
- [ ] Implement Monte Carlo density estimation as a baseline
- [ ] Deploy interactive dashboard via Streamlit
- [ ] Add calibration comparisons (SVI vs SABR)

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
- Rahul Rajesh
- Tinashe Dwemza

---

## TODO List:

### Data Ingestion & Infrastructure

- [ ] Asynchronous API layer – Add support for concurrent fetching of instruments, tickers and order books using asyncio or multi‑threading to reduce latency. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Local caching & retries – Implement caching of API responses and exponential backoff to gracefully handle network failures and rate limits. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Instrument parsing utilities – Create helpers to decode Deribit instrument codes (e.g. `BTC-27DEC24-30000-C`) and normalise inverse vs linear option conventions. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Historical data loader – Provide functions to load option chains from CSV/Parquet files for offline analysis and unit tests. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Risk‑free rate service – Add a module to supply SOFR or another benchmark rate, configurable via a local dataset or constant. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] API rate‑limit handling – Detect API limits and automatically throttle or queue requests. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Unit tests for API calls – Provide mock API responses and test error handling for all data‑layer functions. (Assigned to: **\_\_**, Status: **\_\_**)

### Pre‑Processing Enhancements

- [ ] Bid/ask mid calculation – Compute robust mid prices by combining bid and ask quotes, and handle missing or outlier quotes. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Put‑call parity conversion – Synthesise call (or put) prices from available quotes using spot, strike and discount factors. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Forward price estimation – Derive the implied forward price from spot and discount factors; use it to calculate log‑moneyness for SVI. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Interpolation & smoothing – Interpolate option prices or implied volatilities across strikes using monotonic splines or kernel regressions prior to differentiation. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Strike range automation – Automatically select a moneyness range based on data distribution and coverage. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Logging framework – Implement structured logging for all preprocessing steps. (Assigned to: **\_\_**, Status: **\_\_**)

### Advanced Volatility Surface Modelling

- [ ] SVI calibration routines – Implement robust optimisers (e.g. grid search, Nelder–Mead, L‑BFGS) to fit SVI parameters and apply regularisation where needed. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] No‑arbitrage enforcement – Build constraints and post‑calibration checks to ensure the fitted smile is free from butterfly and calendar arbitrage. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Alternative models – Extend the module to include other smiles such as SABR or arbitrage‑free splines; provide a common API for switching models. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Surface interpolation across maturities – Fit SVI parameters across maturities and smooth them in the time dimension to create a full volatility surface. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Multi‑model calibration comparison – Create utilities to compare fit quality between SVI, SABR, Heston etc. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Volatility surface diagnostics – Visualise and report slope, curvature and skew across moneyness and time. (Assigned to: **\_\_**, Status: **\_\_**)

### Precision Risk‑Neutral Density Extraction

- [ ] High‑order finite differences – Use central or higher‑order difference schemes and adaptive spacing for the Breeden–Litzenberger derivative. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Spline‑based differentiation – Fit smooth curves (e.g. cubic splines) to call prices and differentiate analytically for a more stable density. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] CDF and quantile functions – Integrate the density to produce the cumulative distribution function and implement inverse‑CDF for quantiles. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Generalised COS framework – Allow users to supply characteristic functions for different models and automatically choose integration bounds and series length. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Normalisation & diagnostics – Ensure densities integrate to one, compute moments (mean, variance, skewness, kurtosis) and verify consistency with forwards. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Density interpolation – Interpolate densities across strike to improve stability before plotting. (Assigned to: **\_\_**, Status: **\_\_**)

### Enhanced Visualisation & User Interaction

- [ ] 3D surface plots – Create interactive 3D surfaces of implied volatilities and risk‑neutral densities over strike and maturity. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Animation & dashboard – Build dashboards (e.g. with Streamlit or Dash) and animated GIF/MP4 outputs that show how densities evolve over time. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Density diagnostics – Overlay key statistics (mean, mode, IQR, tail probabilities) and allow switching between BL and COS methods interactively. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Custom themes & export options – Support different themes (light/dark) and allow exporting figures in multiple formats (PNG, SVG, GIF, MP4). (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Parameter sliders – Add interactive controls to tweak model parameters and see real‑time changes in the plots. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Notebook integration – Provide helpers for embedding interactive plots in Jupyter notebooks. (Assigned to: **\_\_**, Status: **\_\_**)

### Command‑Line & API Integration

- [ ] Robust CLI workflow – Extend the CLI to handle batch processing of multiple expiries and output results in JSON/CSV formats. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Configuration files – Let users provide configuration via YAML or TOML for repeatable runs (assets, expiries, risk‑free rates). (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] RESTful microservice – Expose a lightweight API using FastAPI or Flask for programmatic density queries. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Environment management – Add `pyproject.toml` or `setup.cfg` for packaging, and optionally a Dockerfile for containerised execution. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Output formatting – Support additional output formats (e.g. Parquet, Excel) from the CLI. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Error handling & user feedback – Improve CLI messages and exit codes to be more informative. (Assigned to: **\_\_**, Status: **\_\_**)

### Testing, CI/CD & Packaging

- [ ] Unit & integration tests – Cover API parsers, SVI calibration and density algorithms with pytest and mocks. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Property‑based testing – Use Hypothesis or similar to test invariants like density integration equals one. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Continuous integration – Configure GitHub Actions to run tests, lint code, and check formatting on each commit. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Documentation generation – Generate API docs with Sphinx or MkDocs and host them via GitHub Pages. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Package distribution – Configure packaging metadata to publish the project on PyPI for easy installation. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Code quality tools – Integrate flake8/black or similar for style enforcement. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Coverage thresholds – Enforce minimum test coverage in CI. (Assigned to: **\_\_**, Status: **\_\_**)

### Expansion & Future Work

- [ ] Additional asset classes – Add support for equity, FX or swaption options once appropriate data sources are integrated. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Real‑time alerts – Implement notification services (email, Slack) when density metrics move beyond thresholds. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Machine‑learning extensions – Explore neural network volatility surfaces and compare them to SVI or SABR fits. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Educational modules – Develop tutorial notebooks and course‑style explanations that use the codebase for teaching. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Community contributions – Expand `CONTRIBUTING.md` and outline “good first issues” to encourage participation. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Hedging strategies – Explore using the estimated densities to inform hedging and trading strategies. (Assigned to: **\_\_**, Status: **\_\_**)
- [ ] Documentation examples – Create example scripts and notebooks demonstrating each module. (Assigned to: **\_\_**, Status: **\_\_**)

---

## 🙌 Acknowledgments

- Deribit for providing live crypto options data
- Jim Gatheral for the **SVI volatility smile framework**
- Breeden & Litzenberger (1978) for the foundational density extraction method
