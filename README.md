# option-density-viz

*Risk-neutral probability density visualization from crypto options (BTC/ETH) using Deribit data.*

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

- Drew Verzino 
- Rahul Rajesh 
- Tinashe Dwemza

---

## 🙌 Acknowledgments

- Deribit for providing live crypto options data  
- Jim Gatheral for the **SVI volatility smile framework**  
- Breeden & Litzenberger (1978) for the foundational density extraction method  
