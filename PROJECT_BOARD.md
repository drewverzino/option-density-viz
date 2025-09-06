# Option Viz Project — Task Board

## To Do

### data/
- async_fetcher.py — implement async functions to pull instruments, tickers, and order books from Deribit API  
  - Use `httpx` or `aiohttp` for concurrency  
  - Add retries with exponential backoff when HTTP 429/5xx occur  
  - Return normalized JSON objects for downstream use  
  - Provide a synchronous wrapper for quick testing  

- cache.py — add local caching layer for API results  
  - Use `functools.lru_cache` or `diskcache` for in-memory + disk caching  
  - Store timestamp + TTL (e.g., 30s for tickers, 1h for instruments)  
  - Expose functions like `get_cached(endpoint, params)`  

- parser.py — decode instrument codes (e.g., BTC-27DEC24-30000-C)  
  - Extract asset, expiry, strike, and option type  
  - Normalize expiry into `datetime`  
  - Return `Instrument` dataclass with fields `{asset, expiry, strike, type, contract_kind}`  

- historical_loader.py — load historical option chains  
  - Accept CSV or Parquet files  
  - Return same schema as live API (so pipeline code works identically)  
  - Add helper to load all expiries for a given day  

- risk_free.py — provide risk-free rate  
  - Load SOFR from a local CSV (date, rate)  
  - Default to constant (e.g., 0.05) if no data available  
  - Function: `get_rate(date: datetime) -> float`  

---

### preprocess/
- midprice.py — robust mid calculation  
  - Formula: `(bid + ask) / 2`  
  - If bid/ask missing → fallback to available side  
  - Flag crossed markets (bid > ask) and wide spreads  

- pcp.py — put–call parity conversion  
  - C + Ke^{-rT} = P + S  
  - Given spot, strike, expiry, and one leg, synthesize the missing price  
  - Return synthesized quotes + residuals for diagnostics  

- forward.py — forward price & log-moneyness  
  - Forward: `F = S * exp(rT)`  
  - Log-moneyness: `k = log(K/F)`  
  - Ensure consistency with PCP output  

---

### vol/
- svi.py — SVI parameter calibration  
  - Implement SVI total variance function: `w(k) = a + b(ρ(k - m) + sqrt((k - m)^2 + σ^2))`  
  - Use grid search for seeds, then L-BFGS to refine  
  - Regularize parameters (e.g., bounds on ρ in (-1,1))  
  - Return dict `{a, b, rho, m, sigma}`  

- no_arb.py — no-arbitrage enforcement  
  - Add butterfly spread positivity checks  
  - Ensure surface monotonic in maturity (calendar arb)  
  - Raise warnings if violated  

- surface.py — volatility surface across maturities  
  - Fit SVI per expiry  
  - Interpolate parameters smoothly across maturities (e.g., splines)  
  - Expose function `get_iv(strike, expiry)`  

---

### density/
- bl.py — Breeden–Litzenberger density  
  - Approximate ∂²C/∂K² using central differences  
  - Support adaptive strike spacing  
  - Ensure density integrates to 1 ± 0.01  

- cdf.py — CDF and quantiles  
  - Numerical integration of PDF  
  - Implement inverse CDF via monotone spline  
  - Return VaR metrics (5%, 95%)  

---

### viz/
- plots.py — static plotting  
  - Smile plot: implied vol vs log-moneyness  
  - Density plot: PDF and CDF with mean/median lines  
  - Use matplotlib (default) and support dark/light themes  

- dashboard.py — Streamlit dashboard  
  - Upload option chain (CSV or live fetch)  
  - Interactive controls for expiry, strike window  
  - Display smile + density plots  
  - Export results to PNG  

---

### cli/
- main.py — CLI entrypoint  
  - Command: `oviz run --asset BTC --expiry 2024-12-27`  
  - Options for input (API vs local CSV), risk-free rate, output dir  
  - Outputs: JSON (params, diagnostics) + PNG plots  

---

### tests/
- Unit tests for each module  
  - Mock API responses for `data/`  
  - Synthetic option chain for calibration/density  
  - Coverage > 80%  
  - Use `pytest` with fixtures  

---

### docs/
- Example notebook  
  - Walkthrough: fetch chain → preprocess → fit SVI → extract density → plot  
  - Include explanations + inline comments  

- README  
  - Installation  
  - Quick start (API + CLI)  
  - Module overview  

---

## In Progress
- (move tasks here as you start working on them)  

## Done
- (move tasks here once merged into main)  
