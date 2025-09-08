# Option Viz — Project Board

## Legend

Status: 🟦 To Do · 🟨 In Progress · 🟩 Done  
Tags: 🧱 Infra · 🔑 Auth · 📈 Modeling · 📐 Density · 🧮 Preprocess · 📊 Viz · 🧪 Tests · 📦 CLI/Docs · 🪵 Logging

Tip: Assign an owner by replacing `Owner: ___`. Move items between the three status sections below.

---

## 🟦 To Do

### vol/ (smiles, surfaces) 📈

- [ ] vol/surface.py — maturity smoothing & term-structure diagnostics  
  Owner: Drew · Tags: 📈  
  <details><summary>Spec & DoD</summary>

  - Smooth per-expiry SVI parameters over T (spline / low-order poly).  
  - Add calendar monotonicity checks on total variance across maturities.  
  - DoD: term surface yields continuous Σ(k,T); calendar constraints pass on sample snapshot.
  </details>

### density/ (RND) 📐

- [ ] density/cos.py — COS (characteristic-function) method  
  Owner: ___ · Tags: 📐  
  <details><summary>Spec & DoD</summary>

  - Implement generic COS engine with cumulant-based [a,b] bounds and series control N.  
  - Provide a Black–Scholes CF baseline and parity check vs closed form.  
  - DoD: prices match BS within 1e-4; density normalizes to 1±1e-3.
  </details>

### viz/ (plots, dashboard) 📊

- [ ] viz/dashboard.py — Streamlit demo  
  Owner: ___ · Tags: 📊  
  <details><summary>Spec & DoD</summary>

  - Select backend (equity/crypto), ticker, expiry; run full pipeline; export plots.  
  - DoD: one-file app runs locally and updates interactively.
  </details>

### cli/, tests/, docs/ 📦

- [ ] tests/cli_smoke.py — CLI smoke + artifact checks 🧪  
  Owner: ___ · Tags: 🧪  
  <details><summary>Spec & DoD</summary>

  - Run CLI for AAPL (equity) and BTC (crypto) with a short grid; assert artifacts exist.  
  - DoD: green locally; skips network by flag when needed.
  </details>

- [ ] docs/gallery — curated artifact images 📦  
  Owner: ___ · Tags: 📦  
  <details><summary>Spec & DoD</summary>

  - Commit a few PNGs (smile + density) for BTC and AAPL with timestamps.  
  - DoD: README links to these images.
  </details>

---

## 🟨 In Progress

- [ ] vol/surface.py — basic multi-expiry helpers (wiring)  
  Owner: Drew · Tags: 📈  
  <details><summary>Now</summary>

  - Utility to fit SVI per-expiry and return a dict of fits; expose `get_iv(strike, expiry)` via nearest/linear blend.  
  - Next: smoothing across T and calendar checks.
  </details>

---

## 🟩 Done

### data/ 🧱

- [x] data/base.py — shared types and interface  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>

  - `OptionQuote`, `OptionChain` dataclasses; `OptionFetcher` protocol (`list_expiries`, `fetch_chain`).  
  - Backend-agnostic for crypto/equity.
  </details>

- [x] data/registry.py — backend factory  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>

  - `get_fetcher("equity"|"crypto")` → `YFinanceFetcher` / `OKXFetcher`; kwargs forwarded.  
  - Smoke test returns ≥1 quote for AAPL and BTC.
  </details>

- [x] data/yf_fetcher.py — equities via yfinance (sync → `asyncio.to_thread`)  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>

  - Robust spot retrieval; `.option_chain()` handling; IV forwarded via `extra["iv"]` when present.
  </details>

- [x] data/okx_fetcher.py — crypto via OKX (public endpoints)  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>

  - YYMMDD/ YYYYMMDD expiry parsing; server-time sync; `AsyncRateLimiter` + backoff for ticker calls; instrument list cached via `KVCache` (1h TTL).  
  - Public instruments/tickers/index spot; helpful error body on failures.
  </details>

- [x] data/cache.py — in-memory + SQLite TTL cache  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>

  - `KVCache` with TTL and persistence; `get_cached(key, fetch_fn, ttl)`; `vacuum_disk()` helper.
  </details>

- [x] data/historical_loader.py — CSV/Parquet save/load  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>

  - `chain_to_dataframe`, `dataframe_to_chain`; `save_chain_csv/parquet`, `load_chain_csv/parquet`.  
  - Batch helpers; tz-safe expiries/as-of.
  </details>

- [x] data/risk_free.py — SOFR loader + constant fallback  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>

  - `RiskFreeProvider(get_rate)` with CSV support, forward-fill option, and default constant.
  </details>

- [x] data/rate_limit.py — polite throttling utilities  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>

  - `AsyncRateLimiter(max_concurrent)` and `retry_with_backoff(...)` with jitter.  
  - Patterns integrated into OKX fetcher.
  </details>

### preprocess/ 🧮

- [x] preprocess/midprice.py — robust mids + flags  
  Owner: Drew · Tags: 🧮  
  <details><summary>What shipped</summary>

  - Vectorized mids with fallback (bid-only/ask-only), crossed/wide flags, relative spread; configurable thresholds.
  </details>

- [x] preprocess/pcp.py — PCP synthesis + diagnostics  
  Owner: Drew · Tags: 🧮  
  <details><summary>What shipped</summary>

  - `synthesize_missing_leg`, residual computation, strike pivot helpers.
  </details>

- [x] preprocess/forward.py — forward & log-moneyness  
  Owner: Drew · Tags: 🧮  
  <details><summary>What shipped</summary>

  - `estimate_forward_from_pcp` (pairs), `yearfrac`, carry model fallback.
  </details>

### vol/ (smiles, surfaces) 📈

- [x] vol/svi.py — SVI calibration (quotes→fit)  
  Owner: Drew · Tags: 📈  
  <details><summary>What shipped</summary>

  - Price-to-IV bootstrapping; vega/price-weighted losses; bounds/regularization; stable seeds; returns `SVIFit` with diagnostics.
  </details>

- [x] vol/no_arb.py — butterfly & calendar checks (per-expiry)  
  Owner: Drew · Tags: 📈  
  <details><summary>What shipped</summary>

  - Discrete convexity screen; simple calendar monotonicity spot checks with reasons.
  </details>

- [x] vol/surface.py — per-expiry SVI fit collection (basic)  
  Owner: Drew · Tags: 📈  
  <details><summary>What shipped</summary>

  - Fit SVI per expiry; simple API to query IV via nearest/linear blend.  
  - Next: smooth params across T (see To Do).
  </details>

### density/ 📐

- [x] density/bl.py — BL finite differences (stabilized)  
  Owner: Drew · Tags: 📐  
  <details><summary>What shipped</summary>

  - Central differences on smoothed price curve; clipping + optional renormalization; integral/negativity diagnostics.
  </details>

- [x] density/cdf.py — CDF, moments & helpers  
  Owner: Drew · Tags: 📐  
  <details><summary>What shipped</summary>

  - `build_cdf`, `moments_from_pdf`, `interpolate_pdf`, `grid_from_calls` utilities.
  </details>

### viz/ 📊

- [x] viz/plots.py — smiles & densities  
  Owner: Drew · Tags: 📊  
  <details><summary>What shipped</summary>

  - Market smile, SVI vs market, PDF+CDF with markers; light/dark themes.
  </details>

### cli/, notebooks, docs 📦

- [x] cli/main.py — unified CLI (Windows-safe asyncio)  
  Owner: Drew · Tags: 📦  
  <details><summary>What shipped</summary>

  - Single-event-loop runner; artifacts: `chain.csv`, `results.json`, `smile_*.png`, `density_pdf_cdf.png`.
  </details>

- [x] notebooks/data_test.ipynb — data validation  
  Owner: Drew · Tags: 📦

- [x] notebooks/suite_test.ipynb — data → preprocess → vol → density  
  Owner: Drew · Tags: 📦  

- [x] README — updated (OKX+yfinance, CLI, Results/Theory)  
  Owner: Drew · Tags: 📦  

### logging 🪵

- [x] utils/logging.py — project-wide structured logging  
  Owner: Drew · Tags: 🪵  
  <details><summary>What shipped</summary>

  - `setup_logging`, `get_logger`, request-level `set_context`, timing decorator `log_timing`, and `span` context.  
  - JSON or console output via env flags: `OVIZ_LOG_JSON`, `OVIZ_LOG_LEVEL`.
  </details>

- [x] **Instrumentation across modules**  
  Owner: Drew · Tags: 🪵  
  <details><summary>What shipped</summary>

  - Data (OKX/yfinance/cache/risk-free), Preprocess (mids/PCP/forward), Vol (SVI/no-arb/surface), Density (BL/CDF), Viz, and CLI now emit concise, contextual logs (counts, params, diagnostics, artifact paths).
  </details>

---

## New card template (copy below)

- [ ] path/to/file.py — short task title  
  Owner: ___ · Tags: (choose from 🧱 🔑 📈 📐 🧮 📊 🧪 📦 🪵)  
  <details><summary>Spec & DoD</summary>

  - What to build:  
  - API:  
  - Edge cases:  
  - DoD:
  </details>
