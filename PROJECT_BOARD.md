# Option Viz — Project Board

## Legend

Status: 🟦 To Do · 🟨 In Progress · 🟩 Done  
Tags: 🧱 Infra · 🔑 Auth · 📈 Modeling · 📐 Density · 🧮 Preprocess · 📊 Viz · 🧪 Tests · 📦 CLI/Docs

Tip: Assign an owner by replacing `Owner: ___`. Move items between the three status sections below.

---

## 🟦 To Do

### data/ (backends, plumbing) 🧱

*(Core data layer is shipped; new data tasks can be added here as needed.)*

- [ ] data/historical_loader.py — add Parquet dependency docs & CLI example  
  Owner: ___ · Tags: 🧱  
  <details><summary>Spec & DoD</summary>
  
  - Document `pyarrow`/`fastparquet` install; provide a short CLI example to save/load.  
  - DoD: README snippet runs successfully after installing optional deps.
  </details>

### preprocess/ (cleaning, transforms) 🧮

*(Core preprocessing helpers are shipped; add enhancements or edge-case handling here.)*

- [ ] preprocess/midprice.py — enhancements  
  Owner: ___ · Tags: 🧮  
  <details><summary>Spec & DoD</summary>
  
  - Add configurable outlier rejection and timestamp staleness checks.  
  - DoD: unit tests for outlier clipping & staleness pass.
  </details>

- [ ] preprocess/pcp.py — diagnostics extras  
  Owner: ___ · Tags: 🧮  
  <details><summary>Spec & DoD</summary>
  
  - Add residual histogram helper & threshold alert callback.  
  - DoD: function returns bins/alerts; plotted in notebook.
  </details>

- [ ] preprocess/forward.py — robustness  
  Owner: ___ · Tags: 🧮  
  <details><summary>Spec & DoD</summary>
  
  - Add fallback to put–call average forward if pairs are sparse; warn on instability.  
  - DoD: unit test with sparse pairs passes; warning surfaced.
  </details>

### vol/ (smiles, surfaces) 📈

- [ ] vol/svi.py — SVI calibration  
  Owner: ___ · Tags: 📈  
  <details><summary>Spec & DoD</summary>
  
  - `w(k)=a+b(ρ(k−m)+sqrt((k−m)^2+σ^2))`; grid seeds + L-BFGS; bounds & mild reg.  
  - DoD: returns finite params; loss < ATM-only baseline on synthetic data.
  </details>

- [ ] vol/no_arb.py — arbitrage checks  
  Owner: ___ · Tags: 📈  
  <details><summary>Spec & DoD</summary>
  
  - Butterfly positivity screens; calendar monotonicity spot checks.  
  - DoD: violations <1% of strikes; flagged with reasons.
  </details>

- [ ] vol/surface.py — across maturities  
  Owner: ___ · Tags: 📈  
  <details><summary>Spec & DoD</summary>
  
  - Fit per-expiry SVI, smooth params over T (spline or low-order poly).  
  - DoD: continuous Σ(k,T); basic calendar constraints pass on sample.
  </details>

### density/ (RND) 📐

- [ ] density/bl.py — BL finite differences  
  Owner: ___ · Tags: 📐  
  <details><summary>Spec & DoD</summary>
  
  - Central / higher-order differences; adaptive spacing near kinks.  
  - DoD: pdf ≥0 on ≥98% grid; ∫pdf=1±0.01.
  </details>

- [ ] density/cdf.py — CDF & quantiles  
  Owner: ___ · Tags: 📐  
  <details><summary>Spec & DoD</summary>
  
  - Integrate pdf; inverse-CDF via monotone spline; VaR stats.  
  - DoD: median/quantiles consistent with forward/variance.
  </details>

### viz/ (plots, dashboard) 📊

- [ ] viz/plots.py — smiles & densities  
  Owner: ___ · Tags: 📊  
  <details><summary>Spec & DoD</summary>
  
  - IV vs log-moneyness; PDF/CDF with mean/median overlays; light/dark theme.  
  - DoD: PNGs saved; functions return `matplotlib.Figure`.
  </details>

- [ ] viz/dashboard.py — Streamlit demo  
  Owner: ___ · Tags: 📊  
  <details><summary>Spec & DoD</summary>
  
  - Select backend (equity/crypto), ticker, expiry; export plots.  
  - DoD: one-file app runs locally and updates interactively.
  </details>

### cli/, tests/, docs/ 📦

- [ ] cli/main.py — minimal CLI  
  Owner: ___ · Tags: 📦  
  <details><summary>Spec & DoD</summary>
  
  - `oviz fetch --asset-class equity --underlying AAPL --expiry 2025-12-19`  
    `oviz fetch --asset-class crypto --underlying BTC --expiry YYMMDD`  
  - DoD: writes normalized JSON/CSV + PNGs; exit codes on failure.
  </details>

- [ ] tests/ — unit & async tests 🧪  
  Owner: ___ · Tags: 🧪  
  <details><summary>Spec & DoD</summary>
  
  - Remaining: yfinance/OKX network smoke tests, failure-path tests, and coverage consolidation.  
  - DoD: coverage ≥80%; network tests opt-in via env flag.
  </details>

---

## 🟨 In Progress

- (Move items here as you pick them up; keep the same card format.)

---

## 🟩 Done

### data/ 🧱

- [x] data/base.py — shared types and interface  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>
  
  - `OptionQuote`, `OptionChain` dataclasses; `OptionFetcher` protocol (`list_expiries`, `fetch_chain`).  
  - Typed throughout; designed to be backend-agnostic (equity/crypto).
  </details>

- [x] data/registry.py — backend factory  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>
  
  - `get_fetcher("equity"|"crypto")` → `YFinanceFetcher` / `OKXFetcher`; kwargs forwarded.  
  - Smoke test returns ≥1 quote for AAPL and BTC.
  </details>

- [x] data/cache.py — in-memory+disk caching  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>
  
  - `KVCache` with TTL; fast in-memory + SQLite persistence; `get_cached(key, fetch_fn, ttl)`.  
  - `vacuum_disk()` and memory promotion on disk hit.
  </details>

- [x] data/historical_loader.py — normalized offline chains  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>
  
  - `save_chain_csv/parquet`, `load_chain_csv/parquet`; `chain_to_dataframe`, `dataframe_to_chain`.  
  - Batch helpers; transparent timezone handling for expiries/as-of.
  </details>

- [x] data/risk_free.py — SOFR loader + fallback  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>
  
  - `RiskFreeProvider(get_rate)` with CSV support, forward-fill option, and constant fallback.  
  - Graceful behavior outside CSV range.
  </details>

- [x] data/rate_limit.py — polite throttling utilities  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>
  
  - `AsyncRateLimiter(max_concurrent)` and `retry_with_backoff(...)` with jitter.  
  - Integrated into OKX ticker fetch loop; patterns documented.
  </details>

- [x] data/yf_fetcher.py — equities via yfinance (sync → `asyncio.to_thread`)  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>
  
  - Fixed `.option_chain()` unpack bug; robust spot retrieval; IV forwarded via `extra["iv"]`.
  </details>

- [x] data/okx_fetcher.py — crypto via OKX (public endpoints)  
  Owner: Drew · Tags: 🧱  
  <details><summary>What shipped</summary>
  
  - YYMMDD/ YYYYMMDD expiry parsing; server-time sync; **integrated `AsyncRateLimiter` + `retry_with_backoff`** for ticker calls; **instrument list cached** via `KVCache` (1h TTL).  
  - Public instruments/tickers/index spot; helpful error body on failures.
  </details>

### preprocess/ 🧮

- [x] preprocess/midprice.py — robust mids + flags  
  Owner: Drew · Tags: 🧮  
  <details><summary>What shipped</summary>
  
  - Vectorized mids with fallback (bid-only/ask-only), crossed/wide flags, relative spread.  
  - Side-used label for auditing; NaN-safe; configurable thresholds.
  </details>

- [x] preprocess/pcp.py — put–call parity diagnostics  
  Owner: Drew · Tags: 🧮  
  <details><summary>What shipped</summary>
  
  - Synthetic legs from PCP; residual computation; strike-pivot helper.  
  - Clean API for plugging into notebooks and tests.
  </details>

- [x] preprocess/forward.py — forward & log-moneyness  
  Owner: Drew · Tags: 🧮  
  <details><summary>What shipped</summary>
  
  - Forward formula, log-moneyness, and a robust forward estimator from PCP pairs with optional spread-based weights.
  </details>

### tests & notebooks 📦

- [x] tests/preprocess_* — unit tests for preprocess  
  Owner: Drew · Tags: 🧪  
  <details><summary>What shipped</summary>
  
  - `test_preprocess_midprice.py`, `test_preprocess_pcp.py`, `test_preprocess_forward.py` covering mids, PCP identities, and forward estimation.
  </details>

- [x] notebooks/data_test.ipynb — end-to-end data validation  
  Owner: Drew · Tags: 📦  
  <details><summary>What shipped</summary>
  
  - PYTHONPATH setup; equity + crypto chain tests; cache/risk-free/limiter demos; CSV round-trip.
  </details>

- [x] notebooks/full_pipeline_test.ipynb — data → preprocess pipeline  
  Owner: Drew · Tags: 📦  
  <details><summary>What shipped</summary>
  
  - Fetch chain (equity/crypto), compute mids & PCP diagnostics, estimate forward, log-moneyness, quick visuals, save CSV/PNGs.
  </details>

- [x] README — updated with Results & Theory Deep Dive  
  Owner: Drew · Tags: 📦  
  <details><summary>What shipped</summary>
  
  - Pitch-style overview, artifacts/metrics in Results, and a deeper theory section (RN measure, SVI, BL, COS).
  </details>

---

## New card template (copy below)

- [ ] path/to/file.py — short task title  
  Owner: ___ · Tags: (choose from 🧱 🔑 📈 📐 🧮 📊 🧪 📦)  
  <details><summary>Spec & DoD</summary>
  
  - What to build:  
  - API:  
  - Edge cases:  
  - DoD:
  </details>
