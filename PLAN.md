# FinMod — Development Plan

> Agreed: 2026-11-22. Work through the phases in order; each phase ends with a measurable "done when".
> Background and sources: [RESEARCH.md](RESEARCH.md). Known code issues: [FINDINGS.md](FINDINGS.md).

## Decisions already made

| # | Decision |
|---|---|
| D1 | **The final buy/sell/hold call comes from one trained tabular model** (the decision model) that takes all signals as input and outputs a calibrated probability. LLMs never make the final call. |
| D2 | **News is read by hosted decision models** (OpenRouter Decisions API: Jev / pplx-decider), which return probabilities for typed questions (sentiment, event flags). The chat-LLM extraction is the fallback. |
| D3 | **Target device: original Jetson Nano 4 GB**, in a Python 3.12 Docker container on CPU. No local LLM; remote calls go through OpenRouter. Training runs on the PC, and the Nano only runs inference. |
| D4 | **Measure before improving.** Every model change (technical features included) is judged with the same walk-forward evaluation harness, against fixed baselines, after costs. |
| D5 | **Go/no-go bar is set in advance** (Phase A). If a model doesn't clear it, we change features or the label, never the bar. |

Realistic goal: a **small but stable positive expected value after costs**, plus the discipline to not trade when there's none. For large-cap 3–10 day direction, professional hit rates are ~51–56%.

---

## ✅ Step 1 — DecisionInput schema + daily logger (done)
`utils/data_models.py` (DecisionInput v1), `scripts/feature_builder.py`, `utils/decision_store.py`, `scripts/log_decision_inputs.py`, OpenRouter provider, tests.

---

## Phase A — Evaluation harness (measure the current system honestly)

**Why:** the reported 58.1% is measured only on big-move days (the ATR filter removes quiet days before the test split), comes from one test window, and its "+0.9 ATR per trade" doesn't match the label. Without a trustworthy measurement, nothing after this can be judged.

- [ ] `evaluation/walk_forward.py`: rolling/expanding walk-forward splits by date, with **purge + embargo** (≥ label horizon) between train and test.
- [ ] `evaluation/labels.py`: **triple-barrier labels** matching the risk manager: +3×ATR target, −2×ATR stop, max holding N days (start N=10). Outputs: which barrier was hit first, the return, holding days. Also keep the old 3-day ATR label for comparison.
- [ ] `evaluation/backtest.py`: turns probabilities into trades (threshold τ, long/short), applies **costs + slippage** (default 5 bps per side) and reports per fold: hit rate, coverage, EV in R (1R = 2×ATR stop), Sharpe-like ratio, max drawdown, Brier score, calibration curve.
- [ ] Baselines, run through the same harness: **always long**, **random with the same coverage**, **simple trend rule** (Close > MA50).
- [ ] Re-evaluate the **current per-ticker ensemble** honestly (all days, triple-barrier, walk-forward). Replace the README's 58.1% table with the real numbers.
- [ ] Write the **go/no-go bar** into this file, e.g. EV > 0 after costs in ≥ 70% of folds, beats every baseline on EV at matched coverage, calibration error < 0.05.

**Done when:** one command (`uv run python -m evaluation.report --model current`) produces a report for any model, and the current system has an honest scorecard.

## Phase B — Technical analysis upgrade

**Why:** the current technicals are standard but have issues: scale-dependent raw values (`Volume`, `ATR`, `MACD`, `Gap`) that drift as the price grows 100×, single timeframe, no relative strength, no volatility regime. Each change below is kept only if Phase A's harness shows it helps.

### B1 — Test classic TA strategies as stand-alone rules
- [ ] Run classic rules through the harness as baselines and to learn what works on this universe: MA crossover (20/50, 50/200), Donchian 20/55 breakout, RSI(2) mean reversion in an uptrend (Close > MA200), MACD signal cross, Bollinger squeeze breakout, 52-week-high momentum.
- [ ] Report per rule: EV after costs, hit rate, coverage, by regime (bull/bear by SPY vs MA200; VIX high/low).
- [ ] Rules that show an edge become **binary features** for the model (`rule_donchian20_long`, …).

### B2 — Rebuild the feature set (`features/technical.py`, pure pandas, Nano-friendly)
- [ ] **Normalise everything:** prices as % of Close or in ATR units, volume as z-score / ratio to its own history, MACD as % of price. Nothing that lets the model recognise a ticker by its price level.
- [ ] **Multi-timeframe trend:** distance to MA20/50/200 in ATR units, MA slopes (5-day change of MA20/MA50), MA alignment (20>50>200).
- [ ] **Momentum:** returns over 1/5/20/60/120 days, 12-1 month momentum, RSI(2) and RSI(14), distance from 20/55/252-day high/low.
- [ ] **Volatility regime:** realised vol 10/20/60, vol ratio (10/60), ATR percentile over 1 year, Bollinger width percentile (squeeze), overnight gap vs ATR.
- [ ] **Volume / flow:** volume z-score, dollar-volume z-score, OBV slope, up-volume ratio, close location value ((C−L)/(H−L)).
- [ ] **Relative strength:** stock return − SPY return (5/20/60 days), stock vs its sector ETF (XLK, XLY, XLP, …), rolling beta and residual (idiosyncratic) volatility.
- [ ] **Market context:** SPY trend + vol, VIX level/change, VIX term structure (VIX/VIX3M) if available, market breadth proxy (share of universe above MA50).
- [ ] Remove redundant features (correlation > 0.95), keeping the one that is easier to interpret.

### B3 — Measure feature value
- [ ] Ablation by group (trend / momentum / volatility / volume / relative / market) on walk-forward folds.
- [ ] Permutation importance and SHAP on **test folds only** (importance on training data is misleading).
- [ ] Check stability: a feature that helps only in one year is noise.

**Done when:** a documented feature set (`features/technical.py`, unit-tested) beats the old 23 features in the harness, with a group ablation table in `results/`.

## Phase C — Multi-stock training dataset

**Why:** ~2k rows per ticker is too little. One pooled model on 200–500 liquid stocks gets 50–100× more data.

- [ ] Universe: S&P 500 (or the top 300 by dollar volume) with membership **as of each date** (avoid survivorship bias; at minimum, document the bias if using today's list).
- [ ] Data pipeline: daily OHLCV (yfinance/Polygon), cached to Parquet in `data/` (gitignored), incremental updates.
- [ ] Polygon news `insights` aggregated per ticker/day from 2024-07-02 on (16:00 ET cutoff, already in `feature_builder.py`).
- [ ] Earnings dates per ticker (point-in-time).
- [ ] `build_dataset.py`: one row per (ticker, day) with Phase B features, news aggregates, events and triple-barrier labels → `data/dataset_v1.parquet`.
- [ ] Keep the watchlist tickers in the universe so per-ticker results can be reported.

**Done when:** the dataset builds reproducibly with one command, and row counts, date coverage and label distribution are reported.

## Phase D — The decision model

- [ ] Models: **logistic regression** (baseline), **LightGBM** (main), **TabPFN** (challenger; check its licence and runtime). Same walk-forward folds.
- [ ] **Calibration** (isotonic/Platt fit on an inner validation fold), shown as a reliability plot.
- [ ] **Threshold τ chosen by EV after costs** on validation folds, never by eye.
- [ ] Inputs tested by ablation: (a) technicals only, (b) + news, (c) + events, (d) + current ML ensemble / GRU outputs (out-of-fold only). Keep a block only if it helps; (d) probably adds little because it's built from the same price data.
- [ ] Optional: a separate model per direction (long-signal vs short-signal), if short trades behave differently.
- [ ] `DecisionOutput` with top SHAP drivers, saved with `feature_hash` and model version.
- [ ] Apply the **go/no-go bar** from Phase A.

**Done when:** the decision model clears the bar (→ Phase F), or we have a documented "no edge yet" result and go back to B/C/E.

## Phase E — News via Decisions API (start early, runs alongside A–D)

- [ ] Switch `news.llm` in DecisionInput to probabilities (`p_short_bullish`, `p_short_bearish`, `p_ev_earnings`, …). No live data has been logged yet, so changing v1 costs nothing.
- [ ] `run_llm_extraction` → OpenRouter Decisions API (`choice` for sentiment, `noul` per event), with the chat-LLM path as fallback. Model is configurable (Jev / pplx-decider).
- [ ] Optional: log a direct "p_up" decision-model question as a candidate feature (live only; never used for decisions until it has a track record).
- [ ] Start the **daily logger** (cron 16:30 ET) as soon as this lands, because live news features can't be back-filled without hindsight.
- [ ] After 3–6 months of live rows: test in Phase D whether the news block adds value.

**Done when:** the daily logger runs unattended and stores Decisions-API news probabilities.

## Phase F — Integrate into the app

- [ ] `ai_market_agent.py` / `app.py`: replace the LLM confluence step with `DecisionModel.predict()`. The LLM only writes the explanation from the SHAP drivers.
- [ ] Fix the risk manager for SELL direction (FINDINGS §2.1) and the yfinance D/E percent bug in `fetch_current_fundamentals`.
- [ ] Fix `predict_next_day()` to drop the incomplete intraday bar.
- [ ] `SignalLogger` logs decision-model outputs and fills outcomes using triple-barrier rules.

## Phase G — Jetson Nano deployment

- [ ] Dependency extras: `bot` (runtime), `train`, `ui`. The bot image excludes browser-use, streamlit, torch and the training stack.
- [ ] GRU → ONNX (only if Phase D keeps it).
- [ ] `Dockerfile` (python:3.12-slim, arm64) + `docker-compose.yml` (restart policy, volumes for `models/`, `data/`, `results/`, TZ=America/New_York).
- [ ] Nano setup notes: update Docker if newer base images fail; 4–8 GB swap on SSD; `nvpmodel`/`jetson_clocks`.
- [ ] Weekly retrain on the PC → `rsync` models to the Nano; the model refuses to load on a `feature_hash` mismatch.

## Phase H — Telegram bot

- [ ] `bot/` with python-telegram-bot v22 (polling, JobQueue).
- [ ] Commands: `/signal TICKER`, `/watchlist`, `/report`, `/why TICKER`, `/status`, `/add`, `/remove`.
- [ ] Daily job after the close: build inputs → predict → push only signals above τ.
- [ ] `ALLOWED_CHAT_IDS` allowlist, token in `.env`, heartbeat message, "not financial advice" in `/start`.

---

## Progress log

| Date | Step | Result |
|---|---|---|
| 2026-11-21 | Research | RESEARCH.md |
| 2026-11-21 | Step 1 | DecisionInput v1, feature builder, daily logger, OpenRouter |
| 2026-11-22 | Plan | This file |
