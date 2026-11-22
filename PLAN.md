# FinMod — Development Plan

> Agreed: 2026-11-22. Work through the phases in order; each phase ends with a measurable "done when".
> Background and sources: [RESEARCH.md](RESEARCH.md). Known code issues: [FINDINGS.md](FINDINGS.md).
> Phases A–D need a PC (or a cloud notebook) for data downloads and training. Design work can happen without one.

## Decisions already made

| # | Decision |
|---|---|
| D1 | **One job per layer.** Each layer answers one question that no other layer answers (table below). A tool is added only if we can name the question it answers. |
| D2 | **The final call comes from one trained decision model** that takes the layer outputs as input and gives a calibrated probability. LLMs never make the final call. The separate ML ensemble is retired once the decision model beats it, because it is a model on the same technical features. |
| D3 | **News is read by hosted decision models** (OpenRouter Decisions API: Jev / pplx-decider) returning probabilities. The chat-LLM extraction is the fallback. |
| D4 | **Technical analysis = 7 indicators, one per market dimension.** No overlapping indicators. |
| D5 | **Agreement is an input, not a gate.** Disagreement lowers the probability; it doesn't force "hold". Actions are graded, and there are only 2–3 named hard vetoes. |
| D6 | **Measure before improving.** Every change is judged with the same walk-forward harness, against fixed baselines, after costs. The go/no-go bar is set before results are seen. |
| D7 | **Learn from real moves and from our own mistakes.** Backtests produce a move study (how indicators behave in real up/down moves) and a signal journal (every right and wrong call, with its full context). |
| D8 | **Target device: original Jetson Nano 4 GB.** Python 3.12 Docker on CPU, no local LLM (OpenRouter instead), training on the PC, inference only on the Nano. |

Realistic goal: a **small but stable positive expected value after costs**, trading often enough to matter, and staying out when there's no edge. For large-cap 3–10 day direction, professional hit rates are ~51–56%.

## System layers

```
LAYER          QUESTION IT ANSWERS                          INPUT           OUTPUT
───────────────────────────────────────────────────────────────────────────────────────────
1 News         "What does the news say about this stock?"   text only       up / down / neutral + probability
2 Technical    "What is the market situation right now?"    daily prices    7 state labels + values
3 Sequential   "Where is the price path heading next?"      price sequence  up / down + expected move (GRU)
4 Decision     "Given 1–3, do we act, and how strongly?"    outputs of 1–3  p(up), graded action, reasons
5 Risk         "How much, and where are stop/target?"       price, ATR, p   shares, stop, target
```

Caution: layers 2 and 3 both read prices, so their agreement is **correlated evidence**, not two independent votes. The decision model learns from outcomes how much each layer deserves. News is the only independent source.

### Layer 2: the 7 indicators

| # | Dimension | Indicator | State label |
|---|---|---|---|
| 1 | Long-term regime | 200-day MA: price above/below, slope | `regime`: bull / bear |
| 2 | Medium trend | 20 / 50 EMA alignment | `trend`: up / down / sideways |
| 3 | Trend strength | ADX(14) | `strength`: strong / weak |
| 4 | Momentum / exhaustion | RSI(14) | `momentum`: rising / fading / stretched |
| 5 | Volatility | ATR(14) as a 1-year percentile | `volatility`: low / normal / high |
| 6 | Participation | Volume vs 20-day average (or OBV slope) | `volume`: confirming / not |
| 7 | Relative strength | 20-day return minus SPY's | `relative`: leader / laggard |

Left out because they overlap one of the seven: MACD (trend + momentum), Stochastic / Williams %R / CCI (same as RSI), Bollinger (volatility + stretch), Ichimoku, Parabolic SAR. B1 may test MACD or Bollinger once; a winner **replaces** its group's pick, never sits beside it.

### Graded actions and vetoes (Layer 4)

| p(up) | Action | Size |
|---|---|---|
| ≥ τ_strong (e.g. 0.62) | strong buy | full |
| τ_buy … τ_strong (e.g. 0.56–0.62) | buy | half |
| in between | hold | — |
| mirror image below 0.5 | sell / strong sell | half / full |

- Thresholds are fitted on validation data to maximise EV after costs, **subject to a minimum activity target** (e.g. 15–30% of ticker-days get a non-hold action). If no threshold satisfies both, that is reported as "no edge", not hidden.
- **Hard vetoes, max 3, each named:** e.g. `EARNINGS_2D` (earnings within 2 days), `VIX_EXTREME` (VIX > 35), `STOP_TOO_WIDE` (stop > X% of price).
- **Every hold stores its reason** (`LOW_PROB`, or one of the veto names).
- **Hold-rate alarm:** if more than ~85% of signals over 2 weeks are holds, report the breakdown by reason.

---

## ✅ Step 1 — DecisionInput schema + daily logger (done)
`utils/data_models.py` (DecisionInput v1), `scripts/feature_builder.py`, `utils/decision_store.py`, `scripts/log_decision_inputs.py`, OpenRouter provider, tests.

---

## Phase A — Evaluation harness (measure the current system honestly)

**Why:** the reported 58.1% is measured only on big-move days (the ATR filter drops quiet days before the test split), comes from one test window, and its "+0.9 ATR per trade" doesn't match the label.

- [ ] `evaluation/walk_forward.py`: walk-forward splits by date with **purge + embargo** (≥ label horizon) between train and test.
- [ ] `evaluation/labels.py`: **triple-barrier labels** matching the risk manager (+3×ATR target, −2×ATR stop, max N days, start N=10): which barrier was hit first, return, days held. Keep the old 3-day ATR label for comparison.
- [ ] `evaluation/backtest.py`: probabilities → graded actions → trades with **costs + slippage** (default 5 bps per side). Per-fold metrics: hit rate, coverage, hold rate, EV in R (1R = 2×ATR), drawdown, Brier score, calibration curve.
- [ ] Baselines through the same harness: **always long**, **random at the same coverage**, **trend rule** (Close > MA50).
- [ ] Honest scorecard for the **current** per-ticker ensemble. Replace the README's 58.1% table.
- [ ] Write the **go/no-go bar** here before any new model is tested, e.g. EV > 0 after costs in ≥ 70% of folds, beats every baseline at matched coverage, calibration error < 0.05, hold rate within the activity target.

**Done when:** one command produces a report for any model, and the current system has an honest scorecard.

## Phase B — Technical analysis (Layer 2)

### B1 — Implement the 7 indicators + state labels
- [ ] `features/technical.py`, pure pandas (Nano-friendly), unit-tested: the 7 indicators, all normalised (ATR units / percentiles / ratios; nothing that reveals a ticker by its price level).
- [ ] `market_state()` → the 7 labels with documented cut-offs, e.g. ADX > 25 = strong, ATR percentile > 80 = high volatility. Cut-offs are checked against the move study (B2), not just taken from textbooks.
- [ ] One-off check: MACD vs the EMA alignment and Bollinger vs the ATR percentile. A challenger only replaces the incumbent if it wins in the harness.

### B2 — Move study: how indicators behave in *real* moves
**Goal:** learn what the 7 indicators look like before, during and at the end of real up and down moves, ignoring small day-to-day changes.

- [ ] **Define a real move** with a ZigZag / swing detector in ATR units: a swing counts when price travels **≥ k×ATR** (start k = 3) from a swing low to a swing high (up move) or the reverse (down move), over ≥ 3 days. Everything smaller is "chop". Thresholds are in ATR so they adapt to each stock's volatility.
- [ ] **Event table** (`results/moves/moves.parquet`): ticker, direction, start/end date, size in ATR and %, duration, max drawdown inside the move.
- [ ] **Indicator profiles around each move:** the 7 indicators and state labels at days −10 … +10 around the move **start**, at the move's **midpoint**, and around its **end**. Compare against the same values on ordinary days (the baseline).
- [ ] **Control group: false starts.** Moves that began the same way but reversed before reaching k×ATR. At decision time you can't know which kind you are in, so the useful question is: *which indicator states separate real starts from false starts?* Without this control the study is hindsight.
- [ ] Questions to answer, per direction and per regime:
  - Which states are common at the start of real up moves and rare at false starts?
  - Which states typically mark the **end** (exhaustion, e.g. RSI stretched + volume fading)?
  - Do down moves look like mirrored up moves? (Often not: down moves tend to be faster and more volatile.)
  - How early do the signals appear, i.e. how many days and ATR of the move are already gone when the state flips?
- [ ] Output: a short report (`results/moves/report.md` + charts) and a list of **candidate features** (e.g. "ADX rising from < 20", "volume confirming on the breakout day"). They go into the model only if B3 shows they help.

### B3 — Measure what each indicator is worth
- [ ] Ablation: drop one indicator at a time on walk-forward folds.
- [ ] Permutation importance and SHAP on **test folds only**.
- [ ] Stability check: an indicator that only helps in one year is noise.

**Done when:** the 7-indicator set is documented, tested, beats the old 23 features in the harness, and the move study report exists.

## Phase C — Multi-stock training dataset

**Why:** ~2k rows per ticker is too little. One pooled model on 200–500 liquid stocks gets 50–100× more data.

- [ ] Universe: S&P 500 (or top 300 by dollar volume). Use membership **as of each date** if available; otherwise document the survivorship bias.
- [ ] Daily OHLCV cached to Parquet in `data/` (gitignored), incremental updates.
- [ ] Polygon news `insights` per ticker/day from 2024-07-02 (16:00 ET cutoff, already in `feature_builder.py`).
- [ ] Point-in-time earnings dates.
- [ ] `build_dataset.py`: one row per (ticker, day): Layer 2 indicators + states, news aggregates, events, GRU output (out-of-fold), triple-barrier labels, and the B2 move tag (which move / phase of move the day sits in, **for analysis only, never a feature**, because it uses the future).

**Done when:** the dataset builds reproducibly with one command; row counts, date coverage and label balance are reported.

## Phase D — The decision model (Layer 4)

- [ ] Models: **logistic regression** (baseline), **LightGBM** (main), **TabPFN** (challenger; check licence and runtime). Same folds.
- [ ] Calibration (isotonic/Platt on an inner validation fold) with a reliability plot.
- [ ] Graded-action thresholds fitted under the activity target (see Layer 4 above).
- [ ] Ablation by layer: (a) technicals only, (b) + news, (c) + GRU, (d) + old ML ensemble outputs. Keep a layer only if it helps. If (d) adds nothing, retire the ensemble (D2).
- [ ] `DecisionOutput` with action, size tier, hold reason, top SHAP drivers, `feature_hash`, model version.
- [ ] Apply the go/no-go bar from Phase A.

**Done when:** the model clears the bar (→ Phase F), or there is a documented "no edge yet" result that sends us back to B/C/E, guided by the signal journal.

## Phase J — Signal journal & error analysis (runs with every backtest from Phase A on)

**Goal:** know *when* the system is right and wrong, so the next experiment targets a real weakness instead of a guess.

- [ ] **Journal row per signal** (`results/journal/<run_id>.parquet`): date, ticker, each layer's output (news label + p, 7 state labels, GRU direction), decision p, action, size, hold reason, outcome (barrier hit, R, days held), right/wrong, and the B2 move context: inside a real up/down move or chop, and how far into it (early / middle / late).
- [ ] Standard error report per run:
  - Hit rate and EV **by market state** (e.g. "bull + strong trend" vs "bear + high volatility").
  - Hit rate **by layer agreement pattern** (all agree / news disagrees / GRU disagrees, …).
  - **Timing:** share of trades entered early / mid / late in a move. Late entries are usually where losses come from.
  - **Missed moves:** real moves (from B2) where we held, broken down by hold reason. This is the direct check on the "always hold" problem.
  - **Worst 20 losses and best 20 wins**, with full context, for manual review.
- [ ] **Experiment log** (`EXPERIMENTS.md`): one row per experiment: hypothesis (from the journal), change, run id, result, keep / drop. Experiments must come from journal findings, e.g. "losses cluster in high-volatility chop → test a volatility veto".

**Done when:** every backtest run automatically produces its journal and error report, and the experiment log has its first entries.

## Phase E — News via Decisions API (Layer 1; start early, runs alongside A–D)

- [ ] Switch `news.llm` in DecisionInput to probabilities (`p_short_bullish`, `p_short_bearish`, `p_ev_earnings`, …). No live data logged yet, so changing v1 is free.
- [ ] `run_llm_extraction` → OpenRouter Decisions API (`choice` for sentiment, `noul` per event), with the chat-LLM path as fallback. Model configurable (Jev / pplx-decider).
- [ ] Output a clear news label (up / down / neutral) + probability for the journal and the bot.
- [ ] Start the **daily logger** (cron 16:30 ET) as soon as this lands; live news features can't be back-filled without hindsight. (Can run on the Nano itself, which needs no training.)
- [ ] After 3–6 months of live rows: test in Phase D whether news adds value.

**Done when:** the daily logger runs unattended and stores Decisions-API news probabilities.

## Phase F — Integrate into the app

- [ ] `ai_market_agent.py` / `app.py`: replace the LLM confluence rules with `DecisionModel.predict()`. The LLM only writes the explanation from the SHAP drivers + market-state labels.
- [ ] Graded actions, named vetoes, hold reasons, hold-rate alarm.
- [ ] Fix: risk manager SELL direction (FINDINGS §2.1); yfinance D/E percent bug in `fetch_current_fundamentals` (it makes almost every stock "HIGH_RISK"); `predict_next_day()` incomplete intraday bar.
- [ ] `SignalLogger` writes the same journal format as Phase J, so live results can be compared with backtests.

## Phase G — Jetson Nano deployment

- [ ] Dependency extras: `bot` (runtime), `train`, `ui`. The bot image excludes browser-use, streamlit, torch and the training stack.
- [ ] GRU → ONNX (only if Phase D keeps it).
- [ ] `Dockerfile` (python:3.12-slim, arm64) + `docker-compose.yml` (restart policy; volumes for `models/`, `data/`, `results/`; TZ=America/New_York).
- [ ] Nano setup notes: update Docker if newer images fail; 4–8 GB swap on SSD; `nvpmodel` / `jetson_clocks`.
- [ ] Weekly retrain on the PC → `rsync` to the Nano; the model refuses to load on a `feature_hash` mismatch.

## Phase H — Telegram bot

- [ ] `bot/` with python-telegram-bot v22 (polling, JobQueue).
- [ ] `/signal TICKER` reply shows each layer's answer: news label, market-state line (e.g. "bull · uptrend · strong · momentum fading · high vol · volume weak · leader"), GRU direction, then the decision, size and stop/target.
- [ ] Other commands: `/watchlist`, `/report` (live journal stats + hold reasons), `/why TICKER`, `/status`, `/add`, `/remove`.
- [ ] Daily job after the close pushes only non-hold actions, plus the hold-rate alarm.
- [ ] `ALLOWED_CHAT_IDS` allowlist, token in `.env`, heartbeat, "not financial advice" in `/start`.

---

## Order of work

```
A (harness) ──► B1 (7 indicators) ──► B2 (move study) ──► B3 ──► C (dataset) ──► D (decision model) ──► F ──► G ──► H
     └──────────── J (journal) is built with A and reused by every later backtest ────────────┘
E (news + daily logger) runs in parallel from the start
```

## Progress log

| Date | Step | Result |
|---|---|---|
| 2026-11-21 | Research | RESEARCH.md |
| 2026-11-21 | Step 1 | DecisionInput v1, feature builder, daily logger, OpenRouter |
| 2026-11-22 | Plan | This file: layers, 7 indicators, graded actions, move study, signal journal |
