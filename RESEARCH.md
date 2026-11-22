# FinMod — Research: Decision Model + Jetson / Telegram Deployment

> Written: 2026-11-21
> Scope: (1) a trained **decision model** that replaces the rule-based LLM confluence step and predicts UP/DOWN from structured + text data, and the **input data structure** it needs; (2) running FinMod on a **Jetson Nano** and using it through a **Telegram bot**.

---

## 0. TL;DR

| Topic | Recommendation |
|---|---|
| What the decision model is | A **stacked meta-model**: gradient-boosted trees (LightGBM or XGBoost) with a logistic-regression baseline. Its inputs are the outputs of the existing models (ML ensemble, GRU) plus news features turned into numbers, fundamentals and market regime. It predicts the same target the ML ensemble already uses: **3-day ATR-filtered direction**. |
| Role of the LLM | Change it from *decider* to *feature extractor* (text → fixed JSON fields) and *explainer* (writes the reasoning text). The trained model makes the call. |
| Input data structure | One versioned, point-in-time JSON record per `(ticker, as_of_date)`: `DecisionInput` (§3). The same record is used for training rows, live inference and the Telegram reply. |
| Largest risk | **Leakage.** Base-model predictions must be *out-of-fold* (walk-forward), news must be cut off at the market close, and LLM-scored history is contaminated because the LLM already knows how 2024–25 turned out. |
| Largest data limit | Polygon/Massive per-ticker `insights` sentiment only starts **2 July 2024**. That gives about 10 tickers × ~600 trading days ≈ 6k rows before the ATR filter. Keep the model small. |
| Jetson Nano (original, 4 GB) | Works, but only as a **CPU box inside Docker**. JetPack 4 is end-of-life and stuck on Ubuntu 18.04 / glibc 2.27. FinMod needs Python ≥ 3.12 and python-telegram-bot needs ≥ 3.11, so run a `python:3.12-slim` arm64 container. No local LLM (`gpt-oss:20b` will not fit). Keep using a cloud LLM, or skip it with the decision model. |
| Jetson **Orin** Nano (8 GB) | Much better fit: JetPack 6, Ubuntu 22.04, CUDA 12, and it can run a 1.5–3B LLM locally via llama.cpp. |
| Telegram | `python-telegram-bot` v22 using **long polling**, so no public IP or port forwarding is needed. Use a `JobQueue` daily job after the US market close, plus `/signal TICKER`, `/watchlist`, `/report` commands. Run it as a systemd service or with Docker `restart: unless-stopped`. |

**Open question for you:** which board do you have, the original **Jetson Nano (4 GB, 2019)** or the **Jetson Orin Nano / Orin Nano Super (8 GB)**? §5 covers both, but the LLM plan depends on it.

---

## 1. What the current system does (baseline)

```
StockPredictor (per ticker)   → RF/XGB/LR/GB probs, Ensemble_Prediction UP/DOWN, Ensemble_Confidence
AlphaPredictor (global GRU)   → predicted_return, position_size, direction Bullish/Bearish/Neutral
NewsAnalyzer (Polygon + LLM)  → short/long-term sentiment + confidence, key events, daily breakdown
fetch_current_fundamentals()  → PE, D/E, margins, market cap + text assessments
          │
          ▼
LLM "Confluence Agent" (prompt.py, 5 fixed rules) → MarketPrediction{action, confidence, reasoning}
```

Problems with the final step that a decision model fixes:

1. **The rules are hand-picked, not learned.** The 0.60 / 0.65 / 0.3 / 0.4 thresholds and the −0.20 fundamental penalties were never fit on data. `validate_signal.py` only tests the ML ensemble, not the final decision.
2. **The confidence is not a probability.** The LLM's `confidence` is a number it writes down, so you cannot check it against outcomes or compute an expected value from it.
3. **It is not reproducible.** The same inputs can give different outputs, and the decision depends on which model provider is used (gpt-oss vs Gemini).
4. **Cost and latency.** Each ticker needs 1 + N_days + 1 LLM calls. That is the hardest part to run on a Jetson.
5. **News is only used as a single label.** Only `short_term_sentiment` actually affects the rules. Article volume, sentiment change, disagreement between sources and earnings proximity are thrown away.

---

## 2. Decision model design

### 2.1 Target
Reuse `StockPredictor`'s label so results are comparable with the existing 58.1% @ ≥60% baseline:

- `y = 1` if `Close[t+3] − Close[t] > 0.5·ATR[t]`, `y = 0` if `< −0.5·ATR[t]`, otherwise drop the row (noise).
- Also store the raw `fwd_ret_3d` and `fwd_move_atr = (Close[t+3]-Close[t])/ATR[t]` so you can later try regression or EV-weighted training without rebuilding the dataset.

### 2.2 Model choice (small data → simple models)

| Model | Why | Notes |
|---|---|---|
| **Logistic regression** (L2, standardized) | Baseline you can read; coefficients show which inputs matter | Always train it first. If the tree model can't beat it on walk-forward, the extra complexity isn't paying off |
| **LightGBM / XGBoost** (shallow: depth 3–4, ≤300 trees, strong regularization) | Handles missing values (no news that day), interactions (sentiment × regime) and mixed scales | XGBoost is already a dependency. LightGBM is faster on CPU and the Jetson |
| Probability calibration (isotonic or Platt, fit on the last walk-forward fold) | Makes "confidence 0.62" actually mean about 62% hit rate | Needed for the risk manager and the Telegram message |
| **Abstain rule** | Trade only when `p_up ≥ τ` or `p_up ≤ 1−τ`. Choose τ by walk-forward expected value, not by eye | Replaces the hand-set STEP 1/2 thresholds |

Not recommended yet: fine-tuning a transformer end to end on text + numbers. With ~3–6k labelled rows it will overfit, and it is heavy on a Jetson.

### 2.3 Stacking without leakage (the critical part)
The decision model consumes `ml.*` and `dl.*` predictions. If those predictions were made by models that **saw that day in training**, the meta-model learns that ML is always right, and everything collapses in live trading.

Required procedure:
1. **Walk-forward the base models.** For example, retrain `StockPredictor` every 63 trading days (quarterly) on data `< t`, then predict the next 63 days. Save these **out-of-fold** predictions into the dataset. `validate_signal.py` already has walk-forward logic you can extend.
2. GRU: the checkpoint in `models/alpha_gru_v1` was trained through ~2025 on AAPL/MSFT/NVDA/AMZN/GOOGL/META. Its history predictions for those tickers are **in-sample**. Either retrain the GRU walk-forward too, or keep `dl.*` as features only for dates after the checkpoint's training cutoff.
3. Train the decision model on fold k using only rows from folds < k. Put a **3-day embargo** between train and test, because the target looks 3 days ahead.

### 2.4 Text → features (where the text goes)
Three layers, from cheapest to most expensive. Build them in order and keep a layer only if it improves walk-forward results.

| Layer | Source | Cost | History available | Contamination risk |
|---|---|---|---|---|
| **A. Polygon/Massive `insights`** | Already returned by `list_ticker_news` (`insights[].sentiment` per ticker + `sentiment_reasoning`) | Free, no model | From **2024-07-02** | Low (scored at publish time) |
| **B. FinBERT** (`ProsusAI/finbert`, ~110M params) on title + description | Local, CPU-friendly; ~50 headlines/ticker/day is seconds on a Jetson | Any date you have headlines for | Low–medium |
| **C. LLM structured extraction** (current `NewsSentiment` prompt, plus event flags) | Gemini / OpenAI / local small LLM | $$, slow | Any date | **High** for back-fill: the model already knows 2023–2025 outcomes, so back-filled labels will look too good |

Rule: **back-fill history with A + B only**. Use C only **going forward** (logged live from now on). Once C has 6+ months of live data, test whether it adds anything over A+B.

Optional layer D: sentence embeddings of headlines (e.g. `all-MiniLM-L6-v2`, 384-d) averaged per day and reduced to ~8–16 dims with PCA fit on the training fold only. Try it only after A–C, because high-dimensional inputs on ~5k rows mostly add noise.

### 2.5 Point-in-time rules for news
- US close is 16:00 America/New_York. News with `published_utc` **after 16:00 ET on day t belongs to day t+1**. The current `parse_news_by_date` groups by calendar date, which leaks after-hours news (earnings!) into the same-day row.
- Weekend and holiday news rolls forward to the next trading day.
- Use lookback windows of 1 day (`news_1d_*`) and 5 days (`news_5d_*`) so the model sees both level and change.

---

## 3. The input data structure — `DecisionInput` (v1)

One record per `(ticker, as_of_date)`, where `as_of_date` is the trading day whose **close** is the last known information. Every field must be computable from data available at 16:00 ET on that day.

Design rules:
- **Flat numeric leaves.** The tree model sees a flattened vector (`technical.RSI`, `news.d1.pos_ratio`, …). Nested JSON is only for readability and transport.
- **`null` means missing, never 0.** LightGBM/XGBoost handle NaN natively, and "no news" is different from "neutral news".
- **Enums become ordinals** (`bearish=-1, neutral=0, bullish=1`). Text is kept for logging and explanations only and never fed raw to the tree model.
- **Version the schema** (`schema_version`) and store `feature_hash` so a model trained on v1 refuses a v2 record.

### 3.1 Example record

```json
{
  "schema_version": "decision_input.v1",
  "ticker": "NVDA",
  "as_of_date": "2026-11-20",
  "generated_at": "2026-11-20T21:35:12Z",

  "technical": {
    "Price_to_MA5": 0.84, "Price_to_MA20": 2.91, "Price_to_MA50": 5.12,
    "RSI": 61.3, "MACD": 2.41, "MACD_Signal": 1.97, "MACD_Hist": 0.44,
    "Stochastic_K": 78.2, "Volatility": 0.021, "ATR_pct": 0.032,
    "BB_Width": 0.118, "BB_Position": 0.81, "VolumeSpike": 1.37,
    "Gap_pct": 0.004, "Return_Lag1": 0.011
  },

  "market": {
    "Beta_90d": 1.98, "Alpha_90d": -0.13, "VolAdj_Return": -0.02,
    "VIX": 17.4, "VIX_chg_5d": -1.2, "SPY_above_200MA": 1, "SPY_ATR_pct": 0.011,
    "SPY_ret_5d": 0.008, "sector_ret_5d": 0.015
  },

  "ml": {
    "p_up_rf": 0.567, "p_up_xgb": 0.497, "p_up_lr": 0.659, "p_up_gb": 0.618,
    "p_up_ensemble": 0.585, "models_agree_frac": 0.75, "oof": true
  },

  "dl": {
    "pred_log_return": 0.0042, "position_size": 0.39, "direction": 1,
    "in_sample": false
  },

  "news": {
    "d1": {
      "n_articles": 14, "n_publishers": 9,
      "pos_ratio": 0.57, "neg_ratio": 0.14, "net_sentiment": 0.43,
      "finbert_mean": 0.31, "finbert_std": 0.22
    },
    "d5": {
      "n_articles": 61, "net_sentiment": 0.21, "finbert_mean": 0.18,
      "volume_z": 1.8
    },
    "sentiment_change_1d_vs_5d": 0.22,
    "llm": {
      "short_term": 1, "short_term_conf": 0.72,
      "long_term": 1, "long_term_conf": 0.65,
      "event_flags": { "earnings": 0, "guidance": 0, "regulatory": 1, "product": 1, "m_and_a": 0, "analyst_action": 1 },
      "source": "live"
    }
  },

  "events": {
    "days_to_earnings": 4, "days_since_earnings": 59, "is_earnings_window": 1,
    "ex_dividend_within_3d": 0
  },

  "fundamentals": {
    "log_market_cap": 29.1, "trailingPE": 52.3, "forwardPE": 31.0,
    "debtToEquity": 0.41, "operatingMargins": 0.62, "earnings_growth": 0.55
  },

  "calendar": { "day_of_week": 4, "month": 11, "is_opex_week": 1 },

  "text_for_explainer": {
    "top_headlines": ["…", "…", "…"],
    "llm_short_term_reasoning": "…"
  },

  "label": {
    "fwd_ret_3d": null, "fwd_move_atr": null, "y": null, "resolved_on": null
  }
}
```

### 3.2 Field notes and changes from the current code

| Field | Source in repo | Change needed |
|---|---|---|
| `technical.*` | `StockPredictor._calculate_features` | Normalise the scale-dependent ones: `ATR_pct = ATR/Close`, `Gap_pct = Gap/Close_prev`, `MACD/Close`. Raw `Volume` and raw `ATR` let a pooled multi-ticker model "recognise" the ticker by price level. |
| `market.*` | `_add_market_features` | Add `SPY_ret_5d`, `VIX_chg_5d`, sector ETF return (XLK, XLY, …). |
| `ml.*` | `predict_next_day()` | Must be **OOF** in training data (`oof: true`). Add `models_agree_frac`. |
| `dl.*` | `AlphaPredictor.predict_next_price()` | Encode `direction` as −1/0/1. Flag `in_sample` for training-universe dates before the checkpoint cutoff. |
| `news.d1/d5` | `polygon.get_news` → `insights` | New aggregator. Apply the 16:00 ET cutoff. Use `strict=True` (only articles whose insights name this ticker). |
| `news.llm` | `NewsAnalyzer` (`NewsSentiment`) | Add an `event_flags` object to the prompt/schema. `source: "live" \| "backfill"`. Train with `backfill` rows excluded or flagged. |
| `events.*` | `yf.Ticker(t).calendar` / earnings dates | New. Earnings proximity is one of the strongest conditioning variables for news impact. |
| `fundamentals.*` | `fetch_current_fundamentals` | `yf.info` is a **current snapshot**, so as-of history is not available from it. For training, use point-in-time quarterly data (WRDS is already in `utils/connect_wrds.py`, or Polygon financials keyed by filing date). Otherwise leave fundamentals out of training and keep them as a post-model filter. |
| `label.*` | Filled ~3 trading days later | `SignalLogger.update_outcomes()` already does a similar job and can fill these. |

### 3.3 Where it lives in code (suggested)

```
utils/data_models.py      + DecisionInput (Pydantic, mirrors §3.1) + DecisionOutput
scripts/feature_builder.py   build_decision_input(ticker, as_of_date) -> DecisionInput   # one code path for train + live
scripts/build_dataset.py     walk-forward loop → data/decision_v1.parquet (one row per ticker-day)
scripts/decision_model.py    train / evaluate / calibrate / save (LogReg + LightGBM)
scripts/ai_market_agent.py   replace the LLM confluence call with DecisionModel.predict(); LLM only writes `reasoning`
```

`DecisionOutput`:
```json
{ "ticker": "NVDA", "as_of_date": "2026-11-20", "p_up": 0.64, "action": "buy",
  "threshold": 0.60, "model_version": "decision_v1_2026-11-20",
  "top_drivers": [["ml.p_up_ensemble", +0.07], ["news.d1.net_sentiment", +0.04], ["market.VIX", -0.02]],
  "reasoning": "LLM-written explanation of the drivers" }
```
`top_drivers` come from SHAP values (XGBoost/LightGBM provide `pred_contribs` natively) and feed both the Telegram message and the LLM explainer prompt.

### 3.4 Evaluation protocol
- Walk-forward folds by **time**, with all tickers pooled per fold, and a 3-day embargo.
- Report per threshold τ: hit rate, coverage (% of days traded), **EV in ATR units** with 3×ATR target and 2×ATR stop, and Brier score / calibration curve.
- Compare four models: (a) ML ensemble alone, (b) current LLM rules (back-test using live-logged rows only), (c) LogReg meta, (d) GBM meta. Adopt the decision model only if it beats (a) on EV at matched coverage.
- Ablations: drop the `news.*` block, then the `dl.*` block. This shows whether each data source is worth its cost on the Jetson.

---

## 4. Data collection plan

1. **Start logging `DecisionInput` live now**, even before the model exists. Every day spent doing so adds clean, non-contaminated rows with LLM features.
2. Back-fill **2024-07-02 → today** with technical + market + OOF ML + Polygon insights + FinBERT. That's ~10 tickers × ~600 days.
3. If you want a longer history (back to ~2018), pull headlines from Polygon's archive (paid plans keep full history) and score them with FinBERT only. Expect `insights` to be null before July 2024, so the model must handle that block being missing.
4. Store data as Parquet (`data/decision_v1.parquet`) and gitignore it (the repo already ignores `*.csv`/`*.json`).

---

## 5. Jetson deployment

### 5.1 Which Jetson matters a lot

| | **Jetson Nano (original, 4 GB)** | **Jetson Orin Nano / Orin Nano Super (8 GB)** |
|---|---|---|
| JetPack / OS | JetPack **4.6.x** (final: 4.6.6), L4T R32, **Ubuntu 18.04**, glibc 2.27. **JetPack 4 is EOL (Nov 2024)**, cannot upgrade the OS | JetPack **6.x** (6.2 added "Super" mode), Ubuntu 22.04, Python 3.10, CUDA 12 |
| System Python | 3.6 | 3.10 |
| FinMod needs | Python ≥ 3.12 (`pyproject.toml`); python-telegram-bot ≥ 3.11 | Same |
| How to get 3.12 | **Docker** `python:3.12-slim` (arm64/Debian has its own glibc). Host-native 3.12 + manylinux wheels will fail on glibc 2.27 | Docker or `uv python install 3.12` on the host. Both work |
| GPU use | Practically none from a 3.12 container (NVIDIA's Nano GPU containers are Py3.6/3.8-era). **Run everything on CPU** | CUDA via `dustynv/*` / jetson-containers images |
| Local LLM | **No.** `gpt-oss:20b` needs ~13 GB+, and even 1B models are slow with 4 GB shared RAM | Yes: 1.5–3B Q4 models via **llama.cpp** (~38 tok/s for Qwen2.5-1.5B Q4_K_M; NVIDIA reports Llama-3.2-3B at ~43 tok/s INT4 on MLC in Super mode). Ollama on JetPack 6.2 has open bug reports, so prefer llama.cpp |
| RAM budget | 4 GB shared CPU/GPU, **add 4–8 GB swap on an SSD/NVMe**, not the SD card | 8 GB shared |

### 5.2 What should run on the device

| Component | On Nano 4 GB | On Orin Nano 8 GB |
|---|---|---|
| Data fetch (yfinance, Polygon) | ✅ | ✅ |
| ML ensemble **inference** (load `models/ml_predictors/*.pkl`) | ✅ fast | ✅ |
| ML ensemble **training / tuning** | ⚠️ possible but slow (minutes–hours with tuning). Better: train on your PC weekly, `rsync` the `.pkl` files over | ✅ |
| GRU inference | ✅ CPU. Export to **ONNX + onnxruntime** to avoid loading full PyTorch (~1 GB RAM) | ✅ PyTorch CUDA or ONNX/TensorRT |
| GRU training | ❌ do it on PC/cloud | ⚠️ possible |
| FinBERT scoring | ✅ via ONNX (int8 quantised ~110 MB) | ✅ |
| Decision model (LightGBM/XGB) | ✅ trivial | ✅ |
| LLM news extraction / explainer | ☁️ Gemini/OpenAI API (`utils/llm/llm_models.py` already supports it) | ✅ local llama.cpp (Qwen2.5-3B / Llama-3.2-3B, JSON grammar) or cloud |
| Streamlit dashboard | ⚠️ works but memory-heavy. Optional | ✅ |

Note: the decision model is what makes the **4 GB Nano viable**. Today every ticker needs several LLM calls and the default provider is a 20B local model. With the decision model, the device only needs cheap CPU inference, and the LLM becomes optional (just for writing the explanation).

### 5.3 Packaging for Jetson

- Split dependencies into extras so the device image stays small: `browser-use`, `streamlit`, `plotly`, `pymysql`, several `langchain-*` providers and training-only code are not needed for the bot.
  ```toml
  [project.optional-dependencies]
  bot    = ["python-telegram-bot[job-queue]>=22", "onnxruntime", "lightgbm"]
  train  = ["torch", "optuna", ...]
  ui     = ["streamlit", "plotly"]
  ```
- Dockerfile sketch (works on both boards, CPU path):
  ```dockerfile
  FROM python:3.12-slim
  RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 tzdata && rm -rf /var/lib/apt/lists/*
  COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
  WORKDIR /app
  COPY pyproject.toml uv.lock ./
  RUN uv sync --frozen --no-dev --extra bot
  COPY . .
  ENV TZ=America/New_York
  CMD ["uv", "run", "python", "-m", "bot.main"]
  ```
  Build on the device or use `docker buildx --platform linux/arm64` on your PC. Mount `models/` and `results/` as volumes so models trained on the PC can be swapped in without rebuilding.
- Fix before deploying: `dl_alpha_production.py` hard-codes `artifacts_dir = /home/sd/FinMod/...`. Make it relative to `PROJECT_ROOT` or an env var. `models/` is gitignored and only `robust_scaler.pkl` is in the repo, so the GRU weights must be copied to the device separately.
- Set `nvpmodel` / `jetson_clocks` for a stable power mode, and put the Docker data root and swap on SSD (SD cards wear out with daily writes).

---

## 6. Telegram bot

### 6.1 Library and transport
- **`python-telegram-bot` v22.x** (latest 22.8, June 2026). It requires **Python ≥ 3.11**, which is another reason for the 3.12 container on the old Nano. Install with the `[job-queue]` extra for scheduling.
- Alternative: `aiogram` 3.x (also async, also modern Python only).
- Use **long polling** (`application.run_polling()`). It only needs outbound HTTPS, so no public IP, domain, TLS cert or port forwarding is required. That suits a home Jetson. Webhooks are only worth it if you already have a reverse proxy.

### 6.2 Proposed commands

| Command | Action |
|---|---|
| `/start`, `/help` | Register chat, show commands |
| `/signal NVDA` | Build `DecisionInput` → run decision model → reply with action, p_up, risk levels (`RiskManager`, with the SELL-direction fix from FINDINGS §2.1), top drivers |
| `/watchlist` | Latest signal for every ticker in `WATCHLIST` (from cache, instant) |
| `/add TSLA`, `/remove TSLA` | Edit the watchlist (persist to SQLite/JSON) |
| `/report` | `SignalLogger.print_report()`: hit rate and EV of live signals so far |
| `/why NVDA` | LLM explanation of the last signal (cloud LLM or local on Orin) |
| `/status` | Device health: model ages, last run, disk, RAM, temperature (`/sys/devices/virtual/thermal/...`) |
| Daily job (JobQueue) | Mon–Fri ~16:30 ET: build inputs, predict, log to `signal_log.csv`, **push** only signals with `p ≥ τ`. Also call `update_outcomes()` |

Example message:
```
📈 NVDA — BUY  (p_up 0.64, τ 0.60)
Entry 182.40 · Stop 170.70 (2×ATR) · Target 199.95 (3×ATR) · 11 sh
Drivers: ML ensemble +0.07 · news sentiment +0.04 · VIX −0.02
Earnings in 4 days ⚠️
```

### 6.3 Practical concerns
- **Long-running work must not block the bot.** Run the model pipeline in `asyncio.to_thread(...)` or a process pool, reply "⏳ working…" first, then edit the message. Training must never happen inside the bot process on the Nano.
- **Access control.** Telegram bots are public. Use an `ALLOWED_CHAT_IDS` allowlist and drop everyone else.
- **Secrets.** `TELEGRAM_BOT_TOKEN` goes in `.env` (already gitignored). Add it to `.env.example`. Never commit it.
- **Rate limits.** Telegram allows ~30 msgs/s globally and ~1 msg/s per chat. A 10-ticker watchlist should be one combined message.
- **Resilience.** Use a systemd unit or `docker compose` with `restart: unless-stopped`, plus a nightly `/status` heartbeat so you notice when the device dies.
- **Disclaimer.** Include "not financial advice" text in `/start`, especially if anyone else gets access.

### 6.4 Architecture on device

```
┌──────────────── Jetson (Docker, TZ=America/New_York) ────────────────┐
│  bot.main (python-telegram-bot, polling)                             │
│     ├── command handlers ──► service.signal(ticker)                  │
│     └── JobQueue 16:30 ET ─► service.daily_run(watchlist)            │
│                                   │                                  │
│  service: feature_builder → DecisionModel → RiskManager → Logger    │
│              │        │           │                                 │
│        yfinance  Polygon     models/ (pkl, onnx) ◄── rsync from PC  │
│                   + FinBERT(onnx)                                    │
│  optional: llama.cpp server (Orin only) or cloud LLM for /why       │
└──────────────────────────────────────────────────────────────────────┘
           ▲ outbound HTTPS only
           └── api.telegram.org, Polygon/Massive, Yahoo, LLM API
```

---

## 7. Suggested order of work

1. **Schema first:** add `DecisionInput` / `DecisionOutput` to `utils/data_models.py` and a `feature_builder.build_decision_input()`. Start **logging live records daily** right away.
2. Fix the news timing (16:00 ET cutoff) and add the Polygon-insights + FinBERT aggregator.
3. Walk-forward OOF predictions for the ML ensemble, then `build_dataset.py` covering 2024-07 → now.
4. Train LogReg + LightGBM meta-models, calibrate, compare against the ML ensemble alone (§3.4).
5. Swap the LLM confluence step for the decision model in `ai_market_agent.py`. The LLM only writes the explanation.
6. Slim dependency extras, Dockerfile, ONNX export of the GRU (and FinBERT).
7. Telegram bot (`bot/` package): `/signal`, `/watchlist`, daily job, allowlist.
8. Deploy to the Jetson with compose + restart policy, plus weekly retraining on PC + `rsync`.

---

## Sources
- NVIDIA — [End of Life for JetPack 4 (4.6.6)](https://forums.developer.nvidia.com/t/announcing-end-of-life-for-nvidia-jetpack-4-with-the-release-of-jetpack-4-6-6/314301), [JetPack 4 reaches EOL](https://forums.developer.nvidia.com/t/jetpack-4-reaches-end-of-life/267563), [Jetson Nano EOL compatibility](https://forums.developer.nvidia.com/t/jetson-nano-eol-compatibility-solutions/269290)
- Canonical — [JetPack 4 EOL, Ubuntu Pro ESM to 2028](https://ubuntu.com/blog/jetpack-4-eol)
- NVIDIA — [JetPack 6.2 Super mode (Orin Nano benchmarks)](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules), [Models that run on Orin Nano Super 8 GB](https://forums.developer.nvidia.com/t/ai-models-that-run-on-jetson-orin-nano-super-8gb-a-practical-guide/365412), [Ollama issues on Orin Nano JP6.2](https://forums.developer.nvidia.com/t/llama3-2-3b-randomly-outputting-gggggggg-when-running-under-ollama-on-jetson-orin-nano-super-jp6-2/342786), [Qwen2.5-1.5B llama.cpp benchmark](https://deepseek.csdn.net/6a61e084662f9a54cb938f41.html)
- python-telegram-bot — [docs (Python 3.11+)](https://docs.python-telegram-bot.org/en/stable/index.html), [changelog (v22.8)](https://docs.python-telegram-bot.org/en/latest/changelog.html)
- Polygon/Massive — [News API sentiment insights](https://www.massive.com/blog/sentiment-analysis-with-ticker-news-api-insights), [How far back news & sentiment go](https://massive.com/knowledge-base/article/how-far-back-does-massives-news-and-sentiment-data-go)
