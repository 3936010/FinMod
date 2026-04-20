# FinMod

Swing trading signal system that combines ensemble ML, a GRU deep learning model, and LLM-powered news sentiment into a single confluence decision.

Built for a 10-ticker watchlist. Designed to be run daily, validated against real outcomes, and improved over time.

---

## How It Works

```
yfinance / Polygon API
        │
        ├── ML Ensemble (per-ticker)
        │     XGBoost + Random Forest + Logistic Regression + Gradient Boosting
        │     23 features: price ratios, momentum, volatility, volume,
        │                  Beta/Alpha, VIX, SPY regime
        │     Target: 3-day forward return, ATR-filtered (noise rows dropped)
        │
        ├── DL Alpha (global GRU, supporting context)
        │     PyTorch GRU trained on 6 major tech stocks
        │     Outputs: predicted log return + position sizing
        │
        └── News Sentiment (Polygon + LLM)
              Last 10 days of news → short/long-term sentiment + key events

All three feed into an LLM Confluence Agent (5-step algorithm)
        │
        ▼
Final signal: Buy / Sell / Hold  +  Confidence score
        │
        ▼
Risk Manager (ATR-based)
        Stop-loss: 2× ATR    Take-profit: 3× ATR    Max position: 20% portfolio
        │
        ▼
Signal Logger → results/signal_log.csv
        Tracks every signal + actual outcome for 3-month demo evaluation
```

---

## Signal Quality

Validated on held-out test data across the watchlist. Use the validation script before trusting any signal:

```bash
uv run python scripts/validate_signal.py
```

Current results (3-day ATR-filtered target, 23 features):

| Confidence threshold | Accuracy | Verdict |
|---|---|---|
| ≥ 50% | 52.4% | coin flip |
| ≥ 55% | 53.4% | marginal |
| **≥ 60%** | **58.1%** | **usable edge** |
| ≥ 65% | 55.0% | too few signals |

**Rule: only act on signals where confidence ≥ 60%.**

With 1.5:1 reward-to-risk (3× ATR target vs 2× ATR stop):
```
Expected value at 58.1% accuracy = +0.9 ATR per trade
```

---

## Project Structure

```
FinMod/
├── app.py                          # Streamlit dashboard (main UI)
├── main.py                         # Batch runner for full watchlist
│
├── scripts/
│   ├── ml_market_movement.py       # Ensemble ML predictor (per-ticker)
│   ├── dl_alpha_production.py      # GRU deep learning model (global)
│   ├── ai_market_agent.py          # Confluence orchestrator
│   ├── news.py                     # News sentiment via Polygon + LLM
│   └── validate_signal.py          # Walk-forward validation script
│
├── utils/
│   ├── risk_manager.py             # ATR-based position sizing
│   ├── signal_logger.py            # Signal outcome tracker (3-month log)
│   ├── data_models.py              # Pydantic schemas
│   └── llm/
│       ├── api_call.py             # LangChain LLM wrapper with retry
│       ├── llm_models.py           # Provider abstraction (Gemini, Ollama)
│       └── prompt.py               # Confluence algorithm prompt
│
├── models/
│   ├── alpha_gru_v1/               # Trained GRU weights + scaler
│   └── ml_predictors/              # Persisted per-ticker ML models (auto-generated)
│
├── results/
│   ├── signal_log.csv              # Live signal + outcome log
│   ├── validation_report.csv       # Last validation run output
│   └── daily_predictions.csv       # Batch run predictions
│
├── FINDINGS.md                     # Full code review and gap analysis
└── TODO.txt                        # Development roadmap
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

### Install

```bash
git clone https://github.com/3936010/FinMod.git
cd FinMod
uv sync
```

### Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```env
POLYGON_API_KEY=your_key       # news data
GOOGLE_API_KEY=your_key        # Gemini LLM
OPENAI_API_KEY=your_key        # optional
```

---

## Usage

### Interactive dashboard

```bash
uv run streamlit run app.py
```

Enter a ticker, set portfolio cash, click **Run Analysis**. The system trains (or loads a cached model), fetches news, runs all three signals, and outputs a Buy/Sell/Hold with full risk levels.

Every BUY or SELL signal is automatically saved to `results/signal_log.csv`.

### Batch run (full watchlist)

```bash
uv run python main.py
```

Runs all tickers in `WATCHLIST`. Models are loaded from cache if under 7 days old; retrained and saved otherwise. Results appended to `results/daily_predictions.csv`.

### Validate signal quality

```bash
uv run python scripts/validate_signal.py                    # full watchlist
uv run python scripts/validate_signal.py --tickers AAPL NVDA   # specific tickers
```

Shows accuracy at each confidence threshold on held-out test data. Run this before starting live demo trading and after any model changes.

### Review 3-month demo results

```python
from utils.signal_logger import SignalLogger

logger = SignalLogger()
logger.update_outcomes()   # fetches actual prices, marks WIN / LOSS
logger.print_report()      # accuracy by confidence tier and by ticker
```

### Train the GRU model from scratch

Only needed if you want to retrain on new data or add tickers to the training universe:

```bash
uv run python scripts/dl_alpha_production.py --train
```

---

## ML Model Details

**Target variable:** 3-day forward return, ATR-filtered.
Only days where `|Close_3d - Close| ≥ 0.5 × ATR` are used for training. Small moves (noise) are excluded. This aligns the model with actual swing trade holding periods and improves signal-to-noise ratio.

**Features (23 total):**

| Group | Features |
|---|---|
| Price vs MAs | `Price_to_MA5`, `Price_to_MA20`, `Price_to_MA50` |
| Momentum | `RSI`, `MACD`, `MACD_Signal`, `MACD_Hist`, `Stochastic_K` |
| Volatility | `Volatility`, `ATR`, `BB_Width`, `BB_Position` |
| Volume | `Volume`, `VolumeSpike` |
| Price action | `Gap`, `Return_Lag1` |
| Market proxy | `Beta_90d`, `Alpha_90d`, `VolAdj_Return`, `PriceVolume_Ratio` |
| Regime | `VIX`, `SPY_above_200MA`, `SPY_ATR_pct` |

**Ensemble:** RF + XGB + LR + GradientBoosting. Weights calibrated on validation set (not test set).

**Model persistence:** Trained models saved to `models/ml_predictors/{TICKER}_predictor.pkl`. Reloaded on subsequent runs. Retrained automatically after 7 days.

---

## Confluence Algorithm (5 steps)

The LLM applies these rules in strict order:

1. **Confidence threshold** — if ML confidence < 60% and DL signal is weak → HOLD
2. **ML vs DL disagreement** — if they point in opposite directions with low conviction → HOLD
3. **Directional alignment** — both ML and DL bullish + sentiment bullish → BUY; bearish → SELL
4. **Technical vs sentiment conflict** — if signals disagree → HOLD
5. **Fundamental filter** — for BUY signals only: PE > 50 or D/E > 1.5 downgrades confidence

---

## Roadmap

- [x] Ensemble ML predictor (RF + XGB + LR + GB) with per-ticker training
- [x] GRU deep learning model as supporting signal
- [x] LLM confluence agent with 5-step algorithm
- [x] ATR-filtered 3-day forward return target
- [x] VIX + SPY regime features
- [x] Model persistence with staleness check (7-day)
- [x] Signal outcome logger for 3-month demo evaluation
- [x] Walk-forward validation script with confidence threshold analysis
- [ ] Full backtesting engine (P&L simulation with ATR stops)
- [ ] Probability calibration (isotonic regression on ensemble outputs)
- [ ] Automated daily run via cron / scheduler
- [ ] Crypto and forex asset class support

---

## License

MIT License — see [LICENSE](LICENSE) for details.
