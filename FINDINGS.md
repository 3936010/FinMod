# FinMod — Code Review & Gap Analysis

> Analyzed: 2026-04-16  
> Scope: Full codebase — `app.py`, `main.py`, `scripts/`, `utils/`, `models/`

---

## Table of Contents

1. [Critical — Crashes & Silent Data Corruption](#1-critical--crashes--silent-data-corruption)
2. [Logic Flaws — Wrong Behavior, No Crash](#2-logic-flaws--wrong-behavior-no-crash)
3. [Development Artifacts in Production](#3-development-artifacts-in-production)
4. [Data & Performance Issues](#4-data--performance-issues)
5. [Configuration & Schema Issues](#5-configuration--schema-issues)
6. [Strategic Gaps — Missing Capabilities](#6-strategic-gaps--missing-capabilities)
7. [Summary Table](#7-summary-table)

---

## 1. Critical — Crashes & Silent Data Corruption

---

### 1.1 `app.py:84` — Calling a removed parameter on `StockPredictor`

**File:** `app.py`, line 84  
**Severity:** Critical — crashes on every UI run

```python
# BROKEN
predictor = StockPredictor(ticker_symbol, use_fundamentals=False)

# StockPredictor.__init__ signature (ml_market_movement.py:31)
def __init__(self, ticker):   # use_fundamentals was removed in a prior refactor
```

`use_fundamentals` was intentionally deleted from `StockPredictor.__init__` (noted as FIX #5 in the ML module). The `app.py` cached factory `get_trained_predictor()` was never updated. This raises `TypeError` every time the Streamlit UI runs analysis.

**Fix:** Remove the `use_fundamentals=False` argument.

```python
predictor = StockPredictor(ticker_symbol)
```

---

### 1.2 `app.py:127-130` — `analyze_news()` called with wrong signature

**File:** `app.py`, lines 127–130  
**Severity:** Critical — crashes; date overrides silently ignored even if it doesn't crash

```python
# BROKEN — analyze_news() takes no arguments
sentiment_result, sentiment_by_date = news_analyzer.analyze_news(
    start_date=start_date_str,
    end_date=end_date_str
)

# Actual signature (news.py:46)
def analyze_news(self):   # dates are set in __init__, not passed here
```

The date range the user selects in the UI is never applied. `NewsAnalyzer` stores `self.start_date` and `self.end_date` at `__init__` time, but `app.py` constructs the analyzer before the user picks dates, then tries to pass them at call time — which isn't supported.

**Fix option A:** Pass dates at `__init__` time (construct `NewsAnalyzer` after date selection).  
**Fix option B:** Add `start_date` / `end_date` parameters to `analyze_news()` and override `self` attributes if provided.

---

### 1.3 `dl_alpha_production.py:273-275` — Random split on time-series data (data leakage)

**File:** `scripts/dl_alpha_production.py`, lines 273–275  
**Severity:** Critical — GRU model trained with look-ahead bias; reported metrics are inflated

```python
# BROKEN — random_split on time-series allows future windows into training
train_size = int(0.9 * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
```

For a sliding-window time-series dataset, shuffling windows mixes future data into training. A window starting at day 2000 could be in `train_ds` while a window from day 100 is in `val_ds` — the model effectively sees the future during training. The ML module (`ml_market_movement.py`) correctly uses a strict chronological split; the DL module does not.

**Fix:** Replace with a sequential index split.

```python
# FIXED
train_size = int(0.9 * len(dataset))
train_ds   = torch.utils.data.Subset(dataset, range(0, train_size))
val_ds     = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
# Do NOT shuffle train_loader for time-series
train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=False, pin_memory=True)
```

---

### 1.4 `dl_alpha_production.py:243-247` — Scaler fit on full history before split

**File:** `scripts/dl_alpha_production.py`, lines 243–247  
**Severity:** Critical — second form of data leakage in the DL pipeline

```python
# BROKEN — scaler sees the entire dataset including the held-out validation period
all_X  = np.concatenate([p[0] for p in processed_data], axis=0)
scaler = RobustScaler()
scaler.fit(all_X)   # learns median/IQR from future data
```

The `RobustScaler` is fit on the full concatenated feature matrix before any train/val split. This means the scaler's median and IQR are informed by future market data — a subtler but equally invalid form of look-ahead bias.

**Fix:** Fit the scaler only on training windows, then apply to validation windows.

```python
# FIXED — approximate chronological split for scaler fitting
split_idx = int(0.9 * len(all_X))
scaler    = RobustScaler()
scaler.fit(all_X[:split_idx])   # train portion only
```

---

### 1.5 `ml_market_movement.py:236-237` — Last training row has a fabricated Target label

**File:** `scripts/ml_market_movement.py`, lines 236–237  
**Severity:** Critical — last row trained on an incorrect label (always `0`)

```python
data["Tomorrow"] = data["Close"].shift(-1)          # last row → NaN
data['Target']   = np.where(data['Tomorrow'] > data['Close'], 1, 0)
# numpy: NaN > float → False → Target = 0
```

Then:

```python
data = data.ffill()     # Target is already 0, not NaN → ffill does not fix it
data.dropna(inplace=True)  # Target=0 is not NaN → row survives into training
```

The last row has a valid feature vector but a fabricated `Target=0` label (because `NaN > Close` evaluates to `False` in numpy). This row is included in training and silently biases the model toward predicting DOWN for the most recent market state.

**Fix:** Drop rows with missing `Tomorrow` before computing `Target`.

```python
data["Tomorrow"] = data["Close"].shift(-1)
data.dropna(subset=["Tomorrow"], inplace=True)   # remove last row before labeling
data['Target'] = np.where(data['Tomorrow'] > data['Close'], 1, 0)
```

---

## 2. Logic Flaws — Wrong Behavior, No Crash

---

### 2.1 `risk_manager.py` — Sell signal receives inverted risk levels

**File:** `utils/risk_manager.py`, `app.py:232`  
**Severity:** High — SELL trades get stop-loss below entry and take-profit above entry (opposite of correct)

`calculate_entry()` is designed for long positions only:

```python
stop_loss_price   = current_price - (2 * atr)   # correct for LONG
take_profit_price = current_price + (3 * atr)   # correct for LONG
```

`app.py` calls it for both buy and sell:

```python
if action in ["buy", "sell"]:
    risk_output = risk_manager.calculate_entry(
        total_cash, current_price, atr, final_prediction.confidence
    )
```

For a SHORT position the stop-loss must be **above** entry (`current_price + 2*ATR`) and take-profit **below** entry (`current_price - 3*ATR`). Currently a SELL signal displays the wrong levels — it looks like a long trade setup.

**Fix:** Add a `direction` parameter to `calculate_entry()`.

```python
def calculate_entry(self, total_cash, current_price, atr, llm_confidence, direction="buy"):
    if direction == "buy":
        stop_loss_price   = max(0.0, current_price - (2 * atr))
        take_profit_price = current_price + (3 * atr)
    else:  # sell / short
        stop_loss_price   = current_price + (2 * atr)
        take_profit_price = max(0.0, current_price - (3 * atr))
```

---

### 2.2 `prompt.py` — ML and DL Alpha signal formats conflict

**File:** `utils/llm/prompt.py`, lines 101–102  
**Severity:** High — LLM receives inconsistent signal vocabulary with no reconciliation guidance

The confluence algorithm states:

> "If ML_Prediction/DL Alpha == 'UP' AND Sentiment == 'bullish' → buy"

But the two models use different vocabularies:

| Source | Direction field | Values |
|---|---|---|
| ML Ensemble (`ml_analysis`) | `Ensemble_Prediction` | `"UP"` / `"DOWN"` |
| DL Alpha (`alpha_analysis`) | `direction` | `"Bullish"` / `"Bearish"` / `"Neutral"` |

The prompt conflates them as if they produce identical output. The LLM must silently resolve this mismatch. This can cause inconsistent signal generation across providers.

**Fix:** Normalize both signals before building the prompt (in `ai_market_agent.py` / `app.py`), or explicitly define the mapping in the system prompt.

---

### 2.3 `prompt.py` — Confluence algorithm has no ML vs DL disagreement rule

**File:** `utils/llm/prompt.py`, lines 94–108  
**Severity:** High — undefined behavior when ML and DL disagree with each other

The algorithm defines:
- Step 2: Both agree → BUY or SELL
- Step 3: Tech vs Sentiment conflict → HOLD

But it never defines what to do when **ML says UP and DL Alpha says Bearish** (or vice versa). The LLM is left to interpret this case with no guiding rule — different providers will handle it differently, making the system non-deterministic.

**Fix:** Add an explicit Step 2.5 to the prompt.

```
STEP 2.5 — ML vs DL Disagreement:
If ML Ensemble and DL Alpha point in opposite directions:
    → Treat as weak/conflicted technical signal
    → If both DL position_size magnitude < 0.3 AND ML confidence < 0.65:
        action: "hold", Reasoning: "[DIVERGENCE]: ML/DL technical disagreement."
    → Otherwise, weight toward the higher-confidence signal.
```

---

### 2.4 `ai_market_agent.py:141` — By-date sentiment silently discarded

**File:** `scripts/ai_market_agent.py`, line 141  
**Severity:** Medium — data fetched and paid for (LLM calls) but never used

```python
sentiment_result, sentiment_date = self.news_analyzer.analyze_news()
# sentiment_date is never referenced again in run_analysis()
```

`analyze_news()` returns `(sentiment_result, sentiment_results_by_date)`. The agent unpacks the second value as `sentiment_date` and immediately discards it. The per-day sentiment breakdown — including daily catalysts and stock price movement signals — never reaches the LLM confluence prompt.

**Fix:** Either pass `sentiment_results_by_date` into the LLM prompt as additional context, or stop computing it if it won't be used (saves LLM calls).

---

### 2.5 `api_call.py:60-84` — Default fallback response ignores `Literal` type constraints

**File:** `utils/llm/api_call.py`, lines 60–84  
**Severity:** Medium — LLM failure produces a fake valid-looking response with an invalid `action`

```python
def create_default_response(model_class):
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
```

`MarketPrediction.action` is typed as `str`, so it gets `"Error in analysis, using default"`. Downstream in `app.py`:

```python
action = final_prediction.action.lower()
# → "error in analysis, using default"
# Falls through to `else` branch → renders as HOLD with no indication of failure
```

A silent LLM failure is presented to the user as a legitimate HOLD signal.

**Fix (short-term):** Define `action` as `Literal["buy", "sell", "hold"]` in `MarketPrediction` and handle the default as an explicit `"hold"`.

```python
class MarketPrediction(BaseModel):
    ticker:     str
    action:     Literal["buy", "sell", "hold"]
    confidence: float
    reasoning:  str
```

**Fix (default response):** Return `action="hold"` with `confidence=0.0` and `reasoning="[ERROR]: LLM unavailable."` explicitly rather than using the generic field-type sniffer.

---

## 3. Development Artifacts in Production

---

### 3.1 `news.py:23-28` — `importlib.reload()` at module import time

**File:** `scripts/news.py`, lines 23–28  
**Severity:** Medium — double initialization, hard-to-trace side effects

```python
import importlib
import utils.polygon_utils
import utils.llm.prompt
import utils.llm.api_call
importlib.reload(utils.polygon_utils)
importlib.reload(utils.llm.prompt)
importlib.reload(utils.llm.api_call)
```

This was added during development to pick up live edits without restarting the kernel. Left in production, it reloads three modules every time `news.py` is imported — running their module-level code twice, resetting any module state, and making import order significant. In a Streamlit app with `@st.cache_resource`, this can cause subtle cache invalidation.

**Fix:** Delete these six lines entirely.

---

### 3.2 `news.py:47` — NVIDIA variable names for a generic function

**File:** `scripts/news.py`, line 47  
**Severity:** Low — no functional impact; indicates the function was never properly abstracted

```python
nvdia_news, nvdia_news_df = self.pl.get_news(self.ticker, ...)
```

Copy-pasted from an NVIDIA dev session. Variable names signal that the abstraction was incomplete. While harmless now, it increases the cost of reading and debugging the code.

**Fix:** Rename to `news_results, news_df`.

---

### 3.3 `news.py:33` — Default argument evaluated at class definition time

**File:** `scripts/news.py`, line 33  
**Severity:** Medium — in a long-running Streamlit session, date defaults become stale

```python
def __init__(
    self,
    ticker,
    start_date = datetime.date.today() - datetime.timedelta(days=7),  # evaluated ONCE at import
    end_date   = datetime.date.today(),                                # same problem
    provider   = "Ollama"
):
```

Python evaluates default argument values at function **definition** time (module import), not at call time. If the Streamlit app stays running across midnight, `NewsAnalyzer()` constructed without explicit dates will use the stale import-time date as its default.

**Fix:** Use `None` as the sentinel and compute inside the body.

```python
def __init__(self, ticker, start_date=None, end_date=None, provider="Ollama"):
    self.start_date = start_date or (datetime.date.today() - datetime.timedelta(days=7))
    self.end_date   = end_date   or datetime.date.today()
```

---

## 4. Data & Performance Issues

---

### 4.1 `ml_market_movement.py:171` — SPY timezone alignment silently fails

**File:** `scripts/ml_market_movement.py`, line 171  
**Severity:** High — Beta and Alpha silently become NaN; defaulted to 1.0/0.0 with no user warning

```python
spy_data = spy.history(period="max")           # timezone-aware index
data['SPY_Return'] = spy_data['SPY_Return'].reindex(data.index)
# If index timezones differ → all NaN → Beta=1.0, Alpha=0.0
```

`yf.Ticker().history()` returns a timezone-aware `DatetimeIndex` (UTC or exchange-local). The stock's `data` index may be timezone-naive (from `yf.download()`) or in a different timezone. A mismatch causes `.reindex()` to find no matching dates — every SPY return becomes NaN — and Beta/Alpha silently default to their neutral values.

**Fix:** Normalize both indexes to UTC before alignment.

```python
spy_data.index = spy_data.index.tz_convert("UTC")
data.index     = pd.DatetimeIndex(data.index).tz_localize("UTC") \
                 if data.index.tzinfo is None else data.index.tz_convert("UTC")
data['SPY_Return'] = spy_data['SPY_Return'].reindex(data.index)
```

---

### 4.2 `news.py:63-68` — Sequential blocking LLM calls per day of news

**File:** `scripts/news.py`, lines 63–68  
**Severity:** Medium — 7-day window = 8+ serial LLM calls; 30-day window = 31+ calls

```python
for date, news in parse_news_by_date(nvdia_news.results).items():
    sentiment_result_by_date = call_llm(...)   # blocking, serial
    sentiment_results_by_date.append(sentiment_result_by_date)
```

Each day triggers a synchronous LLM call. For a week of news this typically takes 30–60 seconds end-to-end. Combined with finding 1.4 (the results are discarded), this is pure waste.

**Fix (short-term):** Remove per-date analysis entirely until the output is wired into the prompt.  
**Fix (long-term):** Use `asyncio` / `concurrent.futures.ThreadPoolExecutor` to parallelize calls if the per-date breakdown is restored.

---

### 4.3 `app.py:135` — Full `MarketAgent` instantiated to call one method

**File:** `app.py`, line 135  
**Severity:** Medium — loads PyTorch GRU weights from disk just to fetch 5 yfinance fields

```python
fundamental_analysis = MarketAgent(ticker)._fetch_current_fundamentals()
```

`MarketAgent.__init__` constructs `StockPredictor` (no issue) and `AlphaPredictor`, which opens and deserializes the PyTorch model file from disk. The entire object is then garbage-collected immediately — only the five fundamental fields are kept.

**Fix:** Extract `_fetch_current_fundamentals` as a standalone function (or a `@staticmethod`) so it can be called without a full `MarketAgent`.

```python
# utils/fundamentals.py
def fetch_current_fundamentals(ticker: str) -> dict:
    ...
```

---

## 5. Configuration & Schema Issues

---

### 5.1 `ai_market_agent.py:40` — Hardcoded absolute path for GRU model

**File:** `scripts/ai_market_agent.py`, line 40  
**Severity:** Medium — breaks on any machine other than the original dev environment

```python
self.alpha_predictor = AlphaPredictor("/home/sd/FinMod/models/alpha_gru_v1")
```

Same hardcoded path appears in `app.py:113`. If the repo is cloned elsewhere, moved, or run in a container, this silently fails or raises `FileNotFoundError`.

**Fix:** Derive the path relative to the project root.

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
self.alpha_predictor = AlphaPredictor(PROJECT_ROOT / "models" / "alpha_gru_v1")
```

---

### 5.2 `app.py:59-60` and `app.py:152` — LLM provider dropdown is cosmetic

**File:** `app.py`, lines 59–60 and 152  
**Severity:** Medium — user selection is silently ignored

```python
model_options  = ["gemini-2.5-flash"]     # one option; cannot be changed without editing code
selected_model = st.sidebar.selectbox("LLM Model", model_options)
...
call_llm(prompt, selected_model, "Gemini", ...)   # "Gemini" hardcoded; ignores provider selection
```

The provider string `"Gemini"` is hardcoded in the `call_llm` invocation regardless of what `selected_model` is. Adding more model options to the list would have no effect without also wiring the provider correctly.

**Fix:** Derive the provider from the selected model name, or surface a separate provider dropdown.

---

### 5.3 `app.py:66-67` — News date range defaults to hardcoded past dates

**File:** `app.py`, lines 66–67  
**Severity:** Low — poor UX; defaults to dates over 2 months in the past

```python
news_start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2026-02-01"))
news_end_date   = st.sidebar.date_input("End Date",   value=pd.to_datetime("2026-02-05"))
```

**Fix:** Default to the last 7 days dynamically.

```python
from datetime import date, timedelta
news_start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=7))
news_end_date   = st.sidebar.date_input("End Date",   value=date.today())
```

---

### 5.4 `data_models.py:241` — `MarketPrediction.action` has no validation

**File:** `utils/data_models.py`, line 241–244  
**Severity:** Medium — an LLM returning `"strong buy"` or `"sell short"` passes schema validation

```python
class MarketPrediction(BaseModel):
    ticker:     str
    action:     str       # any string accepted
    confidence: float
    reasoning:  str
```

**Fix:** Constrain to valid values.

```python
from typing import Literal

class MarketPrediction(BaseModel):
    ticker:     str
    action:     Literal["buy", "sell", "hold"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning:  str
```

---

## 6. Strategic Gaps — Missing Capabilities

---

### 6.1 No backtesting framework

The TODO acknowledges this. Without backtesting there is no way to know whether the confluence signals have positive expectancy. The system can generate confident signals that lose money on every trade — and there is currently no mechanism to detect that.

**What's needed:** A `Backtester` class that:
1. Iterates through historical dates using only data available at each point (walk-forward)
2. Applies the ML/DL/LLM signal generation for each date
3. Simulates entry/exit using the RiskManager levels
4. Calculates Sharpe ratio, max drawdown, win rate, profit factor

---

### 6.2 ML models retrain from scratch on every run

`save_models()` and `load_models()` are implemented in `StockPredictor` but are commented out / never called in the main flow. `@st.cache_resource` in the UI prevents retraining within a session but not across restarts. Each cold start re-downloads all data and re-trains 4 models + GridSearchCV, which can take 5–15 minutes per ticker.

**What's needed:** A model staleness check — load persisted models if they exist and were trained within N days; only retrain if stale.

---

### 6.3 Ensemble probabilities are not calibrated

Random Forest and Gradient Boosting classifiers are known to produce poorly calibrated probability outputs — RF tends to be overconfident toward 0 and 1; GBM can be underconfident near the center. The `Ensemble_Confidence` value passed to the `RiskManager` and LLM prompt is raw, uncalibrated probability.

**What's needed:** Apply `sklearn.calibration.CalibratedClassifierCV` (with `method='isotonic'` or `'sigmoid'`) to each model after training, using the validation set. This makes the 70% confidence number actually mean "right 70% of the time."

---

### 6.4 No principled ML vs DL signal weighting

When ML and DL Alpha agree, the system is confident. When they disagree, the LLM is left to figure it out with no explicit weighting rule. There is no tracked record of which signal has been more reliable for which ticker or market condition.

**What's needed:** A simple performance tracker that logs `(ML_signal, DL_signal, actual_next_day_return)` and periodically updates a per-ticker weight between the two signals based on recent accuracy.

---

### 6.5 No market regime detection

The same technical signals and the same confluence rules are applied regardless of whether the market is:
- **Trending** (momentum works well)
- **Mean-reverting / ranging** (momentum fails; contrarian works)
- **High-volatility / crisis** (all signals are noise; risk reduction is priority)

A signal from a momentum-trained model in a mean-reverting regime has negative expected value.

**What's needed:** A lightweight regime classifier (e.g., VIX level + SPY 20-day ATR percentile + Hurst exponent on recent returns) that gates or modifies the confluence algorithm's thresholds per regime.

---

## 7. Summary Table

| # | File | Issue | Severity |
|---|---|---|---|
| 1.1 | `app.py:84` | `StockPredictor` called with removed `use_fundamentals` param | **Critical** |
| 1.2 | `app.py:127` | `analyze_news()` called with args it doesn't accept; dates ignored | **Critical** |
| 1.3 | `dl_alpha_production.py:273` | `random_split` on time-series → look-ahead bias in GRU training | **Critical** |
| 1.4 | `dl_alpha_production.py:243` | Scaler fit on full history before train/val split | **Critical** |
| 1.5 | `ml_market_movement.py:237` | Last training row has fabricated `Target=0` label | **Critical** |
| 2.1 | `risk_manager.py` | SELL signal gets inverted stop-loss and take-profit levels | High |
| 2.2 | `prompt.py:101` | ML (`UP/DOWN`) vs DL (`Bullish/Bearish`) format mismatch in prompt | High |
| 2.3 | `prompt.py:94` | Confluence has no rule for ML vs DL disagreement | High |
| 2.4 | `ai_market_agent.py:141` | Per-date sentiment result fetched and discarded | Medium |
| 2.5 | `api_call.py:60` | Default fallback ignores `Literal` constraints → silent invalid `action` | Medium |
| 3.1 | `news.py:23` | `importlib.reload()` calls at module import time | Medium |
| 3.2 | `news.py:47` | `nvdia_news` variable names in a generic function | Low |
| 3.3 | `news.py:33` | Mutable default date evaluated once at import time | Medium |
| 4.1 | `ml_market_movement.py:171` | SPY timezone mismatch → Beta/Alpha silently NaN → defaults | High |
| 4.2 | `news.py:63` | Sequential blocking LLM call per day of news (8+ serial calls) | Medium |
| 4.3 | `app.py:135` | Full `MarketAgent` (loads PyTorch) instantiated to fetch 5 fields | Medium |
| 5.1 | `ai_market_agent.py:40` | GRU model path hardcoded to `/home/sd/FinMod/...` | Medium |
| 5.2 | `app.py:152` | LLM provider dropdown is cosmetic; `"Gemini"` hardcoded | Medium |
| 5.3 | `app.py:66` | News date defaults hardcoded to past dates in Feb 2026 | Low |
| 5.4 | `data_models.py:241` | `MarketPrediction.action` unvalidated (`str` not `Literal`) | Medium |
| 6.1 | — | No backtesting framework | Strategic |
| 6.2 | — | ML models retrain from scratch every run | Strategic |
| 6.3 | — | Ensemble probabilities not calibrated | Strategic |
| 6.4 | — | No ML vs DL signal weighting / performance tracking | Strategic |
| 6.5 | — | No market regime detection | Strategic |

---

*Generated from full codebase read of `app.py`, `scripts/ai_market_agent.py`, `scripts/ml_market_movement.py`, `scripts/dl_alpha_production.py`, `scripts/news.py`, `utils/risk_manager.py`, `utils/llm/prompt.py`, `utils/llm/api_call.py`, `utils/data_models.py`.*
