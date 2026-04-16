"""
main.py — Batch signal generator for the watchlist.

Run this daily (or on demand) to produce next-day predictions for every
ticker. Models are persisted to disk and only retrained when stale
(older than MODEL_STALENESS_DAYS), so the 10-ticker watchlist does not
require a full retrain on every run.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta

from scripts.ml_market_movement import StockPredictor

# ── Config ────────────────────────────────────────────────────────────────────
WATCHLIST = ['QCOM', 'NVDA', 'AAPL', 'AMZN', 'AMD', 'TSCO', 'WMT']
MODELS_DIR = Path(__file__).parent / "models" / "ml_predictors"
MODEL_STALENESS_DAYS = 7   # retrain if model is older than this
# ─────────────────────────────────────────────────────────────────────────────


def is_stale(model_path: Path) -> bool:
    """Return True if the model file is older than MODEL_STALENESS_DAYS."""
    age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
    return age > timedelta(days=MODEL_STALENESS_DAYS)


def train_and_save(predictor: StockPredictor, model_path: Path):
    """Full training cycle — processes data, trains, evaluates, saves."""
    predictor.data_processing()
    predictor.tune_and_train_models(tune=False)
    predictor.evaluate_models()
    predictor.save_models(str(model_path))


def run_ticker(ticker: str):
    model_path = MODELS_DIR / f"{ticker}_predictor.pkl"
    predictor  = StockPredictor(ticker)

    if model_path.exists() and not is_stale(model_path):
        age_hours = (datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)).seconds // 3600
        age_days  = (datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)).days
        print(f"\n[{ticker}] Loading cached model (age: {age_days}d {age_hours}h) — skipping retrain")
        predictor.load_models(str(model_path))
        # predict_next_day() fetches fresh data itself when self.data is None
    else:
        reason = "stale" if model_path.exists() else "first run"
        print(f"\n[{ticker}] {reason.capitalize()} — training from scratch...")
        train_and_save(predictor, model_path)

    prediction = predictor.predict_next_day()
    predictor.save_final_prediction(prediction)
    return prediction


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  FinMod Batch Run — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Watchlist : {', '.join(WATCHLIST)}")
    print(f"  Models dir: {MODELS_DIR}")
    print("=" * 60)

    results = {}
    for ticker in WATCHLIST:
        try:
            results[ticker] = run_ticker(ticker)
        except Exception as e:
            print(f"\n[{ticker}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[ticker] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for ticker, pred in results.items():
        if "error" in pred:
            print(f"  {ticker:6s}  ERROR — {pred['error']}")
        else:
            signal = pred.get("Ensemble_Prediction", "?")
            conf   = pred.get("Ensemble_Confidence", 0)
            print(f"  {ticker:6s}  {signal:4s}  confidence={conf:.1%}")
