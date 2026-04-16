"""
scripts/validate_signal.py — Walk-forward ML signal validation.

Answers the key question before you start demo trading:
"At each confidence level, how often was the ML signal actually correct
over the last 6 months of held-out data?"

Usage
-----
    python scripts/validate_signal.py                    # all WATCHLIST tickers
    python scripts/validate_signal.py --tickers AAPL NVDA   # specific tickers
    python scripts/validate_signal.py --months 3         # shorter lookback

How it works
------------
StockPredictor's data_processing() already creates a clean train/val/test
split (15% test, 10% val, 75% train) in chronological order.  This script
uses that same test set — which the model never trained on — and evaluates
accuracy at different confidence thresholds.  The result tells you which
confidence level is actually meaningful for this system.
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime

from scripts.ml_market_movement import StockPredictor


# ── Config ────────────────────────────────────────────────────────────────────
WATCHLIST = ['QCOM', 'NVDA', 'AAPL', 'AMZN', 'AMD', 'TSCO', 'WMT']
CONFIDENCE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
# ─────────────────────────────────────────────────────────────────────────────


def validate_ticker(ticker: str) -> dict:
    """
    Train on the full history (minus test set) and evaluate the ensemble
    on the held-out test set at multiple confidence thresholds.

    Returns a dict with per-threshold accuracy, count, and win rate.
    """
    print(f"\n  [{ticker}] Loading and training...")
    predictor = StockPredictor(ticker)
    predictor.data_processing()
    predictor.tune_and_train_models(tune=False)

    # ── Get test set predictions ───────────────────────────────────────────
    X_test = predictor.X_test_scaled
    y_test = predictor.y_test.values

    # Individual model probabilities
    prob_RF  = predictor.model_RF.predict_proba(X_test)[:, 1]
    prob_XGB = predictor.model_XGB.predict_proba(X_test)[:, 1]
    prob_LR  = predictor.model_LR.predict_proba(X_test)[:, 1]
    prob_GB  = predictor.model_GB.predict_proba(X_test)[:, 1]

    w = predictor.model_weights
    ensemble_prob_up = (
        prob_RF  * w["RF"]  +
        prob_XGB * w["XGB"] +
        prob_LR  * w["LR"]  +
        prob_GB  * w["GB"]
    )

    # Confidence = how sure the ensemble is (distance from 0.5 → 1.0)
    # ensemble_prob_up > 0.5 → predicting UP, confidence = ensemble_prob_up
    # ensemble_prob_up < 0.5 → predicting DOWN, confidence = 1 - ensemble_prob_up
    predicted_up   = ensemble_prob_up >= 0.5
    confidence_arr = np.where(predicted_up, ensemble_prob_up, 1 - ensemble_prob_up)

    # Correct = predicted direction matches actual
    actual_up   = y_test == 1
    correct_arr = predicted_up == actual_up

    # Test set date range for display
    test_dates = predictor.X_test.index
    date_range = f"{test_dates[0].date()} → {test_dates[-1].date()}"

    results = {
        "ticker":     ticker,
        "date_range": date_range,
        "test_rows":  len(y_test),
        "thresholds": {},
    }

    # ── Accuracy at each confidence threshold ─────────────────────────────
    for thresh in CONFIDENCE_THRESHOLDS:
        mask  = confidence_arr >= thresh
        count = mask.sum()
        if count == 0:
            results["thresholds"][thresh] = {"count": 0, "accuracy": None, "buy_signals": 0, "sell_signals": 0}
            continue

        accuracy    = correct_arr[mask].mean() * 100
        buy_signals  = (predicted_up[mask]).sum()
        sell_signals = (~predicted_up[mask]).sum()

        results["thresholds"][thresh] = {
            "count":        int(count),
            "accuracy":     round(accuracy, 1),
            "buy_signals":  int(buy_signals),
            "sell_signals": int(sell_signals),
        }

    return results


def print_report(all_results: list):
    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print()
    print("  How to read this: 'Accuracy' is how often the signal was correct")
    print("  on the held-out test set at or above each confidence threshold.")
    print("  Aim for ≥58% accuracy at your chosen threshold before trading.")
    print()

    header = f"  {'Ticker':<7} {'Test Period':<24} {'N':>5}"
    for t in CONFIDENCE_THRESHOLDS:
        header += f"  {'≥'+str(int(t*100))+'%':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in all_results:
        row = f"  {r['ticker']:<7} {r['date_range']:<24} {r['test_rows']:>5}"
        for t in CONFIDENCE_THRESHOLDS:
            stats = r["thresholds"].get(t, {})
            acc   = stats.get("accuracy")
            n     = stats.get("count", 0)
            if acc is None or n == 0:
                row += f"  {'—':>9}"
            else:
                marker = "✓" if acc >= 58 else " "
                row += f"  {acc:>6.1f}%{marker} "
        print(row)

    print()
    print("  ✓ = ≥58% accuracy (usable signal threshold)")
    print()

    # ── Aggregate summary ─────────────────────────────────────────────────
    print("  --- Aggregate (all tickers combined) ---")
    agg = {t: {"correct": 0, "total": 0} for t in CONFIDENCE_THRESHOLDS}
    for r in all_results:
        for t in CONFIDENCE_THRESHOLDS:
            stats = r["thresholds"].get(t, {})
            n     = stats.get("count", 0)
            acc   = stats.get("accuracy")
            if acc is not None and n > 0:
                agg[t]["correct"] += int(n * acc / 100)
                agg[t]["total"]   += n

    agg_row = f"  {'ALL':<7} {'':<24} {'':<5}"
    for t in CONFIDENCE_THRESHOLDS:
        total   = agg[t]["total"]
        correct = agg[t]["correct"]
        if total == 0:
            agg_row += f"  {'—':>9}"
        else:
            acc    = correct / total * 100
            marker = "✓" if acc >= 58 else " "
            agg_row += f"  {acc:>6.1f}%{marker} "
    print(agg_row)

    print()

    # ── Recommended confidence threshold ──────────────────────────────────
    print("  --- Recommendation ---")
    best_threshold = None
    for t in sorted(CONFIDENCE_THRESHOLDS, reverse=True):
        total = agg[t]["total"]
        if total == 0:
            continue
        acc = agg[t]["correct"] / total * 100
        if acc >= 58:
            best_threshold = t
            break

    if best_threshold:
        total = agg[best_threshold]["total"]
        acc   = agg[best_threshold]["correct"] / total * 100
        print(f"  Suggested minimum confidence threshold: {int(best_threshold*100)}%")
        print(f"  At this level: {acc:.1f}% accuracy across {total} test signals")
        print(f"  → Only act on signals where LLM+ML confidence ≥ {int(best_threshold*100)}%")
    else:
        print("  WARNING: No confidence threshold reached 58% accuracy.")
        print("  The ML signal may not have a reliable edge on this data.")
        print("  Consider retraining with more data or different feature engineering.")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Walk-forward ML signal validation")
    parser.add_argument("--tickers", nargs="+", default=WATCHLIST,
                        help="Tickers to validate (default: full watchlist)")
    args = parser.parse_args()

    print(f"Validating {len(args.tickers)} ticker(s): {', '.join(args.tickers)}")
    print("This trains a model per ticker — takes a few minutes...\n")

    all_results = []
    for ticker in args.tickers:
        try:
            result = validate_ticker(ticker)
            all_results.append(result)
        except Exception as e:
            print(f"  [{ticker}] ERROR: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        print_report(all_results)

        # Save to CSV for reference
        rows = []
        for r in all_results:
            for t, stats in r["thresholds"].items():
                rows.append({
                    "ticker":       r["ticker"],
                    "date_range":   r["date_range"],
                    "test_rows":    r["test_rows"],
                    "threshold":    t,
                    "count":        stats.get("count", 0),
                    "accuracy_pct": stats.get("accuracy"),
                    "buy_signals":  stats.get("buy_signals", 0),
                    "sell_signals": stats.get("sell_signals", 0),
                })
        out_path = Path(__file__).resolve().parent.parent / "results" / "validation_report.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
