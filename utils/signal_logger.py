"""
utils/signal_logger.py — Signal outcome tracker for 3-month demo evaluation.

Usage
-----
Logging a signal (call this right after a prediction is generated):

    from utils.signal_logger import SignalLogger
    logger = SignalLogger()
    logger.log_signal(
        ticker          = "AAPL",
        action          = final_prediction.action,
        confidence      = final_prediction.confidence,
        reasoning       = final_prediction.reasoning,
        ml_prediction   = ml_analysis["Ensemble_Prediction"],
        ml_confidence   = ml_analysis["Ensemble_Confidence"],
        dl_direction    = alpha_analysis.get("direction", "N/A"),
        dl_position_size= alpha_analysis.get("position_size", 0.0),
        news_sentiment  = news_analysis.get("short_term_sentiment", {}).get("short_term_sentiment", "N/A"),
        current_price   = current_price,
        stop_loss       = risk_output["stop_loss_price"],
        take_profit     = risk_output["take_profit_price"],
        shares          = risk_output["shares_to_buy"],
        atr             = atr,
    )

Updating outcomes (run this daily — it fills in actual prices for open trades):

    logger.update_outcomes()

Printing the 3-month report:

    logger.print_report()
"""

import os
import csv
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional

import yfinance as yf
import pandas as pd

# ── File location ─────────────────────────────────────────────────────────────
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SIGNAL_LOG   = _RESULTS_DIR / "signal_log.csv"

# ── CSV columns ───────────────────────────────────────────────────────────────
COLUMNS = [
    "log_id",
    "timestamp",
    "signal_date",
    "ticker",
    "action",
    "confidence",
    "ml_prediction",
    "ml_confidence",
    "dl_direction",
    "dl_position_size",
    "news_sentiment",
    "current_price",
    "stop_loss",
    "take_profit",
    "shares",
    "atr",
    "reasoning",
    # Filled in by update_outcomes()
    "next_day_price",
    "next_day_return_pct",
    "week_price",
    "week_return_pct",
    "outcome",        # WIN / LOSS / NEUTRAL / OPEN
    "outcome_updated",
]
# ─────────────────────────────────────────────────────────────────────────────


class SignalLogger:
    """
    Appends trading signals to a CSV and periodically resolves outcomes
    by fetching actual market prices from yfinance.
    """

    def __init__(self, log_path: Path = SIGNAL_LOG):
        self.log_path = log_path
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self._init_csv()

    # ── Public API ────────────────────────────────────────────────────────────

    def log_signal(
        self,
        ticker:          str,
        action:          str,
        confidence:      float,
        reasoning:       str,
        ml_prediction:   str,
        ml_confidence:   float,
        dl_direction:    str,
        dl_position_size:float,
        news_sentiment:  str,
        current_price:   float,
        stop_loss:       float,
        take_profit:     float,
        shares:          int,
        atr:             float,
    ) -> str:
        """
        Append a new signal row.  Returns the log_id for reference.
        Only called when action is 'buy' or 'sell' — HOLD signals are
        skipped because there is nothing to track.
        """
        if action.lower() == "hold":
            print(f"[SignalLogger] HOLD for {ticker} — not logged (nothing to track).")
            return ""

        log_id = f"{ticker}_{date.today().isoformat()}_{int(datetime.now().timestamp())}"
        row = {
            "log_id":           log_id,
            "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "signal_date":      date.today().isoformat(),
            "ticker":           ticker,
            "action":           action.lower(),
            "confidence":       round(confidence, 4),
            "ml_prediction":    ml_prediction,
            "ml_confidence":    round(ml_confidence, 4),
            "dl_direction":     dl_direction,
            "dl_position_size": round(dl_position_size, 4),
            "news_sentiment":   news_sentiment,
            "current_price":    round(current_price, 4),
            "stop_loss":        round(stop_loss, 4),
            "take_profit":      round(take_profit, 4),
            "shares":           shares,
            "atr":              round(atr, 4),
            "reasoning":        reasoning.replace("\n", " "),
            # Outcomes filled later
            "next_day_price":   "",
            "next_day_return_pct": "",
            "week_price":       "",
            "week_return_pct":  "",
            "outcome":          "OPEN",
            "outcome_updated":  "",
        }
        self._append_row(row)
        print(f"[SignalLogger] Logged {action.upper()} signal for {ticker} (id={log_id})")
        return log_id

    def update_outcomes(self):
        """
        Fetch actual prices for every OPEN signal whose signal_date is at
        least 1 trading day in the past.  Marks each row as WIN, LOSS, or
        NEUTRAL based on whether the price moved in the predicted direction.

        WIN  — price moved in the signal direction (buy→up / sell→down)
        LOSS — price moved against the signal direction
        NEUTRAL — change within ±0.1% (flat day)
        """
        df = self._load()
        if df.empty:
            print("[SignalLogger] No signals to update.")
            return

        open_mask = df["outcome"] == "OPEN"
        open_rows = df[open_mask].copy()

        if open_rows.empty:
            print("[SignalLogger] All signals already resolved.")
            return

        updated = 0
        for idx, row in open_rows.iterrows():
            signal_date = pd.to_datetime(row["signal_date"]).date()
            # Need at least 1 trading day to have passed
            if signal_date >= date.today():
                continue

            ticker        = row["ticker"]
            entry_price   = float(row["current_price"])
            action        = row["action"]

            try:
                hist = yf.Ticker(ticker).history(
                    start=str(signal_date),
                    end=str(date.today() + timedelta(days=1)),
                    auto_adjust=True,
                )
                if len(hist) < 2:
                    continue  # not enough data yet

                next_day_price = float(hist["Close"].iloc[1])
                next_day_return = (next_day_price - entry_price) / entry_price * 100

                # Week price (5 trading days later if available)
                week_price, week_return = "", ""
                if len(hist) >= 6:
                    week_price  = float(hist["Close"].iloc[5])
                    week_return = (week_price - entry_price) / entry_price * 100

                # Determine outcome based on next-day direction vs signal
                if abs(next_day_return) < 0.1:
                    outcome = "NEUTRAL"
                elif action == "buy" and next_day_return > 0:
                    outcome = "WIN"
                elif action == "sell" and next_day_return < 0:
                    outcome = "WIN"
                else:
                    outcome = "LOSS"

                df.at[idx, "next_day_price"]      = round(next_day_price, 4)
                df.at[idx, "next_day_return_pct"] = round(next_day_return, 3)
                df.at[idx, "week_price"]          = round(week_price, 4) if week_price != "" else ""
                df.at[idx, "week_return_pct"]     = round(week_return, 3) if week_return != "" else ""
                df.at[idx, "outcome"]             = outcome
                df.at[idx, "outcome_updated"]     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                updated += 1

            except Exception as e:
                print(f"[SignalLogger] Could not update {ticker} ({row['signal_date']}): {e}")

        self._save(df)
        print(f"[SignalLogger] Updated {updated} signal(s).")

    def print_report(self):
        """
        Print a concise performance report — the key table to review
        after 3 months of demo trading.
        """
        df = self._load()
        if df.empty:
            print("[SignalLogger] No signals logged yet.")
            return

        resolved = df[df["outcome"].isin(["WIN", "LOSS", "NEUTRAL"])].copy()
        if resolved.empty:
            print("[SignalLogger] No resolved signals yet — run update_outcomes() first.")
            return

        total     = len(resolved)
        wins      = (resolved["outcome"] == "WIN").sum()
        losses    = (resolved["outcome"] == "LOSS").sum()
        neutrals  = (resolved["outcome"] == "NEUTRAL").sum()
        win_rate  = wins / total * 100 if total > 0 else 0

        print("\n" + "=" * 60)
        print("  SIGNAL LOG — PERFORMANCE REPORT")
        print("=" * 60)
        print(f"  Total resolved signals : {total}")
        print(f"  WIN   : {wins}  ({wins/total*100:.1f}%)")
        print(f"  LOSS  : {losses}  ({losses/total*100:.1f}%)")
        print(f"  NEUTRAL: {neutrals}  ({neutrals/total*100:.1f}%)")
        print(f"  Overall win rate       : {win_rate:.1f}%")

        # Accuracy by confidence tier
        print("\n  --- Win Rate by Confidence Tier ---")
        resolved["confidence"] = pd.to_numeric(resolved["confidence"], errors="coerce")
        bins   = [0.0, 0.55, 0.60, 0.65, 0.70, 1.01]
        labels = ["<55%", "55-60%", "60-65%", "65-70%", "≥70%"]
        resolved["conf_tier"] = pd.cut(resolved["confidence"], bins=bins, labels=labels, right=False)

        tier_stats = (
            resolved.groupby("conf_tier", observed=True)["outcome"]
            .apply(lambda x: {
                "count":    len(x),
                "win_rate": (x == "WIN").sum() / len(x) * 100 if len(x) > 0 else 0,
            })
        )
        print(f"  {'Tier':<10} {'Count':>6}  {'Win Rate':>10}")
        print(f"  {'-'*30}")
        for tier, stats in tier_stats.items():
            print(f"  {str(tier):<10} {stats['count']:>6}  {stats['win_rate']:>9.1f}%")

        # Per-ticker breakdown
        print("\n  --- Win Rate by Ticker ---")
        ticker_stats = (
            resolved.groupby("ticker")["outcome"]
            .apply(lambda x: {
                "count":    len(x),
                "win_rate": (x == "WIN").sum() / len(x) * 100 if len(x) > 0 else 0,
            })
        )
        print(f"  {'Ticker':<8} {'Count':>6}  {'Win Rate':>10}")
        print(f"  {'-'*30}")
        for ticker, stats in ticker_stats.items():
            print(f"  {ticker:<8} {stats['count']:>6}  {stats['win_rate']:>9.1f}%")

        print("=" * 60)
        print(f"  Full log: {self.log_path}\n")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_csv(self):
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()

    def _append_row(self, row: dict):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writerow(row)

    def _load(self) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame(columns=COLUMNS)
        return pd.read_csv(self.log_path, dtype=str)

    def _save(self, df: pd.DataFrame):
        df.to_csv(self.log_path, index=False)
