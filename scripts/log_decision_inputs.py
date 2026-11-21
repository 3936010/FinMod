"""
scripts/log_decision_inputs.py — daily logger for the decision model's inputs.

Run once per trading day after the US close (e.g. 16:30 America/New_York):

    uv run python scripts/log_decision_inputs.py              # watchlist from main.py
    uv run python scripts/log_decision_inputs.py NVDA AAPL    # specific tickers
    uv run python scripts/log_decision_inputs.py --no-llm     # skip the OpenRouter call
    uv run python scripts/log_decision_inputs.py --resolve-only

Each run:
  1. builds a DecisionInput for each ticker's last completed session and
     upserts it into data/decision_inputs/decision_input_v1.jsonl
  2. resolves labels for older records whose 3-trading-day outcome is now known.

Starting this now matters: live rows are the only clean source of LLM news
features (back-filled LLM scores are contaminated by hindsight).

Cron example (Jetson, container TZ=America/New_York):
    30 16 * * 1-5  cd /app && uv run python scripts/log_decision_inputs.py >> logs/decision.log 2>&1
"""

from __future__ import annotations

import os
import sys
import argparse
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

from scripts.feature_builder import (
    build_decision_input, compute_label, drop_incomplete_bar, _to_date, LABEL_HORIZON,
)
from utils.decision_store import DecisionStore

DEFAULT_WATCHLIST = ['QCOM', 'NVDA', 'AAPL', 'AMZN', 'AMD', 'TSCO', 'WMT']   # mirrors main.py


def resolve_labels(store: DecisionStore, now: datetime | None = None) -> int:
    """Fill label for records whose as_of + LABEL_HORIZON sessions have closed."""
    import yfinance as yf

    records = store.load()
    pending = [r for r in records if r.label.resolved_on is None
               and r.reference.close and r.reference.atr]
    if not pending:
        return 0

    resolved = 0
    for ticker in sorted({r.ticker for r in pending}):
        earliest = min(r.as_of_date for r in pending if r.ticker == ticker)
        try:
            hist = yf.Ticker(ticker).history(start=earliest, auto_adjust=True)
            hist = drop_incomplete_bar(hist, now)
        except Exception as e:
            print(f"[{ticker}] label fetch failed: {e}")
            continue
        dates = [_to_date(i) for i in hist.index]
        closes = list(hist["Close"])
        for r in (r for r in pending if r.ticker == ticker):
            as_of = date.fromisoformat(r.as_of_date)
            if as_of not in dates:
                continue
            i = dates.index(as_of)
            if i + LABEL_HORIZON >= len(dates):
                continue                      # outcome not known yet
            # Use the stored as_of close so later dividend adjustments don't shift the base
            r.label = compute_label(r.reference.close, r.reference.atr, float(closes[i + LABEL_HORIZON]))
            resolved += 1

    if resolved:
        store.replace_all(records)
    return resolved


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="*", help="defaults to the main.py watchlist")
    p.add_argument("--no-llm", action="store_true", help="skip the LLM news extraction")
    p.add_argument("--no-dl", action="store_true", help="skip the GRU model")
    p.add_argument("--resolve-only", action="store_true", help="only fill in labels")
    args = p.parse_args(argv)

    load_dotenv()
    store = DecisionStore()
    tickers = [t.upper() for t in args.tickers] or DEFAULT_WATCHLIST

    if not args.resolve_only:
        built = []
        for t in tickers:
            try:
                rec = build_decision_input(t, use_llm=not args.no_llm, use_dl=not args.no_dl)
                built.append(rec)
                fv = rec.feature_vector()
                filled = sum(v is not None for v in fv.values())
                print(f"[{t}] as_of={rec.as_of_date} features={filled}/{len(fv)} "
                      f"p_up_ml={rec.ml.p_up_ensemble} news_d1={rec.news.d1.n_articles} "
                      f"errors={rec.provenance.errors or '-'}")
            except Exception as e:
                print(f"[{t}] FAILED: {e}")
        if built:
            store.upsert(built)
            print(f"Saved {len(built)} record(s) to {store.path}")

    n = resolve_labels(store)
    print(f"Resolved {n} label(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
