"""Offline tests for the DecisionInput schema, feature builder helpers and store."""

import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from utils.data_models import DecisionInput, NewsLLMExtraction, DECISION_FEATURE_BLOCKS
from utils.decision_store import DecisionStore
from scripts.feature_builder import (
    ET, drop_incomplete_bar, session_bounds, aggregate_news_window, build_news_block,
    compute_label, events_block, fundamentals_block, calendar_block, llm_block,
    dl_block_from_return, technical_block,
)


def utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def article(ts, sentiment=None, ticker="NVDA", publisher="Reuters"):
    return {
        "published_utc": ts, "title": f"news {ts}", "description": "", "publisher": publisher,
        "tickers": [ticker],
        "insights": [{"ticker": ticker, "sentiment": sentiment}] if sentiment else [],
    }


# Thu 2026-11-19, Fri 2026-11-20, Mon 2026-11-23 ... (plus earlier sessions)
TRADING = [date(2026, 11, d) for d in (13, 16, 17, 18, 19, 20, 23)]


class TestSchema(unittest.TestCase):
    def test_feature_names_stable_and_numeric(self):
        names = DecisionInput.feature_names()
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(n.split(".")[0] in DECISION_FEATURE_BLOCKS for n in names))
        # non-feature blocks must not leak into the vector
        self.assertFalse(any(n.startswith(("label.", "reference.", "provenance.")) for n in names))
        self.assertIn("news.d1.net_sentiment", names)
        self.assertIn("news.llm.ev_earnings", names)
        self.assertEqual(DecisionInput.feature_hash(), DecisionInput.feature_hash())

    def test_missing_is_none_and_bools_become_floats(self):
        r = DecisionInput(ticker="X", as_of_date="2026-11-20", generated_at="")
        r.events.is_earnings_window = True
        fv = r.feature_vector()
        self.assertIsNone(fv["technical.RSI"])
        self.assertEqual(fv["events.is_earnings_window"], 1.0)

    def test_extra_fields_rejected(self):
        with self.assertRaises(Exception):
            DecisionInput(ticker="X", as_of_date="2026-11-20", generated_at="", surprise=1)


class TestPointInTime(unittest.TestCase):
    def test_drop_incomplete_bar(self):
        idx = pd.DatetimeIndex([pd.Timestamp("2026-11-19", tz=ET), pd.Timestamp("2026-11-20", tz=ET)])
        hist = pd.DataFrame({"Close": [1.0, 2.0]}, index=idx)
        during = datetime(2026, 11, 20, 15, 0, tzinfo=ET)
        after = datetime(2026, 11, 20, 16, 5, tzinfo=ET)
        self.assertEqual(len(drop_incomplete_bar(hist, during)), 1)
        self.assertEqual(len(drop_incomplete_bar(hist, after)), 2)

    def test_session_bounds_monday_includes_weekend(self):
        start, end = session_bounds(TRADING, date(2026, 11, 23), 1)
        self.assertEqual(start, datetime(2026, 11, 20, 16, 0, tzinfo=ET))
        self.assertEqual(end, datetime(2026, 11, 23, 16, 0, tzinfo=ET))

    def test_after_close_news_goes_to_next_session(self):
        # 16:00 ET on 2026-11-19 = 21:00 UTC (EST, UTC-5)
        before_close = article(utc(2026, 11, 19, 20, 30), "positive")
        after_close = article(utc(2026, 11, 19, 21, 30), "negative")   # e.g. earnings after the bell
        arts = [before_close, after_close]
        thu = aggregate_news_window(arts, "NVDA", *session_bounds(TRADING, date(2026, 11, 19), 1))
        fri = aggregate_news_window(arts, "NVDA", *session_bounds(TRADING, date(2026, 11, 20), 1))
        self.assertEqual((thu.n_articles, thu.net_sentiment), (1, 1.0))
        self.assertEqual((fri.n_articles, fri.net_sentiment), (1, -1.0))

    def test_news_block_aggregates(self):
        arts = [
            article(utc(2026, 11, 20, 14), "positive"),
            article(utc(2026, 11, 20, 15), "neutral", publisher="CNBC"),
            article(utc(2026, 11, 20, 16)),                       # no insight → unscored
            article(utc(2026, 11, 18, 15), "negative"),
            article(utc(2026, 11, 20, 15), "positive", ticker="AAPL"),  # other ticker
        ]
        b = build_news_block(arts, "NVDA", TRADING, date(2026, 11, 20))
        self.assertEqual((b.d1.n_articles, b.d1.n_scored, b.d1.n_publishers), (3, 2, 2))
        self.assertAlmostEqual(b.d1.net_sentiment, 0.5)
        self.assertEqual(b.d5.n_articles, 4)
        self.assertAlmostEqual(b.d5.net_sentiment, 0.0)   # 1 pos, 1 neg, 1 neutral
        self.assertAlmostEqual(b.volume_ratio_1d_vs_5d, 3 / (4 / 5))
        self.assertAlmostEqual(b.sentiment_change_1d_vs_5d, 0.5)

    def test_no_news_is_zero_count_not_missing_sentiment_zero(self):
        b = build_news_block([], "NVDA", TRADING, date(2026, 11, 20))
        self.assertEqual(b.d1.n_articles, 0)
        self.assertIsNone(b.d1.net_sentiment)


class TestBlocks(unittest.TestCase):
    def test_label(self):
        up = compute_label(100.0, 4.0, 103.0)       # +0.75 ATR
        down = compute_label(100.0, 4.0, 97.0)
        noise = compute_label(100.0, 4.0, 101.0)    # +0.25 ATR
        self.assertEqual((up.y, up.is_noise), (1, False))
        self.assertEqual((down.y, down.is_noise), (0, False))
        self.assertEqual((noise.y, noise.is_noise), (None, True))
        self.assertAlmostEqual(up.fwd_move_atr, 0.75)

    def test_events(self):
        dates = [date(2026, 8, 26), date(2026, 11, 24)]
        e = events_block(dates, date(2026, 11, 20))
        self.assertEqual((e.days_to_earnings, e.days_since_earnings, e.is_earnings_window), (4, 86, True))
        self.assertIsNone(events_block([], date(2026, 11, 20)).days_to_earnings)

    def test_fundamentals_de_percent_to_ratio(self):
        f = fundamentals_block({"debtToEquity": 150.0, "marketCap": 1e12, "trailingPE": "Infinity"})
        self.assertAlmostEqual(f.debt_to_equity, 1.5)
        self.assertIsNone(f.trailingPE)
        self.assertAlmostEqual(f.log_market_cap, np.log(1e12))

    def test_calendar_opex(self):
        self.assertTrue(calendar_block(date(2026, 11, 20)).is_opex_week)   # 3rd Friday of Nov 2026
        self.assertFalse(calendar_block(date(2026, 11, 23)).is_opex_week)

    def test_llm_and_dl(self):
        ext = NewsLLMExtraction(short_term_sentiment="bullish", short_term_confidence=0.7,
                                long_term_sentiment="neutral", long_term_confidence=0.5,
                                events={"earnings": True, "unknown_key": True})
        b = llm_block(ext)
        self.assertEqual((b.short_term, b.long_term, b.ev_earnings, b.ev_macro), (1, 0, True, False))
        self.assertEqual(dl_block_from_return(0.002).direction, 1)
        self.assertEqual(dl_block_from_return(0.0001).direction, 0)

    def test_technical_block_normalises_price_scale(self):
        n = 80
        close = pd.Series(np.linspace(100, 120, n))
        cols = ["Price_to_MA5", "Price_to_MA20", "Price_to_MA50", "RSI", "MACD", "MACD_Signal",
                "MACD_Hist", "Stochastic_K", "Volatility", "ATR", "BB_Width", "BB_Position",
                "VolumeSpike", "Gap", "Return", "Return_Lag1"]
        df = pd.DataFrame({c: 1.0 for c in cols}, index=range(n))
        df["Close"] = close
        df["ATR"] = 2.4
        t = technical_block(df)
        self.assertAlmostEqual(t.ATR_pct, 2.0)               # 2.4 / 120 * 100
        self.assertAlmostEqual(t.Return_5d, 120 / close.iloc[-6] - 1)


class TestStore(unittest.TestCase):
    def test_upsert_replaces_and_keeps_resolved_label(self):
        with tempfile.TemporaryDirectory() as d:
            store = DecisionStore(Path(d) / "x.jsonl")
            r = DecisionInput(ticker="NVDA", as_of_date="2026-11-20", generated_at="a")
            r.label = compute_label(100, 4, 103)
            store.upsert([r, DecisionInput(ticker="AAPL", as_of_date="2026-11-20", generated_at="a")])
            again = DecisionInput(ticker="NVDA", as_of_date="2026-11-20", generated_at="b")
            store.upsert([again])
            recs = {x.key: x for x in store.load()}
            self.assertEqual(len(recs), 2)
            self.assertEqual(recs[("NVDA", "2026-11-20")].generated_at, "b")
            self.assertEqual(recs[("NVDA", "2026-11-20")].label.y, 1)
            frame = store.to_frame()
            self.assertIn("feature.technical.RSI", frame.columns)
            self.assertIn("label.y", frame.columns)


if __name__ == "__main__":
    unittest.main()
