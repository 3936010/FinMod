"""
scripts/feature_builder.py — builds one DecisionInput record per (ticker, day).

This is the single code path that turns market data, model outputs and news
into the decision model's input (utils/data_models.DecisionInput). The live
daily logger uses it now; the historical dataset builder will use it later, so
training rows and live rows are built the same way.

Point-in-time rules applied here:
  - The as_of day is the last *completed* session. If the script runs before
    16:00 ET, today's partial bar is dropped.
  - News counts toward day t only if published at or before 16:00 ET on t;
    after-hours news belongs to the next session.
  - Fundamentals come from today's yfinance snapshot, so they are marked
    fundamentals_point_in_time=False (fine for live rows, not for back-fill).

Pure helpers (no network) are kept separate from fetchers so they can be tested.
"""

from __future__ import annotations

import os
import sys
import math
import json
from datetime import datetime, date, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from utils.data_models import (
    DecisionInput, TechnicalBlock, MarketBlock, MLBlock, DLBlock, NewsWindow,
    NewsBlock, NewsLLMBlock, EventsBlock, FundamentalsBlock, CalendarBlock,
    ReferenceBlock, LabelBlock, NewsLLMExtraction,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_MODELS_DIR = PROJECT_ROOT / "models" / "ml_predictors"
DL_MODEL_DIR = PROJECT_ROOT / "models" / "alpha_gru_v1"

ET = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)
LABEL_HORIZON = 3          # trading days, same as StockPredictor's target
LABEL_ATR_MULT = 0.5       # |move| < 0.5 ATR = noise (same as StockPredictor)
MAX_LLM_ARTICLES = 25
EARNINGS_WINDOW_DAYS = 5

SENTIMENT_ORDINAL = {"bearish": -1, "neutral": 0, "bullish": 1}
LLM_EVENT_KEYS = (
    "earnings", "guidance", "analyst_action", "product",
    "regulatory_legal", "m_and_a", "macro", "management",
)


# =============================================================================
#  Pure helpers
# =============================================================================

def _f(value) -> float | None:
    """float, or None for NaN / inf / missing."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def drop_incomplete_bar(hist: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Drop today's bar if the US session hasn't closed yet (yfinance returns the
    in-progress day during market hours, which would leak intraday prices into a
    row meant to be "as of the close")."""
    if hist.empty:
        return hist
    now_et = (now or datetime.now(timezone.utc)).astimezone(ET)
    last = pd.Timestamp(hist.index[-1])
    last_date = (last.tz_convert(ET) if last.tzinfo else last).date()
    if last_date == now_et.date() and now_et.time() < MARKET_CLOSE:
        return hist.iloc[:-1]
    return hist


def close_et(d: date) -> datetime:
    return datetime.combine(d, MARKET_CLOSE, tzinfo=ET)


def session_bounds(trading_dates: list[date], as_of: date, n_sessions: int) -> tuple[datetime, datetime]:
    """(start, end] window covering the last n_sessions sessions ending at the
    as_of close. start = close of the session n_sessions before as_of, so news
    after that close (incl. weekends/holidays/after-hours) is included."""
    dates = sorted(d for d in trading_dates if d <= as_of)
    if not dates or dates[-1] != as_of:
        raise ValueError(f"{as_of} is not in the trading calendar")
    i = len(dates) - 1
    start_idx = i - n_sessions
    if start_idx >= 0:
        start = close_et(dates[start_idx])
    else:  # not enough history: fall back to calendar days
        start = close_et(as_of - timedelta(days=n_sessions + 2))
    return start, close_et(as_of)


def _article_sentiment(article: dict, ticker: str) -> str | None:
    for ins in article.get("insights") or []:
        if (ins.get("ticker") or "").upper() == ticker.upper():
            s = (ins.get("sentiment") or "").lower()
            return s if s in ("positive", "negative", "neutral") else None
    return None


def aggregate_news_window(articles: Iterable[dict], ticker: str,
                          start: datetime, end: datetime) -> NewsWindow:
    """Aggregate articles with start < published_utc <= end that mention ticker."""
    in_window = [
        a for a in articles
        if start < a["published_utc"] <= end
        and ticker.upper() in [t.upper() for t in (a.get("tickers") or [ticker])]
    ]
    if not in_window:
        return NewsWindow(n_articles=0, n_scored=0, n_publishers=0)

    sentiments = [s for s in (_article_sentiment(a, ticker) for a in in_window) if s]
    n_scored = len(sentiments)
    window = NewsWindow(
        n_articles=len(in_window),
        n_scored=n_scored,
        n_publishers=len({a.get("publisher") for a in in_window if a.get("publisher")}),
    )
    if n_scored:
        pos = sentiments.count("positive") / n_scored
        neg = sentiments.count("negative") / n_scored
        window.pos_ratio, window.neg_ratio, window.net_sentiment = pos, neg, pos - neg
    return window


def build_news_block(articles: list[dict], ticker: str,
                     trading_dates: list[date], as_of: date) -> NewsBlock:
    d1 = aggregate_news_window(articles, ticker, *session_bounds(trading_dates, as_of, 1))
    d5 = aggregate_news_window(articles, ticker, *session_bounds(trading_dates, as_of, 5))
    block = NewsBlock(d1=d1, d5=d5)
    if d5.n_articles:
        block.volume_ratio_1d_vs_5d = d1.n_articles / (d5.n_articles / 5)
    if d1.net_sentiment is not None and d5.net_sentiment is not None:
        block.sentiment_change_1d_vs_5d = d1.net_sentiment - d5.net_sentiment
    return block


def articles_in_session(articles: list[dict], ticker: str,
                        trading_dates: list[date], as_of: date) -> list[dict]:
    """Articles in the d1 window, newest first (input for headlines / LLM)."""
    start, end = session_bounds(trading_dates, as_of, 1)
    picked = [a for a in articles if start < a["published_utc"] <= end
              and ticker.upper() in [t.upper() for t in (a.get("tickers") or [ticker])]]
    return sorted(picked, key=lambda a: a["published_utc"], reverse=True)


def technical_block(hist: pd.DataFrame) -> TechnicalBlock:
    """hist must already contain StockPredictor._calculate_features columns."""
    row, close = hist.iloc[-1], hist["Close"]
    c = _f(row["Close"])
    prev_c = _f(close.iloc[-2]) if len(close) > 1 else None
    pct = lambda v: (_f(v) / c * 100) if (c and _f(v) is not None) else None
    ret_5d = _f(close.iloc[-1] / close.iloc[-6] - 1) if len(close) > 5 else None
    return TechnicalBlock(
        Price_to_MA5=_f(row["Price_to_MA5"]),
        Price_to_MA20=_f(row["Price_to_MA20"]),
        Price_to_MA50=_f(row["Price_to_MA50"]),
        RSI=_f(row["RSI"]),
        MACD_pct=pct(row["MACD"]),
        MACD_Signal_pct=pct(row["MACD_Signal"]),
        MACD_Hist_pct=pct(row["MACD_Hist"]),
        Stochastic_K=_f(row["Stochastic_K"]),
        Volatility=_f(row["Volatility"]),
        ATR_pct=pct(row["ATR"]),
        BB_Width=_f(row["BB_Width"]),
        BB_Position=_f(row["BB_Position"]),
        VolumeSpike=_f(row["VolumeSpike"]),
        Gap_pct=(_f(row["Gap"]) / prev_c * 100) if (prev_c and _f(row["Gap"]) is not None) else None,
        Return_1d=_f(row["Return"]),
        Return_Lag1=_f(row["Return_Lag1"]),
        Return_5d=ret_5d,
    )


def market_block(hist: pd.DataFrame, spy_close: pd.Series | None) -> MarketBlock:
    """hist must already contain StockPredictor._add_market_features columns."""
    row = hist.iloc[-1]
    vix = hist["VIX"] if "VIX" in hist else None
    block = MarketBlock(
        Beta_90d=_f(row.get("Beta_90d")),
        Alpha_90d=_f(row.get("Alpha_90d")),
        VolAdj_Return=_f(row.get("VolAdj_Return")),
        VIX=_f(row.get("VIX")),
        VIX_chg_5d=_f(vix.iloc[-1] - vix.iloc[-6]) if vix is not None and len(vix) > 5 else None,
        SPY_above_200MA=_f(row.get("SPY_above_200MA")),
        SPY_ATR_pct=_f(row.get("SPY_ATR_pct")),
    )
    if spy_close is not None:
        as_of = _to_date(hist.index[-1])
        s = spy_close[[_to_date(i) <= as_of for i in spy_close.index]]
        if len(s) > 5:
            block.SPY_ret_5d = _f(s.iloc[-1] / s.iloc[-6] - 1)
    return block


def ml_block(payload: dict, hist: pd.DataFrame) -> MLBlock:
    """Run a persisted StockPredictor ensemble (joblib payload) on the last row."""
    X = hist[payload["features"]].iloc[[-1]]
    if X.isna().any(axis=None):
        raise ValueError(f"NaN ML features: {list(X.columns[X.isna().iloc[0]])}")
    Xs = payload["scaler"].transform(X)
    probs = {name: float(payload[f"model_{name}"].predict_proba(Xs)[0][1])
             for name in ("RF", "XGB", "LR", "GB")}
    w = payload.get("model_weights") or {k: 0.25 for k in probs}
    p_ens = sum(probs[k] * w.get(k, 0.25) for k in probs) / (sum(w.get(k, 0.25) for k in probs) or 1)
    up = p_ens > 0.5
    agree = sum((p > 0.5) == up for p in probs.values()) / len(probs)
    return MLBlock(p_up_rf=probs["RF"], p_up_xgb=probs["XGB"], p_up_lr=probs["LR"],
                   p_up_gb=probs["GB"], p_up_ensemble=p_ens, models_agree_frac=agree)


def dl_block_from_return(pred_log_return: float, sizing_scale: float = 0.01) -> DLBlock:
    """Same mapping as AlphaPredictor.predict_next_price()."""
    pct_change = (math.exp(pred_log_return) - 1) * 100
    direction = 1 if pct_change > 0.05 else -1 if pct_change < -0.05 else 0
    return DLBlock(pred_log_return=pred_log_return,
                   position_size=float(np.tanh(pred_log_return / sizing_scale)),
                   direction=direction)


def events_block(earnings_dates: list[date], as_of: date) -> EventsBlock:
    if not earnings_dates:
        return EventsBlock()
    future = [d for d in earnings_dates if d > as_of]
    past = [d for d in earnings_dates if d <= as_of]
    to_next = (min(future) - as_of).days if future else None
    return EventsBlock(
        days_to_earnings=to_next,
        days_since_earnings=(as_of - max(past)).days if past else None,
        is_earnings_window=(to_next is not None and to_next <= EARNINGS_WINDOW_DAYS),
    )


def fundamentals_block(info: dict) -> FundamentalsBlock:
    mcap = _f(info.get("marketCap"))
    de = _f(info.get("debtToEquity"))
    return FundamentalsBlock(
        log_market_cap=math.log(mcap) if mcap and mcap > 0 else None,
        trailingPE=_f(info.get("trailingPE")),
        forwardPE=_f(info.get("forwardPE")),
        # yfinance reports D/E in percent (e.g. 150.0 means 1.5x)
        debt_to_equity=de / 100 if de is not None else None,
        operating_margin=_f(info.get("operatingMargins")),
        earnings_growth=_f(info.get("earningsGrowth")),
        revenue_growth=_f(info.get("revenueGrowth")),
    )


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def calendar_block(as_of: date) -> CalendarBlock:
    opex = _third_friday(as_of.year, as_of.month)
    return CalendarBlock(day_of_week=as_of.weekday(), month=as_of.month,
                         is_opex_week=as_of.isocalendar()[:2] == opex.isocalendar()[:2])


def llm_block(ext: NewsLLMExtraction) -> NewsLLMBlock:
    ev = {k.lower(): bool(v) for k, v in (ext.events or {}).items()}
    return NewsLLMBlock(
        short_term=SENTIMENT_ORDINAL[ext.short_term_sentiment],
        short_term_conf=ext.short_term_confidence,
        long_term=SENTIMENT_ORDINAL[ext.long_term_sentiment],
        long_term_conf=ext.long_term_confidence,
        **{f"ev_{k}": ev.get(k, False) for k in LLM_EVENT_KEYS},
    )


def compute_label(close: float, atr: float, close_fwd: float,
                  atr_mult: float = LABEL_ATR_MULT) -> LabelBlock:
    """StockPredictor's target: 1 if move > 0.5 ATR, 0 if < -0.5 ATR, else noise."""
    move = close_fwd - close
    noise = abs(move) <= atr_mult * atr
    return LabelBlock(
        close_fwd_3d=close_fwd,
        fwd_ret_3d=close_fwd / close - 1,
        fwd_move_atr=move / atr if atr else None,
        y=None if noise else int(move > 0),
        is_noise=noise,
        resolved_on=date.today().isoformat(),
    )


def _to_date(ts) -> date:
    ts = pd.Timestamp(ts)
    return (ts.tz_convert(ET) if ts.tzinfo else ts).date()


# =============================================================================
#  Fetchers (network)
# =============================================================================

@lru_cache(maxsize=1)
def _spy_close() -> pd.Series | None:
    import yfinance as yf
    try:
        return yf.Ticker("SPY").history(period="2y")["Close"]
    except Exception as e:
        print(f"Warning: SPY download failed: {e}")
        return None


def load_feature_frame(ticker: str, now: datetime | None = None, period: str = "3y") -> pd.DataFrame:
    """OHLCV + StockPredictor technical/market features, last completed session last."""
    import yfinance as yf
    from scripts.ml_market_movement import StockPredictor

    hist = yf.Ticker(ticker).history(period=period)
    hist = hist.drop(columns=[c for c in ("Dividends", "Stock Splits", "Capital Gains") if c in hist])
    hist = drop_incomplete_bar(hist, now)
    if len(hist) < 260:
        raise ValueError(f"{ticker}: only {len(hist)} rows of price history")

    sp = StockPredictor(ticker)          # constructor only sets feature lists, no I/O
    hist = sp._calculate_features(hist)
    hist = sp._add_market_features(hist)
    hist.replace([np.inf, -np.inf], np.nan, inplace=True)
    return hist.ffill()                   # same cleaning as StockPredictor.predict_next_day


def fetch_polygon_news(ticker: str, start_utc: datetime, end_utc: datetime) -> list[dict]:
    """Ticker news from Polygon/Massive as plain dicts with aware datetimes.
    Uses RESTClient directly; utils.polygon_utils.get_news also opens a DB
    connection and drops the whole batch if one article fails validation."""
    from polygon import RESTClient

    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY is not set")
    client = RESTClient(api_key=api_key, retries=5)
    out = []
    for n in client.list_ticker_news(
        ticker=ticker,
        published_utc_gte=start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        published_utc_lte=end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        order="desc", sort="published_utc", limit=1000,
    ):
        published = getattr(n, "published_utc", None)
        if not published:
            continue
        publisher = getattr(n, "publisher", None)
        out.append({
            "id": getattr(n, "id", None),
            "published_utc": pd.Timestamp(published).to_pydatetime().astimezone(timezone.utc),
            "title": getattr(n, "title", "") or "",
            "description": getattr(n, "description", "") or "",
            "publisher": getattr(publisher, "name", None) if publisher else None,
            "tickers": list(getattr(n, "tickers", None) or []),
            "insights": [
                {"ticker": getattr(i, "ticker", None), "sentiment": getattr(i, "sentiment", None),
                 "sentiment_reasoning": getattr(i, "sentiment_reasoning", None)}
                for i in (getattr(n, "insights", None) or [])
            ],
        })
    return out


def fetch_earnings_dates(ticker: str) -> list[date]:
    import yfinance as yf
    df = yf.Ticker(ticker).get_earnings_dates(limit=12)
    return sorted({_to_date(ts) for ts in df.index}) if df is not None else []


def fetch_info(ticker: str) -> dict:
    import yfinance as yf
    return yf.Ticker(ticker).info or {}


@lru_cache(maxsize=1)
def _alpha_predictor():
    """GRU predictor, or None if torch / weights are unavailable (e.g. on the Nano
    before the ONNX export, or when models/alpha_gru_v1 only has the scaler)."""
    try:
        from scripts.dl_alpha_production import AlphaPredictor
        return AlphaPredictor(DL_MODEL_DIR)
    except Exception as e:
        print(f"DL Alpha unavailable: {e}")
        return None


def llm_settings() -> tuple[str, str]:
    provider = os.getenv("DECISION_LLM_PROVIDER", "OpenRouter")
    if provider == "OpenRouter":
        from utils.llm.llm_models import OPENROUTER_DEFAULT_MODEL
        return provider, os.getenv("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL)
    return provider, os.getenv("DECISION_LLM_MODEL", "gemini-2.5-flash")


def run_llm_extraction(ticker: str, as_of: date, articles: list[dict]) -> NewsLLMExtraction | None:
    from utils.llm.prompt import prompts
    from utils.llm.api_call import call_llm

    provider, model = llm_settings()
    payload = [
        {"published_utc": a["published_utc"].strftime("%Y-%m-%d %H:%M UTC"),
         "publisher": a.get("publisher"), "title": a["title"],
         "description": (a.get("description") or "")[:400]}
        for a in articles[:MAX_LLM_ARTICLES]
    ]
    prompt = prompts.news_features_template.invoke({
        "ticker": ticker, "as_of_date": as_of.isoformat(),
        "articles": json.dumps(payload, indent=1),
    })
    # default_factory=None-returning so a failed call leaves the block empty
    # instead of call_llm's placeholder values ("Error...", 0.0) becoming features.
    return call_llm(prompt, model, provider, NewsLLMExtraction,
                    max_retries=3, default_factory=lambda: None)


# =============================================================================
#  Builder
# =============================================================================

def build_decision_input(ticker: str, use_llm: bool = True, use_dl: bool = True,
                         now: datetime | None = None) -> DecisionInput:
    """Build the live DecisionInput for the last completed session of `ticker`.
    Each block is fetched independently; a failing source leaves its block empty
    and is noted in provenance.errors rather than aborting the record."""
    now = now or datetime.now(timezone.utc)
    hist = load_feature_frame(ticker, now)
    as_of = _to_date(hist.index[-1])
    trading_dates = [_to_date(i) for i in hist.index]

    rec = DecisionInput(ticker=ticker, as_of_date=as_of.isoformat(),
                        generated_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    prov = rec.provenance
    errors = prov.errors

    rec.technical = technical_block(hist)
    rec.market = market_block(hist, _spy_close())
    rec.calendar = calendar_block(as_of)
    rec.reference = ReferenceBlock(close=_f(hist["Close"].iloc[-1]), atr=_f(hist["ATR"].iloc[-1]))

    # ── ML ensemble (persisted by main.py) ────────────────────────────────────
    model_path = ML_MODELS_DIR / f"{ticker}_predictor.pkl"
    if model_path.exists():
        try:
            import joblib
            rec.ml = ml_block(joblib.load(model_path), hist)
            trained = datetime.fromtimestamp(model_path.stat().st_mtime).date()
            prov.ml_model_path = str(model_path.relative_to(PROJECT_ROOT))
            prov.ml_model_trained_at = trained.isoformat()
            # training drops the last LABEL_HORIZON rows, so as_of >= trained date is unseen
            prov.ml_out_of_sample = as_of >= trained
        except Exception as e:
            errors.append(f"ml: {e}")
    else:
        errors.append(f"ml: no model at {model_path.relative_to(PROJECT_ROOT)} (run main.py)")

    # ── DL GRU (optional) ─────────────────────────────────────────────────────
    if use_dl and (ap := _alpha_predictor()) is not None:
        try:
            r = ap.predict_next_return(hist[["Open", "High", "Low", "Close", "Volume"]])
            rec.dl = dl_block_from_return(r, ap.config.get("sizing_scale", 0.01))
            prov.dl_available, prov.dl_in_sample = True, False
        except Exception as e:
            errors.append(f"dl: {e}")

    # ── News (Polygon insights) + optional LLM extraction ─────────────────────
    try:
        start, end = session_bounds(trading_dates, as_of, 5)
        articles = fetch_polygon_news(ticker, start.astimezone(timezone.utc), end.astimezone(timezone.utc))
        rec.news = build_news_block(articles, ticker, trading_dates, as_of)
        prov.news_source = "polygon"
        session_articles = articles_in_session(articles, ticker, trading_dates, as_of)
        rec.text_for_explainer.top_headlines = [a["title"] for a in session_articles[:5]]
        if use_llm and session_articles:
            provider, model = llm_settings()
            ext = run_llm_extraction(ticker, as_of, session_articles)
            if ext is not None:
                rec.news.llm = llm_block(ext)
                rec.text_for_explainer.llm_reasoning = ext.reasoning
                prov.llm_model, prov.llm_source = f"{provider}:{model}", "live"
            else:
                errors.append("llm: extraction failed")
    except Exception as e:
        errors.append(f"news: {e}")

    # ── Events + fundamentals (yfinance) ──────────────────────────────────────
    try:
        rec.events = events_block(fetch_earnings_dates(ticker), as_of)
    except Exception as e:
        errors.append(f"events: {e}")
    try:
        rec.fundamentals = fundamentals_block(fetch_info(ticker))
        prov.fundamentals_point_in_time = False
    except Exception as e:
        errors.append(f"fundamentals: {e}")

    return rec


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(build_decision_input(t, use_llm="--no-llm" not in sys.argv).model_dump_json(indent=2))
