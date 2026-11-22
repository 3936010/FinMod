## This file is copied from the repo https://github.com/virattt/ai-hedge-fund
## Original author: Virat Singh https://github.com/virattt
## Modified by Zexi Chen https://github.com/Chen-zexi

from typing import Literal
from pydantic import BaseModel, Field


class Price(BaseModel):
    open: float
    close: float
    high: float
    low: float
    volume: int
    time: str


class PriceResponse(BaseModel):
    ticker: str
    prices: list[Price]


class FinancialMetrics(BaseModel):
    ticker: str
    report_period: str
    period: str
    currency: str | None = None
    
    # Valuation & Comparison
    market_cap: float | None = None
    earnings_growth: float | None = None
    fcf_yield: float | None = None
    
    # Profitability
    return_on_equity: float | None = None
    return_on_invested_capital: float | None = None
    operating_margin: float | None = None
    gross_margin: float | None = None
    earnings_per_share: float | None = None
    basic_eps: float | None = None
    diluted_eps: float | None = None
    net_income: float | None = None
    revenue: float | None = None
    
    # Financial Health & Risk
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    total_assets: float | None = None
    total_liabilities: float | None = None
    
    # Share Structure & Dividends
    book_value_per_share: float | None = None
    outstanding_shares: float | None = None
    dividends: float | None = None
    special_distributions: float | None = None
    
    # Other
    free_cash_flow: float | None = None
    operating_cash_flow: float | None = None
    capital_expenditure: float | None = None


class FinancialMetricsResponse(BaseModel):
    financial_metrics: list[FinancialMetrics]


class LineItem(BaseModel):
    ticker: str
    report_period: str
    period: str
    currency: str

    # Allow additional fields dynamically
    model_config = {"extra": "allow"}


class LineItemResponse(BaseModel):
    search_results: list[LineItem]


class InsiderTrade(BaseModel):
    ticker: str
    issuer: str | None
    name: str | None
    title: str | None
    is_board_director: bool | None
    transaction_date: str | None
    transaction_shares: float | None
    transaction_price_per_share: float | None
    transaction_value: float | None
    shares_owned_before_transaction: float | None
    shares_owned_after_transaction: float | None
    security_title: str | None
    filing_date: str


class InsiderTradeResponse(BaseModel):
    insider_trades: list[InsiderTrade]


# polygon news

class insights(BaseModel):
    ticker: str
    sentiment: str
    sentiment_reasoning: str
    
    model_config = {
        "extra": "ignore",
    }
    
    
class CompanyNews(BaseModel):
    article_url: str
    author: str
    description: str
    id: str
    insights: list[insights]
    keywords: list[str]
    published_utc: str
    tickers: list[str]
    title: str
    publisher: str  
    
    model_config = {
        "extra": "ignore",
        "arbitrary_types_allowed": True,
    }
    
class CompanyNewsResponse(BaseModel):
    count: int
    results: list[CompanyNews]
    
    model_config = {
        "extra": "ignore",
        "arbitrary_types_allowed": True,
    }


class RetailActivity(BaseModel):
    ticker: str
    date: str
    activity: float
    sentiment: int
    
class RetailActivityResponse(BaseModel):
    results: list[RetailActivity]


class ShortTermSentiment(BaseModel):
    short_term_sentiment: str = Field(alias="short_term_sentiment")
    short_term_confidence: float = Field(alias="short_term_confidence")
    short_term_reasoning: str = Field(alias="short_term_reasoning")
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore" 
    }
    
class LongTermSentiment(BaseModel):
    long_term_sentiment: str = Field(alias="long_term_sentiment")
    long_term_confidence: float = Field(alias="long_term_confidence")
    long_term_reasoning: str = Field(alias="long_term_reasoning")
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore" 
    }
    
class KeyEventOutlook(BaseModel):
    event: str = Field(alias="event")
    event_date: str = Field(alias="event_date")
    event_type: str = Field(alias="event_type")
    event_impact: str = Field(alias="event_impact")
    event_reasoning: str = Field(alias="event_reasoning")
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore" 
    }
    
class NewsSentiment(BaseModel):
    short_term_sentiment: ShortTermSentiment = Field(alias="short_term_sentiment")
    long_term_sentiment: LongTermSentiment = Field(alias="long_term_sentiment")
    key_events_outlook: list[KeyEventOutlook] = Field(alias="key_events_outlook")

    model_config = {
        "populate_by_name": True,
        "extra": "ignore" 
    }
    
class NewsSentimentByDate(BaseModel):
    summary: str = Field(alias="Summary")
    sentiment: str = Field(alias="Sentiment")
    reasoning: str = Field(alias="Reasoning")
    date: str = Field(alias="Date")
    stock_price_movement: str = Field(alias="Stock Price Movement")
    
    model_config = {
        "populate_by_name": True,
        "extra": "ignore" 
    }

class Position(BaseModel):
    cash: float = 0.0
    shares: int = 0
    ticker: str


class Portfolio(BaseModel):
    positions: dict[str, Position]  # ticker -> Position mapping
    total_cash: float = 0.0


class AnalystSignal(BaseModel):
    signal: str | None = None
    confidence: float | None = None
    reasoning: dict | str | None = None
    max_position_size: float | None = None  # For risk management signals


class TickerAnalysis(BaseModel):
    ticker: str
    analyst_signals: dict[str, AnalystSignal]  # agent_name -> signal mapping


class AgentStateData(BaseModel):
    tickers: list[str]
    portfolio: Portfolio
    start_date: str
    end_date: str
    ticker_analyses: dict[str, TickerAnalysis]  # ticker -> analysis mapping


class AgentStateMetadata(BaseModel):
    show_reasoning: bool = False
    model_config = {"extra": "allow"}

class MarketPrediction(BaseModel):
    ticker: str
    # "hold" is listed first so create_default_response() picks it as the safe
    # fallback when the LLM fails (it uses __args__[0] for Literal fields).
    action: Literal["hold", "buy", "sell"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str



# =============================================================================
#  Decision model — input / output schema (v1)
#
#  One DecisionInput record per (ticker, as_of_date). as_of_date is the trading
#  day whose 16:00 ET close is the last information used. Every feature must be
#  computable from data available at that moment (point-in-time).
#
#  Rules:
#    - Feature leaves are numeric (float / int / bool) or None.
#    - None means "missing" (e.g. no news), never "zero". Tree models handle NaN.
#    - Categories are encoded as ordinals (bearish=-1, neutral=0, bullish=1).
#    - Non-feature data (prices for labelling, provenance, text) lives in
#      blocks that feature_vector() ignores.
#    - Adding/removing/renaming a feature requires a new schema_version.
# =============================================================================

DECISION_SCHEMA_VERSION = "decision_input.v1"

# Blocks flattened into the model's feature vector, in this order.
DECISION_FEATURE_BLOCKS = (
    "technical", "market", "ml", "dl", "news", "events", "fundamentals", "calendar",
)


class _Block(BaseModel):
    model_config = {"extra": "forbid"}


class TechnicalBlock(_Block):
    """Per-ticker technicals. Scale-dependent values are normalised by price so a
    model pooled across tickers can't identify a ticker by its price level."""
    Price_to_MA5: float | None = None        # % above/below 5-day MA
    Price_to_MA20: float | None = None
    Price_to_MA50: float | None = None
    RSI: float | None = None                 # 0-100
    MACD_pct: float | None = None            # MACD / Close * 100
    MACD_Signal_pct: float | None = None
    MACD_Hist_pct: float | None = None
    Stochastic_K: float | None = None        # 0-100
    Volatility: float | None = None          # 10-day std of daily returns
    ATR_pct: float | None = None             # ATR(14) / Close * 100
    BB_Width: float | None = None
    BB_Position: float | None = None
    VolumeSpike: float | None = None         # volume / prior 20-day avg volume
    Gap_pct: float | None = None             # (Open - prev Close) / prev Close * 100
    Return_1d: float | None = None           # today's return
    Return_Lag1: float | None = None         # yesterday's return
    Return_5d: float | None = None


class MarketBlock(_Block):
    Beta_90d: float | None = None
    Alpha_90d: float | None = None           # annualised
    VolAdj_Return: float | None = None
    VIX: float | None = None
    VIX_chg_5d: float | None = None          # VIX points change over 5 sessions
    SPY_above_200MA: float | None = None     # 1 bull / 0 bear
    SPY_ATR_pct: float | None = None
    SPY_ret_5d: float | None = None


class MLBlock(_Block):
    """Per-ticker ensemble (scripts/ml_market_movement.py) probabilities of UP."""
    p_up_rf: float | None = None
    p_up_xgb: float | None = None
    p_up_lr: float | None = None
    p_up_gb: float | None = None
    p_up_ensemble: float | None = None
    models_agree_frac: float | None = None   # share of the 4 models on the ensemble's side


class DLBlock(_Block):
    """Global GRU (scripts/dl_alpha_production.py)."""
    pred_log_return: float | None = None
    position_size: float | None = None       # tanh-scaled, -1..1
    direction: int | None = None             # -1 bearish / 0 neutral / 1 bullish


class NewsWindow(_Block):
    """Aggregates over articles in a window that ends at the as_of 16:00 ET close."""
    n_articles: int | None = None            # articles tagged with this ticker
    n_scored: int | None = None              # of which have a Polygon insight for this ticker
    n_publishers: int | None = None
    pos_ratio: float | None = None           # over scored articles
    neg_ratio: float | None = None
    net_sentiment: float | None = None       # pos_ratio - neg_ratio, -1..1
    finbert_mean: float | None = None        # filled in step 2 (FinBERT), -1..1
    finbert_std: float | None = None


class NewsLLMBlock(_Block):
    """Structured extraction by an LLM. Collected live only — LLM back-fills of
    history are contaminated by the model's knowledge of what happened next."""
    short_term: int | None = None            # -1 / 0 / 1
    short_term_conf: float | None = None
    long_term: int | None = None
    long_term_conf: float | None = None
    ev_earnings: bool | None = None
    ev_guidance: bool | None = None
    ev_analyst_action: bool | None = None
    ev_product: bool | None = None
    ev_regulatory_legal: bool | None = None
    ev_m_and_a: bool | None = None
    ev_macro: bool | None = None
    ev_management: bool | None = None


class NewsBlock(_Block):
    d1: NewsWindow = Field(default_factory=NewsWindow)   # since previous close
    d5: NewsWindow = Field(default_factory=NewsWindow)   # last 5 sessions
    volume_ratio_1d_vs_5d: float | None = None           # d1 count / (d5 count / 5)
    sentiment_change_1d_vs_5d: float | None = None       # d1.net - d5.net
    llm: NewsLLMBlock = Field(default_factory=NewsLLMBlock)


class EventsBlock(_Block):
    days_to_earnings: int | None = None      # calendar days to next report
    days_since_earnings: int | None = None
    is_earnings_window: bool | None = None   # next report within 5 calendar days


class FundamentalsBlock(_Block):
    log_market_cap: float | None = None
    trailingPE: float | None = None
    forwardPE: float | None = None
    debt_to_equity: float | None = None      # ratio (yfinance reports percent; divided by 100)
    operating_margin: float | None = None
    earnings_growth: float | None = None
    revenue_growth: float | None = None


class CalendarBlock(_Block):
    day_of_week: int | None = None           # 0 = Monday
    month: int | None = None
    is_opex_week: bool | None = None         # week of the 3rd-Friday monthly options expiry


class ReferenceBlock(_Block):
    """Raw prices needed to resolve the label. Not features."""
    close: float | None = None
    atr: float | None = None


class LabelBlock(_Block):
    """Filled ~3 trading days later by the label resolver. Same target as
    StockPredictor: 3-day move, ATR-filtered."""
    close_fwd_3d: float | None = None
    fwd_ret_3d: float | None = None
    fwd_move_atr: float | None = None
    y: int | None = None                     # 1 up / 0 down / None (noise or unresolved)
    is_noise: bool | None = None             # |move| < 0.5 ATR
    resolved_on: str | None = None


class ExplainerText(_Block):
    """Text kept for the explainer / Telegram message. Never fed to the model raw."""
    top_headlines: list[str] = Field(default_factory=list)
    llm_reasoning: str | None = None


class Provenance(_Block):
    """Where each block came from — used to filter training rows (e.g. drop
    in-sample base-model predictions, or back-filled LLM features)."""
    ml_model_path: str | None = None
    ml_model_trained_at: str | None = None
    ml_out_of_sample: bool | None = None     # False if the ML model trained on this row
    dl_available: bool = False
    dl_in_sample: bool | None = None
    news_source: str | None = None           # "polygon"
    llm_model: str | None = None             # "OpenRouter:google/gemini-2.5-flash"
    llm_source: Literal["live", "backfill"] | None = None
    fundamentals_point_in_time: bool | None = None  # False = today's snapshot
    errors: list[str] = Field(default_factory=list)


def _flatten(prefix: str, obj: BaseModel, out: dict):
    for name in type(obj).model_fields:
        value = getattr(obj, name)
        key = f"{prefix}.{name}"
        if isinstance(value, BaseModel):
            _flatten(key, value, out)
        elif value is None:
            out[key] = None
        elif isinstance(value, bool):
            out[key] = float(value)
        elif isinstance(value, (int, float)):
            out[key] = float(value)
        else:
            raise TypeError(f"Non-numeric feature {key}: {type(value).__name__}")


class DecisionInput(BaseModel):
    model_config = {"extra": "forbid"}

    schema_version: Literal["decision_input.v1"] = DECISION_SCHEMA_VERSION
    ticker: str
    as_of_date: str                          # YYYY-MM-DD (trading day)
    generated_at: str                        # ISO UTC timestamp

    technical: TechnicalBlock = Field(default_factory=TechnicalBlock)
    market: MarketBlock = Field(default_factory=MarketBlock)
    ml: MLBlock = Field(default_factory=MLBlock)
    dl: DLBlock = Field(default_factory=DLBlock)
    news: NewsBlock = Field(default_factory=NewsBlock)
    events: EventsBlock = Field(default_factory=EventsBlock)
    fundamentals: FundamentalsBlock = Field(default_factory=FundamentalsBlock)
    calendar: CalendarBlock = Field(default_factory=CalendarBlock)

    reference: ReferenceBlock = Field(default_factory=ReferenceBlock)
    text_for_explainer: ExplainerText = Field(default_factory=ExplainerText)
    provenance: Provenance = Field(default_factory=Provenance)
    label: LabelBlock = Field(default_factory=LabelBlock)

    @property
    def key(self) -> tuple[str, str]:
        return (self.ticker, self.as_of_date)

    def feature_vector(self) -> dict[str, float | None]:
        """Flat {"technical.RSI": 61.3, ...} in a fixed order; None = missing."""
        out: dict[str, float | None] = {}
        for block in DECISION_FEATURE_BLOCKS:
            _flatten(block, getattr(self, block), out)
        return out

    @classmethod
    def feature_names(cls) -> list[str]:
        return list(cls(ticker="_", as_of_date="1970-01-01", generated_at="").feature_vector())

    @classmethod
    def feature_hash(cls) -> str:
        """Fingerprint of the feature layout. A trained model stores this and
        refuses records whose layout differs."""
        import hashlib
        return hashlib.sha256("\n".join(cls.feature_names()).encode()).hexdigest()[:12]


class DecisionOutput(BaseModel):
    ticker: str
    as_of_date: str
    p_up: float = Field(ge=0.0, le=1.0)
    action: Literal["hold", "buy", "sell"]
    threshold: float
    model_version: str
    feature_hash: str
    top_drivers: list[tuple[str, float]] = Field(default_factory=list)  # (feature, SHAP contribution)
    reasoning: str | None = None


class NewsLLMExtraction(BaseModel):
    """What the LLM must return for the news.llm block (see prompts.news_features_template)."""
    short_term_sentiment: Literal["bearish", "neutral", "bullish"]
    short_term_confidence: float = Field(ge=0.0, le=1.0)
    long_term_sentiment: Literal["bearish", "neutral", "bullish"]
    long_term_confidence: float = Field(ge=0.0, le=1.0)
    events: dict[str, bool] = Field(default_factory=dict)
    reasoning: str = ""

    model_config = {"extra": "ignore"}
