import sys
import os
import json
from pathlib import Path
from pprint import pprint

# Add the parent folder to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import yfinance as yf
from scripts.news import NewsAnalyzer
from scripts.ml_market_movement import StockPredictor
from scripts.dl_alpha_production import AlphaPredictor
from utils.llm.prompt import prompts as p
from utils.llm.api_call import call_llm
from utils.data_models import MarketPrediction

# Reliable fundamental keys to extract from yfinance
_FUNDAMENTAL_KEYS = [
    'marketCap',
    'trailingPE',
    'dividendYield',
    'debtToEquity',
    'operatingMargins'
]

def fetch_current_fundamentals(ticker: str) -> dict:
    """
    Fetch current fundamental snapshot from yfinance.

    Extracted as a module-level function so app.py can call it directly
    without instantiating a full MarketAgent (which loads PyTorch weights).

    This is a "Real-Time Filter" for the LLM, NOT a training feature.
    Returns:
        Dict with fundamental metrics and human-readable assessments for the LLM.
    """
    print(f"Fetching current fundamentals for {ticker}...")
    fundamentals = {}

    try:
        info = yf.Ticker(ticker).info

        for key in _FUNDAMENTAL_KEYS:
            fundamentals[key] = info.get(key, "N/A")

        # Human-readable PE assessment
        pe = fundamentals.get('trailingPE')
        if isinstance(pe, (int, float)):
            if pe > 50:
                fundamentals['PE_Assessment'] = "OVERVALUED (PE > 50)"
            elif pe > 25:
                fundamentals['PE_Assessment'] = "FAIRLY_VALUED (PE 25-50)"
            else:
                fundamentals['PE_Assessment'] = "UNDERVALUED (PE < 25)"
        else:
            fundamentals['PE_Assessment'] = "UNKNOWN"

        # Human-readable Debt assessment
        de = fundamentals.get('debtToEquity')
        if isinstance(de, (int, float)):
            if de > 1.5:
                fundamentals['Debt_Assessment'] = "HIGH_RISK (D/E > 1.5)"
            elif de > 1.0:
                fundamentals['Debt_Assessment'] = "MODERATE_RISK (D/E 1.0-1.5)"
            else:
                fundamentals['Debt_Assessment'] = "LOW_RISK (D/E < 1.0)"
        else:
            fundamentals['Debt_Assessment'] = "UNKNOWN"

        print(f"Fundamentals: PE={fundamentals.get('trailingPE', 'N/A')}, D/E={fundamentals.get('debtToEquity', 'N/A')}")

    except Exception as e:
        print(f"Warning: Could not fetch fundamentals: {e}")
        fundamentals = {key: "N/A" for key in _FUNDAMENTAL_KEYS}
        fundamentals['PE_Assessment'] = "UNKNOWN"
        fundamentals['Debt_Assessment'] = "UNKNOWN"

    return fundamentals


class MarketAgent:
    """
    Market Agent that combines ML predictions with LLM-filtered fundamentals.

    Key Design Principle:
    - ML Prediction: Based on Technicals + Market Proxies (The "Momentum" signal)
    - LLM Filter: Uses Current Fundamentals to validate the trade (Real-Time Filter)
    """

    def __init__(self, ticker):
        self.ticker = ticker
        self.stock_predictor = StockPredictor(ticker)
        self.alpha_predictor = AlphaPredictor(PROJECT_ROOT / "models" / "alpha_gru_v1")
        self.model_name = "gpt-oss:20b"
        self.model_provider = "Ollama"
        self.max_retries = 3
        self.news_analyzer = NewsAnalyzer(ticker, provider=self.model_provider)

    def _fetch_current_fundamentals(self):
        """Delegates to the module-level fetch_current_fundamentals()."""
        return fetch_current_fundamentals(self.ticker)

    def run_analysis(self):
        print(f"--- Starting Analysis for {self.ticker} ---")

        # 1. ML Analysis (Technical + Market Proxies only, NO fundamentals)
        print("\n--- Running ML Prediction ---")
        ml_analysis = {}
        try:
            self.stock_predictor.data_processing()
            self.stock_predictor.train_models()
            try:
                self.stock_predictor.evaluate_models()
            except Exception as e:
                print(f"Model evaluation failed: {e}")
            ml_analysis = self.stock_predictor.predict_next_day()
            self.stock_predictor.save_final_prediction(ml_analysis)
            pprint(ml_analysis)
        except Exception as e:
            print(f"ML Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            ml_analysis = {"error": str(e)}
            
        # 1.5 DL Alpha Prediction
        print("\n--- Running DL Alpha Prediction ---")
        alpha_analysis = {}
        try:
            alpha_analysis = self.alpha_predictor.predict_next_price(self.ticker)
            pprint(alpha_analysis)
        except Exception as e:
            print(f"DL Alpha Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            alpha_analysis = {"error": str(e)}
        
        # 2. News Analysis
        print("\n--- Running News Analysis ---")
        news_analysis = {}
        try:
            sentiment_result, sentiment_by_date = self.news_analyzer.analyze_news()
            if sentiment_result:
                news_analysis = sentiment_result.model_dump()
                # Include the per-day breakdown so the LLM sees daily catalysts
                if sentiment_by_date:
                    news_analysis["daily_breakdown"] = [
                        s.model_dump() for s in sentiment_by_date if s is not None
                    ]
            else:
                news_analysis = {"info": "No news found"}
        except Exception as e:
            print(f"News Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            news_analysis = {"error": str(e)}
        
        # 3. Fetch Current Fundamentals (Real-Time Filter for LLM)
        print("\n--- Fetching Current Fundamentals ---")
        fundamental_analysis = self._fetch_current_fundamentals()
            
        print("\n--- Generating Final Opinion ---")
        template = p.market_agent_template
        
        prompt = template.invoke({
            "ticker": self.ticker,
            "ml_analysis": json.dumps(ml_analysis, indent=2),
            "alpha_analysis": json.dumps(alpha_analysis, indent=2),
            "news_analysis": json.dumps(news_analysis, indent=2),
            "fundamental_analysis": json.dumps(fundamental_analysis, indent=2)
        })
        
        final_prediction = call_llm(
            prompt, 
            self.model_name, 
            self.model_provider, 
            MarketPrediction, 
            max_retries=self.max_retries
        )
        
        return final_prediction

if __name__ == "__main__":
    ticker = "AAPL" 
    agent = MarketAgent(ticker)
    prediction = agent.run_analysis()
    print("\n==================================")
    print("      FINAL MARKET OPINION      ")
    print("==================================")
    pprint(prediction)
