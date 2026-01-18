import sys
import os
import json
from pprint import pprint

# Add the parent folder to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.news import NewsAnalyzer
from scripts.ml_market_movement import StockPredictor
from utils.llm.prompt import prompts as p
from utils.llm.api_call import call_llm
from utils.data_models import MarketPrediction
import importlib
import utils.llm.prompt
importlib.reload(utils.llm.prompt) # Reload prompts to get the new template

class MarketAgent:
    def __init__(self, ticker):
        self.ticker = ticker
        self.news_analyzer = NewsAnalyzer(ticker)
        self.stock_predictor = StockPredictor(ticker, use_fundamentals=True)
        self.model_name = "qwen3:8b" # Options: llama3.2, llama3.1, mistral, qwen2.5
        self.model_provider = "Ollama"
        self.max_retries = 3

    def run_analysis(self):
        print(f"--- Starting Analysis for {self.ticker} ---")

        # 1. ML Analysis
        print("\n--- Running ML Prediction ---")
        ml_analysis = {}
        try:
            self.stock_predictor.data_processing()
            self.stock_predictor.train_models()
            ml_analysis = self.stock_predictor.predict_next_day()
            pprint(ml_analysis)
        except Exception as e:
            print(f"ML Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            ml_analysis = {"error": str(e)}
        
        # 2. News Analysis
        print("\n--- Running News Analysis ---")
        news_analysis = {}
        try:
            sentiment_result, sentiment_date = self.news_analyzer.analyze_news()
            if sentiment_result:
                # Convert Pydantic to dict for LLM
                news_analysis = sentiment_result.model_dump()
            else:
                news_analysis = {"info": "No news found"}
        except Exception as e:
            print(f"News Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            news_analysis = {"error": str(e)}
            
        print("\n--- Generating Final Opinion ---")
        template = p.market_agent_template
        prompt = template.invoke({
            "ticker": self.ticker,
            "ml_analysis": json.dumps(ml_analysis, indent=2),
            "news_analysis": json.dumps(news_analysis, indent=2)
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
