import sys
import os
import pandas as pd
import warnings
import json
from devtools import pprint

# Add the parent folder to the path so we can import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# warnings.filterwarnings("ignore", category=FutureWarning)

# Import the classes
from utils.polygon_utils import polygon, prepare_news_for_llm, parse_news_by_date
from utils.data_models import NewsSentiment, NewsSentimentByDate
from utils.connect_db import ConnectDB
from utils.llm.prompt import prompts as p
from utils.llm.api_call import call_llm
from langchain_core.prompts import ChatPromptTemplate
# Reload modules to pick up changes
import importlib
import datetime
import utils.polygon_utils
import utils.llm.prompt
import utils.llm.api_call
importlib.reload(utils.polygon_utils)
importlib.reload(utils.llm.prompt)
importlib.reload(utils.llm.api_call)
from utils.polygon_utils import polygon, prepare_news_for_llm, parse_news_by_date
from utils.llm.prompt import prompts as p
from utils.llm.api_call import call_llm

class NewsAnalyzer:
    def __init__(self, ticker, start_date=datetime.date.today() - datetime.timedelta(days=7), end_date=datetime.date.today(), provider="Ollama"):
        self.ticker = ticker
        self.pl = polygon()
        if provider == "Ollama":
            self.model_name = "gpt-oss:20b"
            self.model_provider = "Ollama"
        elif provider == "Gemini":
            self.model_name = "gemini-2.5-flash"
            self.model_provider = "Gemini"
        self.max_retries = 3
        self.start_date = start_date
        self.end_date = end_date

    def analyze_news(self):
        nvdia_news, nvdia_news_df = self.pl.get_news(self.ticker, self.start_date, self.end_date, limit=1000, strict=False)
        
        if nvdia_news_df is None or nvdia_news_df.empty:
            print("No news found")
            return None, None

        nvdia_news_df.sort_index(ascending=False)
        LLM_news = prepare_news_for_llm(nvdia_news.results)
        template = p.news_sentiment_template
        prompt = template.invoke({"llm_news": json.dumps(LLM_news, indent=2), "ticker": self.ticker})

        sentiment_result = call_llm(prompt, self.model_name, self.model_provider, NewsSentiment, max_retries=self.max_retries)
        
        template_by_date = p.news_sentiment_by_date_template
        sentiment_results_by_date = []
        
        for date, news in parse_news_by_date(nvdia_news.results).items():
            llm_news = prepare_news_for_llm(news)
            prompt_news_by_date = template_by_date.invoke({"llm_news": json.dumps(llm_news, indent=2), "ticker": self.ticker, "date": date} )
            print(f'\nProcessing news on {date}: {len(news)} news')
            sentiment_result_by_date = call_llm(prompt_news_by_date, self.model_name, self.model_provider, NewsSentimentByDate, max_retries=self.max_retries)
            sentiment_results_by_date.append(sentiment_result_by_date)

        return sentiment_result, sentiment_results_by_date

if __name__ == "__main__":
    ticker = 'AAPL'
    analyzer = NewsAnalyzer(ticker)
    sentiment, sentiment_by_date = analyzer.analyze_news()
    if sentiment_by_date:
        pprint(sentiment_by_date)
