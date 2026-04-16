import sys
import os
import json
import datetime
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the parent folder to the path so we can import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.polygon_utils import polygon, prepare_news_for_llm, parse_news_by_date
from utils.data_models import NewsSentiment, NewsSentimentByDate
from utils.llm.prompt import prompts as p
from utils.llm.api_call import call_llm

class NewsAnalyzer:
    def __init__(self, ticker, start_date=None, end_date=None, provider="Ollama"):
        self.ticker = ticker
        self.pl = polygon()
        if provider == "Ollama":
            self.model_name = "gpt-oss:20b"
            self.model_provider = "Ollama"
        elif provider == "Gemini":
            self.model_name = "gemini-2.5-flash"
            self.model_provider = "Gemini"
        self.max_retries = 3
        # None sentinel avoids the mutable-default pitfall where date is evaluated
        # once at class definition time and grows stale in long-running processes.
        self.start_date = start_date if start_date is not None else datetime.date.today() - datetime.timedelta(days=10)
        self.end_date   = end_date   if end_date   is not None else datetime.date.today()

    def _analyze_single_date(self, date, news):
        """Analyze news for one date — runs inside a thread pool for parallel execution."""
        llm_news = prepare_news_for_llm(news)
        prompt = p.news_sentiment_by_date_template.invoke(
            {"llm_news": json.dumps(llm_news, indent=2), "ticker": self.ticker, "date": date}
        )
        print(f'\nProcessing news on {date}: {len(news)} articles')
        return call_llm(prompt, self.model_name, self.model_provider, NewsSentimentByDate, max_retries=self.max_retries)

    def analyze_news(self):
        news_results, news_df = self.pl.get_news(self.ticker, self.start_date, self.end_date, limit=1000, strict=False)

        if news_df is None or news_df.empty:
            print("No news found")
            return None, None

        news_df.sort_index(ascending=False)
        LLM_news = prepare_news_for_llm(news_results.results)
        prompt = p.news_sentiment_template.invoke(
            {"llm_news": json.dumps(LLM_news, indent=2), "ticker": self.ticker}
        )
        sentiment_result = call_llm(prompt, self.model_name, self.model_provider, NewsSentiment, max_retries=self.max_retries)

        # Parallelize per-date LLM calls — previously sequential (1 blocking call per day).
        # ThreadPoolExecutor is appropriate because call_llm is I/O-bound (network).
        date_news_map = parse_news_by_date(news_results.results)
        sentiment_results_by_date = []

        with ThreadPoolExecutor(max_workers=min(4, len(date_news_map))) as executor:
            futures = {
                executor.submit(self._analyze_single_date, date, news): date
                for date, news in date_news_map.items()
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    sentiment_results_by_date.append(result)

        return sentiment_result, sentiment_results_by_date

if __name__ == "__main__":
    ticker = 'AAPL'
    analyzer = NewsAnalyzer(ticker)
    sentiment, sentiment_by_date = analyzer.analyze_news()
    if sentiment_by_date:
        pprint(sentiment_by_date)
