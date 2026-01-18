import pandas as pd
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

class prompts:
    news_sentiment_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a financial analyst. You are given a list of news articles and a ticker. You need to analyze the news and provide sentiment analysis.
                    
                    You must respond in JSON format with this exact structure:
                    {{
                        "short_term_sentiment": {{
                            "short_term_sentiment": "bullish/bearish/neutral",
                            "short_term_confidence": 0.0-1.0,
                            "short_term_reasoning": "explanation"
                        }},
                        "long_term_sentiment": {{
                            "long_term_sentiment": "bullish/bearish/neutral",
                            "long_term_confidence": 0.0-1.0,
                            "long_term_reasoning": "explanation"
                        }},
                        "key_events_outlook": [
                            {{
                                "event": "event name",
                                "event_date": "YYYY-MM-DD or approximate",
                                "event_type": "earnings/product_launch/regulatory/etc",
                                "event_impact": "positive/negative/neutral",
                                "event_reasoning": "why this matters"
                            }}
                        ]
                    }}
                """,
            ),
            (
                "human",
                """Based on the following news, create the investment signal:
                    News Data for {ticker}
                    {llm_news}
                    
                    Provide your analysis in JSON format following the exact schema specified.
                """,
            ),
        ]
    )

        
    news_sentiment_by_date_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                    """You are a financial analyst. You are given a list of news articles and a ticker on a specific date. You need to analyze the news and provide a summary of the news and the sentiment of the news.
                    You should explain why the stock price might move up or down based on the news.
                    """,
                ),
                (
                    "human",
                    """Based on the following news, create the investment signal:

                    News Data for {ticker} on {date}:
                    {llm_news}
                    You response should be in the following format:
                    Summary: a short summary of the news
                    Sentiment: categorized by strongly positive, positive, neutral, negative, strongly negative
                    Reasoning: a short reasoning for the sentiment
                    Date: {date}
                    Stock Price Movement: does the news suggest the moevment of stock price on that day?
                    """
                    ,
                ),
            ]
        )

    market_agent_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a senior hedge fund manager. You make the final trading decision based on inputs from your team.
                Your team has provided:
                1. Technical & Fundamental Analysis (ML Models): Predictions from Random Forest and XGBoost models, including confidence/probability.
                2. News Sentiment Analysis: Sentiment derived from recent news coverage, including short-term and long-term outlooks.

                Your task is to synthesize these inputs and provide a final recommendation (Buy, Sell, or Hold), a confidence score (0.0 to 1.0), and a detailed reasoning explaining your decision.
                Consider conflicting signals carefully. If ML says UP but News is Bearish, explain how you weigh them.
                
                You must respond in JSON format with this exact structure:
                {{
                    "ticker": "{ticker}",
                    "action": "buy/sell/hold",
                    "confidence": 0.0-1.0,
                    "reasoning": "your detailed synthesis and explanation"
                }}
                """
            ),
            (
                "human",
                """Symbol: {ticker}
                
                Technical & Fundamental ML Analysis:
                {ml_analysis}
                
                News Sentiment Analysis:
                {news_analysis}
                
                Make your final decision.
                """
            )
        ]
    )
