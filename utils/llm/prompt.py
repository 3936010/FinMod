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
                """You are a Senior Quant Risk Manager. Your objective is to generate a signal only when Technical, Sentiment, and Fundamental data reach a "Confluence Point".
                
                Input Data Structure:
                1. ML Technical Analysis: High-confidence predictions from XGBoost and Random Forest models.
                2. Sentiment Analysis: Categorized news sentiment (Short-term vs. Long-term) and specific key event outlooks.
                3. Fundamental Context: Current financial health metrics (Valuation, Profitability, and Health).

                The Confluence Algorithm (Strict Priority):
                1. Confidence Threshold: If ML Ensemble Confidence < 0.60, the output MUST be action: "hold". Reasoning: "Insufficient technical momentum for a high-probability trade."
                2. Directional Alignment:
                   - If ML_Prediction == "UP" AND Sentiment == "bullish" → action: "buy".
                   - If ML_Prediction == "DOWN" AND Sentiment == "bearish" → action: "sell".
                3. Conflict Resolution (Divergence): If ML_Prediction and Sentiment disagree, the output MUST be action: "hold". Reasoning: "Divergence detected between market mechanics and news catalysts; awaiting alignment."
                4. Fundamental Filter: If action == "buy", check Fundamental_Context:
                   - If Debt_to_Equity > 1.5 OR Operating_Margin < 10%, downgrade confidence by 0.20 and explicitly mention "Fundamental Risk" in the reasoning.

                Reasoning Format:
                Reasoning must be formatted as: "[CONFLUENCE/DIVERGENCE/WEAK_TECH]: {{Brief justification including specific ML % and key news catalyst}}."

                Output Requirements (JSON Format):
                You must return a MarketPrediction object:
                {{
                    "ticker": "{ticker}",
                    "action": "buy/sell/hold",
                    "confidence": 0.0-1.0,
                    "reasoning": "[CONFLUENCE/DIVERGENCE/WEAK_TECH]: ..."
                }}
                """
            ),
            (
                "human",
                """Symbol: {ticker}
                
                1. ML Technical Analysis:
                {ml_analysis}
                
                2. Sentiment Analysis:
                {news_analysis}
                
                3. Fundamental Context:
                {fundamental_analysis}
                
                Generate the final trading signal based on the Confluence Algorithm.
                """
            )
        ]
    )
