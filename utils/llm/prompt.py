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
                
                === INPUT DATA STRUCTURE ===
                1. ML Technical Analysis: High-confidence predictions from XGBoost and Random Forest models, trained on Technicals + Market Proxies (Beta, Alpha).
                2. Sentiment Analysis: Categorized news sentiment (Short-term vs. Long-term) and specific key event outlooks.
                3. Current Fundamentals: REAL-TIME snapshot of financial health (PE, D/E, Operating Margins). This is a FILTER, not a training feature.

                === THE TRUTH HIERARCHY ===
                Priority Order:
                1. ML Prediction = The "Momentum Signal" (based on Technicals + Market Proxies)
                2. LLM Filter = Uses Current Fundamentals to VALIDATE the trade
                3. Sentiment = Confirms or warns of catalysts

                === THE CONFLUENCE ALGORITHM (Strict Priority) ===
                
                STEP 1 - Confidence Threshold:
                If ML Ensemble Confidence < 0.60:
                    → action: "hold"
                    → Reasoning: "[WEAK_TECH]: Insufficient technical momentum for a high-probability trade."
                
                STEP 2 - Directional Alignment:
                - If ML_Prediction == "UP" AND Sentiment == "bullish" → action: "buy"
                - If ML_Prediction == "DOWN" AND Sentiment == "bearish" → action: "sell"
                
                STEP 3 - Conflict Resolution (Divergence):
                If ML_Prediction and Sentiment disagree:
                    → action: "hold"
                    → Reasoning: "[DIVERGENCE]: Technical vs Sentiment conflict; awaiting alignment."
                
                STEP 4 - FUNDAMENTAL FILTER (CRITICAL):
                This step ONLY applies if action == "buy". Check Current Fundamentals:
                
                ⚠️ OVERVALUATION CHECK:
                If PE_Assessment contains "OVERVALUED" or trailingPE > 50:
                    → DOWNGRADE confidence by 0.20
                    → Include "Fundamental Risk: Overvalued (PE > 50)" in reasoning
                
                ⚠️ HIGH DEBT CHECK:
                If Debt_Assessment contains "HIGH_RISK" or debtToEquity > 1.5:
                    → DOWNGRADE confidence by 0.20
                    → Include "Fundamental Risk: High Debt (D/E > 1.5)" in reasoning
                
                If BOTH risks are present:
                    → DOWNGRADE confidence by 0.40 (0.20 + 0.20)
                    → Consider downgrading action to "hold" if confidence drops below 0.50

                === REASONING FORMAT ===
                Reasoning must follow this format:
                "[CONFLUENCE/DIVERGENCE/WEAK_TECH]: {{Brief justification including specific ML confidence %, key news catalyst, and any fundamental risks}}."

                === OUTPUT REQUIREMENTS (JSON Format) ===
                You must return a MarketPrediction object:
                {{
                    "ticker": "{ticker}",
                    "action": "buy/sell/hold",
                    "confidence": 0.0-1.0 (after any fundamental downgrades),
                    "reasoning": "[TYPE]: ..."
                }}
                """
            ),
            (
                "human",
                """Symbol: {ticker}
                
                1. ML Technical Analysis (Momentum Signal):
                {ml_analysis}
                
                2. Sentiment Analysis:
                {news_analysis}
                
                3. Current Fundamentals (Real-Time Filter):
                {fundamental_analysis}
                
                Generate the final trading signal based on the Confluence Algorithm and Truth Hierarchy.
                Remember: Apply the fundamental filter if action is "buy" - check PE_Assessment and Debt_Assessment fields.
                """
            )
        ]
    )
