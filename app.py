import streamlit as st
import sys
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import yfinance as yf
from scripts.ai_market_agent import MarketAgent
from scripts.ml_market_movement import StockPredictor
from scripts.dl_alpha_production import AlphaPredictor
from scripts.news import NewsAnalyzer
from utils.risk_manager import RiskManager
from utils.llm.prompt import prompts as p
from utils.llm.api_call import call_llm
from utils.data_models import MarketPrediction

import plotly.graph_objects as go
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="FinMod Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Signal Cards
st.markdown("""
<style>
    .signal-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
    }
    .signal-buy { background: linear-gradient(135deg, #00c853, #1de9b6); }
    .signal-sell { background: linear-gradient(135deg, #ff1744, #ff5252); }
    .signal-hold { background: linear-gradient(135deg, #607d8b, #90a4ae); }
    .metric-card {
        background: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
total_cash = st.sidebar.slider("Portfolio Cash ($)", min_value=100, max_value=1000, value=500, step=50)
# model_options = ["llama3.2:3b", "gpt-oss:20b", "gpt-oss:20b"]
model_options = ["gemini-2.5-flash"]
selected_model = st.sidebar.selectbox("LLM Model", model_options)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 News Date Range")
news_start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2026-02-01"),
    help="Start date for news analysis"
)
news_end_date = st.sidebar.date_input(
    "End Date",
    value=pd.to_datetime("2026-02-05"),
    help="End date for news analysis"
)

st.sidebar.markdown("---")
run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)


# --- Cached Model Training ---
@st.cache_resource
def get_trained_predictor(ticker_symbol: str):
    """Cache the trained StockPredictor to avoid retraining on UI refresh."""
    # use_fundamentals=False because fundamentals are now fetched separately as real-time filter
    predictor = StockPredictor(ticker_symbol, use_fundamentals=False)
    predictor.data_processing()
    predictor.train_models()
    return predictor

@st.cache_resource
def get_trained_alpha_predictor(model_dir: str):
    """Cache the DL Alpha Predictor to avoid reloading PyTorch models on UI refresh."""
    return AlphaPredictor(model_dir)

# --- Main Header ---
st.title(f"📊 Trading Dashboard - {ticker}")

# --- Main Content ---
if run_analysis:
    with st.spinner(f"Running analysis for {ticker}... This may take a few minutes."):
        try:
            # 1. ML Prediction (Cached)
            st.info("🔄 Training ML models...")
            predictor = get_trained_predictor(ticker)
            ml_analysis = predictor.predict_next_day()
            
            # Get current price and ATR from latest data (for Risk & Execution)
            current_price = predictor.data['Close'].iloc[-1]
            atr = predictor.data['ATR'].iloc[-1]
            
            # 1.5 DL Alpha Prediction
            st.info("🧠 Running DL Alpha prediction...")
            try:
                alpha_predictor = get_trained_alpha_predictor("/home/sd/FinMod/models/alpha_gru_v1")
                alpha_analysis = alpha_predictor.predict_next_price(ticker)
            except Exception as e:
                import traceback
                traceback.print_exc()
                alpha_analysis = {"error": str(e)}
            
            # 2. News Analysis
            st.info("📰 Analyzing news sentiment...")
            news_analyzer = NewsAnalyzer(ticker)
            news_analyzer.model_name = selected_model
            # Convert date objects to string format (YYYY-MM-DD)
            start_date_str = news_start_date.strftime('%Y-%m-%d')
            end_date_str = news_end_date.strftime('%Y-%m-%d')
            sentiment_result, sentiment_by_date = news_analyzer.analyze_news(
                start_date=start_date_str,
                end_date=end_date_str
            )
            news_analysis = sentiment_result.model_dump() if sentiment_result else {"info": "No news found"}
            
            # 3. Fetch Current Fundamentals (Real-Time Filter for LLM)
            st.info("🏢 Fetching current fundamentals...")
            fundamental_analysis = MarketAgent(ticker)._fetch_current_fundamentals()
            
            # 4. LLM Final Decision
            st.info("🤖 Generating final trading signal...")
            
            template = p.market_agent_template
            prompt = template.invoke({
                "ticker": ticker,
                "ml_analysis": json.dumps(ml_analysis, indent=2),
                "alpha_analysis": json.dumps(alpha_analysis, indent=2),
                "news_analysis": json.dumps(news_analysis, indent=2),
                "fundamental_analysis": json.dumps(fundamental_analysis, indent=2)
            })
            
            final_prediction = call_llm(
                prompt,
                selected_model,
                "Gemini", # Was "Ollama"
                MarketPrediction,
                max_retries=3
            )
            
            # --- Display Results ---
            st.success("✅ Analysis Complete!")
            
            # Header Row: Price + Signal Card
            col_price, col_signal = st.columns([1, 1])
            
            with col_price:
                st.metric(label="Current Price", value=f"${current_price:.2f}")
                st.metric(label="14-Day ATR", value=f"${atr:.2f}")
            
            with col_signal:
                action = final_prediction.action.lower()
                if action == "buy":
                    st.markdown('<div class="signal-card signal-buy">🟢 BUY</div>', unsafe_allow_html=True)
                elif action == "sell":
                    st.markdown('<div class="signal-card signal-sell">🔴 SELL</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="signal-card signal-hold">⬜ HOLD</div>', unsafe_allow_html=True)
            
            # Metrics Row
            met_col1, met_col2 = st.columns(2)
            with met_col1:
                st.metric("Ensemble Confidence", f"{ml_analysis['Ensemble_Confidence']*100:.1f}%")
            with met_col2:
                st.metric("LLM Confidence", f"{final_prediction.confidence*100:.1f}%")
            
            # === TECHNICAL LAB TAB ===
            st.markdown("---")
            st.subheader("📈 Technical Lab")
            
            # Market Proxy Metrics Row (Beta/Alpha)
            market_proxies = ml_analysis.get('Market_Proxies', {})
            proxy_col1, proxy_col2, proxy_col3, proxy_col4 = st.columns(4)
            with proxy_col1:
                beta = market_proxies.get('Beta_90d', 'N/A')
                st.metric("90-Day Beta", f"{beta:.2f}" if isinstance(beta, (int, float)) else beta)
            with proxy_col2:
                alpha = market_proxies.get('Alpha_90d', 'N/A')
                st.metric("90-Day Alpha", f"{alpha:.2%}" if isinstance(alpha, (int, float)) else alpha)
            with proxy_col3:
                vol_adj = market_proxies.get('VolAdj_Return', 'N/A')
                st.metric("Vol-Adj Return", f"{vol_adj:.2f}" if isinstance(vol_adj, (int, float)) else vol_adj)
            with proxy_col4:
                pv_ratio = market_proxies.get('PriceVolume_Ratio', 'N/A')
                st.metric("Price/Vol Ratio", f"{pv_ratio:.2f}" if isinstance(pv_ratio, (int, float)) else pv_ratio)
            
            # Two-Column Analysis Pane
            st.markdown("---")
            st.subheader("📊 Analysis Details")
            
            col_ml, col_sent = st.columns(2)
            
            with col_ml:
                st.markdown("### 🔢 ML Technical Analysis")
                # Display ML analysis without market proxies (shown above)
                ml_display = {k: v for k, v in ml_analysis.items() if k != 'Market_Proxies'}
                st.json(ml_display)
                
                # Current Fundamentals (Real-Time Filter)
                st.markdown("### 🏢 Current Fundamentals (LLM Filter)")
                st.json(fundamental_analysis)

                # DL Alpha Prediction
                st.markdown("### 🧠 DL Alpha Analysis")
                st.json(alpha_analysis)
            
            with col_sent:
                st.markdown("### 📰 Sentiment / News Analysis")
                st.caption(f"📅 Date Range: {start_date_str} to {end_date_str}")
                st.json(news_analysis)
                
                st.markdown("### 🎯 LLM Reasoning")
                st.info(final_prediction.reasoning)
            
            # Risk Management (if BUY or SELL)
            if action in ["buy", "sell"]:
                st.markdown("---")
                st.subheader("⚠️ Risk Management")
                
                risk_manager = RiskManager()
                risk_output = risk_manager.calculate_entry(
                    total_cash=total_cash,
                    current_price=current_price,
                    atr=atr,
                    llm_confidence=final_prediction.confidence
                )
                
                # Position Value Metric
                st.metric("Total Position Value", f"${risk_output['total_position_value']:,.2f}")
                
                # Risk Table
                risk_df = pd.DataFrame([{
                    "Shares to Buy": risk_output['shares_to_buy'],
                    "Entry Price": f"${current_price:.2f}",
                    "Stop-Loss (2x ATR)": f"${risk_output['stop_loss_price']:.2f}",
                    "Take-Profit (3x ATR)": f"${risk_output['take_profit_price']:.2f}",
                    "Position Value": f"${risk_output['total_position_value']:,.2f}"
                }])
                st.table(risk_df)
                
                # Plotly Chart for Entry/Exit Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=["Stop-Loss", "Entry", "Take-Profit"],
                    y=[risk_output['stop_loss_price'], current_price, risk_output['take_profit_price']],
                    mode='markers+lines+text',
                    text=[f"${risk_output['stop_loss_price']:.2f}", f"${current_price:.2f}", f"${risk_output['take_profit_price']:.2f}"],
                    textposition="top center",
                    marker=dict(size=15, color=['red', 'blue', 'green']),
                    line=dict(width=2, dash='dash')
                ))
                fig.update_layout(
                    title="Trade Levels",
                    yaxis_title="Price ($)",
                    template="plotly_dark",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

else:
    st.info("👈 Configure your analysis in the sidebar and click **Run Analysis** to start.")
    
    # Placeholder content
    st.markdown("""
    ## How to Use
    1. Enter a **Ticker Symbol** (e.g., AAPL, MSFT, NVDA)
    2. Set your **Portfolio Cash** amount
    3. Select an **LLM Model** for the final decision
    4. Click **Run Analysis** to generate trading signals
    
    ## What You'll Get
    - **ML Technical Predictions**: Random Forest + XGBoost ensemble
    - **News Sentiment Analysis**: LLM-powered news interpretation
    - **Final Trading Signal**: Confluence-based decision (Buy/Sell/Hold)
    - **Risk Management**: Position sizing with ATR-based stops
    """)
