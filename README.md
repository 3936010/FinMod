# FinMod 📈

**FinMod** is an advanced financial technology platform that integrates Deep Learning, Ensemble Machine Learning, and AI Agents to provide institutional-grade market analysis and trading signals.

The system leverages a hybrid approach:
- **Deep Learning (Alpha Engine)**: A multi-asset GRU model for high-precision return forecasting.
- **Machine Learning (Technical Lab)**: Ensemble models (XGBoost, Random Forest) for multi-horizon technical analysis.
- **AI Agents**: Large Language Models (LLMs) used for sentiment extraction from financial news and final confluence-based decision making.

---

## 🚀 Key Features

### 🧠 Deep Learning Alpha Engine
- **Global Multi-Asset GRU**: A production-ready PyTorch implementation trained on major tech tickers.
- **Robust Feature Engineering**: Includes RSI, MACD, Parkinson Volatility, ATR, and more.
- **Production-Ready Inference**: Thread-safe predictor module with tanh-based position sizing.

### 🤖 AI Market Agents
- **Sentiment Analysis**: Multi-agent system to analyze financial news using Gemini/GPT models.
- **Confluence Logic**: Combines ML technical analysis, DL alpha predictions, and news sentiment into a final "Buy/Sell/Hold" signal.

### 📊 Interactive Dashboard
- **Streamlit UI**: A professional trading dashboard for real-time ticker analysis.
- **Risk Management**: Automated position sizing, stop-loss, and take-profit calculation using ATR volatility.
- **Technical Lab**: Deep dive into market proxies, beta/alpha metrics, and price-volume ratios.

---

## 📁 Project Structure

```text
FinMod/
├── app.py                  # Streamlit Dashboard UI
├── scripts/
│   ├── dl_alpha_production.py # Production DL Engine (PyTorch)
│   ├── ai_market_agent.py   # AI Agent Logic
│   └── price.py             # Sandbox for model experimentation
├── utils/
│   ├── risk_manager.py      # ATR-based risk calculations
│   ├── llm/                 # LLM API wrappers and prompts
│   └── data_fetchers/       # Polygon, WRDS, YFinance connectors
├── models/                  # Saved model weights and artifacts
└── results/                 # Performance reports and predictions
```

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Highly recommended for dependency management)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/3936010/FinMod.git
   cd FinMod
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Configure Environment**:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   POLYGON_API_KEY=your_key
   GOOGLE_API_KEY=your_key
   OPENAI_API_KEY=your_key
   # See .env.example for full list
   ```

---

## 📖 Usage

### Launching the Dashboard
```bash
streamlit run app.py
```

### Training the Alpha Engine
```bash
python scripts/dl_alpha_production.py --train
```

---

## 🗺️ Roadmap
- [ ] Integrate AlphaPredictor directly into the agent decision loop.
- [ ] Implement backtesting engine for historical performance validation.
- [ ] Add support for crypto and forex asset classes.
- [ ] Enhance UI with real-time order execution simulations.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
