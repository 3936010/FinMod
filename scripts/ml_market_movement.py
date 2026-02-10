import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


class StockPredictor:
    def __init__(self, ticker, use_fundamentals=True):
        self.ticker = ticker
        # use_fundamentals flag is kept for backward compatibility but fundamentals 
        # are now only used as real-time LLM filters, not for training
        self.use_fundamentals = use_fundamentals
        self.model_RF = None
        self.model_XGB = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data = None
        self.market_proxy_features = {}  # Latest market proxy values for display
        
        # Technical features
        self.technical_features = [
            # Price features
            'High', 'Low', 'Volume',
            # Moving averages
            'MA5', 'MA10', 'MA20', 'MA50',
            # Price relative to MAs
            'Price_to_MA5', 'Price_to_MA20',
            # Momentum indicators
            'RSI', 'Momentum', 'MACD', 'MACD_Signal',
            # Volatility indicators
            'Volatility', 'ATR', 'BB_Width', 'BB_Position',
            # Volume indicators
            'VolumeSpike', 'Volume_MA_Ratio',
            # Other
            'Gap', 'Stochastic_K', 'Stochastic_D',
            # Lag features
            'Return_Lag1', 'Return_Lag2', 'Return_Lag3'
        ]
        
        # Market proxy features (replaces fundamental features to prevent data leakage)
        self.market_proxy_feature_names = [
            'Beta_90d',           # 90-day rolling correlation with SPY
            'Alpha_90d',          # Excess return relative to Beta-adjusted SPY
            'VolAdj_Return',      # Return divided by 20-day ATR/StdDev
            'PriceVolume_Ratio'   # Close / 20-day average volume
        ]
        
        # Training uses ONLY technical + market proxy features (NO fundamentals)
        self.features = self.technical_features + self.market_proxy_feature_names

    def _calculate_features(self, data):
        """Calculate all technical indicators and features."""
        # Daily Returns
        data['Return'] = data['Close'].pct_change()
        
        # Moving Averages
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Price relative to moving averages (%)
        data['Price_to_MA5'] = (data['Close'] / data['MA5'] - 1) * 100
        data['Price_to_MA20'] = (data['Close'] / data['MA20'] - 1) * 100
        
        # Rolling Standard Deviation (Volatility)
        data['Volatility'] = data['Return'].rolling(window=10).std()
        
        # RSI (Relative Strength Index)
        data['Delta'] = data['Close'].diff()
        data['Gain'] = data['Delta'].where(data['Delta'] > 0, 0)
        data['Loss'] = -data['Delta'].where(data['Delta'] < 0, 0)
        data['AvgGain'] = data['Gain'].rolling(window=14).mean()
        data['AvgLoss'] = data['Loss'].rolling(window=14).mean()
        data['RS'] = data['AvgGain'] / data['AvgLoss']
        data['RSI'] = 100 - (100 / (1 + data['RS']))
        
        # Momentum
        data['Momentum'] = data['Close'] - data['Close'].shift(5)
        
        # Gap (overnight)
        data['Gap'] = data['Open'] - data['Close'].shift(1)
        
        # Volume indicators
        data['VolumeSpike'] = data['Volume'] / data['Volume'].rolling(20).mean()
        data['Volume_MA_Ratio'] = data['Volume'] / data['Volume'].rolling(10).mean()
        
        # MACD
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        # ATR (Average True Range)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = tr.rolling(window=14).mean()
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + 2 * bb_std
        data['BB_Lower'] = data['BB_Middle'] - 2 * bb_std
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        data['Stochastic_K'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)
        data['Stochastic_D'] = data['Stochastic_K'].rolling(window=3).mean()
        
        # Lag features (previous day returns)
        data['Return_Lag1'] = data['Return'].shift(1)
        data['Return_Lag2'] = data['Return'].shift(2)
        data['Return_Lag3'] = data['Return'].shift(3)
        
        return data

    def _add_market_proxy_features(self, data):
        """
        Calculate market proxy features to replace fundamentals.
        These are rolling metrics that can be calculated from price/volume data only.
        
        Features:
        - Beta_90d: 90-day rolling correlation with SPY (market sensitivity)
        - Alpha_90d: Excess return relative to Beta-adjusted SPY return
        - VolAdj_Return: Current return divided by 20-day volatility (risk-adjusted momentum)
        - PriceVolume_Ratio: Close price divided by 20-day average volume (liquidity proxy)
        """
        print("Calculating market proxy features...")
        
        # Fetch SPY data for Beta/Alpha calculation
        try:
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="max")
            spy_data['SPY_Return'] = spy_data['Close'].pct_change()
            
            # Align SPY data with ticker data by date
            data['SPY_Return'] = spy_data['SPY_Return'].reindex(data.index)
            
            # Beta: 90-day rolling covariance / variance with SPY
            rolling_cov = data['Return'].rolling(window=90).cov(data['SPY_Return'])
            rolling_var = data['SPY_Return'].rolling(window=90).var()
            data['Beta_90d'] = rolling_cov / rolling_var
            
            # Alpha: Excess return = Stock Return - (Beta * SPY Return)
            # Use 90-day rolling mean of excess returns
            excess_return = data['Return'] - (data['Beta_90d'] * data['SPY_Return'])
            data['Alpha_90d'] = excess_return.rolling(window=90).mean() * 252  # Annualized
            
            # Clean up temporary column
            del data['SPY_Return']
            
        except Exception as e:
            print(f"Warning: Could not calculate Beta/Alpha: {e}")
            data['Beta_90d'] = 1.0  # Default to market beta
            data['Alpha_90d'] = 0.0  # Default to no alpha
        
        # VolAdj_Return: Return / 20-day rolling volatility (Sharpe-like ratio)
        rolling_vol = data['Return'].rolling(window=20).std()
        data['VolAdj_Return'] = data['Return'] / rolling_vol
        
        # PriceVolume_Ratio: Close / 20-day average volume (normalized)
        avg_volume = data['Volume'].rolling(window=20).mean()
        data['PriceVolume_Ratio'] = data['Close'] / avg_volume
        # Scale to reasonable range (values can be very small)
        data['PriceVolume_Ratio'] = data['PriceVolume_Ratio'] * 1e6
        
        # Store latest values for display
        self.market_proxy_features = {
            'Beta_90d': float(data['Beta_90d'].iloc[-1]) if not pd.isna(data['Beta_90d'].iloc[-1]) else 1.0,
            'Alpha_90d': float(data['Alpha_90d'].iloc[-1]) if not pd.isna(data['Alpha_90d'].iloc[-1]) else 0.0,
            'VolAdj_Return': float(data['VolAdj_Return'].iloc[-1]) if not pd.isna(data['VolAdj_Return'].iloc[-1]) else 0.0,
            'PriceVolume_Ratio': float(data['PriceVolume_Ratio'].iloc[-1]) if not pd.isna(data['PriceVolume_Ratio'].iloc[-1]) else 0.0
        }
        
        print(f"Market Proxy Features: Beta={self.market_proxy_features['Beta_90d']:.2f}, Alpha={self.market_proxy_features['Alpha_90d']:.2%}")
        
        return data

    def data_processing(self, period="max", start_date="2015-01-01"):
        """
        Process data with technical features and market proxies.
        
        Training Constraint: Uses ONLY technical_features + market_proxy_features.
        Fundamentals are NOT used in training to prevent historical data leakage.
        """
        ticker_obj = yf.Ticker(self.ticker)
        data = ticker_obj.history(period=period)
        
        # Remove unnecessary columns
        for col in ["Dividends", "Stock Splits"]:
            if col in data.columns:
                del data[col]
        
        # Create target variable
        data["Tomorrow"] = data["Close"].shift(-1)
        data['Target'] = np.where(data['Tomorrow'] > data['Close'], 1, 0)
        
        # Filter by start date
        data = data.loc[start_date:].copy()
        
        # Calculate technical features
        data = self._calculate_features(data)
        
        # Add market proxy features (replaces fundamentals to prevent leakage)
        data = self._add_market_proxy_features(data)
        
        # Clean Data Pipe: Handle holidays/gaps properly
        # 1. Replace infinities with NaN
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 2. Forward fill to handle holiday/gap days
        data = data.ffill()
        # 3. Backward fill for any remaining NaN at the start
        data = data.bfill()
        # 4. Drop any remaining NaN rows
        data.dropna(inplace=True)
        
        X = data[self.features]
        y = data['Target']
        
        # Check class balance
        class_counts = y.value_counts()
        print(f"\nClass distribution: Up={class_counts.get(1, 0)}, Down={class_counts.get(0, 0)}")
        
        # Time-based split (last 100 for test)
        self.X_train = X.iloc[:-100]
        self.X_test = X.iloc[-100:]
        self.y_train = y.iloc[:-100]
        self.y_test = y.iloc[-100:]
        self.data = data
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training samples: {len(self.X_train)}, Test samples: {len(self.X_test)}")
        print(f"Features used: {len(self.features)}")


    def tune_and_train_models(self, tune=True):
        """Train models with optional hyperparameter tuning."""
        
        # Calculate class weight for imbalanced data
        class_counts = self.y_train.value_counts()
        scale_pos_weight = class_counts.get(0, 1) / class_counts.get(1, 1)
        
        if tune:
            print("\n--- Tuning Random Forest ---")
            rf_params = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced']
            }
            
            tscv = TimeSeriesSplit(n_splits=5)
            rf_grid = GridSearchCV(
                RandomForestClassifier(random_state=42),
                rf_params,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            rf_grid.fit(self.X_train_scaled, self.y_train)
            self.model_RF = rf_grid.best_estimator_
            print(f"Best RF params: {rf_grid.best_params_}")
            print(f"Best RF CV score: {rf_grid.best_score_:.4f}")
            
            print("\n--- Tuning XGBoost ---")
            xgb_params = {
                'n_estimators': [200, 500],
                'learning_rate': [0.01, 0.05],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
            }
            
            xgb_grid = GridSearchCV(
                XGBClassifier(
                    random_state=42,
                    scale_pos_weight=scale_pos_weight,
                    use_label_encoder=False,
                    eval_metric='logloss'
                ),
                xgb_params,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            xgb_grid.fit(self.X_train_scaled, self.y_train)
            self.model_XGB = xgb_grid.best_estimator_
            print(f"Best XGB params: {xgb_grid.best_params_}")
            print(f"Best XGB CV score: {xgb_grid.best_score_:.4f}")
        else:
            # Use default improved parameters
            self.model_RF = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42
            )
            self.model_XGB = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42
            )
            self.model_RF.fit(self.X_train_scaled, self.y_train)
            self.model_XGB.fit(self.X_train_scaled, self.y_train)

    def train_models(self):
        """Backward compatible training without tuning."""
        self.tune_and_train_models(tune=False)

    def evaluate_models(self, show_feature_importance=True):
        """Evaluate models with detailed metrics."""
        preds_RF = self.model_RF.predict(self.X_test_scaled)
        preds_XGB = self.model_XGB.predict(self.X_test_scaled)
        
        acc_rf = accuracy_score(self.y_test, preds_RF)
        acc_xgb = accuracy_score(self.y_test, preds_XGB)
        
        print(f"\n=== Model Evaluation ===")
        print(f"Accuracy RF:  {acc_rf:.2f}")
        print(f"Accuracy XGB: {acc_xgb:.2f}")
        
        # Ensemble prediction (majority vote)
        ensemble_preds = np.round((preds_RF + preds_XGB) / 2).astype(int)
        acc_ensemble = accuracy_score(self.y_test, ensemble_preds)
        print(f"Accuracy Ensemble: {acc_ensemble:.2f}")
        
        # Show results
        results_df = pd.DataFrame({
            'Actual': self.y_test.values,
            'RF_Pred': preds_RF,
            'XGB_Pred': preds_XGB,
            'Ensemble': ensemble_preds
        }, index=self.y_test.index)
        print(f"\nLast 10 predictions:")
        print(results_df.tail(10))
        
        # Feature importance
        if show_feature_importance:
            print(f"\n=== Top 10 Features (Random Forest) ===")
            importances = pd.Series(
                self.model_RF.feature_importances_,
                index=self.features
            ).sort_values(ascending=False)
            print(importances.head(10))
        
        return acc_rf, acc_xgb, acc_ensemble

    def predict_next_day(self):
        """Predict next day movement using technical + market proxy features."""
        ticker_obj = yf.Ticker(self.ticker)
        hist = ticker_obj.history(period="max")
        
        # Remove unnecessary columns
        for col in ["Dividends", "Stock Splits"]:
            if col in hist.columns:
                del hist[col]
        
        # Calculate technical features
        hist = self._calculate_features(hist)
        
        # Add market proxy features (same as training)
        hist = self._add_market_proxy_features(hist)
        
        # Clean data pipe
        hist.replace([np.inf, -np.inf], np.nan, inplace=True)
        hist = hist.ffill()
        hist = hist.bfill()
        hist.dropna(inplace=True)
        
        # Get latest data point
        X_latest = hist[self.features].iloc[-1:]
        X_latest_scaled = self.scaler.transform(X_latest)
        
        # Predictions
        pred_RF = self.model_RF.predict(X_latest_scaled)[0]
        pred_XGB = self.model_XGB.predict(X_latest_scaled)[0]
        
        # Probabilities
        prob_RF = self.model_RF.predict_proba(X_latest_scaled)[0]
        prob_XGB = self.model_XGB.predict_proba(X_latest_scaled)[0]
        
        print(f"\n=== Next Day Prediction for {self.ticker} ===")
        print(f"RF Prediction:  {'UP' if pred_RF == 1 else 'DOWN'} (confidence: {max(prob_RF)*100:.1f}%)")
        print(f"XGB Prediction: {'UP' if pred_XGB == 1 else 'DOWN'} (confidence: {max(prob_XGB)*100:.1f}%)")
        
        # Ensemble
        avg_prob_up = (prob_RF[1] + prob_XGB[1]) / 2
        ensemble_pred = 1 if avg_prob_up > 0.5 else 0
        ensemble_conf = avg_prob_up if ensemble_pred == 1 else 1 - avg_prob_up
        print(f"Ensemble:       {'UP' if ensemble_pred == 1 else 'DOWN'} (confidence: {ensemble_conf*100:.1f}%)")
        
        return {
            "RF_Prediction": "UP" if pred_RF == 1 else "DOWN",
            "RF_Probability_Up": float(prob_RF[1]),
            "XGB_Prediction": "UP" if pred_XGB == 1 else "DOWN",
            "XGB_Probability_Up": float(prob_XGB[1]),
            "Ensemble_Prediction": "UP" if ensemble_pred == 1 else "DOWN",
            "Ensemble_Confidence": float(ensemble_conf),
            # Include market proxy features in output for dashboard display
            "Market_Proxies": self.market_proxy_features
        }


if __name__ == "__main__":
    ticker = "AMD"
    
    print("=" * 60)
    print(f"Stock Prediction for {ticker} with Fundamental Analysis")
    print("=" * 60)
    
    # Create predictor with fundamentals enabled
    predictor = StockPredictor(ticker, use_fundamentals=True)
    predictor.data_processing()
    
    # Use tune=False for faster execution (tune=True for full hyperparameter search)
    predictor.tune_and_train_models(tune=False)
    
    predictor.evaluate_models()
    predictor.predict_next_day()