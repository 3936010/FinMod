import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import pandas as pd
import numpy as np
import traceback
import joblib  # FIX #9: Added joblib for model persistence
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import warnings
import json
from datetime import datetime

# FIX #6: Suppress only specific known-harmless warnings instead of all warnings globally
warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class StockPredictor:
    # FIX #5: Removed deprecated `use_fundamentals` parameter entirely.
    # It was stored but never read — dead code that caused confusion.
    def __init__(self, ticker):
        self.ticker = ticker
        self.model_RF = None
        self.model_XGB = None
        self.model_LR = None
        self.model_GB = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data = None
        self.market_proxy_features = {}  # Latest market proxy values for display
        self.model_weights = {'RF': 0.25, 'XGB': 0.25, 'LR': 0.25, 'GB': 0.25}  # Default weights for ensemble

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
            'VolAdj_Return',      # Return divided by 20-day ATR/StdDev (shifted to prevent leakage)
            'PriceVolume_Ratio'   # Close / 20-day average volume (shifted to prevent leakage)
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
        data['RS'] = data['AvgGain'] / (data['AvgLoss'] + 1e-10)
        data['RSI'] = 100 - (100 / (1 + data['RS']))

        # Momentum
        data['Momentum'] = data['Close'] - data['Close'].shift(5)

        # Gap (overnight)
        data['Gap'] = data['Open'] - data['Close'].shift(1)

        # Volume indicators
        # Shift rolling mean to avoid using current volume in the denominator (leakage prevention)
        data['VolumeSpike'] = data['Volume'] / (data['Volume'].rolling(20).mean().shift(1) + 1e-10)
        data['Volume_MA_Ratio'] = data['Volume'] / (data['Volume'].rolling(10).mean().shift(1) + 1e-10)

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
        - Beta_90d: 90-day rolling covariance/variance with SPY (market sensitivity)
        - Alpha_90d: Excess return relative to Beta-adjusted SPY return
        - VolAdj_Return: Prior day's return divided by prior 20-day volatility (risk-adjusted momentum)
          FIX #4: Both rolling_vol and return are now shifted by 1 to prevent today's data leaking in.
        - PriceVolume_Ratio: Prior close divided by prior 20-day average volume (liquidity proxy)
          FIX #4: avg_volume is now shifted by 1 to prevent today's volume leaking in.
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
            traceback.print_exc()
            data['Beta_90d'] = 1.0  # Default to market beta
            data['Alpha_90d'] = 0.0  # Default to no alpha

        # FIX #4: VolAdj_Return — shift rolling vol by 1 so today's return is not divided
        # by a volatility window that includes today. Consistent with volume feature treatment.
        rolling_vol = data['Return'].rolling(window=20).std().shift(1)
        data['VolAdj_Return'] = data['Return'].shift(1) / rolling_vol

        # FIX #4: PriceVolume_Ratio — shift avg_volume by 1 to prevent today's volume leaking in.
        avg_volume = data['Volume'].rolling(window=20).mean().shift(1)
        data['PriceVolume_Ratio'] = data['Close'].shift(1) / avg_volume
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

    def data_processing(self, period="max", start_date="2015-01-01", test_pct=0.15):
        """
        Process data with technical features and market proxies.

        Training Constraint: Uses ONLY technical_features + market_proxy_features.
        Fundamentals are NOT used in training to prevent historical data leakage.

        FIX #3: test_pct parameter replaces hard-coded 100-row test split.
        Default 15% of data is used for test, giving more robust out-of-sample evaluation.
        A validation split (val_pct) is also carved out from training for ensemble weighting
        and XGB early stopping — keeping the test set completely clean.
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
        # 3. Drop any remaining NaN rows (avoid bfill to prevent look-ahead bias)
        data.dropna(inplace=True)

        X = data[self.features]
        y = data['Target']

        # Check class balance
        class_counts = y.value_counts()
        print(f"\nClass distribution: Up={class_counts.get(1, 0)}, Down={class_counts.get(0, 0)}")

        # FIX #3: Percentage-based time split instead of fixed 100-row test split.
        # Also creates a validation set from training data for ensemble weighting
        # and XGB early stopping — keeping test set completely untouched.
        n = len(X)
        n_test = int(n * test_pct)
        n_val = int(n * 0.10)   # 10% validation carved from training end
        n_train = n - n_test - n_val

        self.X_train = X.iloc[:n_train]
        self.X_val = X.iloc[n_train:n_train + n_val]
        self.X_test = X.iloc[n_train + n_val:]

        self.y_train = y.iloc[:n_train]
        self.y_val = y.iloc[n_train:n_train + n_val]
        self.y_test = y.iloc[n_train + n_val:]

        self.data = data

        # Scale features — fit only on train to prevent leakage
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"Training samples: {len(self.X_train)}, Validation samples: {len(self.X_val)}, Test samples: {len(self.X_test)}")
        print(f"Features used: {len(self.features)}")

    def tune_and_train_models(self, tune=True):
        """
        Train models with optional hyperparameter tuning.

        FIX #1: XGBoost early stopping now uses self.X_val_scaled / self.y_val
                (validation set), NOT the test set, to prevent test-set leakage.
        FIX #2: Ensemble weights are derived from the validation set in evaluate_models(),
                so the test set remains a clean, final held-out evaluation.
        FIX #7: Removed deprecated `use_label_encoder=False` from XGBClassifier
                (removed in XGBoost >= 1.6).
        """
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

            # FIX #7: Removed use_label_encoder=False (deprecated/removed in XGBoost >= 1.6)
            xgb_grid = GridSearchCV(
                XGBClassifier(
                    random_state=42,
                    scale_pos_weight=scale_pos_weight,
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

            print("\n--- Tuning Logistic Regression ---")
            lr_params = {
                'C': [0.1, 1.0, 10.0],
                'class_weight': ['balanced']
            }
            lr_grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), lr_params, cv=tscv, scoring='accuracy', n_jobs=-1, verbose=0)
            lr_grid.fit(self.X_train_scaled, self.y_train)
            self.model_LR = lr_grid.best_estimator_
            print(f"Best LR params: {lr_grid.best_params_}")
            print(f"Best LR CV score: {lr_grid.best_score_:.4f}")

            print("\n--- Tuning Gradient Boosting ---")
            gb_params = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.05],
                'max_depth': [3, 5]
            }
            gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=tscv, scoring='accuracy', n_jobs=-1, verbose=0)
            gb_grid.fit(self.X_train_scaled, self.y_train)
            self.model_GB = gb_grid.best_estimator_
            print(f"Best GB params: {gb_grid.best_params_}")
            print(f"Best GB CV score: {gb_grid.best_score_:.4f}")
        else:
            # Use default improved parameters
            self.model_RF = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42
            )
            # FIX #7: Removed use_label_encoder=False (deprecated/removed in XGBoost >= 1.6)
            self.model_XGB = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                early_stopping_rounds=10,
                eval_metric='logloss'
            )
            self.model_LR = LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=42)
            self.model_GB = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)

            self.model_RF.fit(self.X_train_scaled, self.y_train)

            # FIX #1: Early stopping uses the VALIDATION set, not the test set.
            # Previously eval_set=[(self.X_test_scaled, self.y_test)] caused test-set leakage
            # because XGBoost used test labels to decide when to stop training.
            self.model_XGB.fit(
                self.X_train_scaled,
                self.y_train,
                eval_set=[(self.X_val_scaled, self.y_val)],  # <-- FIXED: val set, not test set
                verbose=False
            )
            self.model_LR.fit(self.X_train_scaled, self.y_train)
            self.model_GB.fit(self.X_train_scaled, self.y_train)

        # FIX #2: Compute ensemble weights on the VALIDATION set immediately after training,
        # so they are calibrated without touching the test set.
        self._calibrate_ensemble_weights()

    def _calibrate_ensemble_weights(self):
        """
        FIX #2: Derive ensemble weights from validation set accuracy.
        Called right after training so test set is never used for weight calibration.
        Previously, weights were computed inside evaluate_models() using test-set accuracy,
        which contaminated the reported test metrics.
        """
        val_acc_rf  = accuracy_score(self.y_val, self.model_RF.predict(self.X_val_scaled))
        val_acc_xgb = accuracy_score(self.y_val, self.model_XGB.predict(self.X_val_scaled))
        val_acc_lr  = accuracy_score(self.y_val, self.model_LR.predict(self.X_val_scaled))
        val_acc_gb  = accuracy_score(self.y_val, self.model_GB.predict(self.X_val_scaled))

        total = val_acc_rf + val_acc_xgb + val_acc_lr + val_acc_gb
        if total > 0:
            self.model_weights['RF']  = val_acc_rf  / total
            self.model_weights['XGB'] = val_acc_xgb / total
            self.model_weights['LR']  = val_acc_lr  / total
            self.model_weights['GB']  = val_acc_gb  / total

        print(f"\nEnsemble weights (from validation set): RF={self.model_weights['RF']:.3f}, "
              f"XGB={self.model_weights['XGB']:.3f}, LR={self.model_weights['LR']:.3f}, "
              f"GB={self.model_weights['GB']:.3f}")

    def train_models(self):
        """Backward compatible training without tuning."""
        self.tune_and_train_models(tune=False)

    def evaluate_models(self, show_feature_importance=True):
        """
        Evaluate models on the held-out test set with detailed metrics.

        FIX #2: Ensemble weights are no longer computed here from test accuracy.
        They are now pre-calibrated on the validation set in _calibrate_ensemble_weights(),
        which is called at the end of tune_and_train_models(). This makes test accuracy
        a genuinely clean out-of-sample estimate.
        """
        preds_RF  = self.model_RF.predict(self.X_test_scaled)
        preds_XGB = self.model_XGB.predict(self.X_test_scaled)
        preds_LR  = self.model_LR.predict(self.X_test_scaled)
        preds_GB  = self.model_GB.predict(self.X_test_scaled)

        acc_rf  = accuracy_score(self.y_test, preds_RF)
        acc_xgb = accuracy_score(self.y_test, preds_XGB)
        acc_lr  = accuracy_score(self.y_test, preds_LR)
        acc_gb  = accuracy_score(self.y_test, preds_GB)

        print(f"\n=== Model Evaluation (Test Set) ===")
        print(f"Accuracy RF:  {acc_rf:.2f}")
        print(f"Accuracy XGB: {acc_xgb:.2f}")
        print(f"Accuracy LR:  {acc_lr:.2f}")
        print(f"Accuracy GB:  {acc_gb:.2f}")

        # Get probabilities for test set using pre-calibrated weights (no test leakage)
        probs_RF  = self.model_RF.predict_proba(self.X_test_scaled)[:, 1]
        probs_XGB = self.model_XGB.predict_proba(self.X_test_scaled)[:, 1]
        probs_LR  = self.model_LR.predict_proba(self.X_test_scaled)[:, 1]
        probs_GB  = self.model_GB.predict_proba(self.X_test_scaled)[:, 1]

        # Calculate weighted ensemble probabilities using validation-calibrated weights
        ensemble_probs = (probs_RF  * self.model_weights['RF'] +
                          probs_XGB * self.model_weights['XGB'] +
                          probs_LR  * self.model_weights['LR'] +
                          probs_GB  * self.model_weights['GB'])
        ensemble_preds = (ensemble_probs > 0.5).astype(int)

        acc_ensemble = accuracy_score(self.y_test, ensemble_preds)
        print(f"Accuracy Ensemble: {acc_ensemble:.2f}")

        # Show results
        results_df = pd.DataFrame({
            'Actual': self.y_test.values,
            'RF_Pred': preds_RF,
            'XGB_Pred': preds_XGB,
            'LR_Pred': preds_LR,
            'GB_Pred': preds_GB,
            'Ensemble_Prob': ensemble_probs,
            'Ensemble_Pred': ensemble_preds
        }, index=self.y_test.index)
        print(f"\nLast 10 predictions:")
        print(results_df.tail(10))

        # Feature importance
        if show_feature_importance:
            print(f"\n=== Top 10 Features (Random Forest) ===")
            importances_rf = pd.Series(
                self.model_RF.feature_importances_,
                index=self.features
            ).sort_values(ascending=False)
            print(importances_rf.head(10))

            print(f"\n=== Top 10 Features (XGBoost) ===")
            importances_xgb = pd.Series(
                self.model_XGB.feature_importances_,
                index=self.features
            ).sort_values(ascending=False)
            print(importances_xgb.head(10))

        return acc_rf, acc_xgb, acc_lr, acc_gb, acc_ensemble

    def predict_next_day(self):
        """Predict next day movement using technical + market proxy features."""
        # Optimization: Use already processed data if available to avoid re-downloading
        if self.data is not None and not self.data.empty:
            X_latest = self.data[self.features].iloc[-1:]
        else:
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
            hist.dropna(inplace=True)

            # Get latest data point
            X_latest = hist[self.features].iloc[-1:]

        X_latest_scaled = self.scaler.transform(X_latest)

        # Predictions
        pred_RF  = self.model_RF.predict(X_latest_scaled)[0]
        pred_XGB = self.model_XGB.predict(X_latest_scaled)[0]
        pred_LR  = self.model_LR.predict(X_latest_scaled)[0]
        pred_GB  = self.model_GB.predict(X_latest_scaled)[0]

        # Probabilities
        prob_RF  = self.model_RF.predict_proba(X_latest_scaled)[0]
        prob_XGB = self.model_XGB.predict_proba(X_latest_scaled)[0]
        prob_LR  = self.model_LR.predict_proba(X_latest_scaled)[0]
        prob_GB  = self.model_GB.predict_proba(X_latest_scaled)[0]

        print(f"\n=== Next Day Prediction for {self.ticker} ===")
        print(f"RF Prediction:  {'UP' if pred_RF == 1 else 'DOWN'} (confidence: {max(prob_RF)*100:.1f}%)")
        print(f"XGB Prediction: {'UP' if pred_XGB == 1 else 'DOWN'} (confidence: {max(prob_XGB)*100:.1f}%)")
        print(f"LR Prediction:  {'UP' if pred_LR == 1 else 'DOWN'} (confidence: {max(prob_LR)*100:.1f}%)")
        print(f"GB Prediction:  {'UP' if pred_GB == 1 else 'DOWN'} (confidence: {max(prob_GB)*100:.1f}%)")

        # Ensemble with validation-calibrated weights (FIX #2)
        w_rf  = self.model_weights.get('RF', 0.25)
        w_xgb = self.model_weights.get('XGB', 0.25)
        w_lr  = self.model_weights.get('LR', 0.25)
        w_gb  = self.model_weights.get('GB', 0.25)
        avg_prob_up = (prob_RF[1] * w_rf + prob_XGB[1] * w_xgb + prob_LR[1] * w_lr + prob_GB[1] * w_gb)
        ensemble_pred = 1 if avg_prob_up > 0.5 else 0
        ensemble_conf = avg_prob_up if ensemble_pred == 1 else 1 - avg_prob_up
        print(f"Ensemble:       {'UP' if ensemble_pred == 1 else 'DOWN'} (confidence: {ensemble_conf*100:.1f}%)")

        return {
            "RF_Prediction": "UP" if pred_RF == 1 else "DOWN",
            "RF_Probability_Up": float(prob_RF[1]),
            "XGB_Prediction": "UP" if pred_XGB == 1 else "DOWN",
            "XGB_Probability_Up": float(prob_XGB[1]),
            "LR_Prediction": "UP" if pred_LR == 1 else "DOWN",
            "LR_Probability_Up": float(prob_LR[1]),
            "GB_Prediction": "UP" if pred_GB == 1 else "DOWN",
            "GB_Probability_Up": float(prob_GB[1]),
            "Ensemble_Prediction": "UP" if ensemble_pred == 1 else "DOWN",
            "Ensemble_Confidence": float(ensemble_conf),
            # Include market proxy features in output for dashboard display
            "Market_Proxies": self.market_proxy_features,
            "Model_Weights": self.model_weights,
            "Date": datetime.now().strftime("%Y-%m-%d")
        }

    def save_models(self, path: str):
        """
        FIX #9: Persist trained models and scaler to disk using joblib.
        Avoids re-training and re-downloading SPY data on every run.

        Usage:
            predictor.save_models("models/qcom_predictor.pkl")
        """
        payload = {
            'model_RF': self.model_RF,
            'model_XGB': self.model_XGB,
            'model_LR': self.model_LR,
            'model_GB': self.model_GB,
            'scaler': self.scaler,
            'model_weights': self.model_weights,
            'features': self.features,
            'ticker': self.ticker,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump(payload, path)
        print(f"Models saved to {path}")

    def load_models(self, path: str):
        """
        FIX #9: Restore trained models and scaler from disk.

        Usage:
            predictor = StockPredictor("QCOM")
            predictor.load_models("models/qcom_predictor.pkl")
            prediction = predictor.predict_next_day()
        """
        payload = joblib.load(path)
        self.model_RF      = payload['model_RF']
        self.model_XGB     = payload['model_XGB']
        self.model_LR      = payload['model_LR']
        self.model_GB      = payload['model_GB']
        self.scaler        = payload['scaler']
        self.model_weights = payload['model_weights']
        self.features      = payload['features']
        self.ticker        = payload['ticker']
        print(f"Models loaded from {path}")

    def save_final_prediction(self, prediction_dict, save=True):
        """Append the daily prediction to a tracking file."""
        if not save:
            return

        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        tracking_file = os.path.join(results_dir, "daily_predictions.csv")

        # Flatten dictionary for CSV
        row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": self.ticker,
            **prediction_dict
        }
        # Remove nested dictionary if present (Market_Proxies)
        if "Market_Proxies" in row:
            for k, v in row["Market_Proxies"].items():
                row[f"Proxy_{k}"] = v
            del row["Market_Proxies"]

        if "Model_Weights" in row:
            for k, v in row["Model_Weights"].items():
                row[f"Weight_{k}"] = v
            del row["Model_Weights"]

        df = pd.DataFrame([row])

        if not os.path.exists(tracking_file):
            df.to_csv(tracking_file, index=False)
        else:
            df.to_csv(tracking_file, mode='a', header=False, index=False)
        print(f"Daily prediction appended to {tracking_file}")

        # FIX #8: Simplified JSON structure — plain dict keyed by ticker instead of [{}] singleton.
        # Old pattern: data = [{}]; data[0][ticker] = ... was fragile and non-standard.
        json_file = os.path.join(results_dir, "market_predictions.json")
        ticker_store = {}

        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    content = json.load(f)
                    if isinstance(content, dict):
                        ticker_store = content
            except (json.JSONDecodeError, AttributeError):
                pass

        # Update the latest state for this ticker
        ticker_store[self.ticker] = prediction_dict

        with open(json_file, 'w') as f:
            json.dump(ticker_store, f, indent=4)
        print(f"Latest prediction updated in {json_file}")


if __name__ == "__main__":
    ticker = "QCOM"

    print("=" * 60)
    # FIX #10: Updated headline — fundamentals are not used in training
    print(f"Stock Prediction for {ticker} (Technical + Market Proxy Features)")
    print("=" * 60)

    # FIX #5: Removed use_fundamentals parameter (was dead code, never read)
    predictor = StockPredictor(ticker)
    predictor.data_processing()

    # Use tune=False for faster execution (tune=True for full hyperparameter search)
    predictor.tune_and_train_models(tune=False)

    predictor.evaluate_models()
    prediction = predictor.predict_next_day()
    predictor.save_final_prediction(prediction, save=False)

    # FIX #9: Optionally persist trained models to skip re-training next run
    # predictor.save_models("models/qcom_predictor.pkl")
