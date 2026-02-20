"""
=============================================================================
  Production Alpha Prediction Service (PyTorch)
  
  Model:    Global Multi-Asset GRU
  Features: Robust Financial Engineering (RSI, MACD, Volatility, etc.)
  Target:   Next-Day Log Return
  Status:   Production-Ready Inference Module
  
  Usage:
    - Train:  python dl_alpha_production.py --train
    - Predict: (Import AlphaPredictor class)
=============================================================================
"""

import json
import random
import warnings
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Suppress non-critical warnings
warnings.filterwarnings("ignore")


# =============================================================================
#  CONFIG & CONSTANTS
# =============================================================================

CONFIG = {
    # ── Assets (Training Universe) ──
    "tickers":        ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"],
    "start_date":     "2010-01-01",
    
    # ── Model Architecture (GRU) ──
    "seq_len":        60,
    "hidden_size":    128,
    "num_layers":     2,
    "dropout":        0.3,
    
    # ── Training Hparams ──
    "batch_size":     1024,
    "epochs":         60,
    "patience":       10,   # Early stopping patience
    "lr":             1e-3,
    "weight_decay":   1e-5,
    "grad_clip":      1.0,
    
    # ── Trading Logic ──
    "sizing_scale":   0.01, # For tanh sizing
    
    # ── System ──
    "seed":           42,
    "artifacts_dir":  Path("/home/sd/FinMod/models/alpha_gru_v1"),
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
#  UTILITIES
# =============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device(CONFIG["device"])


# =============================================================================
#  1. FEATURE ENGINEERING (PRODUCTION EXACT MATCH)
# =============================================================================

class FeatureEngineer:
    """
    Production-grade feature engineering.
    Must reliably process both historical batches and live updates.
    """
    
    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all alpha features from single-asset history.
        """
        df = df.copy()
        
        # 1. Returns
        df['log_return'] = np.log(df['Close']).diff()
        
        # 2. Momentum Indicators
        # RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_sig'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Stochastic (14)
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-9))
        
        # 3. Volatility Features
        # Parkinson Volatility
        const = 1.0 / (4.0 * np.log(2.0))
        df['parkinson_vol'] = np.sqrt(const * (np.log(df['High'] / df['Low']) ** 2))
        
        # ATR (14)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low'] - df['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(14).mean()
        
        # Spread
        df['spread'] = (df['High'] - df['Low']) / df['Close']
        
        # 4. Trend Features
        for w in [20, 50, 200]:
            ma = df['Close'].rolling(w).mean()
            df[f'price_to_MA{w}'] = df['Close'] / ma
        
        df['MA20_slope'] = df['Close'].rolling(20).mean().diff(5) / df['Close']
        
        # 5. Target (Next Day Return) - ONLY FOR TRAINING
        df['target'] = df['log_return'].shift(-1)
        
        return df

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> List[str]:
        exclude = {'Open', 'High', 'Low', 'Close', 'Volume', 'target'}
        features = [c for c in df.columns if c not in exclude and not c.endswith('_ret')]
        return sorted(features) # Deterministic order for inference


# =============================================================================
#  2. MODEL ARCHITECTURE (GRU GLOBAL)
# =============================================================================

class GlobalGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1) # Regression output
        self._init_weights()
        
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


# =============================================================================
#  3. TRAINING PIPELINE
# =============================================================================

class ProductionTrainer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_device()
        print(f"  [System] Device: {self.device}")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        print("  [Data] Loading assets...")
        tickers = CONFIG["tickers"]
        start_date = CONFIG["start_date"]
        
        data_dict = {}
        for t in tickers:
            df = yf.download(t, start=start_date, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if len(df) > 500:
                data_dict[t] = df
                
        return data_dict

    def train(self):
        set_seed(CONFIG["seed"])
        
        # 1. Data Loading & Feature Engineering
        data_dict = self.load_data()
        
        processed_data = [] # List of (X_raw, y_raw) arrays
        feature_cols = []
        
        print("  [Data] Engineering features for global training...")
        for ticker, df in data_dict.items():
            df_feats = FeatureEngineer.compute_features(df)
            df_feats.dropna(inplace=True)
            
            if not feature_cols:
                feature_cols = FeatureEngineer.get_feature_columns(df_feats)
                # Save feature list for production inference
                with open(self.output_dir / "feature_columns.json", "w") as f:
                    json.dump(feature_cols, f)
            
            X_asset = df_feats[feature_cols].values.astype(np.float32)
            y_asset = df_feats['target'].values.astype(np.float32)
            
            processed_data.append((X_asset, y_asset))
            
        # 2. Fit Scaler (Robust to Outliers)
        print("  [Scaler] Fitting RobustScaler on full history...")
        all_X = np.concatenate([p[0] for p in processed_data], axis=0)
        scaler = RobustScaler()
        scaler.fit(all_X)
        
        # Save Scaler
        with open(self.output_dir / "robust_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
            
        # 3. Windowing
        X_list, y_list = [], []
        seq_len = CONFIG["seq_len"]
        
        for X_raw, y_raw in processed_data:
            X_scaled = scaler.transform(X_raw)
            # Sliding window
            num_samples = len(X_scaled) - seq_len
            for i in range(num_samples):
                X_list.append(X_scaled[i : i+seq_len])
                y_list.append(y_raw[i+seq_len])
                
        X_train = np.array(X_list)
        y_train = np.array(y_list).reshape(-1, 1)
        
        # Convert to Tensor
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(), 
            torch.from_numpy(y_train).float()
        )
        
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
        
        # 4. Model Setup
        input_dim = len(feature_cols)
        model = GlobalGRU(
            input_dim, CONFIG["hidden_size"], 
            CONFIG["num_layers"], CONFIG["dropout"]
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CONFIG["lr"], 
            weight_decay=CONFIG["weight_decay"]
        )
        criterion = nn.MSELoss()
        scaler_amp = GradScaler()
        
        # 5. Training Loop
        print(f"  [Training] Starting GRU training ({CONFIG['epochs']} epochs)...")
        best_loss = float('inf')
        patience = 0
        
        for epoch in range(CONFIG["epochs"]):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                with autocast():
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                
                scaler_amp.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                scaler_amp.step(optimizer)
                scaler_amp.update()
                train_loss += loss.item() * len(X_batch)
            
            # Val Check
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    val_loss += criterion(model(X_batch), y_batch).item() * len(X_batch)
            
            val_loss /= len(val_ds)
            print(f"    Epoch {epoch+1:02d} | Val Loss: {val_loss:.6f}")
            
            # Early Stopping & Saving
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), self.output_dir / "gru_global_model.pt")
                with open(self.output_dir / "config.json", "w") as f:
                    json.dump(CONFIG, f, indent=2, default=str)
            else:
                patience += 1
                if patience >= CONFIG["patience"]:
                    print("    [Stop] Early stopping triggered.")
                    break
        
        print(f"  [Done] Model saved to {self.output_dir}")


# =============================================================================
#  4. PRODUCTION INFERENCE MODULE
# =============================================================================

class AlphaPredictor:
    """
    Clean, thread-safe inference wrapper for the Alpha Model.
    Designed for integration into a trading loop.
    """
    
    def __init__(self, model_dir: Union[str, Path]):
        self.model_dir = Path(model_dir)
        self.device = get_device()  # Unified device handling
        self._load_artifacts()
        
    def _load_artifacts(self):
        # 1. Config
        with open(self.model_dir / "config.json", "r") as f:
            self.config = json.load(f)
            
        # 2. Feature Columns
        with open(self.model_dir / "feature_columns.json", "r") as f:
            self.feature_cols = json.load(f)
            
        # 3. Scaler
        with open(self.model_dir / "robust_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
            
        # 4. Model
        input_dim = len(self.feature_cols)
        self.model = GlobalGRU(
            input_dim, self.config["hidden_size"], 
            self.config["num_layers"], self.config["dropout"]
        )
        self.model.load_state_dict(
            torch.load(self.model_dir / "gru_global_model.pt", map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        
    def predict_next_return(self, df: pd.DataFrame) -> float:
        """
        Predict next-day log return for a single asset.
        """
        if len(df) < 300:
            raise ValueError(f"Insufficient history. Need > 300 rows for MA200+buffer, got {len(df)}")
            
        # 1. Feature Engineering
        df_feats = FeatureEngineer.compute_features(df)
        
        # 2. Extract last sequence
        try:
            latest_window = df_feats.iloc[-self.config["seq_len"]:][self.feature_cols]
        except KeyError as e:
            raise KeyError(f"Missing feature columns in inference: {e}")
            
        # Check for NaNs
        if latest_window.isna().any().any():
            latest_window = latest_window.ffill().fillna(0)
            if latest_window.isna().any().any():
                 raise ValueError("NaNs found in features even after ffill.")
            
        # 3. Scale
        X_raw = latest_window.values
        X_scaled = self.scaler.transform(X_raw)
        
        # 4. Tensor
        X_tensor = torch.from_numpy(X_scaled).float().unsqueeze(0).to(self.device)
        
        if torch.isnan(X_tensor).any():
             raise ValueError("Input tensor contains NaNs after scaling")
             
        # 5. Predict
        with torch.no_grad():
            pred = self.model(X_tensor).item()
            
        if np.isnan(pred):
             if torch.isnan(self.model.fc.weight).any():
                 raise ValueError("Model weights contain NaNs!")
             raise ValueError("Model predicted NaN despite clean input.")
             
        return pred

    def predict_position(self, df: pd.DataFrame) -> float:
        """
        Returns a trading position size in range [-1.0, 1.0].
        """
        raw_pred = self.predict_next_return(df)
        scale = self.config.get("sizing_scale", 0.01)
        return float(np.tanh(raw_pred / scale))

    def predict_next_price(self, ticker: str, period: str = "5y") -> Dict:
        """
        Predict the next trading day's closing price for a given ticker.
        """
        # 1. Download data
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 300:
            raise ValueError(
                f"Insufficient data for {ticker}: got {len(df)} rows, need > 300."
            )

        # 2. Predict (Single execution optimized)
        pred_log_return = self.predict_next_return(df)
        
        # Optimized: Calculate position directly from prediction
        scale = self.config.get("sizing_scale", 0.01)
        position = float(np.tanh(pred_log_return / scale))

        current_price = float(df["Close"].iloc[-1])
        predicted_price = current_price * np.exp(pred_log_return)
        pct_change = (predicted_price - current_price) / current_price * 100.0

        # 3. Determine direction
        if pct_change > 0.05:
            direction = "Bullish"
        elif pct_change < -0.05:
            direction = "Bearish"
        else:
            direction = "Neutral"

        as_of = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])

        return {
            "ticker":           ticker,
            "current_price":    round(current_price, 2),
            "predicted_price":  round(predicted_price, 2),
            "predicted_return": round(pred_log_return, 6),
            "pct_change":       round(pct_change, 3),
            "position_size":    round(position, 4),
            "direction":        direction,
            "as_of_date":       as_of,
        }


# =============================================================================
#  CLI ENTRY POINT (TRAINING ONLY)
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Alpha Model Trainer")
    parser.add_argument("--train", action="store_true", help="Run full model training")
    parser.add_argument("--test-inference", action="store_true", help="Run a test inference on AAPL")
    args = parser.parse_args()
    
    if args.train:
        trainer = ProductionTrainer(CONFIG["artifacts_dir"])
        trainer.train()
        
    if args.test_inference:
        print("\n[Testing Inference Module]")
        try:
            predictor = AlphaPredictor(CONFIG["artifacts_dir"])
            df = yf.download("AAPL", period="5y", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            pred = predictor.predict_next_return(df)
            pos = predictor.predict_position(df)
            
            print(f"  AAPL Prediction (Next Day): {pred:.6f}")
            print(f"  AAPL Position Sizing (tanh): {pos:.4f}")
            print("  ✅ Inference Test Passed")
        except Exception as e:
            print(f"  ❌ Inference Failed: {e}")
