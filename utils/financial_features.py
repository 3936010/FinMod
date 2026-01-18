"""
Financial Feature Extractor

Extracts fundamental financial metrics from yfinance for use in ML stock prediction models.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')


class FinancialFeatureExtractor:
    """
    Extracts fundamental financial features for stock prediction ML models.
    
    Uses yfinance to get:
    - Real-time snapshot metrics (margins, ratios, growth rates)
    - Quarterly financial trends (revenue, EPS, margins over time)
    """
    
    # Features that can be extracted
    SNAPSHOT_FEATURES = [
        'profitMargins',
        'operatingMargins', 
        'grossMargins',
        'revenueGrowth',
        'earningsGrowth',
        'debtToEquity',
        'currentRatio',
        'quickRatio',
        'trailingPE',
        'forwardPE',
        'priceToBook',
        'bookValue',
        'revenuePerShare',
        'trailingPegRatio',
    ]
    
    def __init__(self, ticker: str):
        """
        Initialize with ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
        """
        self.ticker = ticker
        self.yf_ticker = yf.Ticker(ticker)
        self._info_cache = None
        self._quarterly_cache = None
    
    @property
    def info(self) -> Dict:
        """Cached ticker info."""
        if self._info_cache is None:
            try:
                self._info_cache = self.yf_ticker.info
            except Exception as e:
                print(f"Error fetching info for {self.ticker}: {e}")
                self._info_cache = {}
        return self._info_cache
    
    def get_snapshot_features(self) -> Dict[str, float]:
        """
        Get current snapshot of fundamental metrics.
        
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        for feature in self.SNAPSHOT_FEATURES:
            value = self.info.get(feature)
            if value is not None and not pd.isna(value):
                # Handle potential infinities
                if np.isinf(value):
                    value = np.nan
                features[f'fundamental_{feature}'] = value
            else:
                features[f'fundamental_{feature}'] = np.nan
        
        return features
    
    def get_quarterly_trends(self, n_quarters: int = 4) -> Dict[str, float]:
        """
        Calculate trends from quarterly financial statements.
        
        Args:
            n_quarters: Number of quarters to analyze
            
        Returns:
            Dictionary of trend features
        """
        features = {}
        
        try:
            # Get quarterly financials
            financials = self.yf_ticker.quarterly_financials
            balance = self.yf_ticker.quarterly_balance_sheet
            
            if financials.empty:
                return self._empty_quarterly_features()
            
            # Revenue trend (QoQ growth)
            if 'Total Revenue' in financials.index:
                revenues = financials.loc['Total Revenue'].dropna().head(n_quarters)
                if len(revenues) >= 2:
                    features['fundamental_revenue_qoq_growth'] = (
                        (revenues.iloc[0] - revenues.iloc[1]) / revenues.iloc[1]
                    ) if revenues.iloc[1] != 0 else np.nan
                    features['fundamental_revenue_trend'] = self._calculate_trend(revenues)
                else:
                    features['fundamental_revenue_qoq_growth'] = np.nan
                    features['fundamental_revenue_trend'] = np.nan
            
            # Net Income trend
            if 'Net Income' in financials.index:
                net_income = financials.loc['Net Income'].dropna().head(n_quarters)
                if len(net_income) >= 2:
                    features['fundamental_ni_qoq_growth'] = (
                        (net_income.iloc[0] - net_income.iloc[1]) / abs(net_income.iloc[1])
                    ) if net_income.iloc[1] != 0 else np.nan
                else:
                    features['fundamental_ni_qoq_growth'] = np.nan
            
            # EPS trend
            if 'Diluted EPS' in financials.index:
                eps = financials.loc['Diluted EPS'].dropna().head(n_quarters)
                if len(eps) >= 2:
                    features['fundamental_eps_qoq_growth'] = (
                        (eps.iloc[0] - eps.iloc[1]) / abs(eps.iloc[1])
                    ) if eps.iloc[1] != 0 else np.nan
                    features['fundamental_eps_trend'] = self._calculate_trend(eps)
                else:
                    features['fundamental_eps_qoq_growth'] = np.nan
                    features['fundamental_eps_trend'] = np.nan
            
            # Operating margin trend
            if 'Operating Income' in financials.index and 'Total Revenue' in financials.index:
                op_income = financials.loc['Operating Income'].dropna().head(n_quarters)
                revenues = financials.loc['Total Revenue'].dropna().head(n_quarters)
                if len(op_income) >= 2 and len(revenues) >= 2:
                    current_margin = op_income.iloc[0] / revenues.iloc[0] if revenues.iloc[0] != 0 else 0
                    prev_margin = op_income.iloc[1] / revenues.iloc[1] if revenues.iloc[1] != 0 else 0
                    features['fundamental_op_margin_change'] = current_margin - prev_margin
                else:
                    features['fundamental_op_margin_change'] = np.nan
            
            # Debt level changes (from balance sheet)
            if not balance.empty and 'Total Debt' in balance.index:
                debt = balance.loc['Total Debt'].dropna().head(n_quarters)
                if len(debt) >= 2 and debt.iloc[1] != 0:
                    features['fundamental_debt_change'] = (
                        (debt.iloc[0] - debt.iloc[1]) / debt.iloc[1]
                    )
                else:
                    features['fundamental_debt_change'] = np.nan
            else:
                features['fundamental_debt_change'] = np.nan
                
        except Exception as e:
            print(f"Error calculating quarterly trends for {self.ticker}: {e}")
            return self._empty_quarterly_features()
        
        return features
    
    def _empty_quarterly_features(self) -> Dict[str, float]:
        """Return empty quarterly features dict."""
        return {
            'fundamental_revenue_qoq_growth': np.nan,
            'fundamental_revenue_trend': np.nan,
            'fundamental_ni_qoq_growth': np.nan,
            'fundamental_eps_qoq_growth': np.nan,
            'fundamental_eps_trend': np.nan,
            'fundamental_op_margin_change': np.nan,
            'fundamental_debt_change': np.nan,
        }
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """
        Calculate trend direction (-1 to 1) using linear regression slope.
        
        Args:
            series: Time series data (newest first)
            
        Returns:
            Normalized trend value
        """
        if len(series) < 2:
            return np.nan
        
        # Reverse so oldest is first
        values = series.values[::-1]
        x = np.arange(len(values))
        
        # Simple linear regression
        try:
            slope, _ = np.polyfit(x, values, 1)
            # Normalize by mean to get percentage trend
            mean_val = np.mean(values)
            if mean_val != 0:
                return slope / abs(mean_val)
            return 0.0
        except:
            return np.nan
    
    def get_all_features(self) -> Dict[str, float]:
        """
        Get all fundamental features combined.
        
        Returns:
            Dictionary with all fundamental features
        """
        features = {}
        features.update(self.get_snapshot_features())
        features.update(self.get_quarterly_trends())
        return features
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """
        Get list of all fundamental feature names.
        
        Returns:
            List of feature names
        """
        snapshot_names = [f'fundamental_{f}' for f in cls.SNAPSHOT_FEATURES]
        quarterly_names = [
            'fundamental_revenue_qoq_growth',
            'fundamental_revenue_trend',
            'fundamental_ni_qoq_growth',
            'fundamental_eps_qoq_growth',
            'fundamental_eps_trend',
            'fundamental_op_margin_change',
            'fundamental_debt_change',
        ]
        return snapshot_names + quarterly_names


if __name__ == "__main__":
    # Test the feature extractor
    extractor = FinancialFeatureExtractor("AAPL")
    
    print("=== Snapshot Features ===")
    snapshot = extractor.get_snapshot_features()
    for k, v in snapshot.items():
        print(f"  {k}: {v}")
    
    print("\n=== Quarterly Trends ===")
    trends = extractor.get_quarterly_trends()
    for k, v in trends.items():
        print(f"  {k}: {v}")
    
    print(f"\n=== All Feature Names ({len(extractor.get_feature_names())}) ===")
    print(extractor.get_feature_names())
