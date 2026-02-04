
import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager()
        self.total_cash = 100000.0  # $100k
        
    def test_standard_case(self):
        """Standard case: Price $150, ATR $5, Confidence 0.8"""
        current_price = 150.0
        atr = 5.0
        confidence = 0.8
        
        # Expected Calculations:
        # Base Risk = 100,000 * 0.01 = 1000
        # Actual Risk = 1000 * 0.8 = 800
        # Stop Loss = 150 - (2 * 5) = 140
        # Risk Per Share = 150 - 140 = 10
        # Shares = 800 / 10 = 80
        # Position Value = 80 * 150 = 12,000
        # Cap = 20,000 (12,000 < 20,000 OK)
        
        result = self.rm.calculate_entry(self.total_cash, current_price, atr, confidence)
        
        self.assertEqual(result['stop_loss_price'], 140.0)
        self.assertEqual(result['shares_to_buy'], 80)
        self.assertEqual(result['total_position_value'], 12000.0)
        self.assertEqual(result['take_profit_price'], 165.0) # 150 + 15

    def test_portfolio_cap(self):
        """Cap case: Price $1000, ATR $50, Confidence 1.0"""
        current_price = 1000.0
        atr = 50.0
        confidence = 1.0
        
        # Base Risk = 1000
        # Actual Risk = 1000
        # Stop Loss = 900
        # Risk Per Share = 100
        # Raw Shares = 1000 / 100 = 10
        # Position Value = 10 * 1000 = 10,000
        # Cap = 20,000
        # WAIT, let's trigger the cap. Risk per share needs to be small.
        
        # Let's try high confidence, low volatility (tight stop)
        atr = 10.0 # Stop loss 980. Risk/share 20.
        # Shares = 1000 / 20 = 50.
        # Position Value = 50 * 1000 = 50,000.
        # Cap = 20,000.
        # Expected Shares = 20,000 / 1000 = 20.
        
        result = self.rm.calculate_entry(self.total_cash, current_price, atr, confidence)
        
        self.assertEqual(result['shares_to_buy'], 20)
        self.assertEqual(result['total_position_value'], 20000.0)

if __name__ == '__main__':
    unittest.main()
