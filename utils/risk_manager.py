from pydantic import BaseModel

class RiskManager:
    """
    Risk Management Engine for an automated trading system.
    Calculates optimal trade size and exit points.
    """
    def __init__(self):
        pass

    def calculate_entry(self, total_cash: float, current_price: float, atr: float, llm_confidence: float) -> dict:
        """
        Calculates optimal trade size and exit points.

        Args:
            total_cash: Current available portfolio balance.
            current_price: The current market price of the ticker.
            atr: The 14-day Average True Range.
            llm_confidence: The confidence score (0.0 - 1.0) from MarketPrediction.

        Returns:
            dict: {
                "shares_to_buy": int,
                "stop_loss_price": float,
                "take_profit_price": float,
                "total_position_value": float
            }
        """
        # 1. Calculate Base Risk: 1% of total_cash
        base_risk = total_cash * 0.01

        # 2. Adjust for Confidence: Actual_Risk = Base_Risk * llm_confidence
        actual_risk = base_risk * llm_confidence

        # 3. Calculate Stop Loss: current_price - (2 * atr)
        # Ensure stop loss is not negative
        stop_loss_price = max(0.0, current_price - (2 * atr))

        # 4. Calculate Risk Per Share: current_price - Stop_Loss_Price
        risk_per_share = current_price - stop_loss_price
        
        # Avoid division by zero if risk_per_share is 0 (unlikely but safe to handle)
        if risk_per_share <= 0:
            return {
                "shares_to_buy": 0,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": 0.0,
                "total_position_value": 0.0
            }

        # 5. Calculate Shares: Shares_To_Buy = Actual_Risk / Risk_Per_Share
        raw_shares = actual_risk / risk_per_share
        shares_to_buy = int(raw_shares)

        # 6. Apply Portfolio Cap: Max 20% of total_cash
        max_position_value = total_cash * 0.20
        current_position_value = shares_to_buy * current_price

        if current_position_value > max_position_value:
            # Recalculate shares to fit cap
            shares_to_buy = int(max_position_value / current_price)

        # Calculate Move-based Take Profit (Reward 1.5:1 -> 3 * ATR)
        take_profit_price = current_price + (3 * atr)
        
        final_position_value = shares_to_buy * current_price

        return {
            "shares_to_buy": shares_to_buy,
            "stop_loss_price": round(stop_loss_price, 2),
            "take_profit_price": round(take_profit_price, 2),
            "total_position_value": round(final_position_value, 2)
        }
