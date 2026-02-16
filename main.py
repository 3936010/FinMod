from scripts.ml_market_movement import StockPredictor
ticker_list = ['QCOM', 'NVDA', 'AAPL', 'AMZN', 'AMD', 'TSCO', 'WMT',]
for ticker in ticker_list:
    # Create predictor with fundamentals enabled
    predictor = StockPredictor(ticker, use_fundamentals=True)
    predictor.data_processing()
    # Use tune=False for faster execution (tune=True for full hyperparameter search)
    predictor.tune_and_train_models(tune=False)
    predictor.evaluate_models()
    prediction = predictor.predict_next_day()
    predictor.save_final_prediction(prediction)