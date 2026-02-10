import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from scripts.news import NewsAnalyzer
    print("Attempting to instantiate NewsAnalyzer...")
    analyzer = NewsAnalyzer("AAPL")
    print("NewsAnalyzer instantiated successfully!")
    
    # Optional: Try to fetch news to ensure the graceful degradation works deep in the call stack
    # print("Attempting to fetch news...")
    # analyzer.analyze_news(start_date='2025-01-01', end_date='2025-01-02') 
    # Commented out to avoid making actual API calls if not necessary for basic verification
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
