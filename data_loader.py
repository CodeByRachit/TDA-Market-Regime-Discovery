import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(tickers, start_date, end_date):
    """
    Fetches adjusted close prices for a list of tickers.
    """
    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    # Fetch data
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # We only need 'Adj Close'
    if 'Adj Close' in data:
        prices = data['Adj Close']
    elif 'Close' in data:
        prices = data['Close']
    else:
        raise ValueError("No 'Adj Close' or 'Close' column found in fetched data.")
    
    # Handle single ticker case where yf returns a Series instead of DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
        
    # Forward fill missing data, then drop any remaining NaNs
    prices = prices.ffill().dropna()
    
    return prices

def compute_log_returns(prices):
    """
    Computes daily logarithmic returns.
    """
    # Log return formula: ln(P_t / P_{t-1})
    log_returns = np.log(prices / prices.shift(1))
    
    # Drop the first row which will be NaN due to shift(1)
    log_returns = log_returns.dropna()
    
    return log_returns

if __name__ == "__main__":
    # Test data loader
    tickers = ["SPY", "QQQ", "TLT", "GLD", "EFA"]
    prices = fetch_data(tickers, "2019-01-01", "2020-12-31")
    returns = compute_log_returns(prices)
    print("Prices head:")
    print(prices.head())
    print("\nLog Returns head:")
    print(returns.head())
