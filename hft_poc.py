import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tda_pipeline import compute_rolling_distance_matrices, extract_topological_features, compute_persistence_landscapes

def run_hft_poc():
    print("--- Running High-Frequency TDA Proof of Concept ---")
    
    # 1. Fetch 1-minute intraday data
    tickers = ["SPY", "QQQ", "TLT", "GLD", "IWM"]
    print(f"Fetching 1-minute intraday data for {tickers} over the last 5 days...")
    
    # yfinance allows 1m data for max 7 days
    data = yf.download(tickers, period="5d", interval="1m")
    
    if "Adj Close" in data:
        prices = data["Adj Close"].dropna()
    else:
        prices = data["Close"].dropna()
        
    print(f"Fetched {len(prices)} rows of intraday data.")
    
    if len(prices) < 100:
        print("Not enough data fetched. Exiting HFT PoC.")
        return
        
    # 2. Compute 1-minute log returns
    returns = np.log(prices / prices.shift(1)).dropna()
    
    # 3. Rolling 60-minute window for topological features
    window_size = 60
    print(f"Computing rolling distance matrices with window = {window_size} minutes...")
    
    # To save time in PoC, we will just sample every 10th minute for the rolling window evaluation
    sampled_returns = returns.copy()
    
    dist_matrices, valid_dates = compute_rolling_distance_matrices(sampled_returns, window_size=window_size)
    
    # We only take every 5th matrix to speed up demonstration
    step = 5
    dist_matrices_subset = dist_matrices[::step]
    valid_dates_subset = valid_dates[::step]
    
    print(f"Extracting topological features for {len(dist_matrices_subset)} intraday intervals...")
    diagrams = extract_topological_features(dist_matrices_subset, epsilon=0.0)
    
    print("Computing Persistence Landscapes (L1 Norms)...")
    l1_norms, _ = compute_persistence_landscapes(diagrams)
    
    # 4. Plot the HFT topological signal
    plt.figure(figsize=(14, 6))
    plt.plot(valid_dates_subset, l1_norms, label="Intraday Topological Fragmentation (L1 Norm)", color="orange")
    plt.title("Limit Order Book / Microstructure Imbalance Proxy")
    plt.ylabel("L1 Norm")
    plt.xlabel("Time (1-minute intervals)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("hft_poc_results.png")
    print("Saved HFT PoC plot to hft_poc_results.png")

if __name__ == "__main__":
    run_hft_poc()
