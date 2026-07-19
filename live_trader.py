import time
import ccxt
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from tda_pipeline import compute_rolling_distance_matrices, extract_topological_features

def fetch_recent_data(exchange, symbols, timeframe='1h', limit=60):
    """Fetches the most recent `limit` candles for the given symbols."""
    closes = {}
    for sym in symbols:
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                closes[sym] = df['close']
                break
            except ccxt.NetworkError as e:
                print(f"[!] Network error fetching {sym}: {e}. Retrying in 10 seconds...")
                time.sleep(10)
            except ccxt.RateLimitExceeded as e:
                print(f"[!] Rate limit exceeded fetching {sym}: {e}. Sleeping for 60 seconds...")
                time.sleep(60)
            except Exception as e:
                print(f"[!] Unexpected error fetching {sym}: {e}. Retrying in 10 seconds...")
                time.sleep(10)
    
    df = pd.DataFrame(closes)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

def run_live_trader():
    print("==================================================")
    print("      TOPOLOGICAL MACRO-REGIME PAPER TRADER       ")
    print("==================================================")
    
    try:
        clf = joblib.load('rf_model.joblib')
        pi = joblib.load('pi_transformer.joblib')
        print("[+] Successfully loaded pre-trained models.")
    except Exception as e:
        print(f"[-] Failed to load models: {e}. Run train_crypto_model.py first.")
        return
        
    exchange = ccxt.binance({'enableRateLimit': True})
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
    
    portfolio_value = 10000.0
    current_allocation = None # None means we haven't taken an initial position yet
    taker_fee = 0.001 # 0.1% fee
    
    print(f"Starting Portfolio Value: ${portfolio_value:,.2f}")
    print("Entering polling loop (Updates every hour)...")
    
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching latest market data...")
            # We need 60 periods for the rolling window, plus 1 period for the return calculation (61 total)
            df = fetch_recent_data(exchange, symbols, timeframe='1h', limit=61)
            
            if len(df) < 61:
                print(f"[!] Not enough data returned (got {len(df)}). Retrying in 60s...")
                time.sleep(60)
                continue
                
            returns = np.log(df / df.shift(1)).dropna()
            
            # 1. Compute single distance matrix for the most recent window (last 60 returns)
            # compute_rolling_distance_matrices computes for all rolling windows. We only need the last one.
            # But we can just pass the whole 60 rows and it will return 1 matrix.
            dist_matrices, _ = compute_rolling_distance_matrices(returns, window_size=60)
            
            if len(dist_matrices) == 0:
                print("[!] Failed to compute distance matrix. Skipping iteration.")
                time.sleep(60)
                continue
                
            # Take only the last one
            latest_dist_matrix = [dist_matrices[-1]]
            
            # 2. Extract Topological Features
            diagrams = extract_topological_features(latest_dist_matrix, epsilon=0.0)
            
            # 3. Persistence Image Translation
            images = pi.transform(diagrams)
            X_live = images.reshape(images.shape[0], -1)
            
            # 4. Predict Regime
            prediction = clf.predict(X_live)[0]
            
            # Prediction map: 1 = Crash, 0 = Normal
            target_allocation = 0.20 if prediction == 1 else 1.00
            regime_name = "CRASH / FRAGMENTATION" if prediction == 1 else "NORMAL MARKET"
            
            print(f"  -> Topological Regime: {regime_name}")
            print(f"  -> Target Allocation: {target_allocation*100:.0f}% Equity")
            
            # 5. Execute Trade & Apply Fees
            if current_allocation is None:
                # Initial position
                current_allocation = target_allocation
                fees_paid = portfolio_value * taker_fee
                portfolio_value -= fees_paid
                print(f"  -> [TRADE] Initializing portfolio allocation to {target_allocation*100:.0f}%. Paid ${fees_paid:.2f} in fees.")
            elif target_allocation != current_allocation:
                # Rebalancing
                allocation_diff = abs(target_allocation - current_allocation)
                trade_value = portfolio_value * allocation_diff
                fees_paid = trade_value * taker_fee
                portfolio_value -= fees_paid
                current_allocation = target_allocation
                print(f"  -> [TRADE] Rebalancing to {target_allocation*100:.0f}%. Traded ${trade_value:,.2f}, Paid ${fees_paid:.2f} in fees.")
            else:
                print("  -> [HOLD] Target allocation matches current allocation. No fees deducted.")
                
            print(f"  -> Live Portfolio Balance: ${portfolio_value:,.2f}")
            
            # Sleep until the top of the next hour
            now = datetime.now()
            # minutes left in the hour + 1 minute buffer to ensure the candle closed
            sleep_seconds = ((59 - now.minute) * 60) + (60 - now.second) + 60
            print(f"Sleeping for {sleep_seconds} seconds until next hourly candle...")
            time.sleep(sleep_seconds)
            
        except KeyboardInterrupt:
            print("\nLive trader stopped by user.")
            break
        except Exception as e:
            print(f"[!] Critical Error in main loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_live_trader()
