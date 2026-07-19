import os
import time
import datetime
import pandas as pd
import numpy as np
import ccxt
import joblib
from sklearn.ensemble import RandomForestClassifier
from tda_pipeline import compute_rolling_distance_matrices, extract_topological_features, compute_persistence_images

def fetch_binance_ohlcv(symbol, timeframe, since, until, max_retries=5):
    """Fetch historical OHLCV data from Binance using ccxt with pagination and error handling."""
    exchange = ccxt.binance({'enableRateLimit': True})
    all_ohlcv = []
    current_since = since
    
    while current_since < until:
        retries = 0
        while retries < max_retries:
            try:
                # Binance fetch_ohlcv returns 1000 max
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                # update current_since to the last timestamp + 1 ms
                last_ts = ohlcv[-1][0]
                current_since = last_ts + 1
                
                if last_ts >= until:
                    break
                
                break # Success, break out of retry loop
            except ccxt.NetworkError as e:
                print(f"Network error fetching {symbol}: {e}. Retrying in 10s...")
                time.sleep(10)
                retries += 1
            except ccxt.RateLimitExceeded as e:
                print(f"Rate limit exceeded for {symbol}: {e}. Sleeping for 60s...")
                time.sleep(60)
                retries += 1
            except Exception as e:
                print(f"Unexpected error for {symbol}: {e}. Retrying in 10s...")
                time.sleep(10)
                retries += 1
                
        if not ohlcv or current_since >= until:
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df['close']

def get_crypto_data():
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
    timeframe = '1h'
    
    # 2020-01-01 to 2023-12-31
    since = int(datetime.datetime(2020, 1, 1).timestamp() * 1000)
    until = int(datetime.datetime(2024, 1, 1).timestamp() * 1000)
    
    closes = {}
    for sym in symbols:
        print(f"Fetching {sym} 1h data from {datetime.datetime.fromtimestamp(since/1000)} to {datetime.datetime.fromtimestamp(until/1000)}...")
        closes[sym] = fetch_binance_ohlcv(sym, timeframe, since, until)
    
    df = pd.DataFrame(closes)
    # Forward fill then dropna to align timestamps
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

def train_and_persist():
    print("--- 1. Fetching Historical 1H Crypto Data ---")
    data_file = 'crypto_1h_2020_2023.csv'
    if os.path.exists(data_file):
        print(f"Loading cached data from {data_file}")
        df = pd.read_csv(data_file, index_col='timestamp', parse_dates=True)
    else:
        df = get_crypto_data()
        df.to_csv(data_file)
    
    print(f"Data shape: {df.shape}")
    
    print("--- 2. Computing Log Returns & Topological Distance Matrices ---")
    returns = np.log(df / df.shift(1)).dropna()
    
    # Window size 60 periods (hours) -> 2.5 days
    window_size = 60
    dist_matrices, valid_dates = compute_rolling_distance_matrices(returns, window_size=window_size)
    
    # Because there are ~35,000 hours in 4 years, running TDA on every single hour takes a very long time.
    # To expedite training, we will sample the dataset (e.g. 1 point every 4 hours).
    # Since topological regimes shift over days, 4-hour sampling is sufficient for training the model.
    sample_rate = 4
    dist_matrices_sampled = dist_matrices[::sample_rate]
    valid_dates_sampled = valid_dates[::sample_rate]
    
    print(f"Sampled {len(dist_matrices_sampled)} points for TDA out of {len(dist_matrices)}.")
    
    print("--- 3. Extracting Topological Features ---")
    diagrams = extract_topological_features(dist_matrices_sampled, epsilon=0.0) # HFT/Crypto mode uses 0.0 or adaptive
    
    print("--- 4. Computing Persistence Images ---")
    from gtda.diagrams import PersistenceImage
    # Fit the globally bounded PersistenceImage
    pi = PersistenceImage(sigma=0.1, n_bins=10, weight_function=None)
    dummy_diagram = np.array([[[0.0, 1.415, 0], [0.0, 1.415, 1]]])
    pi.fit(dummy_diagram)
    
    images = pi.transform(diagrams)
    X = images.reshape(images.shape[0], -1)
    
    print("--- 5. Defining Crypto Crash Labels ---")
    # We define a crash as forward 24-hour return of the equal-weighted portfolio being < -5%
    # But since we sampled valid_dates_sampled, we must find the forward 24-hour return from the original returns dataframe.
    
    eq_port_returns = returns.mean(axis=1)
    # Forward 24h return = sum of next 24 hourly log returns
    forward_24h_returns = eq_port_returns.rolling(window=24).sum().shift(-24)
    
    y = []
    for date in valid_dates_sampled:
        if date in forward_24h_returns.index:
            ret = forward_24h_returns.loc[date]
            # 1 for Crash, 0 for Normal
            y.append(1 if ret < -0.05 else 0)
        else:
            y.append(0)
            
    y = np.array(y)
    print(f"Class distribution: {np.bincount(y)} (Crash instances: {np.sum(y)})")
    
    print("--- 6. Training the Random Forest Classifier ---")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', n_jobs=-1)
    clf.fit(X, y)
    
    print("--- 7. Persisting Models to Disk ---")
    joblib.dump(clf, 'rf_model.joblib')
    joblib.dump(pi, 'pi_transformer.joblib')
    print("Successfully saved 'rf_model.joblib' and 'pi_transformer.joblib'.")

if __name__ == "__main__":
    train_and_persist()
