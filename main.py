import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

from data_loader import fetch_data, compute_log_returns
from tda_pipeline import compute_rolling_distance_matrices, extract_topological_features, compute_persistence_landscapes
from regime_model import generate_allocations
from backtest import run_backtest, calculate_metrics
from cross_sectional_alpha import compute_topological_sensitivities, compute_rolling_betas, generate_cross_sectional_weights, run_cross_sectional_backtest
from tda_pipeline import compute_persistence_entropy, compute_persistence_images
from ml_regime_model import generate_unsupervised_allocations, generate_supervised_allocations
from hft_poc import run_hft_poc

def plot_macro_results(backtest_df, valid_dates, turbulence_index, era_name):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Plot Wealth Index
    axes[0].plot(backtest_df.index, backtest_df['Strategy_Wealth'], label='Strategy (TDA Causal)', color='blue')
    axes[0].plot(backtest_df.index, backtest_df['Benchmark_Wealth'], label='Benchmark (Equal Weight)', color='gray', alpha=0.7)
    axes[0].set_title(f'[{era_name}] Portfolio Wealth Index')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Target Allocation
    axes[1].plot(backtest_df.index, backtest_df['Target_Allocation'], label='Target Equity Allocation', color='green', drawstyle='steps-post')
    axes[1].set_title('Dynamic Asset Allocation')
    axes[1].set_ylabel('Allocation (%)')
    axes[1].grid(True)
    
    # Plot Turbulence Index aligned with dates
    axes[2].plot(valid_dates, turbulence_index, label='Topological Fragmentation (L1 Norm)', color='red')
    axes[2].set_title('Topological Early Warning Signal')
    axes[2].set_ylabel('Index Value')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'macro_results_{era_name}.png')
    print(f"Saved plot to macro_results_{era_name}.png")

def plot_cross_sectional_results(backtest_df, valid_dates, turbulence_index, era_name):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot Wealth Index
    axes[0].plot(backtest_df.index, backtest_df['Strategy_Wealth'], label='Strategy (Topological Alpha)', color='purple')
    axes[0].set_title(f'[{era_name}] Cross-Sectional Dollar & Beta Neutral Wealth Index')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Turbulence Index aligned with dates
    axes[1].plot(valid_dates, turbulence_index, label='Topological Fragmentation (L1 Norm)', color='red')
    axes[1].set_title('Topological Macro Signal')
    axes[1].set_ylabel('Index Value')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'cross_sectional_results_{era_name}.png')
    print(f"Saved plot to cross_sectional_results_{era_name}.png")


def run_pipeline(mode="macro", era="2020"):
    print(f"--- Running TDA Pipeline | Mode: {mode.upper()} | Era: {era} ---")
    
    if mode == "hft":
        run_hft_poc()
        return
    
    # 1. Configuration
    if mode == "ml_macro":
        # Pre-train the ML models on historical data from 1998
        tickers = ["SPY", "DIA", "QQQ", "XLK", "XLF", "XLE", "XLV"]
        start_date = "1998-01-01"
        end_date = "2023-01-01"
    elif era == "2020":
        tickers = ["SPY", "QQQ", "TLT", "GLD", "EFA", "IWM", "EEM", "VNQ", "LQD", "HYG"]
        start_date = "2019-01-01"
        end_date = "2023-01-01"
    else: # 2000 and 2008
        tickers = ["SPY", "DIA", "QQQ", "XLK", "XLF", "XLE", "XLV"]
        start_date = "1998-01-01"
        end_date = "2011-01-01"
        
    window_size = 40
    burn_in_period = 252
    
    # 2. Fetch Data
    prices = fetch_data(tickers, start_date, end_date)
    returns = compute_log_returns(prices)
    
    # 3. Compute Rolling Distance Matrices
    print(f"Computing rolling distance matrices with window = {window_size} days...")
    dist_matrices, valid_dates = compute_rolling_distance_matrices(returns, window_size=window_size)
    
    # 4. Extract Topological Features
    print("Extracting topological features (Vietoris-Rips persistence + Dynamic Noise Filtering)...")
    diagrams = extract_topological_features(dist_matrices)
    
    # 5. Compute Persistence Landscapes (L1 Norms)
    print("Computing Persistence Landscapes (L1 Norms) for H1 features...")
    l1_norms, _ = compute_persistence_landscapes(diagrams)
    
    if mode == "macro":
        # 6a. Regime Classification and Allocation (Strictly Causal)
        print("Generating strictly causal allocations (Z > 3.0 detection)...")
        # Generate allocations using raw L1 norms, eliminating the double-lag moving average
        quintiles, allocations = generate_allocations(l1_norms, burn_in_period=burn_in_period, z_threshold=3.0, run_length=2)
        
        # 7a. Run Backtest
        print("Running macro backtest...")
        backtest_df = run_backtest(returns, allocations, valid_dates)
        metrics = calculate_metrics(backtest_df)
        
        print("\n--- Macro Performance Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        plot_macro_results(backtest_df, valid_dates, l1_norms, era)
        
    elif mode == "cross_sectional":
        # 6b. Cross-Sectional Alpha Generation
        print("Computing ETF topological sensitivities and rolling market betas...")
        sensitivities = compute_topological_sensitivities(returns, l1_norms, valid_dates, window=20)
        betas = compute_rolling_betas(returns, window=60)
        
        print("Generating factor-neutral and dollar-neutral portfolio weights...")
        weights = generate_cross_sectional_weights(sensitivities, betas)
        
        # 7b. Run Backtest
        print("Running cross-sectional backtest...")
        backtest_df = run_cross_sectional_backtest(returns, weights)
        
        # Simple metrics for dollar-neutral
        strat_tot = backtest_df['Strategy_Wealth'].iloc[-1] - 1.0
        n_years = len(backtest_df) / 252
        strat_ann = (backtest_df['Strategy_Wealth'].iloc[-1]) ** (1 / n_years) - 1.0 if n_years > 0 else 0
        strat_peak = backtest_df['Strategy_Wealth'].cummax()
        strat_max_dd = ((backtest_df['Strategy_Wealth'] - strat_peak) / strat_peak).min()
        
        print("\n--- Cross-Sectional Performance Metrics ---")
        print(f"Strategy Total Return: {strat_tot:.4f}")
        print(f"Strategy Annualized Return: {strat_ann:.4f}")
        print(f"Strategy Max Drawdown: {strat_max_dd:.4f}")
        
        plot_cross_sectional_results(backtest_df, valid_dates, l1_norms, era)
        
    elif mode == "ml_macro":
        print("Computing advanced topological vectorizations for ML...")
        entropies = compute_persistence_entropy(diagrams)
        images = compute_persistence_images(diagrams, n_bins=10, sigma=0.1)
        
        # Combine features: L1 norm, Entropy, and flattened images
        features = np.hstack([
            l1_norms.reshape(-1, 1), 
            entropies.reshape(-1, 1), 
            images
        ])
        
        # We also need the benchmark returns aligned with valid_dates
        # returns has an index that starts before valid_dates. 
        # valid_dates is the index for features.
        benchmark_returns_series = returns.mean(axis=1).loc[valid_dates]
        
        print("Generating allocations via Unsupervised K-Means Clustering...")
        unsup_allocations = generate_unsupervised_allocations(features, burn_in_period=burn_in_period, window=500)
        
        print("Generating allocations via Supervised Random Forest (Purged CV)...")
        sup_allocations = generate_supervised_allocations(features, benchmark_returns_series, burn_in_period=burn_in_period, embargo=20)
        
        print("Running backtest for Unsupervised ML model...")
        unsup_backtest = run_backtest(returns, unsup_allocations, valid_dates)
        
        print("Running backtest for Supervised ML model...")
        sup_backtest = run_backtest(returns, sup_allocations, valid_dates)
        
        # Slice backtest for target era
        if era == "2020":
            target_start = "2019-01-01"
            target_end = "2023-01-01"
        else:
            target_start = "1998-01-01"
            target_end = "2011-01-01"
            
        unsup_backtest_sliced = unsup_backtest.loc[target_start:target_end].copy()
        sup_backtest_sliced = sup_backtest.loc[target_start:target_end].copy()
        
        # Re-base Wealth index to 1.0 at start of slice
        unsup_backtest_sliced['Strategy_Wealth'] = unsup_backtest_sliced['Strategy_Wealth'] / unsup_backtest_sliced['Strategy_Wealth'].iloc[0]
        unsup_backtest_sliced['Benchmark_Wealth'] = unsup_backtest_sliced['Benchmark_Wealth'] / unsup_backtest_sliced['Benchmark_Wealth'].iloc[0]
        
        sup_backtest_sliced['Strategy_Wealth'] = sup_backtest_sliced['Strategy_Wealth'] / sup_backtest_sliced['Strategy_Wealth'].iloc[0]
        sup_backtest_sliced['Benchmark_Wealth'] = sup_backtest_sliced['Benchmark_Wealth'] / sup_backtest_sliced['Benchmark_Wealth'].iloc[0]
        
        # Get matching valid_dates and l1_norms for plotting
        valid_dates_series = pd.to_datetime(valid_dates)
        mask = (valid_dates_series >= pd.to_datetime(target_start)) & (valid_dates_series <= pd.to_datetime(target_end))
        valid_dates_sliced = valid_dates_series[mask].tolist()
        l1_norms_sliced = np.array(l1_norms)[mask]
        
        unsup_metrics = calculate_metrics(unsup_backtest_sliced)
        print("\n--- Unsupervised ML Performance ---")
        for k, v in unsup_metrics.items():
            print(f"{k}: {v:.4f}")
            
        sup_metrics = calculate_metrics(sup_backtest_sliced)
        print("\n--- Supervised ML Performance ---")
        for k, v in sup_metrics.items():
            print(f"{k}: {v:.4f}")
            
        # Plot Supervised results
        plot_macro_results(sup_backtest_sliced, valid_dates_sliced, l1_norms_sliced, f"{era}_supervised_ml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TDA Market Regime Discovery')
    parser.add_argument('--mode', type=str, choices=['macro', 'cross_sectional', 'ml_macro', 'hft'], default='macro', help='Strategy mode to run')
    parser.add_argument('--era', type=str, choices=['2020', '2000'], default='2020', help='Era to backtest')
    args = parser.parse_args()
    
    run_pipeline(mode=args.mode, era=args.era)
