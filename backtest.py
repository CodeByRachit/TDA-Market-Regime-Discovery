import pandas as pd
import numpy as np

def run_backtest(returns_df, allocations, valid_dates):
    """
    Simulates the strategy portfolio.
    returns_df: original daily log returns of the asset universe
    allocations: the target equity allocations from the regime model
    valid_dates: the dates corresponding to the allocations
    """
    # Create a benchmark: equal weighted portfolio of all assets
    n_assets = returns_df.shape[1]
    # Simple average of log returns gives the log return of the equal weight portfolio (approximation)
    benchmark_log_returns = returns_df.mean(axis=1)
    
    # Strategy returns: benchmark return * allocation
    # The allocations correspond to the `valid_dates`.
    # Let's align them.
    # We shift allocations by 1 day because the allocation computed on day T 
    # based on close prices can only be applied to the return on day T+1.
    
    alloc_df = pd.Series(allocations, index=valid_dates)
    
    # We create a dataframe to hold aligned data
    backtest_df = pd.DataFrame(index=returns_df.index)
    backtest_df['Benchmark_Log_Return'] = benchmark_log_returns
    
    # Forward fill the allocations in case of missing days, then shift by 1 to prevent lookahead bias
    backtest_df['Target_Allocation'] = alloc_df
    backtest_df['Target_Allocation'] = backtest_df['Target_Allocation'].shift(1).ffill()
    
    # Fill any NaNs at the beginning with 1.0 (100% allocation)
    backtest_df['Target_Allocation'] = backtest_df['Target_Allocation'].fillna(1.0)
    
    # Calculate daily turnover (absolute change in allocation)
    turnover = backtest_df['Target_Allocation'].diff().abs().fillna(0.0)
    
    # Slippage assumption: 10 basis points per 100% turnover
    slippage_cost = turnover * 0.0010
    
    # Compute strategy returns
    backtest_df['Strategy_Log_Return'] = (backtest_df['Benchmark_Log_Return'] * backtest_df['Target_Allocation']) - slippage_cost
    
    # Convert log returns back to cumulative arithmetic returns (wealth index)
    backtest_df['Benchmark_Wealth'] = np.exp(backtest_df['Benchmark_Log_Return'].cumsum())
    backtest_df['Strategy_Wealth'] = np.exp(backtest_df['Strategy_Log_Return'].cumsum())
    
    return backtest_df

def calculate_metrics(backtest_df):
    """
    Calculates basic performance metrics.
    """
    strategy_wealth = backtest_df['Strategy_Wealth']
    benchmark_wealth = backtest_df['Benchmark_Wealth']
    
    # Total Return
    strat_tot = strategy_wealth.iloc[-1] - 1.0
    bench_tot = benchmark_wealth.iloc[-1] - 1.0
    
    # Annualized Return (assuming 252 trading days)
    n_years = len(backtest_df) / 252
    strat_ann = (strategy_wealth.iloc[-1]) ** (1 / n_years) - 1.0 if n_years > 0 else 0
    bench_ann = (benchmark_wealth.iloc[-1]) ** (1 / n_years) - 1.0 if n_years > 0 else 0
    
    # Max Drawdown
    strat_peak = strategy_wealth.cummax()
    strat_dd = (strategy_wealth - strat_peak) / strat_peak
    strat_max_dd = strat_dd.min()
    
    bench_peak = benchmark_wealth.cummax()
    bench_dd = (benchmark_wealth - bench_peak) / bench_peak
    bench_max_dd = bench_dd.min()
    
    return {
        "Strategy Total Return": strat_tot,
        "Strategy Annualized Return": strat_ann,
        "Strategy Max Drawdown": strat_max_dd,
        "Benchmark Total Return": bench_tot,
        "Benchmark Annualized Return": bench_ann,
        "Benchmark Max Drawdown": bench_max_dd
    }
