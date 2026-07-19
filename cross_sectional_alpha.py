import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def compute_topological_sensitivities(returns_df, l1_norms, valid_dates, window=20):
    """
    Computes the rolling correlation between each asset's returns and 
    the daily change in the L1 norm (Topological Fragmentation Index).
    """
    # L1 norms correspond to valid_dates. We need daily changes.
    l1_series = pd.Series(l1_norms, index=valid_dates)
    l1_changes = l1_series.diff().fillna(0.0)
    
    sensitivities = pd.DataFrame(index=valid_dates, columns=returns_df.columns)
    
    # We only have L1 changes starting from valid_dates[0], which is window_size days into returns_df
    for i in range(window, len(valid_dates)):
        current_date = valid_dates[i]
        start_date = valid_dates[i - window]
        
        # Slicing the data
        y = l1_changes.loc[start_date:current_date]
        X = returns_df.loc[start_date:current_date]
        
        # Calculate correlation for each asset
        corrs = X.corrwith(y)
        sensitivities.loc[current_date] = corrs
        
    return sensitivities.astype(float)

def compute_rolling_betas(returns_df, window=60):
    """
    Computes rolling market beta for each asset against an equal-weight benchmark.
    """
    market_returns = returns_df.mean(axis=1)
    betas = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
    
    for i in range(window, len(returns_df)):
        current_date = returns_df.index[i]
        start_date = returns_df.index[i - window]
        
        X = market_returns.loc[start_date:current_date].values.reshape(-1, 1)
        
        for col in returns_df.columns:
            y = returns_df[col].loc[start_date:current_date].values
            if len(X) > 2:
                model = LinearRegression().fit(X, y)
                betas.loc[current_date, col] = model.coef_[0]
                
    return betas.astype(float)

def generate_cross_sectional_weights(sensitivities, betas):
    """
    Generates dollar-neutral and beta-neutral portfolio weights.
    We want to go long insulated assets (low/negative sensitivity) and 
    short sensitive assets (high positive sensitivity).
    So our target signal S is -1 * sensitivity.
    We project S onto the null space of [1, Beta] to ensure neutrality.
    """
    weights = pd.DataFrame(index=sensitivities.index, columns=sensitivities.columns)
    
    for date in sensitivities.index:
        S = -sensitivities.loc[date].values
        B = betas.loc[date].values
        
        if np.isnan(S).any() or np.isnan(B).any():
            weights.loc[date] = np.nan
            continue
            
        # Form constraint matrix C: shape (2, N)
        # Row 1: Dollar neutrality (all 1s)
        # Row 2: Beta neutrality (betas)
        N = len(S)
        C = np.vstack([np.ones(N), B])
        
        # Project S onto the null space of C
        # W = S - C.T @ inv(C @ C.T) @ C @ S
        try:
            CCT_inv = np.linalg.inv(C @ C.T)
            projection = C.T @ CCT_inv @ C @ S
            W = S - projection
            
            # Scale weights so the gross exposure is 1.0 (sum of absolute weights = 1.0)
            sum_abs = np.sum(np.abs(W))
            if sum_abs > 0:
                W = W / sum_abs
                
            # --- EXECUTION REALITIES: Short Constraints & Borrow Fees ---
            # 1. Hard constraint: limit maximum short position on any single asset to -0.15 (-15%)
            # This simulates avoiding massive concentration in hard-to-borrow names.
            max_short = -0.15
            W_constrained = np.clip(W, max_short, None)
            
            # Re-normalize to dollar neutrality simply by shifting the long side 
            # if we clipped any short positions. (Simplistic approach for constraint satisfaction)
            diff = np.sum(W_constrained)
            if diff != 0:
                # If diff > 0, we clipped shorts. We must reduce longs to compensate.
                long_mask = W_constrained > 0
                if np.sum(long_mask) > 0:
                    W_constrained[long_mask] -= diff * (W_constrained[long_mask] / np.sum(W_constrained[long_mask]))
            
            weights.loc[date] = W_constrained
        except np.linalg.LinAlgError:
            weights.loc[date] = np.nan
            
    # Forward fill the weights to stay invested during flat signal periods
    weights = weights.ffill().fillna(0.0)
    return weights.astype(float)

def run_cross_sectional_backtest(returns_df, weights):
    """
    Simulates the cross-sectional strategy.
    weights are computed on day T, applied to returns on day T+1.
    """
    # Shift weights by 1 day
    target_weights = weights.shift(1).fillna(0.0)
    
    # Portfolio return is the dot product of weights and asset returns
    portfolio_returns = (target_weights * returns_df).sum(axis=1)
    
    backtest_df = pd.DataFrame(index=returns_df.index)
    backtest_df['Strategy_Log_Return'] = portfolio_returns
    
    # For a dollar neutral strategy, returns are arithmetic additive to the base capital
    # Wealth = 1.0 + cumsum(returns)
    # We will use standard cumulative return for simplicity
    backtest_df['Strategy_Wealth'] = np.exp(portfolio_returns.cumsum())
    
    return backtest_df
