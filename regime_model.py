import numpy as np
import pandas as pd

def classify_by_percentile(signal_value, historical_signals):
    """
    Classifies the current signal into a quintile (1-4) based on historical percentiles.
    Quintile 5 is reserved for causal structural break detection.
    """
    q20 = np.percentile(historical_signals, 20)
    q40 = np.percentile(historical_signals, 40)
    q60 = np.percentile(historical_signals, 60)
    q80 = np.percentile(historical_signals, 80)
    
    if signal_value <= q20:
        return 1, 1.20 # 120% Leveraged Long
    elif signal_value <= q40:
        return 2, 1.10 # 110% Slight Leverage
    elif signal_value <= q60:
        return 3, 0.90 # 90% Underweight Equity
    else:
        return 4, 0.60 # 60% Heavily Hedged

def generate_allocations(topological_signals, burn_in_period=252, z_threshold=3.0, run_length=2):
    """
    Generates target allocations for a time series of raw topological signals.
    Uses a strictly causal decision rule with a run-length parameter for detecting crashes.
    """
    allocations = []
    quintiles = []
    consecutive_high_z = 0
    
    for i in range(len(topological_signals)):
        if i < burn_in_period:
            # Burn-in phase: passive observation
            quintiles.append(2)
            allocations.append(1.10)
            continue
            
        historical_signals = topological_signals[:i]
        current_signal = topological_signals[i]
        
        # Compute historical mean and std for Z-score
        mean = np.mean(historical_signals)
        std = np.std(historical_signals)
        z_score = (current_signal - mean) / std if std > 0 else 0.0
        
        # Causal detection rule
        if z_score > z_threshold:
            consecutive_high_z += 1
        else:
            consecutive_high_z = 0
            
        # Determine regime
        if consecutive_high_z >= run_length:
            # Structural collapse detected! Immediate de-risking
            q, alloc = 5, 0.20
        else:
            # Normal market conditions -> use standard topological percentiles
            q, alloc = classify_by_percentile(current_signal, historical_signals)
            
        quintiles.append(q)
        allocations.append(alloc)
        
    return np.array(quintiles), np.array(allocations)
