import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

def generate_unsupervised_allocations(features, burn_in_period=252, window=500):
    """
    Uses rolling K-Means clustering on topological features to dynamically classify market regimes.
    Identifies 2 clusters (Normal vs. Crisis).
    features: A numpy array or DataFrame of shape (n_samples, n_features).
    """
    allocations = []
    
    # We will track rolling clusters
    for i in range(len(features)):
        if i < burn_in_period:
            allocations.append(1.10)
            continue
            
        start_idx = max(0, i - window)
        # historical window
        X_history = features[start_idx:i]
        x_current = features[i].reshape(1, -1)
        
        # Fit K-Means
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        kmeans.fit(X_history)
        
        # Identify the "Crisis" cluster. We assume Crisis regimes have higher topological complexity.
        # We can use the mean of the first feature (e.g., L1 norm) to identify the cluster.
        cluster_means = kmeans.cluster_centers_[:, 0]
        crisis_cluster = np.argmax(cluster_means)
        
        # Predict current regime
        current_cluster = kmeans.predict(x_current)[0]
        
        if current_cluster == crisis_cluster:
            allocations.append(0.20) # De-risk
        else:
            allocations.append(1.10) # Normal leverage
            
    return np.array(allocations)

def generate_supervised_allocations(features, benchmark_returns, burn_in_period=252, embargo=20):
    """
    Trains a RandomForestClassifier using purged cross-validation logic (embargo).
    Label: 1 if the forward 20-day benchmark return is strongly negative (Crash), 0 otherwise.
    """
    # Calculate forward 20-day return
    fwd_returns = benchmark_returns.rolling(window=20).sum().shift(-20).fillna(0.0)
    
    # Define "Crash" as a forward return worse than -5%
    labels = (fwd_returns < -0.05).astype(int).values
    
    allocations = []
    
    # Rolling Random Forest
    for i in range(len(features)):
        if i < burn_in_period:
            allocations.append(1.10)
            continue
            
        # Embargo: we cannot use data where the forward return overlaps with our current prediction day.
        # If we are at day i, the forward return for day i-k uses data up to i-k+20.
        # We must ensure i-k+20 < i -> k > 20.
        # So we can only train on data up to i - 20 (the embargo period).
        train_end = i - embargo
        
        if train_end < 50:
            allocations.append(1.10)
            continue
            
        X_train = features[:train_end]
        y_train = labels[:train_end]
        
        # If we have no crisis examples yet in training, just stay long
        if np.sum(y_train) == 0:
            allocations.append(1.10)
            continue
            
        clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1, class_weight="balanced")
        clf.fit(X_train, y_train)
        
        x_current = features[i].reshape(1, -1)
        pred = clf.predict(x_current)[0]
        
        if pred == 1:
            allocations.append(0.20) # Predicted Crash
        else:
            allocations.append(1.10)
            
    return np.array(allocations)
