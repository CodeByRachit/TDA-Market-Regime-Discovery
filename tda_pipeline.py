import numpy as np
import pandas as pd
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PairwiseDistance, PersistenceLandscape, Filtering, PersistenceEntropy, PersistenceImage

def compute_rolling_distance_matrices(returns_df, window_size=60):
    """
    Computes rolling correlation matrices and transforms them into distance matrices.
    D_ij = sqrt(2 * (1 - C_ij))
    """
    n_assets = returns_df.shape[1]
    n_days = len(returns_df)
    
    distance_matrices = []
    valid_dates = []
    
    for i in range(window_size, n_days + 1):
        window_returns = returns_df.iloc[i - window_size:i]
        
        # Compute Pearson correlation matrix
        corr_matrix = window_returns.corr().values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Handle potential numerical instability (correlation slightly > 1 or < -1)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        
        # Transform to distance matrix
        dist_matrix = np.sqrt(2 * (1 - corr_matrix))
        
        distance_matrices.append(dist_matrix)
        valid_dates.append(returns_df.index[i - 1])
        
    return np.array(distance_matrices), valid_dates

def extract_topological_features(distance_matrices, homology_dimensions=(1,), epsilon=0.02):
    """
    Extracts persistence diagrams from distance matrices.
    By default, isolates 1-dimensional homology (loops) for turbulence calculations.
    """
    vr = VietorisRipsPersistence(metric="precomputed", homology_dimensions=homology_dimensions)
    diagrams = vr.fit_transform(distance_matrices)
    
    if epsilon > 0:
        # Dynamically filter out topological noise
        filtering = Filtering(epsilon=epsilon, homology_dimensions=homology_dimensions)
        diagrams_filtered = filtering.fit_transform(diagrams)
        return diagrams_filtered
    else:
        return diagrams

def compute_wasserstein_distances(diagrams):
    """
    Computes Wasserstein distance between consecutive persistence diagrams.
    """
    pwd = PairwiseDistance(metric="wasserstein")
    n_diagrams = len(diagrams)
    
    wasserstein_distances = [0.0] # First one has no previous diagram
    
    for i in range(1, n_diagrams):
        # We need to reshape the inputs to (1, n_points, 3) as expected by PairwiseDistance
        diag1 = diagrams[i-1:i]
        diag2 = diagrams[i:i+1]
        
        # Compute pairwise distance between these two
        # The result is a distance matrix of size (1, 1)
        pwd.fit(diag1)
        dist = pwd.transform(diag2)[0][0]
        wasserstein_distances.append(dist)
        
    return np.array(wasserstein_distances)

def compute_persistence_landscapes(diagrams):
    """
    Computes persistence landscapes and their L1, L2 norms (for dimensional 1 features).
    """
    pl = PersistenceLandscape(n_layers=1, n_bins=100)
    landscapes = pl.fit_transform(diagrams)
    
    # landscapes shape: depends on number of homology dimensions
    # We are interested in L1/L2 norms of the landscapes.
    
    axes_to_sum = tuple(range(1, landscapes.ndim))
    l1_norms = np.sum(np.abs(landscapes), axis=axes_to_sum)
    l2_norms = np.sqrt(np.sum(landscapes**2, axis=axes_to_sum))
    
    return l1_norms, l2_norms

def compute_persistence_entropy(diagrams):
    """
    Computes Persistent Entropy (Shannon entropy of topological lifetimes).
    """
    pe = PersistenceEntropy()
    entropies = pe.fit_transform(diagrams)
    # pe returns shape (n_samples, n_homology_dimensions). We isolate the 1D entropy.
    return entropies[:, 0]

def compute_persistence_images(diagrams, n_bins=10, sigma=0.1):
    """
    Computes Persistence Images with modest resolution to avoid overfitting trees.
    """
    pi = PersistenceImage(sigma=sigma, n_bins=n_bins, weight_function=None)
    # explicitly lock bounds so we don't look ahead and scale relative to entire set
    dummy_diagram = np.array([[[0.0, 1.415, 0], [0.0, 1.415, 1]]])
    pi.fit(dummy_diagram)
    images = pi.transform(diagrams)
    # Flatten the image matrix per sample into a 1D feature vector
    n_samples = images.shape[0]
    flattened_images = images.reshape(n_samples, -1)
    return flattened_images

if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    dummy_returns = pd.DataFrame(np.random.randn(100, 10))
    dist_matrices, dates = compute_rolling_distance_matrices(dummy_returns, window_size=60)
    print(f"Generated {len(dist_matrices)} distance matrices of shape {dist_matrices[0].shape}")
    
    diagrams = extract_topological_features(dist_matrices)
    print(f"Extracted diagrams of shape {diagrams.shape}")
    
    w_dists = compute_wasserstein_distances(diagrams)
    print(f"Computed {len(w_dists)} Wasserstein distances")
