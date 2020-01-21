"""Functions for generating the  """
import numpy as np


def generate_pdf(p, q, n_bins=None):
    if n_bins is None:
        n_bins = int(np.sqrt(len(p))) + 1
    
    min_value = np.min(np.hstack([p, q]))
    max_value = np.max(np.hstack([p, q]))
    
    n_bins = int(np.sqrt(len(p))) + 1
    p = histogram(p, n_bins, max_value, min_value)
    q = histogram(q, n_bins, max_value, min_value)
    
    return p, q


def histogram(p, n_bins, max_value, min_value):
    bin_count = np.zeros(n_bins)
    bin_thresholds = np.linspace(min_value, max_value, n_bins)
    for i, bin in enumerate(bin_thresholds[:-1]):
        bin_count[i] = p[(p >= bin) & (p < bin_thresholds[i + 1])].shape[0]
    
    hist = bin_count / p.shape[0]  # Change into a density 
    
    epsilon = 1e-6
    hist += epsilon
        
    return hist

def kl_divergence(p, q): return np.sum(p*np.log(p/q), axis=0)

def bootstrap_kl_divergence(dist_1, dist_2, n_simulations=1_000, n_bins=None):
    """Since both of these random variables are draws from a random distribution, """
    simulation_results = []
    for _ in range(n_simulations):
        q = np.random.choice(dist_1, 1_000)
        p = np.random.choice(dist_2, 1_000)
        p, q = generate_pdf(p, q)
        
        simulation_results.append(kl_divergence(p, q))
        
    return simulation_results
    
    