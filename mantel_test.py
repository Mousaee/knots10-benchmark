#!/usr/bin/env python3
"""
Mantel Permutation Test for Knots-10
Compare topological distance matrix with confusion rate matrix via permutation test.

Core idea: Do topologically similar knots get confused more by the CNN?
Method: Mantel test with 9999 permutations, two-tailed p-value
Output: JSON results + null distribution histograms (5 subplots)
"""
import os
import json
import numpy as np

# Try to import matplotlib, fallback gracefully
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 0.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not available, skipping visualization", flush=True)

# Try to import scipy, fallback to pure numpy implementation
try:
    from scipy.stats import spearmanr
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not available, using pure numpy implementation", flush=True)

# =============================================
# 1. Configuration
# =============================================

CLASSES = ['ABK', 'BK', 'CH', 'F8K', 'F8L', 'FSK', 'FMB', 'OHK', 'RK', 'SK']
RESULTS_DIR = 'results'
BASELINE_MODELS = ['resnet18', 'resnet50', 'efficientnet_b0', 'vit', 'swin_t']
N_PERMUTATIONS = 9999

os.makedirs(f'{RESULTS_DIR}/figures', exist_ok=True)

# =============================================
# 2. Topological Properties and Distance
# =============================================

KNOT_PROPERTIES = {
    'OHK': {'crossing_num': 3, 'type': 'prime', 'family': 'stopper', 'components': 1},
    'F8K': {'crossing_num': 4, 'type': 'prime', 'family': 'stopper', 'components': 1},
    'BK': {'crossing_num': 4, 'type': 'loop', 'family': 'loop', 'components': 1},
    'RK': {'crossing_num': 6, 'type': 'composite', 'family': 'binding', 'components': 2},
    'FSK': {'crossing_num': 6, 'type': 'composite', 'family': 'bend', 'components': 2},
    'FMB': {'crossing_num': 8, 'type': 'composite', 'family': 'bend', 'components': 2},
    'F8L': {'crossing_num': 4, 'type': 'loop', 'family': 'loop', 'components': 1},
    'CH': {'crossing_num': 2, 'type': 'hitch', 'family': 'hitch', 'components': 1},
    'SK': {'crossing_num': 3, 'type': 'slip', 'family': 'stopper', 'components': 1},
    'ABK': {'crossing_num': 4, 'type': 'loop', 'family': 'loop', 'components': 1},
}

DERIVATION_PAIRS = {
    ('OHK', 'SK'): 0.1,
    ('F8K', 'FMB'): 0.15,
    ('RK', 'FSK'): 0.1,
    ('F8K', 'F8L'): 0.1,
}


def topological_distance(k1, k2):
    """Multi-factor topological distance between two knot classes."""
    p1, p2 = KNOT_PROPERTIES[k1], KNOT_PROPERTIES[k2]

    # Factor 1: Crossing number difference (normalized)
    d_cross = abs(p1['crossing_num'] - p2['crossing_num']) / 8.0

    # Factor 2: Same family? (0 if same, 1 if different)
    d_family = 0.0 if p1['family'] == p2['family'] else 1.0

    # Factor 3: Same type? (prime/composite/loop/hitch)
    d_type = 0.0 if p1['type'] == p2['type'] else 0.5

    # Factor 4: Component count difference
    d_comp = abs(p1['components'] - p2['components'])

    # Factor 5: Structural derivation penalty
    pair = tuple(sorted([k1, k2]))
    d_deriv = DERIVATION_PAIRS.get(pair, 0.5)

    # Weighted combination
    dist = 0.25 * d_cross + 0.25 * d_family + 0.15 * d_type + 0.1 * d_comp + 0.25 * d_deriv
    return dist


def build_distance_matrix():
    """Build topological distance matrix."""
    n = len(CLASSES)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = topological_distance(CLASSES[i], CLASSES[j])
    return D


def confusion_rate_matrix(cm):
    """Normalize confusion matrix to get pairwise confusion rates."""
    cm = np.array(cm, dtype=float)
    n = cm.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        total = cm[i].sum()
        if total > 0:
            for j in range(n):
                if i != j:
                    C[i, j] = cm[i, j] / total
    # Symmetrize
    return (C + C.T) / 2


# =============================================
# 3. Utility Functions for Correlation
# =============================================

def squareform_numpy(D):
    """Convert square distance matrix to upper triangle vector (pure numpy)."""
    n = D.shape[0]
    indices = np.triu_indices(n, k=1)
    return D[indices]


def spearman_correlation(x, y):
    """Compute Spearman correlation coefficient (pure numpy)."""
    # Convert to ranks
    rank_x = np.argsort(np.argsort(x)) + 1
    rank_y = np.argsort(np.argsort(y)) + 1

    # Compute Pearson correlation on ranks
    mean_x = rank_x.mean()
    mean_y = rank_y.mean()

    cov = np.mean((rank_x - mean_x) * (rank_y - mean_y))
    std_x = rank_x.std(ddof=1)
    std_y = rank_y.std(ddof=1)

    if std_x > 0 and std_y > 0:
        rho = cov / (std_x * std_y)
    else:
        rho = 0.0

    return rho


# =============================================
# 4. Mantel Test Implementation
# =============================================

def mantel_test_permutation(D_topo, C_confusion, n_perms=9999, seed=None):
    """
    Mantel permutation test: test correlation between topological distance
    and confusion rate matrices.

    Args:
        D_topo: topological distance matrix (n x n)
        C_confusion: confusion rate matrix (n x n)
        n_perms: number of permutations (default 9999)
        seed: random seed for reproducibility

    Returns:
        r_obs: observed Spearman correlation
        p_value: two-tailed p-value
        null_dist: array of null distribution (permutation correlations)
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert to vectors (upper triangles)
    if HAS_SCIPY:
        v_topo = squareform(D_topo, checks=False)
        v_conf = squareform(C_confusion, checks=False)
    else:
        v_topo = squareform_numpy(D_topo)
        v_conf = squareform_numpy(C_confusion)

    # Observed correlation
    if HAS_SCIPY:
        r_obs, _ = spearmanr(v_topo, v_conf)
    else:
        r_obs = spearman_correlation(v_topo, v_conf)

    # Permutation test
    null_dist = []
    n = D_topo.shape[0]

    for perm_idx in range(n_perms):
        # Random permutation of rows and columns
        perm = np.random.permutation(n)
        C_perm = C_confusion[np.ix_(perm, perm)]

        # Correlation with permuted confusion matrix
        if HAS_SCIPY:
            v_perm = squareform(C_perm, checks=False)
            r_perm, _ = spearmanr(v_topo, v_perm)
        else:
            v_perm = squareform_numpy(C_perm)
            r_perm = spearman_correlation(v_topo, v_perm)
        null_dist.append(r_perm)

    null_dist = np.array(null_dist)

    # Two-tailed p-value: count how many |r_perm| >= |r_obs|
    count_extreme = np.sum(np.abs(null_dist) >= np.abs(r_obs))
    p_value = (count_extreme + 1) / (n_perms + 1)

    return r_obs, p_value, null_dist


def run_mantel_test(model_name):
    """Run Mantel test for a single model."""
    print(f"\n[{model_name}] Running Mantel test...", flush=True)

    # Load confusion matrix
    fp = f'{RESULTS_DIR}/{model_name}_results.json'
    if not os.path.exists(fp):
        print(f"  [SKIP] {fp} not found", flush=True)
        return None

    with open(fp) as f:
        results = json.load(f)

    cm = np.array(results['confusion_matrix'])
    C = confusion_rate_matrix(cm)

    # Build topological distance matrix
    D = build_distance_matrix()

    # Run Mantel test
    r_obs, p_value, null_dist = mantel_test_permutation(D, C, n_perms=N_PERMUTATIONS, seed=42)

    # Also compute Spearman correlation directly (for reference)
    if HAS_SCIPY:
        v_topo = squareform(D, checks=False)
        v_conf = squareform(C, checks=False)
        rho, p_spear = spearmanr(v_topo, v_conf)
    else:
        v_topo = squareform_numpy(D)
        v_conf = squareform_numpy(C)
        rho = spearman_correlation(v_topo, v_conf)
        p_spear = np.nan  # Cannot compute p-value without scipy

    # Statistics on null distribution
    null_mean = null_dist.mean()
    null_std = null_dist.std()

    result = {
        'model': model_name,
        'r_obs': float(r_obs),
        'p_value_mantel': float(p_value),
        'spearman_rho': float(rho),
        'spearman_p': float(p_spear),
        'null_mean': float(null_mean),
        'null_std': float(null_std),
        'n_permutations': N_PERMUTATIONS,
    }

    print(f"  r_obs = {r_obs:.4f}", flush=True)
    print(f"  p_value (two-tailed) = {p_value:.4e}", flush=True)
    print(f"  Spearman rho = {rho:.4f} (p = {p_spear:.4e})", flush=True)
    print(f"  Null mean = {null_mean:.4f}, std = {null_std:.4f}", flush=True)

    return result, null_dist


# =============================================
# 4. Visualization
# =============================================

def plot_null_distribution(all_results, all_null_dists):
    """
    Plot null distribution histograms for all models (5 subplots).
    Mark the observed correlation on each histogram.
    Falls back gracefully if matplotlib is not available.
    """
    if not HAS_MATPLOTLIB:
        print("[INFO] Skipping visualization (matplotlib not available)", flush=True)
        return

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5))  # 190mm double-column
    axes = axes.flatten()

    for idx, model_name in enumerate(BASELINE_MODELS):
        if model_name not in all_results:
            axes[idx].text(0.5, 0.5, f'{model_name}\n(not found)',
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(model_name, fontsize=12)
            continue

        result = all_results[model_name]
        null_dist = all_null_dists[model_name]
        r_obs = result['r_obs']
        p_value = result['p_value_mantel']

        # Plot histogram
        ax = axes[idx]
        ax.hist(null_dist, bins=50, alpha=0.7, color='skyblue', edgecolor='black')

        # Mark observed value
        ax.axvline(r_obs, color='red', linestyle='--', linewidth=2.5, label=f'r_obs = {r_obs:.4f}')

        # Mark critical values for two-tailed test
        alpha = 0.05
        crit_val = np.percentile(np.abs(null_dist), (1 - alpha) * 100)
        ax.axvline(crit_val, color='orange', linestyle=':', linewidth=2, alpha=0.7, label=f'Critical (α={alpha})')
        ax.axvline(-crit_val, color='orange', linestyle=':', linewidth=2, alpha=0.7)

        ax.set_title(f'{model_name}\np-value = {p_value:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Spearman correlation', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

    # Hide the last (6th) subplot
    axes[5].axis('off')

    plt.suptitle('Mantel Test: Null Distribution of Correlations\n(9999 permutations)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/figures/mantel_null_distribution.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{RESULTS_DIR}/figures/mantel_null_distribution.png', dpi=150, bbox_inches='tight')
    paper_fig = 'paper/figures/mantel_null_distribution.pdf'
    os.makedirs(os.path.dirname(paper_fig), exist_ok=True)
    fig.savefig(paper_fig, dpi=300, bbox_inches='tight')
    print(f'[Saved] {paper_fig}', flush=True)
    plt.close()

    print('[Saved] mantel_null_distribution.pdf/png', flush=True)


# =============================================
# 5. Main
# =============================================

if __name__ == '__main__':
    print('='*70)
    print('MANTEL PERMUTATION TEST FOR KNOTS-10')
    print('='*70)
    print(f'N_PERMUTATIONS = {N_PERMUTATIONS}')
    print(f'BASELINE_MODELS = {BASELINE_MODELS}')

    all_results = {}
    all_null_dists = {}

    # Run test for each model
    for model_name in BASELINE_MODELS:
        result = run_mantel_test(model_name)
        if result is not None:
            result_data, null_dist = result
            all_results[model_name] = result_data
            all_null_dists[model_name] = null_dist

    # Save results to JSON
    output_file = f'{RESULTS_DIR}/mantel_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n[Saved] {output_file}', flush=True)

    # Plot null distributions
    if all_results:
        plot_null_distribution(all_results, all_null_dists)

    # Print summary table
    print('\n' + '='*70)
    print('MANTEL TEST SUMMARY')
    print('='*70)
    print(f"{'Model':<20} | {'r_obs':>8} | {'p-value':>10} | {'Spearman':>10}")
    print('-'*70)
    for model_name in BASELINE_MODELS:
        if model_name in all_results:
            r = all_results[model_name]
            print(f"{model_name:<20} | {r['r_obs']:>8.4f} | {r['p_value_mantel']:>10.4e} | {r['spearman_rho']:>10.4f}")
    print('='*70)
    print('[DONE]', flush=True)
