#!/usr/bin/env python3
"""
Weight Sensitivity Analysis for Topological Distance Metric.
Tests robustness of correlation results across different weight configurations.
Output: weight_sensitivity.json + weight_sensitivity.pdf
"""
import json, os, itertools
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
RESULTS_DIR = 'results'
FIG_DIR = f'{RESULTS_DIR}/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Knot properties
KP = {
    'OHK':{'cn':3,'type':'prime','family':'stopper','comp':1},
    'SK': {'cn':3,'type':'slip','family':'stopper','comp':1},
    'F8K':{'cn':4,'type':'prime','family':'stopper','comp':1},
    'BK': {'cn':4,'type':'loop','family':'loop','comp':1},
    'F8L':{'cn':4,'type':'loop','family':'loop','comp':1},
    'ABK':{'cn':4,'type':'loop','family':'loop','comp':1},
    'CH': {'cn':2,'type':'hitch','family':'hitch','comp':1},
    'RK': {'cn':6,'type':'composite','family':'binding','comp':2},
    'FSK':{'cn':6,'type':'composite','family':'bend','comp':2},
    'FMB':{'cn':8,'type':'composite','family':'bend','comp':2},
}
DR = {('F8K','FMB'):0.15,('F8K','F8L'):0.1,('OHK','SK'):0.1,('RK','FSK'):0.1}

def build_distance_matrix(w1, w2, w3, w4, w5):
    """Build 10x10 topological distance matrix with given weights."""
    D = np.zeros((10, 10))
    for i, ci in enumerate(CLASSES):
        for j, cj in enumerate(CLASSES):
            if i == j: continue
            pi, pj = KP[ci], KP[cj]
            d1 = abs(pi['cn'] - pj['cn']) / 8.0
            d2 = 0 if pi['family'] == pj['family'] else 1
            d3 = 0 if pi['type'] == pj['type'] else 0.5
            d4 = abs(pi['comp'] - pj['comp'])
            d5 = DR.get(tuple(sorted([ci, cj])), 0.5)
            D[i, j] = w1*d1 + w2*d2 + w3*d3 + w4*d4 + w5*d5
    return D

def confusion_rate_matrix(cm):
    """Normalize confusion matrix to get pairwise confusion rates."""
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums
    np.fill_diagonal(cm_norm, 0)
    C = (cm_norm + cm_norm.T) / 2
    return C

def get_correlation(D, C):
    """Compute Spearman correlation between distance and confusion."""
    idx = np.triu_indices(10, k=1)
    dists = D[idx]
    confs = C[idx]
    mask = confs > 0
    if mask.sum() < 3:
        return np.nan, 1.0
    rho, p = spearmanr(dists, confs)
    return rho, p

# Load model results
model_results = {}
for mname in ['resnet18','resnet50','efficientnet_b0','vit','swin_t']:
    fp = f'{RESULTS_DIR}/{mname}_results.json'
    if os.path.exists(fp):
        with open(fp) as f:
            model_results[mname] = json.load(f)
print(f'Loaded: {list(model_results.keys())}')

# ─── Strategy 1: Systematic grid search ───
# Test 5 weight configurations including uniform and extremes
weight_configs = {
    'Default (0.25,0.25,0.15,0.10,0.25)': (0.25, 0.25, 0.15, 0.10, 0.25),
    'Uniform (0.20 each)':                  (0.20, 0.20, 0.20, 0.20, 0.20),
    'Crossing-heavy (0.40,0.15,0.15,0.15,0.15)': (0.40, 0.15, 0.15, 0.15, 0.15),
    'Family-heavy (0.15,0.40,0.15,0.15,0.15)':   (0.15, 0.40, 0.15, 0.15, 0.15),
    'Structure-heavy (0.15,0.15,0.15,0.15,0.40)': (0.15, 0.15, 0.15, 0.15, 0.40),
    'No crossing (0,0.30,0.20,0.20,0.30)':        (0.00, 0.30, 0.20, 0.20, 0.30),
    'No family (0.30,0,0.20,0.20,0.30)':          (0.30, 0.00, 0.20, 0.20, 0.30),
    'Binary family only (0,1,0,0,0)':              (0.00, 1.00, 0.00, 0.00, 0.00),
    'Crossing only (1,0,0,0,0)':                   (1.00, 0.00, 0.00, 0.00, 0.00),
}

# ─── Strategy 2: Random perturbation (Monte Carlo) ───
np.random.seed(42)
n_random = 200
default_w = np.array([0.25, 0.25, 0.15, 0.10, 0.25])

# Target model for sensitivity: use the one with strongest signal
target_model = 'efficientnet_b0'  # strongest correlation
cm = np.array(model_results[target_model]['confusion_matrix'])
C = confusion_rate_matrix(cm)

# Grid search results
print('\n=== Systematic Weight Configurations ===')
print(f'{"Config":<50} | {"Spearman ρ":>12} | {"p-value":>10}')
print('-' * 80)
grid_results = {}
for name, (w1, w2, w3, w4, w5) in weight_configs.items():
    D = build_distance_matrix(w1, w2, w3, w4, w5)
    rho, p = get_correlation(D, C)
    grid_results[name] = {'weights': [w1,w2,w3,w4,w5], 'rho': rho, 'p': p}
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    print(f'{name:<50} | {rho:>10.3f}   | {p:>10.4f} {sig}')

# Monte Carlo perturbation
print(f'\n=== Monte Carlo Perturbation (n={n_random}) ===')
mc_rhos = []
for _ in range(n_random):
    # Perturb default weights by ±50%
    noise = np.random.uniform(0.5, 1.5, 5)
    w_perturbed = default_w * noise
    w_perturbed = w_perturbed / w_perturbed.sum()  # normalize to 1
    D = build_distance_matrix(*w_perturbed)
    rho, p = get_correlation(D, C)
    mc_rhos.append(rho)

mc_rhos = np.array(mc_rhos)
mc_rhos = mc_rhos[~np.isnan(mc_rhos)]
print(f'  Mean ρ: {mc_rhos.mean():.3f} ± {mc_rhos.std():.3f}')
print(f'  Range: [{mc_rhos.min():.3f}, {mc_rhos.max():.3f}]')
print(f'  % significant (ρ < -0.25): {(mc_rhos < -0.25).mean()*100:.0f}%')

# ─── Multi-model sensitivity ───
print('\n=== Multi-Model Sensitivity (Default vs Uniform) ===')
model_display = {'resnet18':'ResNet-18','resnet50':'ResNet-50',
                 'efficientnet_b0':'EfficientNet-B0','vit':'ViT-B/16','swin_t':'Swin-T'}
multi_results = {}
for mname, res in model_results.items():
    cm_m = np.array(res['confusion_matrix'])
    C_m = confusion_rate_matrix(cm_m)
    D_default = build_distance_matrix(0.25, 0.25, 0.15, 0.10, 0.25)
    D_uniform = build_distance_matrix(0.20, 0.20, 0.20, 0.20, 0.20)
    rho_d, p_d = get_correlation(D_default, C_m)
    rho_u, p_u = get_correlation(D_uniform, C_m)
    multi_results[mname] = {
        'default': {'rho': rho_d, 'p': p_d},
        'uniform': {'rho': rho_u, 'p': p_u}
    }
    print(f'  {model_display[mname]:<18} Default ρ={rho_d:.3f} (p={p_d:.3f}) | Uniform ρ={rho_u:.3f} (p={p_u:.3f})')

# ─── Visualization ───
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Monte Carlo histogram
ax = axes[0]
ax.hist(mc_rhos, bins=25, color='#0072B2', alpha=0.7, edgecolor='white')
ax.axvline(mc_rhos.mean(), color='#D55E00', linestyle='--', linewidth=2, label=f'Mean = {mc_rhos.mean():.3f}')
ax.axvline(-0.491, color='#009E73', linestyle='-', linewidth=2, label='Default weights (ρ = −0.491)')
ax.set_xlabel('Spearman ρ (EfficientNet-B0)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Weight Sensitivity: Monte Carlo Perturbation\n(200 random weight vectors, ±50% noise)', fontsize=11)
ax.legend(fontsize=9)

# Right: Multi-model comparison
ax2 = axes[1]
models_plot = [m for m in ['resnet18','resnet50','efficientnet_b0','vit'] if m in multi_results]
x = np.arange(len(models_plot))
w = 0.35
rhos_d = [multi_results[m]['default']['rho'] for m in models_plot]
rhos_u = [multi_results[m]['uniform']['rho'] for m in models_plot]
ax2.bar(x - w/2, rhos_d, w, label='Default weights', color='#0072B2', alpha=0.8)
ax2.bar(x + w/2, rhos_u, w, label='Uniform weights', color='#E69F00', alpha=0.8)
ax2.set_ylabel('Spearman ρ', fontsize=12)
ax2.set_title('Weight Configuration Robustness\nAcross Models', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels([model_display[m] for m in models_plot], fontsize=10)
ax2.legend(fontsize=9)
ax2.axhline(0, color='gray', linewidth=0.5)

plt.tight_layout()
for d in [FIG_DIR, 'paper/figures']:
    os.makedirs(d, exist_ok=True)
    fig.savefig(f'{d}/weight_sensitivity.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{d}/weight_sensitivity.png', dpi=200, bbox_inches='tight')
plt.close()
print(f'\n[Saved] weight_sensitivity.pdf/png')

# Save JSON
output = {
    'grid_search': {k: {'weights': v['weights'], 'rho': float(v['rho']), 'p': float(v['p'])}
                    for k, v in grid_results.items()},
    'monte_carlo': {
        'n_samples': n_random,
        'mean_rho': float(mc_rhos.mean()),
        'std_rho': float(mc_rhos.std()),
        'min_rho': float(mc_rhos.min()),
        'max_rho': float(mc_rhos.max()),
        'pct_significant': float((mc_rhos < -0.25).mean() * 100)
    },
    'multi_model': {k: {kk: {'rho': float(vv['rho']), 'p': float(vv['p'])}
                        for kk, vv in v.items()}
                    for k, v in multi_results.items()}
}
with open(f'{RESULTS_DIR}/weight_sensitivity.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f'[Saved] weight_sensitivity.json')
print('\n[DONE]')
