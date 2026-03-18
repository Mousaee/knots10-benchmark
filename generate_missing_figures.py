#!/usr/bin/env python3
"""
Generate weight_evolution.pdf and mantel_null_distribution.pdf
from existing JSON result data. No GPU required.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 300,
})
plt.style.use('seaborn-v0_8-whitegrid')

RESULTS_DIR = 'results'
FIG_DIR = f'{RESULTS_DIR}/figures'
PAPER_FIG_DIR = 'paper/figures'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

FACTOR_NAMES = [
    'Crossing Number ($w_1$)',
    'Family ($w_2$)',
    'Type ($w_3$)',
    'Components ($w_4$)',
    'Derivation ($w_5$)',
]
FACTOR_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']

# ═══════════════════════════════════════════
# Figure B4: Weight Evolution
# ═══════════════════════════════════════════
print('[1/2] Weight evolution...', flush=True)

with open(f'{RESULTS_DIR}/learnable_weights_resnet18.json') as f:
    r18 = json.load(f)
with open(f'{RESULTS_DIR}/learnable_weights_resnet50.json') as f:
    r50 = json.load(f)

trajectory = r18.get('weight_trajectory', [])
if trajectory:
    traj = np.array(trajectory)  # shape: (epochs, 5)
    epochs = range(1, len(traj) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: ResNet-18 trajectory
    ax = axes[0]
    for i in range(5):
        ax.plot(epochs, traj[:, i], color=FACTOR_COLORS[i],
                label=FACTOR_NAMES[i], linewidth=2.0, marker='o',
                markersize=4, markevery=2)
    ax.axhline(0.2, color='gray', linestyle='--', alpha=0.5, label='Uniform (0.2)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight Value')
    ax.set_title('ResNet-18: Weight Evolution', fontsize=13)
    ax.set_ylim(0.10, 0.30)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # Right: Final comparison R18 vs R50
    ax = axes[1]
    r18_final = r18['learned_weights']
    r50_final = r50['learned_weights']
    x = np.arange(5)
    w = 0.3
    bars1 = ax.bar(x - w/2, r18_final, w, label='ResNet-18', color='#1565C0',
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, r50_final, w, label='ResNet-50', color='#E53935',
                   edgecolor='white', linewidth=0.5)
    ax.axhline(0.2, color='gray', linestyle='--', alpha=0.5, label='Uniform')
    ax.set_xlabel('Distance Factor')
    ax.set_ylabel('Learned Weight')
    ax.set_title('Final Learned Weights', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(['$w_1$\nCrossing', '$w_2$\nFamily', '$w_3$\nType',
                         '$w_4$\nComp.', '$w_5$\nDeriv.'], fontsize=9)
    ax.set_ylim(0.10, 0.30)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    for d in [FIG_DIR, PAPER_FIG_DIR]:
        fig.savefig(f'{d}/weight_evolution.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'{d}/weight_evolution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  [OK] weight_evolution', flush=True)
else:
    print('  [SKIP] No weight trajectory data in R18 JSON', flush=True)


# ═══════════════════════════════════════════
# Figure B5: Mantel Null Distribution
# ═══════════════════════════════════════════
print('[2/2] Mantel null distribution...', flush=True)

with open(f'{RESULTS_DIR}/mantel_test_results.json') as f:
    mantel = json.load(f)

# We need to regenerate null distributions since they weren't saved.
# Use the same methodology as mantel_test.py but with saved confusion matrices.
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform

# Build topological distance matrix
CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
KP = {
    'ABK': {'cn':4,'family':'loop','type':'loop','comp':1},
    'BK':  {'cn':4,'family':'loop','type':'loop','comp':1},
    'CH':  {'cn':2,'family':'hitch','type':'hitch','comp':1},
    'F8K': {'cn':4,'family':'stopper','type':'prime','comp':1},
    'F8L': {'cn':4,'family':'loop','type':'loop','comp':1},
    'FSK': {'cn':6,'family':'bend','type':'composite','comp':2},
    'FMB': {'cn':8,'family':'bend','type':'composite','comp':2},
    'OHK': {'cn':3,'family':'stopper','type':'prime','comp':1},
    'RK':  {'cn':6,'family':'binding','type':'composite','comp':2},
    'SK':  {'cn':3,'family':'stopper','type':'slip','comp':1},
}
DR = {('OHK','SK'):0.1,('F8K','FMB'):0.15,('RK','FSK'):0.1,('F8K','F8L'):0.1}

def td(a, b):
    pa, pb = KP[a], KP[b]
    return (0.25*abs(pa['cn']-pb['cn'])/8
           +0.25*(0 if pa['family']==pb['family'] else 1)
           +0.15*(0 if pa['type']==pb['type'] else 0.5)
           +0.10*abs(pa['comp']-pb['comp'])
           +0.25*DR.get(tuple(sorted([a,b])), 0.5))

K = len(CLASSES)
topo_dist = np.array([[td(CLASSES[i],CLASSES[j]) if i!=j else 0
                        for j in range(K)] for i in range(K)])

# Extract upper triangle indices
topo_vec = squareform(topo_dist, checks=False)

BASELINE_MODELS = ['resnet18', 'resnet50', 'efficientnet_b0', 'vit', 'swin_t']
MODEL_DISPLAY = {
    'resnet18': 'ResNet-18', 'resnet50': 'ResNet-50',
    'efficientnet_b0': 'EfficientNet-B0', 'vit': 'ViT-B/16', 'swin_t': 'Swin-T'
}
N_PERMS = 9999

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, model_name in enumerate(BASELINE_MODELS):
    ax = axes[idx]

    # Load confusion matrix from results
    rpath = f'{RESULTS_DIR}/{model_name}_results.json'
    with open(rpath) as f:
        rdata = json.load(f)
    cm = np.array(rdata['confusion_matrix'], dtype=float)

    # Compute confusion rate matrix (symmetrized, matching mantel_test.py)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_rate = cm / row_sums
    np.fill_diagonal(conf_rate, 0)
    conf_rate = (conf_rate + conf_rate.T) / 2  # symmetrize
    conf_vec = squareform(conf_rate, checks=False)

    # Observed correlation
    r_obs, _ = spearmanr(topo_vec, conf_vec)

    # Permutation test (seed=42 to match original mantel_test.py)
    np.random.seed(42)
    null_dist = np.zeros(N_PERMS)
    for p in range(N_PERMS):
        perm = np.random.permutation(K)
        perm_conf = conf_rate[np.ix_(perm, perm)]
        perm_vec = squareform(perm_conf, checks=False)
        null_dist[p], _ = spearmanr(topo_vec, perm_vec)

    p_value = (np.sum(np.abs(null_dist) >= np.abs(r_obs)) + 1) / (N_PERMS + 1)

    # Plot
    ax.hist(null_dist, bins=50, alpha=0.7, color='#90CAF9', edgecolor='#1565C0',
            linewidth=0.5)
    ax.axvline(r_obs, color='#D32F2F', linestyle='--', linewidth=2.5,
               label=f'$r_{{obs}}$ = {r_obs:.3f}')

    crit_val = np.percentile(np.abs(null_dist), 95)
    ax.axvline(crit_val, color='#FF9800', linestyle=':', linewidth=2, alpha=0.7,
               label=f'Critical ($\\alpha$=0.05)')
    ax.axvline(-crit_val, color='#FF9800', linestyle=':', linewidth=2, alpha=0.7)

    sig_marker = ''
    if p_value < 0.001: sig_marker = '***'
    elif p_value < 0.01: sig_marker = '**'
    elif p_value < 0.05: sig_marker = '*'

    ax.set_title(f'{MODEL_DISPLAY[model_name]}\n$p$ = {p_value:.4f} {sig_marker}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Spearman correlation', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

# Hide last subplot
axes[5].axis('off')

plt.suptitle('Mantel Permutation Test: Null Distributions\n(9,999 permutations, two-tailed)',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
for d in [FIG_DIR, PAPER_FIG_DIR]:
    fig.savefig(f'{d}/mantel_null_distribution.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{d}/mantel_null_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print('  [OK] mantel_null_distribution', flush=True)

print('\nDone. Generated:')
print('  - weight_evolution.pdf/png')
print('  - mantel_null_distribution.pdf/png')
