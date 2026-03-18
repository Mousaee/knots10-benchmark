#!/usr/bin/env python3
"""
Topological Distance Analysis for Knots-10
Core theoretical contribution: correlate topological similarity with visual confusion.
"""
import json, os
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams.update({
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
plt.style.use('seaborn-v0_8-whitegrid')

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
RESULTS_DIR = 'results'
os.makedirs(f'{RESULTS_DIR}/figures', exist_ok=True)

# =============================================
# 1. Define Topological Properties
# =============================================

KNOT_PROPERTIES = {
    'OHK': {'name':'Overhand Knot', 'crossing_num':3, 'type':'prime',
            'notation':'3_1', 'family':'stopper', 'components':1},
    'F8K': {'name':'Figure-8 Knot', 'crossing_num':4, 'type':'prime',
            'notation':'4_1', 'family':'stopper', 'components':1},
    'BK':  {'name':'Bowline Knot', 'crossing_num':4, 'type':'loop',
            'notation':'loop', 'family':'loop', 'components':1},
    'RK':  {'name':'Reef Knot', 'crossing_num':6, 'type':'composite',
            'notation':'3_1#3_1*', 'family':'binding', 'components':2},
    'FSK': {'name':"Fisherman's Knot", 'crossing_num':6, 'type':'composite',
            'notation':'3_1#3_1', 'family':'bend', 'components':2},
    'FMB': {'name':'Flemish Bend', 'crossing_num':8, 'type':'composite',
            'notation':'4_1_traced', 'family':'bend', 'components':2},
    'F8L': {'name':'Figure-8 Loop', 'crossing_num':4, 'type':'loop',
            'notation':'4_1_loop', 'family':'loop', 'components':1},
    'CH':  {'name':'Clove Hitch', 'crossing_num':2, 'type':'hitch',
            'notation':'hitch', 'family':'hitch', 'components':1},
    'SK':  {'name':'Slip Knot', 'crossing_num':3, 'type':'slip',
            'notation':'3_1_slip', 'family':'stopper', 'components':1},
    'ABK': {'name':'Alpine Butterfly', 'crossing_num':4, 'type':'loop',
            'notation':'loop_mid', 'family':'loop', 'components':1},
}

# =============================================
# 2. Topological Distance Matrix
# =============================================

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
    # OHK-SK share 3_1 structure
    # F8K-FMB share 4_1 structure  
    # RK-FSK both composite of 3_1
    derivation_pairs = {
        ('OHK','SK'): 0.1,
        ('F8K','FMB'): 0.15,
        ('RK','FSK'): 0.1,
        ('F8K','F8L'): 0.1,
    }
    pair = tuple(sorted([k1, k2]))
    d_deriv = derivation_pairs.get(pair, 0.5)
    
    # Weighted combination
    dist = 0.25*d_cross + 0.25*d_family + 0.15*d_type + 0.1*d_comp + 0.25*d_deriv
    return dist

def build_distance_matrix():
    n = len(CLASSES)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i,j] = topological_distance(CLASSES[i], CLASSES[j])
    return D

# =============================================
# 3. Confusion Rate Extraction
# =============================================

def confusion_rate_matrix(cm):
    """Normalize confusion matrix to get pairwise confusion rates."""
    n = cm.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        total = cm[i].sum()
        if total > 0:
            for j in range(n):
                if i != j:
                    C[i,j] = cm[i,j] / total
    # Symmetrize
    return (C + C.T) / 2

# =============================================
# 4. Correlation Analysis
# =============================================

def analyze_correlation(D, C, model_name):
    """Correlate topological distance with confusion rate."""
    n = len(CLASSES)
    dists, confs = [], []
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(D[i,j])
            confs.append(C[i,j])
            pairs.append((CLASSES[i], CLASSES[j]))
    
    dists = np.array(dists)
    confs = np.array(confs)
    
    rho, p_spear = spearmanr(dists, confs)
    r, p_pear = pearsonr(dists, confs)
    
    print(f"\n[{model_name}] Correlation Analysis:", flush=True)
    print(f"  Spearman rho = {rho:.4f} (p = {p_spear:.4e})", flush=True)
    print(f"  Pearson  r   = {r:.4f} (p = {p_pear:.4e})", flush=True)
    
    # Top confused pairs
    idx = np.argsort(-confs)[:5]
    print(f"  Top-5 confused pairs:", flush=True)
    for k in idx:
        if confs[k] > 0:
            print(f"    {pairs[k][0]}-{pairs[k][1]}: "
                  f"conf={confs[k]:.3f}, dist={dists[k]:.3f}", flush=True)
    
    return dists, confs, rho, p_spear, r, p_pear, pairs

# =============================================
# 5. Visualization
# =============================================

def plot_distance_heatmap(D):
    """Plot topological distance matrix."""
    fig, ax = plt.subplots(figsize=(3.54, 3.2))  # 90mm single-column
    sns.heatmap(D, xticklabels=CLASSES, yticklabels=CLASSES,
                annot=True, fmt='.2f', cmap='YlOrRd',
                ax=ax, square=True, linewidths=0.5)
    ax.set_title('Topological Distance Matrix', fontsize=14, pad=12)
    ax.set_xlabel('Knot Class'); ax.set_ylabel('Knot Class')
    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/figures/topological_distance.pdf', dpi=300)
    fig.savefig(f'{RESULTS_DIR}/figures/topological_distance.png', dpi=150)
    plt.close()
    print('[Saved] topological_distance.pdf/png', flush=True)

def plot_scatter(dists, confs, rho, p_spear, model_name):
    """Scatter plot: topological distance vs confusion rate."""
    fig, ax = plt.subplots(figsize=(3.54, 3))  # 90mm single-column
    ax.scatter(dists, confs, alpha=0.6, s=60, c='#2196F3', edgecolors='white')
    
    # Fit line
    if len(dists) > 2:
        z = np.polyfit(dists, confs, 1)
        xr = np.linspace(dists.min(), dists.max(), 100)
        ax.plot(xr, np.polyval(z, xr), 'r--', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Topological Distance', fontsize=12)
    ax.set_ylabel('Confusion Rate', fontsize=12)
    ax.set_title(f'{model_name}: Topological Distance vs Confusion Rate\n'
                 f'Spearman ρ = {rho:.3f} (p = {p_spear:.2e})',
                 fontsize=12, pad=12)
    plt.tight_layout()
    fn = model_name.lower().replace('-','_').replace(' ','_')
    fig.savefig(f'{RESULTS_DIR}/figures/topo_vs_conf_{fn}.pdf', dpi=300)
    fig.savefig(f'{RESULTS_DIR}/figures/topo_vs_conf_{fn}.png', dpi=150)
    plt.close()
    print(f'[Saved] topo_vs_conf_{fn}.pdf/png', flush=True)

def plot_difficulty_tiers(D):
    """Visualize difficulty tiers based on topological properties."""
    tiers = {
        'Easy': ['CH','ABK','BK'],
        'Medium': ['F8K','F8L','OHK','SK'],
        'Hard': ['RK','FSK','FMB']
    }
    colors = {'Easy':'#4CAF50', 'Medium':'#FF9800', 'Hard':'#F44336'}
    
    fig, ax = plt.subplots(figsize=(7.5, 3.5))  # 190mm double-column
    y_pos = 0
    yticks, ylabels = [], []
    
    for tier, knots in tiers.items():
        for k in knots:
            p = KNOT_PROPERTIES[k]
            ax.barh(y_pos, p['crossing_num'], color=colors[tier],
                    alpha=0.8, height=0.6, edgecolor='white')
            ax.text(p['crossing_num']+0.1, y_pos,
                    f"{p['name']} ({p['notation']})",
                    va='center', fontsize=9)
            yticks.append(y_pos)
            ylabels.append(k)
            y_pos += 1
        y_pos += 0.5
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel('Crossing Number', fontsize=12)
    ax.set_title('Knot Difficulty Tiers', fontsize=14, pad=12)
    
    from matplotlib.patches import Patch
    legend = [Patch(fc=c, label=t) for t,c in colors.items()]
    ax.legend(handles=legend, loc='lower right')
    
    plt.tight_layout()
    fig.savefig(f'{RESULTS_DIR}/figures/difficulty_tiers.pdf', dpi=300)
    fig.savefig(f'{RESULTS_DIR}/figures/difficulty_tiers.png', dpi=150)
    plt.close()
    print('[Saved] difficulty_tiers.pdf/png', flush=True)

# =============================================
# 6. Main
# =============================================

if __name__ == '__main__':
    D = build_distance_matrix()
    plot_distance_heatmap(D)
    plot_difficulty_tiers(D)
    
    # Load existing results
    results_all = {}
    for fn in ['resnet18', 'vit']:
        fp = f'{RESULTS_DIR}/{fn}_results.json'
        if os.path.exists(fp):
            with open(fp) as f:
                results_all[fn] = json.load(f)

    # Also check for new models
    for fn in ['resnet50', 'efficientnet_b0', 'swin_t']:
        fp = f'{RESULTS_DIR}/{fn}_results.json'
        if os.path.exists(fp):
            with open(fp) as f:
                results_all[fn] = json.load(f)
    
    print(f'\nLoaded results for: {list(results_all.keys())}')
    
    # Correlation analysis for each model
    summary = {}
    for mname, res in results_all.items():
        cm = np.array(res['confusion_matrix'])
        C = confusion_rate_matrix(cm)
        out = analyze_correlation(D, C, mname)
        dists, confs, rho, p_s, r, p_p, pairs = out
        plot_scatter(dists, confs, rho, p_s, mname)
        summary[mname] = {
            'spearman_rho': rho,
            'spearman_p': p_s,
            'pearson_r': r,
            'pearson_p': p_p
        }

    # Save summary
    with open(f'{RESULTS_DIR}/topological_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n[Saved] topological_analysis.json')
    
    # Print summary table
    print('\n' + '='*60)
    print('TOPOLOGICAL-VISUAL CORRELATION SUMMARY')
    print('='*60)
    for m, s in summary.items():
        print(f"  {m:20s} | rho={s['spearman_rho']:.3f} "
              f"(p={s['spearman_p']:.2e}) | "
              f"r={s['pearson_r']:.3f}")
    print('='*60)
    print('[DONE]', flush=True)
