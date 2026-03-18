#!/usr/bin/env python3
"""
Generate McNemar p-value heatmap for the paper.
Two panels: base models (Group 1) and FGVC models (Group 2).
Output: paper/figures/mcnemar_heatmap.pdf
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Elsevier IVC figure standards ---
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

RESULTS_DIR = 'results'

# Load McNemar results
with open(os.path.join(RESULTS_DIR, 'mcnemar_results.json')) as f:
    data = json.load(f)

pairwise = data['pairwise']

# Define model groups (must match ordering groups from McNemar analysis)
GROUP1 = ['ResNet-18 (CE)', 'ResNet-18 (TACA)', 'ResNet-50 (CE)', 'ResNet-50 (TACA)',
          'EfficientNet-B0 (CE)', 'ViT-B/16 (CE)', 'Swin-T (CE)']
GROUP1_SHORT = ['R18-CE', 'R18-TACA', 'R50-CE', 'R50-TACA', 'EffNet', 'ViT-B/16', 'Swin-T']

GROUP2 = ['TransFG', 'PMG', 'Graph-FGVC']
GROUP2_SHORT = ['TransFG', 'PMG', 'Graph-FGVC']

def build_pvalue_matrix(models, pairwise_data):
    """Build symmetric p-value matrix from pairwise results."""
    n = len(models)
    P = np.ones((n, n))  # diagonal = 1.0
    for i in range(n):
        for j in range(i+1, n):
            key1 = f"{models[i]}_vs_{models[j]}"
            key2 = f"{models[j]}_vs_{models[i]}"
            if key1 in pairwise_data:
                P[i, j] = pairwise_data[key1]['p_value']
                P[j, i] = P[i, j]
            elif key2 in pairwise_data:
                P[i, j] = pairwise_data[key2]['p_value']
                P[j, i] = P[i, j]
    return P

def build_acc_vector(models, pairwise_data):
    """Extract accuracy for each model from pairwise data."""
    accs = {}
    for key, val in pairwise_data.items():
        accs[val['model_a']] = val['acc_a']
        accs[val['model_b']] = val['acc_b']
    return [accs.get(m, 0) for m in models]

P1 = build_pvalue_matrix(GROUP1, pairwise)
P2 = build_pvalue_matrix(GROUP2, pairwise)
acc1 = build_acc_vector(GROUP1, pairwise)
acc2 = build_acc_vector(GROUP2, pairwise)

# Custom colormap: green (significant) -> white (borderline) -> red (not significant)
# Actually: use -log10(p) for better visualization
# Or simply: color by significance level

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5),  # 190mm double-column
                                 gridspec_kw={'width_ratios': [6, 3], 'wspace': 0.35})

def plot_pvalue_heatmap(ax, P, labels, accs, title):
    n = len(labels)

    # Use -log10(p) for visualization, cap at 5
    with np.errstate(divide='ignore'):
        logP = -np.log10(P + 1e-20)
    logP = np.clip(logP, 0, 5)
    # Diagonal -> NaN (mask)
    for i in range(n):
        logP[i, i] = np.nan

    # Custom colormap
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad('white')

    im = ax.imshow(logP, cmap=cmap, vmin=0, vmax=5, aspect='equal')

    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, f'{accs[i]*100:.1f}%', ha='center', va='center',
                       fontsize=8, fontweight='bold', color='black')
            else:
                p = P[i, j]
                if p < 0.001:
                    txt = f'{p:.0e}'
                    color = 'white'
                elif p < 0.01:
                    txt = f'{p:.3f}'
                    color = 'white'
                elif p < 0.05:
                    txt = f'{p:.3f}'
                    color = 'black'
                else:
                    txt = f'{p:.2f}'
                    color = 'black'

                # Add significance stars
                if p < 0.001:
                    txt += '\n***'
                elif p < 0.01:
                    txt += '\n**'
                elif p < 0.05:
                    txt += '\n*'
                else:
                    txt += '\nn.s.'

                ax.text(j, i, txt, ha='center', va='center',
                       fontsize=7, color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    return im

im1 = plot_pvalue_heatmap(ax1, P1, GROUP1_SHORT, acc1,
                           'General Architectures\n(paired McNemar test)')
im2 = plot_pvalue_heatmap(ax2, P2, GROUP2_SHORT, acc2,
                           'FGVC-Specialized\n(paired McNemar test)')

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.set_label('$-\\log_{10}(p)$', fontsize=10)
cbar.set_ticks([0, 1.3, 2, 3, 5])
cbar.set_ticklabels(['1.0', '0.05*', '0.01**', '0.001***', '$\\leq10^{-5}$'])

fig.suptitle('McNemar Paired Significance Tests (seed 42, n=480)',
             fontsize=12, y=0.98)

# Note about cross-group comparison
fig.text(0.5, 0.01,
         'Note: Cross-group comparisons are invalid (different test set ordering). Diagonal shows accuracy.',
         ha='center', fontsize=8, style='italic', color='gray')

os.makedirs('paper/figures', exist_ok=True)
fig.savefig('paper/figures/mcnemar_heatmap.pdf', bbox_inches='tight', dpi=300)
fig.savefig('paper/figures/mcnemar_heatmap.png', bbox_inches='tight', dpi=150)
print('[Saved] paper/figures/mcnemar_heatmap.pdf')
print('[Saved] paper/figures/mcnemar_heatmap.png')
plt.close()
