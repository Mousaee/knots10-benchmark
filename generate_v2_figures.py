#!/usr/bin/env python3
"""Generate all figures for paper V2 (5 models)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.style.use('seaborn-v0_8-whitegrid')

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
RESULTS_DIR = 'results'
FIG_DIR = f'{RESULTS_DIR}/figures'
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_NAMES = {
    'resnet18': 'ResNet-18',
    'resnet50': 'ResNet-50',
    'efficientnet_b0': 'EfficientNet-B0',
    'vit': 'ViT-B/16',
    'swin_t': 'Swin-T'
}
MODEL_ORDER = ['resnet18','resnet50','efficientnet_b0','vit','swin_t']
COLORS = ['#2196F3','#1565C0','#4CAF50','#FF9800','#F44336']

# Load all results
results = {}
for k in MODEL_ORDER:
    fp = f'{RESULTS_DIR}/{k}_results.json'
    if os.path.exists(fp):
        with open(fp) as f:
            results[k] = json.load(f)
print(f'Loaded: {list(results.keys())}')

# ===== Fig 1: Overall comparison bar chart =====
def fig_overall_comparison():
    models = [MODEL_NAMES[k] for k in MODEL_ORDER if k in results]
    test_acc = [results[k]['test_acc']*100 for k in MODEL_ORDER if k in results]
    val_acc = [results[k]['best_val_acc']*100 for k in MODEL_ORDER if k in results]
    
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, val_acc, w, label='Validation', color='#90CAF9', edgecolor='white')
    b2 = ax.bar(x + w/2, test_acc, w, label='Test (Set)', color='#1565C0', edgecolor='white')
    
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Comparison on Knots-10', fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(90, 101)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/model_comparison.pdf', dpi=300)
    fig.savefig(f'{FIG_DIR}/model_comparison.png', dpi=150)
    plt.close()
    print('[Saved] model_comparison')

fig_overall_comparison()

# ===== Fig 2: Convergence curves (all 5 models) =====
def fig_convergence():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, k in enumerate(MODEL_ORDER):
        if k not in results: continue
        h = results[k]['history']
        ep = range(1, len(h['train_acc'])+1)
        ax1.plot(ep, [a*100 for a in h['train_acc']],
                 color=COLORS[i], label=MODEL_NAMES[k], linewidth=1.5)
        ax2.plot(ep, [a*100 for a in h['val_acc']],
                 color=COLORS[i], label=MODEL_NAMES[k], linewidth=1.5)
    
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training Accuracy'); ax1.legend(fontsize=8)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy'); ax2.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/convergence_all.pdf', dpi=300)
    fig.savefig(f'{FIG_DIR}/convergence_all.png', dpi=150)
    plt.close()
    print('[Saved] convergence_all')

fig_convergence()

# ===== Fig 3: Confusion matrices (best & worst) =====
def fig_confusion_matrices():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, k, title in zip(axes, ['swin_t','vit'],
                             ['Swin-T (Best: 99.4%)', 'ViT-B/16 (95.2%)']):
        cm = np.array(results[k]['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    ax=ax, square=True, linewidths=0.5, cbar=False)
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/confusion_best_worst.pdf', dpi=300)
    fig.savefig(f'{FIG_DIR}/confusion_best_worst.png', dpi=150)
    plt.close()
    print('[Saved] confusion_best_worst')

fig_confusion_matrices()

# ===== Fig 4: Per-class F1 heatmap =====
def fig_f1_heatmap():
    f1_data = []
    for k in MODEL_ORDER:
        if k not in results: continue
        rpt = results[k]['report']
        row = [rpt[c]['f1-score'] for c in CLASSES]
        f1_data.append(row)
    
    f1_arr = np.array(f1_data)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(f1_arr, annot=True, fmt='.2f',
                cmap='RdYlGn', vmin=0.85, vmax=1.0,
                xticklabels=CLASSES,
                yticklabels=[MODEL_NAMES[k] for k in MODEL_ORDER if k in results],
                ax=ax, linewidths=0.5)
    ax.set_title('Per-Class F1 Score', fontsize=14, pad=12)
    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/f1_heatmap.pdf', dpi=300)
    fig.savefig(f'{FIG_DIR}/f1_heatmap.png', dpi=150)
    plt.close()
    print('[Saved] f1_heatmap')

fig_f1_heatmap()

# ===== Fig 5: Efficiency scatter (acc vs params vs time) =====
def fig_efficiency():
    params_m = {
        'resnet18': 11.2, 'resnet50': 23.5,
        'efficientnet_b0': 5.3, 'vit': 86.6, 'swin_t': 28.3
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, k in enumerate(MODEL_ORDER):
        if k not in results: continue
        t = results[k]['test_acc'] * 100
        p = params_m[k]
        tt = results[k]['train_time']
        ax.scatter(p, t, s=tt/8, c=COLORS[i],
                   alpha=0.7, edgecolors='white', linewidth=1.5)
        ax.annotate(MODEL_NAMES[k], (p, t),
                    textcoords="offset points",
                    xytext=(8, 5), fontsize=9)
    ax.set_xlabel('Parameters (M)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Model Size\n(bubble size = training time)',
                 fontsize=12, pad=12)
    plt.tight_layout()
    fig.savefig(f'{FIG_DIR}/efficiency.pdf', dpi=300)
    fig.savefig(f'{FIG_DIR}/efficiency.png', dpi=150)
    plt.close()
    print('[Saved] efficiency')

fig_efficiency()
print('\n[ALL FIGURES DONE]')
