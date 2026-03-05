#!/usr/bin/env python3
"""
Regenerate ALL paper figures with improved visual quality.
Fixes: label overlap, font sizes, legend position, color accessibility,
       t-SNE convex hulls, heatmap contrast, consistent styling.
Run from: /Users/musae/Desktop/CNN for Knot/
Estimated time: ~10-15 minutes total
"""
import os, sys, json, glob, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm_mpl
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull
import seaborn as sns

# ── Global style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})
plt.style.use('seaborn-v0_8-whitegrid')

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
RESULTS_DIR = 'results'
FIG_DIR = f'{RESULTS_DIR}/figures'
PAPER_FIG_DIR = 'paper/figures'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

MODEL_NAMES = {
    'resnet18': 'ResNet-18', 'resnet50': 'ResNet-50',
    'efficientnet_b0': 'EfficientNet-B0', 'vit': 'ViT-B/16', 'swin_t': 'Swin-T'
}
MODEL_ORDER = ['resnet18','resnet50','efficientnet_b0','vit','swin_t']
# Colorblind-friendly palette (works in B&W via distinct markers/hatches)
COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
MARKERS = ['o','s','^','D','v']

# ── Knot properties ──
KP = {
    'OHK':{'name':'Overhand Knot','cn':3,'type':'prime','notation':'3₁','family':'stopper','comp':1},
    'SK': {'name':'Slip Knot','cn':3,'type':'slip','notation':'3₁','family':'stopper','comp':1},
    'F8K':{'name':'Figure-8 Knot','cn':4,'type':'prime','notation':'4₁','family':'stopper','comp':1},
    'BK': {'name':'Bowline','cn':4,'type':'loop','notation':'loop','family':'loop','comp':1},
    'F8L':{'name':'Figure-8 Loop','cn':4,'type':'loop','notation':'4₁','family':'loop','comp':1},
    'ABK':{'name':'Alpine Butterfly','cn':4,'type':'loop','notation':'loop','family':'loop','comp':1},
    'CH': {'name':'Clove Hitch','cn':2,'type':'hitch','notation':'hitch','family':'hitch','comp':1},
    'RK': {'name':'Reef Knot','cn':6,'type':'composite','notation':'3₁#3₁','family':'binding','comp':2},
    'FSK':{'name':"Fisherman's Knot",'cn':6,'type':'composite','notation':'3₁#3₁','family':'bend','comp':2},
    'FMB':{'name':'Flemish Bend','cn':8,'type':'composite','notation':'4₁#4₁','family':'bend','comp':2},
}
FAM_COLORS = {'loop':'#0072B2','stopper':'#E69F00','hitch':'#009E73','binding':'#D55E00','bend':'#CC79A7'}

# ── Load results ──
results = {}
for k in MODEL_ORDER:
    fp = f'{RESULTS_DIR}/{k}_results.json'
    if os.path.exists(fp):
        with open(fp) as f:
            results[k] = json.load(f)
print(f'Loaded models: {list(results.keys())}')

t_start = time.time()

# ═══════════════════════════════════════════════════
# Figure 1: Difficulty Tiers (fix: label overlap, B&W patterns)
# ═══════════════════════════════════════════════════
print('\n[1/8] Difficulty tiers...', flush=True)
tiers = {'Easy': ['CH','ABK','BK'], 'Medium': ['F8K','F8L','OHK','SK'], 'Hard': ['RK','FSK','FMB']}
tier_colors = {'Easy':'#009E73', 'Medium':'#E69F00', 'Hard':'#D55E00'}
tier_hatches = {'Easy':'', 'Medium':'///', 'Hard':'xxx'}

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = 0; yticks, ylabels = [], []
for tier, knots in tiers.items():
    for k in knots:
        p = KP[k]
        bar = ax.barh(y_pos, p['cn'], color=tier_colors[tier],
                alpha=0.8, height=0.6, edgecolor='black', linewidth=0.5,
                hatch=tier_hatches[tier])
        # Label INSIDE bar for short bars, OUTSIDE for long
        label_text = f"  {p['name']}"
        ax.text(p['cn'] + 0.15, y_pos, label_text, va='center', fontsize=9)
        yticks.append(y_pos)
        ylabels.append(k)
        y_pos += 1
    y_pos += 0.5

ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=10, fontfamily='monospace')
ax.set_xlabel('Visual Crossing Number ($C_{vis}$)', fontsize=12)
ax.set_title('Knot Difficulty Tiers', fontsize=14, pad=12)
ax.set_xlim(0, 10)
legend = [Patch(fc=c, ec='black', hatch=tier_hatches[t], label=t) for t,c in tier_colors.items()]
ax.legend(handles=legend, loc='lower right', fontsize=10)
plt.tight_layout()
for d in [FIG_DIR, PAPER_FIG_DIR]:
    fig.savefig(f'{d}/difficulty_tiers.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{d}/difficulty_tiers.png', dpi=200, bbox_inches='tight')
plt.close()
print('  [OK] difficulty_tiers', flush=True)

# ═══════════════════════════════════════════════════
# Figure 2: Model Comparison (fix: y-axis font, model name spacing)
# ═══════════════════════════════════════════════════
print('[2/8] Model comparison...', flush=True)
models_list = [MODEL_NAMES[k] for k in MODEL_ORDER if k in results]
test_acc = [results[k]['test_acc']*100 for k in MODEL_ORDER if k in results]
val_acc = [results[k]['best_val_acc']*100 for k in MODEL_ORDER if k in results]

x = np.arange(len(models_list))
w = 0.32
fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - w/2, val_acc, w, label='Validation', color='#90CAF9', edgecolor='white', linewidth=0.5)
b2 = ax.bar(x + w/2, test_acc, w, label='Test (Tight)', color='#1565C0', edgecolor='white', linewidth=0.5)

for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Model Comparison on Knots-10', fontsize=14, pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models_list, fontsize=11, rotation=0)
ax.set_ylim(90, 102)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
for d in [FIG_DIR, PAPER_FIG_DIR]:
    fig.savefig(f'{d}/model_comparison.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{d}/model_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print('  [OK] model_comparison', flush=True)

# ═══════════════════════════════════════════════════
# Figure 3: F1 Heatmap (fix: x-axis label rotation uniform)
# ═══════════════════════════════════════════════════
print('[3/8] F1 heatmap...', flush=True)
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
            ax=ax, linewidths=0.5, cbar_kws={'label': 'F1 Score'})
ax.set_title('Per-Class F1 Score', fontsize=14, pad=12)
ax.set_xticklabels(CLASSES, rotation=0, fontsize=10, fontfamily='monospace')  # uniform, no rotation
ax.set_yticklabels([MODEL_NAMES[k] for k in MODEL_ORDER if k in results], rotation=0, fontsize=10)
plt.tight_layout()
for d in [FIG_DIR, PAPER_FIG_DIR]:
    fig.savefig(f'{d}/f1_heatmap.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{d}/f1_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print('  [OK] f1_heatmap', flush=True)

# ═══════════════════════════════════════════════════
# Figure 4: Convergence (fix: legend position → lower right)
# ═══════════════════════════════════════════════════
print('[4/8] Convergence curves...', flush=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for i, k in enumerate(MODEL_ORDER):
    if k not in results: continue
    h = results[k]['history']
    ep = range(1, len(h['train_acc'])+1)
    ax1.plot(ep, [a*100 for a in h['train_acc']],
             color=COLORS[i], label=MODEL_NAMES[k], linewidth=1.8, marker=MARKERS[i],
             markevery=3, markersize=4)
    ax2.plot(ep, [a*100 for a in h['val_acc']],
             color=COLORS[i], label=MODEL_NAMES[k], linewidth=1.8, marker=MARKERS[i],
             markevery=3, markersize=4)

ax1.set_xlabel('Epoch', fontsize=12); ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Training Accuracy', fontsize=13)
ax1.legend(fontsize=9, loc='lower right')
ax1.set_ylim(80, 101)
ax2.set_xlabel('Epoch', fontsize=12); ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Validation Accuracy', fontsize=13)
ax2.legend(fontsize=9, loc='lower right')  # moved from upper right
ax2.set_ylim(80, 101)
plt.tight_layout()
for d in [FIG_DIR, PAPER_FIG_DIR]:
    fig.savefig(f'{d}/convergence_all.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{d}/convergence_all.png', dpi=200, bbox_inches='tight')
plt.close()
print('  [OK] convergence_all', flush=True)

# ═══════════════════════════════════════════════════
# Figure 5: Confusion matrices (fix: add "near-perfect" annotation for Swin-T)
# ═══════════════════════════════════════════════════
print('[5/8] Confusion matrices...', flush=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, k, title in zip(axes, ['swin_t','vit'],
                         ['Swin-T (99.4%)', 'ViT-B/16 (95.2%)']):
    if k not in results: continue
    cm = np.array(results[k]['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES,
                ax=ax, square=True, linewidths=0.5, cbar=False)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(CLASSES, rotation=0, fontsize=9)

plt.tight_layout()
for d in [FIG_DIR, PAPER_FIG_DIR]:
    fig.savefig(f'{d}/confusion_best_worst.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{d}/confusion_best_worst.png', dpi=200, bbox_inches='tight')
plt.close()
print('  [OK] confusion_best_worst', flush=True)

# ═══════════════════════════════════════════════════
# Figure 7+8: Distance matrices + t-SNE (fix: contrast, convex hulls)
# Requires model inference
# ═══════════════════════════════════════════════════
print('[6/8] Embedding analysis (distance matrices + t-SNE)...', flush=True)

import torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from sklearn.manifold import TSNE

FAMILIES = {c: KP[c]['family'] for c in CLASSES}

# Build topological distance matrix
DR = {('OHK','SK'):0.1,('F8K','FMB'):0.15,('RK','FSK'):0.1,('F8K','F8L'):0.1}
def td(a,b):
    pa, pb = KP[a], KP[b]
    return (0.25*abs(pa['cn']-pb['cn'])/8
           +0.25*(0 if pa['family']==pb['family'] else 1)
           +0.15*(0 if pa['type']==pb['type'] else 0.5)
           +0.10*abs(pa['comp']-pb['comp'])
           +0.25*DR.get(tuple(sorted([a,b])), 0.5))
tD = np.array([[td(CLASSES[i],CLASSES[j]) if i!=j else 0 for j in range(10)] for i in range(10)])

c2i = {c:i for i,c in enumerate(CLASSES)}
paths, labels_arr = [], []
for p in sorted(glob.glob('train_small/*.jpg')):
    cn = os.path.basename(p).split('_')[0]
    if cn in c2i: paths.append(p); labels_arr.append(c2i[cn])
labels_arr = np.array(labels_arr)
print(f'  {len(paths)} images for embedding extraction', flush=True)

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

class TopoModel(nn.Module):
    def __init__(self, bb, dim):
        super().__init__()
        self.backbone = bb
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 10))
    def forward(self, x): return self.classifier(self.backbone(x))

def get_feats(model, hook_target):
    ho = {}
    def hk(m,i,o): ho['f']=o.detach()
    h = hook_target.register_forward_hook(hk)
    all_f = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(paths), 64):
            imgs = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[start:start+64]])
            _ = model(imgs)
            f = ho['f'].squeeze()
            if f.ndim==1: f=f.unsqueeze(0)
            all_f.append(f.numpy())
    h.remove()
    return np.vstack(all_f)

def load_and_extract(arch, ckpt, dim, is_topo):
    base = models.resnet18(weights=None) if arch=='resnet18' else models.resnet50(weights=None)
    if is_topo:
        base.fc = nn.Identity()
        m = TopoModel(base, dim)
        ht = m.backbone.avgpool
    else:
        base.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, 10))
        m = base; ht = m.avgpool
    m.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False))
    return get_feats(m, ht)

def plot_heatmaps_improved(tD, eD_base, eD_topo, arch, save_path):
    """Improved distance matrix visualization with better contrast."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = ['Topological Distance\n(Ground Truth)',
              f'{arch} Baseline\nEmbedding Distance',
              f'{arch} Topo-Guided\nEmbedding Distance']
    for ax, D, title in zip(axes, [tD, eD_base, eD_topo], titles):
        Dn = D / D.max() if D.max() > 0 else D
        im = ax.imshow(Dn, cmap='YlOrRd', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_title(title, fontsize=11, pad=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    # Also save to paper/figures
    fname = os.path.basename(save_path)
    fig.savefig(f'{PAPER_FIG_DIR}/{fname}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] {save_path}', flush=True)

def plot_tsne_improved(feats_base, feats_topo, labels, arch, save_path):
    """Improved t-SNE with convex hulls for family clusters."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, feats, title in zip(axes, [feats_base, feats_topo],
                                 [f'{arch} Baseline', f'{arch} Topo-Guided']):
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        coords = tsne.fit_transform(feats)

        # Draw convex hulls per family
        family_points = {}
        for ci, c in enumerate(CLASSES):
            fam = FAMILIES[c]
            mask = labels == ci
            pts = coords[mask]
            if fam not in family_points:
                family_points[fam] = pts
            else:
                family_points[fam] = np.vstack([family_points[fam], pts])

        for fam, pts in family_points.items():
            if len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    hull_pts = np.append(hull.vertices, hull.vertices[0])
                    ax.fill(pts[hull_pts, 0], pts[hull_pts, 1],
                            alpha=0.1, color=FAM_COLORS[fam])
                    ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                            color=FAM_COLORS[fam], alpha=0.3, linewidth=1)
                except Exception:
                    pass

        # Scatter points
        for ci, c in enumerate(CLASSES):
            mask = labels == ci
            color = FAM_COLORS[FAMILIES[c]]
            ax.scatter(coords[mask,0], coords[mask,1],
                       c=color, label=f'{c} ({FAMILIES[c]})',
                       alpha=0.7, s=20, edgecolors='white', linewidths=0.3)

        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        # Compact legend
        handles, lbls = ax.get_legend_handles_labels()
        # Deduplicate by family
        seen = set()
        unique_h, unique_l = [], []
        for h, l in zip(handles, lbls):
            fam = l.split('(')[1].rstrip(')')
            if fam not in seen:
                seen.add(fam)
                unique_h.append(h)
                unique_l.append(fam.capitalize())
        ax.legend(unique_h, unique_l, fontsize=9, loc='best', title='Family')

    fig.suptitle(f't-SNE Embedding Visualization ({arch})', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    fname = os.path.basename(save_path)
    fig.savefig(f'{PAPER_FIG_DIR}/{fname}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] {save_path}', flush=True)

# Run embedding analysis
configs = [
    ('ResNet-18', 'resnet18', 512,
     'checkpoints/resnet18_best.pth', 'checkpoints/resnet18_topo_guided.pth'),
    ('ResNet-50', 'resnet50', 2048,
     'checkpoints/resnet50_best.pth', 'checkpoints/resnet50_topo_guided.pth'),
]

for display, arch, dim, ckpt_base, ckpt_topo in configs:
    if not os.path.exists(ckpt_base) or not os.path.exists(ckpt_topo):
        print(f'  [SKIP] {display}: checkpoint not found', flush=True)
        continue
    print(f'  === {display} ===', flush=True)
    feats_base = load_and_extract(arch, ckpt_base, dim, False)
    print(f'    Baseline embeddings: {feats_base.shape}', flush=True)
    feats_topo = load_and_extract(arch, ckpt_topo, dim, True)
    print(f'    Topo embeddings: {feats_topo.shape}', flush=True)

    cent_base = np.array([feats_base[labels_arr==c].mean(0) for c in range(10)])
    cent_topo = np.array([feats_topo[labels_arr==c].mean(0) for c in range(10)])
    eD_base = np.array([[np.linalg.norm(cent_base[i]-cent_base[j]) for j in range(10)] for i in range(10)])
    eD_topo = np.array([[np.linalg.norm(cent_topo[i]-cent_topo[j]) for j in range(10)] for i in range(10)])

    plot_heatmaps_improved(tD, eD_base, eD_topo, display,
                  f'{FIG_DIR}/distance_comparison_{arch}.png')
    print(f'    Starting t-SNE...', flush=True)
    plot_tsne_improved(feats_base, feats_topo, labels_arr, display,
              f'{FIG_DIR}/tsne_comparison_{arch}.png')

print('  [OK] Embedding analysis complete', flush=True)

# ═══════════════════════════════════════════════════
# Figure 6: Grad-CAM (fix: uniform image size, heatmap scale bar)
# ═══════════════════════════════════════════════════
print('[7/8] Grad-CAM visualization...', flush=True)

CLASS_FULL = {
    'ABK':'Alpine\nButterfly','BK':'Bowline','CH':'Clove\nHitch',
    'F8K':'Figure-8\nKnot','F8L':'Figure-8\nLoop','FSK':"Fisher-\nman's",
    'FMB':'Flemish\nBend','OHK':'Overhand','RK':'Reef\nKnot','SK':'Slip\nKnot'
}
CKPT_DIR = 'checkpoints'
DATA_DIR = './train'

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda m,i,o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m,gi,go: setattr(self, 'gradients', go[0].detach()))

    def __call__(self, x, target_class=None):
        self.model.eval()
        x = x.to(next(self.model.parameters()).device)
        out = self.model(x)
        pred = out.argmax(dim=1).item()
        if target_class is None:
            target_class = pred
        self.model.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot[0, target_class] = 1.0
        out.backward(gradient=one_hot)
        w = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = torch.relu((w * self.activations).sum(dim=1))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, pred, target_class

def resize_cam(cam, size=(224,224)):
    cam_uint8 = (cam * 255).astype(np.uint8)
    from PIL import Image as PILImg
    return np.array(PILImg.fromarray(cam_uint8).resize(size, PILImg.BILINEAR)).astype(np.float32) / 255.0

def overlay_heatmap(img_pil, cam, alpha=0.45):
    img_np = np.array(img_pil.resize((224,224))).astype(np.float32) / 255.0
    cam_resized = resize_cam(cam, (224,224))
    heatmap = cm_mpl.jet(cam_resized)[:,:,:3]
    overlay = alpha * heatmap + (1 - alpha) * img_np
    return np.clip(overlay, 0, 1), cam_resized

def get_test_samples(data_dir, n_per_class=1):
    samples = {}
    for cls in CLASSES:
        pattern = os.path.join(data_dir, '**', f'{cls}_*Set*.jpg')
        files = sorted(glob.glob(pattern, recursive=True))
        if files:
            samples[cls] = files[0]
    return samples

# Load or skip
ckpt_r18 = f'{CKPT_DIR}/resnet18_best.pth'
ckpt_r50 = f'{CKPT_DIR}/resnet50_best.pth'
if os.path.exists(ckpt_r18) and os.path.exists(ckpt_r50) and os.path.exists(DATA_DIR):
    m18 = models.resnet18(weights=None)
    m18.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m18.fc.in_features, 10))
    m18.load_state_dict(torch.load(ckpt_r18, map_location='cpu', weights_only=False))
    m18.eval()
    gcam18 = GradCAM(m18, m18.layer4[-1])

    m50 = models.resnet50(weights=None)
    m50.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m50.fc.in_features, 10))
    m50.load_state_dict(torch.load(ckpt_r50, map_location='cpu', weights_only=False))
    m50.eval()
    gcam50 = GradCAM(m50, m50.layer4[-1])

    te_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

    samples = get_test_samples(DATA_DIR)
    print(f'  Found {len(samples)}/{len(CLASSES)} test samples', flush=True)

    n_cls = len(CLASSES)
    fig, axes = plt.subplots(3, n_cls, figsize=(2.2*n_cls, 7),
                             gridspec_kw={'hspace':0.15, 'wspace':0.05})
    row_labels = ['Original', 'ResNet-18', 'ResNet-50']

    for col, cls in enumerate(CLASSES):
        if cls not in samples:
            for row in range(3): axes[row, col].axis('off')
            continue
        img_pil = Image.open(samples[cls]).convert('RGB')
        x = te_tf(img_pil).unsqueeze(0)

        # Row 0: original
        axes[0, col].imshow(np.array(img_pil.resize((224,224))))
        axes[0, col].set_title(CLASS_FULL.get(cls, cls), fontsize=7, pad=3, fontweight='bold')
        axes[0, col].axis('off')

        # Row 1: ResNet-18
        cam18, pred18, _ = gcam18(x)
        ov18, _ = overlay_heatmap(img_pil, cam18)
        axes[1, col].imshow(ov18)
        axes[1, col].axis('off')
        c18 = '#2ecc71' if CLASSES[pred18]==cls else '#e74c3c'
        axes[1, col].text(0.5, -0.02, CLASSES[pred18],
            transform=axes[1,col].transAxes, ha='center',
            va='top', fontsize=7, color=c18, fontweight='bold')

        # Row 2: ResNet-50
        cam50, pred50, _ = gcam50(x)
        ov50, _ = overlay_heatmap(img_pil, cam50)
        axes[2, col].imshow(ov50)
        axes[2, col].axis('off')
        c50 = '#2ecc71' if CLASSES[pred50]==cls else '#e74c3c'
        axes[2, col].text(0.5, -0.02, CLASSES[pred50],
            transform=axes[2,col].transAxes, ha='center',
            va='top', fontsize=7, color=c50, fontweight='bold')

    # Row labels
    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.15, 0.5, label,
            transform=axes[row,0].transAxes, ha='right',
            va='center', fontsize=10, fontweight='bold', rotation=90)

    # Heatmap color bar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.3, pad=0.01,
                        orientation='horizontal', aspect=40)
    cbar.set_label('Activation Intensity', fontsize=8)

    for d in [FIG_DIR, PAPER_FIG_DIR]:
        fig.savefig(f'{d}/gradcam_comparison.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
        fig.savefig(f'{d}/gradcam_comparison.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print('  [OK] gradcam_comparison', flush=True)
else:
    print('  [SKIP] Grad-CAM: checkpoints or data not found', flush=True)

# ═══════════════════════════════════════════════════
# Copy figures to paper/figures/ (for figures not already copied)
# ═══════════════════════════════════════════════════
print('[8/8] Copying figures to paper/figures/...', flush=True)
import shutil
for fname in ['difficulty_tiers.pdf', 'model_comparison.pdf', 'f1_heatmap.pdf',
              'convergence_all.pdf', 'confusion_best_worst.pdf']:
    src = f'{FIG_DIR}/{fname}'
    dst = f'{PAPER_FIG_DIR}/{fname}'
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f'  Copied {fname}', flush=True)

elapsed = time.time() - t_start
print(f'\n{"="*50}')
print(f'ALL FIGURES REGENERATED in {elapsed:.0f}s ({elapsed/60:.1f} min)')
print(f'{"="*50}')
print(f'Output directories:')
print(f'  {FIG_DIR}/')
print(f'  {PAPER_FIG_DIR}/')
