#!/usr/bin/env python3
"""
Generate UMAP comparison alongside t-SNE for ResNet-18.
Adds convex hulls for family clusters.
"""
import os, glob, json
import numpy as np
import torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print('[WARN] umap-learn not installed, trying pip install...')
    import subprocess
    subprocess.check_call(['pip', 'install', 'umap-learn'])
    from umap import UMAP
    HAS_UMAP = True

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
KP = {'OHK':'stopper','SK':'stopper','F8K':'stopper','BK':'loop','F8L':'loop',
      'ABK':'loop','CH':'hitch','RK':'binding','FSK':'bend','FMB':'bend'}
FAM_COLORS = {'loop':'#0072B2','stopper':'#E69F00','hitch':'#009E73','binding':'#D55E00','bend':'#CC79A7'}

c2i = {c:i for i,c in enumerate(CLASSES)}
paths, labels = [], []
for p in sorted(glob.glob('train_small/*.jpg')):
    cn = os.path.basename(p).split('_')[0]
    if cn in c2i: paths.append(p); labels.append(c2i[cn])
labels = np.array(labels)
print(f'{len(paths)} images')

tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
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

# Load ResNet-18 baseline and topo-guided
base = models.resnet18(weights=None)
base.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 10))
base.load_state_dict(torch.load('checkpoints/resnet18_best.pth', map_location='cpu', weights_only=False))
feats_base = get_feats(base, base.avgpool)
print(f'Baseline: {feats_base.shape}')

base2 = models.resnet18(weights=None)
base2.fc = nn.Identity()
topo = TopoModel(base2, 512)
topo.load_state_dict(torch.load('checkpoints/resnet18_topo_guided.pth', map_location='cpu', weights_only=False))
feats_topo = get_feats(topo, topo.backbone.avgpool)
print(f'Topo-guided: {feats_topo.shape}')

def draw_panel(ax, coords, labels, title):
    # Convex hulls per family
    family_points = {}
    for ci, c in enumerate(CLASSES):
        fam = KP[c]
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
                ax.fill(pts[hull_pts,0], pts[hull_pts,1], alpha=0.1, color=FAM_COLORS[fam])
                ax.plot(pts[hull_pts,0], pts[hull_pts,1], color=FAM_COLORS[fam], alpha=0.3, linewidth=1)
            except: pass
    for ci, c in enumerate(CLASSES):
        mask = labels == ci
        ax.scatter(coords[mask,0], coords[mask,1],
                   c=FAM_COLORS[KP[c]], alpha=0.7, s=20, edgecolors='white', linewidths=0.3,
                   label=f'{c} ({KP[c]})')
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

# 2x2 figure: [t-SNE baseline, t-SNE topo] / [UMAP baseline, UMAP topo]
print('Computing t-SNE (baseline)...')
tsne_base = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000).fit_transform(feats_base)
print('Computing t-SNE (topo-guided)...')
tsne_topo = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000).fit_transform(feats_topo)
print('Computing UMAP (baseline)...')
umap_base = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(feats_base)
print('Computing UMAP (topo-guided)...')
umap_topo = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(feats_topo)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
draw_panel(axes[0,0], tsne_base, labels, 't-SNE: Baseline')
draw_panel(axes[0,1], tsne_topo, labels, 't-SNE: Topo-Guided')
draw_panel(axes[1,0], umap_base, labels, 'UMAP: Baseline')
draw_panel(axes[1,1], umap_topo, labels, 'UMAP: Topo-Guided')

# Shared legend
handles, lbls = axes[0,0].get_legend_handles_labels()
seen = set(); unique_h, unique_l = [], []
for h, l in zip(handles, lbls):
    fam = l.split('(')[1].rstrip(')')
    if fam not in seen:
        seen.add(fam)
        unique_h.append(h)
        unique_l.append(fam.capitalize())
fig.legend(unique_h, unique_l, loc='lower center', ncol=5, fontsize=10, title='Topological Family')
fig.suptitle('ResNet-18 Embedding Visualization: t-SNE vs UMAP', fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.96])

for d in ['results/figures', 'paper/figures']:
    os.makedirs(d, exist_ok=True)
    fig.savefig(f'{d}/tsne_umap_comparison.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(f'{d}/tsne_umap_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print('[Saved] tsne_umap_comparison.pdf/png')
print('[DONE]')
