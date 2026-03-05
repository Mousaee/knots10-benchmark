import torch, torch.nn as nn, numpy as np, glob, os, json
from torchvision import transforms, models
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
KP = {'OHK':(3,'prime','stopper',1),'F8K':(4,'prime','stopper',1),
      'BK':(4,'loop','loop',1),'RK':(6,'composite','binding',2),
      'FSK':(6,'composite','bend',2),'FMB':(8,'composite','bend',2),
      'F8L':(4,'loop','loop',1),'CH':(2,'hitch','hitch',1),
      'SK':(3,'slip','stopper',1),'ABK':(4,'loop','loop',1)}
DR = {('OHK','SK'):0.1,('F8K','FMB'):0.15,('RK','FSK'):0.1,('F8K','F8L'):0.1}
def td(a,b):
    c1,t1,f1,n1=KP[a];c2,t2,f2,n2=KP[b]
    return 0.25*abs(c1-c2)/8+0.25*(0 if f1==f2 else 1)+0.15*(0 if t1==t2 else .5)+.1*abs(n1-n2)+.25*DR.get(tuple(sorted([a,b])),.5)
tD = np.array([[td(CLASSES[i],CLASSES[j]) if i!=j else 0 for j in range(10)] for i in range(10)])

FAMILIES = {c: KP[c][2] for c in CLASSES}
FAM_COLORS = {'loop':'#1f77b4','stopper':'#ff7f0e','hitch':'#2ca02c','binding':'#d62728','bend':'#9467bd'}

c2i = {c:i for i,c in enumerate(CLASSES)}
paths, labels = [], []
for p in sorted(glob.glob('train_small/*.jpg')):
    cn = os.path.basename(p).split('_')[0]
    if cn in c2i: paths.append(p); labels.append(c2i[cn])
labels = np.array(labels)
print(f'{len(paths)} images', flush=True)

tf = transforms.Compose([transforms.ToTensor(),
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
            imgs = torch.stack([tf(Image.open(p)) for p in paths[start:start+64]])
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

# --- Visualization functions ---
def plot_heatmaps(tD, eD_base, eD_topo, arch, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, D, title in zip(axes,
        [tD, eD_base, eD_topo],
        ['Topological Distance', f'{arch} Baseline\nEmbedding Distance', f'{arch} Topo-Guided\nEmbedding Distance']):
        Dn = D / D.max() if D.max() > 0 else D
        im = ax.imshow(Dn, cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, fontsize=8)
        ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'Distance Matrix Comparison ({arch})', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {save_path}', flush=True)

def plot_tsne(feats_base, feats_topo, labels, arch, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, feats, title in zip(axes,
        [feats_base, feats_topo],
        [f'{arch} Baseline', f'{arch} Topo-Guided']):
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        coords = tsne.fit_transform(feats)
        for ci, c in enumerate(CLASSES):
            mask = labels == ci
            color = FAM_COLORS[FAMILIES[c]]
            ax.scatter(coords[mask,0], coords[mask,1],
                       c=color, label=f'{c} ({FAMILIES[c]})', alpha=0.6, s=15)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, loc='best', ncol=2)
    fig.suptitle(f't-SNE Visualization ({arch})', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {save_path}', flush=True)

# --- Main ---
os.makedirs('results/figures', exist_ok=True)

configs = [
    ('ResNet-18', 'resnet18', 512,
     'checkpoints/resnet18_best.pth', 'checkpoints/resnet18_topo_guided.pth'),
    ('ResNet-50', 'resnet50', 2048,
     'checkpoints/resnet50_best.pth', 'checkpoints/resnet50_topo_guided.pth'),
]

for display, arch, dim, ckpt_base, ckpt_topo in configs:
    print(f'\n=== {display} ===', flush=True)
    feats_base = load_and_extract(arch, ckpt_base, dim, False)
    print(f'  Baseline embeddings: {feats_base.shape}', flush=True)
    feats_topo = load_and_extract(arch, ckpt_topo, dim, True)
    print(f'  Topo embeddings: {feats_topo.shape}', flush=True)

    # Centroid distance matrices
    cent_base = np.array([feats_base[labels==c].mean(0) for c in range(10)])
    cent_topo = np.array([feats_topo[labels==c].mean(0) for c in range(10)])
    eD_base = np.array([[np.linalg.norm(cent_base[i]-cent_base[j]) for j in range(10)] for i in range(10)])
    eD_topo = np.array([[np.linalg.norm(cent_topo[i]-cent_topo[j]) for j in range(10)] for i in range(10)])

    plot_heatmaps(tD, eD_base, eD_topo, display,
                  f'results/figures/distance_comparison_{arch}.png')
    print(f'  Starting t-SNE...', flush=True)
    plot_tsne(feats_base, feats_topo, labels, display,
              f'results/figures/tsne_comparison_{arch}.png')

print('\nAll visualizations done!', flush=True)
