#!/usr/bin/env python3
"""
Embedding Alignment Analysis: Do topo-guided models learn topology-aware representations?

Extracts penultimate-layer embeddings, computes class centroid distances,
and compares with topological distance matrix via Mantel test + visualization.
"""
import os, json, numpy as np, torch, torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, pandas as pd

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
DATA_DIR = './train'
RESULTS_DIR = 'results'
CHECKPOINT_DIR = 'checkpoints'
SEED = 42

# --- Topo distance matrix (same as topo_guided_training.py) ---
KNOT_PROPERTIES = {
    'OHK': {'crossing_num':3, 'type':'prime', 'family':'stopper', 'components':1},
    'F8K': {'crossing_num':4, 'type':'prime', 'family':'stopper', 'components':1},
    'BK':  {'crossing_num':4, 'type':'loop',  'family':'loop',    'components':1},
    'RK':  {'crossing_num':6, 'type':'composite','family':'binding','components':2},
    'FSK': {'crossing_num':6, 'type':'composite','family':'bend',   'components':2},
    'FMB': {'crossing_num':8, 'type':'composite','family':'bend',   'components':2},
    'F8L': {'crossing_num':4, 'type':'loop',  'family':'loop',    'components':1},
    'CH':  {'crossing_num':2, 'type':'hitch', 'family':'hitch',   'components':1},
    'SK':  {'crossing_num':3, 'type':'slip',  'family':'stopper', 'components':1},
    'ABK': {'crossing_num':4, 'type':'loop',  'family':'loop',    'components':1},
}
DERIVATION_PAIRS = {
    ('OHK','SK'): 0.1, ('F8K','FMB'): 0.15,
    ('RK','FSK'): 0.1, ('F8K','F8L'): 0.1,
}

def topological_distance(k1, k2):
    p1, p2 = KNOT_PROPERTIES[k1], KNOT_PROPERTIES[k2]
    d_cross = abs(p1['crossing_num'] - p2['crossing_num']) / 8.0
    d_family = 0.0 if p1['family'] == p2['family'] else 1.0
    d_type = 0.0 if p1['type'] == p2['type'] else 0.5
    d_comp = abs(p1['components'] - p2['components'])
    pair = tuple(sorted([k1, k2]))
    d_deriv = DERIVATION_PAIRS.get(pair, 0.5)
    return 0.25*d_cross + 0.25*d_family + 0.15*d_type + 0.10*d_comp + 0.25*d_deriv

def build_topo_distance_matrix():
    n = len(CLASSES)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i,j] = topological_distance(CLASSES[i], CLASSES[j])
    return D

# --- Dataset ---
class KnotDataset(Dataset):
    def __init__(self, df, transform):
        self.df, self.transform = df, transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r['path']).convert('RGB')
        return self.transform(img), r['label']

def load_data():
    rows = []
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    for p in sorted(glob.glob(f'{DATA_DIR}/*.jpg')):
        fname = os.path.basename(p)
        cls_name = fname.split('_')[0]
        if cls_name in class_to_idx:
            rows.append({'path': p, 'label': class_to_idx[cls_name], 'class': cls_name})
    return pd.DataFrame(rows)

# --- Embedding extraction ---
class TopoGuidedModel(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, num_classes))
    def forward(self, x):
        emb = self.backbone(x)
        return self.classifier(emb)

def get_model_and_hook(name, ckpt_path):
    """Load model, return model for embedding extraction."""
    is_topo = 'topo' in ckpt_path
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if 'resnet18' in name:
        base = models.resnet18(weights=None)
        embed_dim = 512
        if is_topo:
            base.fc = nn.Identity()
            m = TopoGuidedModel(base, embed_dim)
        else:
            base.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, 10))
            m = base
    elif 'resnet50' in name:
        base = models.resnet50(weights=None)
        embed_dim = 2048
        if is_topo:
            base.fc = nn.Identity()
            m = TopoGuidedModel(base, embed_dim)
        else:
            base.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, 10))
            m = base
    else:
        raise ValueError(f"Unknown model: {name}")

    m.load_state_dict(state)
    m.eval()
    return m

def extract_embeddings(model, dataloader, device, model_name):
    """Extract penultimate layer embeddings via forward hook."""
    embeddings, labels = [], []
    features = {}

    def hook_fn(module, input, output):
        features['feat'] = output.detach()

    # Register hook on the layer before classifier
    if hasattr(model, 'backbone'):
        handle = model.backbone.avgpool.register_forward_hook(hook_fn)
    elif 'resnet' in model_name:
        handle = model.avgpool.register_forward_hook(hook_fn)
    elif 'swin' in model_name:
        handle = model.norm.register_forward_hook(hook_fn)

    model = model.to(device)
    with torch.no_grad():
        for imgs, labs in dataloader:
            imgs = imgs.to(device)
            _ = model(imgs)
            feat = features['feat'].squeeze().cpu().numpy()
            if feat.ndim == 1:
                feat = feat[np.newaxis, :]
            embeddings.append(feat)
            labels.extend(labs.numpy())

    handle.remove()
    return np.vstack(embeddings), np.array(labels)

# --- Mantel test ---
def mantel_test(D1, D2, perms=999):
    """Mantel test: correlation between two distance matrices."""
    v1 = squareform(D1, checks=False)
    v2 = squareform(D2, checks=False)
    r_obs, _ = spearmanr(v1, v2)
    n = D1.shape[0]
    count = 0
    for _ in range(perms):
        perm = np.random.permutation(n)
        D2_perm = D2[np.ix_(perm, perm)]
        v2_perm = squareform(D2_perm, checks=False)
        r_perm, _ = spearmanr(v1, v2_perm)
        if r_perm >= r_obs:
            count += 1
    p = (count + 1) / (perms + 1)
    return r_obs, p

def compute_centroid_distances(embeddings, labels):
    """Compute pairwise centroid distance matrix."""
    n_classes = len(CLASSES)
    centroids = np.zeros((n_classes, embeddings.shape[1]))
    for c in range(n_classes):
        mask = labels == c
        centroids[c] = embeddings[mask].mean(axis=0)
    D = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            D[i,j] = np.linalg.norm(centroids[i] - centroids[j])
    return D, centroids

# --- Visualization ---
def plot_tsne(embeddings, labels, title, save_path, topo_D):
    """t-SNE with colors reflecting topological families."""
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    coords = tsne.fit_transform(embeddings)

    family_colors = {
        'loop': '#1f77b4', 'stopper': '#ff7f0e', 'hitch': '#2ca02c',
        'binding': '#d62728', 'bend': '#9467bd'
    }
    class_families = [KNOT_PROPERTIES[c]['family'] for c in CLASSES]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for ci, c in enumerate(CLASSES):
        mask = labels == ci
        color = family_colors[class_families[ci]]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=f'{c} ({class_families[ci]})', alpha=0.6, s=20)
    ax.legend(fontsize=8, loc='best')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'[Saved] {save_path}')

def plot_distance_comparison(topo_D, emb_D_baseline, emb_D_topo, save_path):
    """Side-by-side: topo dist vs embedding dist (baseline vs topo-guided)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, D, title in zip(axes,
        [topo_D, emb_D_baseline, emb_D_topo],
        ['Topological Distance', 'Embedding Dist (Baseline)', 'Embedding Dist (Topo-Guided)']):
        im = ax.imshow(D / D.max(), cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, fontsize=8)
        ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'[Saved] {save_path}')

# --- Main ---
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    os.makedirs(f'{RESULTS_DIR}/figures', exist_ok=True)
    topo_D = build_topo_distance_matrix()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    df = load_data()
    dataset = KnotDataset(df, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f'Loaded {len(df)} images')

    # Models to compare
    pairs = [
        ('resnet18_baseline', f'{CHECKPOINT_DIR}/resnet18_best.pth', 'resnet18'),
        ('resnet18_topo',     f'{CHECKPOINT_DIR}/resnet18_topo_guided.pth', 'resnet18'),
        ('resnet50_baseline', f'{CHECKPOINT_DIR}/resnet50_best.pth', 'resnet50'),
        ('resnet50_topo',     f'{CHECKPOINT_DIR}/resnet50_topo_guided.pth', 'resnet50'),
    ]

    results = {}
    all_emb_D = {}

    for name, ckpt, arch in pairs:
        print(f'\n=== {name} ===')
        model = get_model_and_hook(arch, ckpt)
        embeddings, labels = extract_embeddings(model, loader, device, arch)
        print(f'  Embeddings: {embeddings.shape}')

        emb_D, centroids = compute_centroid_distances(embeddings, labels)
        all_emb_D[name] = emb_D

        # Mantel test
        r, p = mantel_test(topo_D, emb_D, perms=999)
        print(f'  Mantel test: rho={r:.4f}, p={p:.4f}')

        # Also simple Spearman on upper triangle
        idx = np.triu_indices(10, k=1)
        rho_s, p_s = spearmanr(topo_D[idx], emb_D[idx])
        rho_p, p_p = pearsonr(topo_D[idx], emb_D[idx])
        print(f'  Spearman: rho={rho_s:.4f}, p={p_s:.4f}')
        print(f'  Pearson:  r={rho_p:.4f}, p={p_p:.4f}')

        results[name] = {
            'mantel_rho': float(r), 'mantel_p': float(p),
            'spearman_rho': float(rho_s), 'spearman_p': float(p_s),
            'pearson_r': float(rho_p), 'pearson_p': float(p_p),
        }

        # t-SNE
        plot_tsne(embeddings, labels, f't-SNE: {name}',
                  f'{RESULTS_DIR}/figures/tsne_{name}.png', topo_D)

    # Distance matrix comparison
    plot_distance_comparison(topo_D,
        all_emb_D['resnet18_baseline'], all_emb_D['resnet18_topo'],
        f'{RESULTS_DIR}/figures/distance_comparison_resnet18.png')
    plot_distance_comparison(topo_D,
        all_emb_D['resnet50_baseline'], all_emb_D['resnet50_topo'],
        f'{RESULTS_DIR}/figures/distance_comparison_resnet50.png')

    # Save results
    with open(f'{RESULTS_DIR}/embedding_alignment.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n[Saved] {RESULTS_DIR}/embedding_alignment.json')

    # Summary
    print('\n===== SUMMARY =====')
    for name, r in results.items():
        print(f'{name}: Mantel rho={r["mantel_rho"]:.4f} (p={r["mantel_p"]:.4f}), '
              f'Spearman={r["spearman_rho"]:.4f}')

if __name__ == '__main__':
    main()
