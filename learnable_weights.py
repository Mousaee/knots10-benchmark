#!/usr/bin/env python3
"""
Learnable Topology Distance Weights Experiment

Core idea: Replace the fixed w1-w5 weights with learnable parameters,
end-to-end optimized together with the classification model.

Loss = L_CE + lambda_topo * L_topo (where L_topo uses learned weights)
"""
import os, sys, json, time, copy, random, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['ABK', 'BK', 'CH', 'F8K', 'F8L', 'FSK', 'FMB', 'OHK', 'RK', 'SK']
SEED = 42
RESULTS_DIR = 'results'
DATA_DIR = './train'

# =============================================
# 1. Topological Properties & Distance Factors
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
    ('OHK', 'SK'): 0.1, ('F8K', 'FMB'): 0.15,
    ('RK', 'FSK'): 0.1, ('F8K', 'F8L'): 0.1,
}


def compute_distance_factors():
    """
    Precompute 5 distance factor matrices (10x10 each).
    Returns: dict with keys 'crossing', 'family', 'type', 'comp', 'deriv'
    """
    n = len(CLASSES)
    factors = {
        'crossing': np.zeros((n, n)),
        'family': np.zeros((n, n)),
        'type': np.zeros((n, n)),
        'comp': np.zeros((n, n)),
        'deriv': np.zeros((n, n)),
    }

    for i in range(n):
        for j in range(n):
            if i != j:
                k1, k2 = CLASSES[i], CLASSES[j]
                p1, p2 = KNOT_PROPERTIES[k1], KNOT_PROPERTIES[k2]

                # Factor 1: crossing number distance
                factors['crossing'][i, j] = abs(p1['crossing_num'] - p2['crossing_num']) / 8.0

                # Factor 2: family distance
                factors['family'][i, j] = 0.0 if p1['family'] == p2['family'] else 1.0

                # Factor 3: type distance
                factors['type'][i, j] = 0.0 if p1['type'] == p2['type'] else 0.5

                # Factor 4: components distance
                factors['comp'][i, j] = abs(p1['components'] - p2['components'])

                # Factor 5: derivation relationship
                pair = tuple(sorted([k1, k2]))
                factors['deriv'][i, j] = DERIVATION_PAIRS.get(pair, 0.5)

    return factors


# =============================================
# 2. Data Loading (same as baseline)
# =============================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class KnotDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['path']).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(row['label'], dtype=torch.long)


def parse_data(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True))
    c2i = {c: i for i, c in enumerate(CLASSES)}
    rows = []
    for f in files:
        fn = os.path.basename(f)
        parts = fn.split('_')
        if parts[0] not in c2i:
            continue
        if 'Loose' in fn or 'VeryLoose' in fn:
            sp = 'train'
        elif 'Set' in fn:
            sp = 'test'
        else:
            continue
        rows.append({'path': f, 'label': c2i[parts[0]], 'split': sp})
    return pd.DataFrame(rows)


def get_transforms(sz=224):
    norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tr = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
    te = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
    return tr, te


# =============================================
# 3. Learnable Topology Distance Module
# =============================================

class LearnableTopologyDistance(nn.Module):
    """
    Learn the 5 weights for topological distance factors.

    Distance = w1*d_crossing + w2*d_family + w3*d_type + w4*d_comp + w5*d_deriv

    Weights are softmax(logits) to ensure they're normalized and positive.
    """

    def __init__(self, distance_factors, device='cpu'):
        """
        Args:
            distance_factors: dict with keys 'crossing', 'family', 'type', 'comp', 'deriv'
                            each is a (10, 10) numpy array
            device: torch device
        """
        super().__init__()

        # Weight logits (trainable parameters)
        self.weight_logits = nn.Parameter(torch.zeros(5))

        # Distance factor matrices (buffers, not trainable)
        factor_matrices = torch.tensor([
            distance_factors['crossing'],
            distance_factors['family'],
            distance_factors['type'],
            distance_factors['comp'],
            distance_factors['deriv'],
        ], dtype=torch.float32)
        self.register_buffer('factor_matrices', factor_matrices)
        self.device_type = device

    def get_weights(self):
        """Return softmax(logits) as normalized weights."""
        return F.softmax(self.weight_logits, dim=0)

    def get_weights_numpy(self):
        """Return weights as numpy array."""
        return self.get_weights().detach().cpu().numpy()

    def get_distance_matrix(self):
        """
        Compute the full distance matrix by weighted combination of factors.
        Returns: (10, 10) tensor, normalized to [0, 1]
        """
        weights = self.get_weights()  # (5,)

        # Weighted sum: (5, 10, 10) @ (5,) -> (10, 10)
        D = torch.einsum('fij,f->ij', self.factor_matrices, weights)

        # Normalize to [0, 1]
        D_max = D.max()
        if D_max > 0:
            D = D / D_max

        return D


# =============================================
# 4. Learnable Topology-Guided Loss
# =============================================

class LearnableTopologyGuidedLoss(nn.Module):
    """
    Modified TopologyGuidedLoss with learnable distance weights.

    L = L_CE + lambda_topo * L_topo

    where L_topo uses distance matrix from LearnableTopologyDistance.
    """

    def __init__(self, distance_factors, lambda_topo=0.1, device='cpu'):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_topo = lambda_topo
        self.num_classes = len(CLASSES)

        # Learnable distance module
        self.distance = LearnableTopologyDistance(distance_factors, device=device)
        self.device_type = device

    def forward(self, logits, labels, embeddings):
        """
        Args:
            logits: (B, num_classes)
            labels: (B,)
            embeddings: (B, embed_dim)

        Returns:
            total_loss, ce_loss, topo_loss, weights_dict
        """
        # Standard cross-entropy
        loss_ce = self.ce(logits, labels)

        # Compute class centroids from current batch
        centroids = []
        present_classes = []
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                centroids.append(embeddings[mask].mean(dim=0))
                present_classes.append(c)

        loss_topo = torch.tensor(0.0, device=embeddings.device)

        if len(present_classes) >= 2:
            centroids = torch.stack(centroids)

            # Pairwise centroid distances
            diff = centroids.unsqueeze(0) - centroids.unsqueeze(1)  # (K, K, D)
            pdist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)  # (K, K)

            # Normalize to [0, 1]
            if pdist.max() > 0:
                cdist_norm = pdist / pdist.max()
            else:
                cdist_norm = pdist

            # Get learned distance matrix
            topo_dist_full = self.distance.get_distance_matrix()

            # Extract submatrix for present classes
            idx = torch.tensor(present_classes, device=embeddings.device)
            topo_sub = topo_dist_full[idx][:, idx]

            # MSE loss between normalized centroid distances and topo distances
            mask_upper = torch.triu(
                torch.ones_like(cdist_norm, dtype=torch.bool), diagonal=1
            )
            loss_topo = F.mse_loss(cdist_norm[mask_upper], topo_sub[mask_upper])

        total = loss_ce + self.lambda_topo * loss_topo

        # Get current weights
        weights = self.distance.get_weights_numpy()
        weights_dict = {
            'w1': float(weights[0]),  # crossing
            'w2': float(weights[1]),  # family
            'w3': float(weights[2]),  # type
            'w4': float(weights[3]),  # comp
            'w5': float(weights[4]),  # deriv
        }

        return total, loss_ce.item(), loss_topo.item(), weights_dict


# =============================================
# 5. Model Creation
# =============================================

class TopoGuidedModel(nn.Module):
    """Wrapper that extracts embeddings before classifier."""

    def __init__(self, backbone, embed_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim, num_classes)
        )
        self._embeddings = None

    def forward(self, x):
        emb = self.backbone(x)
        self._embeddings = emb
        return self.classifier(emb)

    def get_embeddings(self):
        return self._embeddings


def make_topo_model(name, num_classes, device):
    """Create model with exposed embedding layer."""
    if name == 'resnet18':
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        embed_dim = base.fc.in_features
        base.fc = nn.Identity()
    elif name == 'resnet50':
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        embed_dim = base.fc.in_features
        base.fc = nn.Identity()
    elif name == 'efficientnet_b0':
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        embed_dim = base.classifier[1].in_features
        base.classifier = nn.Identity()
    elif name == 'swin_t':
        base = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        embed_dim = base.head.in_features
        base.head = nn.Identity()
    else:
        raise ValueError(f"Unknown model: {name}")

    model = TopoGuidedModel(base, embed_dim, num_classes)
    return model.to(device)


# =============================================
# 6. Training Loop with Learnable Weights
# =============================================

def train_learnable_topo(model, criterion, loaders, device, epochs=20,
                         base_lr=1e-4, weight_lr=1e-3):
    """
    Train with dual learning rates:
    - Backbone parameters: base_lr
    - Weight logits: weight_lr
    """
    # Dual learning rate optimization
    backbone_params = []
    weight_params = []

    for name, param in model.named_parameters():
        backbone_params.append(param)

    for name, param in criterion.distance.named_parameters():
        weight_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': base_lr},
        {'params': weight_params, 'lr': weight_lr},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_wts = copy.deepcopy(model.state_dict())
    best_criterion_wts = copy.deepcopy(criterion.state_dict())
    best_acc = 0.0

    hist = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'ce_loss': [], 'topo_loss': []
    }
    weight_trajectory = []  # (epochs, 5)

    for ep in range(epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            criterion.train() if phase == 'train' else criterion.eval()

            running_loss, running_correct = 0.0, 0
            running_ce, running_topo = 0.0, 0.0
            n_batches = 0
            epoch_weights = None

            for x, y in loaders[phase]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(x)
                    embeddings = model.get_embeddings()
                    total_loss, ce_val, topo_val, weights_dict = criterion(
                        logits, y, embeddings
                    )

                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                    # Record weights from first batch
                    if epoch_weights is None:
                        epoch_weights = [weights_dict[f'w{i+1}'] for i in range(5)]

                _, preds = torch.max(logits, 1)
                running_loss += total_loss.item() * x.size(0)
                running_correct += (preds == y).sum().item()
                running_ce += ce_val
                running_topo += topo_val
                n_batches += 1

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(loaders[phase].dataset)
            epoch_acc = running_correct / len(loaders[phase].dataset)

            hist[f'{phase}_loss'].append(epoch_loss)
            hist[f'{phase}_acc'].append(epoch_acc)

            if phase == 'train':
                hist['ce_loss'].append(running_ce / n_batches)
                hist['topo_loss'].append(running_topo / n_batches)
                if epoch_weights is not None:
                    weight_trajectory.append(epoch_weights)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
                best_criterion_wts = copy.deepcopy(criterion.state_dict())

        print(f'  Ep {ep+1}/{epochs} | '
              f'TrL={hist["train_loss"][-1]:.4f} TrA={hist["train_acc"][-1]:.4f} | '
              f'VaL={hist["val_loss"][-1]:.4f} VaA={hist["val_acc"][-1]:.4f} | '
              f'CE={hist["ce_loss"][-1]:.4f} Topo={hist["topo_loss"][-1]:.4f} | '
              f'Weights={weight_trajectory[-1] if weight_trajectory else "N/A"}',
              flush=True)

    model.load_state_dict(best_wts)
    criterion.load_state_dict(best_criterion_wts)

    return model, criterion, hist, best_acc, np.array(weight_trajectory)


# =============================================
# 7. Evaluation
# =============================================

def evaluate(model, loader, device):
    """Evaluate model and return predictions, labels, embeddings."""
    model.eval()
    preds, labels, all_embs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            emb = model.get_embeddings()
            preds.extend(torch.max(logits, 1)[1].cpu().numpy())
            labels.extend(y.numpy())
            all_embs.append(emb.cpu().numpy())
    return np.array(preds), np.array(labels), np.concatenate(all_embs)


def compute_embedding_spearman(embeddings_train, embeddings_test, labels_test):
    """
    Compute Spearman correlation between embedding distances and labels.

    For test set, compute pairwise distances and correlate with label similarity.
    """
    n_test = len(labels_test)
    if n_test < 10:
        return np.nan

    # Pairwise distances in embedding space
    diff = embeddings_test[:, np.newaxis, :] - embeddings_test[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))

    # Label similarity (0 = same, 1 = different)
    label_sim = (labels_test[:, np.newaxis] != labels_test[np.newaxis, :]).astype(float)

    # Flatten upper triangle
    idx_upper = np.triu_indices(n_test, k=1)
    dists_upper = dists[idx_upper]
    label_sim_upper = label_sim[idx_upper]

    rho, _ = spearmanr(dists_upper, label_sim_upper)
    return float(rho) if not np.isnan(rho) else 0.0


# =============================================
# 8. Main Experiment
# =============================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_learnable_experiment(model_name, df, distance_factors, device, epochs=20):
    """Run learnable weights experiment for one model."""
    print(f'\n{"="*70}', flush=True)
    print(f'  Learnable Topology Distance: {model_name.upper()}', flush=True)
    print(f'{"="*70}', flush=True)

    tr_full = df[df['split'] == 'train']
    te_df = df[df['split'] == 'test']
    tr_df, va_df = train_test_split(
        tr_full, test_size=0.2, stratify=tr_full['label'], random_state=SEED
    )
    tr_tf, te_tf = get_transforms()

    loaders = {
        'train': DataLoader(KnotDataset(tr_df, tr_tf), batch_size=32, shuffle=True),
        'val': DataLoader(KnotDataset(va_df, te_tf), batch_size=32),
    }
    test_loader = DataLoader(KnotDataset(te_df, te_tf), batch_size=32)

    model = make_topo_model(model_name, len(CLASSES), device)
    criterion = LearnableTopologyGuidedLoss(
        distance_factors, lambda_topo=0.1, device=device
    )
    criterion.to(device)

    # Record initial weights
    initial_weights = criterion.distance.get_weights_numpy()

    t0 = time.time()
    model, criterion, hist, best_val, weight_traj = train_learnable_topo(
        model, criterion, loaders, device, epochs=epochs,
        base_lr=1e-4, weight_lr=1e-3
    )
    train_time = time.time() - t0

    # Evaluate on test set
    preds, labels, embeddings = evaluate(model, test_loader, device)
    report = classification_report(
        labels, preds, target_names=CLASSES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(labels, preds)
    test_acc = (preds == labels).mean()

    # Compute embedding quality metric
    _, _, emb_train = evaluate(model, loaders['train'], device)
    spearman_rho = compute_embedding_spearman(emb_train, embeddings, labels)

    # Final weights
    final_weights = criterion.distance.get_weights_numpy()

    print(f'\n  Test Acc: {test_acc:.4f} | Best Val: {best_val:.4f} | '
          f'Time: {train_time:.0f}s', flush=True)
    print(f'  Initial Weights: {initial_weights}', flush=True)
    print(f'  Final Weights:   {final_weights}', flush=True)
    print(f'  Embedding Spearman: {spearman_rho:.4f}', flush=True)
    print(classification_report(
        labels, preds, target_names=CLASSES, zero_division=0
    ), flush=True)

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(),
               f'checkpoints/{model_name}_learnable_weights.pth')
    torch.save(criterion.state_dict(),
               f'checkpoints/{model_name}_learnable_criterion.pth')

    return {
        'model': model_name,
        'method': 'learnable_topology',
        'epochs': epochs,
        'test_acc': float(test_acc),
        'best_val_acc': float(best_val),
        'train_time': train_time,
        'initial_weights': initial_weights.tolist(),
        'learned_weights': final_weights.tolist(),
        'weight_trajectory': weight_traj.tolist(),
        'embedding_spearman_rho': spearman_rho,
        'history': hist,
        'report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': preds.tolist(),
        'labels': labels.tolist(),
    }


def visualize_weight_evolution(results_dict, output_file='results/figures/weight_evolution.pdf'):
    """
    Plot weight evolution across epochs for each model.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fig, axes = plt.subplots(1, len(results_dict), figsize=(15, 5))
    if len(results_dict) == 1:
        axes = [axes]

    weight_names = ['w1 (crossing)', 'w2 (family)', 'w3 (type)', 'w4 (comp)', 'w5 (deriv)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, (model_name, result) in enumerate(results_dict.items()):
        ax = axes[idx]
        traj = np.array(result['weight_trajectory'])  # (epochs, 5)

        for w_idx in range(5):
            ax.plot(traj[:, w_idx], label=weight_names[w_idx],
                   color=colors[w_idx], marker='o', markersize=4)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight Value')
        ax.set_title(f'{model_name.upper()}\n'
                    f'Test Acc: {result["test_acc"]:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 0.5])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'\n[Saved] {output_file}', flush=True)
    paper_fig = 'paper/figures/weight_evolution.pdf'
    os.makedirs(os.path.dirname(paper_fig), exist_ok=True)
    plt.savefig(paper_fig, dpi=300, bbox_inches='tight')
    print(f'[Saved] {paper_fig}', flush=True)
    plt.close()


# =============================================
# 9. Main Entry Point
# =============================================

if __name__ == '__main__':
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print(f'Device: {device}', flush=True)

    # Load data
    df = parse_data(DATA_DIR)
    print(f'Total: {len(df)} '
          f'(train: {len(df[df.split=="train"])}, '
          f'test: {len(df[df.split=="test"])})', flush=True)

    # Precompute distance factors
    distance_factors = compute_distance_factors()
    print(f'Distance factors computed: {list(distance_factors.keys())}', flush=True)

    # Models to run
    model_configs = [
        'resnet18',
        'resnet50',
    ]

    all_results = {}
    for model_name in model_configs:
        set_seed(SEED)
        res = run_learnable_experiment(
            model_name, df, distance_factors, device, epochs=20
        )
        all_results[model_name] = res

        fname = f'{RESULTS_DIR}/learnable_weights_{model_name}.json'
        with open(fname, 'w') as f:
            json.dump(res, f, indent=2, cls=NumpyEncoder)
        print(f'[Saved] {fname}', flush=True)

    # Summary
    print(f'\n{"="*70}', flush=True)
    print('  SUMMARY: Learnable Topology Weights', flush=True)
    print(f'{"="*70}', flush=True)

    for model_name in model_configs:
        result = all_results[model_name]
        print(f'\n{model_name.upper()}:', flush=True)
        print(f'  Test Acc: {result["test_acc"]:.4f}', flush=True)
        print(f'  Initial Weights: {result["initial_weights"]}', flush=True)
        print(f'  Learned Weights: {result["learned_weights"]}', flush=True)
        print(f'  Embedding Spearman: {result["embedding_spearman_rho"]:.4f}', flush=True)

    # Visualization
    visualize_weight_evolution(all_results)

    # Master results file
    master_file = f'{RESULTS_DIR}/learnable_weights_results.json'
    with open(master_file, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f'\n[Saved] {master_file}', flush=True)

    print('\n[DONE] Learnable weights experiment complete.', flush=True)
