#!/usr/bin/env python3
"""
Topology-Guided Training for Physical Knot Classification

Core idea: align the learned feature space with knot-theoretic topology
by adding a regularization term that encourages class centroids in
embedding space to respect topological distances.

Loss = L_CE + lambda * L_topo

where L_topo = MSE(normalized_embedding_distances, topological_distances)
over all class pairs.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
SEED = 42
RESULTS_DIR = 'results'
DATA_DIR = './train'

# =============================================
# 1. Topological Distance Matrix (from knot theory)
# =============================================

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

# =============================================
# 2. Data Loading (same as baseline)
# =============================================

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

class KnotDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df; self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try: img = Image.open(row['path']).convert('RGB')
        except: img = Image.new('RGB', (224,224), (0,0,0))
        if self.transform: img = self.transform(img)
        return img, torch.tensor(row['label'], dtype=torch.long)

def parse_data(data_dir):
    files = glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True)
    c2i = {c:i for i,c in enumerate(CLASSES)}
    rows = []
    for f in files:
        fn = os.path.basename(f); parts = fn.split('_')
        if parts[0] not in c2i: continue
        if 'Loose' in fn or 'VeryLoose' in fn: sp = 'train'
        elif 'Set' in fn: sp = 'test'
        else: continue
        rows.append({'path':f, 'label':c2i[parts[0]], 'split':sp})
    return pd.DataFrame(rows)

def get_transforms(sz=224):
    norm = ([0.485,0.456,0.406],[0.229,0.224,0.225])
    tr = transforms.Compose([
        transforms.Resize((sz,sz)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), transforms.ColorJitter(0.1,0.1),
        transforms.ToTensor(), transforms.Normalize(*norm)])
    te = transforms.Compose([
        transforms.Resize((sz,sz)), transforms.ToTensor(), transforms.Normalize(*norm)])
    return tr, te

# =============================================
# 3. Topology-Guided Model (with embedding extraction)
# =============================================

class TopoGuidedModel(nn.Module):
    """Wrapper that extracts embeddings before the classifier head."""
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
        base.fc = nn.Identity()  # remove original head
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
# 4. Topology-Guided Loss
# =============================================

class TopologyGuidedLoss(nn.Module):
    """
    L = L_CE + lambda_topo * L_topo + lambda_margin * L_margin
    
    L_topo: Align pairwise centroid distances with topological distances.
            MSE(normalized_centroid_dist, topo_dist) over all class pairs.
    
    L_margin: For each sample, push embeddings away from topologically
              similar but different classes (hard negatives from topology).
    """
    def __init__(self, topo_dist_matrix, lambda_topo=0.1, lambda_margin=0.05,
                 margin=1.2, device='cpu'):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_topo = lambda_topo
        self.lambda_margin = lambda_margin
        self.margin = margin
        # Normalize topo distances to [0, 1]
        D = torch.tensor(topo_dist_matrix, dtype=torch.float32)
        D = D / D.max()
        self.topo_dist = D.to(device)
        self.num_classes = D.shape[0]

    def forward(self, logits, labels, embeddings):
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
        loss_margin = torch.tensor(0.0, device=embeddings.device)

        if len(present_classes) >= 2:
            centroids = torch.stack(centroids)
            # Pairwise centroid distances (MPS-compatible, no cdist)
            diff = centroids.unsqueeze(0) - centroids.unsqueeze(1)  # (K,K,D)
            pdist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)    # (K,K)
            # Normalize to [0, 1]
            if pdist.max() > 0:
                cdist_norm = pdist / pdist.max()
            else:
                cdist_norm = pdist

            # Target: topological distances for present classes
            idx = torch.tensor(present_classes, device=embeddings.device)
            topo_sub = self.topo_dist[idx][:, idx]

            # L_topo: MSE between normalized centroid distances and topo distances
            mask_upper = torch.triu(torch.ones_like(cdist_norm, dtype=torch.bool), diagonal=1)
            loss_topo = F.mse_loss(cdist_norm[mask_upper], topo_sub[mask_upper])

        # L_margin: topology-aware triplet-like loss (on L2-normalized embeddings)
        if len(present_classes) >= 2 and self.lambda_margin > 0:
            margin_losses = []
            emb_norm = F.normalize(embeddings, p=2, dim=1)
            cent_norm = F.normalize(centroids, p=2, dim=1)
            for i, c in enumerate(present_classes):
                mask = (labels == c)
                if mask.sum() == 0:
                    continue
                emb_c = emb_norm[mask]
                for j, c2 in enumerate(present_classes):
                    if c == c2:
                        continue
                    topo_d = self.topo_dist[c, c2].item()
                    if topo_d < 0.3:  # topologically close = hard negative
                        dist_to_c2 = torch.norm(emb_c - cent_norm[j].unsqueeze(0), dim=1)
                        margin_loss = F.relu(self.margin - dist_to_c2).mean()
                        margin_losses.append(margin_loss)
            if margin_losses:
                loss_margin = torch.stack(margin_losses).mean()

        total = loss_ce + self.lambda_topo * loss_topo + self.lambda_margin * loss_margin
        return total, loss_ce.item(), loss_topo.item(), loss_margin.item()

# =============================================
# 5. Training Loop
# =============================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def train_topo_guided(model, loaders, criterion, device, epochs=20, lr=1e-4):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    hist = {
        'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[],
        'topo_loss':[], 'margin_loss':[], 'ce_loss':[]
    }

    for ep in range(epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_correct = 0.0, 0
            running_ce, running_topo, running_margin = 0.0, 0.0, 0.0
            n_batches = 0

            for x, y in loaders[phase]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(x)
                    embeddings = model.get_embeddings()
                    total_loss, ce_val, topo_val, margin_val = criterion(
                        logits, y, embeddings)

                    if phase == 'train':
                        total_loss.backward()
                        opt.step()

                _, preds = torch.max(logits, 1)
                running_loss += total_loss.item() * x.size(0)
                running_correct += (preds == y).sum().item()
                running_ce += ce_val
                running_topo += topo_val
                running_margin += margin_val
                n_batches += 1

            if phase == 'train':
                sched.step()

            epoch_loss = running_loss / len(loaders[phase].dataset)
            epoch_acc = running_correct / len(loaders[phase].dataset)
            hist[f'{phase}_loss'].append(epoch_loss)
            hist[f'{phase}_acc'].append(epoch_acc)

            if phase == 'train':
                hist['ce_loss'].append(running_ce / n_batches)
                hist['topo_loss'].append(running_topo / n_batches)
                hist['margin_loss'].append(running_margin / n_batches)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())

        print(f'  Ep {ep+1}/{epochs} | '
              f'TrL={hist["train_loss"][-1]:.4f} TrA={hist["train_acc"][-1]:.4f} | '
              f'VaL={hist["val_loss"][-1]:.4f} VaA={hist["val_acc"][-1]:.4f} | '
              f'CE={hist["ce_loss"][-1]:.4f} Topo={hist["topo_loss"][-1]:.4f} '
              f'Margin={hist["margin_loss"][-1]:.4f}',
              flush=True)

    model.load_state_dict(best_wts)
    return model, hist, best_acc

# =============================================
# 6. Evaluation
# =============================================

def evaluate(model, loader, device):
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

# =============================================
# 7. Main: Run Topology-Guided Experiments
# =============================================

def run_topo_experiment(model_name, df, topo_dist, device,
                        lambda_topo=0.1, lambda_margin=0.05, epochs=20):
    print(f'\n{"="*60}', flush=True)
    print(f'  Topology-Guided Training: {model_name.upper()}', flush=True)
    print(f'  lambda_topo={lambda_topo}, lambda_margin={lambda_margin}', flush=True)
    print(f'{"="*60}', flush=True)

    tr_full = df[df['split']=='train']
    te_df = df[df['split']=='test']
    tr_df, va_df = train_test_split(
        tr_full, test_size=0.2, stratify=tr_full['label'], random_state=SEED)
    tr_tf, te_tf = get_transforms()

    loaders = {
        'train': DataLoader(KnotDataset(tr_df, tr_tf), batch_size=32, shuffle=True),
        'val': DataLoader(KnotDataset(va_df, te_tf), batch_size=32)
    }
    test_loader = DataLoader(KnotDataset(te_df, te_tf), batch_size=32)

    model = make_topo_model(model_name, len(CLASSES), device)
    criterion = TopologyGuidedLoss(
        topo_dist, lambda_topo=lambda_topo,
        lambda_margin=lambda_margin, device=device)

    t0 = time.time()
    model, hist, best_val = train_topo_guided(
        model, loaders, criterion, device, epochs)
    train_time = time.time() - t0

    preds, labels, embeddings = evaluate(model, test_loader, device)
    report = classification_report(
        labels, preds, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds)
    test_acc = (preds == labels).mean()

    print(f'\n  Test Acc: {test_acc:.4f} | Best Val: {best_val:.4f} | '
          f'Time: {train_time:.0f}s', flush=True)
    print(classification_report(
        labels, preds, target_names=CLASSES, zero_division=0), flush=True)

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(),
               f'checkpoints/{model_name}_topo_guided.pth')

    return {
        'model': model_name,
        'method': 'topology_guided',
        'lambda_topo': lambda_topo,
        'lambda_margin': lambda_margin,
        'history': hist,
        'best_val_acc': best_val,
        'test_acc': float(test_acc),
        'train_time': train_time,
        'report': report,
        'confusion_matrix': cm.tolist(),
        'preds': preds.tolist(),
        'labels': labels.tolist()
    }

# =============================================
# 8. Main Entry Point
# =============================================

if __name__ == '__main__':
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print(f'Device: {device}', flush=True)

    df = parse_data(DATA_DIR)
    print(f'Total: {len(df)} '
          f'(train:{len(df[df.split=="train"])}, '
          f'test:{len(df[df.split=="test"])})', flush=True)

    # Build topological distance matrix
    topo_dist = build_topo_distance_matrix()
    print(f'Topological distance matrix: {topo_dist.shape}', flush=True)

    # Models to run
    model_configs = [
        ('resnet18',       0.1,  0.05),
        ('resnet50',       0.1,  0.05),
        ('efficientnet_b0', 0.1, 0.05),
        ('swin_t',         0.1,  0.05),
    ]

    all_results = {}
    for model_name, lt, lm in model_configs:
        set_seed(SEED)
        res = run_topo_experiment(
            model_name, df, topo_dist, device,
            lambda_topo=lt, lambda_margin=lm, epochs=20)
        all_results[model_name] = res

        fname = f'{RESULTS_DIR}/{model_name}_topo_guided_results.json'
        with open(fname, 'w') as f:
            json.dump(res, f, indent=2, cls=NumpyEncoder)
        print(f'[Saved] {fname}', flush=True)

    # Summary comparison
    print(f'\n{"="*60}', flush=True)
    print('  SUMMARY: Topology-Guided vs Baseline', flush=True)
    print(f'{"="*60}', flush=True)

    for model_name, _, _ in model_configs:
        topo_acc = all_results[model_name]['test_acc']
        # Load baseline for comparison
        baseline_file = f'{RESULTS_DIR}/{model_name}_results.json'
        if os.path.exists(baseline_file):
            base = json.load(open(baseline_file))
            base_acc = base['test_acc']
            delta = (topo_acc - base_acc) * 100
            print(f'  {model_name:20s}: '
                  f'baseline={base_acc:.4f} '
                  f'topo={topo_acc:.4f} '
                  f'delta={delta:+.2f}%', flush=True)
        else:
            print(f'  {model_name:20s}: '
                  f'topo={topo_acc:.4f} (no baseline)', flush=True)

    print('\n[DONE] All topology-guided experiments complete.', flush=True)
