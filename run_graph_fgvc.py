#!/usr/bin/env python3
"""Graph-based Fine-Grained Visual Classification for Knot Recognition"""
import os, sys, json, time, copy, random, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.style.use('seaborn-v0_8-whitegrid')

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
SEED = 42
RESULTS_DIR = 'results'
DATA_DIR = './train'

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True))
    c2i = {c:i for i,c in enumerate(CLASSES)}
    rows = []
    for f in files:
        fn = os.path.basename(f); parts = fn.split('_')
        if parts[0] not in c2i: continue
        if 'Loose' in fn or 'VeryLoose' in fn: sp = 'train'
        elif 'Set' in fn: sp = 'test'
        else: continue
        rows.append({'path':f, 'label':c2i[parts[0]], 'split':sp,
                     'light': 'DL' if '_DL_' in fn else ('SLA' if '_SLA_' in fn else 'SLS'),
                     'tightness': 'VeryLoose' if 'VeryLoose' in fn else ('Loose' if 'Loose' in fn else 'Set')})
    return pd.DataFrame(rows)

def get_transforms(sz=224):
    norm = ([0.485,0.456,0.406],[0.229,0.224,0.225])
    tr = transforms.Compose([transforms.Resize((sz,sz)), transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(15), transforms.ColorJitter(0.1,0.1),
         transforms.ToTensor(), transforms.Normalize(*norm)])
    te = transforms.Compose([transforms.Resize((sz,sz)), transforms.ToTensor(), transforms.Normalize(*norm)])
    return tr, te

# ============================================================================
# Graph-Based Fine-Grained Visual Classification Components
# ============================================================================

class AdjacencyLearner(nn.Module):
    """Learn adjacency matrix based on node feature similarity"""
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, nodes):
        """
        Args:
            nodes: (B, N, C) where N is number of spatial positions, C is feature dim
        Returns:
            adj: (B, N, N) adjacency matrix (row-wise normalized)
        """
        h = self.fc(nodes)  # (B, N, hidden_dim)
        # Compute similarity: cosine-like metric via bmm
        adj = torch.bmm(h, h.transpose(1, 2))  # (B, N, N)
        # Add self-loop boost (identity scaled)
        diag = torch.eye(adj.size(1), device=adj.device).unsqueeze(0) * 0.5
        adj = adj + diag
        # Row-wise softmax normalization for numerical stability
        adj = F.softmax(adj, dim=-1)
        return adj

class GraphConvLayer(nn.Module):
    """Graph Convolution Layer"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.bn = nn.LayerNorm(out_dim)

    def forward(self, nodes, adj):
        """
        Args:
            nodes: (B, N, C) node features
            adj: (B, N, N) adjacency matrix
        Returns:
            out: (B, N, out_dim) updated node features
        """
        agg = torch.bmm(adj, nodes)  # Neighborhood aggregation (B, N, C)
        out = self.W(agg)  # Linear transformation (B, N, out_dim)
        out = self.bn(out)  # LayerNorm works directly on (B, N, out_dim)
        return F.relu(out)

class GraphFGVCModel(nn.Module):
    """Graph-based Fine-Grained Visual Classification Model

    Architecture:
    1. ResNet-50 truncated at layer3 -> (B, 1024, 14, 14) feature maps
    2. 196 spatial positions as graph nodes
    3. Learnable adjacency matrix based on node similarity
    4. 2-layer GCN for spatial relationship learning
    5. Global pooling + classification head
    6. Parallel global branch for ensemble fusion with learnable weights
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Feature extractor: ResNet-50 truncated at layer3
        base = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3
        )  # Output: (B, 1024, 14, 14)

        # Graph components
        self.adj_learner = AdjacencyLearner(1024, 256)
        self.gcn1 = GraphConvLayer(1024, 512)
        self.gcn2 = GraphConvLayer(512, 256)

        # Graph branch classifier
        self.classifier = nn.Linear(256, num_classes)

        # Global branch for ensemble
        self.layer4 = base.layer4
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_classifier = nn.Linear(2048, num_classes)

        # Learnable branch fusion weights
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) input images
        Returns:
            logits: (B, num_classes) classification logits
        """
        feat = self.features(x)  # (B, 1024, 14, 14)
        B, C, H, W = feat.shape

        # ===== Graph Branch =====
        # Reshape spatial feature map to node format
        nodes = feat.reshape(B, C, H*W).permute(0, 2, 1)  # (B, 196, 1024)

        # Learn adjacency matrix from node features
        adj = self.adj_learner(nodes)  # (B, 196, 196)

        # GCN layers for spatial relationship learning
        nodes = self.gcn1(nodes, adj)  # (B, 196, 512)
        nodes = self.gcn2(nodes, adj)  # (B, 196, 256)

        # Global pooling over nodes
        graph_out = nodes.mean(dim=1)  # (B, 256)
        graph_logits = self.classifier(graph_out)

        # ===== Global Branch =====
        global_feat = self.layer4(feat)  # (B, 2048, 7, 7)
        global_feat = self.global_pool(global_feat).flatten(1)  # (B, 2048)
        global_logits = self.global_classifier(global_feat)

        # ===== Ensemble Fusion with Learnable Weights =====
        # Weighted average: graph_branch * weight + global_branch * (1 - weight)
        logits = self.fusion_weight * graph_logits + (1.0 - self.fusion_weight) * global_logits

        return logits

def train_model(model, loaders, device, epochs=20, lr=1e-4):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_wts = copy.deepcopy(model.state_dict()); best_acc = 0
    hist = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    for ep in range(epochs):
        for phase in ['train','val']:
            model.train() if phase=='train' else model.eval()
            rl, rc = 0.0, 0
            for x, y in loaders[phase]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    out = model(x); _, p = torch.max(out,1); loss = crit(out,y)
                    if phase=='train': loss.backward(); opt.step()
                rl += loss.item()*x.size(0); rc += (p==y).sum().item()
            if phase=='train': sched.step()
            el = rl/len(loaders[phase].dataset); ea = rc/len(loaders[phase].dataset)
            hist[f'{phase}_loss'].append(el); hist[f'{phase}_acc'].append(ea)
            if phase=='val' and ea > best_acc:
                best_acc = ea; best_wts = copy.deepcopy(model.state_dict())
        print(f'  Ep {ep+1}/{epochs} | TrL={hist["train_loss"][-1]:.4f} TrA={hist["train_acc"][-1]:.4f} | VaL={hist["val_loss"][-1]:.4f} VaA={hist["val_acc"][-1]:.4f}', flush=True)
    model.load_state_dict(best_wts)
    return model, hist, best_acc

def evaluate(model, loader, device):
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); out = model(x)
            preds.extend(torch.max(out,1)[1].cpu().numpy())
            labels.extend(y.numpy())
    return np.array(preds), np.array(labels)

def run_experiment(df, device, epochs=20):
    print(f'\n{"="*50}\n  Running: Graph-based FGVC\n{"="*50}', flush=True)
    tr_full = df[df['split']=='train']; te_df = df[df['split']=='test']
    tr_df, va_df = train_test_split(tr_full, test_size=0.2, stratify=tr_full['label'], random_state=SEED)
    tr_tf, te_tf = get_transforms()
    loaders = {
        'train': DataLoader(KnotDataset(tr_df, tr_tf), batch_size=32, shuffle=True),
        'val': DataLoader(KnotDataset(va_df, te_tf), batch_size=32)
    }
    test_loader = DataLoader(KnotDataset(te_df, te_tf), batch_size=32)
    model = GraphFGVCModel(num_classes=len(CLASSES)).to(device)
    t0 = time.time()
    model, hist, best_val = train_model(model, loaders, device, epochs)
    train_time = time.time() - t0
    preds, labels = evaluate(model, test_loader, device)
    report = classification_report(labels, preds, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds)
    test_acc = (preds == labels).mean()
    print(f'  Test Acc: {test_acc:.4f} | Best Val: {best_val:.4f} | Time: {train_time:.0f}s', flush=True)
    print(classification_report(labels, preds, target_names=CLASSES, zero_division=0), flush=True)
    return {
        'model': 'graph_fgvc', 'history': hist, 'best_val_acc': best_val,
        'test_acc': float(test_acc), 'train_time': train_time,
        'report': report, 'confusion_matrix': cm.tolist(),
        'preds': preds.tolist(), 'labels': labels.tolist()
    }

if __name__ == '__main__':
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print(f'Device: {device}', flush=True)
    df = parse_data(DATA_DIR)
    print(f'Total: {len(df)} (train:{len(df[df.split=="train"])}, test:{len(df[df.split=="test"])})', flush=True)

    # Graph-based Fine-Grained Visual Classification
    res = run_experiment(df, device, epochs=20)
    with open(f'{RESULTS_DIR}/graph_fgvc_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('[Saved] graph_fgvc_results.json', flush=True)

    print('\n[DONE] Graph FGVC experiment complete.', flush=True)
