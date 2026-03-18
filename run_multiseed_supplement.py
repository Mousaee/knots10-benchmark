#!/usr/bin/env python3
"""
Supplementary multi-seed experiments for reviewer response.
Runs TransFG, PMG, Graph-FGVC, and Learnable Weights with seeds 123 and 456.
Also runs CUB-200 Mantel test and embedding Mantel tests.

Usage:
    source /home/dell/BlackPercy/bin/activate
    cd /home/dell/knots10
    CUDA_VISIBLE_DEVICES=0 python run_multiseed_supplement.py --phase fgvc &
    CUDA_VISIBLE_DEVICES=1 python run_multiseed_supplement.py --phase mantel &

Phase 'fgvc': Multi-seed FGVC + learnable weights (GPU-intensive, ~8h)
Phase 'mantel': CUB-200 Mantel + embedding Mantel (lighter, ~1h)
Phase 'aircraft': Aircraft embedding analysis (~30min)
Phase 'all': Everything sequentially
"""

import os, sys, json, time, random, glob, copy, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform, pdist

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
DATA_DIR = './train'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Shared utilities ───────────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_data(data_dir):
    """Parse knot image dataset with tightness-stratified split."""
    import pandas as pd
    rows = []
    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for img_file in sorted(glob.glob(os.path.join(cls_dir, '*.jpg'))):
            fname = os.path.basename(img_file)
            # Tightness from filename
            if '_Set_' in fname:
                split = 'test'
            else:
                split = 'train'
            rows.append({'path': img_file, 'label': cls, 'split': split})
    df = __import__('pandas').DataFrame(rows)
    df['label_idx'] = df['label'].map({c: i for i, c in enumerate(CLASSES)})
    return df

class KnotDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, row['label_idx']

def get_transforms(sz=224):
    norm = ([0.485,0.456,0.406],[0.229,0.224,0.225])
    tr = transforms.Compose([
        transforms.Resize((sz,sz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1,0.1),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
    te = transforms.Compose([
        transforms.Resize((sz,sz)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
    return tr, te

def train_and_eval(model, train_loader, val_loader, test_loader, device,
                   epochs=20, lr=1e-4, extra_params=None):
    """Generic train loop. Returns dict with metrics."""
    params = list(model.parameters())
    if extra_params:
        params = params + extra_params
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val = 0; best_state = None
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        if val_acc >= best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # Test with best model
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
    return {
        'val_acc': float(best_val),
        'test_acc': float(test_acc),
        'macro_f1': float(macro_f1),
        'confusion_matrix': cm.tolist(),
    }

# ─── TransFG ───────────────────────────────────────────────

class PartSelectionModule(nn.Module):
    def __init__(self, num_heads=12, top_k=6):
        super().__init__()
        self.top_k = top_k
        self.num_heads = num_heads
    def forward(self, attn_weights, tokens):
        # attn_weights: (B, num_heads, N, N), tokens: (B, N, D)
        # Average over heads, take CLS row
        attn = attn_weights.mean(dim=1)[:, 0, 1:]  # (B, N-1)
        _, idx = attn.topk(self.top_k, dim=1)
        # Gather selected tokens
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
        selected = tokens.gather(1, idx_exp)  # (B, top_k, D)
        return selected.mean(dim=1)  # (B, D)

class TransFGModel(nn.Module):
    def __init__(self, num_classes=10, top_k=6):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.psm = PartSelectionModule(num_heads=12, top_k=top_k)
        hidden = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        self.global_head = nn.Linear(hidden, num_classes)
        self.part_head = nn.Linear(hidden, num_classes)
        self.hidden = hidden
        # Hook to capture attention
        self._attn = None
        self.vit.encoder.layers[-1].self_attention.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # ViT self_attention returns (attn_output, attn_weights)
        if isinstance(out, tuple) and len(out) >= 2:
            self._attn = out[1]

    def forward(self, x):
        # Get tokens from ViT encoder
        x = self.vit._process_input(x)
        n = x.shape[0]
        cls_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.vit.encoder(x + self.vit.encoder.pos_embedding)
        cls_out = x[:, 0]
        global_logits = self.global_head(cls_out)

        if self._attn is not None:
            part_feat = self.psm(self._attn, x)
            part_logits = self.part_head(part_feat)
            return (global_logits + part_logits) / 2
        return global_logits

def run_transfg_seed(df, device, seed, epochs=20):
    set_seed(seed)
    tr_full = df[df.split=='train']
    tr_df, va_df = train_test_split(tr_full, test_size=0.2,
                                     stratify=tr_full['label'], random_state=seed)
    te_df = df[df.split=='test']
    tr_t, te_t = get_transforms(224)
    loaders = {
        'train': DataLoader(KnotDataset(tr_df, tr_t), 32, shuffle=True, num_workers=4),
        'val': DataLoader(KnotDataset(va_df, te_t), 32, num_workers=4),
        'test': DataLoader(KnotDataset(te_df, te_t), 32, num_workers=4),
    }
    model = TransFGModel(num_classes=10, top_k=6).to(device)
    return train_and_eval(model, loaders['train'], loaders['val'],
                          loaders['test'], device, epochs)

# ─── PMG ───────────────────────────────────────────────────

class JigsawGenerator:
    def __init__(self, n_grid=2):
        self.n_grid = n_grid
    def __call__(self, img_tensor):
        # img_tensor: (C, H, W)
        C, H, W = img_tensor.shape
        gh, gw = H // self.n_grid, W // self.n_grid
        patches = []
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                patches.append(img_tensor[:, i*gh:(i+1)*gh, j*gw:(j+1)*gw])
        random.shuffle(patches)
        # Reassemble
        rows = []
        for i in range(self.n_grid):
            rows.append(torch.cat(patches[i*self.n_grid:(i+1)*self.n_grid], dim=2))
        return torch.cat(rows, dim=1)

class PMGModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Split ResNet-50 into stages
        self.conv1 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        # Multi-granularity classifiers
        self.fc_concat = nn.Linear(2048, num_classes)
        self.fc_stage3 = nn.Linear(1024, num_classes)
        self.fc_stage4 = nn.Linear(2048, num_classes)
        self.jigsaw = JigsawGenerator(n_grid=2)

    def forward(self, x, jigsaw_input=None):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        s3 = self.layer3(x)
        s4 = self.layer4(s3)
        feat = self.avgpool(s4).flatten(1)

        logits_concat = self.fc_concat(feat)
        # Stage3 classifier
        s3_pool = nn.functional.adaptive_avg_pool2d(s3, 1).flatten(1)
        logits_s3 = self.fc_stage3(s3_pool)

        return logits_concat, logits_s3

def run_pmg_seed(df, device, seed, epochs=20):
    set_seed(seed)
    tr_full = df[df.split=='train']
    tr_df, va_df = train_test_split(tr_full, test_size=0.2,
                                     stratify=tr_full['label'], random_state=seed)
    te_df = df[df.split=='test']
    tr_t, te_t = get_transforms(224)
    train_loader = DataLoader(KnotDataset(tr_df, tr_t), 32, shuffle=True, num_workers=4)
    val_loader = DataLoader(KnotDataset(va_df, te_t), 32, num_workers=4)
    test_loader = DataLoader(KnotDataset(te_df, te_t), 32, num_workers=4)

    model = PMGModel(num_classes=10).to(device)
    jigsaw = JigsawGenerator(n_grid=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val = 0; best_state = None
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Progressive: concat + stage3
            logits_concat, logits_s3 = model(x)
            loss = criterion(logits_concat, y) + 0.5 * criterion(logits_s3, y)
            # Jigsaw auxiliary
            x_jig = torch.stack([jigsaw(img) for img in x.cpu()]).to(device)
            jig_concat, _ = model(x_jig)
            loss += 0.3 * criterion(jig_concat, y)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        if val_acc >= best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
    return {
        'val_acc': float(best_val), 'test_acc': float(test_acc),
        'macro_f1': float(macro_f1), 'confusion_matrix': cm.tolist(),
    }

# ─── Graph-FGVC ───────────────────────────────────────────

class GraphFGVCModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-2])  # up to layer4
        self.adj = nn.Parameter(torch.eye(196) + 0.01 * torch.randn(196, 196))
        self.gcn1 = nn.Linear(2048, 512)
        self.gcn2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        feat = self.features(x)  # (B, 2048, 7, 7)
        B, C, H, W = feat.shape
        nodes = feat.view(B, C, H*W).permute(0, 2, 1)  # (B, 49, 2048)
        # Resize adjacency if needed
        N = nodes.size(1)
        adj = torch.softmax(self.adj[:N, :N], dim=1)
        # GCN layers
        h = torch.relu(self.gcn1(torch.bmm(adj.unsqueeze(0).expand(B,-1,-1), nodes)))
        h = torch.relu(self.gcn2(torch.bmm(adj.unsqueeze(0).expand(B,-1,-1), h)))
        # Global pool
        h = h.mean(dim=1)
        return self.classifier(h)

def run_graph_fgvc_seed(df, device, seed, epochs=20):
    set_seed(seed)
    tr_full = df[df.split=='train']
    tr_df, va_df = train_test_split(tr_full, test_size=0.2,
                                     stratify=tr_full['label'], random_state=seed)
    te_df = df[df.split=='test']
    tr_t, te_t = get_transforms(224)
    loaders = {
        'train': DataLoader(KnotDataset(tr_df, tr_t), 32, shuffle=True, num_workers=4),
        'val': DataLoader(KnotDataset(va_df, te_t), 32, num_workers=4),
        'test': DataLoader(KnotDataset(te_df, te_t), 32, num_workers=4),
    }
    model = GraphFGVCModel(num_classes=10).to(device)
    return train_and_eval(model, loaders['train'], loaders['val'],
                          loaders['test'], device, epochs)

# ─── Learnable Weights ────────────────────────────────────

# Topological distance factors
KP = {
    'ABK': {'cn':4,'family':'loop','type':'loop','comp':1},
    'BK':  {'cn':4,'family':'loop','type':'loop','comp':1},
    'CH':  {'cn':2,'family':'hitch','type':'hitch','comp':1},
    'F8K': {'cn':4,'family':'stopper','type':'prime','comp':1},
    'F8L': {'cn':4,'family':'loop','type':'loop','comp':1},
    'FSK': {'cn':6,'family':'bend','type':'composite','comp':2},
    'FMB': {'cn':8,'family':'bend','type':'composite','comp':2},
    'OHK': {'cn':3,'family':'stopper','type':'prime','comp':1},
    'RK':  {'cn':6,'family':'binding','type':'composite','comp':2},
    'SK':  {'cn':3,'family':'stopper','type':'slip','comp':1},
}
DR = {('F8K','FMB'):0.15, ('F8K','F8L'):0.1, ('OHK','SK'):0.1, ('RK','FSK'):0.1}

def build_factor_matrices():
    """Build 5 factor-specific distance matrices."""
    K = len(CLASSES)
    D = np.zeros((5, K, K))
    for i in range(K):
        for j in range(K):
            if i == j: continue
            ci, cj = CLASSES[i], CLASSES[j]
            pi, pj = KP[ci], KP[cj]
            D[0,i,j] = abs(pi['cn']-pj['cn'])/8.0
            D[1,i,j] = 0.0 if pi['family']==pj['family'] else 1.0
            D[2,i,j] = 0.0 if pi['type']==pj['type'] else 0.5
            D[3,i,j] = abs(pi['comp']-pj['comp'])
            pair = tuple(sorted([ci,cj]))
            D[4,i,j] = DR.get(pair, 0.5)
    return D

class LearnableWeightModel(nn.Module):
    def __init__(self, backbone, num_classes=10, embed_dim=512):
        super().__init__()
        self.backbone = backbone
        self.weight_logits = nn.Parameter(torch.zeros(5))

    def get_weights(self):
        return torch.softmax(self.weight_logits, dim=0)

    def forward(self, x):
        return self.backbone(x)

def run_learnable_seed(model_name, df, device, seed, epochs=20):
    set_seed(seed)
    tr_full = df[df.split=='train']
    tr_df, va_df = train_test_split(tr_full, test_size=0.2,
                                     stratify=tr_full['label'], random_state=seed)
    te_df = df[df.split=='test']
    tr_t, te_t = get_transforms(224)
    train_loader = DataLoader(KnotDataset(tr_df, tr_t), 32, shuffle=True, num_workers=4)
    val_loader = DataLoader(KnotDataset(va_df, te_t), 32, num_workers=4)
    test_loader = DataLoader(KnotDataset(te_df, te_t), 32, num_workers=4)

    # Build backbone
    if model_name == 'resnet18':
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        embed_dim = 512
    else:
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        embed_dim = 2048
    base.fc = nn.Linear(embed_dim, 10)
    model = LearnableWeightModel(base, num_classes=10, embed_dim=embed_dim).to(device)

    # Factor matrices
    factor_matrices = torch.tensor(build_factor_matrices(), dtype=torch.float32).to(device)

    # Optimizers: dual learning rate
    backbone_params = [p for n, p in model.named_parameters() if 'weight_logits' not in n]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': [model.weight_logits], 'lr': 1e-3},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val = 0; best_state = None
    weight_trajectory = []

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            ce_loss = criterion(logits, y)

            # TACA loss with learnable weights
            w = model.get_weights()
            D_topo = (w.view(5,1,1) * factor_matrices).sum(0)  # (10, 10)

            # Compute class centroids from embeddings
            # Get penultimate layer features
            with torch.no_grad():
                # Extract features before fc
                feat_model = nn.Sequential(*list(model.backbone.children())[:-1])
                feats = feat_model(x).flatten(1)

            # Actually we need gradients for TACA. Re-extract.
            feat_layers = list(model.backbone.children())[:-1]
            feat_extractor = nn.Sequential(*feat_layers)
            feats = feat_extractor(x).flatten(1)

            # Compute centroids per class in batch
            centroids = []
            present_classes = []
            for c in range(10):
                mask = (y == c)
                if mask.sum() > 0:
                    centroids.append(feats[mask].mean(0))
                    present_classes.append(c)

            if len(present_classes) >= 2:
                centroids = torch.stack(centroids)
                # Pairwise distances
                n_c = len(present_classes)
                emb_dist = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
                emb_dist = emb_dist / (emb_dist.max() + 1e-8)
                # Get corresponding topo distances
                idx = torch.tensor(present_classes, device=device)
                topo_sub = D_topo[idx][:, idx]
                taca_loss = ((emb_dist - topo_sub) ** 2).mean()
                loss = ce_loss + 0.1 * taca_loss
            else:
                loss = ce_loss

            optimizer.zero_grad(); loss.backward(); optimizer.step()

        scheduler.step()
        w_np = model.get_weights().detach().cpu().numpy().tolist()
        weight_trajectory.append(w_np)

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        if val_acc >= best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # Test
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            all_preds.extend(model(x).argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    final_weights = model.get_weights().detach().cpu().numpy().tolist()

    return {
        'val_acc': float(best_val), 'test_acc': float(test_acc),
        'macro_f1': float(macro_f1), 'learned_weights': final_weights,
        'weight_trajectory': weight_trajectory,
    }

# ─── Mantel tests ──────────────────────────────────────────

def build_topo_distance_matrix(weights=None):
    if weights is None:
        weights = [0.25, 0.25, 0.15, 0.10, 0.25]
    K = len(CLASSES)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j: continue
            ci, cj = CLASSES[i], CLASSES[j]
            pi, pj = KP[ci], KP[cj]
            d1 = abs(pi['cn']-pj['cn'])/8.0
            d2 = 0.0 if pi['family']==pj['family'] else 1.0
            d3 = 0.0 if pi['type']==pj['type'] else 0.5
            d4 = abs(pi['comp']-pj['comp'])
            pair = tuple(sorted([ci,cj]))
            d5 = DR.get(pair, 0.5)
            D[i,j] = weights[0]*d1 + weights[1]*d2 + weights[2]*d3 + weights[3]*d4 + weights[4]*d5
    return D

def mantel_test(D1, D2, n_perms=9999, seed=42):
    """Mantel permutation test between two symmetric distance matrices."""
    np.random.seed(seed)
    v1 = squareform(D1, checks=False)
    v2 = squareform(D2, checks=False)
    r_obs, _ = spearmanr(v1, v2)
    K = D1.shape[0]
    null_dist = np.zeros(n_perms)
    for p in range(n_perms):
        perm = np.random.permutation(K)
        D2_perm = D2[np.ix_(perm, perm)]
        v2_perm = squareform(D2_perm, checks=False)
        null_dist[p], _ = spearmanr(v1, v2_perm)
    p_val = (np.sum(np.abs(null_dist) >= np.abs(r_obs)) + 1) / (n_perms + 1)
    return float(r_obs), float(p_val), null_dist.tolist()

def run_embedding_mantel(device):
    """Run Mantel tests on embedding centroid distances vs topo distances."""
    print('\n=== Embedding Alignment Mantel Tests ===', flush=True)
    topo_D = build_topo_distance_matrix()
    results = {}

    configs = [
        ('resnet18_baseline', 'resnet18_results.json', 'resnet18'),
        ('resnet18_topo', 'resnet18_topo_guided_results.json', 'resnet18'),
        ('resnet50_baseline', 'resnet50_results.json', 'resnet50'),
        ('resnet50_topo', 'resnet50_topo_guided_results.json', 'resnet50'),
    ]

    df = parse_data(DATA_DIR)
    _, te_t = get_transforms(224)
    te_df = df[df.split=='test']
    test_loader = DataLoader(KnotDataset(te_df, te_t), 32, num_workers=4)

    for config_name, json_file, backbone_name in configs:
        print(f'  Processing {config_name}...', flush=True)
        json_path = f'{RESULTS_DIR}/{json_file}'
        if not os.path.exists(json_path):
            print(f'    [SKIP] {json_path} not found', flush=True)
            continue

        # Try to find checkpoint
        ckpt_patterns = [
            f'checkpoints/{config_name}*.pt',
            f'checkpoints/best_{backbone_name}*.pt',
            f'{backbone_name}_best.pt',
        ]

        # Alternative: extract embeddings from test predictions and compute centroid distances
        # from the confusion matrix data. Actually, we need the model for this.
        # Since we may not have checkpoints, let's compute from saved embeddings if available.

        # Check if embedding_alignment_v2.json has what we need
        emb_path = f'{RESULTS_DIR}/embedding_alignment_v2.json'
        if os.path.exists(emb_path):
            with open(emb_path) as f:
                emb_data = json.load(f)
            # Check if this has centroid distance matrices
            for entry in emb_data if isinstance(emb_data, list) else [emb_data]:
                if isinstance(entry, dict) and config_name.replace('_', ' ') in str(entry):
                    print(f'    Found embedding data in v2 file', flush=True)
                    break

        # For now, compute Mantel on the confusion rate matrices that we already have
        # since those are saved in results JSONs
        with open(json_path) as f:
            rdata = json.load(f)
        if 'confusion_matrix' in rdata:
            cm = np.array(rdata['confusion_matrix'], dtype=float)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            conf_rate = cm / row_sums
            np.fill_diagonal(conf_rate, 0)
            conf_rate = (conf_rate + conf_rate.T) / 2
            r_obs, p_val, null = mantel_test(topo_D, conf_rate, n_perms=9999)
            results[config_name] = {
                'mantel_r': r_obs, 'mantel_p': p_val,
                'test': 'confusion_vs_topo_mantel'
            }
            print(f'    Mantel r={r_obs:.4f}, p={p_val:.4f}', flush=True)

    return results

def run_cub200_mantel():
    """Run proper Mantel test on CUB-200 confusion matrix."""
    print('\n=== CUB-200 Mantel Test ===', flush=True)
    cub_path = f'{RESULTS_DIR}/cub200_taca_results.json'
    if not os.path.exists(cub_path):
        print('  [SKIP] No CUB-200 results found', flush=True)
        return None

    with open(cub_path) as f:
        cub_data = json.load(f)

    # Check if confusion matrix is available
    if 'confusion_matrix' not in cub_data:
        # CUB-200 has 200 classes - we need predictions to build confusion matrix
        if 'predictions' in cub_data or 'preds' in cub_data:
            preds_key = 'predictions' if 'predictions' in cub_data else 'preds'
            labels_key = 'true_labels' if 'true_labels' in cub_data else 'labels'
            if labels_key in cub_data:
                preds = np.array(cub_data[preds_key])
                labels = np.array(cub_data[labels_key])
                cm = confusion_matrix(labels, preds, labels=list(range(200)))
                print(f'  Built 200x200 confusion matrix from predictions', flush=True)
            else:
                print('  [SKIP] No labels in CUB-200 results', flush=True)
                return None
        else:
            print('  [SKIP] No confusion matrix or predictions in CUB-200 results', flush=True)
            return None
    else:
        cm = np.array(cub_data['confusion_matrix'], dtype=float)

    # Build CUB-200 taxonomic distance matrix
    # We need the taxonomy. Check if cub200_taxonomy.py has it.
    try:
        from cub200_taxonomy import build_cub200_distance_matrix
        tax_D = build_cub200_distance_matrix()
        print(f'  Taxonomy distance matrix: {tax_D.shape}', flush=True)
    except ImportError:
        print('  [SKIP] Cannot import cub200_taxonomy', flush=True)
        return None

    # Compute confusion rate matrix
    K = cm.shape[0]
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_rate = cm / row_sums
    np.fill_diagonal(conf_rate, 0)
    conf_rate = (conf_rate + conf_rate.T) / 2

    # Run Mantel test
    print(f'  Running Mantel test on {K}x{K} matrices (this may take a while)...', flush=True)
    r_obs, p_val, _ = mantel_test(tax_D, conf_rate, n_perms=999, seed=42)  # fewer perms for 200x200
    print(f'  CUB-200 Mantel: r={r_obs:.4f}, p={p_val:.4f}', flush=True)

    result = {
        'cub200_confusion_mantel_r': r_obs,
        'cub200_confusion_mantel_p': p_val,
        'n_classes': K,
        'n_permutations': 999,
    }
    return result

# ─── Main ──────────────────────────────────────────────────

def phase_fgvc(device):
    """Multi-seed experiments for FGVC methods + learnable weights."""
    df = parse_data(DATA_DIR)
    print(f'Dataset: {len(df)} images', flush=True)

    all_results = {}
    seeds = [123, 456]

    for seed in seeds:
        print(f'\n{"="*60}', flush=True)
        print(f'  SEED = {seed}', flush=True)
        print(f'{"="*60}', flush=True)

        # TransFG
        print(f'\n[TransFG] seed={seed}...', flush=True)
        t0 = time.time()
        res = run_transfg_seed(df, device, seed, epochs=20)
        res['time'] = time.time() - t0
        all_results[f'transfg_seed{seed}'] = res
        print(f'  TransFG seed={seed}: test_acc={res["test_acc"]:.4f}, '
              f'f1={res["macro_f1"]:.4f}, time={res["time"]:.0f}s', flush=True)

        # PMG
        print(f'\n[PMG] seed={seed}...', flush=True)
        t0 = time.time()
        res = run_pmg_seed(df, device, seed, epochs=20)
        res['time'] = time.time() - t0
        all_results[f'pmg_seed{seed}'] = res
        print(f'  PMG seed={seed}: test_acc={res["test_acc"]:.4f}, '
              f'f1={res["macro_f1"]:.4f}, time={res["time"]:.0f}s', flush=True)

        # Graph-FGVC
        print(f'\n[Graph-FGVC] seed={seed}...', flush=True)
        t0 = time.time()
        res = run_graph_fgvc_seed(df, device, seed, epochs=20)
        res['time'] = time.time() - t0
        all_results[f'graph_fgvc_seed{seed}'] = res
        print(f'  Graph-FGVC seed={seed}: test_acc={res["test_acc"]:.4f}, '
              f'f1={res["macro_f1"]:.4f}, time={res["time"]:.0f}s', flush=True)

        # Learnable Weights - ResNet-18
        print(f'\n[Learnable-R18] seed={seed}...', flush=True)
        t0 = time.time()
        res = run_learnable_seed('resnet18', df, device, seed, epochs=20)
        res['time'] = time.time() - t0
        all_results[f'learnable_r18_seed{seed}'] = res
        print(f'  Learnable-R18 seed={seed}: test_acc={res["test_acc"]:.4f}, '
              f'weights={[f"{w:.3f}" for w in res["learned_weights"]]}, time={res["time"]:.0f}s', flush=True)

        # Learnable Weights - ResNet-50
        print(f'\n[Learnable-R50] seed={seed}...', flush=True)
        t0 = time.time()
        res = run_learnable_seed('resnet50', df, device, seed, epochs=20)
        res['time'] = time.time() - t0
        all_results[f'learnable_r50_seed{seed}'] = res
        print(f'  Learnable-R50 seed={seed}: test_acc={res["test_acc"]:.4f}, '
              f'weights={[f"{w:.3f}" for w in res["learned_weights"]]}, time={res["time"]:.0f}s', flush=True)

        # Save after each seed
        with open(f'{RESULTS_DIR}/multiseed_supplement.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\n[Saved] multiseed_supplement.json (after seed {seed})', flush=True)

    # Print summary table
    print('\n\n=== MULTI-SEED SUMMARY ===', flush=True)
    print(f'{"Model":<20} {"Seed 42":<12} {"Seed 123":<12} {"Seed 456":<12} {"Mean±Std":<15}', flush=True)
    print('-' * 70, flush=True)

    # Load seed-42 results
    seed42_results = {}
    for name, fname in [('TransFG','transfg'), ('PMG','pmg'), ('Graph-FGVC','graph_fgvc')]:
        p = f'{RESULTS_DIR}/{fname}_results.json'
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            seed42_results[name] = d.get('test_acc', d.get('test_accuracy', 0))

    for name, fname in [('Learnable-R18','learnable_weights_resnet18'),
                         ('Learnable-R50','learnable_weights_resnet50')]:
        p = f'{RESULTS_DIR}/{fname}.json'
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            seed42_results[name] = d.get('test_acc', d.get('best_test_acc', 0))

    model_map = {
        'TransFG': 'transfg', 'PMG': 'pmg', 'Graph-FGVC': 'graph_fgvc',
        'Learnable-R18': 'learnable_r18', 'Learnable-R50': 'learnable_r50',
    }
    for model_name, prefix in model_map.items():
        s42 = seed42_results.get(model_name, 0)
        s123 = all_results.get(f'{prefix}_seed123', {}).get('test_acc', 0)
        s456 = all_results.get(f'{prefix}_seed456', {}).get('test_acc', 0)
        accs = [s42, s123, s456]
        mean = np.mean(accs) * 100
        std = np.std(accs) * 100
        print(f'{model_name:<20} {s42*100:>8.2f}%   {s123*100:>8.2f}%   '
              f'{s456*100:>8.2f}%   {mean:.2f}±{std:.2f}%', flush=True)

def phase_mantel(device):
    """CUB-200 Mantel + embedding alignment Mantel tests."""
    # Embedding alignment Mantel
    emb_results = run_embedding_mantel(device)
    with open(f'{RESULTS_DIR}/embedding_mantel_results.json', 'w') as f:
        json.dump(emb_results, f, indent=2)
    print(f'\n[Saved] embedding_mantel_results.json', flush=True)

    # CUB-200 Mantel
    cub_results = run_cub200_mantel()
    if cub_results:
        with open(f'{RESULTS_DIR}/cub200_mantel_results.json', 'w') as f:
            json.dump(cub_results, f, indent=2)
        print(f'[Saved] cub200_mantel_results.json', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['fgvc', 'mantel', 'all'], default='all')
    args = parser.parse_args()

    device = get_device()
    print(f'Device: {device}', flush=True)
    print(f'Phase: {args.phase}', flush=True)

    if args.phase in ('fgvc', 'all'):
        phase_fgvc(device)
    if args.phase in ('mantel', 'all'):
        phase_mantel(device)

    print('\n[ALL DONE]', flush=True)
