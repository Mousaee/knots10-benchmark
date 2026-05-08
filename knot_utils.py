#!/usr/bin/env python3
"""
Shared utilities for Knots-10 experiments (S0–S4).

Provides canonical data loading, model creation, evaluation,
and McNemar paired tests with a unified interface.
"""
import os, glob, random, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# ── Constants ───────────────────────────────────────────────────────────────
CLASSES = ['ABK', 'BK', 'CH', 'F8K', 'F8L', 'FSK', 'FMB', 'OHK', 'RK', 'SK']
NUM_CLASSES = len(CLASSES)
C2I = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

KNOT_PROPERTIES = {
    'OHK': {'crossing_num': 3, 'type': 'prime',     'family': 'stopper', 'components': 1},
    'F8K': {'crossing_num': 4, 'type': 'prime',     'family': 'stopper', 'components': 1},
    'BK':  {'crossing_num': 4, 'type': 'loop',      'family': 'loop',    'components': 1},
    'RK':  {'crossing_num': 6, 'type': 'composite', 'family': 'binding', 'components': 2},
    'FSK': {'crossing_num': 6, 'type': 'composite', 'family': 'bend',    'components': 2},
    'FMB': {'crossing_num': 8, 'type': 'composite', 'family': 'bend',    'components': 2},
    'F8L': {'crossing_num': 4, 'type': 'loop',      'family': 'loop',    'components': 1},
    'CH':  {'crossing_num': 2, 'type': 'hitch',     'family': 'hitch',   'components': 1},
    'SK':  {'crossing_num': 3, 'type': 'slip',      'family': 'stopper', 'components': 1},
    'ABK': {'crossing_num': 4, 'type': 'loop',      'family': 'loop',    'components': 1},
}

# Visual crossing number per class (ordered by CLASSES)
CROSSING_NUMS = [KNOT_PROPERTIES[c]['crossing_num'] for c in CLASSES]
FAMILY_LABELS = {f: i for i, f in enumerate(sorted(set(
    p['family'] for p in KNOT_PROPERTIES.values())))}

# Hard pairs for focused evaluation (topologically close, visually confusable)
HARD_PAIRS = [
    ('F8K', 'OHK'),   # d=0.156, both prime stopper
    ('FSK', 'FMB'),   # d=0.150, both composite bend
    ('F8K', 'SK'),    # d=0.231, trefoil variants
    ('F8K', 'F8L'),   # d=0.256, figure-8 variants
    ('RK',  'FSK'),   # d=0.188, both composite
]

DERIVATION_PAIRS = {
    ('OHK', 'SK'):  0.1,
    ('F8K', 'FMB'): 0.15,
    ('RK',  'FSK'): 0.1,
    ('F8K', 'F8L'): 0.1,
}


# ── Seed / Device ───────────────────────────────────────────────────────────
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
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Topological Distance ────────────────────────────────────────────────────
def topological_distance(k1, k2):
    p1, p2 = KNOT_PROPERTIES[k1], KNOT_PROPERTIES[k2]
    d_cross = abs(p1['crossing_num'] - p2['crossing_num']) / 8.0
    d_family = 0.0 if p1['family'] == p2['family'] else 1.0
    d_type = 0.0 if p1['type'] == p2['type'] else 0.5
    d_comp = abs(p1['components'] - p2['components'])
    pair = tuple(sorted([k1, k2]))
    d_deriv = DERIVATION_PAIRS.get(pair, 0.5)
    return 0.25 * d_cross + 0.25 * d_family + 0.15 * d_type + 0.10 * d_comp + 0.25 * d_deriv


def build_topo_distance_matrix():
    n = NUM_CLASSES
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = topological_distance(CLASSES[i], CLASSES[j])
    return D


# ── Data ────────────────────────────────────────────────────────────────────
def parse_data(data_dir):
    """Parse 10Knots dataset with metadata (lighting, tightness)."""
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True))
    rows = []
    for f in files:
        fn = os.path.basename(f)
        parts = fn.split('_')
        if parts[0] not in C2I:
            continue
        if 'Loose' in fn or 'VeryLoose' in fn:
            sp = 'train'
        elif 'Set' in fn:
            sp = 'test'
        else:
            continue
        rows.append({
            'path': f,
            'label': C2I[parts[0]],
            'class_name': parts[0],
            'split': sp,
            'light': 'DL' if '_DL_' in fn else ('SLA' if '_SLA_' in fn else 'SLS'),
            'tightness': 'VeryLoose' if 'VeryLoose' in fn else ('Loose' if 'Loose' in fn else 'Set'),
            'filename': fn,
        })
    return pd.DataFrame(rows)


class KnotDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(row['label'], dtype=torch.long)


def get_transforms(sz=224):
    norm = (IMAGENET_MEAN, IMAGENET_STD)
    tr = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(*norm),
    ])
    te = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(*norm),
    ])
    return tr, te


def get_canonical_test_loader(data_dir, batch_size=64, sz=224):
    """Return a deterministic test DataLoader (sorted by path)."""
    df = parse_data(data_dir)
    test_df = df[df['split'] == 'test'].sort_values('path').reset_index(drop=True)
    _, te_transform = get_transforms(sz)
    ds = KnotDataset(test_df, transform=te_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    return loader, test_df


def get_train_val_loaders(data_dir, seed=42, batch_size=32, sz=224, val_ratio=0.2):
    """Return train/val DataLoaders with deterministic split."""
    df = parse_data(data_dir)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    from sklearn.model_selection import train_test_split
    tr_df, val_df = train_test_split(
        train_df, test_size=val_ratio, random_state=seed, stratify=train_df['label'])
    tr_transform, te_transform = get_transforms(sz)
    tr_loader = DataLoader(KnotDataset(tr_df, tr_transform),
                           batch_size=batch_size, shuffle=True,
                           num_workers=2, pin_memory=True)
    val_loader = DataLoader(KnotDataset(val_df, te_transform),
                            batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    return tr_loader, val_loader


# ── Models ──────────────────────────────────────────────────────────────────
# Embedding extraction wrapper used by TACA and other experiments.
class EmbeddingModel(nn.Module):
    """Wraps a backbone so that forward() returns (logits, embeddings)."""
    def __init__(self, backbone, embed_dim, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        emb = self.backbone(x)
        logits = self.classifier(emb)
        return logits, emb


def _make_backbone(name):
    """Return (backbone_with_identity_head, embed_dim)."""
    if name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        dim = m.fc.in_features; m.fc = nn.Identity()
    elif name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        dim = m.fc.in_features; m.fc = nn.Identity()
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        dim = m.classifier[1].in_features; m.classifier = nn.Identity()
    elif name == 'vit':
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        dim = m.heads.head.in_features; m.heads.head = nn.Identity()
    elif name == 'swin_t':
        m = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        dim = m.head.in_features; m.head = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return m, dim


def make_model(name, num_classes=NUM_CLASSES, with_embeddings=False):
    """Create a classification model.

    All models use EmbeddingModel internally (backbone.* + classifier.* keys)
    to match the checkpoint format used by existing training scripts.

    If with_embeddings=True, forward() returns (logits, embeddings).
    If with_embeddings=False (default), same architecture but the caller
    should use evaluate() which handles both tuple and plain outputs.
    """
    bb, dim = _make_backbone(name)
    return EmbeddingModel(bb, dim, num_classes)


EMBED_DIMS = {
    'resnet18': 512, 'resnet50': 2048, 'efficientnet_b0': 1280,
    'vit': 768, 'swin_t': 768,
}


# ── Training utility ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device,
                    extra_loss_fn=None):
    """Generic training loop for one epoch.

    Args:
        extra_loss_fn: callable(logits, labels, embeddings) -> scalar loss.
            Used by TACA / AuxCrossing etc.  May be None.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        if isinstance(out, tuple):
            logits, emb = out
        else:
            logits, emb = out, None
        loss = criterion(logits, labels)
        if extra_loss_fn is not None and emb is not None:
            loss = loss + extra_loss_fn(logits, labels, emb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (accuracy, all_preds, all_labels) on a DataLoader."""
    model.eval()
    preds_list, labels_list = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        logits = out[0] if isinstance(out, tuple) else out
        preds_list.append(logits.argmax(1).cpu())
        labels_list.append(labels)
    preds = torch.cat(preds_list).numpy()
    labels = torch.cat(labels_list).numpy()
    acc = (preds == labels).mean()
    return acc, preds, labels


# ── McNemar ─────────────────────────────────────────────────────────────────
def mcnemar_test(preds_a, preds_b, labels):
    """McNemar paired test between two models' predictions.

    Returns dict with b, c, chi2, p_value.
    """
    from scipy.stats import chi2 as chi2_dist
    a_correct = (preds_a == labels)
    b_correct = (preds_b == labels)
    b = int((~a_correct & b_correct).sum())   # A wrong, B right
    c = int((a_correct & ~b_correct).sum())   # A right, B wrong
    if b + c == 0:
        return {'b': b, 'c': c, 'chi2': 0.0, 'p': 1.0}
    chi2_val = (abs(b - c) - 1) ** 2 / (b + c)  # with continuity correction
    p = 1.0 - chi2_dist.cdf(chi2_val, df=1)
    return {'b': b, 'c': c, 'chi2': round(chi2_val, 3), 'p': round(p, 6)}


def hard_pair_accuracy(preds, labels, class_a, class_b):
    """Accuracy on test samples belonging to class_a or class_b only."""
    ia, ib = C2I[class_a], C2I[class_b]
    mask = (labels == ia) | (labels == ib)
    if mask.sum() == 0:
        return float('nan')
    return float((preds[mask] == labels[mask]).mean())


# ── Checkpoint I/O ──────────────────────────────────────────────────────────
def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
