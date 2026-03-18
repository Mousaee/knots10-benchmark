#!/usr/bin/env python3
"""TransFG (Transformer for Fine-Grained Recognition) - Knot Classification"""
import os, sys, json, time, copy, random, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
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

class TransFGModel(nn.Module):
    """TransFG: Transformer for Fine-Grained Recognition with Part Selection Module"""
    def __init__(self, num_classes=10, pretrained=True, top_k=6):
        super(TransFGModel, self).__init__()
        self.num_classes = num_classes
        self.top_k = top_k

        # Load pre-trained ViT
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        self.hidden_dim = 768  # ViT-B/16 hidden dimension

        # Store ViT components
        self.patch_embed = vit.conv_proj
        self.encoder_ln = vit.encoder.ln
        self.encoder_blocks = nn.ModuleList(vit.encoder.layers)
        self.vit_class_token = nn.Parameter(vit.class_token.clone())
        self.vit_encoder_pos_embedding = nn.Parameter(vit.encoder.pos_embedding.clone())

        # Learnable attention weights for part selection (alternative to direct attention weights)
        # This acts as a part selector that learns which patches are important
        self.part_selector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        # Two classification heads: global (CLS) and part (selected parts)
        self.head_global = nn.Linear(self.hidden_dim, num_classes)
        self.head_part = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass with Part Selection Module

        Args:
            x: Input images (B, 3, 224, 224)

        Returns:
            logits: Combined classification logits (B, num_classes)
        """
        B, C, H, W = x.shape

        # Patch embedding: (B, 3, 224, 224) -> (B, num_patches, hidden_dim)
        # Flatten patches
        x = self.patch_embed(x)  # (B, hidden_dim, 14, 14)
        x = x.reshape(B, self.hidden_dim, -1)  # (B, hidden_dim, 196)
        x = x.permute(0, 2, 1)  # (B, 196, hidden_dim)

        # Add CLS token
        batch_class_token = self.vit_class_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        x = torch.cat([batch_class_token, x], dim=1)  # (B, 197, hidden_dim)

        # Add positional embedding
        x = x + self.vit_encoder_pos_embedding

        # Pass through encoder blocks
        for block in self.encoder_blocks:
            x = block(x)

        # Layer norm
        x = self.encoder_ln(x)

        # x shape: (B, 197, hidden_dim)
        # x[:, 0] is CLS token, x[:, 1:] are patch tokens
        cls_token = x[:, 0]  # (B, hidden_dim)
        patch_tokens = x[:, 1:]  # (B, 196, hidden_dim)

        # Part Selection Module: select top-k patches
        # Use part selector to compute importance scores for each patch
        patch_scores = self.part_selector(patch_tokens)  # (B, 196, 1)
        patch_scores = patch_scores.squeeze(-1)  # (B, 196)

        # Select top-k patches based on importance scores
        actual_k = min(self.top_k, patch_tokens.shape[1])
        _, top_k_indices = torch.topk(patch_scores, actual_k, dim=1)

        # Gather top-k patch tokens
        # top_k_indices: (B, actual_k)
        part_tokens = torch.gather(patch_tokens, 1,
                                  top_k_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim))  # (B, actual_k, hidden_dim)

        # Aggregate selected parts via mean pooling
        part_feat = part_tokens.mean(dim=1)  # (B, hidden_dim)

        # Classification heads
        global_logits = self.head_global(cls_token)  # (B, num_classes)
        part_logits = self.head_part(part_feat)      # (B, num_classes)

        # Combine predictions: average the two heads
        combined_logits = (global_logits + part_logits) / 2.0

        return combined_logits

def make_transfg_model(num_classes=10, device='cpu', pretrained=True, top_k=6):
    """Create TransFG model"""
    model = TransFGModel(num_classes=num_classes, pretrained=pretrained, top_k=top_k)
    return model.to(device)

def train_model(model, loaders, device, epochs=20, lr=1e-4):
    """Train TransFG model"""
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
    """Evaluate model on dataset"""
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); out = model(x)
            preds.extend(torch.max(out,1)[1].cpu().numpy())
            labels.extend(y.numpy())
    return np.array(preds), np.array(labels)

def count_parameters(model):
    """Count total number of parameters"""
    return sum(p.numel() for p in model.parameters()) / 1e6

def run_experiment(df, device, epochs=20, top_k=6):
    """Run TransFG experiment"""
    print(f'\n{"="*50}\n  Running: TransFG (top_k={top_k})\n{"="*50}', flush=True)
    tr_full = df[df['split']=='train']; te_df = df[df['split']=='test']
    tr_df, va_df = train_test_split(tr_full, test_size=0.2, stratify=tr_full['label'], random_state=SEED)
    tr_tf, te_tf = get_transforms()
    loaders = {
        'train': DataLoader(KnotDataset(tr_df, tr_tf), batch_size=32, shuffle=True),
        'val': DataLoader(KnotDataset(va_df, te_tf), batch_size=32)
    }
    test_loader = DataLoader(KnotDataset(te_df, te_tf), batch_size=32)
    model = make_transfg_model(num_classes=len(CLASSES), device=device, pretrained=True, top_k=top_k)
    params = count_parameters(model)
    print(f'  Model parameters: {params:.2f}M', flush=True)
    t0 = time.time()
    model, hist, best_val = train_model(model, loaders, device, epochs)
    train_time = time.time() - t0
    preds, labels = evaluate(model, test_loader, device)
    report = classification_report(labels, preds, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds)
    test_acc = (preds == labels).mean()
    print(f'  Test Acc: {test_acc:.4f} | Best Val: {best_val:.4f} | Time: {train_time:.0f}s', flush=True)
    print(classification_report(labels, preds, target_names=CLASSES, zero_division=0), flush=True)

    # Compute macro F1
    macro_f1 = report['macro avg']['f1-score']

    return {
        'model': 'TransFG',
        'params': f'{params:.2f}M',
        'val_acc': float(best_val),
        'test_acc': float(test_acc),
        'macro_f1': float(macro_f1),
        'training_time': train_time,
        'history': hist,
        'confusion_matrix': cm.tolist(),
        'per_class_report': report,
        'preds': preds.tolist(),
        'labels': labels.tolist()
    }

if __name__ == '__main__':
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print(f'Device: {device}', flush=True)
    df = parse_data(DATA_DIR)
    print(f'Total: {len(df)} (train:{len(df[df.split=="train"])}, test:{len(df[df.split=="test"])})', flush=True)

    # TransFG with top_k=6
    res = run_experiment(df, device, epochs=20, top_k=6)
    with open(f'{RESULTS_DIR}/transfg_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('[Saved] transfg_results.json', flush=True)

    print('\n[DONE] TransFG experiment complete.', flush=True)
