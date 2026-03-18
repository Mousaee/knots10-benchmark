#!/usr/bin/env python3
"""
Multi-Seed Robustness Analysis for Supplementary Material.

Runs key experiments with 3 random seeds and reports mean ± std.
Priority order (most important first):
  1. ResNet-18 baseline (CE only)         ~27 min/seed
  2. ResNet-18 + TACA                     ~27 min/seed
  3. Swin-T baseline                      ~55 min/seed
  4. ResNet-50 baseline                   ~30 min/seed
  5. EfficientNet-B0 baseline             ~28 min/seed
  6. ViT-B/16 baseline                    ~40 min/seed
  7. ResNet-18 + TAML                     ~32 min/seed
  8. ResNet-18 + TACA+TAML                ~32 min/seed

Estimated total: ~15 hours (all 8 configs × 3 seeds)
First 3 configs (most critical): ~5.5 hours
"""
import os, json, time, copy, random, glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
NUM_CLASSES = 10
DATA_DIR = './train'
RESULTS_DIR = 'results'
SEEDS = [42, 123, 456]

TOPO_DIST = np.array([
    [0.000, 0.400, 0.650, 0.406, 0.150, 0.550, 0.700, 0.556, 0.700, 0.556],
    [0.400, 0.000, 0.650, 0.406, 0.250, 0.550, 0.700, 0.556, 0.700, 0.556],
    [0.650, 0.650, 0.000, 0.656, 0.500, 0.400, 0.550, 0.806, 0.550, 0.806],
    [0.406, 0.406, 0.656, 0.000, 0.256, 0.356, 0.506, 0.156, 0.506, 0.231],
    [0.150, 0.250, 0.500, 0.256, 0.000, 0.400, 0.550, 0.406, 0.550, 0.406],
    [0.550, 0.550, 0.400, 0.356, 0.400, 0.000, 0.150, 0.506, 0.188, 0.506],
    [0.700, 0.700, 0.550, 0.506, 0.550, 0.150, 0.000, 0.656, 0.300, 0.656],
    [0.556, 0.556, 0.806, 0.156, 0.406, 0.506, 0.656, 0.000, 0.656, 0.231],
    [0.700, 0.700, 0.550, 0.506, 0.550, 0.188, 0.300, 0.656, 0.000, 0.656],
    [0.556, 0.556, 0.806, 0.231, 0.406, 0.506, 0.656, 0.231, 0.656, 0.000]
])

# ── Utilities ──────────────────────────────────────────────────

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

class KnotDataset(Dataset):
    def __init__(self, df, transform=None, preload=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.images = []
        if preload:
            for _, row in self.df.iterrows():
                try:
                    img = Image.open(row['path']).convert('RGB')
                except Exception:
                    img = Image.new('RGB', (224, 224), (0, 0, 0))
                self.images.append(img)
        self.preloaded = preload
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        if self.preloaded:
            img = self.images[idx]
        else:
            row = self.df.iloc[idx]
            try: img = Image.open(row['path']).convert('RGB')
            except Exception: img = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.df.iloc[idx]['label'], dtype=torch.long)

def parse_data(data_dir):
    import pandas as pd
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True))
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

def get_transforms():
    norm = ([0.485,0.456,0.406],[0.229,0.224,0.225])
    tr = transforms.Compose([
        transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), transforms.ColorJitter(0.1,0.1),
        transforms.ToTensor(), transforms.Normalize(*norm)])
    te = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(*norm)])
    return tr, te

def make_model(name, nc, device):
    if name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, nc))
    elif name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, nc))
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
    elif name == 'vit':
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        m.heads.head = nn.Linear(m.heads.head.in_features, nc)
    elif name == 'swin_t':
        m = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        m.head = nn.Linear(m.head.in_features, nc)
    return m.to(device)

# ── TACA Loss ──────────────────────────────────────────────────

class TopoLoss(nn.Module):
    def __init__(self, topo_dist, lam_taca=0.0, lam_taml=0.0):
        super().__init__()
        self.register_buffer('topo', torch.tensor(topo_dist, dtype=torch.float32))
        self.lam_taca = lam_taca
        self.lam_taml = lam_taml
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels, emb=None):
        dev = logits.device
        ce = self.ce(logits, labels)
        total = ce

        if self.lam_taca > 0 and emb is not None:
            cents = []
            for c in range(NUM_CLASSES):
                mask = (labels == c)
                if mask.sum() > 0:
                    cents.append(emb[mask].mean(0))
                else:
                    cents.append(torch.zeros_like(emb[0]))
            cents = torch.stack(cents)
            diff = cents.unsqueeze(0) - cents.unsqueeze(1)
            edist = torch.sqrt((diff ** 2).sum(-1) + 1e-12)
            edist = edist / (edist.max() + 1e-8)
            total = total + self.lam_taca * nn.functional.mse_loss(edist, self.topo)

        if self.lam_taml > 0 and emb is not None:
            enorm = nn.functional.normalize(emb, dim=1)
            sim = torch.mm(enorm, enorm.t())
            B = emb.size(0)
            # Vectorized: mask for different-index AND different-label pairs
            idx_mask = ~torch.eye(B, dtype=torch.bool, device=dev)
            label_i = labels.unsqueeze(1).expand(B, B)
            label_j = labels.unsqueeze(0).expand(B, B)
            diff_label_mask = label_i != label_j
            mask = idx_mask & diff_label_mask
            if mask.any():
                margins = self.topo[label_i, label_j]  # (B, B)
                losses = torch.relu(sim - (1.0 - margins))
                total = total + self.lam_taml * losses[mask].mean()

        return total

# ── Training ───────────────────────────────────────────────────

def train_and_eval(model_name, df, device, seed, lam_taca=0.0, lam_taml=0.0, epochs=20):
    """Train one model with one seed. Returns dict of metrics."""
    set_seed(seed)
    tr_full = df[df['split']=='train']
    te_df = df[df['split']=='test']
    tr_df, va_df = train_test_split(
        tr_full, test_size=0.2, stratify=tr_full['label'], random_state=seed)

    tr_tf, te_tf = get_transforms()
    use_cuda = (device.type == 'cuda')
    nw = 4 if use_cuda else 2
    loader_kw = dict(batch_size=32, num_workers=nw, persistent_workers=True,
                     pin_memory=use_cuda)
    tr_ld = DataLoader(KnotDataset(tr_df, tr_tf), shuffle=True, **loader_kw)
    va_ld = DataLoader(KnotDataset(va_df, te_tf), **loader_kw)
    te_ld = DataLoader(KnotDataset(te_df, te_tf), **loader_kw)

    model = make_model(model_name, NUM_CLASSES, device)
    use_topo = (lam_taca > 0 or lam_taml > 0)

    if use_topo:
        crit = TopoLoss(TOPO_DIST, lam_taca, lam_taml).to(device)
    else:
        crit = nn.CrossEntropyLoss()

    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_st = 0.0, None
    t0 = time.time()

    for ep in range(1, epochs+1):
        model.train()
        for imgs, labs in tr_ld:
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            if use_topo:
                # Extract embeddings for topo loss
                if model_name in ('resnet18', 'resnet50'):
                    feat = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(
                        model.maxpool(model.relu(model.bn1(model.conv1(imgs)))))))))
                    emb = feat.view(feat.size(0), -1)
                    logits = model.fc(emb)
                else:
                    logits = model(imgs)
                    emb = None
                loss = crit(logits, labs, emb)
            else:
                logits = model(imgs)
                loss = crit(logits, labs)
            loss.backward()
            opt.step()
        sch.step()

        # Validation
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, labs in va_ld:
                imgs, labs = imgs.to(device), labs.to(device)
                vc += (model(imgs).argmax(1)==labs).sum().item()
                vt += imgs.size(0)
        val_acc = vc/vt
        if val_acc > best_val:
            best_val = val_acc
            best_st = copy.deepcopy(model.state_dict())

    train_time = time.time() - t0

    # Test
    model.load_state_dict(best_st)
    model.eval()
    preds, labs_list = [], []
    with torch.no_grad():
        for imgs, l in te_ld:
            imgs = imgs.to(device)
            preds.extend(model(imgs).argmax(1).cpu().numpy())
            labs_list.extend(l.numpy())

    preds, labs_list = np.array(preds), np.array(labs_list)
    test_acc = (preds == labs_list).mean()
    rep = classification_report(labs_list, preds, target_names=CLASSES, output_dict=True, zero_division=0)
    f1 = rep['macro avg']['f1-score']

    return {
        'val_acc': best_val,
        'test_acc': float(test_acc),
        'f1': float(f1),
        'time': train_time
    }

# ── Main ───────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    df = parse_data(DATA_DIR)
    print(f"Device: {device}")
    print(f"Data: {len(df)} images (train={len(df[df.split=='train'])}, test={len(df[df.split=='test'])})")
    print(f"Seeds: {SEEDS}")
    print(f"{'='*70}\n")

    # Define all configs in priority order
    configs = [
        # (display_name, model_name, lam_taca, lam_taml)
        ('ResNet-18 (CE)',        'resnet18',       0.0,   0.0),
        ('ResNet-18 (CE+TACA)',   'resnet18',       0.1,   0.0),
        ('Swin-T (CE)',           'swin_t',         0.0,   0.0),
        ('ResNet-50 (CE)',        'resnet50',       0.0,   0.0),
        ('EfficientNet-B0 (CE)', 'efficientnet_b0', 0.0,   0.0),
        ('ViT-B/16 (CE)',         'vit',            0.0,   0.0),
        ('ResNet-18 (CE+TAML)',   'resnet18',       0.0,   0.005),
        ('ResNet-18 (CE+TACA+TAML)', 'resnet18',    0.1,   0.005),
    ]

    all_results = {}
    total_start = time.time()

    for ci, (disp_name, model_name, lt, lm) in enumerate(configs):
        print(f"\n{'='*70}")
        print(f"  [{ci+1}/{len(configs)}] {disp_name}")
        print(f"{'='*70}")

        seed_results = []
        for si, seed in enumerate(SEEDS):
            print(f"  Seed {seed} ({si+1}/{len(SEEDS)}) ... ", end='', flush=True)
            r = train_and_eval(model_name, df, device, seed, lt, lm, epochs=20)
            seed_results.append(r)
            print(f"Val={r['val_acc']:.4f} Test={r['test_acc']:.4f} "
                  f"F1={r['f1']:.4f} Time={r['time']:.0f}s", flush=True)

        # Compute summary
        vals = [r['val_acc'] for r in seed_results]
        tests = [r['test_acc'] for r in seed_results]
        f1s = [r['f1'] for r in seed_results]

        summary = {
            'config': disp_name,
            'model': model_name,
            'lam_taca': lt,
            'lam_taml': lm,
            'seeds': SEEDS,
            'per_seed': seed_results,
            'val_acc_mean': float(np.mean(vals)),
            'val_acc_std': float(np.std(vals)),
            'test_acc_mean': float(np.mean(tests)),
            'test_acc_std': float(np.std(tests)),
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
        }
        all_results[disp_name] = summary

        print(f"\n  >>> {disp_name}: Test = {summary['test_acc_mean']*100:.2f}% "
              f"± {summary['test_acc_std']*100:.2f}%  "
              f"F1 = {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")

        # Save incrementally (so partial results survive if killed)
        with open(f'{RESULTS_DIR}/robustness_multiseed.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    total_time = time.time() - total_start

    # Final summary table
    print(f"\n{'='*70}")
    print(f"  ROBUSTNESS SUMMARY  (total time: {total_time/3600:.1f} hours)")
    print(f"{'='*70}")
    print(f"{'Config':<28} {'Test Acc':>18} {'Macro F1':>18}")
    print(f"{'-'*28} {'-'*18} {'-'*18}")
    for name, s in all_results.items():
        print(f"{name:<28} {s['test_acc_mean']*100:6.2f}% ± {s['test_acc_std']*100:.2f}%"
              f"   {s['f1_mean']:.4f} ± {s['f1_std']:.4f}")

    print(f"\n[Saved] {RESULTS_DIR}/robustness_multiseed.json")
    print("[DONE]")
