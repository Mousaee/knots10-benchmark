#!/usr/bin/env python3
"""Progressive Multi-Granularity (PMG) Training for Knot Classification"""
import os, sys, json, time, copy, random, glob, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

class JigsawGenerator:
    """Generate jigsaw puzzles by shuffling image patches"""
    def __init__(self, n_grid=2):
        self.n_grid = n_grid
        self.n_patches = n_grid * n_grid
        self.perms = list(itertools.permutations(range(self.n_patches)))

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: (C, H, W) tensor
        Returns:
            shuffled_img: (C, H, W) shuffled tensor
            perm_index: int, index of permutation used
        """
        C, H, W = img_tensor.shape
        patch_h, patch_w = H // self.n_grid, W // self.n_grid

        # Reshape into patches: (C, n_grid, patch_h, n_grid, patch_w)
        patches = img_tensor.view(C, self.n_grid, patch_h, self.n_grid, patch_w)
        # Transpose to: (n_grid, n_grid, C, patch_h, patch_w)
        patches = patches.permute(1, 3, 0, 2, 4).contiguous()
        patches = patches.view(self.n_patches, C, patch_h, patch_w)

        # Random permutation
        perm_idx = random.randint(0, len(self.perms) - 1)
        perm = self.perms[perm_idx]
        shuffled_patches = patches[list(perm)]

        # Reshape back: (n_grid, n_grid, C, patch_h, patch_w)
        shuffled_patches = shuffled_patches.view(self.n_grid, self.n_grid, C, patch_h, patch_w)
        # Transpose to: (C, n_grid, patch_h, n_grid, patch_w)
        shuffled_patches = shuffled_patches.permute(2, 0, 3, 1, 4).contiguous()
        # Reshape to (C, H, W)
        shuffled_img = shuffled_patches.view(C, H, W)

        return shuffled_img, perm_idx

class PMGModel(nn.Module):
    """Progressive Multi-Granularity (PMG) Model"""
    def __init__(self, num_classes=10):
        super().__init__()
        base = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Split ResNet50 into stages
        self.conv_block1 = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool, base.layer1
        )  # Output: 256 channels
        self.conv_block2 = base.layer2  # Output: 512 channels
        self.conv_block3 = base.layer3  # Output: 1024 channels
        self.conv_block4 = base.layer4  # Output: 2048 channels

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Granularity-specific classifiers
        # Granularity 1: whole image (after layer4, 2048 channels)
        self.classifier1 = nn.Linear(2048, num_classes)
        # Granularity 2: 2x2 patches (after layer3, 1024 channels)
        self.classifier2 = nn.Linear(1024, num_classes)
        # Granularity 3: 4x4 patches (after layer2, 512 channels)
        self.classifier3 = nn.Linear(512, num_classes)

        # Concatenated classifier for all granularities
        self.classifier_concat = nn.Linear(2048 + 1024 + 512, num_classes)

        self.num_classes = num_classes

    def forward(self, x, return_features=False):
        """
        Forward pass with multi-granularity features

        Args:
            x: input tensor (B, C, H, W)
            return_features: if True, return intermediate features for visualization

        Returns:
            output: (B, num_classes) final classification logits
            or (output, features_dict) if return_features=True
        """
        # Stage 1: extract coarse features (256 channels)
        f1 = self.conv_block1(x)  # (B, 256, H/4, W/4)

        # Stage 2: medium features (512 channels)
        f2 = self.conv_block2(f1)  # (B, 512, H/8, W/8)

        # Stage 3: fine features (1024 channels)
        f3 = self.conv_block3(f2)  # (B, 1024, H/16, W/16)

        # Stage 4: finest features (2048 channels)
        f4 = self.conv_block4(f3)  # (B, 2048, H/32, W/32)

        # Global average pooling for each stage
        p1 = self.pool(f1).view(f1.size(0), -1)  # (B, 256)
        p2 = self.pool(f2).view(f2.size(0), -1)  # (B, 512)
        p3 = self.pool(f3).view(f3.size(0), -1)  # (B, 1024)
        p4 = self.pool(f4).view(f4.size(0), -1)  # (B, 2048)

        # Classification heads for each granularity
        out1 = self.classifier1(p4)  # (B, num_classes)
        out2 = self.classifier2(p3)  # (B, num_classes)
        out3 = self.classifier3(p2)  # (B, num_classes)

        # Concatenated features
        concat_features = torch.cat([p4, p3, p2], dim=1)  # (B, 2048+1024+512)
        out_concat = self.classifier_concat(concat_features)  # (B, num_classes)

        if return_features:
            return out_concat, {
                'out1': out1, 'out2': out2, 'out3': out3,
                'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4
            }
        return out_concat

def train_pmg(model, loaders, device, epochs=20, lr=1e-4):
    """
    Progressive Multi-Granularity training with staged learning

    Stages:
    - Epoch 0-4: Train granularity 3 (coarse) only
    - Epoch 5-9: Train granularities 2-3 (coarse+medium)
    - Epoch 10-19: Train all granularities (fine+medium+coarse)
    """
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    hist = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'loss1': [], 'loss2': [], 'loss3': [], 'loss_concat': []
    }

    for ep in range(epochs):
        # Determine training stage
        if ep < 5:
            # Stage 1: only train classifier3 + conv_block2
            stage = 1
            for param in model.conv_block1.parameters(): param.requires_grad = False
            for param in model.conv_block2.parameters(): param.requires_grad = True
            for param in model.conv_block3.parameters(): param.requires_grad = False
            for param in model.conv_block4.parameters(): param.requires_grad = False
            for param in model.classifier1.parameters(): param.requires_grad = False
            for param in model.classifier2.parameters(): param.requires_grad = False
            for param in model.classifier3.parameters(): param.requires_grad = True
            for param in model.classifier_concat.parameters(): param.requires_grad = False
        elif ep < 10:
            # Stage 2: train classifiers 2-3 + conv_blocks 2-3
            stage = 2
            for param in model.conv_block1.parameters(): param.requires_grad = False
            for param in model.conv_block2.parameters(): param.requires_grad = True
            for param in model.conv_block3.parameters(): param.requires_grad = True
            for param in model.conv_block4.parameters(): param.requires_grad = False
            for param in model.classifier1.parameters(): param.requires_grad = False
            for param in model.classifier2.parameters(): param.requires_grad = True
            for param in model.classifier3.parameters(): param.requires_grad = True
            for param in model.classifier_concat.parameters(): param.requires_grad = False
        else:
            # Stage 3: train all
            stage = 3
            for param in model.parameters(): param.requires_grad = True

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            rl, rc = 0.0, 0
            loss1_accum, loss2_accum, loss3_accum, loss_concat_accum = 0.0, 0.0, 0.0, 0.0
            n_batches = 0

            for x, y in loaders[phase]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    out, features = model(x, return_features=True)

                    # Multi-task loss from all granularities
                    l1 = crit(features['out1'], y) if stage >= 1 else torch.tensor(0.0, device=device)
                    l2 = crit(features['out2'], y) if stage >= 2 else torch.tensor(0.0, device=device)
                    l3 = crit(features['out3'], y) if stage >= 1 else torch.tensor(0.0, device=device)
                    l_concat = crit(out, y)

                    # Weighted combination
                    loss = l_concat + 0.5 * l1 + 0.5 * l2 + 0.5 * l3

                    _, p = torch.max(out, 1)

                    if phase == 'train':
                        loss.backward()
                        opt.step()

                rl += loss.item() * x.size(0)
                rc += (p == y).sum().item()
                loss1_accum += l1.item() if isinstance(l1, torch.Tensor) else 0
                loss2_accum += l2.item() if isinstance(l2, torch.Tensor) else 0
                loss3_accum += l3.item() if isinstance(l3, torch.Tensor) else 0
                loss_concat_accum += l_concat.item()
                n_batches += 1

            if phase == 'train':
                sched.step()

            el = rl / len(loaders[phase].dataset)
            ea = rc / len(loaders[phase].dataset)
            hist[f'{phase}_loss'].append(el)
            hist[f'{phase}_acc'].append(ea)
            hist['loss1'].append(loss1_accum / n_batches if stage >= 1 else 0)
            hist['loss2'].append(loss2_accum / n_batches if stage >= 2 else 0)
            hist['loss3'].append(loss3_accum / n_batches if stage >= 1 else 0)
            hist['loss_concat'].append(loss_concat_accum / n_batches)

            if phase == 'val' and ea > best_acc:
                best_acc = ea
                best_wts = copy.deepcopy(model.state_dict())

        print(f'  Ep {ep+1}/{epochs} [Stage {stage}] | TrL={hist["train_loss"][-1]:.4f} TrA={hist["train_acc"][-1]:.4f} | VaL={hist["val_loss"][-1]:.4f} VaA={hist["val_acc"][-1]:.4f}', flush=True)

    model.load_state_dict(best_wts)
    return model, hist, best_acc

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.extend(torch.max(out, 1)[1].cpu().numpy())
            labels.extend(y.numpy())
    return np.array(preds), np.array(labels)

def run_pmg(df, device, epochs=20):
    """Run PMG experiment on Knots-10 dataset"""
    print(f'\n{"="*50}\n  Running: PMG (Progressive Multi-Granularity)\n{"="*50}', flush=True)

    tr_full = df[df['split'] == 'train']
    te_df = df[df['split'] == 'test']
    tr_df, va_df = train_test_split(
        tr_full, test_size=0.2, stratify=tr_full['label'], random_state=SEED
    )

    tr_tf, te_tf = get_transforms()
    loaders = {
        'train': DataLoader(KnotDataset(tr_df, tr_tf), batch_size=32, shuffle=True),
        'val': DataLoader(KnotDataset(va_df, te_tf), batch_size=32)
    }
    test_loader = DataLoader(KnotDataset(te_df, te_tf), batch_size=32)

    model = PMGModel(num_classes=len(CLASSES)).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params: {total_params/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M', flush=True)

    t0 = time.time()
    model, hist, best_val = train_pmg(model, loaders, device, epochs=epochs)
    train_time = time.time() - t0

    preds, labels = evaluate(model, test_loader, device)
    report = classification_report(labels, preds, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds)
    test_acc = (preds == labels).mean()
    macro_f1 = report['macro avg']['f1-score']

    print(f'  Test Acc: {test_acc:.4f} | Best Val: {best_val:.4f} | Time: {train_time:.0f}s', flush=True)
    print(classification_report(labels, preds, target_names=CLASSES, zero_division=0), flush=True)

    return {
        'model': 'PMG',
        'params': f'{total_params/1e6:.2f}M',
        'val_acc': float(best_val),
        'test_acc': float(test_acc),
        'macro_f1': float(macro_f1),
        'train_time': float(train_time),
        'history': hist,
        'report': report,
        'confusion_matrix': cm.tolist(),
        'preds': preds.tolist(),
        'labels': labels.tolist()
    }

if __name__ == '__main__':
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print(f'Device: {device}', flush=True)

    df = parse_data(DATA_DIR)
    print(f'Total: {len(df)} (train: {len(df[df.split=="train"])}, test: {len(df[df.split=="test"])})', flush=True)

    # Run PMG
    res = run_pmg(df, device, epochs=20)
    with open(f'{RESULTS_DIR}/pmg_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('[Saved] pmg_results.json', flush=True)

    print('\n[DONE] PMG experiment complete.', flush=True)
