#!/usr/bin/env python3
"""Knot Classification Experiments - Compact Runner"""
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

def make_model(name, nc, device):
    if name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, nc))
    elif name == 'vit':
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        m.heads.head = nn.Linear(m.heads.head.in_features, nc)
    return m.to(device)

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

def run_experiment(model_name, df, device, epochs=20):
    print(f'\n{"="*50}\n  Running: {model_name.upper()}\n{"="*50}', flush=True)
    tr_full = df[df['split']=='train']; te_df = df[df['split']=='test']
    tr_df, va_df = train_test_split(tr_full, test_size=0.2, stratify=tr_full['label'], random_state=SEED)
    tr_tf, te_tf = get_transforms()
    loaders = {
        'train': DataLoader(KnotDataset(tr_df, tr_tf), batch_size=32, shuffle=True),
        'val': DataLoader(KnotDataset(va_df, te_tf), batch_size=32)
    }
    test_loader = DataLoader(KnotDataset(te_df, te_tf), batch_size=32)
    model = make_model(model_name, len(CLASSES), device)
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
        'model': model_name, 'history': hist, 'best_val_acc': best_val,
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

    # 1. ResNet18
    res = run_experiment('resnet18', df, device, epochs=20)
    with open(f'{RESULTS_DIR}/resnet18_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('[Saved] resnet18_results.json', flush=True)

    # 2. ViT
    set_seed(SEED)
    res = run_experiment('vit', df, device, epochs=20)
    with open(f'{RESULTS_DIR}/vit_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('[Saved] vit_results.json', flush=True)

    print('\n[DONE] All experiments complete.', flush=True)
