#!/usr/bin/env python3
"""Run additional baseline models: ResNet-50, EfficientNet-B0, Swin-T"""
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

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
SEED = 42
RESULTS_DIR = 'results'

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
        rows.append({'path':f, 'label':c2i[parts[0]], 'split':sp})
    return pd.DataFrame(rows)

def get_transforms(sz=224):
    norm = ([0.485,0.456,0.406],[0.229,0.224,0.225])
    tr = transforms.Compose([
        transforms.Resize((sz,sz)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1,0.1),
        transforms.ToTensor(),
        transforms.Normalize(*norm)])
    te = transforms.Compose([
        transforms.Resize((sz,sz)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)])
    return tr, te

def make_model(name, nc, device):
    if name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, nc))
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
    elif name == 'swin_t':
        m = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        m.head = nn.Linear(m.head.in_features, nc)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m.to(device)

def train_model(model, loaders, device, epochs=20, lr=1e-4):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    hist = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    for ep in range(epochs):
        for phase in ['train','val']:
            model.train() if phase=='train' else model.eval()
            rl, rc = 0.0, 0
            for x, y in loaders[phase]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    out = model(x)
                    _, p = torch.max(out,1)
                    loss = crit(out,y)
                    if phase=='train':
                        loss.backward()
                        opt.step()
                rl += loss.item()*x.size(0)
                rc += (p==y).sum().item()
            if phase=='train': sched.step()
            el = rl/len(loaders[phase].dataset)
            ea = rc/len(loaders[phase].dataset)
            hist[f'{phase}_loss'].append(el)
            hist[f'{phase}_acc'].append(ea)
            if phase=='val' and ea > best_acc:
                best_acc = ea
                best_wts = copy.deepcopy(model.state_dict())
        print(f'  Ep {ep+1}/{epochs} | TrA={hist["train_acc"][-1]:.4f} VaA={hist["val_acc"][-1]:.4f}', flush=True)
    model.load_state_dict(best_wts)
    return model, hist, best_acc

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.extend(torch.max(out,1)[1].cpu().numpy())
            labels.extend(y.numpy())
    return np.array(preds), np.array(labels)

def run_one(name, df, device, epochs=20):
    print(f'\n{"="*50}\n  {name.upper()}\n{"="*50}', flush=True)
    tr_full = df[df['split']=='train']
    te_df = df[df['split']=='test']
    tr_df, va_df = train_test_split(
        tr_full, test_size=0.2,
        stratify=tr_full['label'],
        random_state=SEED)
    tr_tf, te_tf = get_transforms()
    loaders = {
        'train': DataLoader(KnotDataset(tr_df, tr_tf), batch_size=32, shuffle=True),
        'val': DataLoader(KnotDataset(va_df, te_tf), batch_size=32)
    }
    tl = DataLoader(KnotDataset(te_df, te_tf), batch_size=32)
    model = make_model(name, len(CLASSES), device)
    t0 = time.time()
    model, hist, bv = train_model(model, loaders, device, epochs)
    tt = time.time() - t0
    preds, labels = evaluate(model, tl, device)
    rpt = classification_report(labels, preds, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds)
    ta = (preds==labels).mean()
    print(f'  Test={ta:.4f} BestVal={bv:.4f} Time={tt:.0f}s', flush=True)
    print(classification_report(labels, preds, target_names=CLASSES, zero_division=0), flush=True)
    return {'model':name, 'history':hist, 'best_val_acc':bv,
            'test_acc':float(ta), 'train_time':tt,
            'report':rpt, 'confusion_matrix':cm.tolist(),
            'preds':preds.tolist(), 'labels':labels.tolist()}

if __name__ == '__main__':
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print(f'Device: {device}', flush=True)
    df = parse_data('./train')
    print(f'Total: {len(df)}', flush=True)

    for name in ['resnet50', 'efficientnet_b0', 'swin_t']:
        set_seed(SEED)
        res = run_one(name, df, device, epochs=20)
        out = f'{RESULTS_DIR}/{name}_results.json'
        with open(out, 'w') as f:
            json.dump(res, f, indent=2)
        print(f'[Saved] {out}', flush=True)

    print('\n[DONE] All extra models complete.', flush=True)
