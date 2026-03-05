#!/usr/bin/env python3
"""
Ablation Study: Cross-condition evaluation.
1. Lighting ablation: train on 2 lighting, test on held-out lighting
2. Tightness ablation: train on single tightness, test on Set
3. Combined factor analysis
"""
import os, sys, json, time, copy, random, glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
NUM_CLASSES = len(CLASSES)
SEED = 42
RESULTS_DIR = 'results'
DATA_DIR = './train'

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

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
        except: img = Image.new('RGB', (224,224))
        if self.transform: img = self.transform(img)
        return img, torch.tensor(row['label'], dtype=torch.long)

def parse_all_data(data_dir):
    """Parse all images with full metadata."""
    files = glob.glob(os.path.join(data_dir, '**', '*.jpg'),
                      recursive=True)
    c2i = {c:i for i,c in enumerate(CLASSES)}
    rows = []
    for f in files:
        fn = os.path.basename(f)
        parts = fn.split('_')
        if parts[0] not in c2i:
            continue
        # Parse tightness
        if 'VeryLoose' in fn:
            tight = 'VeryLoose'
        elif 'Loose' in fn:
            tight = 'Loose'
        elif 'Set' in fn:
            tight = 'Set'
        else:
            continue
        # Parse lighting
        if '_DL_' in fn:
            light = 'DL'
        elif '_SLA_' in fn:
            light = 'SLA'
        elif '_SLS_' in fn:
            light = 'SLS'
        else:
            continue
        rows.append({
            'path': f,
            'label': c2i[parts[0]],
            'class': parts[0],
            'light': light,
            'tightness': tight
        })
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

def train_and_eval(train_df, test_df, tag, epochs=20):
    """Train ResNet-18 and evaluate."""
    set_seed(SEED)
    device = get_device()
    print(f"\n--- {tag} ---")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    tr_df, val_df = train_test_split(
        train_df, test_size=0.2,
        stratify=train_df['label'],
        random_state=SEED)
    tr_tf, te_tf = get_transforms()

    tr_loader = DataLoader(
        KnotDataset(tr_df, tr_tf), batch_size=32, shuffle=True)
    val_loader = DataLoader(
        KnotDataset(val_df, te_tf), batch_size=32)
    test_loader = DataLoader(
        KnotDataset(test_df, te_tf), batch_size=32)

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, NUM_CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    best_val, best_state = 0, None
    for ep in range(1, epochs+1):
        model.train()
        correct, total = 0, 0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1)==labels).sum().item()
            total += imgs.size(0)
        scheduler.step()
        tr_acc = correct/total

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                vc += (model(imgs).argmax(1)==labels).sum().item()
                vt += imgs.size(0)
        val_acc = vc/vt
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
        if ep % 5 == 0:
            print(f"  Ep {ep}/{epochs} TrA={tr_acc:.4f} VaA={val_acc:.4f}")

    # Test
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_acc = np.mean(
        np.array(all_preds)==np.array(all_labels))
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    f1 = report['macro avg']['f1-score']
    print(f"  Test={test_acc:.4f} Val={best_val:.4f} F1={f1:.4f}")
    return {
        'tag': tag, 'test_acc': test_acc,
        'best_val': best_val, 'f1': f1,
        'report': report, 'cm': cm.tolist()
    }

# ============================================
# Main: Ablation Experiments
# ============================================
if __name__ == '__main__':
    set_seed(SEED)
    df = parse_all_data(DATA_DIR)
    print(f"Total images: {len(df)}")
    print(f"Tightness: {df['tightness'].value_counts().to_dict()}")
    print(f"Lighting:  {df['light'].value_counts().to_dict()}")

    test_df = df[df['tightness']=='Set'].reset_index(drop=True)
    loose_df = df[df['tightness'].isin(
        ['Loose','VeryLoose'])].reset_index(drop=True)

    results = []

    # ---- Ablation 1: Tightness ----
    print("\n" + "="*50)
    print("ABLATION 1: Tightness Effect")
    print("="*50)

    # 1a. Baseline: Loose+VeryLoose → Set
    r = train_and_eval(loose_df, test_df,
                       "Loose+VeryLoose → Set")
    results.append(r)

    # 1b. Only Loose → Set
    only_loose = df[df['tightness']=='Loose'].reset_index(drop=True)
    r = train_and_eval(only_loose, test_df,
                       "OnlyLoose → Set")
    results.append(r)

    # 1c. Only VeryLoose → Set
    only_vl = df[df['tightness']=='VeryLoose'].reset_index(drop=True)
    r = train_and_eval(only_vl, test_df,
                       "OnlyVeryLoose → Set")
    results.append(r)

    # ---- Ablation 2: Lighting ----
    print("\n" + "="*50)
    print("ABLATION 2: Lighting Effect")
    print("="*50)

    for held_out in ['DL', 'SLA', 'SLS']:
        train_lights = [l for l in ['DL','SLA','SLS']
                        if l != held_out]
        tr = loose_df[loose_df['light'].isin(
            train_lights)].reset_index(drop=True)
        te = test_df[test_df['light']==held_out
                     ].reset_index(drop=True)
        r = train_and_eval(
            tr, te,
            f"Train({'+'.join(train_lights)}) → "
            f"Test({held_out})")
        results.append(r)

    # ---- Ablation 3: Data size effect ----
    print("\n" + "="*50)
    print("ABLATION 3: Training Size Effect")
    print("="*50)

    for frac in [0.25, 0.5, 0.75, 1.0]:
        if frac < 1.0:
            sub = loose_df.groupby('label').apply(
                lambda x: x.sample(
                    frac=frac, random_state=SEED)
            ).reset_index(drop=True)
        else:
            sub = loose_df
        r = train_and_eval(
            sub, test_df,
            f"Size={frac:.0%} ({len(sub)} imgs)")
        results.append(r)

    # ---- Summary ----
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['tag']:40s} | "
              f"Test={r['test_acc']:.4f} "
              f"F1={r['f1']:.4f}")
    print("="*60)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fp = f'{RESULTS_DIR}/ablation_results.json'
    with open(fp, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {fp}")
    print("[ALL DONE]")
