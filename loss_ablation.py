#!/usr/bin/env python3
"""Loss Ablation: CE / CE+TALS / CE+TAML / CE+TALS+TAML"""
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
SEED = 42
DATA_DIR = './train'

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

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

class KnotDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df; self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, torch.tensor(row['label'], dtype=torch.long)

def parse_data(data_dir, test=False):
    import pandas as pd
    files = glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True)
    c2i = {c:i for i,c in enumerate(CLASSES)}
    rows = []
    for f in files:
        fn = os.path.basename(f)
        parts = fn.split('_')
        if parts[0] not in c2i: continue
        is_set = 'Set' in fn
        if test and not is_set: continue
        if not test and is_set: continue
        rows.append({'path': f, 'label': c2i[parts[0]]})
    return pd.DataFrame(rows)

def get_transforms():
    norm = ([0.485,0.456,0.406],[0.229,0.224,0.225])
    tr = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.1,0.1),
        transforms.ToTensor(),
        transforms.Normalize(*norm)])
    te = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)])
    return tr, te

class TopoLoss(nn.Module):
    def __init__(self, topo_dist, lam_tals=0.0, lam_taml=0.0):
        super().__init__()
        self.topo = torch.tensor(topo_dist, dtype=torch.float32)
        self.lam_tals = lam_tals
        self.lam_taml = lam_taml
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels, emb=None):
        dev = logits.device
        self.topo = self.topo.to(dev)
        ce = self.ce(logits, labels)
        total = ce
        tals = torch.tensor(0.0, device=dev)
        taml = torch.tensor(0.0, device=dev)
        
        if self.lam_tals > 0 and emb is not None:
            cents = []
            for c in range(NUM_CLASSES):
                mask = (labels == c)
                if mask.sum() > 0:
                    cents.append(emb[mask].mean(0))
                else:
                    cents.append(torch.zeros_like(emb[0]))
            cents = torch.stack(cents)
            # Manual pairwise L2 distance (MPS-compatible, avoids cdist backward)
            diff = cents.unsqueeze(0) - cents.unsqueeze(1)
            edist = torch.sqrt((diff ** 2).sum(-1) + 1e-12)
            edist = edist / (edist.max() + 1e-8)
            tals = nn.functional.mse_loss(edist, self.topo)
            total = total + self.lam_tals * tals
        
        if self.lam_taml > 0 and emb is not None:
            enorm = nn.functional.normalize(emb, dim=1)
            sim = torch.mm(enorm, enorm.t())
            mloss = []
            for i in range(emb.size(0)):
                yi = labels[i]
                for j in range(emb.size(0)):
                    if i == j: continue
                    yj = labels[j]
                    if yi == yj: continue
                    margin = self.topo[yi, yj]
                    mloss.append(torch.relu(sim[i,j] - (1.0 - margin)))
            if len(mloss) > 0:
                taml = torch.stack(mloss).mean()
                total = total + self.lam_taml * taml
        
        return total, ce, tals, taml

def train_model(tr_df, val_df, te_df, name, lam_tals, lam_taml, epochs=20):
    set_seed(SEED)
    dev = get_device()
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    
    tr_tf, te_tf = get_transforms()
    tr_ld = DataLoader(KnotDataset(tr_df, tr_tf), batch_size=32, shuffle=True)
    val_ld = DataLoader(KnotDataset(val_df, te_tf), batch_size=32)
    te_ld = DataLoader(KnotDataset(te_df, te_tf), batch_size=32)
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, NUM_CLASSES))
    model = model.to(dev)
    
    crit = TopoLoss(TOPO_DIST, lam_tals, lam_taml)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    best_val, best_st = 0.0, None
    t0 = time.time()
    
    for ep in range(1, epochs+1):
        model.train()
        cor, tot = 0, 0
        for imgs, labs in tr_ld:
            imgs, labs = imgs.to(dev), labs.to(dev)
            opt.zero_grad()
            feat = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(
                model.maxpool(model.relu(model.bn1(model.conv1(imgs)))))))))
            emb = feat.view(feat.size(0), -1)
            logits = model.fc(emb)
            loss, _, _, _ = crit(logits, labs, emb)
            loss.backward()
            opt.step()
            cor += (logits.argmax(1)==labs).sum().item()
            tot += imgs.size(0)
        sch.step()
        tr_acc = cor/tot
        
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, labs in val_ld:
                imgs, labs = imgs.to(dev), labs.to(dev)
                logits = model(imgs)
                vc += (logits.argmax(1)==labs).sum().item()
                vt += imgs.size(0)
        val_acc = vc/vt
        if val_acc > best_val:
            best_val = val_acc
            best_st = copy.deepcopy(model.state_dict())
        if ep % 5 == 0:
            print(f"  Ep {ep}/{epochs} Tr={tr_acc:.4f} Val={val_acc:.4f}")
    
    tt = time.time() - t0
    
    model.load_state_dict(best_st)
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for imgs, l in te_ld:
            imgs, l = imgs.to(dev), l.to(dev)
            p = model(imgs).argmax(1)
            preds.extend(p.cpu().numpy())
            labs.extend(l.cpu().numpy())
    
    te_acc = np.mean(np.array(preds)==np.array(labs))
    rep = classification_report(labs, preds, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(labs, preds)
    
    print(f"  Val={best_val:.4f} Test={te_acc:.4f} F1={rep['macro avg']['f1-score']:.4f} Time={tt:.1f}s")
    
    return {
        'config': name, 'lam_tals': lam_tals, 'lam_taml': lam_taml,
        'val_acc': best_val, 'test_acc': te_acc,
        'f1': rep['macro avg']['f1-score'], 'time': tt,
        'report': rep, 'cm': cm.tolist()
    }

if __name__ == '__main__':
    set_seed(SEED)
    tr_df = parse_data(DATA_DIR, test=False)
    te_df = parse_data(DATA_DIR, test=True)
    tr_df, val_df = train_test_split(tr_df, test_size=0.2, stratify=tr_df['label'], random_state=SEED)
    print(f"Train={len(tr_df)} Val={len(val_df)} Test={len(te_df)}")
    
    configs = [
        ('CE_only', 0.0, 0.0),
        ('CE+TALS', 0.1, 0.0),
        ('CE+TAML', 0.0, 0.005),
        ('CE+TALS+TAML', 0.1, 0.005)
    ]
    
    results = []
    for name, lt, lm in configs:
        r = train_model(tr_df, val_df, te_df, name, lt, lm, epochs=20)
        results.append(r)
    
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    for r in results:
        print(f"{r['config']:<20} Val={r['val_acc']:.4f} Test={r['test_acc']:.4f} F1={r['f1']:.4f} Time={r['time']:.1f}s")
    
    os.makedirs('results', exist_ok=True)
    with open('results/loss_ablation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n[Saved] results/loss_ablation.json")
