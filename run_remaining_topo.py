#!/usr/bin/env python3
"""Run remaining topology-guided experiments (EfficientNet-B0, Swin-T)."""
import os, sys, json, time, copy, random, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
SEED = 42; RESULTS_DIR = 'results'; DATA_DIR = './train'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topo_guided_training import (
    set_seed, get_device, KnotDataset, parse_data, get_transforms,
    build_topo_distance_matrix, make_topo_model, TopologyGuidedLoss,
    train_topo_guided, evaluate, NumpyEncoder
)

def run_one(model_name, df, topo_dist, device, lt=0.1, lm=0.05, epochs=20):
    fname = f'{RESULTS_DIR}/{model_name}_topo_guided_results.json'
    if os.path.exists(fname):
        print(f'[SKIP] {fname} exists', flush=True)
        return
    print(f'\n{"="*60}\n  Topology-Guided: {model_name.upper()}\n{"="*60}', flush=True)
    tr_full = df[df['split']=='train']; te_df = df[df['split']=='test']
    tr_df, va_df = train_test_split(tr_full, test_size=0.2, stratify=tr_full['label'], random_state=SEED)
    tr_tf, te_tf = get_transforms()
    loaders = {
        'train': DataLoader(KnotDataset(tr_df, tr_tf), batch_size=32, shuffle=True),
        'val': DataLoader(KnotDataset(va_df, te_tf), batch_size=32)
    }
    test_loader = DataLoader(KnotDataset(te_df, te_tf), batch_size=32)
    model = make_topo_model(model_name, len(CLASSES), device)
    criterion = TopologyGuidedLoss(topo_dist, lambda_topo=lt, lambda_margin=lm, device=device)
    t0 = time.time()
    model, hist, best_val = train_topo_guided(model, loaders, criterion, device, epochs)
    train_time = time.time() - t0
    preds, labels, embs = evaluate(model, test_loader, device)
    report = classification_report(labels, preds, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds)
    test_acc = (preds == labels).mean()
    print(f'  Test Acc: {test_acc:.4f} | Best Val: {best_val:.4f} | Time: {train_time:.0f}s', flush=True)
    print(classification_report(labels, preds, target_names=CLASSES, zero_division=0), flush=True)
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/{model_name}_topo_guided.pth')
    res = {'model': model_name, 'method': 'topology_guided',
           'lambda_topo': lt, 'lambda_margin': lm, 'history': hist,
           'best_val_acc': best_val, 'test_acc': float(test_acc),
           'train_time': train_time, 'report': report,
           'confusion_matrix': cm.tolist()}
    with open(fname, 'w') as f:
        json.dump(res, f, indent=2, cls=NumpyEncoder)
    print(f'[Saved] {fname}', flush=True)

if __name__ == '__main__':
    set_seed(SEED)
    device = get_device()
    df = parse_data(DATA_DIR)
    topo_dist = build_topo_distance_matrix()
    print(f'Device: {device} | Data: {len(df)}', flush=True)
    for mn in ['efficientnet_b0', 'swin_t']:
        set_seed(SEED)
        run_one(mn, df, topo_dist, device)
    print(f'\n{"="*60}\n  SUMMARY\n{"="*60}', flush=True)
    for mn in ['resnet18','resnet50','efficientnet_b0','swin_t']:
        tf = f'{RESULTS_DIR}/{mn}_topo_guided_results.json'
        bf = f'{RESULTS_DIR}/{mn}_results.json'
        if os.path.exists(tf) and os.path.exists(bf):
            ta = json.load(open(tf))['test_acc']
            ba = json.load(open(bf))['test_acc']
            print(f'  {mn:20s}: base={ba:.4f} topo={ta:.4f} d={((ta-ba)*100):+.2f}%', flush=True)
    print('[DONE]', flush=True)
