#!/usr/bin/env python3
"""
Single-config single-seed runner for distributed execution.

Usage:
    python run_single.py --model resnet18 --lam_taca 0 --lam_taml 0 --seed 42
    python run_single.py --model swin_t --lam_taca 0 --lam_taml 0 --seed 123 --device cuda
    python run_single.py --model resnet18 --lam_taca 0.1 --lam_taml 0.005 --seed 456 --device mps

Output: results/single_{model}_{lam_taca}_{lam_taml}_seed{N}.json
"""
import argparse
import os
import json
import time

from robustness_multiseed import (
    TOPO_DIST, CLASSES, NUM_CLASSES, SEEDS,
    set_seed, get_device, parse_data, get_transforms,
    KnotDataset, make_model, TopoLoss,
)

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_and_eval(model_name, df, device, seed, lam_taca=0.0, lam_taml=0.0, epochs=20):
    """Train one model with one seed. Returns dict of metrics."""
    set_seed(seed)
    tr_full = df[df['split'] == 'train']
    te_df = df[df['split'] == 'test']
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

    for ep in range(1, epochs + 1):
        model.train()
        for imgs, labs in tr_ld:
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            if use_topo:
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
                vc += (model(imgs).argmax(1) == labs).sum().item()
                vt += imgs.size(0)
        val_acc = vc / vt
        if val_acc > best_val:
            best_val = val_acc
            best_st = copy.deepcopy(model.state_dict())

        elapsed = time.time() - t0
        print(f"  Epoch {ep:2d}/{epochs}  val_acc={val_acc:.4f}  "
              f"best={best_val:.4f}  elapsed={elapsed:.0f}s", flush=True)

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
    rep = classification_report(labs_list, preds, target_names=CLASSES,
                                output_dict=True, zero_division=0)
    f1 = rep['macro avg']['f1-score']

    return {
        'val_acc': best_val,
        'test_acc': float(test_acc),
        'f1': float(f1),
        'time': train_time,
    }


def main():
    parser = argparse.ArgumentParser(description='Run single knot experiment')
    parser.add_argument('--model', required=True,
                        choices=['resnet18', 'resnet50', 'efficientnet_b0', 'vit', 'swin_t'])
    parser.add_argument('--lam_taca', type=float, default=0.0)
    parser.add_argument('--lam_taml', type=float, default=0.0)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', type=str, default=None,
                        help='Force device (cuda/mps/cpu). Auto-detect if omitted.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default='./train')
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    df = parse_data(args.data_dir)
    print(f"Device: {device}")
    print(f"Data: {len(df)} images")
    print(f"Config: model={args.model} lam_taca={args.lam_taca} "
          f"lam_taml={args.lam_taml} seed={args.seed} epochs={args.epochs}")
    print(f"{'=' * 60}")

    t0 = time.time()
    result = train_and_eval(
        args.model, df, device, args.seed,
        args.lam_taca, args.lam_taml, args.epochs)
    wall_time = time.time() - t0

    # Build output
    output = {
        'model': args.model,
        'lam_taca': args.lam_taca,
        'lam_taml': args.lam_taml,
        'seed': args.seed,
        'epochs': args.epochs,
        'device': str(device),
        'val_acc': result['val_acc'],
        'test_acc': result['test_acc'],
        'f1': result['f1'],
        'train_time': result['time'],
        'wall_time': wall_time,
    }

    fname = (f"single_{args.model}_taca{args.lam_taca}_taml{args.lam_taml}"
             f"_seed{args.seed}.json")
    out_path = os.path.join(args.results_dir, fname)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE  val={result['val_acc']:.4f}  test={result['test_acc']:.4f}  "
          f"f1={result['f1']:.4f}  time={wall_time:.0f}s")
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
