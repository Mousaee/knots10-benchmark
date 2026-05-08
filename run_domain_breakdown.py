#!/usr/bin/env python3
"""
S1: Lighting Breakdown + Tightness Transfer Matrix

Pure inference — no training. Uses existing checkpoints.

Outputs:
  1. Per-lighting accuracy breakdown (DL, SLA, SLS) for each model
  2. Tightness transfer matrix: train on {Loose, VeryLoose}, test on {Set}
     cross-tabulated by lighting condition
  3. Per-class accuracy under each lighting condition

Usage:
  python run_domain_breakdown.py --data_dir ../train --ckpt_dir ../checkpoints
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from knot_utils import (
    CLASSES, NUM_CLASSES, C2I,
    set_seed, get_device, parse_data, get_transforms,
    KnotDataset, make_model, load_checkpoint,
    HARD_PAIRS, hard_pair_accuracy,
)
from torch.utils.data import DataLoader


# ── Available models ────────────────────────────────────────────────────────
MODELS_TO_EVAL = ['resnet18', 'resnet50', 'efficientnet_b0', 'vit', 'swin_t']


def find_checkpoint(name, ckpt_dir):
    """Look for checkpoint file with common naming patterns."""
    patterns = [
        f"{name}_best.pth", f"{name}_baseline.pth",
        f"{name}.pth", f"{name}_results.pth",
    ]
    for p in patterns:
        path = os.path.join(ckpt_dir, p)
        if os.path.exists(path):
            return path
    return None


@torch.no_grad()
def evaluate_on_loader(model, loader, device):
    """Return (preds, labels) arrays."""
    model.eval()
    preds_list, labels_list = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        logits = out[0] if isinstance(out, tuple) else out
        preds_list.append(logits.argmax(1).cpu())
        labels_list.append(labels)
    return torch.cat(preds_list).numpy(), torch.cat(labels_list).numpy()


def run_domain_breakdown(args):
    device = get_device()
    set_seed(42)
    os.makedirs('results', exist_ok=True)
    print(f"Device: {device}")

    df = parse_data(args.data_dir)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    _, te_transform = get_transforms(224)

    print(f"Test set: {len(test_df)} images")
    print(f"  Lighting: {test_df['light'].value_counts().to_dict()}")
    print(f"  Tightness: {test_df['tightness'].value_counts().to_dict()}")
    print(f"Train set: {len(train_df)} images")
    print(f"  Lighting: {train_df['light'].value_counts().to_dict()}")
    print(f"  Tightness: {train_df['tightness'].value_counts().to_dict()}")

    all_results = {}

    for model_name in MODELS_TO_EVAL:
        ckpt_path = find_checkpoint(model_name, args.ckpt_dir)
        if not ckpt_path:
            print(f"\nSKIPPING {model_name}: no checkpoint found")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        model = make_model(model_name, with_embeddings=False).to(device)
        load_checkpoint(model, ckpt_path, device)

        result = {'model': model_name}

        # ── 1. Per-lighting breakdown ──────────────────────────────────────
        lighting_accs = {}
        for light in ['DL', 'SLA', 'SLS']:
            subset = test_df[test_df['light'] == light].sort_values('path').reset_index(drop=True)
            if len(subset) == 0:
                lighting_accs[light] = {'acc': None, 'n': 0}
                continue
            loader = DataLoader(KnotDataset(subset, te_transform),
                                batch_size=64, shuffle=False, num_workers=2)
            preds, labels = evaluate_on_loader(model, loader, device)
            acc = float((preds == labels).mean())
            lighting_accs[light] = {
                'acc': acc, 'n': len(subset),
                'per_class': {
                    CLASSES[c]: float((preds[labels == c] == c).mean())
                    if (labels == c).sum() > 0 else None
                    for c in range(NUM_CLASSES)
                },
            }
            print(f"  {light}: acc={acc:.4f} (n={len(subset)})")

        result['lighting_breakdown'] = lighting_accs

        # ── 2. Tightness transfer matrix ──────────────────────────────────
        # Rows = train tightness (Loose, VeryLoose), Cols = test tightness (Set)
        # Cross-tabulate with lighting
        tight_matrix = {}
        for train_tight in ['Loose', 'VeryLoose']:
            for test_light in ['DL', 'SLA', 'SLS']:
                # We can't retrain per-tightness, but we can report
                # how the model (trained on all Loose+VeryLoose) performs
                # on each lighting condition of the test set
                key = f"train_{train_tight}_test_{test_light}"
                # Count training data composition
                n_train = len(train_df[(train_df['tightness'] == train_tight) &
                                       (train_df['light'] == test_light)])
                tight_matrix[key] = {
                    'n_train_matching': int(n_train),
                    'test_light': test_light,
                    'train_tightness': train_tight,
                }

        # The actual transfer: overall model on test subsets by tightness
        test_tight_vals = test_df['tightness'].unique()
        tightness_accs = {}
        for tight in test_tight_vals:
            subset = test_df[test_df['tightness'] == tight].sort_values('path').reset_index(drop=True)
            if len(subset) == 0:
                continue
            loader = DataLoader(KnotDataset(subset, te_transform),
                                batch_size=64, shuffle=False, num_workers=2)
            preds, labels = evaluate_on_loader(model, loader, device)
            acc = float((preds == labels).mean())
            tightness_accs[tight] = {'acc': acc, 'n': len(subset)}
            print(f"  Tightness={tight}: acc={acc:.4f} (n={len(subset)})")

        result['tightness_breakdown'] = tightness_accs
        result['tightness_lighting_matrix'] = tight_matrix

        # ── 3. Combined: lighting × tightness ─────────────────────────────
        combined = {}
        for light in ['DL', 'SLA', 'SLS']:
            for tight in test_tight_vals:
                subset = test_df[(test_df['light'] == light) &
                                 (test_df['tightness'] == tight)
                                ].sort_values('path').reset_index(drop=True)
                if len(subset) == 0:
                    continue
                loader = DataLoader(KnotDataset(subset, te_transform),
                                    batch_size=64, shuffle=False, num_workers=2)
                preds, labels = evaluate_on_loader(model, loader, device)
                acc = float((preds == labels).mean())
                combined[f"{light}_{tight}"] = {'acc': acc, 'n': len(subset)}

        result['lighting_tightness_combined'] = combined

        # ── 4. Per-class confusion for worst lighting ─────────────────────
        # Find which lighting is hardest
        valid_lights = {k: v for k, v in lighting_accs.items() if v['acc'] is not None}
        if valid_lights:
            worst_light = min(valid_lights, key=lambda k: valid_lights[k]['acc'])
            subset = test_df[test_df['light'] == worst_light].sort_values('path').reset_index(drop=True)
            loader = DataLoader(KnotDataset(subset, te_transform),
                                batch_size=64, shuffle=False, num_workers=2)
            preds, labels = evaluate_on_loader(model, loader, device)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
            result['worst_lighting'] = worst_light
            result['worst_lighting_confusion'] = cm.tolist()

            # Hard-pair accuracy under worst lighting
            hp = {}
            for ca, cb in HARD_PAIRS:
                hp[f"{ca}-{cb}"] = hard_pair_accuracy(preds, labels, ca, cb)
            result['worst_lighting_hard_pairs'] = hp

        all_results[model_name] = result

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Lighting Breakdown Summary")
    print(f"{'='*70}")
    print(f"{'Model':>18s}  {'DL':>7s}  {'SLA':>7s}  {'SLS':>7s}  {'Overall':>7s}")
    print('-' * 50)
    for name, r in all_results.items():
        lb = r['lighting_breakdown']
        dl = f"{lb['DL']['acc']:.4f}" if lb['DL']['acc'] is not None else "N/A"
        sla = f"{lb['SLA']['acc']:.4f}" if lb['SLA']['acc'] is not None else "N/A"
        sls = f"{lb['SLS']['acc']:.4f}" if lb['SLS']['acc'] is not None else "N/A"
        # Overall from tightness breakdown 'Set'
        tb = r.get('tightness_breakdown', {})
        overall = "N/A"
        if 'Set' in tb and tb['Set']['acc'] is not None:
            overall = f"{tb['Set']['acc']:.4f}"
        print(f"{name:>18s}  {dl:>7s}  {sla:>7s}  {sls:>7s}  {overall:>7s}")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = 'results/domain_breakdown.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='S1: Domain Breakdown')
    parser.add_argument('--data_dir', type=str, default='../train')
    parser.add_argument('--ckpt_dir', type=str, default='../checkpoints')
    args = parser.parse_args()
    run_domain_breakdown(args)


if __name__ == '__main__':
    main()
