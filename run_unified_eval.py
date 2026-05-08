#!/usr/bin/env python3
"""
S0: Unified Evaluation Pipeline with Cross-Group Paired Statistics

Fixes: "Cross-group comparisons were omitted because different sample orderings"

All models are evaluated on the SAME canonical test DataLoader (sorted by path)
so per-sample pairing is valid for any model pair, including general vs FGVC.

Modes:
  --retrain   Retrain all 5 general-purpose models and save checkpoints.
              FGVC models (TransFG, PMG, Graph-FGVC) are imported from
              their existing scripts if checkpoints are provided.

  (default)   Evaluate whatever checkpoints exist, run McNemar on all pairs.

Usage:
  python run_unified_eval.py --data_dir ../train --ckpt_dir ../checkpoints
  python run_unified_eval.py --data_dir ../train --ckpt_dir ../checkpoints --retrain --seed 42
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from knot_utils import (
    CLASSES, NUM_CLASSES, C2I, EMBED_DIMS,
    set_seed, get_device, parse_data, get_transforms,
    KnotDataset, get_canonical_test_loader, get_train_val_loaders,
    make_model, EmbeddingModel, _make_backbone,
    train_one_epoch, evaluate, mcnemar_test, hard_pair_accuracy,
    save_checkpoint, load_checkpoint, HARD_PAIRS,
)
from torch.utils.data import DataLoader

# ── FGVC model imports (from existing scripts) ──────────────────────────────
def _import_transfg():
    from run_transfg import TransFGModel
    return TransFGModel

def _import_pmg():
    from run_pmg import PMGModel
    return PMGModel

def _import_graph_fgvc():
    from run_graph_fgvc import GraphFGVCModel
    return GraphFGVCModel


# ── Model registry ──────────────────────────────────────────────────────────
GENERAL_MODELS = ['resnet18', 'resnet50', 'efficientnet_b0', 'vit', 'swin_t']
FGVC_MODELS = ['transfg', 'pmg', 'graph_fgvc']
ALL_MODELS = GENERAL_MODELS + FGVC_MODELS


def create_model(name, device):
    """Create model by name; returns (model, needs_special_eval)."""
    if name in GENERAL_MODELS:
        return make_model(name, with_embeddings=False).to(device), False
    elif name == 'transfg':
        cls = _import_transfg()
        return cls(num_classes=NUM_CLASSES).to(device), True
    elif name == 'pmg':
        cls = _import_pmg()
        return cls(num_classes=NUM_CLASSES).to(device), True
    elif name == 'graph_fgvc':
        cls = _import_graph_fgvc()
        return cls(num_classes=NUM_CLASSES).to(device), True
    raise ValueError(f"Unknown model: {name}")


def find_checkpoint(name, ckpt_dir):
    """Look for checkpoint file in ckpt_dir with common naming patterns."""
    patterns = [
        f"{name}_best.pth",
        f"{name}_baseline.pth",
        f"{name}.pth",
        f"{name}_results.pth",
    ]
    for p in patterns:
        path = os.path.join(ckpt_dir, p)
        if os.path.exists(path):
            return path
    return None


# ── Retraining (general models only) ────────────────────────────────────────
def retrain_general(name, data_dir, ckpt_dir, seed, device, epochs=20):
    """Retrain a general-purpose model and save checkpoint."""
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"  Retraining {name} (seed={seed}, epochs={epochs})")
    print(f"{'='*60}")

    tr_loader, val_loader = get_train_val_loaders(
        data_dir, seed=seed, batch_size=32, sz=224)

    model = make_model(name, with_embeddings=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        loss, tr_acc = train_one_epoch(model, tr_loader, optimizer, criterion, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == epochs:
            print(f"  Epoch {ep:3d}  loss={loss:.4f}  train={tr_acc:.4f}  val={val_acc:.4f}")

    model.load_state_dict(best_state)
    ckpt_path = os.path.join(ckpt_dir, f"{name}_best.pth")
    save_checkpoint(model, ckpt_path)
    print(f"  Saved: {ckpt_path}  (best val={best_val_acc:.4f})")
    return model


# ── Main evaluation ─────────────────────────────────────────────────────────
def run_evaluation(data_dir, ckpt_dir, retrain, seed, device, epochs):
    """Evaluate all models on canonical test set, run all-pairs McNemar."""
    os.makedirs('results', exist_ok=True)
    test_loader, test_df = get_canonical_test_loader(data_dir, batch_size=64)
    test_labels = test_df['label'].values
    n_test = len(test_df)
    print(f"\nCanonical test set: {n_test} samples, sorted by path")
    print(f"Label distribution: {np.bincount(test_labels, minlength=NUM_CLASSES).tolist()}")

    # ── Collect predictions ──────────────────────────────────────────────
    results = {}  # name -> {acc, preds, group}

    for name in ALL_MODELS:
        group = 'general' if name in GENERAL_MODELS else 'fgvc'
        ckpt_path = find_checkpoint(name, ckpt_dir)

        if retrain and name in GENERAL_MODELS:
            model = retrain_general(name, data_dir, ckpt_dir, seed, device, epochs)
            model.to(device)
        elif ckpt_path:
            print(f"\nLoading {name} from {ckpt_path}")
            model, _ = create_model(name, device)
            try:
                load_checkpoint(model, ckpt_path, device)
            except Exception as e:
                print(f"  WARNING: Could not load {name}: {e}")
                continue
        else:
            print(f"\nSKIPPING {name}: no checkpoint in {ckpt_dir}")
            continue

        acc, preds, labels = evaluate(model, test_loader, device)
        assert np.array_equal(labels, test_labels), \
            f"Label mismatch for {name}! Canonical ordering violated."
        results[name] = {'acc': float(acc), 'preds': preds.tolist(), 'group': group}
        print(f"  {name:20s}  acc={acc:.4f}  group={group}")

    if len(results) < 2:
        print("\nNeed at least 2 models for McNemar tests. Exiting.")
        return results

    # ── All-pairs McNemar ────────────────────────────────────────────────
    model_names = sorted(results.keys())
    n_pairs = len(model_names) * (len(model_names) - 1) // 2
    bonferroni_alpha = 0.05 / max(n_pairs, 1)

    print(f"\n{'='*80}")
    print(f"  McNemar Paired Tests (n={n_test}, Bonferroni α={bonferroni_alpha:.6f})")
    print(f"{'='*80}")
    print(f"{'Model A':>22s}  {'Model B':>22s}  {'b':>4s}  {'c':>4s}  {'χ²':>7s}  {'p':>10s}  Sig")
    print(f"{'-'*80}")

    mcnemar_results = []
    for i, ma in enumerate(model_names):
        for mb in model_names[i + 1:]:
            pa = np.array(results[ma]['preds'])
            pb = np.array(results[mb]['preds'])
            res = mcnemar_test(pa, pb, test_labels)
            sig = '***' if res['p'] < 0.001 else ('**' if res['p'] < 0.01 else
                  ('*' if res['p'] < bonferroni_alpha else 'n.s.'))
            cross = '(cross)' if results[ma]['group'] != results[mb]['group'] else ''
            print(f"{ma:>22s}  {mb:>22s}  {res['b']:4d}  {res['c']:4d}  "
                  f"{res['chi2']:7.2f}  {res['p']:10.6f}  {sig:4s} {cross}")
            mcnemar_results.append({
                'model_a': ma, 'model_b': mb,
                'group_a': results[ma]['group'], 'group_b': results[mb]['group'],
                **res, 'sig': sig,
            })

    # ── Hard-pair analysis ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Hard-Pair Accuracy")
    print(f"{'='*60}")
    hp_header = f"{'Model':>20s}"
    for ca, cb in HARD_PAIRS:
        hp_header += f"  {ca}-{cb:>5s}"
    print(hp_header)
    print('-' * len(hp_header))

    hard_pair_results = {}
    for name in model_names:
        pa = np.array(results[name]['preds'])
        row = f"{name:>20s}"
        hp_row = {}
        for ca, cb in HARD_PAIRS:
            hp_acc = hard_pair_accuracy(pa, test_labels, ca, cb)
            row += f"  {hp_acc:8.3f}"
            hp_row[f"{ca}-{cb}"] = hp_acc
        print(row)
        hard_pair_results[name] = hp_row

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        'n_test': n_test,
        'test_paths': test_df['path'].tolist(),
        'test_labels': test_labels.tolist(),
        'models': {k: {'acc': v['acc'], 'group': v['group'], 'preds': v['preds']}
                   for k, v in results.items()},
        'mcnemar': mcnemar_results,
        'hard_pairs': hard_pair_results,
        'bonferroni_alpha': bonferroni_alpha,
    }
    out_path = 'results/unified_eval.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='S0: Unified Evaluation Pipeline')
    parser.add_argument('--data_dir', type=str, default='../train')
    parser.add_argument('--ckpt_dir', type=str, default='../checkpoints')
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain general-purpose models before evaluation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    run_evaluation(args.data_dir, args.ckpt_dir, args.retrain,
                   args.seed, device, args.epochs)


if __name__ == '__main__':
    main()
