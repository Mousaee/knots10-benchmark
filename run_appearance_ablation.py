#!/usr/bin/env python3
"""
S2: Structure vs Appearance Ablation (Two-Layer)

Layer 1 — Diagnostic (pure inference, no retraining):
  Uses existing checkpoints to evaluate on 4 test conditions:
  (a) Original test images            — full appearance
  (b) Grayscale test images           — remove colour cue
  (c) Background-masked (rope only)   — remove background cue
  (d) Colour-histogram-equalized      — flatten per-class colour bias

Layer 2 — Intervention (requires training):
  Train a new model on background-masked images only, evaluate on:
  (a) Background-masked test set
  (b) Original test set (transfer)

Requires: rembg (pip install rembg) for background removal.
Falls back to simple thresholding if rembg is not available.

Usage:
  python run_appearance_ablation.py --data_dir ../train --ckpt_dir ../checkpoints
  python run_appearance_ablation.py --data_dir ../train --ckpt_dir ../checkpoints --train_masked --epochs 20
"""
import os, sys, json, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageOps
import io

sys.path.insert(0, os.path.dirname(__file__))
from knot_utils import (
    CLASSES, NUM_CLASSES, C2I, IMAGENET_MEAN, IMAGENET_STD,
    set_seed, get_device, parse_data, get_transforms,
    KnotDataset, get_canonical_test_loader, get_train_val_loaders,
    make_model, load_checkpoint, evaluate, save_checkpoint,
    HARD_PAIRS, hard_pair_accuracy, mcnemar_test,
)


# ── Background removal ─────────────────────────────────────────────────────
_REMBG_SESSION = None

def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        try:
            from rembg import new_session
            _REMBG_SESSION = new_session("u2net")
        except ImportError:
            _REMBG_SESSION = "fallback"
    return _REMBG_SESSION


def remove_background(img_pil):
    """Remove background from PIL image, return rope-only on white background."""
    session = _get_rembg_session()
    if session == "fallback":
        return _fallback_bg_removal(img_pil)
    try:
        from rembg import remove
        # rembg returns RGBA
        result = remove(img_pil, session=session)
        # Composite on white background
        bg = Image.new('RGB', result.size, (255, 255, 255))
        bg.paste(result, mask=result.split()[3])
        return bg
    except Exception:
        return _fallback_bg_removal(img_pil)


def _fallback_bg_removal(img_pil):
    """Simple center-crop as fallback when rembg is unavailable."""
    w, h = img_pil.size
    crop_ratio = 0.6
    left = int(w * (1 - crop_ratio) / 2)
    top = int(h * (1 - crop_ratio) / 2)
    right = w - left
    bottom = h - top
    cropped = img_pil.crop((left, top, right, bottom))
    bg = Image.new('RGB', (w, h), (255, 255, 255))
    bg.paste(cropped, (left, top))
    return bg


# ── Image transforms for ablation conditions ───────────────────────────────
def to_grayscale(img_pil):
    """Convert to grayscale, keep 3 channels."""
    return ImageOps.grayscale(img_pil).convert('RGB')


def equalize_histogram(img_pil):
    """Per-channel histogram equalization to flatten colour distribution."""
    return ImageOps.equalize(img_pil)


# ── Custom dataset with on-the-fly transforms ──────────────────────────────
class AblationDataset(Dataset):
    """KnotDataset with an extra pre-transform (grayscale, bg removal, etc.)."""
    def __init__(self, df, pre_transform_fn=None, tensor_transform=None):
        self.df = df.reset_index(drop=True)
        self.pre_transform_fn = pre_transform_fn
        self.tensor_transform = tensor_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.pre_transform_fn is not None:
            img = self.pre_transform_fn(img)
        if self.tensor_transform is not None:
            img = self.tensor_transform(img)
        return img, torch.tensor(row['label'], dtype=torch.long)


# ── Diagnostic evaluation (Layer 1) ────────────────────────────────────────
def find_checkpoint(name, ckpt_dir):
    patterns = [f"{name}_best.pth", f"{name}_baseline.pth",
                f"{name}.pth", f"{name}_results.pth"]
    for p in patterns:
        path = os.path.join(ckpt_dir, p)
        if os.path.exists(path):
            return path
    return None


def run_diagnostic(args):
    """Layer 1: Evaluate existing models on 4 test conditions."""
    device = get_device()
    set_seed(42)
    os.makedirs('results', exist_ok=True)
    print(f"Device: {device}")

    df = parse_data(args.data_dir)
    test_df = df[df['split'] == 'test'].sort_values('path').reset_index(drop=True)
    n_test = len(test_df)
    test_labels = test_df['label'].values
    print(f"Test set: {n_test} samples")

    _, te_transform = get_transforms(224)

    conditions = {
        'original': None,
        'grayscale': to_grayscale,
        'bg_removed': remove_background,
        'hist_eq': equalize_histogram,
    }

    models_to_eval = ['resnet18', 'resnet50', 'swin_t']
    all_results = {}

    for model_name in models_to_eval:
        ckpt_path = find_checkpoint(model_name, args.ckpt_dir)
        if not ckpt_path:
            print(f"\nSKIPPING {model_name}: no checkpoint")
            continue

        print(f"\n{'='*60}")
        print(f"  Diagnostic: {model_name}")
        print(f"{'='*60}")

        model = make_model(model_name, with_embeddings=False).to(device)
        load_checkpoint(model, ckpt_path, device)

        model_results = {}
        model_preds = {}

        for cond_name, pre_fn in conditions.items():
            print(f"  Condition: {cond_name} ...", end=' ', flush=True)
            ds = AblationDataset(test_df, pre_transform_fn=pre_fn,
                                 tensor_transform=te_transform)
            # num_workers=0 for bg_removed: rembg ONNX runtime segfaults in forked workers
            nw = 0 if cond_name == 'bg_removed' else 2
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=nw)
            acc, preds, labels = evaluate(model, loader, device)
            assert np.array_equal(labels, test_labels)

            hp = {}
            for ca, cb in HARD_PAIRS:
                hp[f"{ca}-{cb}"] = hard_pair_accuracy(preds, test_labels, ca, cb)

            model_results[cond_name] = {
                'acc': float(acc), 'hard_pairs': hp, 'preds': preds.tolist(),
            }
            model_preds[cond_name] = preds
            print(f"acc={acc:.4f}")

        # McNemar: original vs each ablation
        mcnemar = {}
        for cond_name in ['grayscale', 'bg_removed', 'hist_eq']:
            res = mcnemar_test(
                model_preds['original'], model_preds[cond_name], test_labels)
            mcnemar[f"original_vs_{cond_name}"] = res
            sig = '***' if res['p'] < 0.001 else ('**' if res['p'] < 0.01 else
                  ('*' if res['p'] < 0.05 else 'n.s.'))
            print(f"    McNemar original vs {cond_name}: "
                  f"b={res['b']} c={res['c']} p={res['p']:.6f} {sig}")

        # Accuracy drop
        orig_acc = model_results['original']['acc']
        for cond_name in ['grayscale', 'bg_removed', 'hist_eq']:
            drop = orig_acc - model_results[cond_name]['acc']
            model_results[cond_name]['acc_drop'] = round(drop, 4)

        model_results['mcnemar'] = mcnemar
        all_results[model_name] = model_results

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Diagnostic Summary (Layer 1)")
    print(f"{'='*70}")
    print(f"{'Model':>15s}  {'Original':>8s}  {'Gray':>8s}  {'BG-Rem':>8s}  {'HistEq':>8s}")
    print('-' * 55)
    for name, r in all_results.items():
        vals = [r.get(c, {}).get('acc', None) for c in
                ['original', 'grayscale', 'bg_removed', 'hist_eq']]
        strs = [f"{v:.4f}" if v is not None else "N/A" for v in vals]
        print(f"{name:>15s}  {'  '.join(f'{s:>8s}' for s in strs)}")

    return all_results


# ── Intervention (Layer 2) ──────────────────────────────────────────────────
def run_intervention(args):
    """Layer 2: Train on background-removed images, test on original + masked."""
    device = get_device()
    set_seed(args.seed)
    os.makedirs('results', exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Intervention: Train on background-removed images")
    print(f"{'='*60}")

    df = parse_data(args.data_dir)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].sort_values('path').reset_index(drop=True)
    test_labels = test_df['label'].values

    from sklearn.model_selection import train_test_split
    tr_df, va_df = train_test_split(
        train_df, test_size=0.2, random_state=args.seed, stratify=train_df['label'])

    tr_transform, te_transform = get_transforms(224)

    # Training datasets with bg removal
    tr_ds = AblationDataset(tr_df, pre_transform_fn=remove_background,
                            tensor_transform=tr_transform)
    va_ds = AblationDataset(va_df, pre_transform_fn=remove_background,
                            tensor_transform=te_transform)
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=0)

    # Test loaders
    test_masked_ds = AblationDataset(test_df, pre_transform_fn=remove_background,
                                     tensor_transform=te_transform)
    test_masked_loader = DataLoader(test_masked_ds, batch_size=64, shuffle=False, num_workers=0)
    test_orig_ds = AblationDataset(test_df, pre_transform_fn=None,
                                   tensor_transform=te_transform)
    test_orig_loader = DataLoader(test_orig_ds, batch_size=64, shuffle=False, num_workers=2)

    backbone = args.backbone
    print(f"  Backbone: {backbone}  Epochs: {args.epochs}")

    model = make_model(backbone, with_embeddings=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = 0.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        scheduler.step()

        val_acc, _, _ = evaluate(model, va_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs:
            print(f"    Epoch {ep:3d}  loss={total_loss/total:.4f}  "
                  f"train={correct/total:.4f}  val={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = os.path.join(args.ckpt_dir, f"{backbone}_masked_best.pth")
    save_checkpoint(model, ckpt_path)
    print(f"  Saved: {ckpt_path}")

    # Evaluate
    acc_masked, preds_masked, _ = evaluate(model, test_masked_loader, device)
    acc_orig, preds_orig, labels_orig = evaluate(model, test_orig_loader, device)

    print(f"  Test on masked: {acc_masked:.4f}")
    print(f"  Test on original (transfer): {acc_orig:.4f}")

    # Compare with standard-trained model
    std_ckpt = find_checkpoint(backbone, args.ckpt_dir)
    intervention_result = {
        'backbone': backbone,
        'best_val': float(best_val_acc),
        'test_masked_acc': float(acc_masked),
        'test_original_acc': float(acc_orig),
    }

    if std_ckpt:
        std_model = make_model(backbone, with_embeddings=False).to(device)
        load_checkpoint(std_model, std_ckpt, device)
        std_acc, std_preds, _ = evaluate(std_model, test_orig_loader, device)
        print(f"  Standard model on original: {std_acc:.4f}")
        intervention_result['standard_original_acc'] = float(std_acc)

        # McNemar: masked-trained vs standard on original test
        res = mcnemar_test(preds_orig, std_preds, test_labels)
        intervention_result['mcnemar_vs_standard'] = res
        sig = '***' if res['p'] < 0.001 else ('**' if res['p'] < 0.01 else
              ('*' if res['p'] < 0.05 else 'n.s.'))
        print(f"  McNemar masked-trained vs standard: p={res['p']:.6f} {sig}")

    return intervention_result


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='S2: Appearance Ablation')
    parser.add_argument('--data_dir', type=str, default='../train')
    parser.add_argument('--ckpt_dir', type=str, default='../checkpoints')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--train_masked', action='store_true',
                        help='Run Layer 2 intervention (train on masked images)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Layer 1: Diagnostic (always run)
    diagnostic_results = run_diagnostic(args)

    # Layer 2: Intervention (optional)
    intervention_result = None
    if args.train_masked:
        intervention_result = run_intervention(args)

    # Save combined results
    output = {
        'diagnostic': diagnostic_results,
        'intervention': intervention_result,
    }
    out_path = 'results/appearance_ablation.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
