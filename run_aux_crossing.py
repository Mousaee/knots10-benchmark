#!/usr/bin/env python3
"""
S4: Auxiliary Crossing-Number Head with Ordinal Regression

Four training conditions (all use the same backbone + canonical test set):
  1. CE only          — standard cross-entropy baseline
  2. CE + TACA(real)  — topology-aware centroid alignment with real distances
  3. CE + TACA(rand)  — same but with random permuted distance matrix
  4. CE + AuxCrossing — auxiliary ordinal regression on visual crossing number

AuxCrossing:  L = L_CE + λ_aux * L_ordinal(C_vis_pred, C_vis_true)
  where C_vis ∈ {2, 3, 4, 6, 8} are ordinal, not categorical.

Usage:
  python run_aux_crossing.py --data_dir ../train
  python run_aux_crossing.py --data_dir ../train --backbone swin_t --epochs 25
"""
import os, sys, json, time, argparse, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from knot_utils import (
    CLASSES, NUM_CLASSES, C2I, CROSSING_NUMS,
    set_seed, get_device, parse_data, get_transforms,
    KnotDataset, get_canonical_test_loader, get_train_val_loaders,
    _make_backbone, EmbeddingModel, evaluate,
    build_topo_distance_matrix, save_checkpoint,
    HARD_PAIRS, hard_pair_accuracy, mcnemar_test,
)


# ── Crossing number targets ────────────────────────────────────────────────
# Per-class visual crossing numbers (ordered by CLASSES index)
# Normalize to [0, 1] for ordinal regression: (c - min) / (max - min)
_C_VIS = np.array(CROSSING_NUMS, dtype=np.float32)  # [4,4,2,4,4,6,8,3,6,3]
C_VIS_MIN, C_VIS_MAX = _C_VIS.min(), _C_VIS.max()
C_VIS_NORM = (_C_VIS - C_VIS_MIN) / (C_VIS_MAX - C_VIS_MIN)  # [0,1]


def get_crossing_targets(labels):
    """Map class labels to normalized crossing-number targets."""
    return torch.tensor(C_VIS_NORM[labels], dtype=torch.float32)


# ── Dual-head model ────────────────────────────────────────────────────────
class DualHeadModel(nn.Module):
    """Backbone + classification head + ordinal crossing-number head.

    forward() returns (class_logits, crossing_pred, embeddings).
    """
    def __init__(self, backbone, embed_dim, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        # Ordinal regression head: predicts scalar in [0, 1]
        self.crossing_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        emb = self.backbone(x)
        logits = self.classifier(emb)
        crossing = self.crossing_head(emb).squeeze(-1)
        return logits, crossing, emb


# ── TACA loss (centroid alignment) ──────────────────────────────────────────
class TACALoss(nn.Module):
    """Topology-Aware Centroid Alignment loss."""
    def __init__(self, topo_dist_matrix, device, lambda_topo=0.1):
        super().__init__()
        D = torch.tensor(topo_dist_matrix, dtype=torch.float32)
        D = D / D.max()
        self.topo_dist = D.to(device)
        self.lambda_topo = lambda_topo
        self.num_classes = D.shape[0]

    def forward(self, logits, labels, embeddings):
        """Returns scalar TACA loss (no CE included)."""
        centroids, present = [], []
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                centroids.append(embeddings[mask].mean(dim=0))
                present.append(c)
        if len(present) < 2:
            return torch.tensor(0.0, device=embeddings.device)

        centroids = torch.stack(centroids)
        diff = centroids.unsqueeze(0) - centroids.unsqueeze(1)
        pdist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)
        if pdist.max() > 0:
            cdist_norm = pdist / pdist.max()
        else:
            cdist_norm = pdist

        idx = torch.tensor(present, device=embeddings.device)
        topo_sub = self.topo_dist[idx][:, idx]
        mask_upper = torch.triu(torch.ones_like(cdist_norm, dtype=torch.bool), diagonal=1)
        return self.lambda_topo * F.mse_loss(cdist_norm[mask_upper], topo_sub[mask_upper])


# ── Training conditions ─────────────────────────────────────────────────────
def train_condition(condition, backbone_name, tr_loader, va_loader,
                    device, seed, epochs=20, lr=1e-4):
    """Train one condition, return (model, best_val_acc)."""
    set_seed(seed)
    bb, dim = _make_backbone(backbone_name)

    topo_D = build_topo_distance_matrix()

    if condition == 'ce_auxcrossing':
        model = DualHeadModel(bb, dim).to(device)
    else:
        model = EmbeddingModel(bb, dim).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_loss = nn.CrossEntropyLoss()

    # Condition-specific loss
    taca_loss = None
    if condition == 'ce_taca_real':
        taca_loss = TACALoss(topo_D, device, lambda_topo=0.1)
    elif condition == 'ce_taca_rand':
        # Permute rows and columns of topo distance matrix
        rng = np.random.RandomState(seed + 999)
        perm = rng.permutation(NUM_CLASSES)
        topo_D_rand = topo_D[perm][:, perm]
        taca_loss = TACALoss(topo_D_rand, device, lambda_topo=0.1)

    lambda_aux = 0.5  # Weight for AuxCrossing loss

    best_val_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            out = model(imgs)
            if condition == 'ce_auxcrossing':
                logits, crossing_pred, emb = out
            else:
                logits, emb = out

            loss = ce_loss(logits, labels)

            # Add TACA
            if taca_loss is not None:
                loss = loss + taca_loss(logits, labels, emb)

            # Add AuxCrossing
            if condition == 'ce_auxcrossing':
                c_targets = get_crossing_targets(labels.cpu().numpy()).to(device)
                loss = loss + lambda_aux * F.mse_loss(crossing_pred, c_targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_acc, _, _ = evaluate(model, va_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == epochs:
            print(f"    Epoch {ep:3d}  loss={total_loss/total:.4f}  "
                  f"train={correct/total:.4f}  val={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc


# ── Main experiment ─────────────────────────────────────────────────────────
def run_aux_crossing_experiment(args):
    device = get_device()
    set_seed(args.seed)
    os.makedirs('results', exist_ok=True)
    print(f"Device: {device}")

    conditions = ['ce_only', 'ce_taca_real', 'ce_taca_rand', 'ce_auxcrossing']
    backbones = args.backbone.split(',')

    # Data
    tr_loader, va_loader = get_train_val_loaders(
        args.data_dir, seed=args.seed, batch_size=32)
    canon_loader, canon_df = get_canonical_test_loader(args.data_dir, batch_size=64)
    canon_labels = canon_df['label'].values
    n_test = len(canon_df)
    print(f"Train: {len(tr_loader.dataset)}  Val: {len(va_loader.dataset)}  "
          f"Test: {n_test}")

    all_results = {}
    all_preds = {}  # for McNemar

    for bb in backbones:
        for cond in conditions:
            name = f"{bb}_{cond}"
            print(f"\n{'='*60}")
            print(f"  Training: {name}")
            print(f"{'='*60}")

            t0 = time.time()
            model, best_val = train_condition(
                cond, bb, tr_loader, va_loader, device,
                args.seed, args.epochs)
            train_time = time.time() - t0

            # Save checkpoint
            ckpt_path = os.path.join(args.ckpt_dir, f"{name}_best.pth")
            os.makedirs(args.ckpt_dir, exist_ok=True)
            save_checkpoint(model, ckpt_path)

            # Evaluate on canonical test
            acc, preds, labels = evaluate(model, canon_loader, device)
            assert np.array_equal(labels, canon_labels)

            # Hard-pair accuracy
            hp = {}
            for ca, cb in HARD_PAIRS:
                hp[f"{ca}-{cb}"] = hard_pair_accuracy(preds, canon_labels, ca, cb)

            result = {
                'condition': cond,
                'backbone': bb,
                'best_val': float(best_val),
                'test_acc': float(acc),
                'train_time': round(train_time, 1),
                'hard_pairs': hp,
                'preds': preds.tolist(),
            }

            # AuxCrossing: also report crossing prediction quality
            if cond == 'ce_auxcrossing':
                model.eval()
                crossing_preds_all = []
                with torch.no_grad():
                    for imgs, _ in canon_loader:
                        imgs = imgs.to(device)
                        _, cp, _ = model(imgs)
                        crossing_preds_all.append(cp.cpu())
                crossing_preds = torch.cat(crossing_preds_all).numpy()
                # De-normalize
                c_pred_raw = crossing_preds * (C_VIS_MAX - C_VIS_MIN) + C_VIS_MIN
                c_true_raw = _C_VIS[canon_labels]
                crossing_mae = float(np.abs(c_pred_raw - c_true_raw).mean())
                result['crossing_mae'] = crossing_mae
                print(f"  Crossing MAE: {crossing_mae:.3f}")

            print(f"  {name}: acc={acc:.4f}  val={best_val:.4f}  time={train_time:.0f}s")
            all_results[name] = result
            all_preds[name] = preds

    # ── McNemar across conditions ──────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  McNemar Paired Tests — AuxCrossing Ablation")
    print(f"{'='*80}")
    names = sorted(all_preds.keys())
    n_pairs = len(names) * (len(names) - 1) // 2
    bonf = 0.05 / max(n_pairs, 1)

    mcnemar_results = []
    print(f"{'Model A':>30s}  {'Model B':>30s}  {'b':>4s}  {'c':>4s}  "
          f"{'chi2':>7s}  {'p':>10s}  Sig")
    print('-' * 100)
    for i, ma in enumerate(names):
        for mb in names[i + 1:]:
            pa, pb = all_preds[ma], all_preds[mb]
            res = mcnemar_test(pa, pb, canon_labels)
            sig = '***' if res['p'] < 0.001 else ('**' if res['p'] < 0.01 else
                  ('*' if res['p'] < bonf else 'n.s.'))
            print(f"{ma:>30s}  {mb:>30s}  {res['b']:4d}  {res['c']:4d}  "
                  f"{res['chi2']:7.2f}  {res['p']:10.6f}  {sig}")
            mcnemar_results.append({
                'model_a': ma, 'model_b': mb, **res, 'sig': sig,
            })

    # ── Hard-pair summary table ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Hard-Pair Accuracy by Condition")
    print(f"{'='*80}")
    hp_header = f"{'Model':>30s}"
    for ca, cb in HARD_PAIRS:
        hp_header += f"  {ca}-{cb:>5s}"
    hp_header += f"  {'Overall':>7s}"
    print(hp_header)
    print('-' * len(hp_header))
    for name in names:
        r = all_results[name]
        row = f"{name:>30s}"
        for ca, cb in HARD_PAIRS:
            row += f"  {r['hard_pairs'][f'{ca}-{cb}']:8.3f}"
        row += f"  {r['test_acc']:7.4f}"
        print(row)

    # ── Save ───────────────────────────────────────────────────────────────
    output = {
        'n_test': n_test,
        'conditions': conditions,
        'backbones': backbones,
        'models': {k: {kk: vv for kk, vv in v.items() if kk != 'preds'}
                   for k, v in all_results.items()},
        'all_preds': {k: v.tolist() for k, v in all_preds.items()},
        'mcnemar': mcnemar_results,
        'bonferroni_alpha': bonf,
    }
    out_path = 'results/aux_crossing.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='S4: AuxCrossing Ablation')
    parser.add_argument('--data_dir', type=str, default='../train')
    parser.add_argument('--ckpt_dir', type=str, default='../checkpoints')
    parser.add_argument('--backbone', type=str, default='resnet50,swin_t',
                        help='Comma-separated backbone names')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_aux_crossing_experiment(args)


if __name__ == '__main__':
    main()
