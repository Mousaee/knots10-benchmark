#!/usr/bin/env python3
"""
Independent validation experiments for TACA (Knots-10).
Runs on SERVER with GPU. Uses existing checkpoints.

Experiments:
1. k-NN retrieval accuracy (k=1,3,5,10) — independent of alignment metric
2. Embedding Mantel test (centroid distance vs topological distance)

These address the reviewer concern about TACA's circular evaluation:
TACA optimizes embedding-topology alignment → measuring alignment is tautological.
k-NN accuracy is an INDEPENDENT downstream metric.

Output: results/independent_validation.json
"""
import os, sys, json, time, copy, random, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report
import pandas as pd
from collections import Counter

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
SEED = 42; DATA_DIR = './train'; RESULTS_DIR = 'results'

# ============================================
# Reuse infrastructure from topo_guided_training
# ============================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topo_guided_training import (
    set_seed, get_device, KnotDataset, parse_data, get_transforms,
    build_topo_distance_matrix, make_topo_model, evaluate
)

# ============================================
# k-NN evaluation
# ============================================

def knn_accuracy(train_embs, train_labels, test_embs, test_labels, k_values=[1, 3, 5, 10]):
    """
    Compute k-NN classification accuracy.
    Uses L2 distance in embedding space.
    This is INDEPENDENT of TACA's alignment objective.
    """
    # Normalize embeddings
    train_norm = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8)
    test_norm = test_embs / (np.linalg.norm(test_embs, axis=1, keepdims=True) + 1e-8)

    # Compute pairwise distances (test x train)
    # Using cosine similarity for efficiency
    sim = test_norm @ train_norm.T  # (N_test, N_train)

    results = {}
    for k in k_values:
        # Get k nearest neighbors
        topk_indices = np.argsort(-sim, axis=1)[:, :k]
        topk_labels = train_labels[topk_indices]

        # Majority vote
        preds = []
        for i in range(len(test_labels)):
            counts = Counter(topk_labels[i])
            preds.append(counts.most_common(1)[0][0])
        preds = np.array(preds)

        acc = (preds == test_labels).mean()
        results[f'k={k}'] = {
            'accuracy': float(acc),
            'correct': int((preds == test_labels).sum()),
            'total': int(len(test_labels)),
        }
        print(f"    k={k}: acc={acc:.4f} ({(preds==test_labels).sum()}/{len(test_labels)})")

    return results

# ============================================
# Embedding Mantel test
# ============================================

def compute_centroid_distances(embeddings, labels, n_classes=10):
    """Compute pairwise L2 distances between class centroids."""
    centroids = []
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            centroids.append(embeddings[mask].mean(axis=0))
        else:
            centroids.append(np.zeros(embeddings.shape[1]))
    centroids = np.array(centroids)

    D = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            D[i, j] = np.linalg.norm(centroids[i] - centroids[j])

    # Normalize to [0, 1]
    if D.max() > 0:
        D = D / D.max()
    return D, centroids

def mantel_test(D1, D2, n_permutations=9999):
    """
    Mantel permutation test between two distance matrices.
    Uses Spearman correlation on upper triangle.
    """
    n = D1.shape[0]
    idx = np.triu_indices(n, k=1)
    v1 = D1[idx]
    v2 = D2[idx]

    # Observed correlation
    try:
        from scipy.stats import spearmanr
        obs_rho, _ = spearmanr(v1, v2)
    except ImportError:
        # Fallback to rank correlation
        def rank_array(x):
            temp = np.argsort(x)
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(x))
            return ranks.astype(float)
        r1, r2 = rank_array(v1), rank_array(v2)
        n_v = len(v1)
        obs_rho = 1 - 6 * np.sum((r1 - r2)**2) / (n_v * (n_v**2 - 1))

    # Permutation test
    count = 0
    null_rhos = []
    for _ in range(n_permutations):
        perm = np.random.permutation(n)
        D1_perm = D1[np.ix_(perm, perm)]
        v1_perm = D1_perm[idx]
        try:
            from scipy.stats import spearmanr
            rho_perm, _ = spearmanr(v1_perm, v2)
        except ImportError:
            r1p = rank_array(v1_perm)
            rho_perm = 1 - 6 * np.sum((r1p - r2)**2) / (n_v * (n_v**2 - 1))
        null_rhos.append(rho_perm)
        if rho_perm >= obs_rho:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return float(obs_rho), float(p_value), null_rhos

# ============================================
# Extract embeddings from a model
# ============================================

def extract_embeddings(model, loader, device):
    """Extract embeddings and labels from a model."""
    model.eval()
    all_embs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _ = model(x)  # forward pass
            emb = model.get_embeddings()
            all_embs.append(emb.cpu().numpy())
            all_labels.extend(y.numpy())
    return np.concatenate(all_embs), np.array(all_labels)

# ============================================
# Main
# ============================================

if __name__ == '__main__':
    set_seed(SEED)
    device = get_device()
    df = parse_data(DATA_DIR)
    topo_dist = build_topo_distance_matrix()
    # Normalize topo_dist to [0,1]
    topo_dist = topo_dist / topo_dist.max()

    print(f"Device: {device}")
    print(f"Data: {len(df)} samples")
    print(f"Topological distance matrix: {topo_dist.shape}")

    # Prepare data loaders
    from sklearn.model_selection import train_test_split
    tr_full = df[df['split']=='train']
    te_df = df[df['split']=='test']
    tr_df, va_df = train_test_split(
        tr_full, test_size=0.2, stratify=tr_full['label'], random_state=SEED)

    _, te_tf = get_transforms()
    train_loader = DataLoader(KnotDataset(tr_df, te_tf), batch_size=32, shuffle=False)
    test_loader = DataLoader(KnotDataset(te_df, te_tf), batch_size=32, shuffle=False)

    # Models to evaluate
    MODEL_CONFIGS = [
        ('resnet18', 'checkpoints/resnet18_baseline.pth', 'CE'),
        ('resnet18', 'checkpoints/resnet18_topo_guided.pth', 'TACA'),
        ('resnet50', 'checkpoints/resnet50_best.pth', 'CE'),
        ('resnet50', 'checkpoints/resnet50_topo_guided.pth', 'TACA'),
    ]

    # Also try loading from results if checkpoints don't exist
    # The models need to be retrained to extract embeddings if no checkpoints exist
    # But let's first check what's available

    all_results = {}

    for model_name, ckpt_path, method in MODEL_CONFIGS:
        label = f"{model_name}_{method}"
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            print(f"  Will train from scratch...")

            # Re-train to get embeddings
            from topo_guided_training import TopologyGuidedLoss, train_topo_guided
            tr_tf, _ = get_transforms()
            train_loader_aug = DataLoader(KnotDataset(tr_df, tr_tf), batch_size=32, shuffle=True)
            val_loader = DataLoader(KnotDataset(va_df, te_tf), batch_size=32)
            loaders = {'train': train_loader_aug, 'val': val_loader}

            set_seed(SEED)
            model = make_topo_model(model_name, len(CLASSES), device)

            if method == 'TACA':
                criterion = TopologyGuidedLoss(
                    topo_dist * topo_dist.max(),  # un-normalize for loss
                    lambda_topo=0.1, lambda_margin=0.05, device=device)
                model, _, _ = train_topo_guided(model, loaders, criterion, device, epochs=20)
            else:
                # CE only training
                from topo_guided_training import TopologyGuidedLoss
                criterion = TopologyGuidedLoss(
                    topo_dist * topo_dist.max(),
                    lambda_topo=0.0, lambda_margin=0.0, device=device)
                model, _, _ = train_topo_guided(model, loaders, criterion, device, epochs=20)

            # Save checkpoint for future use
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")
        else:
            print(f"  Loading checkpoint: {ckpt_path}")
            model = make_topo_model(model_name, len(CLASSES), device)
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            # Handle raw model checkpoints (keys without 'backbone.' prefix)
            if any(k.startswith('conv1') or k.startswith('layer1') for k in state_dict):
                new_sd = {}
                for k, v in state_dict.items():
                    if k.startswith('fc.'):
                        new_sd['classifier.' + k[3:]] = v  # fc.X -> classifier.X
                    else:
                        new_sd['backbone.' + k] = v
                state_dict = new_sd
            model.load_state_dict(state_dict, strict=False)

        # Extract embeddings
        print("  Extracting train embeddings...")
        train_embs, train_labels = extract_embeddings(model, train_loader, device)
        print(f"    Shape: {train_embs.shape}")

        print("  Extracting test embeddings...")
        test_embs, test_labels = extract_embeddings(model, test_loader, device)
        print(f"    Shape: {test_embs.shape}")

        # 1. k-NN accuracy
        print("\n  k-NN Retrieval Accuracy:")
        knn_results = knn_accuracy(train_embs, train_labels, test_embs, test_labels)

        # 2. Embedding Mantel test
        print("\n  Embedding Mantel Test:")
        emb_dist, centroids = compute_centroid_distances(test_embs, test_labels)
        mantel_rho, mantel_p, null_dist = mantel_test(emb_dist, topo_dist)
        print(f"    Spearman rho: {mantel_rho:.4f}")
        print(f"    p-value: {mantel_p:.6f}")
        print(f"    Significant (p<0.05): {'YES' if mantel_p < 0.05 else 'NO'}")

        # Standard accuracy via proper forward pass on test images
        model.eval()
        all_preds = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                all_preds.extend(torch.max(logits, 1)[1].cpu().numpy())
        test_acc = (np.array(all_preds) == test_labels).mean()

        all_results[label] = {
            'model': model_name,
            'method': method,
            'test_accuracy': float(test_acc),
            'knn': knn_results,
            'embedding_mantel': {
                'spearman_rho': mantel_rho,
                'p_value': mantel_p,
                'significant': mantel_p < 0.05,
            },
            'centroid_distance_matrix': emb_dist.tolist(),
        }

    # ============================================
    # Compare CE vs TACA using k-NN (independent metric!)
    # ============================================
    print("\n" + "=" * 70)
    print("CE vs TACA COMPARISON (k-NN as independent metric)")
    print("=" * 70)

    for backbone in ['resnet18', 'resnet50']:
        ce_key = f"{backbone}_CE"
        taca_key = f"{backbone}_TACA"
        if ce_key in all_results and taca_key in all_results:
            print(f"\n{backbone.upper()}:")
            print(f"  Standard accuracy: CE={all_results[ce_key]['test_accuracy']:.4f}, "
                  f"TACA={all_results[taca_key]['test_accuracy']:.4f}")
            for k in ['k=1', 'k=3', 'k=5', 'k=10']:
                ce_knn = all_results[ce_key]['knn'].get(k, {}).get('accuracy', 'N/A')
                taca_knn = all_results[taca_key]['knn'].get(k, {}).get('accuracy', 'N/A')
                if isinstance(ce_knn, float) and isinstance(taca_knn, float):
                    diff = (taca_knn - ce_knn) * 100
                    print(f"  {k} accuracy: CE={ce_knn:.4f}, TACA={taca_knn:.4f} ({diff:+.2f}pp)")
            print(f"  Mantel rho: CE={all_results[ce_key]['embedding_mantel']['spearman_rho']:.4f}, "
                  f"TACA={all_results[taca_key]['embedding_mantel']['spearman_rho']:.4f}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outfile = os.path.join(RESULTS_DIR, 'independent_validation.json')
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] {outfile}")
    print("[DONE]")
