#!/usr/bin/env python3
"""
Random Distance Matrix Ablation for TACA (Knots-10).
Runs on SERVER with GPU. Trains from scratch (3 random matrices × 20 epochs each).

Experiment: If TACA works because the topological distance is meaningful,
then replacing it with a RANDOM distance matrix should produce:
- Similar classification accuracy (CE dominates)
- BUT worse embedding-topology alignment and worse k-NN retrieval

This directly addresses the circularity critique:
- If random TACA ≈ real TACA on all metrics → TACA is just noise
- If random TACA < real TACA on k-NN/alignment → the structure matters

Output: results/random_ablation.json
"""
import os, sys, json, time, copy, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
SEED = 42; DATA_DIR = './train'; RESULTS_DIR = 'results'
N_RANDOM = 3  # number of random distance matrices to try

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topo_guided_training import (
    set_seed, get_device, KnotDataset, parse_data, get_transforms,
    build_topo_distance_matrix, make_topo_model, TopologyGuidedLoss,
    train_topo_guided, evaluate
)

def generate_random_distance_matrix(n=10, seed=None):
    """Generate a random symmetric distance matrix with zero diagonal."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # Random upper triangle
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = rng.uniform(0.1, 1.0)
            D[j, i] = D[i, j]
    return D

def generate_permuted_distance_matrix(D_orig, seed=None):
    """Permute rows/columns of the real distance matrix.
    This preserves the distribution of distances but scrambles class assignments.
    This is a stronger control than fully random: same distribution, wrong labels.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    perm = rng.permutation(D_orig.shape[0])
    return D_orig[np.ix_(perm, perm)]

def knn_accuracy(train_embs, train_labels, test_embs, test_labels, k=5):
    """Quick k-NN accuracy."""
    train_norm = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8)
    test_norm = test_embs / (np.linalg.norm(test_embs, axis=1, keepdims=True) + 1e-8)
    sim = test_norm @ train_norm.T
    topk = np.argsort(-sim, axis=1)[:, :k]
    preds = []
    for i in range(len(test_labels)):
        counts = Counter(train_labels[topk[i]])
        preds.append(counts.most_common(1)[0][0])
    return float((np.array(preds) == test_labels).mean())

def compute_alignment(embeddings, labels, topo_dist, n_classes=10):
    """Compute embedding-topology Spearman correlation."""
    centroids = []
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            centroids.append(embeddings[mask].mean(axis=0))
        else:
            centroids.append(np.zeros(embeddings.shape[1]))
    centroids = np.array(centroids)

    D_emb = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            D_emb[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    if D_emb.max() > 0:
        D_emb = D_emb / D_emb.max()

    idx = np.triu_indices(n_classes, k=1)
    v1 = D_emb[idx]
    v2 = topo_dist[idx]

    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(v1, v2)
    except ImportError:
        def rank_array(x):
            temp = np.argsort(x)
            ranks = np.empty_like(temp, dtype=float)
            ranks[temp] = np.arange(len(x), dtype=float)
            return ranks
        r1, r2 = rank_array(v1), rank_array(v2)
        n_v = len(v1)
        rho = 1 - 6 * np.sum((r1 - r2)**2) / (n_v * (n_v**2 - 1))
        p = -1  # can't compute without scipy

    return float(rho), float(p)

# ============================================
# Main
# ============================================

if __name__ == '__main__':
    set_seed(SEED)
    device = get_device()
    df = parse_data(DATA_DIR)
    topo_dist_raw = build_topo_distance_matrix()
    topo_dist_norm = topo_dist_raw / topo_dist_raw.max()

    print(f"Device: {device}")
    print(f"Data: {len(df)} samples")

    # Prepare data
    tr_full = df[df['split']=='train']
    te_df = df[df['split']=='test']
    tr_df, va_df = train_test_split(
        tr_full, test_size=0.2, stratify=tr_full['label'], random_state=SEED)
    tr_tf, te_tf = get_transforms()

    train_loader_eval = DataLoader(KnotDataset(tr_df, te_tf), batch_size=32, shuffle=False)
    test_loader = DataLoader(KnotDataset(te_df, te_tf), batch_size=32, shuffle=False)

    results = {}

    # ============================================
    # Condition 1: Real TACA (reference)
    # ============================================
    print("\n" + "=" * 60)
    print("CONDITION 1: Real TACA (topological distance)")
    print("=" * 60)

    set_seed(SEED)
    model = make_topo_model('resnet18', len(CLASSES), device)
    loaders = {
        'train': DataLoader(KnotDataset(tr_df, tr_tf), batch_size=32, shuffle=True),
        'val': DataLoader(KnotDataset(va_df, te_tf), batch_size=32)
    }
    criterion = TopologyGuidedLoss(topo_dist_raw, lambda_topo=0.1, lambda_margin=0.05, device=device)

    t0 = time.time()
    model, hist, best_val = train_topo_guided(model, loaders, criterion, device, epochs=20)
    train_time = time.time() - t0

    preds, labels, test_embs = evaluate(model, test_loader, device)
    _, _, train_embs_eval = evaluate(model, train_loader_eval, device)
    train_labels_eval = np.array([y for _, y in train_loader_eval.dataset])

    test_acc = float((preds == labels).mean())
    knn_5 = knn_accuracy(train_embs_eval, train_labels_eval, test_embs, labels, k=5)
    knn_1 = knn_accuracy(train_embs_eval, train_labels_eval, test_embs, labels, k=1)
    align_rho, align_p = compute_alignment(test_embs, labels, topo_dist_norm)

    print(f"  Test acc: {test_acc:.4f}")
    print(f"  k-NN(k=1): {knn_1:.4f}, k-NN(k=5): {knn_5:.4f}")
    print(f"  Alignment rho: {align_rho:.4f}")

    results['real_taca'] = {
        'distance_type': 'topological',
        'test_accuracy': test_acc,
        'knn_1': knn_1, 'knn_5': knn_5,
        'alignment_rho': align_rho,
        'alignment_p': align_p,
        'train_time': train_time,
    }

    # ============================================
    # Condition 2: CE only (no TACA)
    # ============================================
    print("\n" + "=" * 60)
    print("CONDITION 2: CE only (no TACA)")
    print("=" * 60)

    set_seed(SEED)
    model_ce = make_topo_model('resnet18', len(CLASSES), device)
    loaders_ce = {
        'train': DataLoader(KnotDataset(tr_df, tr_tf), batch_size=32, shuffle=True),
        'val': DataLoader(KnotDataset(va_df, te_tf), batch_size=32)
    }
    criterion_ce = TopologyGuidedLoss(topo_dist_raw, lambda_topo=0.0, lambda_margin=0.0, device=device)

    t0 = time.time()
    model_ce, _, _ = train_topo_guided(model_ce, loaders_ce, criterion_ce, device, epochs=20)
    train_time_ce = time.time() - t0

    preds_ce, labels_ce, test_embs_ce = evaluate(model_ce, test_loader, device)
    _, _, train_embs_ce = evaluate(model_ce, train_loader_eval, device)

    test_acc_ce = float((preds_ce == labels_ce).mean())
    knn_5_ce = knn_accuracy(train_embs_ce, train_labels_eval, test_embs_ce, labels_ce, k=5)
    knn_1_ce = knn_accuracy(train_embs_ce, train_labels_eval, test_embs_ce, labels_ce, k=1)
    align_rho_ce, align_p_ce = compute_alignment(test_embs_ce, labels_ce, topo_dist_norm)

    print(f"  Test acc: {test_acc_ce:.4f}")
    print(f"  k-NN(k=1): {knn_1_ce:.4f}, k-NN(k=5): {knn_5_ce:.4f}")
    print(f"  Alignment rho: {align_rho_ce:.4f}")

    results['ce_only'] = {
        'distance_type': 'none',
        'test_accuracy': test_acc_ce,
        'knn_1': knn_1_ce, 'knn_5': knn_5_ce,
        'alignment_rho': align_rho_ce,
        'alignment_p': align_p_ce,
        'train_time': train_time_ce,
    }

    # ============================================
    # Condition 3: Permuted TACA (3 random permutations)
    # ============================================
    print("\n" + "=" * 60)
    print(f"CONDITION 3: Permuted TACA ({N_RANDOM} random permutations)")
    print("=" * 60)

    random_results = []
    for r in range(N_RANDOM):
        rand_seed = 1000 + r
        print(f"\n  --- Random permutation {r+1}/{N_RANDOM} (seed={rand_seed}) ---")

        perm_dist = generate_permuted_distance_matrix(topo_dist_raw, seed=rand_seed)
        print(f"  Permuted distance matrix generated")

        set_seed(SEED)
        model_rand = make_topo_model('resnet18', len(CLASSES), device)
        loaders_rand = {
            'train': DataLoader(KnotDataset(tr_df, tr_tf), batch_size=32, shuffle=True),
            'val': DataLoader(KnotDataset(va_df, te_tf), batch_size=32)
        }
        criterion_rand = TopologyGuidedLoss(
            perm_dist, lambda_topo=0.1, lambda_margin=0.05, device=device)

        t0 = time.time()
        model_rand, _, _ = train_topo_guided(
            model_rand, loaders_rand, criterion_rand, device, epochs=20)
        train_time_r = time.time() - t0

        preds_r, labels_r, test_embs_r = evaluate(model_rand, test_loader, device)
        _, _, train_embs_r = evaluate(model_rand, train_loader_eval, device)

        test_acc_r = float((preds_r == labels_r).mean())
        knn_5_r = knn_accuracy(train_embs_r, train_labels_eval, test_embs_r, labels_r, k=5)
        knn_1_r = knn_accuracy(train_embs_r, train_labels_eval, test_embs_r, labels_r, k=1)
        # Alignment with REAL topological distance (not the permuted one!)
        align_rho_r, align_p_r = compute_alignment(test_embs_r, labels_r, topo_dist_norm)

        print(f"  Test acc: {test_acc_r:.4f}")
        print(f"  k-NN(k=1): {knn_1_r:.4f}, k-NN(k=5): {knn_5_r:.4f}")
        print(f"  Alignment rho (vs real topo): {align_rho_r:.4f}")

        random_results.append({
            'permutation_seed': rand_seed,
            'test_accuracy': test_acc_r,
            'knn_1': knn_1_r, 'knn_5': knn_5_r,
            'alignment_rho': align_rho_r,
            'alignment_p': align_p_r,
            'train_time': train_time_r,
        })

    results['permuted_taca'] = random_results

    # ============================================
    # Summary comparison
    # ============================================
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    rand_accs = [r['test_accuracy'] for r in random_results]
    rand_knn1 = [r['knn_1'] for r in random_results]
    rand_knn5 = [r['knn_5'] for r in random_results]
    rand_rhos = [r['alignment_rho'] for r in random_results]

    print(f"\n{'Condition':<25} {'Test Acc':>10} {'k-NN(1)':>10} {'k-NN(5)':>10} {'Align ρ':>10}")
    print("-" * 70)
    print(f"{'CE only':<25} {results['ce_only']['test_accuracy']:>10.4f} {results['ce_only']['knn_1']:>10.4f} {results['ce_only']['knn_5']:>10.4f} {results['ce_only']['alignment_rho']:>10.4f}")
    print(f"{'Real TACA':<25} {results['real_taca']['test_accuracy']:>10.4f} {results['real_taca']['knn_1']:>10.4f} {results['real_taca']['knn_5']:>10.4f} {results['real_taca']['alignment_rho']:>10.4f}")
    print(f"{'Permuted TACA (mean)':<25} {np.mean(rand_accs):>10.4f} {np.mean(rand_knn1):>10.4f} {np.mean(rand_knn5):>10.4f} {np.mean(rand_rhos):>10.4f}")
    print(f"{'Permuted TACA (std)':<25} {np.std(rand_accs):>10.4f} {np.std(rand_knn1):>10.4f} {np.std(rand_knn5):>10.4f} {np.std(rand_rhos):>10.4f}")

    print("\nInterpretation:")
    if np.mean(rand_rhos) < results['real_taca']['alignment_rho'] - 0.1:
        print("  ✓ Alignment: Real TACA >> Permuted → distance structure matters")
    else:
        print("  ✗ Alignment: Real TACA ≈ Permuted → alignment may be trivial")

    if np.mean(rand_knn5) < results['real_taca']['knn_5'] - 0.01:
        print("  ✓ k-NN: Real TACA > Permuted → TACA improves retrieval (independent metric!)")
    else:
        print("  ? k-NN: Real TACA ≈ Permuted → retrieval improvement unclear")

    results['summary'] = {
        'real_taca_vs_ce': {
            'acc_diff': results['real_taca']['test_accuracy'] - results['ce_only']['test_accuracy'],
            'knn5_diff': results['real_taca']['knn_5'] - results['ce_only']['knn_5'],
            'align_diff': results['real_taca']['alignment_rho'] - results['ce_only']['alignment_rho'],
        },
        'real_taca_vs_permuted_mean': {
            'acc_diff': results['real_taca']['test_accuracy'] - np.mean(rand_accs),
            'knn5_diff': results['real_taca']['knn_5'] - np.mean(rand_knn5),
            'align_diff': results['real_taca']['alignment_rho'] - np.mean(rand_rhos),
        }
    }

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outfile = os.path.join(RESULTS_DIR, 'random_ablation.json')
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {outfile}")
    print("[DONE]")
