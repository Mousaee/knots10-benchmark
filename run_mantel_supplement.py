#!/usr/bin/env python3
"""
Supplementary Mantel tests:
1. Embedding alignment Mantel (for R18/R50 baseline & topo-guided)
2. CUB-200 confusion-vs-taxonomy Mantel (the one the paper wrongly claimed)

Usage:
    source /home/dell/BlackPercy/bin/activate
    cd /home/dell/knots10
    python run_mantel_supplement.py
"""
import os, json, time
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform

CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']
RESULTS_DIR = 'results'

KP = {
    'ABK': {'cn':4,'family':'loop','type':'loop','comp':1},
    'BK':  {'cn':4,'family':'loop','type':'loop','comp':1},
    'CH':  {'cn':2,'family':'hitch','type':'hitch','comp':1},
    'F8K': {'cn':4,'family':'stopper','type':'prime','comp':1},
    'F8L': {'cn':4,'family':'loop','type':'loop','comp':1},
    'FSK': {'cn':6,'family':'bend','type':'composite','comp':2},
    'FMB': {'cn':8,'family':'bend','type':'composite','comp':2},
    'OHK': {'cn':3,'family':'stopper','type':'prime','comp':1},
    'RK':  {'cn':6,'family':'binding','type':'composite','comp':2},
    'SK':  {'cn':3,'family':'stopper','type':'slip','comp':1},
}
DR = {('F8K','FMB'):0.15, ('F8K','F8L'):0.1, ('OHK','SK'):0.1, ('RK','FSK'):0.1}

def build_topo_distance_matrix():
    K = len(CLASSES)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j: continue
            ci, cj = CLASSES[i], CLASSES[j]
            pi, pj = KP[ci], KP[cj]
            d1 = abs(pi['cn']-pj['cn'])/8.0
            d2 = 0.0 if pi['family']==pj['family'] else 1.0
            d3 = 0.0 if pi['type']==pj['type'] else 0.5
            d4 = abs(pi['comp']-pj['comp'])
            pair = tuple(sorted([ci,cj]))
            d5 = DR.get(pair, 0.5)
            D[i,j] = 0.25*d1 + 0.25*d2 + 0.15*d3 + 0.10*d4 + 0.25*d5
    return D

def mantel_test(D1, D2, n_perms=9999, seed=42):
    """Mantel permutation test."""
    np.random.seed(seed)
    v1 = squareform(D1, checks=False)
    v2 = squareform(D2, checks=False)
    r_obs, _ = spearmanr(v1, v2)
    K = D1.shape[0]
    null_dist = np.zeros(n_perms)
    for p in range(n_perms):
        perm = np.random.permutation(K)
        D2_perm = D2[np.ix_(perm, perm)]
        v2_perm = squareform(D2_perm, checks=False)
        null_dist[p], _ = spearmanr(v1, v2_perm)
    p_val = (np.sum(np.abs(null_dist) >= np.abs(r_obs)) + 1) / (n_perms + 1)
    return float(r_obs), float(p_val)

def confusion_rate_matrix(cm):
    """Compute symmetrized confusion rate matrix."""
    cm = np.array(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    conf_rate = cm / row_sums
    np.fill_diagonal(conf_rate, 0)
    return (conf_rate + conf_rate.T) / 2

# ─── Part 1: Embedding Mantel for Knots-10 ─────────────────

print('=== Part 1: Embedding Alignment Mantel Tests (Knots-10) ===')
print('Note: We test whether embedding centroid distances correlate with')
print('topological distances using Mantel (not parametric Spearman).')
print()

topo_D = build_topo_distance_matrix()
emb_results = {}

# Check embedding_alignment_v2.json for centroid distance matrices
emb_v2_path = f'{RESULTS_DIR}/embedding_alignment_v2.json'
if os.path.exists(emb_v2_path):
    with open(emb_v2_path) as f:
        emb_v2 = json.load(f)
    print(f'Loaded {emb_v2_path}')
    print(f'Keys: {list(emb_v2.keys()) if isinstance(emb_v2, dict) else "list"}')

    # If it contains centroid distance matrices, use those
    if isinstance(emb_v2, dict):
        for config_name in ['resnet18_baseline', 'resnet18_topo_guided',
                            'resnet50_baseline', 'resnet50_topo_guided']:
            if config_name in emb_v2:
                data = emb_v2[config_name]
                if 'centroid_distances' in data:
                    cd = np.array(data['centroid_distances'])
                    # Ensure symmetric
                    cd = (cd + cd.T) / 2
                    np.fill_diagonal(cd, 0)
                    r, p = mantel_test(topo_D, cd, n_perms=9999)
                    emb_results[config_name] = {
                        'mantel_r': r, 'mantel_p': p,
                        'spearman_r': float(data.get('spearman_rho', 'N/A')),
                    }
                    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
                    print(f'  {config_name}: Mantel r={r:.4f}, p={p:.4f} {sig} '
                          f'(original Spearman: {data.get("spearman_rho", "N/A")})')

# Fallback: use confusion matrices from result JSONs
configs = [
    ('resnet18', 'resnet18_results.json'),
    ('resnet18_topo', 'resnet18_topo_guided_results.json'),
    ('resnet50', 'resnet50_results.json'),
    ('resnet50_topo', 'resnet50_topo_guided_results.json'),
]

print('\nConfusion-rate Mantel tests (for comparison):')
for config_name, json_file in configs:
    json_path = f'{RESULTS_DIR}/{json_file}'
    if not os.path.exists(json_path):
        print(f'  [SKIP] {json_path} not found')
        continue
    with open(json_path) as f:
        rdata = json.load(f)
    if 'confusion_matrix' not in rdata:
        print(f'  [SKIP] {json_path} has no confusion_matrix')
        continue
    cm = confusion_rate_matrix(rdata['confusion_matrix'])
    r, p = mantel_test(topo_D, cm, n_perms=9999)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    emb_results[f'{config_name}_confusion'] = {'mantel_r': r, 'mantel_p': p}
    print(f'  {config_name}: Mantel r={r:.4f}, p={p:.4f} {sig}')

with open(f'{RESULTS_DIR}/mantel_supplement_results.json', 'w') as f:
    json.dump(emb_results, f, indent=2)
print(f'\n[Saved] mantel_supplement_results.json')

# ─── Part 2: CUB-200 Confusion Mantel ──────────────────────

print('\n\n=== Part 2: CUB-200 Confusion-vs-Taxonomy Mantel Test ===')
print('This is the test the paper SHOULD have run but didn\'t.')
print()

cub_path = f'{RESULTS_DIR}/cub200_taca_results.json'
if not os.path.exists(cub_path):
    print(f'[SKIP] {cub_path} not found')
else:
    with open(cub_path) as f:
        cub_data = json.load(f)

    print(f'CUB-200 result keys: {list(cub_data.keys())}')

    has_cm = 'confusion_matrix' in cub_data
    has_preds = 'preds' in cub_data or 'predictions' in cub_data
    has_labels = 'labels' in cub_data or 'true_labels' in cub_data

    print(f'  Has confusion_matrix: {has_cm}')
    print(f'  Has predictions: {has_preds}')
    print(f'  Has labels: {has_labels}')

    if has_cm:
        cm = np.array(cub_data['confusion_matrix'], dtype=float)
    elif has_preds and has_labels:
        preds = np.array(cub_data.get('preds', cub_data.get('predictions')))
        labels = np.array(cub_data.get('labels', cub_data.get('true_labels')))
        n_classes = max(max(preds), max(labels)) + 1
        from sklearn.metrics import confusion_matrix as cm_func
        cm = cm_func(labels, preds, labels=list(range(n_classes))).astype(float)
        print(f'  Built {cm.shape[0]}x{cm.shape[0]} confusion matrix from predictions')
    else:
        print('  [SKIP] Cannot build confusion matrix - need to rerun CUB-200 with prediction saving')
        cm = None

    if cm is not None:
        K = cm.shape[0]
        print(f'  Matrix size: {K}x{K}')

        # Build taxonomy distance
        try:
            from cub200_taxonomy import build_cub200_distance_matrix
            tax_D = build_cub200_distance_matrix()
            if tax_D.shape[0] != K:
                print(f'  [WARN] Taxonomy matrix is {tax_D.shape[0]}x{tax_D.shape[0]} but confusion is {K}x{K}')
                # Truncate or pad
                min_K = min(tax_D.shape[0], K)
                tax_D = tax_D[:min_K, :min_K]
                cm = cm[:min_K, :min_K]
                K = min_K

            conf_rate = confusion_rate_matrix(cm)
            print(f'  Running Mantel test ({K}x{K}, 999 permutations)...')
            t0 = time.time()
            r, p = mantel_test(tax_D, conf_rate, n_perms=999, seed=42)
            elapsed = time.time() - t0
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
            print(f'  CUB-200 Confusion Mantel: r={r:.4f}, p={p:.4f} {sig} ({elapsed:.1f}s)')

            cub_mantel = {
                'cub200_confusion_mantel_r': r,
                'cub200_confusion_mantel_p': p,
                'n_classes': K,
                'n_permutations': 999,
                'note': 'This is the CORRECT Mantel test (confusion vs taxonomy). '
                        'The paper originally reported embedding-taxonomy Spearman instead.'
            }
            with open(f'{RESULTS_DIR}/cub200_confusion_mantel.json', 'w') as f:
                json.dump(cub_mantel, f, indent=2)
            print(f'  [Saved] cub200_confusion_mantel.json')

        except ImportError:
            print('  [SKIP] cub200_taxonomy module not available')
        except Exception as e:
            print(f'  [ERROR] {e}')

# ─── Part 3: Aircraft embedding analysis ───────────────────

print('\n\n=== Part 3: Aircraft Embedding Analysis ===')
aircraft_path = f'{RESULTS_DIR}/aircraft_taca_results.json'
if not os.path.exists(aircraft_path):
    print(f'[SKIP] {aircraft_path} not found')
else:
    with open(aircraft_path) as f:
        aircraft_data = json.load(f)
    print(f'Aircraft result keys: {list(aircraft_data.keys())}')

    # Check what data is available
    for key in ['test_acc', 'macro_f1', 'embed_spearman_rho', 'confusion_matrix',
                'preds', 'predictions', 'labels', 'per_class_report', 'report']:
        if key in aircraft_data:
            val = aircraft_data[key]
            if isinstance(val, (int, float, str)):
                print(f'  {key}: {val}')
            elif isinstance(val, list):
                print(f'  {key}: list[{len(val)}]')
            elif isinstance(val, dict):
                print(f'  {key}: dict with {len(val)} keys')

print('\n[ALL DONE]')
