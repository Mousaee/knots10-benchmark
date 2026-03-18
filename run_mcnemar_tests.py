#!/usr/bin/env python3
"""
McNemar paired significance tests for Knots-10.
Compares CE vs TACA, and pairwise across all models.
Runs LOCALLY from existing result JSONs (no GPU needed).

Output: results/mcnemar_results.json
"""
import json, os
import numpy as np

RESULTS_DIR = 'results'
CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']

def load_preds_labels(filepath):
    """Load predictions and labels from result JSON."""
    with open(filepath) as f:
        d = json.load(f)
    if isinstance(d, list):
        d = d[0]  # aircraft format
    preds = np.array(d.get('preds', []))
    labels = np.array(d.get('labels', []))
    return preds, labels

def mcnemar_test(preds_a, preds_b, labels):
    """
    McNemar test: compare two classifiers on the same test set.
    Returns: b (A wrong, B right), c (A right, B wrong), chi2, p_value
    """
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # b: A wrong, B right
    b = np.sum(~correct_a & correct_b)
    # c: A right, B wrong
    c = np.sum(correct_a & ~correct_b)

    # McNemar chi-squared (with continuity correction)
    if b + c == 0:
        return b, c, 0.0, 1.0

    chi2 = (abs(b - c) - 1)**2 / (b + c)

    # p-value from chi2 distribution with 1 df
    # Using survival function approximation
    from math import exp, sqrt, pi
    # Simple chi2 p-value for 1 df
    if chi2 == 0:
        p = 1.0
    else:
        # Use scipy if available, else approximate
        try:
            from scipy.stats import chi2 as chi2_dist
            p = chi2_dist.sf(chi2, df=1)
        except ImportError:
            # Approximation for chi2 with 1 df
            x = sqrt(chi2)
            # Normal approximation
            t = 1.0 / (1.0 + 0.2316419 * x)
            d = 0.3989422804014327
            p_approx = d * exp(-x*x/2.0) * t * (0.3193815 + t*(-0.3565638 + t*(1.781478 + t*(-1.821256 + t*1.330274))))
            p = 2 * p_approx  # two-tailed

    return int(b), int(c), float(chi2), float(p)

# ============================================
# Load all available model predictions
# ============================================

MODEL_FILES = {
    'ResNet-18 (CE)': 'resnet18_results.json',
    'ResNet-18 (TACA)': 'resnet18_topo_guided_results.json',
    'ResNet-50 (CE)': 'resnet50_results.json',
    'ResNet-50 (TACA)': 'resnet50_topo_guided_results.json',
    'EfficientNet-B0 (CE)': 'efficientnet_b0_results.json',
    'ViT-B/16 (CE)': 'vit_results.json',
    'Swin-T (CE)': 'swin_t_results.json',
    'TransFG': 'transfg_results.json',
    'PMG': 'pmg_results.json',
    'Graph-FGVC': 'graph_fgvc_results.json',
}

print("=" * 70)
print("McNemar Paired Significance Tests — Knots-10")
print("=" * 70)

all_preds = {}
all_labels = {}

for name, fname in MODEL_FILES.items():
    fpath = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(fpath):
        try:
            preds, labels = load_preds_labels(fpath)
            if len(preds) > 0 and len(labels) > 0:
                all_preds[name] = preds
                all_labels[name] = labels
                acc = (preds == labels).mean()
                print(f"  Loaded {name}: {len(preds)} samples, acc={acc:.4f}")
            else:
                print(f"  SKIP {name}: no preds/labels in file")
        except Exception as e:
            print(f"  SKIP {name}: {e}")
    else:
        print(f"  SKIP {name}: file not found")

# ============================================
# Key comparisons: CE vs TACA (same backbone)
# ============================================

print("\n" + "=" * 70)
print("KEY COMPARISONS: CE vs TACA (same backbone)")
print("=" * 70)

results = {}

ce_taca_pairs = [
    ('ResNet-18 (CE)', 'ResNet-18 (TACA)'),
    ('ResNet-50 (CE)', 'ResNet-50 (TACA)'),
]

for ce_name, taca_name in ce_taca_pairs:
    if ce_name in all_preds and taca_name in all_preds:
        labels = all_labels[ce_name]
        b, c, chi2, p = mcnemar_test(all_preds[ce_name], all_preds[taca_name], labels)

        acc_ce = (all_preds[ce_name] == labels).mean()
        acc_taca = (all_preds[taca_name] == labels).mean()

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

        print(f"\n{ce_name} vs {taca_name}:")
        print(f"  Acc: {acc_ce:.4f} vs {acc_taca:.4f} (diff: {(acc_taca-acc_ce)*100:+.2f}pp)")
        print(f"  CE wrong/TACA right (b): {b}")
        print(f"  CE right/TACA wrong (c): {c}")
        print(f"  McNemar chi2: {chi2:.4f}, p = {p:.6f} {sig}")

        results[f"{ce_name}_vs_{taca_name}"] = {
            'model_a': ce_name, 'model_b': taca_name,
            'acc_a': float(acc_ce), 'acc_b': float(acc_taca),
            'b_a_wrong_b_right': b, 'c_a_right_b_wrong': c,
            'chi2': chi2, 'p_value': p, 'significant': p < 0.05
        }

# ============================================
# Group models by test set ordering (McNemar requires paired samples)
# ============================================

print("\n" + "=" * 70)
print("GROUPING MODELS BY TEST SET ORDERING")
print("=" * 70)
print("(McNemar requires paired samples — only comparing within groups)")

ordering_groups = {}
for name in all_preds:
    label_key = tuple(all_labels[name][:20].tolist())  # first 20 as fingerprint
    if label_key not in ordering_groups:
        ordering_groups[label_key] = []
    ordering_groups[label_key].append(name)

for i, (key, members) in enumerate(ordering_groups.items()):
    print(f"\n  Group {i+1}: {', '.join(members)}")

# ============================================
# All VALID pairwise comparisons (within groups only)
# ============================================

print("\n" + "=" * 70)
print("ALL PAIRWISE McNemar TESTS (within-group only)")
print("=" * 70)

pairwise = {}

for group_members in ordering_groups.values():
    group_members_sorted = sorted(group_members)
    for i, name_a in enumerate(group_members_sorted):
        for name_b in group_members_sorted[i+1:]:
            labels = all_labels[name_a]

            b, c, chi2, p = mcnemar_test(all_preds[name_a], all_preds[name_b], labels)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

            acc_a = (all_preds[name_a] == labels).mean()
            acc_b = (all_preds[name_b] == labels).mean()

            print(f"  {name_a} vs {name_b}: chi2={chi2:.2f}, p={p:.4f} {sig} (b={b}, c={c})")

            key = f"{name_a}_vs_{name_b}"
            pairwise[key] = {
                'model_a': name_a, 'model_b': name_b,
                'acc_a': float(acc_a), 'acc_b': float(acc_b),
                'b': b, 'c': c, 'chi2': chi2, 'p_value': p,
                'significant_0.05': p < 0.05,
                'significant_0.01': p < 0.01,
            }

results['pairwise'] = pairwise

# ============================================
# Summary: which differences are significant?
# ============================================

print("\n" + "=" * 70)
print("SIGNIFICANCE SUMMARY (p < 0.05)")
print("=" * 70)

sig_pairs = [(k, v) for k, v in pairwise.items() if v['significant_0.05']]
nonsig_pairs = [(k, v) for k, v in pairwise.items() if not v['significant_0.05']]

print(f"\nSignificant differences ({len(sig_pairs)}):")
for k, v in sig_pairs:
    print(f"  {v['model_a']} vs {v['model_b']}: p={v['p_value']:.4f}")

print(f"\nNon-significant differences ({len(nonsig_pairs)}):")
for k, v in nonsig_pairs:
    print(f"  {v['model_a']} vs {v['model_b']}: p={v['p_value']:.4f}")

# Bonferroni correction
n_tests = len(pairwise)
alpha_bonf = 0.05 / n_tests if n_tests > 0 else 0.05
print(f"\nBonferroni-corrected alpha (for {n_tests} tests): {alpha_bonf:.6f}")

sig_bonf = [(k, v) for k, v in pairwise.items() if v['p_value'] < alpha_bonf]
print(f"Significant after Bonferroni correction: {len(sig_bonf)}")
for k, v in sig_bonf:
    print(f"  {v['model_a']} vs {v['model_b']}: p={v['p_value']:.6f}")

results['bonferroni_alpha'] = alpha_bonf
results['n_tests'] = n_tests

# Save
outfile = os.path.join(RESULTS_DIR, 'mcnemar_results.json')
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[Saved] {outfile}")
print("[DONE]")
