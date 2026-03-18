#!/usr/bin/env python3
"""
Integrate multi-seed FGVC results into the paper.
Run after downloading seed-123 and seed-456 results from server.

Outputs:
1. LaTeX table rows for Table S1
2. Ranking stability analysis
3. Whether main text claims need softening
"""
import json, os, sys
import numpy as np

RESULTS_DIR = 'results'
CLASSES = ['ABK','BK','CH','F8K','F8L','FSK','FMB','OHK','RK','SK']

def get_f1(d):
    """Extract macro F1 from various JSON formats."""
    if 'macro_f1' in d:
        return d['macro_f1']
    if 'f1_score' in d:
        return d['f1_score']
    if 'report' in d:
        rpt = d['report']
        if isinstance(rpt, dict) and CLASSES[0] in rpt:
            return np.mean([rpt[c]['f1-score'] for c in CLASSES if c in rpt])
    if 'per_class_report' in d:
        rpt = d['per_class_report']
        if isinstance(rpt, dict) and CLASSES[0] in rpt:
            return np.mean([rpt[c]['f1-score'] for c in CLASSES if c in rpt])
    return None

def load_model_seeds(model_key, seed_pattern, seed42_file=None):
    """Load results for a model across all available seeds."""
    results = {}
    # Seed 42 (original)
    if seed42_file:
        f42 = f'{RESULTS_DIR}/{seed42_file}'
    else:
        f42 = f'{RESULTS_DIR}/{model_key}_results.json'
    if os.path.exists(f42):
        with open(f42) as f:
            d = json.load(f)
        results[42] = {
            'test_acc': d.get('test_acc', d.get('best_test_acc')),
            'macro_f1': get_f1(d)
        }
    # Seeds 123, 456
    for seed in [123, 456]:
        fp = f'{RESULTS_DIR}/{seed_pattern.format(seed=seed)}'
        if os.path.exists(fp):
            with open(fp) as f:
                d = json.load(f)
            results[seed] = {
                'test_acc': d.get('test_acc', d.get('best_test_acc')),
                'macro_f1': get_f1(d)
            }
    return results

# Model configurations to process
MODELS = {
    'TransFG (CE)': {
        'key': 'transfg',
        'seed_pattern': 'transfg_results_seed{seed}.json',
    },
    'PMG (CE)': {
        'key': 'pmg',
        'seed_pattern': 'pmg_results_seed{seed}.json',
    },
    'Graph-FGVC (CE)': {
        'key': 'graph_fgvc',
        'seed_pattern': 'graph_fgvc_results_seed{seed}.json',
    },
    'ResNet-18 (CE+TACA learnable)': {
        'key': 'learnable_weights_resnet18',
        'seed_pattern': 'learnable_weights_resnet18_seed{seed}.json',
        'seed42_file': 'learnable_weights_resnet18.json',
    },
    'ResNet-50 (CE+TACA learnable)': {
        'key': 'learnable_weights_resnet50',
        'seed_pattern': 'learnable_weights_resnet50_seed{seed}.json',
        'seed42_file': 'learnable_weights_resnet50.json',
    },
}

print("=" * 70)
print("Multi-Seed FGVC Integration Report")
print("=" * 70)

all_results = {}
for name, cfg in MODELS.items():
    seeds = load_model_seeds(cfg['key'], cfg['seed_pattern'], cfg.get('seed42_file'))
    all_results[name] = seeds

    n_seeds = len(seeds)
    available_seeds = sorted(seeds.keys())

    print(f"\n{name}:")
    print(f"  Seeds available: {available_seeds} ({n_seeds}/3)")

    if n_seeds == 0:
        print("  [NO DATA]")
        continue

    for s, r in sorted(seeds.items()):
        f1_str = f"{r['macro_f1']:.3f}" if r['macro_f1'] is not None else "N/A"
        print(f"  Seed {s}: acc={r['test_acc']*100:.2f}%, F1={f1_str}")

    if n_seeds >= 2:
        accs = [r['test_acc'] for r in seeds.values()]
        f1s = [r['macro_f1'] for r in seeds.values() if r['macro_f1'] is not None]

        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs, ddof=0) * 100  # population std for consistency with existing table

        print(f"  Mean ± std: {mean_acc:.2f} ± {std_acc:.2f}%")
        if f1s:
            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s, ddof=0)
            print(f"  F1 mean ± std: {mean_f1:.3f} ± {std_f1:.3f}")

# Generate LaTeX table rows
print("\n" + "=" * 70)
print("LaTeX Table S1 Rows (add after existing rows)")
print("=" * 70)
print("\\midrule")
print("\\multicolumn{3}{@{}l}{\\textit{FGVC-Specialized Methods}} \\\\")
print("\\addlinespace[2pt]")

for name in ['TransFG (CE)', 'PMG (CE)', 'Graph-FGVC (CE)']:
    seeds = all_results[name]
    if len(seeds) >= 3:
        accs = [seeds[s]['test_acc'] for s in sorted(seeds.keys())]
        f1s = [seeds[s]['macro_f1'] for s in sorted(seeds.keys()) if seeds[s]['macro_f1'] is not None]
        ma, sa = np.mean(accs)*100, np.std(accs, ddof=0)*100
        if f1s:
            mf, sf = np.mean(f1s), np.std(f1s, ddof=0)
            print(f"{name:30s} & ${ma:.2f} \\pm {sa:.2f}$ & ${mf:.3f} \\pm {sf:.3f}$ \\\\")
        else:
            print(f"{name:30s} & ${ma:.2f} \\pm {sa:.2f}$ & --- \\\\")
    elif len(seeds) == 1:
        s = list(seeds.values())[0]
        print(f"{name:30s} & {s['test_acc']*100:.2f}\\% (seed 42 only) & --- \\\\")
    else:
        print(f"{name:30s} & [WAITING FOR DATA] \\\\")

print("\\midrule")
print("\\multicolumn{3}{@{}l}{\\textit{Learnable Weights}} \\\\")
print("\\addlinespace[2pt]")

for name in ['ResNet-18 (CE+TACA learnable)', 'ResNet-50 (CE+TACA learnable)']:
    seeds = all_results[name]
    if len(seeds) >= 3:
        accs = [seeds[s]['test_acc'] for s in sorted(seeds.keys())]
        ma, sa = np.mean(accs)*100, np.std(accs, ddof=0)*100
        print(f"{name:30s} & ${ma:.2f} \\pm {sa:.2f}$ & --- \\\\")
    elif len(seeds) == 1:
        s = list(seeds.values())[0]
        print(f"{name:30s} & {s['test_acc']*100:.2f}\\% (seed 42 only) & --- \\\\")
    else:
        print(f"{name:30s} & [WAITING FOR DATA] \\\\")

# Ranking stability analysis
print("\n" + "=" * 70)
print("Ranking Stability Analysis")
print("=" * 70)

# Check FGVC ranking across seeds
fgvc_models = ['TransFG (CE)', 'PMG (CE)', 'Graph-FGVC (CE)']
for seed in [42, 123, 456]:
    accs = {}
    for name in fgvc_models:
        if seed in all_results[name]:
            accs[name] = all_results[name][seed]['test_acc']
    if accs:
        ranked = sorted(accs.items(), key=lambda x: -x[1])
        print(f"\nSeed {seed}: {' > '.join(f'{n.split()[0]}({a*100:.1f}%)' for n,a in ranked)}")

# Check main text claims
print("\n" + "=" * 70)
print("Main Text Claim Verification")
print("=" * 70)

# Claim 1: TACA ResNet-18 learnable (98.1%) > TransFG (97.7%)
if 42 in all_results['TransFG (CE)'] and 42 in all_results['ResNet-18 (CE+TACA learnable)']:
    taca = all_results['ResNet-18 (CE+TACA learnable)'][42]['test_acc'] * 100
    transfg = all_results['TransFG (CE)'][42]['test_acc'] * 100
    print(f"Seed 42: TACA-learnable ({taca:.1f}%) vs TransFG ({transfg:.1f}%): {'HOLDS' if taca > transfg else 'VIOLATED'}")

# Claim 2: PMG < baseline ResNet-18 (95.8%)
if 42 in all_results['PMG (CE)']:
    pmg = all_results['PMG (CE)'][42]['test_acc'] * 100
    print(f"Seed 42: PMG ({pmg:.1f}%) < ResNet-18 baseline (95.8%): {'HOLDS' if pmg < 95.8 else 'VIOLATED'}")

# Multi-seed ranking stability
if all(len(all_results[m]) >= 3 for m in fgvc_models):
    print("\nMulti-seed mean ranking:")
    means = {m: np.mean([all_results[m][s]['test_acc'] for s in sorted(all_results[m].keys())])
             for m in fgvc_models}
    ranked = sorted(means.items(), key=lambda x: -x[1])
    print(f"  {' > '.join(f'{n.split()[0]}({v*100:.1f}%)' for n,v in ranked)}")

    # Check if ranking is stable across all seeds
    rankings_by_seed = {}
    for seed in [42, 123, 456]:
        seed_accs = {m: all_results[m][seed]['test_acc'] for m in fgvc_models if seed in all_results[m]}
        if len(seed_accs) == 3:
            rankings_by_seed[seed] = sorted(seed_accs, key=lambda m: -seed_accs[m])

    if len(rankings_by_seed) >= 2:
        seeds = sorted(rankings_by_seed.keys())
        stable = all(rankings_by_seed[seeds[0]] == rankings_by_seed[s] for s in seeds[1:])
        print(f"  Ranking stable across seeds: {'YES' if stable else 'NO'}")
        if not stable:
            print("  WARNING: FGVC ranking is NOT stable. Need to soften main text claims.")
            for s, r in sorted(rankings_by_seed.items()):
                print(f"    Seed {s}: {' > '.join(m.split()[0] for m in r)}")
else:
    print("\n[Cannot check ranking stability - not all seeds available yet]")

print("\n" + "=" * 70)
print("DONE. When all 3 seeds are available, update paper Table S1 and S1 text.")
print("=" * 70)
