#!/usr/bin/env python3
"""
Merge single-run results into the unified robustness_multiseed.json format.

Scans results/single_*.json, groups by (model, lam_taca, lam_taml),
computes mean ± std across seeds, and outputs the same format as
robustness_multiseed.py's main loop.

Usage:
    python merge_results.py
    python merge_results.py --results_dir results --output results/robustness_multiseed.json
"""
import argparse
import json
import os
import glob
from collections import defaultdict

import numpy as np

# Display name mapping: (model, lam_taca, lam_taml) -> display name
CONFIG_NAMES = {
    ('resnet18', 0.0, 0.0):     'ResNet-18 (CE)',
    ('resnet18', 0.1, 0.0):     'ResNet-18 (CE+TACA)',
    ('swin_t', 0.0, 0.0):       'Swin-T (CE)',
    ('resnet50', 0.0, 0.0):     'ResNet-50 (CE)',
    ('efficientnet_b0', 0.0, 0.0): 'EfficientNet-B0 (CE)',
    ('vit', 0.0, 0.0):          'ViT-B/16 (CE)',
    ('resnet18', 0.0, 0.005):   'ResNet-18 (CE+TAML)',
    ('resnet18', 0.1, 0.005):   'ResNet-18 (CE+TACA+TAML)',
}

# Priority order for output sorting
PRIORITY_ORDER = [
    'ResNet-18 (CE)',
    'ResNet-18 (CE+TACA)',
    'Swin-T (CE)',
    'ResNet-50 (CE)',
    'EfficientNet-B0 (CE)',
    'ViT-B/16 (CE)',
    'ResNet-18 (CE+TAML)',
    'ResNet-18 (CE+TACA+TAML)',
]


def main():
    parser = argparse.ArgumentParser(description='Merge single-run results')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path. Default: {results_dir}/robustness_multiseed.json')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.results_dir, 'robustness_multiseed.json')

    # Scan all single result files
    pattern = os.path.join(args.results_dir, 'single_*.json')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files matching {pattern}")
        return

    print(f"Found {len(files)} single-run result files")

    # Group by config key
    groups = defaultdict(list)
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        key = (data['model'], data['lam_taca'], data['lam_taml'])
        groups[key].append(data)

    # Build merged results
    all_results = {}
    for key, runs in groups.items():
        model, lt, lm = key
        disp_name = CONFIG_NAMES.get(key, f"{model}_taca{lt}_taml{lm}")

        seeds = [r['seed'] for r in runs]
        per_seed = [
            {'val_acc': r['val_acc'], 'test_acc': r['test_acc'],
             'f1': r['f1'], 'time': r['train_time']}
            for r in runs
        ]

        vals = [r['val_acc'] for r in runs]
        tests = [r['test_acc'] for r in runs]
        f1s = [r['f1'] for r in runs]

        summary = {
            'config': disp_name,
            'model': model,
            'lam_taca': lt,
            'lam_taml': lm,
            'seeds': seeds,
            'per_seed': per_seed,
            'val_acc_mean': float(np.mean(vals)),
            'val_acc_std': float(np.std(vals)),
            'test_acc_mean': float(np.mean(tests)),
            'test_acc_std': float(np.std(tests)),
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
        }
        all_results[disp_name] = summary

    # Sort by priority order
    sorted_results = {}
    for name in PRIORITY_ORDER:
        if name in all_results:
            sorted_results[name] = all_results[name]
    # Append any configs not in the priority list
    for name in all_results:
        if name not in sorted_results:
            sorted_results[name] = all_results[name]

    # Save
    with open(args.output, 'w') as f:
        json.dump(sorted_results, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"  MERGED ROBUSTNESS SUMMARY  ({len(files)} runs → {len(sorted_results)} configs)")
    print(f"{'=' * 70}")
    print(f"{'Config':<28} {'Seeds':>6} {'Test Acc':>18} {'Macro F1':>18}")
    print(f"{'-' * 28} {'-' * 6} {'-' * 18} {'-' * 18}")
    for name, s in sorted_results.items():
        n_seeds = len(s['seeds'])
        print(f"{name:<28} {n_seeds:>6} "
              f"{s['test_acc_mean'] * 100:6.2f}% ± {s['test_acc_std'] * 100:.2f}%"
              f"   {s['f1_mean']:.4f} ± {s['f1_std']:.4f}")

    # Warn about incomplete configs
    for name in PRIORITY_ORDER:
        if name not in sorted_results:
            print(f"  [MISSING] {name}")
        elif len(sorted_results[name]['seeds']) < 3:
            print(f"  [INCOMPLETE] {name}: only {len(sorted_results[name]['seeds'])}/3 seeds")

    print(f"\n[Saved] {args.output}")


if __name__ == '__main__':
    main()
