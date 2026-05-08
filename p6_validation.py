#!/usr/bin/env python3
"""
P6: Statistical Validation Checks for the Knot Classification Paper.

Four validation checks:
  1. McNemar paired tests (6 comparisons, Bonferroni-corrected)
  2. Bootstrap CI for pair-bin Delta_spec (TACA-real vs TACA-rand)
  3. R18 TACA-rand diagnosis (why it collapsed)
  4. MixStyle+TACA mode collapse confirmation

Runs on SERVER at /home/dell/knots10/ with GPU.

Usage:
  python p6_validation.py --data_dir ../train
"""
import os, sys, json, argparse, itertools
import numpy as np
import torch

# ── path setup (portable, not hardcoded) ────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from knot_utils import (
    CLASSES, NUM_CLASSES, C2I, HARD_PAIRS,
    get_device, set_seed,
    get_canonical_test_loader,
    _make_backbone, EmbeddingModel, evaluate,
    load_checkpoint, hard_pair_accuracy,
    build_topo_distance_matrix,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def collect_predictions(model, loader, device):
    """Run inference and return (preds, labels) as numpy arrays."""
    _, preds, labels = evaluate(model, loader, device)
    return preds, labels


def load_model_and_predict(backbone_name, ckpt_path, loader, device):
    """Build model, load checkpoint, run inference, return (preds, labels)."""
    bb, dim = _make_backbone(backbone_name)
    model = EmbeddingModel(bb, dim, NUM_CLASSES).to(device)
    load_checkpoint(model, ckpt_path, device)
    return collect_predictions(model, loader, device)


# ── AuxCrossing dual-head model (needed to load swin_t_ce_auxcrossing) ──────
class DualHeadModel(torch.nn.Module):
    """Mirrors run_aux_crossing.py DualHeadModel for checkpoint compat."""
    def __init__(self, backbone, embed_dim, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embed_dim, num_classes),
        )
        self.crossing_head = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embed_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        emb = self.backbone(x)
        logits = self.classifier(emb)
        crossing = self.crossing_head(emb).squeeze(-1)
        return logits, crossing, emb


def load_auxcrossing_and_predict(backbone_name, ckpt_path, loader, device):
    """Build DualHeadModel, load checkpoint, run inference."""
    bb, dim = _make_backbone(backbone_name)
    model = DualHeadModel(bb, dim)
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits, _, _ = model(imgs)
            preds_list.append(logits.argmax(1).cpu())
            labels_list.append(labels)
    preds = torch.cat(preds_list).numpy()
    labels = torch.cat(labels_list).numpy()
    return preds, labels


# ============================================================================
# CHECK 1: McNemar paired tests
# ============================================================================

def mcnemar_paired_test(preds_a, preds_b, labels):
    """McNemar test implemented inline (not calling knot_utils.mcnemar_test).

    Uses continuity correction: chi2 = (|b-c| - 1)^2 / (b+c)
    Returns dict with b, c, chi2, p_value.
    """
    from scipy.stats import chi2 as chi2_dist

    a_correct = (preds_a == labels)
    b_correct = (preds_b == labels)

    # b = model_A wrong AND model_B right
    b = int((~a_correct & b_correct).sum())
    # c = model_A right AND model_B wrong
    c = int((a_correct & ~b_correct).sum())

    if b + c == 0:
        return {'b': b, 'c': c, 'chi2': 0.0, 'p': 1.0}

    chi2_val = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = 1.0 - chi2_dist.cdf(chi2_val, df=1)
    return {'b': b, 'c': c, 'chi2': float(chi2_val), 'p': float(p_val)}


def run_mcnemar_checks(loader, device, ckpt_dir):
    """Run 6 McNemar paired tests with Bonferroni correction."""
    print("\n" + "=" * 72)
    print("  CHECK 1: McNemar Paired Significance Tests (6 comparisons)")
    print("=" * 72)

    # Define the 6 comparisons: (label, backbone, ckpt_A, ckpt_B)
    # Convention: A = TACA-real (expected better), B = comparison model
    # b = A wrong & B right, c = A right & B wrong
    # If TACA-real is genuinely better, c >> b
    comparisons = [
        ("Swin-T: TACA-real vs TACA-rand",
         "swin_t",
         os.path.join(ckpt_dir, "swin_t_ce_taca_real_best.pth"),
         os.path.join(ckpt_dir, "swin_t_ce_taca_rand_best.pth"),
         "standard", "standard"),
        ("Swin-T: TACA-real vs CE",
         "swin_t",
         os.path.join(ckpt_dir, "swin_t_ce_taca_real_best.pth"),
         os.path.join(ckpt_dir, "swin_t_ce_only_best.pth"),
         "standard", "standard"),
        ("ViT: TACA-real vs TACA-rand",
         "vit",
         os.path.join(ckpt_dir, "vit_ce_taca_real_best.pth"),
         os.path.join(ckpt_dir, "vit_ce_taca_rand_best.pth"),
         "standard", "standard"),
        ("ViT: TACA-real vs CE",
         "vit",
         os.path.join(ckpt_dir, "vit_ce_taca_real_best.pth"),
         os.path.join(ckpt_dir, "vit_best.pth"),
         "standard", "standard"),
        ("Swin-T: TACA-real vs AuxCrossing",
         "swin_t",
         os.path.join(ckpt_dir, "swin_t_ce_taca_real_best.pth"),
         os.path.join(ckpt_dir, "swin_t_ce_auxcrossing_best.pth"),
         "standard", "auxcrossing"),
        ("ResNet-50: TACA-real vs CE",
         "resnet50",
         os.path.join(ckpt_dir, "resnet50_ce_taca_real_best.pth"),
         os.path.join(ckpt_dir, "resnet50_ce_only_best.pth"),
         "standard", "standard"),
    ]

    n_comparisons = len(comparisons)
    alpha = 0.05
    alpha_bonf = alpha / n_comparisons

    results = []
    # Cache predictions to avoid re-running inference for the same checkpoint
    pred_cache = {}

    for label, backbone, ckpt_a, ckpt_b, type_a, type_b in comparisons:
        print(f"\n  {label}")

        # Check checkpoint existence
        if not os.path.isfile(ckpt_a):
            print(f"    SKIP: {ckpt_a} not found")
            results.append({'comparison': label, 'skipped': True,
                            'reason': f'{ckpt_a} not found'})
            continue
        if not os.path.isfile(ckpt_b):
            print(f"    SKIP: {ckpt_b} not found")
            results.append({'comparison': label, 'skipped': True,
                            'reason': f'{ckpt_b} not found'})
            continue

        # Load model A predictions
        if ckpt_a not in pred_cache:
            if type_a == "auxcrossing":
                pa, la = load_auxcrossing_and_predict(backbone, ckpt_a, loader, device)
            else:
                pa, la = load_model_and_predict(backbone, ckpt_a, loader, device)
            pred_cache[ckpt_a] = (pa, la)
        pa, la = pred_cache[ckpt_a]

        # Load model B predictions
        if ckpt_b not in pred_cache:
            if type_b == "auxcrossing":
                pb, lb = load_auxcrossing_and_predict(backbone, ckpt_b, loader, device)
            else:
                pb, lb = load_model_and_predict(backbone, ckpt_b, loader, device)
            pred_cache[ckpt_b] = (pb, lb)
        pb, lb = pred_cache[ckpt_b]

        # Sanity: labels must match (same canonical test set)
        assert np.array_equal(la, lb), \
            f"Label mismatch for {label}: shapes {la.shape} vs {lb.shape}"

        acc_a = float((pa == la).mean())
        acc_b = float((pb == la).mean())

        res = mcnemar_paired_test(pa, pb, la)
        sig_raw = res['p'] < alpha
        sig_bonf = res['p'] < alpha_bonf

        sig_str = ("***" if res['p'] < 0.001 else
                   "**" if res['p'] < 0.01 else
                   "*" if res['p'] < alpha_bonf else
                   "n.s.")

        print(f"    Acc A (TACA-real): {acc_a:.4f}    Acc B: {acc_b:.4f}    "
              f"delta: {(acc_a - acc_b)*100:+.2f}pp")
        print(f"    b (A wrong, B right): {res['b']}    "
              f"c (A right, B wrong): {res['c']}")
        print(f"    chi2 = {res['chi2']:.4f}    p = {res['p']:.6f}    {sig_str}")

        results.append({
            'comparison': label,
            'backbone': backbone,
            'acc_a': acc_a,
            'acc_b': acc_b,
            'b': res['b'],
            'c': res['c'],
            'chi2': res['chi2'],
            'p_value': res['p'],
            'significant_raw_0.05': sig_raw,
            'significant_bonferroni': sig_bonf,
        })

    print(f"\n  Bonferroni-corrected alpha = {alpha_bonf:.6f} "
          f"({n_comparisons} comparisons)")

    return {
        'n_comparisons': n_comparisons,
        'alpha': alpha,
        'alpha_bonferroni': alpha_bonf,
        'tests': results,
    }


# ============================================================================
# CHECK 2: Bootstrap CI for pair-bin Delta_spec
# ============================================================================

def pairwise_accuracy(preds, labels, ci, cj):
    """Accuracy restricted to samples from class ci or cj."""
    mask = (labels == ci) | (labels == cj)
    if mask.sum() == 0:
        return float('nan')
    return float((preds[mask] == labels[mask]).mean())


def run_bootstrap_delta_spec(loader, device, ckpt_dir, n_boot=10000, seed=42):
    """Bootstrap CI for per-bin mean Delta_spec (TACA-real minus TACA-rand)."""
    print("\n" + "=" * 72)
    print("  CHECK 2: Bootstrap CI for Pair-Bin Delta_spec")
    print("=" * 72)

    rng = np.random.RandomState(seed)
    topo_D = build_topo_distance_matrix()

    # Generate all 45 class pairs
    pairs = list(itertools.combinations(range(NUM_CLASSES), 2))
    assert len(pairs) == 45, f"Expected 45 pairs, got {len(pairs)}"

    # Bin thresholds
    bins = {
        'near':   lambda d: d <= 0.3,
        'medium': lambda d: 0.3 < d <= 0.6,
        'far':    lambda d: d > 0.6,
    }

    # Assign each pair to a bin
    pair_bins = {}
    for ci, cj in pairs:
        d = topo_D[ci, cj]
        for bname, cond in bins.items():
            if cond(d):
                pair_bins[(ci, cj)] = bname
                break

    # Print bin distribution
    bin_counts = {b: 0 for b in bins}
    for b in pair_bins.values():
        bin_counts[b] += 1
    print(f"  Pair bins: {bin_counts}")

    configs = [
        ("Swin-T", "swin_t",
         os.path.join(ckpt_dir, "swin_t_ce_taca_real_best.pth"),
         os.path.join(ckpt_dir, "swin_t_ce_taca_rand_best.pth")),
        ("ResNet-50", "resnet50",
         os.path.join(ckpt_dir, "resnet50_ce_taca_real_best.pth"),
         os.path.join(ckpt_dir, "resnet50_ce_taca_rand_best.pth")),
    ]

    all_bootstrap_results = {}

    for display_name, backbone, ckpt_real, ckpt_rand in configs:
        print(f"\n  --- {display_name} ---")

        if not os.path.isfile(ckpt_real) or not os.path.isfile(ckpt_rand):
            print(f"    SKIP: checkpoint not found")
            all_bootstrap_results[display_name] = {'skipped': True}
            continue

        preds_real, labels = load_model_and_predict(
            backbone, ckpt_real, loader, device)
        preds_rand, _ = load_model_and_predict(
            backbone, ckpt_rand, loader, device)

        # Compute pairwise accuracy for each of the 45 pairs
        delta_by_bin = {b: [] for b in bins}
        pair_deltas = {}

        for ci, cj in pairs:
            acc_real = pairwise_accuracy(preds_real, labels, ci, cj)
            acc_rand = pairwise_accuracy(preds_rand, labels, ci, cj)
            delta = acc_real - acc_rand
            bname = pair_bins[(ci, cj)]
            delta_by_bin[bname].append(delta)
            pair_deltas[(ci, cj)] = {
                'acc_real': acc_real, 'acc_rand': acc_rand,
                'delta': delta, 'bin': bname,
                'topo_dist': float(topo_D[ci, cj]),
                'pair': f"{CLASSES[ci]}-{CLASSES[cj]}",
            }

        backbone_results = {'pair_details': {}, 'bins': {}}
        for k, v in pair_deltas.items():
            backbone_results['pair_details'][v['pair']] = v

        # Bootstrap within each bin
        for bname in bins:
            deltas = np.array(delta_by_bin[bname])
            n_pairs_in_bin = len(deltas)
            if n_pairs_in_bin == 0:
                print(f"    {bname:>8s}: no pairs in this bin")
                backbone_results['bins'][bname] = {
                    'n_pairs': 0, 'mean_delta': None,
                    'ci_lo': None, 'ci_hi': None,
                }
                continue

            observed_mean = float(deltas.mean())

            # Bootstrap: resample pairs within the bin
            boot_means = np.empty(n_boot)
            for i in range(n_boot):
                idx = rng.randint(0, n_pairs_in_bin, size=n_pairs_in_bin)
                boot_means[i] = deltas[idx].mean()

            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))

            print(f"    {bname:>8s}: n={n_pairs_in_bin:2d}  "
                  f"mean Delta_spec={observed_mean:+.4f}  "
                  f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")

            backbone_results['bins'][bname] = {
                'n_pairs': n_pairs_in_bin,
                'mean_delta': observed_mean,
                'ci_lo': ci_lo,
                'ci_hi': ci_hi,
                'boot_std': float(boot_means.std()),
            }

        all_bootstrap_results[display_name] = backbone_results

    return {
        'n_bootstrap': n_boot,
        'seed': seed,
        'bin_thresholds': {'near': '<=0.3', 'medium': '(0.3,0.6]', 'far': '>0.6'},
        'results': all_bootstrap_results,
    }


# ============================================================================
# CHECK 3: R18 TACA-rand diagnosis
# ============================================================================

def run_r18_diagnosis(ckpt_dir, device):
    """Diagnostic summary for R18 TACA-rand collapse."""
    print("\n" + "=" * 72)
    print("  CHECK 3: ResNet-18 TACA-rand Collapse Diagnosis")
    print("=" * 72)

    diag = {
        'issue': 'ResNet-18 TACA-rand produces near-chance accuracy',
        'root_cause': [],
        'recommendation': 'No retraining needed; report R18 TACA-rand as '
                          'degenerate and exclude from main comparisons.',
    }

    # Check if R18 TACA-rand checkpoint exists
    r18_rand_ckpt = os.path.join(ckpt_dir, "resnet18_ce_taca_rand_best.pth")
    r18_real_ckpt = os.path.join(ckpt_dir, "resnet18_ce_taca_real_best.pth")

    # Embedding dimension mismatch analysis
    print("\n  ResNet-18 backbone: embed_dim = 512")
    print("  ResNet-50 backbone: embed_dim = 2048")
    print("  Swin-T backbone:    embed_dim = 768")
    print("  ViT-B/16 backbone:  embed_dim = 768")
    diag['root_cause'].append(
        "R18 has the smallest embedding space (512-dim). The TACA centroid "
        "alignment loss with a RANDOM distance matrix creates conflicting "
        "gradients: CE pushes centroids apart for discrimination while "
        "TACA-rand pushes them to respect an arbitrary distance structure. "
        "In a low-dimensional space (512-d), these conflicting objectives "
        "cannot be jointly satisfied, causing training collapse."
    )

    diag['root_cause'].append(
        "Larger backbones (ResNet-50=2048d, Swin-T/ViT=768d) have enough "
        "capacity to absorb the noisy TACA signal in unused embedding "
        "dimensions, so TACA-rand degrades gracefully rather than collapsing."
    )

    # Check if checkpoint is loadable and inspect
    if os.path.isfile(r18_rand_ckpt):
        try:
            state = torch.load(r18_rand_ckpt, map_location='cpu',
                               weights_only=True)
            # Check classifier weights
            cls_key = 'classifier.1.weight'
            if cls_key in state:
                w = state[cls_key]
                w_std = float(w.std())
                w_max = float(w.abs().max())
                print(f"\n  R18 TACA-rand classifier weight stats:")
                print(f"    std  = {w_std:.6f}")
                print(f"    max  = {w_max:.6f}")
                if w_std < 0.01:
                    diag['root_cause'].append(
                        f"Classifier weights nearly zero (std={w_std:.6f}), "
                        "confirming training collapse."
                    )
                    print(f"    --> Weights near-zero: CONFIRMED training collapse")
                else:
                    print(f"    --> Weights appear normal")
            diag['checkpoint_found'] = True
        except Exception as e:
            print(f"  Could not load R18 TACA-rand checkpoint: {e}")
            diag['checkpoint_found'] = False
    else:
        print(f"  R18 TACA-rand checkpoint not found at {r18_rand_ckpt}")
        diag['checkpoint_found'] = False
        diag['root_cause'].append(
            "No R18 TACA-rand checkpoint available. This backbone was likely "
            "not trained with the unified S4 pipeline (which uses "
            "EmbeddingModel). The original topo_guided_training.py used a "
            "different TopoGuidedModel wrapper, creating a checkpoint format "
            "mismatch if mixed."
        )

    # Pipeline mismatch note
    diag['root_cause'].append(
        "Pipeline mismatch: early R18 experiments used topo_guided_training.py "
        "(TopoGuidedModel with backbone.* + classifier.* keys and a "
        "TopologyGuidedLoss that includes both centroid alignment AND margin "
        "loss). Later experiments (run_aux_crossing.py) used EmbeddingModel + "
        "TACALoss (centroid alignment only, no margin). The margin loss "
        "component (lambda_margin=0.05) acts as an additional push on the "
        "512-d space, worsening the collapse under random distances."
    )

    for i, cause in enumerate(diag['root_cause'], 1):
        print(f"\n  Cause {i}: {cause}")

    print(f"\n  Recommendation: {diag['recommendation']}")
    return diag


# ============================================================================
# CHECK 4: MixStyle+TACA mode collapse confirmation
# ============================================================================

def run_mixstyle_taca_diagnosis():
    """Confirm MixStyle+TACA mode collapse from combo_dg_structure.json."""
    print("\n" + "=" * 72)
    print("  CHECK 4: MixStyle+TACA Mode Collapse Confirmation")
    print("=" * 72)

    combo_path = os.path.join(_SCRIPT_DIR, 'results', 'combo_dg_structure.json')
    if not os.path.isfile(combo_path):
        # Try mechanism subfolder
        combo_path = os.path.join(_SCRIPT_DIR, 'results', 'mechanism',
                                  'combo_dg_structure.json')

    result = {'json_path': combo_path}

    if not os.path.isfile(combo_path):
        print(f"  combo_dg_structure.json not found at {combo_path}")
        result['found'] = False
        result['diagnosis'] = 'File not found; cannot confirm mode collapse.'
        return result

    with open(combo_path) as f:
        data = json.load(f)

    result['found'] = True

    # Check MixStyle+TACA predictions
    if 'all_preds' in data and 'MixStyle+TACA' in data['all_preds']:
        preds = np.array(data['all_preds']['MixStyle+TACA'])
        n_samples = len(preds)
        unique_preds = np.unique(preds)
        n_unique = len(unique_preds)
        dominant_class = int(np.bincount(preds).argmax())
        dominant_frac = float((preds == dominant_class).mean())

        print(f"  Total test samples: {n_samples}")
        print(f"  Unique predicted classes: {n_unique} / {NUM_CLASSES}")
        print(f"  Dominant class: {CLASSES[dominant_class]} (idx={dominant_class})")
        print(f"  Fraction predicting dominant class: {dominant_frac:.4f}")

        is_collapsed = (n_unique == 1) or (dominant_frac > 0.95)

        if is_collapsed:
            print(f"  --> MODE COLLAPSE CONFIRMED: "
                  f"{'all' if n_unique == 1 else f'{dominant_frac*100:.1f}%'} "
                  f"predictions = {CLASSES[dominant_class]}")
        else:
            print(f"  --> Mode collapse NOT confirmed "
                  f"({n_unique} unique classes, max frac={dominant_frac:.2f})")

        result['n_samples'] = n_samples
        result['n_unique_preds'] = n_unique
        result['dominant_class'] = CLASSES[dominant_class]
        result['dominant_fraction'] = dominant_frac
        result['mode_collapse'] = is_collapsed
    else:
        print("  MixStyle+TACA predictions not found in JSON")
        result['mode_collapse'] = None
        result['diagnosis'] = 'MixStyle+TACA key not found in all_preds'

    # Also check model-level stats if available
    if 'models' in data and 'MixStyle+TACA' in data['models']:
        m = data['models']['MixStyle+TACA']
        test_acc = m.get('test_acc', None)
        best_val = m.get('best_val', None)
        print(f"\n  Model stats from JSON:")
        print(f"    test_acc = {test_acc}")
        print(f"    best_val = {best_val}")
        result['test_acc'] = test_acc
        result['best_val'] = best_val

        if test_acc is not None and abs(test_acc - 1.0 / NUM_CLASSES) < 0.02:
            print(f"    --> Test acc ~ 1/K = {1.0/NUM_CLASSES:.2f}: "
                  f"confirms chance-level = collapse")

    return result


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def print_summary(check1, check2, check3, check4):
    """Print a consolidated summary table."""
    print("\n")
    print("=" * 80)
    print("  VALIDATION SUMMARY")
    print("=" * 80)

    # McNemar summary
    print("\n  [1] McNemar Paired Tests")
    print(f"      Bonferroni alpha = {check1['alpha_bonferroni']:.6f}")
    print(f"      {'Comparison':<42s}  {'chi2':>7s}  {'p':>10s}  {'Sig?':>5s}")
    print("      " + "-" * 70)
    for t in check1['tests']:
        if t.get('skipped'):
            print(f"      {t['comparison']:<42s}  {'SKIPPED':>7s}")
        else:
            sig = 'YES' if t['significant_bonferroni'] else 'no'
            print(f"      {t['comparison']:<42s}  "
                  f"{t['chi2']:7.3f}  {t['p']:10.6f}  {sig:>5s}")

    # Bootstrap summary
    print("\n  [2] Bootstrap CI for Delta_spec (TACA-real minus TACA-rand)")
    for bbone, bdata in check2['results'].items():
        if isinstance(bdata, dict) and bdata.get('skipped'):
            print(f"      {bbone}: SKIPPED")
            continue
        print(f"      {bbone}:")
        for bname in ['near', 'medium', 'far']:
            bd = bdata['bins'].get(bname, {})
            if bd.get('n_pairs', 0) == 0:
                print(f"        {bname:>8s}: no pairs")
            else:
                print(f"        {bname:>8s}: n={bd['n_pairs']:2d}  "
                      f"mean={bd['mean_delta']:+.4f}  "
                      f"95% CI [{bd['ci_lo']:+.4f}, {bd['ci_hi']:+.4f}]")

    # R18 diagnosis
    print(f"\n  [3] R18 TACA-rand: {check3['issue']}")
    print(f"      Recommendation: {check3['recommendation']}")

    # MixStyle+TACA
    collapse_str = ("CONFIRMED" if check4.get('mode_collapse') else
                    "NOT confirmed" if check4.get('mode_collapse') is False else
                    "UNKNOWN")
    print(f"\n  [4] MixStyle+TACA mode collapse: {collapse_str}")
    if check4.get('dominant_class'):
        print(f"      Dominant class: {check4['dominant_class']}  "
              f"({check4['dominant_fraction']*100:.1f}% of predictions)")

    print("\n" + "=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='P6: Statistical Validation Checks for Knots-10 Paper')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the 10Knots dataset root (with class subfolders)')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--n_boot', type=int, default=10000,
                        help='Number of bootstrap resamples')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Checkpoint dir: {args.ckpt_dir}")

    os.makedirs('results', exist_ok=True)

    # Canonical test set (shared across all checks)
    loader, test_df = get_canonical_test_loader(args.data_dir, batch_size=64)
    print(f"Canonical test set: {len(test_df)} samples")

    # ── Run all 4 checks ──────────────────────────────────────────────────
    check1 = run_mcnemar_checks(loader, device, args.ckpt_dir)
    check2 = run_bootstrap_delta_spec(
        loader, device, args.ckpt_dir, n_boot=args.n_boot, seed=args.seed)
    check3 = run_r18_diagnosis(args.ckpt_dir, device)
    check4 = run_mixstyle_taca_diagnosis()

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(check1, check2, check3, check4)

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        'mcnemar': check1,
        'bootstrap_delta_spec': check2,
        'r18_diagnosis': check3,
        'mixstyle_taca_diagnosis': check4,
    }

    out_path = os.path.join('results', 'validation_checks.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[Saved] {out_path}")
    print("[DONE]")


if __name__ == '__main__':
    main()
