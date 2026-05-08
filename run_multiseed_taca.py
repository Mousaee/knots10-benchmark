#!/usr/bin/env python3
"""
Multi-seed TACA Δ_spec ablation for Swin-T, ViT-B/16, and ResNet-50.

Seeds: 42, 123, 456
Conditions: CE-only, TACA(real), TACA(rand)
Output: results/multiseed_delta_spec.json

Usage:
  # Run on GPU 0 (swin_t + resnet50):
  python run_multiseed_taca.py --data_dir ./train --gpu 0 --backbones swin_t,resnet50

  # Run on GPU 1 (vit):
  python run_multiseed_taca.py --data_dir ./train --gpu 1 --backbones vit

  # Run everything on default GPU:
  python run_multiseed_taca.py --data_dir ./train
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from knot_utils import (
    CLASSES, NUM_CLASSES, C2I, CROSSING_NUMS,
    HARD_PAIRS,
    set_seed, get_device, parse_data, get_transforms,
    KnotDataset, get_canonical_test_loader, get_train_val_loaders,
    _make_backbone, EmbeddingModel, evaluate,
    build_topo_distance_matrix, save_checkpoint, load_checkpoint,
    hard_pair_accuracy, mcnemar_test,
)

# ── Constants ──────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456]
ALL_BACKBONES = ["swin_t", "vit", "resnet50"]
CONDITIONS = ["ce_only", "taca_real", "taca_rand"]
EPOCHS = 20
LR = 1e-4
WD = 1e-4
BATCH_SIZE = 32


# ── TACA Loss ──────────────────────────────────────────────────────────────
class TACALoss(nn.Module):
    def __init__(self, topo_dist_matrix, device, lambda_topo=0.1):
        super().__init__()
        D = torch.tensor(topo_dist_matrix, dtype=torch.float32)
        D = D / D.max()
        self.topo_dist = D.to(device)
        self.lambda_topo = lambda_topo

    def forward(self, logits, labels, embeddings):
        centroids, present = [], []
        for c in range(NUM_CLASSES):
            mask = (labels == c)
            if mask.sum() > 0:
                centroids.append(embeddings[mask].mean(dim=0))
                present.append(c)
        if len(present) < 2:
            return torch.tensor(0.0, device=embeddings.device)
        centroids = torch.stack(centroids)
        diff = centroids.unsqueeze(0) - centroids.unsqueeze(1)
        pdist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)
        cdist_norm = pdist / pdist.max() if pdist.max() > 0 else pdist
        idx = torch.tensor(present, device=embeddings.device)
        topo_sub = self.topo_dist[idx][:, idx]
        mask_upper = torch.triu(
            torch.ones_like(cdist_norm, dtype=torch.bool), diagonal=1
        )
        return self.lambda_topo * F.mse_loss(
            cdist_norm[mask_upper], topo_sub[mask_upper]
        )


# ── Training ───────────────────────────────────────────────────────────────
def train_one_condition(backbone_name, condition, data_dir, device, seed):
    """Train one (backbone, condition, seed) combination. Return best model."""
    set_seed(seed)
    bb, dim = _make_backbone(backbone_name)
    model = EmbeddingModel(bb, dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ce_loss = nn.CrossEntropyLoss()

    # Build TACA loss based on condition
    taca_loss = None
    topo_D = build_topo_distance_matrix()
    if condition == "taca_real":
        taca_loss = TACALoss(topo_D, device, lambda_topo=0.1)
    elif condition == "taca_rand":
        rng = np.random.RandomState(seed + 999)
        perm = rng.permutation(NUM_CLASSES)
        topo_D_rand = topo_D[perm][:, perm]
        taca_loss = TACALoss(topo_D_rand, device, lambda_topo=0.1)

    tr_loader, va_loader = get_train_val_loaders(
        data_dir, seed=seed, batch_size=BATCH_SIZE
    )

    best_val_acc, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, emb = model(imgs)
            loss = ce_loss(logits, labels)
            if taca_loss is not None:
                loss = loss + taca_loss(logits, labels, emb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_acc, _, _ = evaluate(model, va_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == EPOCHS:
            print(f"      Ep {ep:2d}/{EPOCHS}  val={val_acc:.4f}  best={best_val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-seed TACA Δ_spec")
    parser.add_argument("--data_dir", type=str, default="./train")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--backbones", type=str, default=",".join(ALL_BACKBONES),
        help="Comma-separated backbones to run",
    )
    parser.add_argument("--out", type=str, default="results/multiseed_delta_spec.json")
    parser.add_argument(
        "--skip_seed42", action="store_true",
        help="Load seed-42 results from existing delta_spec.json instead of retraining",
    )
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    backbones = [b.strip() for b in args.backbones.split(",")]
    print(f"Device: {device}")
    print(f"Backbones: {backbones}")
    print(f"Seeds: {SEEDS}")
    print(f"Conditions: {CONDITIONS}")
    print("=" * 70)

    # Load existing seed-42 results if requested
    existing_seed42 = {}
    if args.skip_seed42 and os.path.exists("results/delta_spec.json"):
        with open("results/delta_spec.json") as f:
            raw = json.load(f)
        # Map to our key format
        cond_map = {"ce": "ce_only", "taca_real": "taca_real", "taca_rand": "taca_rand"}
        for arch, arch_data in raw.items():
            for old_cond, new_cond in cond_map.items():
                acc = arch_data.get("acc", {}).get(old_cond)
                hp = arch_data.get("hard_pair_acc", {}).get(old_cond, {})
                if acc is not None:
                    existing_seed42[(arch, new_cond)] = {
                        "acc": acc, "hard_pairs": hp
                    }
        print(f"Loaded {len(existing_seed42)} existing seed-42 results")

    # ── Run all combinations ───────────────────────────────────────────
    # results[(backbone, condition, seed)] = {acc, hard_pairs, preds, labels}
    all_results = {}
    canon_loader, canon_df = get_canonical_test_loader(args.data_dir, batch_size=64)
    canon_labels = canon_df["label"].values
    n_test = len(canon_df)
    print(f"Test set: {n_test} images\n")

    total_runs = len(backbones) * len(CONDITIONS) * len(SEEDS)
    run_idx = 0

    for bb in backbones:
        for cond in CONDITIONS:
            for seed in SEEDS:
                run_idx += 1
                key = (bb, cond, seed)

                # Check if we can skip seed 42
                if seed == 42 and (bb, cond) in existing_seed42:
                    r = existing_seed42[(bb, cond)]
                    all_results[key] = {
                        "acc": r["acc"],
                        "hard_pairs": r["hard_pairs"],
                        "preds": None,  # not available from JSON
                        "source": "cached",
                    }
                    print(
                        f"[{run_idx}/{total_runs}] {bb} | {cond} | seed={seed} "
                        f"-> CACHED acc={r['acc']*100:.2f}%"
                    )
                    continue

                print(f"\n[{run_idx}/{total_runs}] {bb} | {cond} | seed={seed}")
                print("-" * 50)
                t0 = time.time()

                model, best_val = train_one_condition(
                    bb, cond, args.data_dir, device, seed
                )

                # Save checkpoint
                ckpt_name = f"{bb}_{cond}_seed{seed}.pth"
                save_checkpoint(model, os.path.join("checkpoints", ckpt_name))

                # Evaluate on canonical test set
                acc, preds, labels = evaluate(model, canon_loader, device)
                assert np.array_equal(labels, canon_labels)

                # Hard-pair accuracy
                hp = {}
                for ca, cb in HARD_PAIRS:
                    hp[f"{ca}_vs_{cb}"] = float(
                        hard_pair_accuracy(preds, labels, ca, cb)
                    )
                hp_mean = float(np.mean(list(hp.values())))

                elapsed = time.time() - t0
                all_results[key] = {
                    "acc": float(acc),
                    "hard_pairs": hp,
                    "hp_mean": hp_mean,
                    "preds": preds.tolist(),
                    "source": "trained",
                }
                print(
                    f"    => acc={acc*100:.2f}%  HP_mean={hp_mean*100:.2f}%  "
                    f"time={elapsed:.0f}s"
                )

    # ── Aggregate per (backbone, condition): mean ± std ────────────────
    print("\n" + "=" * 70)
    print("Aggregated Results (mean ± std across 3 seeds)")
    print("=" * 70)

    summary = {}
    for bb in backbones:
        summary[bb] = {}
        for cond in CONDITIONS:
            accs = []
            hp_means = []
            for seed in SEEDS:
                r = all_results.get((bb, cond, seed))
                if r is not None:
                    accs.append(r["acc"])
                    # Compute HP mean from hard_pairs dict
                    if r.get("hard_pairs"):
                        hp_vals = [v for v in r["hard_pairs"].values()
                                   if isinstance(v, (int, float))]
                        if hp_vals:
                            hp_means.append(np.mean(hp_vals))

            summary[bb][cond] = {
                "accs": [round(a * 100, 2) for a in accs],
                "mean": round(np.mean(accs) * 100, 2) if accs else None,
                "std": round(np.std(accs) * 100, 2) if len(accs) > 1 else None,
                "hp_mean": round(np.mean(hp_means) * 100, 2) if hp_means else None,
                "hp_std": round(np.std(hp_means) * 100, 2) if len(hp_means) > 1 else None,
            }

    # Print summary table
    print(f"\n{'Backbone':<12} {'Condition':<12} {'Seed 42':>8} {'Seed 123':>8} "
          f"{'Seed 456':>8} {'Mean':>8} {'Std':>6}")
    print("-" * 72)
    for bb in backbones:
        for cond in CONDITIONS:
            s = summary[bb][cond]
            accs = s["accs"]
            row = f"{bb:<12} {cond:<12}"
            for i in range(3):
                row += f" {accs[i]:7.2f}%" if i < len(accs) else "     N/A"
            row += f" {s['mean']:7.2f}%" if s["mean"] else "     N/A"
            row += f" {s['std']:5.2f}" if s["std"] is not None else "   N/A"
            print(row)
        print()

    # ── Compute Δ_spec per seed and aggregated ─────────────────────────
    print("\n" + "=" * 70)
    print("Δ_spec = Acc(TACA_real) - Acc(TACA_rand)")
    print("=" * 70)

    delta_spec_summary = {}
    for bb in backbones:
        deltas = []
        for seed in SEEDS:
            r_real = all_results.get((bb, "taca_real", seed))
            r_rand = all_results.get((bb, "taca_rand", seed))
            if r_real and r_rand:
                d = (r_real["acc"] - r_rand["acc"]) * 100
                deltas.append(d)

        delta_spec_summary[bb] = {
            "per_seed": [round(d, 2) for d in deltas],
            "mean": round(np.mean(deltas), 2) if deltas else None,
            "std": round(np.std(deltas), 2) if len(deltas) > 1 else None,
        }
        print(
            f"  {bb:<12}: per_seed={delta_spec_summary[bb]['per_seed']}  "
            f"mean={delta_spec_summary[bb]['mean']:+.2f} ± "
            f"{delta_spec_summary[bb]['std']:.2f} pp"
        )

    # ── McNemar tests (real vs rand, per seed) ─────────────────────────
    print("\n" + "=" * 70)
    print("McNemar Tests (TACA_real vs TACA_rand, per seed)")
    print("=" * 70)

    mcnemar_results = {}
    for bb in backbones:
        mcnemar_results[bb] = {}
        for seed in SEEDS:
            r_real = all_results.get((bb, "taca_real", seed))
            r_rand = all_results.get((bb, "taca_rand", seed))
            if (r_real and r_rand
                    and r_real.get("preds") and r_rand.get("preds")):
                pa = np.array(r_real["preds"])
                pb = np.array(r_rand["preds"])
                mc = mcnemar_test(pa, pb, canon_labels)
                mcnemar_results[bb][str(seed)] = mc
                sig = "***" if mc["p"] < 0.001 else (
                    "**" if mc["p"] < 0.01 else (
                        "*" if mc["p"] < 0.05 else "n.s."))
                print(
                    f"  {bb:<12} seed={seed}: b={mc['b']} c={mc['c']} "
                    f"chi2={mc['chi2']:.2f} p={mc['p']:.4f} {sig}"
                )
            else:
                print(f"  {bb:<12} seed={seed}: preds not available (cached)")

    # ── Save everything ────────────────────────────────────────────────
    output = {
        "config": {
            "seeds": SEEDS,
            "backbones": backbones,
            "conditions": CONDITIONS,
            "epochs": EPOCHS,
            "n_test": n_test,
        },
        "per_run": {
            f"{bb}_{cond}_seed{seed}": {
                k: v for k, v in r.items() if k != "preds"
            }
            for (bb, cond, seed), r in all_results.items()
        },
        "summary": summary,
        "delta_spec": delta_spec_summary,
        "mcnemar_real_vs_rand": mcnemar_results,
    }

    # Also save full preds for McNemar (separate file, can be large)
    preds_output = {}
    for (bb, cond, seed), r in all_results.items():
        if r.get("preds"):
            preds_output[f"{bb}_{cond}_seed{seed}"] = r["preds"]

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")

    preds_path = args.out.replace(".json", "_preds.json")
    with open(preds_path, "w") as f:
        json.dump(preds_output, f)
    print(f"Predictions saved to {preds_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
