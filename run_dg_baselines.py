#!/usr/bin/env python3
"""
S3: Domain Generalization Baselines — MixStyle + AmpMix + Held-Out Lighting

Three test settings per model:
  (a) Canonical test set (Set-tightness images, all lighting)
  (b) Held-out lighting: train on DL+SLA, test on SLS
  (c) Phone photos (cross-domain)

DG methods:
  - Baseline (ERM): standard training
  - MixStyle: feature-level style perturbation after BN layers
  - AmpMix: Fourier amplitude mixing as data augmentation

Usage:
  python run_dg_baselines.py --data_dir ../train --photo_dir ../phone_photos
  python run_dg_baselines.py --data_dir ../train --backbone resnet50 --epochs 25
"""
import os, sys, json, time, argparse, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from knot_utils import (
    CLASSES, NUM_CLASSES, C2I, IMAGENET_MEAN, IMAGENET_STD,
    set_seed, get_device, parse_data, get_transforms,
    KnotDataset, get_canonical_test_loader,
    _make_backbone, evaluate, save_checkpoint, HARD_PAIRS, hard_pair_accuracy,
)


# ── MixStyle ────────────────────────────────────────────────────────────────
class MixStyle(nn.Module):
    """Feature-level style perturbation (Zhou et al., 2021).

    Mixes instance-level mean/std of BN features between randomly paired
    samples in a mini-batch.  Applied probabilistically during training only.
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        x_norm = (x - mu) / sig

        # Shuffle within batch to create random pairings
        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]

        # Interpolate statistics
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu2
        sig_mix = lam * sig + (1 - lam) * sig2

        return x_norm * sig_mix + mu_mix


def inject_mixstyle(model, layer_names=('layer1', 'layer2')):
    """Insert MixStyle after specified layers (for ResNet-family)."""
    ms = MixStyle(p=0.5, alpha=0.3)
    for name in layer_names:
        if hasattr(model, name):
            orig = getattr(model, name)
            setattr(model, name, nn.Sequential(orig, ms))
    return model


# ── AmpMix (Fourier-domain augmentation) ────────────────────────────────────
class AmpMixTransform:
    """Fourier amplitude mixing augmentation (Xu et al., 2023).

    Randomly mixes the amplitude spectrum of the input image with another
    image from the batch.  Preserves phase (= structure), shifts style.
    Applied as a collate-level transform.
    """
    def __init__(self, alpha=1.0, ratio=0.5):
        self.alpha = alpha  # Beta distribution parameter
        self.ratio = ratio  # Probability of applying augmentation

    def __call__(self, batch_imgs):
        """
        Args:
            batch_imgs: (B, C, H, W) tensor batch
        Returns:
            augmented batch (same shape)
        """
        if random.random() > self.ratio:
            return batch_imgs
        B, C, H, W = batch_imgs.shape
        perm = torch.randperm(B)
        lam = np.random.beta(self.alpha, self.alpha)

        # FFT
        fft_x = torch.fft.fft2(batch_imgs, dim=(-2, -1))
        amp_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        fft_y = torch.fft.fft2(batch_imgs[perm], dim=(-2, -1))
        amp_y = torch.abs(fft_y)

        # Mix amplitude in low-frequency center
        # Create a low-frequency mask (center crop in frequency domain)
        h_center, w_center = H // 2, W // 2
        h_r = int(H * 0.1)  # 10% radius
        w_r = int(W * 0.1)
        mask = torch.zeros(1, 1, H, W)
        mask[:, :, h_center - h_r:h_center + h_r, w_center - w_r:w_center + w_r] = 1.0
        mask = mask.to(batch_imgs.device)

        # Shift amplitude to center, mix, shift back
        amp_x_shifted = torch.fft.fftshift(amp_x, dim=(-2, -1))
        amp_y_shifted = torch.fft.fftshift(amp_y, dim=(-2, -1))
        amp_mixed = amp_x_shifted * (1 - lam * mask) + amp_y_shifted * (lam * mask)
        amp_mixed = torch.fft.ifftshift(amp_mixed, dim=(-2, -1))

        # Reconstruct with mixed amplitude and original phase
        fft_mixed = amp_mixed * torch.exp(1j * pha_x)
        result = torch.fft.ifft2(fft_mixed, dim=(-2, -1)).real
        return result


# ── Data utilities ──────────────────────────────────────────────────────────
def get_lighting_split_loaders(data_dir, held_out_light='SLS', seed=42,
                               batch_size=32, sz=224, val_ratio=0.2):
    """Train on 2 lighting conditions, test on held-out lighting.

    Returns (train_loader, val_loader, heldout_test_loader, heldout_df).
    """
    df = parse_data(data_dir)
    train_lights = [l for l in ['DL', 'SLA', 'SLS'] if l != held_out_light]

    # Train split: train-split images from non-held-out lighting
    train_pool = df[(df['split'] == 'train') & (df['light'].isin(train_lights))].reset_index(drop=True)
    from sklearn.model_selection import train_test_split
    tr_df, va_df = train_test_split(
        train_pool, test_size=val_ratio, random_state=seed, stratify=train_pool['label'])

    # Held-out test: test-split images from held-out lighting
    ho_test = df[(df['split'] == 'test') & (df['light'] == held_out_light)].sort_values('path').reset_index(drop=True)

    tr_transform, te_transform = get_transforms(sz)
    tr_loader = DataLoader(KnotDataset(tr_df, tr_transform),
                           batch_size=batch_size, shuffle=True, num_workers=2)
    va_loader = DataLoader(KnotDataset(va_df, te_transform),
                           batch_size=batch_size, shuffle=False, num_workers=2)
    ho_loader = DataLoader(KnotDataset(ho_test, te_transform),
                           batch_size=batch_size, shuffle=False, num_workers=2)
    return tr_loader, va_loader, ho_loader, ho_test


def load_phone_photos(photo_dir, sz=224):
    """Load phone photos into a DataLoader (compatible with evaluate())."""
    _, te_transform = get_transforms(sz)
    rows = []
    if not os.path.isdir(photo_dir):
        return None, None

    for cls_name in sorted(os.listdir(photo_dir)):
        cls_path = os.path.join(photo_dir, cls_name)
        if not os.path.isdir(cls_path) or cls_name not in C2I:
            continue
        for img_file in sorted(os.listdir(cls_path)):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                rows.append({
                    'path': os.path.join(cls_path, img_file),
                    'label': C2I[cls_name],
                })
    if not rows:
        return None, None
    import pandas as pd
    phone_df = pd.DataFrame(rows)
    loader = DataLoader(KnotDataset(phone_df, te_transform),
                        batch_size=32, shuffle=False, num_workers=2)
    return loader, phone_df


# ── Model creation ──────────────────────────────────────────────────────────
def create_dg_model(backbone, method, device):
    """Create model with DG method applied.

    Uses EmbeddingModel (backbone.* + classifier.* keys) for consistency.
    Returns (model, needs_ampmix: bool).
    """
    from knot_utils import EmbeddingModel
    bb, dim = _make_backbone(backbone)

    if method == 'mixstyle':
        # Insert MixStyle into backbone before wrapping
        inject_mixstyle(bb, layer_names=('layer1', 'layer2'))

    model = EmbeddingModel(bb, dim, NUM_CLASSES).to(device)
    return model, (method == 'ampmix')


# ── Training ────────────────────────────────────────────────────────────────
def train_dg_model(model, tr_loader, va_loader, device, epochs=20,
                   lr=1e-4, use_ampmix=False):
    """Train with optional AmpMix augmentation."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    ampmix = AmpMixTransform(alpha=1.0, ratio=0.5) if use_ampmix else None

    best_val_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if ampmix is not None:
                imgs = ampmix(imgs)
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
        tr_loss = total_loss / total
        tr_acc = correct / total
        val_acc, _, _ = evaluate(model, va_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == epochs:
            print(f"    Epoch {ep:3d}  loss={tr_loss:.4f}  "
                  f"train={tr_acc:.4f}  val={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc


# ── Main experiment ─────────────────────────────────────────────────────────
def run_dg_experiment(args):
    device = get_device()
    set_seed(args.seed)
    os.makedirs('results', exist_ok=True)
    print(f"Device: {device}")

    methods = ['erm', 'mixstyle', 'ampmix']
    backbones = args.backbone.split(',')

    # Canonical test loader
    canon_loader, canon_df = get_canonical_test_loader(args.data_dir, batch_size=64)
    canon_labels = canon_df['label'].values
    print(f"Canonical test: {len(canon_df)} samples")

    # Held-out lighting loaders
    tr_loader, va_loader, ho_loader, ho_df = get_lighting_split_loaders(
        args.data_dir, held_out_light='SLS', seed=args.seed, batch_size=32)
    ho_labels = ho_df['label'].values
    print(f"Held-out SLS test: {len(ho_df)} samples")
    print(f"Train pool (DL+SLA): {len(tr_loader.dataset)} samples")

    # Phone photos
    phone_loader, phone_df = load_phone_photos(args.photo_dir)
    if phone_loader:
        print(f"Phone photos: {len(phone_df)} samples")
    else:
        print("Phone photos: not found, skipping")

    all_results = {}
    for bb in backbones:
        for method in methods:
            name = f"{bb}_{method}"
            print(f"\n{'='*60}")
            print(f"  Training: {name}")
            print(f"{'='*60}")

            set_seed(args.seed)
            model, use_ampmix = create_dg_model(bb, method, device)
            model, best_val = train_dg_model(
                model, tr_loader, va_loader, device,
                epochs=args.epochs, use_ampmix=use_ampmix)

            # Save checkpoint
            ckpt_path = os.path.join(args.ckpt_dir, f"{name}_best.pth")
            os.makedirs(args.ckpt_dir, exist_ok=True)
            save_checkpoint(model, ckpt_path)

            result = {'method': method, 'backbone': bb, 'best_val': float(best_val)}

            # (a) Canonical test
            acc_canon, preds_canon, labels_canon = evaluate(model, canon_loader, device)
            assert np.array_equal(labels_canon, canon_labels)
            result['canonical_acc'] = float(acc_canon)
            result['canonical_preds'] = preds_canon.tolist()

            # Hard-pair accuracy
            hp = {}
            for ca, cb in HARD_PAIRS:
                hp[f"{ca}-{cb}"] = hard_pair_accuracy(preds_canon, canon_labels, ca, cb)
            result['canonical_hard_pairs'] = hp

            # (b) Held-out lighting (SLS)
            acc_ho, preds_ho, labels_ho = evaluate(model, ho_loader, device)
            result['heldout_sls_acc'] = float(acc_ho)
            result['heldout_sls_preds'] = preds_ho.tolist()

            # (c) Phone photos
            if phone_loader:
                acc_phone, preds_phone, labels_phone = evaluate(model, phone_loader, device)
                result['phone_acc'] = float(acc_phone)
                result['phone_preds'] = preds_phone.tolist()
            else:
                result['phone_acc'] = None

            print(f"  Results: canonical={acc_canon:.4f}  "
                  f"held-out-SLS={acc_ho:.4f}  "
                  f"phone={result['phone_acc'] if result['phone_acc'] is not None else 'N/A'}")

            all_results[name] = result

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Domain Generalization Summary")
    print(f"{'='*80}")
    print(f"{'Model':>25s}  {'Canonical':>9s}  {'HO-SLS':>7s}  {'Phone':>7s}  {'BestVal':>7s}")
    print('-' * 65)
    for name, r in all_results.items():
        phone_str = f"{r['phone_acc']:.4f}" if r['phone_acc'] is not None else "N/A"
        print(f"{name:>25s}  {r['canonical_acc']:9.4f}  "
              f"{r['heldout_sls_acc']:7.4f}  {phone_str:>7s}  {r['best_val']:7.4f}")

    # Save
    out_path = 'results/dg_baselines.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='S3: DG Baselines')
    parser.add_argument('--data_dir', type=str, default='../train')
    parser.add_argument('--photo_dir', type=str, default='../phone_photos')
    parser.add_argument('--ckpt_dir', type=str, default='../checkpoints')
    parser.add_argument('--backbone', type=str, default='resnet50,swin_t',
                        help='Comma-separated backbone names')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_dg_experiment(args)


if __name__ == '__main__':
    main()
