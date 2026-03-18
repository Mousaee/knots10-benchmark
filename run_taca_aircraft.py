"""
TACA (Task-specific Contrastive Learning) Training on FGVC-Aircraft Dataset
Validates generalization of TACA method across fine-grained datasets.
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from aircraft_hierarchy import build_aircraft_distance_matrix


# Configuration
AIRCRAFT_DIR = os.environ.get('AIRCRAFT_DIR', os.path.expanduser('~/data/fgvc-aircraft-2013b'))
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class AircraftDataset(Dataset):
    """FGVC-Aircraft dataset."""

    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root (str): Path to fgvc-aircraft-2013b directory
            split (str): 'train', 'val', or 'test'
            transform: Image transforms
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []

        # Load variant names (labels)
        variants_file = self.root / "data" / "variants.txt"
        with open(variants_file, 'r') as f:
            self.variant_names = [line.strip() for line in f if line.strip()]
        self.variant_to_idx = {v: i for i, v in enumerate(self.variant_names)}

        # Load image-variant pairs
        images_file = self.root / "data" / f"images_variant_{split}.txt"
        with open(images_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_id = parts[0]
                    variant = ' '.join(parts[1:])
                    if variant in self.variant_to_idx:
                        # Construct image path: images/{id:06d}.jpg
                        image_path = self.root / "data" / "images" / f"{image_id}.jpg"
                        if image_path.exists():
                            self.images.append(str(image_path))
                            self.labels.append(self.variant_to_idx[variant])

        print(f"Loaded {len(self.images)} images for {split} split")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.images[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label


class TACArithmeticLoss(nn.Module):
    """
    Task-specific Contrastive Arithmetic Loss
    Encourages learned features to respect task-specific similarity structure.
    """

    def __init__(self, distance_matrix, temperature=0.07, margin=0.5):
        """
        Args:
            distance_matrix (np.ndarray): (num_classes, num_classes) distance matrix
            temperature (float): Temperature for contrastive scaling
            margin (float): Margin for triplet-like loss
        """
        super().__init__()
        self.register_buffer('distance_matrix',
                            torch.from_numpy(distance_matrix).float())
        self.temperature = temperature
        self.margin = margin

    def forward(self, logits, labels):
        """
        Args:
            logits: (batch_size, num_classes) model predictions
            labels: (batch_size,) ground truth class indices

        Returns:
            loss: scalar tensor
        """
        batch_size = logits.shape[0]

        # Normalize logits as similarity scores
        logits = logits / self.temperature

        # Extract distances for label pairs in batch
        distances = self.distance_matrix[labels]  # (batch_size, num_classes)

        # Contrastive loss: penalize small similarity with distant classes
        # and large similarity with close classes
        loss = 0.0
        for b in range(batch_size):
            label = labels[b]
            logit = logits[b]

            # For each class j, use task-specific distance to determine target
            dists = self.distance_matrix[label]  # (num_classes,)

            # Target: similarity should be inversely proportional to distance
            # s_ij should be high when d_ij is low, and vice versa
            targets = 1.0 - dists  # (num_classes,)

            # MSE between normalized logits and targets
            loss = loss + torch.mean((logit - targets) ** 2)

        return loss / batch_size


class ResNet50Classifier(nn.Module):
    """ResNet-50 for Aircraft variant classification."""

    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)

        # Replace classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)


def train_epoch(model, train_loader, criterion_ce, criterion_taca, optimizer, device, use_taca=False, lambda_taca=0.1):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss_ce = criterion_ce(logits, labels)

        if use_taca:
            loss_taca = criterion_taca(logits, labels)
            loss = loss_ce + lambda_taca * loss_taca
        else:
            loss = loss_ce

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

    return total_loss / len(train_loader), correct / total


def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def run_experiment(use_taca=False, lambda_taca=0.1, epochs=100, batch_size=16, lr=1e-4):
    """Run single experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: TACA={use_taca}, λ={lambda_taca if use_taca else 'N/A'}")
    print(f"{'='*60}")

    # Data loading
    print("Loading FGVC-Aircraft dataset...")
    transform_train = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = AircraftDataset(AIRCRAFT_DIR, split='train', transform=transform_train)
    val_dataset = AircraftDataset(AIRCRAFT_DIR, split='val', transform=transform_val)
    test_dataset = AircraftDataset(AIRCRAFT_DIR, split='test', transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Build distance matrix
    print("Building aircraft distance matrix...")
    distance_matrix, variants = build_aircraft_distance_matrix(AIRCRAFT_DIR)

    # Model setup
    print("Initializing ResNet-50...")
    model = ResNet50Classifier(num_classes=100, pretrained=True).to(DEVICE)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_taca = TACArithmeticLoss(distance_matrix).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion_ce, criterion_taca,
                                            optimizer, DEVICE, use_taca=use_taca, lambda_taca=lambda_taca)
        val_acc = evaluate(model, val_loader, DEVICE)
        test_acc = evaluate(model, test_loader, DEVICE)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"TrLoss: {train_loss:.4f} | "
                  f"TrAcc: {train_acc:.4f} | "
                  f"ValAcc: {val_acc:.4f} | "
                  f"TestAcc: {test_acc:.4f}")

        scheduler.step()

    # Final evaluation
    final_test_acc = evaluate(model, test_loader, DEVICE)

    result = {
        'use_taca': use_taca,
        'lambda': lambda_taca if use_taca else None,
        'best_val_acc': float(best_val_acc),
        'final_test_acc': float(final_test_acc),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'seed': SEED,
        'device': DEVICE,
        'history': history
    }

    return result


def main():
    """Run all experiments."""
    results = []

    # Experiment 1: Baseline (CE only)
    print("\n" + "="*80)
    print("EXPERIMENT 1: ResNet-50 + CE (Baseline)")
    print("="*80)
    result1 = run_experiment(use_taca=False, epochs=100, batch_size=16, lr=1e-4)
    results.append({
        'name': 'ResNet-50 + CE',
        **result1
    })

    # Experiment 2: TACA
    print("\n" + "="*80)
    print("EXPERIMENT 2: ResNet-50 + CE + TACA (λ=0.1)")
    print("="*80)
    result2 = run_experiment(use_taca=True, lambda_taca=0.1, epochs=100, batch_size=16, lr=1e-4)
    results.append({
        'name': 'ResNet-50 + CE + TACA',
        **result2
    })

    # Save results
    results_file = RESULTS_DIR / 'aircraft_taca_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Best Val Accuracy: {r['best_val_acc']:.4f}")
        print(f"  Final Test Accuracy: {r['final_test_acc']:.4f}")


if __name__ == "__main__":
    main()
