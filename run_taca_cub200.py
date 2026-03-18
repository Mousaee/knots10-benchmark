"""
TACA (Taxonomy-Aware Contrastive Alignment) Training on CUB-200-2011

This script trains and evaluates a ResNet-50 classifier on the CUB-200-2011 bird species
dataset using two approaches:
1. Cross-Entropy (CE) baseline
2. CE + TACA loss (taxonomy-guided loss)

The TACA loss encourages learned embeddings to align with bird taxonomic hierarchy.

Environment:
    CUB_DIR: Path to CUB_200_2011 dataset root (default: ~/data/CUB_200_2011)

Usage:
    python run_taca_cub200.py

Output:
    results/cub200_taca_results.json
        - baseline: CE-only performance metrics
        - taca: CE+TACA performance metrics
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from PIL import Image
from torchvision import transforms, models
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from cub200_taxonomy import build_cub200_distance_matrix

# Suppress warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Device Detection
# ============================================================================

def get_device() -> torch.device:
    """Detect and return appropriate device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


# ============================================================================
# Dataset
# ============================================================================

class CUBDataset(Dataset):
    """
    CUB-200-2011 Dataset loader.

    Expected directory structure:
        CUB_200_2011/
        ├── images/
        │   ├── 001.Black_footed_Albatross/
        │   │   └── Black_Footed_Albatross_0001_796111.jpg
        │   ├── 002.Laysan_Albatross/
        │   ├── ...
        │   └── 200.Common_Yellowthroat/
        ├── classes.txt
        ├── image_class_labels.txt
        ├── images.txt
        └── train_test_split.txt
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize CUB dataset.

        Args:
            root: Path to CUB_200_2011 directory
            split: 'train' or 'test'
            transform: Optional image transform pipeline
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []

        # Parse images.txt: image_id filepath
        images_file = os.path.join(root, 'images.txt')
        id_to_filepath = {}
        if os.path.exists(images_file):
            with open(images_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    image_id = int(parts[0])
                    filepath = ' '.join(parts[1:])
                    id_to_filepath[image_id] = filepath

        # Parse image_class_labels.txt: image_id class_id (1-indexed)
        labels_file = os.path.join(root, 'image_class_labels.txt')
        id_to_label = {}
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    image_id = int(parts[0])
                    class_id = int(parts[1])
                    id_to_label[image_id] = class_id - 1  # Convert to 0-indexed

        # Parse train_test_split.txt: image_id is_train (1=train, 0=test)
        split_file = os.path.join(root, 'train_test_split.txt')
        id_to_split = {}
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    image_id = int(parts[0])
                    is_train = int(parts[1])
                    id_to_split[image_id] = is_train

        # Build dataset with split filtering
        split_value = 1 if split == 'train' else 0
        for image_id in sorted(id_to_filepath.keys()):
            if id_to_split.get(image_id, 1) != split_value:
                continue

            filepath = id_to_filepath[image_id]
            label = id_to_label.get(image_id, -1)

            if label >= 0:
                self.images.append(filepath)
                self.labels.append(label)

        print(f"CUB {split} split: {len(self.images)} images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath = self.images[idx]
        label = self.labels[idx]

        # Construct full image path
        img_path = os.path.join(self.root, 'images', filepath)

        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            # Return a dummy black image as fallback
            img = Image.new('RGB', (448, 448), color='black')

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================================
# TACA Loss (Taxonomy-Aware Contrastive Alignment)
# ============================================================================

class TaxonomyGuidedLoss(nn.Module):
    """
    Taxonomy-guided contrastive loss for aligning learned embeddings with
    taxonomic hierarchy.

    The loss encourages embeddings of taxonomically similar species to be
    closer in embedding space than taxonomically distant species.

    Args:
        taxonomy_dist_matrix: 200×200 numpy array of taxonomic distances
        lambda_taca: Weight of taxonomy loss vs CE loss
        temperature: Temperature for softmax scaling (default: 0.1)
        device: Device to place tensors on
    """

    def __init__(
        self,
        taxonomy_dist_matrix: np.ndarray,
        lambda_taca: float = 0.1,
        temperature: float = 0.1,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__()
        self.lambda_taca = lambda_taca
        self.temperature = temperature

        # Store taxonomy distance matrix as buffer (no gradient)
        self.register_buffer(
            'taxonomy_dist',
            torch.from_numpy(taxonomy_dist_matrix).float().to(device)
        )

    def forward(
        self,
        embeddings: torch.Tensor,  # (batch_size, embed_dim)
        labels: torch.Tensor,  # (batch_size,)
    ) -> torch.Tensor:
        """
        Compute taxonomy-guided contrastive loss.

        Loss encourages the learned pairwise distances in embedding space
        to correlate with taxonomic distances.

        Args:
            embeddings: Batch of learned embeddings
            labels: Batch of class labels (0-199)

        Returns:
            Scalar loss value
        """
        batch_size = embeddings.shape[0]

        # Compute pairwise L2 distances in embedding space
        # dist[i,j] = ||embedding[i] - embedding[j]||_2
        dists = torch.cdist(embeddings, embeddings, p=2)  # (batch_size, batch_size)

        # Extract corresponding taxonomic distances
        # tax_dists[i,j] = taxonomy_dist[labels[i], labels[j]]
        labels_expanded_i = labels.unsqueeze(1)  # (batch_size, 1)
        labels_expanded_j = labels.unsqueeze(0)  # (1, batch_size)
        tax_dists = self.taxonomy_dist[labels_expanded_i, labels_expanded_j]

        # Normalize distances to [0, 1]
        dists_norm = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)

        # Compute contrastive alignment loss
        # MSE between normalized embedding distances and taxonomy distances
        loss = ((dists_norm - tax_dists) ** 2).mean()

        return loss


class CombinedLoss(nn.Module):
    """Cross-Entropy + TACA loss."""

    def __init__(
        self,
        taxonomy_dist_matrix: np.ndarray,
        lambda_taca: float = 0.1,
        num_classes: int = 200,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.taca_loss = TaxonomyGuidedLoss(
            taxonomy_dist_matrix,
            lambda_taca=lambda_taca,
            device=device,
        )
        self.lambda_taca = lambda_taca

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Classification logits (batch_size, num_classes)
            embeddings: Feature embeddings (batch_size, embed_dim)
            labels: Class labels (batch_size,)

        Returns:
            Combined loss: CE + lambda_taca * TACA
        """
        ce = self.ce_loss(logits, labels)
        taca = self.taca_loss(embeddings, labels)
        return ce + self.lambda_taca * taca


# ============================================================================
# Model
# ============================================================================

class ResNet50Classifier(nn.Module):
    """ResNet-50 with learnable embedding layer for TACA."""

    def __init__(self, num_classes: int = 200, embed_dim: int = 512):
        super().__init__()
        # Load pretrained ResNet-50
        backbone = models.resnet50(pretrained=True)

        # Remove final FC layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone_out_dim = backbone.fc.in_features

        # Add embedding layer
        self.embedding = nn.Linear(self.backbone_out_dim, embed_dim)
        self.embed_dim = embed_dim

        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images (batch_size, 3, H, W)

        Returns:
            logits: Classification logits (batch_size, num_classes)
            embeddings: Learned embeddings (batch_size, embed_dim)
        """
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        embeddings = self.embedding(features)
        logits = self.classifier(embeddings)
        return logits, embeddings


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, embeddings = model(images)

        # Loss computation depends on loss function type
        if isinstance(loss_fn, CombinedLoss):
            loss = loss_fn(logits, embeddings, labels)
        else:
            loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.shape[0]

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_embeddings = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits, embeddings = model(images)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_embeddings.append(embeddings.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_embeddings = np.vstack(all_embeddings)

    # Accuracy and F1
    test_acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Embedding-taxonomy alignment (Spearman correlation)
    embed_spearman_rho, embed_spearman_p = compute_embed_spearman_rho(
        all_embeddings, all_labels, device
    )

    return {
        'test_acc': float(test_acc),
        'macro_f1': float(macro_f1),
        'embed_spearman_rho': float(embed_spearman_rho),
        'embed_spearman_p': float(embed_spearman_p),
    }


def compute_embed_spearman_rho(
    embeddings: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Compute Spearman correlation between pairwise distances in embedding space
    and taxonomic distances.

    This metric measures how well the learned embeddings align with the
    taxonomic hierarchy.

    Args:
        embeddings: (n_samples, embed_dim)
        labels: (n_samples,) class labels 0-199
        device: Device (for loading distance matrix)

    Returns:
        Spearman ρ and p-value
    """
    # Limit to 500 samples to avoid memory issues
    if len(embeddings) > 500:
        idx = np.random.choice(len(embeddings), 500, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform

    embed_dist = squareform(pdist(embeddings, metric='euclidean'))

    # Get taxonomy distances
    tax_dist_matrix = build_cub200_distance_matrix()
    tax_dist = tax_dist_matrix[np.ix_(labels, labels)]

    # Flatten upper triangles
    embed_dist_vec = embed_dist[np.triu_indices_from(embed_dist, k=1)]
    tax_dist_vec = tax_dist[np.triu_indices_from(tax_dist, k=1)]

    # Normalize to [0, 1]
    embed_dist_vec = (embed_dist_vec - embed_dist_vec.min()) / (
        embed_dist_vec.max() - embed_dist_vec.min() + 1e-8
    )

    # Compute Spearman correlation
    rho, p_value = spearmanr(embed_dist_vec, tax_dist_vec)

    return rho, p_value


def train_model(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    model_name: str = 'model',
) -> Dict[str, float]:
    """Train and evaluate model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    best_acc = 0.0

    for epoch in range(epochs):
        avg_loss = train_epoch(model, loss_fn, optimizer, train_loader, device)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Final evaluation
    print("\nFinal Evaluation...")
    metrics = evaluate(model, test_loader, device)

    print(f"Test Accuracy: {metrics['test_acc']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Embedding-Taxonomy Spearman ρ: {metrics['embed_spearman_rho']:.4f}")
    print(f"Embedding-Taxonomy Spearman p: {metrics['embed_spearman_p']:.6f}")

    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TACA on CUB-200-2011 Bird Classification'
    )
    parser.add_argument(
        '--cub_dir',
        type=str,
        default=os.path.expanduser('~/data/CUB_200_2011'),
        help='Path to CUB_200_2011 dataset',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory for results',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate',
    )
    parser.add_argument(
        '--lambda_taca',
        type=float,
        default=0.1,
        help='Weight for TACA loss',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = get_device()

    # Check dataset
    cub_dir = args.cub_dir
    if not os.path.exists(cub_dir):
        print(
            f"\n{'!'*60}\n"
            f"CUB-200-2011 dataset not found at {cub_dir}\n\n"
            f"Download instructions:\n"
            f"1. Visit: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html\n"
            f"2. Download: CUB_200_2011.tgz\n"
            f"3. Extract to: {os.path.dirname(cub_dir)}/\n"
            f"   Or set CUB_DIR environment variable\n"
            f"{'!'*60}\n"
        )
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Load datasets
    print("\nLoading CUB-200-2011 dataset...")
    train_dataset = CUBDataset(cub_dir, split='train', transform=train_transform)
    test_dataset = CUBDataset(cub_dir, split='test', transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Load taxonomy distance matrix
    print("Building taxonomy distance matrix...")
    tax_dist_matrix = build_cub200_distance_matrix()

    # ========== Experiment 1: CE Baseline ==========
    print("\n" + "="*60)
    print("EXPERIMENT 1: CE Baseline")
    print("="*60)

    model_baseline = ResNet50Classifier(num_classes=200, embed_dim=512).to(device)
    optimizer_baseline = optim.AdamW(model_baseline.parameters(), lr=args.lr)
    scheduler_baseline = CosineAnnealingLR(optimizer_baseline, T_max=args.epochs)
    loss_fn_baseline = nn.CrossEntropyLoss()

    baseline_metrics = train_model(
        model_baseline,
        loss_fn_baseline,
        optimizer_baseline,
        scheduler_baseline,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        model_name='ResNet-50 + CE',
    )

    # ========== Experiment 2: CE + TACA ==========
    print("\n" + "="*60)
    print("EXPERIMENT 2: CE + TACA")
    print("="*60)

    model_taca = ResNet50Classifier(num_classes=200, embed_dim=512).to(device)
    optimizer_taca = optim.AdamW(model_taca.parameters(), lr=args.lr)
    scheduler_taca = CosineAnnealingLR(optimizer_taca, T_max=args.epochs)
    loss_fn_taca = CombinedLoss(
        tax_dist_matrix,
        lambda_taca=args.lambda_taca,
        num_classes=200,
        device=device,
    ).to(device)

    taca_metrics = train_model(
        model_taca,
        loss_fn_taca,
        optimizer_taca,
        scheduler_taca,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        model_name='ResNet-50 + CE + TACA',
    )
    taca_metrics['lambda_taca'] = args.lambda_taca

    # ========== Save Results ==========
    results = {
        'baseline': baseline_metrics,
        'taca': taca_metrics,
    }

    output_file = os.path.join(args.output_dir, 'cub200_taca_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}\n")

    # Print summary
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<35} {'Baseline':<15} {'TACA':<15}")
    print(f"{'-'*65}")
    print(f"{'Test Accuracy':<35} {baseline_metrics['test_acc']:<15.4f} {taca_metrics['test_acc']:<15.4f}")
    print(f"{'Macro F1':<35} {baseline_metrics['macro_f1']:<15.4f} {taca_metrics['macro_f1']:<15.4f}")
    print(f"{'Embedding-Taxonomy Spearman ρ':<35} {baseline_metrics['embed_spearman_rho']:<15.4f} {taca_metrics['embed_spearman_rho']:<15.4f}")
    print(f"{'-'*65}\n")


if __name__ == '__main__':
    main()
