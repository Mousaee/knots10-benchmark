# Knots-10: A Benchmark for Physical Knot Classification with Topology-Guided Representation Learning

This repository contains the code, results, and manuscript for the paper:

> **Knots-10: A Benchmark for Physical Knot Classification with Topology-Guided Representation Learning**
>
> Shiheng Nie, College of Science, Shihezi University

## Overview

We present **Knots-10**, a benchmark for fine-grained visual classification of physical knots from real-world photographs. Built on the [10Knots dataset](https://www.kaggle.com/datasets/josephcameron/10knots) (1,440 images, 10 knot types), we introduce:

- A **tightness-stratified evaluation protocol** that trains on loosely tied knots and tests on tightly dressed ones
- A **topological distance metric** linking knot-theoretic properties to visual classification difficulty
- **Topology-Aware Centroid Alignment (TACA)**, a regularization loss that aligns learned embeddings with knot-theoretic structure

### Key Results

| Model | Test Acc | Macro F1 |
|-------|----------|----------|
| ResNet-18 | 95.83% | 0.96 |
| ResNet-50 | 96.04% | 0.96 |
| EfficientNet-B0 | 95.21% | 0.95 |
| ViT-B/16 | 95.21% | 0.95 |
| **Swin-T** | **99.38%** | **0.99** |

Topology-guided training (TACA) improves ResNet-18 accuracy by **1.25 percentage points** (95.83% -> 97.08%) and increases embedding-topology alignment by **40%** (Spearman rho: 0.46 -> 0.65).

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

- Python >= 3.10
- PyTorch >= 2.0 (MPS backend supported for Apple Silicon)

### Dataset

Download the [10Knots dataset](https://www.kaggle.com/datasets/josephcameron/10knots) from Kaggle and place it in a `train/` directory at the project root:

```
knots10-benchmark/
  train/
    ABK_*.jpg
    BK_*.jpg
    ...
```

## Reproducing Results

### 1. Baseline Experiments (5 architectures)

```bash
python run_experiments.py
```

Trains ResNet-18, ResNet-50, EfficientNet-B0, ViT-B/16, and Swin-T with the tightness-stratified protocol. Results are saved to `results/`.

### 2. Topological Analysis

```bash
python topological_analysis.py
```

Computes the topological distance matrix and correlation analysis between topological distance and pairwise confusion rates.

### 3. Topology-Guided Training

```bash
python topo_guided_training.py
```

Trains ResNet-18 and ResNet-50 with TACA regularization and evaluates embedding alignment.

### 4. Loss Ablation Study

```bash
python loss_ablation.py
```

Evaluates four loss configurations: CE only, CE+TACA, CE+TAML, CE+TACA+TAML.

### 5. Additional Analyses

```bash
python embedding_analysis.py    # Embedding alignment analysis
python weight_sensitivity.py    # Weight sensitivity Monte Carlo
python gradcam_viz.py           # Grad-CAM visualizations
python umap_comparison.py       # t-SNE and UMAP projections
python ablation_study.py        # Cross-condition ablation
```

### 6. Regenerate All Figures

```bash
python regenerate_all_figures.py
python convert_to_pdf.py
```

## Repository Structure

```
knots10-benchmark/
├── README.md
├── requirements.txt
├── .gitignore
│
├── run_experiments.py           # Main baseline experiments
├── run_extra_models.py          # Additional model training
├── topo_guided_training.py      # Topology-guided training (TACA + TAML)
├── loss_ablation.py             # Loss component ablation study
├── topological_analysis.py      # Topological distance and correlation
├── embedding_analysis.py        # Embedding alignment analysis
├── weight_sensitivity.py        # Weight sensitivity analysis
├── gradcam_viz.py               # Grad-CAM visualization
├── umap_comparison.py           # t-SNE and UMAP projections
├── ablation_study.py            # Cross-condition ablation
├── regenerate_all_figures.py    # Regenerate all paper figures
├── generate_v2_figures.py       # Generate v2 figures
├── convert_to_pdf.py            # Convert PNG figures to PDF
├── viz_embed.py                 # Embedding visualization
├── run_embed.py                 # Embedding extraction runner
├── run_loss_ablation.py         # Loss ablation runner
├── run_remaining_topo.py        # Remaining topo experiments runner
│
├── results/                     # Experiment results
│   ├── *_results.json           # Per-model result files
│   ├── loss_ablation.json       # Ablation study results
│   ├── topological_analysis.json
│   ├── embedding_alignment.json
│   ├── weight_sensitivity.json
│   └── figures/                 # Generated figures
│
└── paper/                       # Manuscript
    ├── main.tex
    ├── references.bib
    └── figures/                 # Paper figures
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{nie2026knots10,
  title={Knots-10: A Benchmark for Physical Knot Classification with Topology-Guided Representation Learning},
  author={Nie, Shiheng},
  journal={Scientific Reports},
  year={2026}
}
```

## License

The 10Knots dataset is released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) by Joseph Cameron. Our code is released under the MIT License.

## Acknowledgements

This work uses the [10Knots dataset](https://www.kaggle.com/datasets/josephcameron/10knots) created by Joseph Cameron.
