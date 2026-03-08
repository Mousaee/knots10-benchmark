# Knots-10: A Benchmark for Physical Knot Classification with Topology-Guided Representation Learning

> **Shiheng Nie, Xiaoli Liu, Yunguang Yue**
>
> College of Science, Shihezi University

## Overview

**Knots-10** is a benchmark for fine-grained visual classification of physical knots from real-world photographs. Built on the [10Knots dataset](https://www.kaggle.com/datasets/josephcameron/10knots) (1,440 images, 10 knot types), we introduce:

- A **tightness-stratified evaluation protocol** that trains on loosely tied knots and tests on tightly dressed ones
- A **topological distance metric** linking knot-theoretic properties to visual classification difficulty
- **Topology-Aware Centroid Alignment (TACA)**, a regularization loss that aligns learned embeddings with knot-theoretic structure

### Key Results

| Model | Test Acc | Macro F1 |
|-------|----------|----------|
| ResNet-18 | 95.83% | 0.959 |
| ResNet-50 | 96.04% | 0.961 |
| EfficientNet-B0 | 95.21% | 0.952 |
| ViT-B/16 | 95.21% | 0.952 |
| **Swin-T** | **99.38%** | **0.994** |

TACA improves ResNet-18 accuracy by **+1.25 pp** (95.83% → 97.08%) and increases embedding–topology alignment by **40%** (Spearman ρ: 0.46 → 0.65).

## Setup

```bash
pip install -r requirements.txt
```

- Python ≥ 3.10
- PyTorch ≥ 2.0 (MPS backend supported for Apple Silicon)

### Dataset

Download the [10Knots dataset](https://www.kaggle.com/datasets/josephcameron/10knots) from Kaggle and place it in a `train/` directory:

```
train/
  ABK_DL_Loose_01.jpg
  ABK_DL_Set_01.jpg
  ...
```

## Reproducing Results

### 1. Baseline Experiments (Table 1)

```bash
python run_experiments.py
```

Trains ResNet-18, ResNet-50, EfficientNet-B0, ViT-B/16, and Swin-T with the tightness-stratified protocol.

### 2. Topological Analysis (Table 2)

```bash
python topological_analysis.py
```

Computes the topological distance matrix and correlation between topological distance and pairwise confusion rates.

### 3. Topology-Guided Training (Table 4)

```bash
python topo_guided_training.py
```

Trains ResNet-18 and ResNet-50 with TACA regularization and evaluates embedding alignment.

### 4. Loss Ablation (Table 3)

```bash
python loss_ablation.py
```

Evaluates four loss configurations: CE only, CE+TACA, CE+TAML, CE+TACA+TAML.

### 5. Additional Analyses

```bash
python embedding_analysis.py     # Embedding alignment (Table 4)
python weight_sensitivity.py     # Weight sensitivity Monte Carlo
python robustness_multiseed.py   # Multi-seed robustness (Supp. S1)
python phone_inference.py        # Cross-domain evaluation (Supp. S2)
python gradcam_viz.py            # Grad-CAM visualizations
python umap_comparison.py        # t-SNE / UMAP projections (Figure 3)
```

## Repository Structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── run_experiments.py           # Baseline experiments (5 architectures)
├── topo_guided_training.py      # Topology-guided training (TACA + TAML)
├── loss_ablation.py             # Loss component ablation
├── topological_analysis.py      # Topological distance metric
├── embedding_analysis.py        # Embedding alignment analysis
├── weight_sensitivity.py        # Weight sensitivity analysis
├── robustness_multiseed.py      # Multi-seed robustness analysis
├── phone_inference.py           # Cross-domain phone photo evaluation
├── gradcam_viz.py               # Grad-CAM visualization
├── umap_comparison.py           # t-SNE and UMAP projections
├── results/                     # Experiment results (JSON)
└── paper/                       # Manuscript source and figures
```

## Citation

```bibtex
@article{nie2026knots10,
  title={Knots-10: A Benchmark for Physical Knot Classification
         with Topology-Guided Representation Learning},
  author={Nie, Shiheng and Liu, Xiaoli and Yue, Yunguang},
  journal={Scientific Reports},
  year={2026}
}
```

## License

Code: [MIT License](LICENSE). Dataset: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (Joseph Cameron).

## Acknowledgements

This work uses the [10Knots dataset](https://www.kaggle.com/datasets/josephcameron/10knots) by Joseph Cameron. Inspired by the graduate course *Deep Intelligent Computing and Practice* at Shihezi University.
