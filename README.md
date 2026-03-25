# Knots-10: A Tightness-Stratified Benchmark for Physical Knot Classification with Topological Difficulty Analysis

> **Shiheng Nie, Yunguang Yue**
>
> College of Science, Shihezi University
>
> [[arXiv:2603.23286]](https://arxiv.org/abs/2603.23286)

## Overview

**Knots-10** is a tightness-stratified evaluation protocol for fine-grained visual classification of physical knots from real-world photographs. Built on the publicly available [10Knots dataset](https://www.kaggle.com/datasets/josephcameron/10knots) (1,440 images, 10 knot types, CC BY-SA 4.0), we introduce:

- A **tightness-stratified evaluation protocol** that trains on loosely tied knots and tests on tightly dressed ones
- A **diagnostic framework** (Mantel permutation test + random-distance ablation) for evaluating structure-guided training in any FGVC domain
- **Topology-Aware Centroid Alignment (TACA)**, a regularization loss that aligns learned embeddings with knot-theoretic structure

### Key Results

| Model | Test Acc (mean +/- std) | Macro F1 |
|-------|-------------------------|----------|
| ResNet-18 | 96.88 +/- 1.06% | 0.969 |
| ResNet-50 | 95.83 +/- 1.03% | 0.959 |
| EfficientNet-B0 | 96.25 +/- 0.45% | 0.963 |
| ViT-B/16 | 96.39 +/- 0.35% | 0.964 |
| **Swin-T** | **97.22 +/- 1.09%** | **0.972** |
| TransFG | 97.15 +/- 0.94% | 0.972 |
| PMG | 94.51 +/- 1.75% | 0.945 |

TACA improves embedding-topology alignment by **40%** (Spearman rho: 0.46 -> 0.65) without harming classification accuracy.

## Setup

```bash
pip install -r requirements.txt
```

- Python >= 3.10
- PyTorch >= 2.0 (MPS backend supported for Apple Silicon)

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
├── benchmark_config.py             # Class definitions (10-class / 28P)
├── run_experiments.py              # Baseline experiments (5 architectures)
├── topo_guided_training.py         # Topology-guided training (TACA + TAML)
├── loss_ablation.py                # Loss component ablation
├── topological_analysis.py         # Topological distance metric
├── embedding_analysis.py           # Embedding alignment analysis
├── weight_sensitivity.py           # Weight sensitivity analysis
├── robustness_multiseed.py         # Multi-seed robustness analysis
├── phone_inference.py              # Cross-domain phone photo evaluation
├── gradcam_viz.py                  # Grad-CAM visualization
├── umap_comparison.py              # t-SNE and UMAP projections
└── results/                        # Experiment results (JSON)
```

## Citation

```bibtex
@article{nie2026knots10,
  title={Knots-10: A Tightness-Stratified Benchmark for Physical Knot
         Classification with Topological Difficulty Analysis},
  author={Nie, Shiheng and Yue, Yunguang},
  journal={arXiv preprint arXiv:2603.23286},
  year={2026}
}
```

## License

Code: [MIT License](LICENSE). Dataset: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (Joseph Cameron).

## Acknowledgements

This work uses the [10Knots dataset](https://www.kaggle.com/datasets/josephcameron/10knots) by Joseph Cameron. Inspired by the graduate course *Deep Intelligent Computing and Practice* at Shihezi University.
