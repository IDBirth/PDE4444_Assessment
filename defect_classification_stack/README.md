# PDE4444 – Defect Classification Stack

ML-Based Visual Quality Inspection System – assessment submission code.

## Assessment alignment

| Section | Requirement | Script |
|---------|-------------|--------|
| S1 | Engineering problem definition | *(report)* |
| S2 | Dataset, features, dimensionality N | `prepare_dataset.py`, `train_sklearn_baseline.py` |
| S3 | CNN ≥3 hidden layers, ≥2 activations | `train_cnn_activation_sweep.py` |
| S3 | Zero-order / first-order / second-order optimisers | `train_optimizer_comparison.py` |
| S3 | Convergence plots | generated automatically by both S3 scripts |
| S4 | Baseline comparison (SVM, sklearn MLP) | `train_sklearn_baseline.py` |
| S5 | Train/val/test split | `prepare_dataset.py` |
| S5 | Cross-validation | `train_cross_validation.py` |
| S5 | Overfitting analysis | `train_cross_validation.py` → `overfitting_analysis.png` |

## Dataset layout (linked from scaffold)

```
data/  -> ../zeroq_cup_classification_scaffold/data
├── raw/
│   ├── defective/
│   └── non_defective/
├── processed/
│   ├── train/  defective/ non_defective/
│   ├── val/    defective/ non_defective/
│   └── test/   defective/ non_defective/
└── interim/   (augmented / balanced splits)
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything
bash run.sh data/processed runs/
```

Or run individual scripts:

### Section 4 – scikit-learn baselines (HOG + SVM / sklearn MLP)

```bash
python train_sklearn_baseline.py \
  --data-dir data/processed \
  --output-dir runs/sklearn_baseline
```

### Section 3 – CNN activation sweep (PyTorch)

Compares **ReLU, ELU, GELU, SELU, LeakyReLU** on a 3-block CNN.
Outputs `activation_convergence.png` per-activation confusion matrices.

```bash
python train_cnn_activation_sweep.py \
  --data-dir data/processed \
  --output-dir runs/cnn_activation_sweep \
  --epochs 20
```

### Section 3 – Optimiser comparison

Compares **Nelder-Mead** (zero-order), **SGD** (first-order),
**Adam** (first-order), and **L-BFGS** (second-order) on the same
small MLP trained on HOG + PCA features.
Outputs `optimizer_convergence.png` and `optimizer_metrics_bar.png`.

```bash
python train_optimizer_comparison.py \
  --data-dir   data/processed \
  --output-dir runs/optimizer_comparison \
  --epochs 50 \
  --nm-iters 800
```

### Section 5 – Cross-validation + overfitting analysis

Stratified 5-fold CV on pooled train+val, final eval on held-out test.
Outputs `overfitting_analysis.png` and per-model `learning_curve.png`.

```bash
python train_cross_validation.py \
  --data-dir   data/processed \
  --output-dir runs/cross_validation
```

### MLP hyperparameter random search (PyTorch)

```bash
python train_keras_mlp_random_search.py \
  --data-dir   data/processed \
  --output-dir runs/mlp_hparam_search \
  --max-trials 12 \
  --img-size 64
```

### YOLO26n classification

```bash
python train_yolo26_cls.py \
  --data-dir   data/processed \
  --output-dir runs/yolo26_cls \
  --epochs 30
```

### Aggregate comparison table

```bash
python compare_results.py \
  --sklearn-dir runs/sklearn_baseline \
  --keras-dir   runs/mlp_hparam_search \
  --cnn-dir     runs/cnn_activation_sweep \
  --yolo-dir    runs/yolo26_cls \
  --output-file runs/model_comparison.csv
```

## Feature dimensionality (Section 2)

| Model family | Feature representation | Dimensionality N |
|---|---|---|
| SVM / sklearn MLP | HOG on 128×128 grey | 8 100 |
| Optimiser MLP | HOG → PCA | 20 |
| CNN | Raw RGB pixels (224×224×3) | 150 528 |
| YOLO26n | Pretrained backbone features | internal |

## Notes

- The environment uses **PyTorch** (not TensorFlow). The
  `train_keras_mlp_random_search.py` file has been rewritten to use
  PyTorch; the filename is kept for backwards compatibility with
  `compare_results.py`.
- Class imbalance is handled via `pos_weight` (PyTorch scripts) and
  `class_weight="balanced"` (sklearn scripts).
- All scripts accept `--seed` for reproducibility.
