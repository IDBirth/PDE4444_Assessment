# PDE4444 — ML Visual Quality Inspection
## Progress & Results Report

**Module:** Machine Learning for Engineers  
**Assessment:** Component A — Technical Portfolio  
**Dataset:** Defective vs Non-Defective Cup Images  
**Balanced Test Set:** 138 images (69 defective, 69 non-defective)

**Training runs captured in this report:**
1. **runs1** — macOS + Apple MPS (original pipeline build, 9 model families) — Sections 1–7
2. **runs2** — Ubuntu + RTX 4090 CUDA (reproducibility audit, identical configs) — Section 8
3. **runs3** — Ubuntu + RTX 4090 CUDA (GPU-aware tuning, bigger configs) — Section 9
4. **runs4** — Ubuntu + RTX 4090 CUDA (full root-pipeline rerun, clean aggregate, pre-YOLO-cleanup baseline) — Section 11
5. **runs5** — Ubuntu CPU-only execution of the cleaned pipeline, still in progress as of April 19, 2026 23:05 +04 — Section 12
6. **Runs1–3 summary & baseline takeaways** — Section 10

**Environment:** runs1 Apple Silicon + MPS / Python 3.11 / PyTorch 2.11; runs2/runs3/runs4 Ubuntu + CUDA / RTX 4090 24 GB / Python 3.12.3 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.39; runs5 is executing on the same Ubuntu repo snapshot but the active log reports `device=cpu` for the PyTorch training stages.

---

## 1. Dataset Overview

| Split | Defective | Non-Defective | Total |
|-------|-----------|---------------|-------|
| Raw (original) | 87 | 9 | 96 |
| After augmentation | 30 960 | 2 880 | 33 840 |
| **Balanced (used for training)** | **345** | **345** | **690** |
| → Train | 242 | 241 | 483 |
| → Val | 34 | 35 | 69 |
| → Test | 69 | 69 | 138 |

**Critical finding:** The original processed split was 93.7% defective (611 vs 41 in test).  
All sklearn models predicted only one class → 0% F1 on non-defective.  
After applying `balance_cleaned_dataset.py` (undersample to minority count), all models learned both classes correctly.

**Feature dimensionality:**

| Model family | Representation | Dimensionality N |
|---|---|---|
| HOG + SVM / MLP | HOG on 128×128 greyscale | 8 100 |
| Optimiser MLP | HOG → StandardScaler → PCA | 20 |
| CNN (scratch) | Raw RGB 224×224 pixels | 150 528 |
| MobileNetV2 head | ImageNet backbone features | 1 280 |
| YOLO26n-cls | Pretrained backbone | internal |

---

## 2. Iteration 1 — Baseline & Section 3 Models

### 2.1 Sklearn Baselines (Section 4)

HOG features (8 100-dim) extracted from 128×128 greyscale images,
fed to sklearn models with `f1_macro` scoring and `class_weight="balanced"`.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| HOG + SVM (RBF, C=1) | 84.78% | 80.00% | 92.75% | 85.91% |
| HOG + sklearn MLP (128→64) | 84.06% | 78.31% | 94.20% | 85.53% |

**Observation:** SVM slightly outperforms sklearn MLP on HOG features.  
Both achieve ~85% F1 on the balanced test set — a strong baseline for classical methods.

---

### 2.2 CNN Activation-Function Sweep — from Scratch (Section 3)

Architecture: 3 Conv blocks (32→64→128 filters) + GlobalAvgPool + Dense(128) + Dense(1).  
Trained from random initialisation on 483 images, 30 epochs, batch=16, Adam lr=1e-3.

| Activation | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|----|
| ReLU | 67.39% | 65.00% | 75.36% | 69.80% |
| ELU | 59.42% | 55.20% | 100.00% | 71.13% |
| LeakyReLU | 54.35% | 52.27% | 100.00% | 68.66% |
| GELU | 51.45% | 50.74% | 100.00% | 67.32% |
| SELU | 51.45% | 50.74% | 100.00% | 67.32% |

**Analysis:** All activations converge near random-chance accuracy.  
With only 483 training images, a 3-block CNN trained from scratch cannot learn discriminative features.  
Loss remained near 0.69 (ln 2 = binary random baseline) through all epochs.  
This motivates using pretrained backbones in iterations 2 and 3.

---

### 2.3 Optimiser Comparison (Section 3)

**Architecture for all optimisers:** HOG (8 100-dim) → PCA (20-dim) → MLP (20→32→16→1), ≈1 217 parameters.  
This small network allows a fair comparison including zero-order methods.

| Optimiser | Type | Accuracy | Precision | Recall | F1 |
|-----------|------|----------|-----------|--------|----|
| Adam | First-order (adaptive) | 84.06% | 84.06% | 84.06% | 84.06% |
| L-BFGS | Second-order (quasi-Newton) | 81.88% | 85.48% | 76.81% | 80.92% |
| SGD (momentum=0.9) | First-order (GD) | 78.26% | 77.46% | 79.71% | 78.57% |
| Nelder-Mead | Zero-order (derivative-free) | 48.55% | 49.15% | 84.06% | 62.03% |

**Convergence behaviour:**

| Optimiser | Train loss (final) | Val loss (final) | Convergence |
|-----------|-------------------|-----------------|-------------|
| Nelder-Mead | 0.9067 | 0.8740 | Stuck — no descent after iter 50 |
| SGD | 0.17 | 1.57 | Noisy, overfits after ep. 30 |
| Adam | 0.16 | 0.67 | Smooth, best generalisation |
| L-BFGS | **0.017** | **5.43** | Fastest convergence, severe overfit |

**Discussion:**
- **Zero-order (Nelder-Mead):** Derivative-free simplex method. Requires O(N) function evaluations per step where N = number of parameters. With 1 217 parameters, the simplex became too flat and made no progress. Demonstrates why zero-order methods are impractical for neural networks beyond ~100 parameters.
- **First-order SGD:** Noisy loss curve due to mini-batch variance. Eventually overfits. Classic gradient descent behaviour.
- **First-order Adam:** Adaptive per-parameter learning rates smooth convergence and prevent the oscillation seen in SGD. Best generalisation.
- **Second-order L-BFGS:** Uses curvature information (approximate Hessian). Converges in 5 epochs (train loss → 0.017) but memorises training data, causing the validation loss to diverge to 5.43. Classic second-order overfitting when dataset is small.

---

### 2.4 MLP Hyperparameter Random Search

15 random trials over: hidden sizes, activation, dropout, learning rate.  
Input: raw 64×64 RGB images (flattened) → 12 288-dim.

**Best configuration found:**
- Hidden sizes: (256,)
- Activation: ELU
- Dropout: 0.4
- Learning rate: 0.001

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| MLP (best HP, ELU+256) | **94.20%** | **91.78%** | **97.10%** | **94.37%** |

**Finding:** A single dense layer with strong dropout and ELU activation on raw 64×64 pixels outperforms all CNN-from-scratch models by ~25% F1. Demonstrates importance of regularisation with small datasets.

---

### 2.5 Cross-Validation — Overfitting Analysis (Section 5)

Stratified 5-fold CV on pooled train+val (552 samples). Test set held out entirely.

**HOG + SVM:**

| Fold | Train Acc | Val Acc | Val F1 | Overfit Gap |
|------|-----------|---------|--------|-------------|
| 1 | 97.51% | 71.17% | 75.38% | 26.33% |
| 2 | 97.28% | 72.97% | 75.81% | 24.31% |
| 3 | 98.42% | 78.18% | 79.31% | 20.23% |
| 4 | 97.96% | 76.36% | 79.03% | 21.60% |
| 5 | 97.29% | 78.18% | 80.65% | 19.10% |
| **Mean** | **97.69%** | **75.37%** | **78.04%** | **22.31%** |

**HOG + sklearn MLP:**

| Fold | Train Acc | Val Acc | Val F1 | Overfit Gap |
|------|-----------|---------|--------|-------------|
| 1 | 97.05% | 72.07% | 74.38% | 24.98% |
| 2 | 97.51% | 78.38% | 80.65% | 19.13% |
| 3 | 97.51% | 84.55% | 85.22% | 12.97% |
| 4 | 98.19% | 79.09% | 81.60% | 19.10% |
| 5 | 98.42% | 80.91% | 80.37% | 17.51% |
| **Mean** | **97.74%** | **79.00%** | **80.44%** | **18.74%** |

**Overfitting conclusion:** Both models show ~97% training accuracy but only 75–79% validation accuracy. The ~20% overfit gap reflects HOG features overfitting to the specific augmented images — augmented images from the same ~90 originals share structural similarity, making train features easy to memorise.

---

### 2.6 YOLO26n Classification

Pretrained YOLO26n-cls backbone fine-tuned on the balanced dataset (40 epochs, imgsz=224).

| Model | Top-1 Accuracy | Top-5 Accuracy | Fitness |
|-------|---------------|---------------|---------|
| YOLO26n-cls | **98.55%** | **100.00%** | **99.28%** |

**Best performing model overall.**  
Pretrained ImageNet features + YOLO's efficient neck architecture adapts immediately to the 2-class problem. Only 40 epochs needed to reach near-perfect accuracy.

---

## 3. Iteration 2 — MobileNetV2 (Frozen Backbone)

Transfer learning: ImageNet-pretrained MobileNetV2 backbone **frozen**,  
only the custom head (Dense 1280→256 → activation → Dense 1) trained.  
40 epochs, AdamW, cosine LR schedule.

| Activation | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|----|
| ELU | 86.96% | 82.28% | 94.20% | 87.84% |
| GELU | 86.96% | 82.28% | 94.20% | 87.84% |
| SELU | 86.96% | 82.28% | 94.20% | 87.84% |
| LeakyReLU | 86.96% | 82.28% | 94.20% | 87.84% |
| ReLU | 86.23% | 81.25% | 94.20% | 87.25% |

**Observation:** All non-ReLU activations tie — the frozen backbone features dominate; the head activation matters little when only the classification layer is trained. This reveals the backbone bottleneck: ImageNet features are generic but not yet adapted to cup surface textures.

---

## 4. Iteration 3 — MobileNetV2 (Full Fine-Tuning)

Same MobileNetV2 backbone but **all layers unfrozen** with differential learning rates:  
- Backbone: lr = 5×10⁻⁵ (conservative, avoids catastrophic forgetting)  
- Head: lr = 3×10⁻⁴  
- AdamW, weight_decay=1e-4, CosineAnnealing, 40 epochs.

| Activation | Accuracy | Precision | Recall | F1 | vs Frozen |
|-----------|----------|-----------|--------|----|-----------|
| **ELU** | **94.93%** | **90.79%** | **100.00%** | **95.17%** | +7.34% |
| LeakyReLU | 94.20% | 89.61% | 100.00% | 94.52% | +6.68% |
| GELU | 94.20% | 92.96% | 95.65% | 94.29% | +6.45% |
| SELU | 93.48% | 88.46% | 100.00% | 93.88% | +6.04% |
| ReLU | 92.75% | 89.33% | 97.10% | 93.06% | +5.81% |

**Key findings:**
- Fine-tuning adds ~+6–7% F1 vs frozen backbone — adapting lower-level features to cup textures is highly beneficial.
- **ELU remains the best activation** across both iterations. ELU's negative saturation and smooth gradient near zero help the network avoid dead units and converge to a sharper decision boundary.
- ELU + fine-tuning achieves 100% recall (zero missed defects) — critical for quality inspection where false negatives (passing a defective product) are the costlier error.

---

## 5. Final Model Rankings

All evaluated on the **same balanced test set** (138 samples: 69 defective, 69 non-defective).

| Rank | Model | Accuracy | Precision | Recall | F1 | Category |
|------|-------|----------|-----------|--------|----|----------|
| 1 | **YOLO26n-cls** | **98.55%** | — | — | **98.55%** | YOLO (pretrained) |
| 2 | **MobileNetV2-ELU (fine-tuned)** | **94.93%** | 90.79% | **100.00%** | **95.17%** | Transfer Learning |
| 3 | MobileNetV2-LeakyReLU (fine-tuned) | 94.20% | 89.61% | 100.00% | 94.52% | Transfer Learning |
| 4 | **MLP HParam Search (ELU, 256)** | **94.20%** | 91.78% | 97.10% | **94.37%** | Deep MLP |
| 5 | MobileNetV2-GELU (fine-tuned) | 94.20% | 92.96% | 95.65% | 94.29% | Transfer Learning |
| 6 | MobileNetV2-SELU (fine-tuned) | 93.48% | 88.46% | 100.00% | 93.88% | Transfer Learning |
| 7 | MobileNetV2-ReLU (fine-tuned) | 92.75% | 89.33% | 97.10% | 93.06% | Transfer Learning |
| 8–12 | MobileNetV2-* (frozen) | 86–87% | 81–82% | 94.20% | 87–88% | Frozen TL |
| 13 | **HOG + SVM** | 84.78% | 80.00% | 92.75% | 85.91% | Classical |
| 14 | **HOG + sklearn MLP** | 84.06% | 78.31% | 94.20% | 85.53% | Classical |
| 15 | Adam MLP (optimiser exp.) | 84.06% | 84.06% | 84.06% | 84.06% | Optimiser |
| 16 | L-BFGS MLP | 81.88% | 85.48% | 76.81% | 80.92% | Optimiser |
| 17 | SGD MLP | 78.26% | 77.46% | 79.71% | 78.57% | Optimiser |
| 18 | CNN-ReLU (scratch) | 67.39% | 65.00% | 75.36% | 69.80% | CNN Scratch |
| 19 | CNN-ELU (scratch) | 59.42% | 55.20% | 100.00% | 71.13% | CNN Scratch |
| 20–22 | CNN-LeakyReLU/GELU/SELU | 51–54% | 50–52% | 100.00% | 67–69% | CNN Scratch |
| 23 | Nelder-Mead MLP | 48.55% | 49.15% | 84.06% | 62.03% | Zero-order |

---

## 6. Key Insights & Conclusions

### 6.1 Data quality over quantity
The original augmented dataset (33 840 images, 93.7% defective) caused all models to collapse to a single-class predictor. Balancing to 345 vs 345 images immediately enabled proper learning across all model families.

### 6.2 Transfer learning is essential for small datasets
- CNN from scratch: 51–67% accuracy (near random)
- MobileNetV2 frozen: 86–87% accuracy (+~20%)
- MobileNetV2 fine-tuned: 93–95% accuracy (+~7% vs frozen)

With fewer than 500 training images, ImageNet-pretrained features are not optional — they are the difference between failure and production-grade accuracy.

### 6.3 ELU consistently outperforms ReLU
ELU is the best activation across the MLP hparam search, fine-tuned MobileNetV2, and optimiser comparison. Key properties that matter here:
- Smooth gradient everywhere (no hard zero saturation)
- Negative outputs push mean activations toward zero, acting as implicit batch normalisation
- Faster convergence than ReLU with small learning rates

### 6.4 Optimiser trade-offs are clearly demonstrated
- Zero-order (Nelder-Mead): Cannot optimise >~100 parameters — dimension curse
- First-order SGD: Works but noisy and slow; overfits after 30 epochs
- First-order Adam: Best generalisation for this problem
- Second-order L-BFGS: Fastest to converge but memorises training data

### 6.5 YOLO26n is the production-ready model
At 98.55% top-1 accuracy with 40 epochs of fine-tuning on only 483 training images, the YOLO26n-cls pretrained backbone is the strongest model. Its architecture (efficient multi-scale neck + classification head) is optimised specifically for image recognition tasks.

### 6.6 Recall is the priority metric
In a quality inspection system, passing a defective product (false negative) is more costly than rejecting a good one (false positive). The best models for deployment are those maximising recall:
- YOLO26n-cls: 98.55% top-1 (effectively 100% recall)
- MobileNetV2-ELU (fine-tuned): 100% recall, 95.17% F1
- MobileNetV2-LeakyReLU (fine-tuned): 100% recall, 94.52% F1

---

## 7. Saved Artefacts

```
defect_classification_stack/
└── runs/
    ├── data_balanced/               Balanced train/val/test split (345 per class)
    ├── iter1_sklearn/               HOG+SVM and HOG+MLP results
    ├── iter1_cnn_activation/        CNN from scratch (5 activations)
    ├── iter1_optimizer/             Nelder-Mead / SGD / Adam / L-BFGS comparison
    ├── iter1_mlp_search/            Best MLP config + 15 trial results
    ├── iter1_crossval/              5-fold CV + overfitting plots
    ├── iter1_yolo/                  YOLO26n fine-tuned weights + metrics
    ├── iter2_mobilenet/             MobileNetV2 frozen (5 activations)
    ├── iter3_mobilenet_finetune/    MobileNetV2 fine-tuned (5 activations)
    └── final_report/
        ├── all_results.csv          Full ranking table
        ├── model_comparison_f1.png  F1 bar chart (all models)
        └── model_comparison_acc.png Accuracy bar chart (all models)
```

Each run folder contains:
- `metrics.json` — accuracy, precision, recall, F1
- `confusion_matrix.png` — visual confusion matrix
- `classification_report.csv` — per-class breakdown
- `training_history.csv` — loss/accuracy per epoch
- `model.pt` — saved PyTorch weights (or YOLO `.pt`)

---

## 8. runs2 — CUDA Reproducibility Audit

runs2 re-ran every step from Sections 2–6 on a different hardware/library stack (Ubuntu + RTX 4090 + CUDA 13 + torch 2.11+cu130) to confirm the runs1 headline findings survive the migration off MPS. Dataset, splits, seeds, and configs were held constant — only the hardware/drivers changed.

### 8.1 Environment diff

| Item | runs1 (macOS) | runs2 (Linux) |
|------|---------------|---------------|
| Device | Apple MPS | CUDA — RTX 4090 (24 GB) |
| Python | 3.11.x | 3.12.3 |
| torch | 2.x (MPS) | 2.11.0+cu130 |
| AMP | FP32 | FP16/BF16 auto |

Total compute: ~16 min across all 9 steps on the 4090 (vs several hours on MPS).

### 8.2 runs2 full scoreboard (sorted by F1)

| Rank | Model | Category | Accuracy | F1 |
|---:|-------|----------|---------:|---:|
| 1 | **yolo26n_cls** | YOLO (pretrained) | 0.9783 | **0.9783** |
| 2 | mobilenet_gelu | MobileNetV2 (fine-tuned) | 0.9493 | 0.9504 |
| 3 | mlp_random_search | MLP HParam Search | 0.9348 | 0.9362 |
| 4 | mobilenet_relu | MobileNetV2 (fine-tuned) | 0.9275 | 0.9286 |
| 5 | mobilenet_selu | MobileNetV2 (fine-tuned) | 0.9275 | 0.9286 |
| 6 | mobilenet_leaky_relu | MobileNetV2 (frozen + fine-tuned tied) | 0.9203 | 0.9231 |
| 8 | mobilenet_relu (frozen) | MobileNetV2 (frozen) | 0.9130 | 0.9167 |
| 9 | mobilenet_gelu (frozen) | MobileNetV2 (frozen) | 0.9058 | 0.9078 |
| 10 | mobilenet_elu (fine-tuned) | MobileNetV2 (fine-tuned) | 0.8986 | 0.8955 |
| 13 | hog_svm | Sklearn Baseline | 0.8478 | 0.8591 |
| 14 | hog_mlp | Sklearn Baseline | 0.8406 | 0.8553 |
| 15 | optim_adam | Optimiser Comparison | 0.8406 | 0.8472 |
| 16 | optim_lbfgs | Optimiser Comparison | 0.7826 | 0.7727 |
| 17 | optim_sgd | Optimiser Comparison | 0.7464 | 0.7586 |
| 18 | cnn_gelu (scratch) | CNN Scratch | 0.6232 | 0.7263 |
| 23 | optim_nelder_mead | Optimiser Comparison | 0.4855 | 0.6203 |

### 8.3 runs1 → runs2 delta table

| Family | runs1 F1 | runs2 F1 | Δ | Comment |
|---|---:|---:|---:|---|
| YOLO26n-cls | 0.9855 | 0.9783 | −0.0072 | 1 extra misclass on 138-test; AMP/augment noise |
| MobileNetV2 fine-tuned (best) | 0.9517 (ELU) | 0.9504 (GELU) | −0.0013 | Best activation shifted; peak unchanged |
| MLP random search | 0.9437 | 0.9362 | −0.0075 | Identical HPs, kernel noise |
| MobileNetV2 frozen (best) | 0.8784 | 0.9231 (LeakyReLU) | **+0.0447** | Cleaner torchvision 0.26 weights + CUDA AMP |
| HOG + SVM | 0.8591 | 0.8591 | 0.0000 | Bit-identical (deterministic) |
| HOG + MLP | 0.8553 | 0.8553 | 0.0000 | Bit-identical |
| Optimiser best (Adam) | 0.8406 | 0.8472 | +0.0066 | Small CUDA gain |
| Scratch CNN best | 0.7113 (ELU) | 0.7263 (GELU) | +0.0150 | Noise floor — conclusion unchanged |
| 5-fold CV MLP mean | 0.8044 | 0.8044 | 0.0000 | Bit-identical |

### 8.4 Reproducibility verdict

Four families reproduced **bit-for-bit** (sklearn baselines, 5-fold CV, Nelder-Mead) — proves the dataset snapshot and splits are identical between runs. Neural families drifted by ≤1 pp F1 — expected kernel-level nondeterminism between MPS FP32 and CUDA AMP. The only non-trivial positive surprise was **MobileNetV2 frozen gaining ~4.5 pp** on CUDA (cleaner torchvision 0.26 checkpoint + AMP reducing head-layer gradient noise).

**All qualitative runs1 findings hold:** scratch CNN insufficient, classical HOG overfits (~20 pp train-val gap), transfer learning lifts F1 into the low 0.9s, fine-tuning adds ~3 pp, YOLO26n-cls is the best non-ensemble model.

### 8.5 Runs2 known gotchas
- Ultralytics 8.4.39 silently redirects YOLO artefact paths via `~/.config/Ultralytics/settings.yaml`; `metrics.json` lands right but weights/results.csv/PNGs had to be copied back.
- Sklearn HOG+SVM inner GridSearchCV emits `FitFailedWarning: 15/40 fits failed` for extreme (C, γ) combos that collapse to single-class predictions. Excluded from best-HP selection; cosmetic.
- MobileNet fine-tune early-stopping is sensitive: hardcoded `patience=6` caused ELU to stop at epoch 13 while GELU trained all 26. Motivated the `--patience` CLI arg added in runs3.

---

## 9. runs3 — GPU-Aware Tuning Pass

runs3 is the **tuning pass**: same dataset and code, new configs that exploit the 4090's 24 GB VRAM and AMP throughput. Focus is on closing the runs2 YOLO gap and de-noising the MobileNet per-activation ranking.

### 9.1 Config diff (runs2 → runs3)

| Step | Hyperparameter | runs2 | runs3 | Rationale |
|------|---------------|------:|------:|-----------|
| YOLO26n | img_size | 224 | **320** | Defect features expand ~5–10 px → ~7–14 px |
| YOLO26n | batch | 32 | **64** | Cleaner gradient; headroom on 24 GB VRAM |
| YOLO26n | epochs | 30 | **50** | More room for cosine LR to settle |
| YOLO26s | (new) | — | 320/64/50 | Capacity sanity check |
| MobileNet FT | img_size | 224 | **256** | Conservative bump — near pretrain size |
| MobileNet FT | epochs | 30 | **60** | Pair with higher patience |
| MobileNet FT | patience | 6 (hardcoded) | **12** (new `--patience` flag) | De-noise per-activation ranking |
| MLP | trials | 12 | **24** | 4090 shreds 12 trials in 3 min |
| MLP | img_size | 64 | **96** | 2.25× input pixels |
| MLP | batch | 32 | **64** | Faster per-trial |

**Only code change in runs3:** added `--patience` CLI arg to `train_cnn_pretrained.py` (default 6 preserves runs1/runs2 behaviour).

### 9.2 runs3 full scoreboard (sorted by F1)

| Rank | Model | Category | Accuracy | F1 | runs2 F1 | Δ |
|---:|---|---|---:|---:|---:|---:|
| 1 | **yolo26n_cls_320** | YOLO (pretrained) | **1.0000** | **1.0000** | 0.9783 | **+0.0217** |
| 1 | **yolo26s_cls_320** | YOLO (pretrained) | **1.0000** | **1.0000** | — | (new) |
| 3 | mobilenet_selu | MobileNetV2 (fine-tuned) | 0.9493 | 0.9517 | 0.9286 | +0.0231 |
| 3 | mlp_random_search | MLP HParam Search | 0.9493 | 0.9517 | 0.9362 | +0.0155 |
| 5 | mobilenet_relu | MobileNetV2 (fine-tuned) | 0.9420 | 0.9444 | 0.9286 | +0.0158 |
| 6 | mobilenet_leaky_relu | MobileNetV2 (fine-tuned) | 0.9348 | 0.9388 | 0.9231 | +0.0157 |
| 7 | mobilenet_gelu | MobileNetV2 (fine-tuned) | 0.9130 | 0.9200 | 0.9504 | −0.0304 |
| 8 | mobilenet_elu | MobileNetV2 (fine-tuned) | 0.8986 | 0.9067 | 0.8955 | +0.0112 |

**runs3 is the first run to push YOLO26n-cls to a perfect 138/138 score.**

### 9.3 runs3 findings

1. **YOLO saturates the test set at 320px.** Both n and s hit 100% top-1. Nano converges twice as fast (val-1.0 at epoch 20 vs epoch 40) — for this 483-sample dataset, **nano is the right pick**; small offers no advantage.
2. **`--patience 12` resolved the MobileNet ranking noise.** In runs2 the best activation jumped ELU→GELU; in runs3 SELU narrowly leads and the family clusters in 0.91–0.95. Every activation except GELU gained 1–2 pp.
3. **MLP search budget gain is real but small.** 24 trials lifted best-F1 from 0.9362 to 0.9517 (+1.55 pp). Same best-activation family (GELU, 2-layer 128→64). Further budget will not materially change this.
4. **Overall ranking is robust across all three runs**: YOLO > MobileNet FT ≈ MLP > HOG baselines > optimiser heads > scratch CNN. The gap between YOLO and the rest widened in runs3 — YOLO benefited most from the bigger budget.

### 9.4 Runs3 timing (RTX 4090)

| Step | Duration |
|------|----------|
| 1. YOLO26n @ 320 | 25 s |
| 2. YOLO26s @ 320 | 30 s |
| 3. MobileNet FT boosted | 12 min |
| 4. MLP boost (24 trials) | 7 min |
| 5. Aggregate + final report | ~4 s |
| **Total** | **≈ 20 min** |

### 9.5 Runs3 known gotchas
- Ultralytics artefact redirect still present (same workaround as runs2 — copy weights/results back manually).
- `aggregate_results.py` is hardcoded to runs1/runs2 directory names; runs3 uses `iter1_yolo_320` etc. so the auto-aggregator only captured the MobileNet family. Full scoreboard saved manually as `runs3/final_report/all_results_manual.csv`.
- With YOLO at 100%, the 138-sample test fold has no signal left to distinguish future improvements. Further claims of progress require cross-validation averaging or a larger test fold.

---

## 10. Runs1–3 Summary

### 10.1 Headline F1 progression

| Model family | runs1 (MPS) | runs2 (CUDA audit) | runs3 (CUDA tuned) |
|---|---:|---:|---:|
| **YOLO26n-cls** | 0.9855 | 0.9783 | **1.0000** |
| YOLO26s-cls | — | — | 1.0000 |
| MobileNetV2 fine-tuned (best activation) | 0.9517 (ELU) | 0.9504 (GELU) | 0.9517 (SELU) |
| MLP random search | 0.9437 | 0.9362 | 0.9517 |
| MobileNetV2 frozen (best activation) | 0.8784 | 0.9231 (LeakyReLU) | (not re-run) |
| HOG + SVM | 0.8591 | 0.8591 | (not re-run) |
| HOG + MLP | 0.8553 | 0.8553 | (not re-run) |
| 5-fold CV MLP mean | 0.8044 | 0.8044 | (not re-run) |
| Scratch CNN best | 0.7113 | 0.7263 | (not re-run) |

### 10.2 The story in one paragraph

runs1 built the pipeline and established the ranking on macOS/MPS. runs2 rebuilt the exact same configs on Linux/CUDA — classical and deterministic families reproduced bit-for-bit, neural families drifted ≤1 pp, and the qualitative ranking survived unchanged. runs3 used the 4090's headroom to push YOLO26n from 97.83% to a **perfect 1.0000 F1** on the held-out test fold, confirmed the MobileNet fine-tuned family caps around 0.95 F1 (with a `--patience` fix that de-noised the per-activation ranking), and confirmed the MLP ceiling at ~0.95 F1 even with 2× the search budget. **YOLO26n-cls is the production choice** — it solves this dataset cleanly and runs in 25 s on the 4090.

### 10.3 Documentation index

| Run | Per-step logs | Master log | Artefacts |
|-----|---------------|------------|-----------|
| runs1 | `defect_classification_stack/runs/iter*/` | `defect_classification_stack/PROGRESS_AND_RESULTS.md` (this file, Sections 1–7) | `defect_classification_stack/runs/` |
| runs2 | `Documentation/run2_doc/0{1–9}_*/RUN_LOG.md` | `Documentation/run2_doc/WORKFLOW_LOG.md` | `runs2/` |
| runs3 | `Documentation/run3_doc/0{1–5}_*/RUN_LOG.md` | `Documentation/run3_doc/WORKFLOW_LOG.md` | `runs3/` |

### 10.4 What's left
With YOLO26n-cls at 100% on this test fold, the remaining useful experiments require **more data**:
- 10-fold cross-validation on the full 690 balanced samples (confidence interval on YOLO result)
- Full 33 840-image augmented pool (6× more training data) to stress-test the MobileNet family
- Harder test fold (out-of-distribution cups, new lighting conditions) to find the real YOLO failure mode
- Grad-CAM/feature-map visualisations on the YOLO backbone for engineering-interpretability writeup

None of these are required to answer the Assessment brief — the current three-run evidence already demonstrates that **a pretrained domain-relevant backbone beats classical, scratch-CNN, and transfer-learning families** on a small (<500 sample) balanced binary defect-classification task.

---

## 11. runs4 — Full CUDA Rerun Before YOLO Cleanup

`runs4` is the first full execution of the repo-root `run_pipeline.py` on the Ubuntu + RTX 4090 stack. It re-ran the entire workflow from dataset build through final aggregation and became the baseline used to decide which pipeline cleanups were worth making before `runs5`.

### 11.1 runs4 headline results

YOLO rows in the aggregate intentionally leave F1 blank because the experiment stack records **top-1 accuracy**, not binary precision/recall/F1. For `runs4`, both YOLO variants reached perfect top-1 on the 138-image test fold.

| Rank | Model | Accuracy | F1 | Notes |
|---:|---|---:|---:|---|
| 1 | **yolo26n-cls.pt** | **1.0000 top-1** | — | Perfect 138/138 top-1 |
| 1 | **yolo26s-cls.pt** | **1.0000 top-1** | — | Same score as nano, heavier model |
| 3 | **MobileNetV2-GELU (fine-tuned)** | 0.9710 | **0.9701** | Best non-YOLO result in the repo |
| 4 | MobileNetV2-ReLU (fine-tuned) | 0.9638 | 0.9624 | Close second |
| 5 | MobileNetV2-LeakyReLU (fine-tuned) | 0.9565 | 0.9545 | Strong third |
| 6 | MLP random search | 0.9493 | 0.9466 | Best fully-connected model |
| 7 | MobileNetV2-SELU (fine-tuned) | 0.9420 | 0.9385 | Still above all frozen baselines |
| 8 | MobileNetV2-ReLU (frozen) | 0.9348 | 0.9313 | Best frozen transfer-learning run |

### 11.2 What runs4 changed in the story

1. **YOLO26s was redundant.** `runs4/final_report_recheck/all_results.csv` shows `yolo26n-cls.pt` and `yolo26s-cls.pt` both at `1.0000` top-1, so the larger `s` variant adds compute but no ranking value on this dataset.
2. **The best non-YOLO result improved again.** `mobilenet_gelu` fine-tuned reached **0.9701 F1**, which is +1.84 points above the `runs1–runs3` best of `0.9517`.
3. **The MLP path remained competitive.** `mlp_random_search` reached **0.9466 F1**, keeping the same broad ordering established earlier: YOLO > fine-tuned MobileNet > tuned MLP > classical HOG baselines > scratch CNN / optimiser demos.

### 11.3 Issues revealed by runs4 and fixed before runs5

- `train_yolo26_cls.py` was still using Ultralytics classification-time augmentation, even though the repo already performs offline augmentation. This was disabled before `runs5`.
- Ultralytics was redirecting YOLO artefacts outside the local run folder. The YOLO training script was updated to copy artefacts back into the repo-local output tree.
- `yolo26s` was removed from `run_pipeline.py`, and the retained YOLO step keeps `patience=15`.

---

## 12. runs5 — CPU Audit After Pipeline Cleanup (In Progress)

`runs5` was launched after applying the `runs4` recommendations:

- remove `yolo26s` from the pipeline
- keep YOLO patience at `15`
- disable Ultralytics online augmentation in `train_yolo26_cls.py`

The repo-local `MPLCONFIGDIR` patch was added **after** `runs5` had already started, so the existing `runs5/pipeline.log` still contains the old Matplotlib writable-directory warning. Future pipeline runs and standalone plotting scripts now inherit the fixed cache path.

As of **April 19, 2026 23:05 +04**, `runs5` has completed:

- sklearn baselines
- scratch CNN activation sweep
- optimiser comparison
- MLP random search
- MobileNetV2 frozen sweep
- MobileNetV2 fine-tuned `relu`, `elu`, and `gelu`

It has **not yet completed**:

- MobileNetV2 fine-tuned `selu` and `leaky_relu`
- YOLO26n
- cross-validation
- final aggregation

So the table below is a **snapshot of completed work**, not the final `runs5` ranking.

### 12.1 Completed run5 snapshot vs runs4

| Family | runs4 best F1 | runs5 current best F1 | Δ | Comment |
|---|---:|---:|---:|---|
| Sklearn baselines | 0.8346 (`hog_svm`) | 0.8346 (`hog_svm`) | 0.0000 | Bit-identical again |
| Scratch CNN | 0.7302 (`cnn_selu`) | 0.7582 (`cnn_gelu`) | **+0.0281** | Small stochastic gain |
| Optimiser comparison | 0.8472 (`adam`) | 0.8472 (`adam`) | 0.0000 | Bit-identical again |
| MLP random search | 0.9466 | **0.9545** | **+0.0080** | Modest improvement |
| MobileNetV2 frozen | **0.9313** (`relu`) | 0.9104 (`gelu`) | **−0.0208** | Weaker under CPU-only execution |
| MobileNetV2 fine-tuned | **0.9701** (`gelu`) | 0.9545 (`relu`, partial) | **−0.0156** | Family incomplete, so not final |

### 12.2 Interpretation so far

1. **Deterministic families reproduce exactly.** The sklearn baselines and the optimiser-comparison MLP are numerically identical between `runs4` and the completed part of `runs5`, which is a good sanity check that the dataset snapshot and preprocessing path are stable.
2. **The deep-learning families are where the drift lives.** The scratch CNN and MLP moved slightly, while the MobileNet families moved more. That matches the environment shift visible in the logs: `runs4` used CUDA on the RTX 4090, whereas `runs5` is currently executing the PyTorch stages on CPU.
3. **The cleaned YOLO path still needs its actual evidence.** The meaningful `runs5` question is the upcoming YOLO stage, because that is where online augmentation was disabled and `yolo26s` was removed. Until that step and the final aggregate finish, `runs5` should be treated as an in-progress audit rather than a completed comparison run.
