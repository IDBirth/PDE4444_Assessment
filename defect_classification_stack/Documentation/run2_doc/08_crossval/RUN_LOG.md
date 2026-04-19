# Step 8 — 5-Fold Cross-Validation (HOG + SVM / MLP)

## Command
```bash
.venv/bin/python defect_classification_stack/train_cross_validation.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter1_crossval
```

Pipeline: HOG (8100-dim) → per-model inner search (GridSearchCV, stratified 5-fold on train+val pool) → 5-fold CV with best HPs on the full 552-sample pool → final retrain and test evaluation on 138-sample held-out test.

## Environment
- Device: CPU (HOG + sklearn estimators — GPU irrelevant)
- torch 2.11.0+cu130, scikit-learn 1.8.0, scikit-image 0.26.0
- Seed: 42

## Timing
- Start: 2026-04-19T15:28:30+04:00
- End:   2026-04-19T15:29:08+04:00
- Duration: 38 s

## Results (5-fold CV on 552-sample train+val pool)

| Model | Mean val accuracy | Mean val F1 | Mean overfit gap |
|-------|------------------:|------------:|-----------------:|
| HOG + SVM  | 0.7537 ± 0.0317 | 0.7804 ± 0.0231 | 0.2232 |
| HOG + MLP  | **0.7900 ± 0.0455** | **0.8044 ± 0.0390** | 0.1874 |

Per-fold detail in [cv_combined.csv](../../../runs2/iter1_crossval/cv_combined.csv).

## Comparison vs runs/iter1_crossval (MPS / macOS)

| Model | runs1 mean val F1 | runs2 mean val F1 | Δ |
|-------|---:|---:|---:|
| HOG + SVM | 0.7804 | 0.7804 | 0.0000 |
| HOG + MLP | 0.8044 | 0.8044 | 0.0000 |

**Audit verdict:** bit-for-bit reproduction — per-fold `train_accuracy`, `val_accuracy`, `val_precision`, `val_recall`, `val_f1`, and `overfit_gap` values match to all 16 decimals printed in the CSV. This confirms the CV pipeline is fully deterministic (HOG is pure numpy, sklearn SVM + MLP with `random_state=42` are reproducible across CPU backends). runs1 vs runs2 numerical drift seen in neural steps does not apply here.

## Key findings (unchanged from runs1)
- Train accuracy is 97–98% while val accuracy is 75–79% ⇒ **significant overfitting gap** (~0.19–0.22). This is the main scientific justification for adopting transfer learning (MobileNetV2 / YOLO) over HOG-based classical ML.
- HOG + MLP beats HOG + SVM by ~2.4 pp mean F1 on CV and shows a smaller overfit gap — consistent with the dropout regularisation in the MLP.
- Fold-to-fold std ≈ 3–4 pp ⇒ the balanced test-set F1 values reported elsewhere in the stack are within one CV fold's noise of the true generalisation performance.

## Known warning (cosmetic)
Sklearn prints `FitFailedWarning: 15 fits failed out of a total of 40` during the inner GridSearchCV over SVM hyperparameters. Cause: some HP combinations (extreme C/gamma) produce a single-class prediction on certain inner folds, which makes `pos_label=1` invalid for that particular scorer call. Those parameter combinations receive NaN score and are correctly excluded from the best-HP selection. Same warning was present in runs1 — not a regression introduced by the CUDA environment.

## Artefacts
- `runs2/iter1_crossval/cv_combined.csv` — all 10 fold rows (5 folds × 2 models)
- `runs2/iter1_crossval/overfitting_analysis.png` — train vs val accuracy per fold + overfit gap
- `runs2/iter1_crossval/hog_svm_cv/{cv_results.csv, classification_report_test.csv, confusion_matrix_test.png, learning_curve.png}`
- `runs2/iter1_crossval/hog_mlp_cv/{cv_results.csv, classification_report_test.csv, confusion_matrix_test.png, learning_curve.png}`
- `Documentation/run2_doc/08_crossval/stdout.log`
