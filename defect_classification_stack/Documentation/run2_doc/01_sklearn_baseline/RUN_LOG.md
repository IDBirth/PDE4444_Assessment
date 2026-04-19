# Step 1 — Sklearn Baselines

## Command
```bash
.venv/bin/python defect_classification_stack/train_sklearn_baseline.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter1_sklearn
```

## Environment
- Device: CPU (sklearn is CPU-bound; HOG + SVM / MLPClassifier)
- torch 2.11.0+cu130, scikit-learn 1.8.0, scikit-image 0.26.0
- Dataset: balanced snapshot (train 483, val 69, test 138)

## Timing
- Start: 2026-04-19T15:00:45+04:00
- End:   2026-04-19T15:01:14+04:00
- Duration: 29 s

## Results (test set, 138 samples)

| Model   | Accuracy | Precision | Recall | F1 |
|---------|---------:|----------:|-------:|---:|
| HOG + SVM | 0.8478 | 0.8000 | 0.9275 | **0.8591** |
| HOG + sklearn MLP | 0.8406 | 0.7831 | 0.9420 | 0.8553 |

## Comparison vs runs/iter1_sklearn (MPS / macOS)

| Model | runs1 F1 | runs2 F1 | Δ |
|-------|---:|---:|---:|
| HOG + SVM | 0.8591 | 0.8591 | 0.0000 |
| HOG + sklearn MLP | 0.8553 | 0.8553 | 0.0000 |

**Audit verdict:** identical to runs1. Bit-for-bit reproduction (classical sklearn, deterministic with same seed).

## Artefacts
- `runs2/iter1_sklearn/summary.csv`
- `runs2/iter1_sklearn/hog_svm/{best_params.json, classification_report.csv, confusion_matrix.png, metrics.json, search_results.csv}`
- `runs2/iter1_sklearn/hog_mlp/{best_params.json, classification_report.csv, confusion_matrix.png, metrics.json, search_results.csv}`
- `Documentation/run2_doc/01_sklearn_baseline/stdout.log`
