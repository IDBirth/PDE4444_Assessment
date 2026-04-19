# Step 9 — Aggregate + Compare

## Commands
```bash
# Aggregate all model families into one scoreboard
.venv/bin/python defect_classification_stack/aggregate_results.py \
    --runs-dir runs2 \
    --output-dir runs2/final_report

# Four-family comparison (sklearn / keras-mlp / cnn-scratch / yolo) per the original report script
.venv/bin/python defect_classification_stack/compare_results.py \
    --sklearn-dir runs2/iter1_sklearn \
    --keras-dir   runs2/iter1_mlp_search \
    --cnn-dir     runs2/iter1_cnn_activation \
    --yolo-dir    runs2/iter1_yolo \
    --output-file runs2/final_report/comparison.csv
```

## Environment
- Device: CPU (pure pandas / matplotlib)
- torch 2.11.0+cu130, pandas 2.3.x, matplotlib 3.10.x

## Timing
- Aggregate: ~3 s
- Compare:   ~1 s

## runs2 full scoreboard (sorted by F1)

| Rank | Model | Category | Accuracy | F1 |
|---:|-------|----------|---------:|---:|
| 1 | **yolo26n_cls** | YOLO (pretrained) | 0.9783 | **0.9783** |
| 2 | mobilenet_gelu | MobileNetV2 (fine-tuned) | 0.9493 | 0.9504 |
| 3 | mlp_random_search | MLP HParam Search | 0.9348 | 0.9362 |
| 4 | mobilenet_relu | MobileNetV2 (fine-tuned) | 0.9275 | 0.9286 |
| 5 | mobilenet_selu | MobileNetV2 (fine-tuned) | 0.9275 | 0.9286 |
| 6 | mobilenet_leaky_relu | MobileNetV2 (frozen) | 0.9203 | 0.9231 |
| 6 | mobilenet_leaky_relu | MobileNetV2 (fine-tuned) | 0.9203 | 0.9231 |
| 8 | mobilenet_relu | MobileNetV2 (frozen) | 0.9130 | 0.9167 |
| 9 | mobilenet_gelu | MobileNetV2 (frozen) | 0.9058 | 0.9078 |
| 10 | mobilenet_elu | MobileNetV2 (fine-tuned) | 0.8986 | 0.8955 |
| 11 | mobilenet_selu | MobileNetV2 (frozen) | 0.8696 | 0.8767 |
| 12 | mobilenet_elu | MobileNetV2 (frozen) | 0.8623 | 0.8742 |
| 13 | hog_svm | Sklearn Baseline | 0.8478 | 0.8591 |
| 14 | hog_mlp | Sklearn Baseline | 0.8406 | 0.8553 |
| 15 | optim_adam | Optimiser Comparison | 0.8406 | 0.8472 |
| 16 | optim_lbfgs | Optimiser Comparison | 0.7826 | 0.7727 |
| 17 | optim_sgd | Optimiser Comparison | 0.7464 | 0.7586 |
| 18 | cnn_gelu | CNN Scratch | 0.6232 | 0.7263 |
| 19 | cnn_selu | CNN Scratch | 0.6159 | 0.7225 |
| 20 | cnn_leaky_relu | CNN Scratch | 0.5507 | 0.6900 |
| 21 | cnn_relu | CNN Scratch | 0.5362 | 0.6832 |
| 21 | cnn_elu | CNN Scratch | 0.5362 | 0.6832 |
| 23 | optim_nelder_mead | Optimiser Comparison | 0.4855 | 0.6203 |

Best model: **yolo26n_cls** (accuracy 97.83%, F1 97.83%).

## Headline ranking vs runs1

| Family best | runs1 F1 | runs2 F1 | Δ |
|---|---:|---:|---:|
| YOLO26n-cls | 0.9855 | 0.9783 | −0.0072 |
| MobileNetV2 fine-tuned (best act) | 0.9517 (ELU) | 0.9504 (GELU) | −0.0013 |
| MLP random search | 0.9437 | 0.9362 | −0.0075 |
| MobileNetV2 frozen (best act) | 0.8784 (tied) | 0.9231 (LeakyReLU) | **+0.0447** |
| HOG + SVM | 0.8591 | 0.8591 | 0.0000 |
| HOG + MLP | 0.8553 | 0.8553 | 0.0000 |
| Optimiser (Adam) | 0.8406 | 0.8472 | +0.0066 |
| Scratch CNN best | 0.7113 (ELU) | 0.7263 (GELU) | +0.0150 |
| 5-fold CV (MLP F1 mean) | 0.8044 | 0.8044 | 0.0000 |

**Overall ranking preserved**: YOLO > MobileNetV2-finetuned > MLP > MobileNetV2-frozen > HOG baselines > optimiser heads > scratch CNN.

## Artefacts
- `runs2/final_report/all_results.csv` — full 23-row scoreboard with category labels
- `runs2/final_report/comparison.csv` — 4-family subset matching the original report table
- `runs2/final_report/model_comparison_f1.png`, `model_comparison_acc.png`
- `Documentation/run2_doc/09_final_report/stdout_aggregate.log`, `stdout_compare.log`
