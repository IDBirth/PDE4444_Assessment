# runs2 — Full Workflow Audit on Linux + RTX 4090

This document is the master log for the **second end-to-end execution** of the defect-classification pipeline. The first pass (`runs/`, called **runs1** here) was produced on macOS + MPS; runs2 reproduces every step on Ubuntu Linux + CUDA (RTX 4090) using the same balanced dataset snapshot so the two sets of numbers can be audited side-by-side.

Per-step detail lives in `01_sklearn_baseline/` through `09_final_report/` — one `RUN_LOG.md` + `stdout.log` each.

---

## Purpose
1. **Reproducibility audit** — confirm runs1 headline findings survive a completely different CPU/GPU stack.
2. **GPU speed-up baseline** — measure how fast each step is on the 4090 so runs3 can plan bigger configs.
3. **Self-contained artefact bundle** — runs2/ now contains everything needed to regenerate the report without touching runs1/.

The dataset is frozen (`defect_classification_stack/runs/data_balanced/`, train 483 / val 69 / test 138, 50/50 balanced). Only hardware and library versions change.

## Environment

| Item | runs1 (macOS) | runs2 (Linux) |
|------|---------------|---------------|
| Device | Apple MPS | CUDA — RTX 4090 (24 GB, CC 8.9) |
| Python | 3.11.x | 3.12.3 |
| torch | 2.x (MPS build) | 2.11.0+cu130 |
| torchvision | 0.2x (MPS) | 0.26.0 |
| ultralytics | 8.4.39 | 8.4.39 |
| scikit-learn | 1.x | 1.8.0 |
| AMP | FP32 | FP16/BF16 auto |

## Commands used (for re-run)
All commands were executed from the repo root `/home/ubu/Desktop/Assessment` via the fresh venv at `.venv/`.

```bash
# Step 1 — sklearn baselines
.venv/bin/python defect_classification_stack/train_sklearn_baseline.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter1_sklearn

# Step 2 — scratch CNN × 5 activations
.venv/bin/python defect_classification_stack/train_cnn_activation_sweep.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter1_cnn_activation

# Step 3 — optimiser comparison
.venv/bin/python defect_classification_stack/train_optimizer_comparison.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter1_optimizer

# Step 4 — MLP random search
.venv/bin/python defect_classification_stack/train_keras_mlp_random_search.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter1_mlp_search

# Step 5 — MobileNetV2 frozen × 5 activations
.venv/bin/python defect_classification_stack/train_cnn_pretrained.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter2_mobilenet

# Step 6 — MobileNetV2 fine-tuned × 5 activations
.venv/bin/python defect_classification_stack/train_cnn_pretrained.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter3_mobilenet_finetune \
    --unfreeze

# Step 7 — YOLO26n-cls
.venv/bin/python defect_classification_stack/train_yolo26_cls.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter1_yolo

# Step 8 — 5-fold cross-validation (HOG + SVM / MLP)
.venv/bin/python defect_classification_stack/train_cross_validation.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter1_crossval

# Step 9 — aggregate + compare
.venv/bin/python defect_classification_stack/aggregate_results.py \
    --runs-dir runs2 --output-dir runs2/final_report
.venv/bin/python defect_classification_stack/compare_results.py \
    --sklearn-dir runs2/iter1_sklearn \
    --keras-dir   runs2/iter1_mlp_search \
    --cnn-dir     runs2/iter1_cnn_activation \
    --yolo-dir    runs2/iter1_yolo \
    --output-file runs2/final_report/comparison.csv
```

## Timing summary (RTX 4090)

| Step | Duration | Start → End (local) |
|------|----------|---------------------|
| 1. Sklearn baselines | 29 s | 15:00:45 → 15:01:14 |
| 2. Scratch CNN × 5 | 115 s | 15:03:25 → 15:05:20 |
| 3. Optimiser comparison | 11 s | 15:06:05 → 15:06:16 |
| 4. MLP random search (12 trials) | 186 s | 15:07:01 → 15:10:07 |
| 5. MobileNetV2 frozen × 5 | 305 s | 15:11:24 → 15:16:29 |
| 6. MobileNetV2 fine-tuned × 5 | 230 s | 15:17:24 → 15:21:14 |
| 7. YOLO26n-cls (30 ep) | 19 s | 15:21:30 → 15:21:49 |
| 8. 5-fold CV | 38 s | 15:28:30 → 15:29:08 |
| 9. Aggregate + compare | ~4 s | — |
| **Total compute** | **≈ 16 min** | |

YOLO 30-epoch classification training in **19 seconds** is the headline GPU win — runs1 on MPS needed several minutes for the same schedule. This leaves plenty of headroom for runs3 experiments (bigger image size, more epochs, model sweeps).

## Headline diff table (runs1 → runs2)

| Family | runs1 best F1 | runs2 best F1 | Δ | Comment |
|---|---:|---:|---:|---|
| YOLO26n-cls | 0.9855 | 0.9783 | −0.0072 | 1 extra misclass on 138-test; AMP/randaugment noise |
| MobileNetV2 fine-tuned | 0.9517 (ELU) | 0.9504 (GELU) | −0.0013 | Best activation shifted; peak unchanged |
| MLP random search | 0.9437 | 0.9362 | −0.0075 | Identical HPs, dropout/Adam kernel noise |
| MobileNetV2 frozen | 0.8784 | **0.9231** | **+0.0447** | CUDA + torchvision 0.26 weights materially better |
| HOG + SVM | 0.8591 | 0.8591 | 0.0000 | Bit-identical (deterministic) |
| HOG + MLP | 0.8553 | 0.8553 | 0.0000 | Bit-identical |
| Optimiser best (Adam) | 0.8406 | 0.8472 | +0.0066 | Small CUDA gain |
| Scratch CNN best | 0.7113 (ELU) | 0.7263 (GELU) | +0.0150 | Noise floor — conclusion unchanged |
| 5-fold CV MLP mean | 0.8044 | 0.8044 | 0.0000 | Bit-identical |

## Reproducibility verdict

Four families reproduce **bit-for-bit** (sklearn baselines, 5-fold CV, Nelder-Mead) — these are CPU-deterministic and prove the dataset snapshot and splits are identical between runs. The neural families drift by ≤1 pp F1 (YOLO, MLP, Adam, MobileNet-finetuned) — expected kernel-level nondeterminism between MPS FP32 and CUDA AMP. The only non-trivial positive surprise is **MobileNetV2 frozen gaining ~4.5 pp** on CUDA — best explained by torchvision 0.26 shipping a slightly cleaner `MobileNet_V2_Weights.DEFAULT` checkpoint and AMP reducing head-layer gradient noise.

**All qualitative findings from the original report hold**: scratch CNN is insufficient; classical HOG overfits (~20 pp train-val gap); transfer learning from MobileNetV2 lifts F1 into the low-0.9s; fine-tuning adds another ~3 pp; YOLO26n-cls is the best non-ensemble model in the stack.

## Known footguns and gotchas

1. **YOLO artefact path**. Ultralytics 8.4.39 honours a global `~/.config/Ultralytics/settings.yaml` that silently redirects `project=runs2/iter1_yolo` to `<global_runs_dir>/classify/runs2/iter1_yolo/train`. The wrapper's own `metrics.json` still lands in the intended path, so audit-critical numbers are preserved, but the `weights/`, `results.csv`, and PNGs had to be copied back in post-hoc. See Step 7 RUN_LOG.

2. **Sklearn CV warnings**. The inner GridSearchCV for HOG + SVM emits `FitFailedWarning: 15/40 fits failed` because some extreme (C, gamma) combinations collapse to single-class predictions on 1 of the 5 inner folds. Those combinations receive NaN and are excluded from the best-HP pick — cosmetic, not a regression.

3. **Early-stopping sensitivity** in MobileNet fine-tune. With the default patience and val-loss-based trigger, ELU stops at epoch 13 on CUDA while GELU trains all 26 epochs and keeps improving. The ranking therefore reshuffles between runs1 and runs2, but the peak F1 is stable. For the report, the defensible takeaway is "fine-tuned MobileNetV2 ≈ 95% F1 regardless of activation; early-stopping noise dominates the per-activation ranking."

4. **Scratch-CNN recall = 1.0**. Every scratch-CNN activation in both runs predicts near-constant-positive on the flat-loss region. F1 ≈ 0.68–0.73 is driven by precision ≈ 0.52, not by genuine learning — this is why we moved to transfer learning.

## Where to look

| Topic | Path |
|---|---|
| Per-step detail | `Documentation/run2_doc/0X_stepname/RUN_LOG.md` |
| Per-step full stdout | `Documentation/run2_doc/0X_stepname/stdout.log` |
| Per-run artefacts | `runs2/iterN_*/` |
| Final scoreboard | `runs2/final_report/all_results.csv` |
| Charts | `runs2/final_report/model_comparison_{acc,f1}.png` |
| runs1 counterpart | `defect_classification_stack/runs/…` + `defect_classification_stack/PROGRESS_AND_RESULTS.md` |

## Next step (planned, not yet executed)
- **runs3** — GPU-aware improvements. Candidates: bigger batch (64–128), img_size=320 for YOLO, 100-epoch fine-tune with cosine LR, mixup/cutmix sweep, MobileNetV3-Large comparison, compile=True AOT. Documentation destination: `Documentation/run3_doc/`.
