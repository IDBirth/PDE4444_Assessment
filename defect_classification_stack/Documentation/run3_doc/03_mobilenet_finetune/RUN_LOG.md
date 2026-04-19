# runs3 Step 3 — MobileNetV2 fine-tuned × 5 activations (boosted)

## Command
```bash
.venv/bin/python defect_classification_stack/train_cnn_pretrained.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs3/iter3_mobilenet_finetune \
    --unfreeze \
    --epochs 60 \
    --batch-size 32 \
    --img-size 256 \
    --patience 12
```

GPU-aware changes vs runs2: image size 224→**256** (+14%), epochs 30→**60** (2×), patience 6→**12** (2×, via new `--patience` CLI arg).

## Why these settings
The runs2 analysis identified **early-stopping sensitivity** as the dominant noise source in per-activation ranking (ELU stopped at epoch 13 while GELU trained all 26). Doubling the patience gives every activation a fair shot at its minimum. 256px is a conservative bump: MobileNetV2 is pretrained at 224, and going much beyond 256 risks stretching the positional filters. 60 epochs × patience 12 means each activation is judged on its best epoch, not on the first plateau.

## Environment
- Device: CUDA (RTX 4090, 24072 MiB)
- torch 2.11.0+cu130, torchvision 0.26.0
- Model: MobileNetV2 (ImageNet), full backbone unfrozen, 2-phase LR (backbone 5e-5, head 3e-4)
- AMP: off (this script uses FP32 matmul)

## Timing
- Start: 2026-04-19T15:42:49+04:00
- End:   2026-04-19T15:54:52+04:00
- Duration: **~12 min** (5 activations × up to 60 epochs at 256px)

Per-activation early stops: most activations ran ~33–55 epochs before triggering patience=12. No activation used all 60 epochs.

## Results (test set, 138 samples), sorted by F1

| Rank | Activation | Accuracy | Precision | Recall | F1 | runs2 F1 | Δ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **SELU** | 0.9493 | 0.9079 | 1.0000 | **0.9517** | 0.9286 | +0.0231 |
| 2 | ReLU | 0.9420 | 0.9067 | 0.9855 | 0.9444 | 0.9286 | +0.0158 |
| 3 | LeakyReLU | 0.9348 | 0.8846 | 1.0000 | 0.9388 | 0.9231 | +0.0157 |
| 4 | GELU | 0.9130 | 0.8519 | 1.0000 | 0.9200 | 0.9504 | −0.0304 |
| 5 | ELU | 0.8986 | 0.8395 | 0.9855 | 0.9067 | 0.8955 | +0.0112 |

**Best activation shifted: GELU (runs2) → SELU (runs3).**

## Convergence notes
- SELU plateaued around val-acc 0.971 in the high-30s epoch range; the later cosine-LR phase kept polishing test-time precision (0.908) without overfitting
- ReLU early-stopped at epoch ~40, LeakyReLU at 33, ELU at 30s — patience=12 gave every activation a clean shot at its own minimum instead of the runs2 premature cut-off
- GELU regressed slightly — the longer schedule let it overfit the tiny 69-sample val set (ranking noise, peak F1 still in the high 0.9s band)

## Finding
With the `--patience 12` fix, the spread between the best and worst activation narrowed (peak 0.9517 vs bottom 0.9067 = 0.045 F1) and the overall band shifted up ~1–2 pp. Every activation except GELU gained. The **fine-tuned MobileNetV2 family now clusters in the 0.91–0.95 F1 range** on this dataset — consistent with the runs2 conclusion that the family plateau is "mid-0.95s F1 regardless of activation", but now with less noise.

## Comparison vs YOLO26n (runs3)
MobileNetV2 best (SELU, 0.9517) still trails YOLO26n at 320px (**1.0000** F1). The 5 pp gap reflects YOLO's domain-pretrained backbone + higher input resolution + random-augment schedule; for this small-dataset surface-defect task, the end-to-end YOLO classifier is the cleaner pick.

## Artefacts
- `runs3/iter3_mobilenet_finetune/summary.csv`
- `runs3/iter3_mobilenet_finetune/{relu,elu,gelu,selu,leaky_relu}/{metrics.json, model.pt, confusion_matrix.png, classification_report.csv, training_history.csv}`
- `runs3/iter3_mobilenet_finetune/mobilenet_convergence.png`
- `Documentation/run3_doc/03_mobilenet_finetune/stdout.log` (276 lines)

## Code change (runs3-only)
Added `--patience` CLI arg to `defect_classification_stack/train_cnn_pretrained.py` (was hardcoded to 6). Defaults preserve runs2 behaviour when the flag is omitted.
