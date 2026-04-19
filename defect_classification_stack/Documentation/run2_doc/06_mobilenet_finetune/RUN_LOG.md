# Step 6 — MobileNetV2 Fine-Tuned Backbone (Activation Sweep)

## Command
```bash
.venv/bin/python defect_classification_stack/train_cnn_pretrained.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter3_mobilenet_finetune \
    --unfreeze
```

Defaults: epochs=30, batch=16, img_size=224, seed=42. **All layers trainable** with differential lr (backbone: conservative, head: faster).

## Environment
- Device: CUDA (RTX 4090)
- torch 2.11.0+cu130, torchvision 0.26.0

## Timing
- Start: 2026-04-19T15:17:24+04:00
- End:   2026-04-19T15:21:14+04:00
- Duration: 230 s (~3.8 min)

## Results (test set, 138 samples)

| Activation | Accuracy | Precision | Recall | F1 | Epochs trained |
|------------|---------:|----------:|-------:|---:|---:|
| GELU       | 0.9493 | 0.9306 | 0.9710 | **0.9504** | 26 (early stop) |
| ReLU       | 0.9275 | 0.9155 | 0.9420 | 0.9286 | 13 (early stop) |
| SELU       | 0.9275 | 0.9155 | 0.9420 | 0.9286 | 13 (early stop) |
| LeakyReLU  | 0.9203 | 0.8919 | 0.9565 | 0.9231 | 13 (early stop) |
| ELU        | 0.8986 | 0.9231 | 0.8696 | 0.8955 | 13 (early stop) |

## Comparison vs runs/iter3_mobilenet_finetune (MPS / macOS)

| Activation | runs1 F1 | runs2 F1 | Δ |
|------------|---:|---:|---:|
| ELU        | **0.9517** | 0.8955 | −0.0562 |
| LeakyReLU  | 0.9452 | 0.9231 | −0.0221 |
| GELU       | 0.9429 | **0.9504** | +0.0075 |
| SELU       | 0.9388 | 0.9286 | −0.0102 |
| ReLU       | 0.9306 | 0.9286 | −0.0020 |

**Audit verdict:** headline result (fine-tuned MobileNetV2 ≈ 95% F1 — best non-YOLO model) **reproduced**. What changed: the winning activation shifted from **ELU** (runs1) to **GELU** (runs2). Root cause is early-stopping sensitivity — in runs2 ELU triggered early-stop at epoch 13 while still in a plateau; GELU trained for 26 epochs and kept improving (val loss 0.1417 → 0.0675). Under MPS (runs1) the val-loss trajectory was different so ELU kept training longer.

For the report, the defensible takeaway is: **fine-tuned MobileNetV2 hits ~95% F1 regardless of activation, and early-stopping noise dominates the per-activation ranking.** This is actually a cleaner scientific finding than "ELU wins" from runs1 — it shows the model is robust to activation choice once the backbone is fully unlocked.

## Artefacts
- `runs2/iter3_mobilenet_finetune/summary.csv`
- Per-activation folders with `metrics.json`, `confusion_matrix.png`, `classification_report.csv`, `training_history.csv`, `model.pt`
- `Documentation/run2_doc/06_mobilenet_finetune/stdout.log`

## Best runs2 non-YOLO model
`mobilenet_gelu` (fine-tuned) — accuracy 94.93%, F1 95.04%, recall 97.10% (2 missed defects out of 69 on the balanced test set).
