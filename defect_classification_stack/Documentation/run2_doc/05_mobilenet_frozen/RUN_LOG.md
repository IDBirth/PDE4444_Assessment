# Step 5 — MobileNetV2 Frozen Backbone (Activation Sweep)

## Command
```bash
.venv/bin/python defect_classification_stack/train_cnn_pretrained.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter2_mobilenet
```

Defaults: epochs=30, batch=16, img_size=224, seed=42. ImageNet-pretrained MobileNetV2 backbone **frozen**; only the head (1280 → 256 → 1 with activation ∈ {relu, elu, gelu, selu, leaky_relu}) is trained.

## Environment
- Device: CUDA (RTX 4090), torchvision 0.26.0
- Pretrained weights: `mobilenet_v2-7ebf99e0.pth` (auto-downloaded to `~/.cache/torch/hub/checkpoints/`)

## Timing
- Start: 2026-04-19T15:11:24+04:00
- End:   2026-04-19T15:16:29+04:00
- Duration: 305 s (~5.1 min)

## Results (test set, 138 samples)

| Activation | Accuracy | Precision | Recall | F1 |
|------------|---------:|----------:|-------:|---:|
| LeakyReLU  | 0.9203 | 0.8919 | 0.9565 | **0.9231** |
| ReLU       | 0.9130 | 0.8800 | 0.9565 | 0.9167 |
| GELU       | 0.9058 | 0.8889 | 0.9275 | 0.9078 |
| SELU       | 0.8696 | 0.8312 | 0.9275 | 0.8767 |
| ELU        | 0.8623 | 0.8049 | 0.9565 | 0.8742 |

## Comparison vs runs/iter2_mobilenet (MPS / macOS)

| Activation | runs1 F1 | runs2 F1 | Δ |
|------------|---:|---:|---:|
| ELU        | 0.8784 | 0.8742 | −0.0042 |
| GELU       | 0.8784 | 0.9078 | **+0.0294** |
| SELU       | 0.8784 | 0.8767 | −0.0017 |
| LeakyReLU  | 0.8784 | **0.9231** | **+0.0447** |
| ReLU       | 0.8725 | 0.9167 | **+0.0442** |

**Audit verdict:** runs2 is *better* than runs1 for ReLU / LeakyReLU / GELU by 3–5 pp. Two plausible causes:
1. **torchvision 0.26.0** may ship a slightly different `MobileNet_V2_Weights.DEFAULT` checkpoint vs whatever version was resolved on the original MPS run.
2. **CUDA vs MPS kernel precision** — TF32 / bf16 autotuning on the 4090 tends to give cleaner gradient updates for conv layers than MPS FP32, reducing head-layer noise during the short 30-epoch training.

Activation ranking is different (runs1: all tied near 0.878 because of the frozen-backbone ceiling; runs2: LeakyReLU > ReLU > GELU > SELU > ELU). Interesting side finding for the report: **on this GPU the frozen MobileNetV2 is already a 92% F1 model**, which is competitive with the tuned MLP (93.6%).

## Artefacts
- `runs2/iter2_mobilenet/summary.csv`
- Per-activation folders with `metrics.json`, `confusion_matrix.png`, `classification_report.csv`, `training_history.csv`, `model.pt`
- `Documentation/run2_doc/05_mobilenet_frozen/stdout.log`
