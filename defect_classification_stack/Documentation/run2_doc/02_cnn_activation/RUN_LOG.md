# Step 2 — Scratch CNN Activation Sweep

## Command
```bash
.venv/bin/python defect_classification_stack/train_cnn_activation_sweep.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter1_cnn_activation
```

Defaults: epochs=20, batch=32, img_size=224, seed=42. Architecture: 3 conv blocks (32→64→128) + dense head 128→128→1.

## Environment
- Device: CUDA (RTX 4090), compute capability 8.9
- torch 2.11.0+cu130
- Dataset: balanced snapshot (train 483, val 69, test 138)

## Timing
- Start: 2026-04-19T15:03:25+04:00
- End:   2026-04-19T15:05:20+04:00
- Duration: 115 s

## Results (test set, 138 samples)

| Activation | Accuracy | Precision | Recall | F1 |
|------------|---------:|----------:|-------:|---:|
| GELU       | 0.6232 | 0.5702 | 1.0000 | **0.7263** |
| SELU       | 0.6159 | 0.5656 | 1.0000 | 0.7225 |
| LeakyReLU  | 0.5507 | 0.5267 | 1.0000 | 0.6900 |
| ReLU       | 0.5362 | 0.5188 | 1.0000 | 0.6832 |
| ELU        | 0.5362 | 0.5188 | 1.0000 | 0.6832 |

## Comparison vs runs/iter1_cnn_activation (MPS / macOS)

| Activation | runs1 F1 | runs2 F1 | Δ | Note |
|---|---:|---:|---:|---|
| ELU        | 0.7113 | 0.6832 | −0.0281 | ranking moved down |
| ReLU       | 0.6980 | 0.6832 | −0.0148 | |
| LeakyReLU  | 0.6866 | 0.6900 | +0.0034 | |
| GELU       | 0.6732 | 0.7263 | +0.0531 | now best |
| SELU       | 0.6732 | 0.7225 | +0.0493 | |

**Audit verdict:** results are numerically different but conclusion is identical — scratch CNN hovers near random on this tiny dataset regardless of activation. Divergence is expected because random-op kernels differ between MPS and CUDA even with the same seed, and early-stopping triggers stochastically when val loss is flat. Headline finding (scratch CNN insufficient; transfer learning required) unchanged.

## Artefacts
- `runs2/iter1_cnn_activation/summary.csv`
- `runs2/iter1_cnn_activation/activation_convergence.png`
- Per-activation folders (`relu`, `elu`, `gelu`, `selu`, `leaky_relu`) with `metrics.json`, `confusion_matrix.png`, `classification_report.csv`, `training_history.csv`, `model.pt`
- `Documentation/run2_doc/02_cnn_activation/stdout.log`
