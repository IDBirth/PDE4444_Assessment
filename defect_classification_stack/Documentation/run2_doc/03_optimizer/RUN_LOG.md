# Step 3 — Optimiser Comparison

## Command
```bash
.venv/bin/python defect_classification_stack/train_optimizer_comparison.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter1_optimizer
```

Architecture (shared): HOG (8100-dim) → PCA (20-dim) → MLP 20→32→16→1 (~1217 params).
Optimisers tested: **Nelder-Mead** (zero-order), **SGD** + **Adam** (first-order), **L-BFGS** (second-order).
Defaults: epochs=50, batch=32, nm_iters=800, seed=42.

## Environment
- Device: CUDA (RTX 4090) — though the 20-dim MLP is so small CPU vs GPU makes little difference
- torch 2.11.0+cu130

## Timing
- Start: 2026-04-19T15:06:05+04:00
- End:   2026-04-19T15:06:16+04:00
- Duration: 11 s

## Results (test set, 138 samples)

| Optimiser | Order | Accuracy | Precision | Recall | F1 |
|-----------|-------|---------:|----------:|-------:|---:|
| Adam | First-order (adaptive) | 0.8406 | 0.8133 | 0.8841 | **0.8472** |
| L-BFGS | Second-order | 0.7826 | 0.8095 | 0.7391 | 0.7727 |
| SGD | First-order | 0.7464 | 0.7237 | 0.7971 | 0.7586 |
| Nelder-Mead | Zero-order | 0.4855 | 0.4915 | 0.8406 | 0.6203 |

## Convergence behaviour (final epoch)

| Optimiser | Train loss | Val loss | Interpretation |
|-----------|----------:|---------:|----------------|
| Nelder-Mead | 0.9067 | 0.8740 | No descent after ~iter 50 — simplex flat in 1217-dim space |
| SGD         | 0.2702 | 0.8623 | Noisy; overfits after ~ep 25 |
| Adam        | 0.2397 | 0.6379 | Smooth descent, best generalisation |
| L-BFGS      | 0.0172 | 6.6357 | Fastest to converge, catastrophic overfit |

## Comparison vs runs/iter1_optimizer (MPS / macOS)

| Optimiser | runs1 F1 | runs2 F1 | Δ |
|-----------|---:|---:|---:|
| Adam | 0.8406 | 0.8472 | +0.0066 |
| L-BFGS | 0.8092 | 0.7727 | −0.0365 |
| SGD | 0.7857 | 0.7586 | −0.0271 |
| Nelder-Mead | 0.6203 | 0.6203 | 0.0000 (deterministic) |

**Audit verdict:** headline ranking (Adam > L-BFGS > SGD > Nelder-Mead) unchanged. All three qualitative findings reproduced:
1. Zero-order is stuck on ~1000 parameters (dimension curse).
2. First-order SGD is noisy and overfits.
3. Second-order L-BFGS converges fastest but memorises training data (train 0.017 / val 6.6).
4. Adam is the best-generalising optimiser.

Small F1 deltas come from kernel-level nondeterminism between MPS and CUDA across SGD/L-BFGS stochastic paths. Nelder-Mead is deterministic and reproduces exactly.

## Artefacts
- `runs2/iter1_optimizer/summary.csv`
- `runs2/iter1_optimizer/optimizer_convergence.png`
- `runs2/iter1_optimizer/pca_info.json` (20 retained PCs, ~58.3% variance)
- Per-optimiser folders with `metrics.json`, `confusion_matrix.png`, `classification_report.csv`, `training_history.csv`
- `Documentation/run2_doc/03_optimizer/stdout.log`
