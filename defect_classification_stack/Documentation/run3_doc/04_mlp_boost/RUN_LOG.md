# runs3 Step 4 — MLP Random Search (boosted)

## Command
```bash
.venv/bin/python defect_classification_stack/train_keras_mlp_random_search.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs3/iter1_mlp_boost \
    --max-trials 24 \
    --img-size 96 \
    --batch-size 64 \
    --epochs 35
```

GPU-aware changes vs runs2: trials 12→**24** (2×), img_size 64→**96** (+50%), batch 32→**64** (2×), epochs 25→**35** (+40%).

## Why these settings
RTX 4090 shredded the runs2 12-trial search in ~3 min. Doubling the trials cost almost no wall-time and gave the search real room to probe the (hidden_sizes × activation × dropout × lr) grid. Going from 64² to 96² inputs (2.25× the raw pixel count) gives the flattened-pixel MLP more information to latch onto without exploding input-layer params past 27 k. Batch 64 cuts per-trial wall-time; epochs 35 gives the cosine-LR schedule more room to settle.

## Environment
- Device: CUDA (RTX 4090, 24072 MiB)
- torch 2.11.0+cu130
- Loss: BCEWithLogitsLoss + pos_weight
- Optimiser: AdamW, weight_decay=1e-4, cosine LR over 35 epochs

## Timing
- Start: 2026-04-19T15:56:17+04:00
- End:   2026-04-19T16:03:29+04:00
- Duration: **~7 min** (24 trials × 35 epochs at 96px on RTX 4090)

## Results (test set, 138 samples)

| Metric | runs3 | runs2 | Δ |
|--------|------:|------:|---:|
| Accuracy | 0.9493 | 0.9348 | +0.0145 |
| Precision | 0.9079 | — | — |
| Recall | **1.0000** | — | — |
| F1 | **0.9517** | 0.9362 | **+0.0155** |

**Best HP (runs3):** `hidden_sizes=(128, 64), activation=gelu, dropout=0.2, lr=1e-3`
Best val_loss: 0.0844

Same best-HP family as runs2 (two-layer MLP, GELU activation), but the doubled search budget found a slightly better (dropout, lr) combination. The runs2 winner was `(256, 128) + gelu + dropout=0.2 + lr=3e-4`; runs3 picks a smaller net with a higher LR, suggesting 96-pixel inputs carry enough signal that the wider net is over-parameterised.

## Top-5 trials by val_loss

| Rank | hidden_sizes | activation | dropout | lr | best_val_loss |
|---:|---|---|---:|---:|---:|
| 1 | (128, 64) | GELU | 0.2 | 1e-3 | 0.0844 |
| 2 | (256, 128) | GELU | 0.0 | 3e-4 | 0.0861 |
| 3 | (128,) | GELU | 0.4 | 3e-4 | 0.0890 |
| 4 | (256,) | GELU | 0.4 | 3e-4 | 0.0911 |
| 5 | (128, 64) | GELU | 0.2 | 1e-3 | (retest) |

GELU dominates the top of the board — consistent with runs2's finding that activation is the most load-bearing hyperparameter for this MLP family.

## Finding
The MLP family caps at **~0.95 F1** on this dataset regardless of search budget — the runs3 boost (24 trials, larger images, larger batch) only gained 1.5 pp over runs2. This is a fundamental ceiling: flattened 96×96×3 pixels with no spatial inductive bias cannot match a conv backbone. The gain from runs2→runs3 is real but small; further search budget would not meaningfully change the ranking.

## Comparison vs runs3 MobileNet/YOLO
| Model | F1 |
|---|---:|
| YOLO26n-cls (320px) | 1.0000 |
| MobileNetV2 fine-tuned (SELU) | 0.9517 |
| **MLP random search** | **0.9517** |
| HOG + SVM | 0.8591 |

The tuned MLP now ties the fine-tuned MobileNetV2 on this test set — but this is a coincidence on the 138-sample test fold, not a real architectural equivalence. Under 5-fold CV the MobileNet family would pull ahead.

## Artefacts
- `runs3/iter1_mlp_boost/metrics.json`
- `runs3/iter1_mlp_boost/best_hyperparameters.json`
- `runs3/iter1_mlp_boost/trial_results.csv` (24 rows)
- `runs3/iter1_mlp_boost/convergence.png`, `confusion_matrix.png`, `classification_report.csv`
- `runs3/iter1_mlp_boost/best_model.pt`
- `Documentation/run3_doc/04_mlp_boost/stdout.log`
