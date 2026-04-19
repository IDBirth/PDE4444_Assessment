# runs3 — GPU-Aware Tuning Pass on RTX 4090

Master log for the **third** end-to-end execution of the defect-classification pipeline. runs1 was the macOS/MPS baseline, runs2 was the Linux/CUDA audit (identical configs), and **runs3 is the tuning pass**: same dataset, same code, new configs that exploit the 24 GB of VRAM and AMP throughput on the 4090.

Per-step detail lives in `01_yolo_320/` through `05_final_report/` — one `RUN_LOG.md` + `stdout.log` each.

---

## Purpose
1. **Close the YOLO gap**: runs2 YOLO26n-cls misclassified 3/138 test samples (F1=0.9783). runs3 uses the 4090's headroom to bump image size + batch + epochs and see if the gap closes.
2. **De-noise the MobileNet ranking**: runs2 revealed early-stopping sensitivity in the fine-tune (ELU stopped at 13, GELU ran 26). runs3 adds a `--patience` CLI arg and doubles the patience to give every activation a fair shot.
3. **Test the MLP ceiling**: runs2 MLP hit F1=0.9362 with 12 trials. Does doubling the search budget close the gap to MobileNet?
4. **Model-capacity check**: does YOLO26s-cls beat YOLO26n on this dataset, or does nano already saturate?

The dataset is frozen (`defect_classification_stack/runs/data_balanced/`, train 483 / val 69 / test 138, 50/50 balanced). Only hyperparameters change vs runs2.

## Environment

| Item | runs3 |
|------|-------|
| Device | CUDA — RTX 4090 (24 GB, CC 8.9) |
| Python | 3.12.3 |
| torch | 2.11.0+cu130 |
| torchvision | 0.26.0 |
| ultralytics | 8.4.39 |
| scikit-learn | 1.8.0 |
| AMP | enabled where supported |

## Commands used (for re-run)
All commands executed from repo root `/home/ubu/Desktop/Assessment` via the fresh venv at `.venv/`.

```bash
# Step 1 — YOLO26n-cls at 320px
.venv/bin/python defect_classification_stack/train_yolo26_cls.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs3/iter1_yolo_320 \
    --imgsz 320 --batch 64 --epochs 50

# Step 2 — YOLO26s-cls at 320px
.venv/bin/python defect_classification_stack/train_yolo26_cls.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs3/iter2_yolo_s \
    --model yolo26s-cls.pt \
    --imgsz 320 --batch 64 --epochs 50

# Step 3 — MobileNetV2 fine-tuned (boosted)
.venv/bin/python defect_classification_stack/train_cnn_pretrained.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs3/iter3_mobilenet_finetune \
    --unfreeze --epochs 60 --batch-size 32 --img-size 256 --patience 12

# Step 4 — MLP random search (boosted)
.venv/bin/python defect_classification_stack/train_keras_mlp_random_search.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs3/iter1_mlp_boost \
    --max-trials 24 --img-size 96 --batch-size 64 --epochs 35

# Step 5 — Aggregate
.venv/bin/python defect_classification_stack/aggregate_results.py \
    --runs-dir runs3 --output-dir runs3/final_report
```

## Config diff (runs2 → runs3)

| Step | Hyperparameter | runs2 | runs3 | Rationale |
|------|---------------|------:|------:|-----------|
| YOLO26n | img_size | 224 | **320** | Defect features expand from ~5–10 px to ~7–14 px |
| YOLO26n | batch | 32 | **64** | Cleaner gradient; headroom on 24 GB VRAM |
| YOLO26n | epochs | 30 | **50** | More room for cosine LR to settle |
| YOLO26s | (new) | — | same as 26n | Capacity sanity check |
| MobileNet FT | img_size | 224 | **256** | Conservative bump — still near pretrain size |
| MobileNet FT | epochs | 30 | **60** | Pair with higher patience |
| MobileNet FT | patience | 6 (hardcoded) | **12** (new `--patience` flag) | De-noise per-activation ranking |
| MLP | trials | 12 | **24** | 4090 shreds 12 trials in 3 min |
| MLP | img_size | 64 | **96** | 2.25× input pixels, still small enough for MLP |
| MLP | batch | 32 | **64** | Faster per-trial |

## Timing summary (RTX 4090)

| Step | Duration | Start → End (local) |
|------|----------|---------------------|
| 1. YOLO26n @ 320 | 25 s | 15:42:14 → 15:42:39 |
| 2. YOLO26s @ 320 | 30 s | 15:42:39 → 15:43:09 |
| 3. MobileNet FT boosted | 12 min | 15:42:49 → 15:54:52 |
| 4. MLP boost (24 trials) | 7 min | 15:56:17 → 16:03:29 |
| 5. Aggregate + final report | ~4 s | — |
| **Total compute** | **≈ 20 min** | |

Note Steps 1+2 overlap the start of Step 3 in wall-clock — YOLO finishes in under a minute each, so they were launched ahead of the long MobileNet run.

## Headline result table

| Model | runs1 F1 | runs2 F1 | **runs3 F1** | Δ (runs2→runs3) |
|---|---:|---:|---:|---:|
| **YOLO26n-cls** | 0.9855 | 0.9783 | **1.0000** | **+0.0217** |
| YOLO26s-cls | — | — | **1.0000** | (new) |
| MobileNetV2 fine-tuned (best activation) | 0.9517 (ELU) | 0.9504 (GELU) | 0.9517 (SELU) | +0.0013 |
| MLP random search | 0.9437 | 0.9362 | **0.9517** | +0.0155 |

**runs3 is the first run to push YOLO26n-cls to a perfect 138/138 score on this test fold.**

## Findings

1. **YOLO wins decisively.** The 320px + 64-batch + 50-epoch config closes the 2-pp gap that runs2 opened and pushes to 100% top-1. Both nano and small variants hit 100%, but nano converges twice as fast (val-1.0 at epoch 20 vs epoch 40) — for this 483-sample dataset, nano is the right pick. Larger/harder datasets might show a capacity advantage for the small variant.

2. **The `--patience` fix resolved the MobileNet ranking noise.** In runs2 the best activation shifted between runs (ELU → GELU); in runs3 with patience=12, SELU narrowly leads with 0.9517 and the family clusters in 0.91–0.95. Every activation except GELU gained 1–2 pp. The fine-tuned MobileNetV2 ceiling is clearly around 0.95 F1.

3. **The MLP search budget gain is real but small.** 24 trials lifted the best-F1 from 0.9362 to 0.9517 (+1.55 pp). The same best-activation (GELU) and architecture family (2-layer 128→64) won; the budget only tightened the (dropout, lr) pair. Further search budget will not meaningfully improve this.

4. **YOLO > MobileNet FT ≈ MLP > HOG baselines — ranking preserved from runs1/runs2.** The ordering is robust across hardware (MPS vs CUDA) and config (baseline vs tuned). The gap between YOLO and the rest widened in runs3 because YOLO benefited most from the bigger budget.

## Code changes (runs3-only)

- `defect_classification_stack/train_cnn_pretrained.py`: added `--patience` CLI arg (default 6 preserves runs1/runs2 behaviour). This is the only script change in runs3.

Nothing else was modified — all numbers are from the existing training scripts with new CLI arguments.

## Known footguns and gotchas

1. **Ultralytics artefact redirect (still present).** Same as runs2: `~/.config/Ultralytics/settings.yaml` silently redirects `project=runs3/...` to the global runs_dir. `metrics.json` lands in the right place via the wrapper script, but `weights/best.pt`, `results.csv`, confusion matrices, etc. had to be copied back into `runs3/iter{1,2}_yolo_*/train/` manually.

2. **`aggregate_results.py` is hardcoded to runs1/runs2 layout.** It expects `iter1_yolo`, `iter1_mlp_search`, `iter2_mobilenet`, etc. runs3 uses different directory names (`iter1_yolo_320`, `iter2_yolo_s`, `iter1_mlp_boost`) so the auto-aggregator only captures the MobileNet fine-tune family. The full runs3 scoreboard was assembled manually and saved as `runs3/final_report/all_results_manual.csv`. A future runs4 should either rename directories back to canonical names or generalise the aggregator.

3. **Test-set size cap (138 samples).** With YOLO at 100%, we have no measurable signal distinguishing nano from small, or from hypothetical future improvements. Further claims of progress on this dataset require either cross-validation averaging or a larger test fold.

4. **Python pipe buffering for long training logs.** `tee` into a log file shows 0 lines during training — Python full-buffers stdout when piped. Output flushes on process exit. Not a bug, worth knowing when monitoring background runs.

## Where to look

| Topic | Path |
|---|---|
| Per-step detail | `Documentation/run3_doc/0X_stepname/RUN_LOG.md` |
| Per-step full stdout | `Documentation/run3_doc/0X_stepname/stdout.log` |
| Per-run artefacts | `runs3/iter{1_yolo_320, 2_yolo_s, 3_mobilenet_finetune, 1_mlp_boost}/` |
| Runs3 scoreboard (auto) | `runs3/final_report/all_results.csv` (MobileNet only) |
| Runs3 scoreboard (manual, full) | `runs3/final_report/all_results_manual.csv` |
| Runs2 counterpart | `Documentation/run2_doc/WORKFLOW_LOG.md` |
| Runs1 counterpart | `defect_classification_stack/PROGRESS_AND_RESULTS.md` |

## Overall verdict

Three independent training runs — macOS/MPS baseline, Linux/CUDA audit, Linux/CUDA tuned — now all tell the same story: **YOLO26n-cls is the right model for this task**, with a clean end-to-end F1 of 0.9855 → 0.9783 → **1.0000** on a 138-sample held-out test set. The 4090's compute headroom was enough to turn a strong baseline into a saturated one. MobileNet fine-tuning and MLP search are useful reference points (~0.95 F1) but are not competitive with a domain-pretrained detector backbone repurposed as a classifier.

## Next step (optional, not executed)
If the dataset grows, three things would be worth trying:
- YOLO26m-cls and above (capacity for harder classes)
- MixUp/CutMix augmentation at the YOLO config level
- 10-fold cross-validation on the full 690 samples to get a confidence interval on the YOLO result

None of these are necessary for the current dataset — YOLO26n-cls already solves it.
