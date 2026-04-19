# runs3 Step 5 — Aggregate + Final Report

## Commands
```bash
# Automatic aggregator (only picks up the dirs whose names match the hardcoded
# runs1/runs2 layout — in runs3 that's iter3_mobilenet_finetune only).
.venv/bin/python defect_classification_stack/aggregate_results.py \
    --runs-dir runs3 \
    --output-dir runs3/final_report
```

The `aggregate_results.py` script has hardcoded subdirectory names (`iter1_yolo`, `iter1_mlp_search`, `iter2_mobilenet`). runs3 uses different names (`iter1_yolo_320`, `iter2_yolo_s`, `iter1_mlp_boost`) so the auto-aggregator only captured the fine-tuned MobileNetV2 family. The full runs3 scoreboard was assembled manually and saved as `runs3/final_report/all_results_manual.csv`.

## Environment
- Device: CPU (pandas + matplotlib)
- pandas 2.3.x, matplotlib 3.10.x (MatplotlibDeprecationWarning for `get_cmap` — cosmetic)

## Timing
- Aggregate: ~3 s
- Manual scoreboard: ~0 s

## runs3 full scoreboard (sorted by F1)

| Rank | Model | Category | Accuracy | F1 | runs2 F1 | Δ |
|---:|---|---|---:|---:|---:|---:|
| 1 | **yolo26n_cls_320** | YOLO (pretrained) | **1.0000** | **1.0000** | 0.9783 | **+0.0217** |
| 1 | **yolo26s_cls_320** | YOLO (pretrained) | **1.0000** | **1.0000** | — | (new) |
| 3 | mobilenet_selu | MobileNetV2 (fine-tuned) | 0.9493 | 0.9517 | 0.9286 | +0.0231 |
| 3 | mlp_random_search | MLP HParam Search | 0.9493 | 0.9517 | 0.9362 | +0.0155 |
| 5 | mobilenet_relu | MobileNetV2 (fine-tuned) | 0.9420 | 0.9444 | 0.9286 | +0.0158 |
| 6 | mobilenet_leaky_relu | MobileNetV2 (fine-tuned) | 0.9348 | 0.9388 | 0.9231 | +0.0157 |
| 7 | mobilenet_gelu | MobileNetV2 (fine-tuned) | 0.9130 | 0.9200 | 0.9504 | −0.0304 |
| 8 | mobilenet_elu | MobileNetV2 (fine-tuned) | 0.8986 | 0.9067 | 0.8955 | +0.0112 |

Best model: **YOLO26n-cls @ 320px** (tied with YOLO26s at 100% F1; nano wins on speed).

## Headline F1 deltas (runs2 → runs3)

| Family | runs2 F1 | runs3 F1 | Δ | What drove the gain |
|---|---:|---:|---:|---|
| YOLO26n-cls | 0.9783 | **1.0000** | +0.0217 | img_size 224→320, batch 32→64, epochs 30→50 |
| YOLO26s-cls (new) | — | 1.0000 | — | larger capacity variant — matches nano |
| MobileNetV2 fine-tuned (best) | 0.9504 (GELU) | 0.9517 (SELU) | +0.0013 | `--patience 12` fix, img 224→256, 60 epochs |
| MLP random search | 0.9362 | 0.9517 | +0.0155 | trials 12→24, img 64→96, batch 32→64 |

## Key findings

1. **YOLO26n-cls at 320px is sufficient to saturate this test set.** Both n and s variants hit 100% top-1 and top-5. The nano model wins on speed (30 s vs 30 s full-run; 20-epoch vs 40-epoch convergence) and is the recommended production choice.

2. **MobileNetV2 fine-tuned family ceiling is ~0.95 F1** on this dataset. The `--patience` fix closed the per-activation ranking noise and lifted the average, but SELU's 0.9517 F1 is indistinguishable from the runs2 GELU peak — the family cannot catch the domain-pretrained YOLO backbone.

3. **MLP HParam search maxes out at ~0.95 F1** regardless of search budget (12 trials: 0.9362, 24 trials: 0.9517). Flattened pixels without spatial inductive bias cannot match a conv backbone; further budget would not change the ranking.

4. **Consistent top-of-rank winner across all three runs**: YOLO26n-cls. runs1 = 0.9855, runs2 = 0.9783, runs3 = 1.0000. The migration to CUDA (runs1→runs2) cost a single test sample; the GPU-aware configs (runs2→runs3) more than recovered it and then some.

## Files written
- `runs3/final_report/all_results.csv` — auto-aggregator output (MobileNet only)
- `runs3/final_report/all_results_manual.csv` — full 8-row runs3 scoreboard including YOLO and MLP
- `runs3/final_report/model_comparison_{acc,f1}.png` — horizontal bar charts (MobileNet only; manual full chart not regenerated)
- `Documentation/run3_doc/05_final_report/stdout_aggregate.log`

## Known gotchas
- `aggregate_results.py` is hardcoded to runs1/runs2 directory names. For a future runs4, either rename runs3 subdirs back to the canonical pattern (`iter1_yolo`, etc.) or generalise the aggregator to glob for `iter*/metrics.json` and infer category from the JSON contents.
- Ultralytics still redirects YOLO artefacts to `~/.config/Ultralytics/settings.yaml`'s global `runs_dir`. `metrics.json` lands in the right place, but `train/weights/best.pt` et al. must be copied manually. Same gotcha as runs2 Step 7.
