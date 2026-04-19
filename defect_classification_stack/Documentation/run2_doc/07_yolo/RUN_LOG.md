# Step 7 — YOLO26n-cls Fine-Tune

## Command
```bash
.venv/bin/python defect_classification_stack/train_yolo26_cls.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs2/iter1_yolo
```

Defaults: epochs=30, batch=32, img_size=224, seed=42, optimizer=auto (SGD with cosine warmup), deterministic=True. Pretrained weights `yolo26n-cls.pt` auto-downloaded from ultralytics/assets v8.4.0.

## Environment
- Device: CUDA (RTX 4090, 24072 MiB), AMP=True
- Ultralytics 8.4.39, torch 2.11.0+cu130, Python 3.12.3
- Model: YOLO26n-cls, 47 layers (fused), 1.53M params, 3.2 GFLOPs
- Classes: `defect`, `non_defect`

## Timing
- Start: 2026-04-19T15:21:30+04:00
- End:   2026-04-19T15:21:49+04:00
- Duration: 19 s (30 epochs on RTX 4090 w/ AMP)

## Results (test set, 138 samples)

| Metric | Value |
|--------|------:|
| top-1 accuracy | **0.9783** |
| top-5 accuracy | 1.0000 |
| fitness | 0.9891 |
| val top-1 (epoch 30) | 0.9855 |
| val loss (epoch 30) | 0.0156 |

Training converged fast: top-1 ≥ 0.95 from epoch 3, first 1.0 val-top-1 at epoch 12, thereafter oscillating 0.971–1.000 with val loss settling at 0.01–0.06 (see `train/results.csv`).

## Comparison vs runs/iter1_yolo (MPS / macOS)

| Metric | runs1 | runs2 | Δ |
|--------|---:|---:|---:|
| test top-1 | 0.9855 | 0.9783 | −0.0072 |
| test top-5 | 1.0000 | 1.0000 | 0.0000 |
| fitness   | 0.9928 | 0.9891 | −0.0037 |

**Audit verdict:** headline result (YOLO26n-cls ≈ 98% top-1, best overall model) **reproduced**. The ~0.7 pp drop is one additional misclassification on the 138-sample test set (3 errors instead of 2) — well within the noise floor given CUDA vs MPS kernel differences in AMP + randaugment paths. Conclusion unchanged: transfer-learning from a YOLO classification backbone is the strongest non-ensemble model in the stack.

## Artefact-path note
Ultralytics 8.4.39 honours a global `settings.yaml` (datasets_dir/runs_dir) that the wrapper cannot override via its own `project=` arg without `exist_ok`. As a result, the trainer wrote artefacts to `/home/ubu/Desktop/MDX-PDE4444/Bilal--PDE4444/runs/classify/runs2/iter1_yolo/train/` (the global ultralytics runs_dir) while `metrics.json` was written to the intended `runs2/iter1_yolo/`. Artefacts have been copied back to `runs2/iter1_yolo/train/` for a self-contained submission bundle.

## Artefacts
- `runs2/iter1_yolo/metrics.json` — top-1 / top-5 / fitness summary
- `runs2/iter1_yolo/train/results.csv` — per-epoch train/val loss + top-1/top-5
- `runs2/iter1_yolo/train/results.png` — training curves
- `runs2/iter1_yolo/train/confusion_matrix.png` + `confusion_matrix_normalized.png`
- `runs2/iter1_yolo/train/weights/best.pt`, `last.pt`
- `runs2/iter1_yolo/train/args.yaml` — exact training config
- `runs2/iter1_yolo/train/{train,val}_batch*.jpg` — augmentation samples
- `Documentation/run2_doc/07_yolo/stdout.log`

## Best runs2 model overall
`yolo26n-cls` — test top-1 97.83%, fitness 0.9891 (3 misclassified out of 138 on balanced test set). Best-in-class for this pipeline.
