# runs3 Step 2 — YOLO26s-cls, img_size=320, batch=64, epochs=50

## Command
```bash
.venv/bin/python defect_classification_stack/train_yolo26_cls.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs3/iter2_yolo_s \
    --model yolo26s-cls.pt \
    --imgsz 320 \
    --batch 64 \
    --epochs 50
```

New backbone: **YOLO26s** (small variant) vs YOLO26n (nano). All other settings match Step 1 for a controlled comparison.

## Environment
- Device: CUDA (RTX 4090, 24072 MiB), AMP=True
- Ultralytics 8.4.39, torch 2.11.0+cu130
- Model: YOLO26s-cls (larger than nano — more C3k2 channels and deeper C2PSA)

## Timing
- Start: 2026-04-19T15:42:39+04:00
- End:   2026-04-19T15:43:09+04:00 (approx)
- Duration: **~30 s** (50 epochs at 320px on RTX 4090 with AMP)

## Results (test set, 138 samples)

| Metric | YOLO26s (runs3) | YOLO26n runs3 | YOLO26n runs2 |
|--------|----------------:|---------------:|---------------:|
| top-1 accuracy | **1.0000** | **1.0000** | 0.9783 |
| top-5 accuracy | 1.0000 | 1.0000 | 1.0000 |
| fitness | **1.0000** | **1.0000** | 0.9891 |

Both nano and small hit 100%. The small model converges slightly more smoothly (lower mid-training oscillation) but the test set is too small to distinguish them statistically.

## Convergence

| Epoch | Val top-1 | Val loss |
|------:|----------:|---------:|
| 5     | 0.9130    | 0.1444 |
| 10    | 0.9710    | 0.0890 |
| 20    | 0.9565    | 0.1429 |
| 30    | 0.9710    | 0.0409 |
| 40    | **1.0000** | 0.0270 |
| 50    | 1.0000    | 0.0215 |

YOLO26s reaches val-1.0 at epoch 40 vs YOLO26n at epoch 20. The nano model's lighter architecture lets it converge faster on this small dataset; the small model is more regularised but needs more epochs.

## Finding
For this 483-sample dataset with 320px input, **YOLO26n is the better choice**: same final accuracy, twice as fast per epoch, faster convergence. The small model offers no advantage here. For larger or harder datasets the capacity gap would matter more.

## Artefacts
- `runs3/iter2_yolo_s/metrics.json`
- `runs3/iter2_yolo_s/train/{results.csv, results.png, confusion_matrix.png, weights/best.pt}`
- `Documentation/run3_doc/02_yolo_s/stdout.log`
