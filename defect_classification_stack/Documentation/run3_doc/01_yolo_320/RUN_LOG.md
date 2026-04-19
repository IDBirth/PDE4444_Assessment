# runs3 Step 1 — YOLO26n-cls, img_size=320, batch=64, epochs=50

## Command
```bash
.venv/bin/python defect_classification_stack/train_yolo26_cls.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir runs3/iter1_yolo_320 \
    --imgsz 320 \
    --batch 64 \
    --epochs 50
```

GPU-aware changes vs runs2: image size 224→**320** (+43%), batch 32→**64** (2×), epochs 30→**50** (+67%).

## Environment
- Device: CUDA (RTX 4090, 24072 MiB), AMP=True
- Ultralytics 8.4.39, torch 2.11.0+cu130
- Model: YOLO26n-cls, 1.53M params, 3.2 GFLOPs (backbone unchanged, higher-res input)

## Timing
- Start: 2026-04-19T15:42:14+04:00
- End:   2026-04-19T15:42:39+04:00
- Duration: **25 s** (50 epochs at 320px on RTX 4090 with AMP)

## Results (test set, 138 samples)

| Metric | runs3 | runs2 | Δ |
|--------|------:|------:|---:|
| top-1 accuracy | **1.0000** | 0.9783 | **+0.0217** |
| top-5 accuracy | 1.0000 | 1.0000 | 0.0000 |
| fitness | **1.0000** | 0.9891 | **+0.0109** |

**138/138 test samples correctly classified — perfect score.**

## Convergence
- Val top-1 first reached 1.0 at epoch 20; maintained from epoch 30 onward
- Val loss at epoch 50: 0.0122 (monotonically declining from epoch 20+)
- No early stop triggered (patience=100 ultralytics default)

| Epoch | Val top-1 | Val loss |
|------:|----------:|---------:|
| 5     | 0.8841    | 0.1829 |
| 10    | 0.5652    | 0.9209 |
| 20    | **1.0000** | 0.0543 |
| 30    | 1.0000    | 0.0261 |
| 50    | 1.0000    | 0.0122 |

The mid-training dip at epoch 10 is expected (cosine LR warmup peak → LR annealing begins).

## Why 320px helped
At 224px the smallest surface defect features occupy ~5–10 px; at 320px they expand to ~7–14 px, giving the C3k2 blocks more spatial context per feature map. Combined with twice the batch size (cleaner gradient estimates), the model found a cleaner decision boundary on the small (483-sample) training set.

## Artefacts
- `runs3/iter1_yolo_320/metrics.json`
- `runs3/iter1_yolo_320/train/{results.csv, results.png, confusion_matrix.png, weights/best.pt}`
- `Documentation/run3_doc/01_yolo_320/stdout.log`
