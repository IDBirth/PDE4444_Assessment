# Top-10 Model Archive

This directory is the **single source of truth** for the best-performing checkpoints from this project.
Each subdirectory contains a symlink `model.pt → ../../<run_path>/...` so the weights are not duplicated.

> **Reproducing**: if you re-run training, point `--output-dir` to a **new** directory (e.g. `runs4/`).
> Never overwrite the runs that produced these checkpoints.

## Scoreboard

| Rank | Name | Type | Run | F1 (defect=pos) | Accuracy | Notes |
|---:|---|---|---|---:|---:|---|
| 1 | `yolo26n_320` | YOLO cls | runs3 | N/A (top1=**1.000**) | **1.000** | img=320 b=64 ep=50 |
| 2 | `yolo26s_320` | YOLO cls | runs3 | N/A (top1=**1.000**) | **1.000** | img=320 b=64 ep=50 |
| 3 | `mobilenet_selu` | MobileNetV2 FT | runs3 | **0.9517** | 0.9493 | img=256 ep=60 p=12 |
| 4 | `mlp_boost` | PyTorch MLP | runs3 | **0.9517** | 0.9493 | 96px 24-trial GELU (128→64) |
| 5 | `mobilenet_elu` | MobileNetV2 FT | runs1 | 0.9517 | 0.9493 | macOS/MPS baseline run |
| 6 | `mobilenet_gelu` | MobileNetV2 FT | runs2 | 0.9504 | 0.9493 | CUDA audit run |
| 7 | `mobilenet_leakyrelu` | MobileNetV2 FT | runs1 | 0.9452 | 0.9420 | macOS/MPS baseline |
| 8 | `mobilenet_relu` | MobileNetV2 FT | runs3 | 0.9444 | 0.9420 | img=256 ep=60 p=12 |
| 9 | `mlp_search` | PyTorch MLP | runs1 | 0.9437 | 0.9420 | 64px 12-trial ELU (256,) |
| 10 | `mobilenet_gelu` | MobileNetV2 FT | runs1 | 0.9429 | 0.9420 | macOS/MPS baseline |

> **F1 note**: F1 is computed with **defect as positive class** (class 0, alphabetical).
> YOLO reports top-1 accuracy only — binary F1 is not available from Ultralytics,
> so the `f1_defect_positive` column is left blank for YOLO rows in `manifest.csv`
> and the top-1 value lives in `top1_accuracy`. This keeps the manifest cleanly
> machine-readable (numeric columns stay numeric, string placeholders removed).

## Directory layout

```
top_models/
├── README.md                   ← this file
├── manifest.csv                ← machine-readable scoreboard
├── eval_all.py                 ← re-evaluate all checkpoints on test set
└── 01_yolo26n_320/
│   └── model.pt                ← copy of runs3/iter1_yolo_320/train/weights/best.pt
├── 02_yolo26s_320/
│   └── model.pt                ← copy of runs3/iter2_yolo_s/train/weights/best.pt
... (10 subdirs total)
```

All checkpoints are **PyTorch .pt files** — no TensorRT engine / ONNX /
TorchScript export is produced by the project. Load directly via
`torch.load(...)` or `YOLO(path)` / `ultralytics.YOLO(path)` for YOLO
classify checkpoints.

## Re-evaluating on the test set

```bash
cd /home/ubu/Desktop/Assessment
.venv/bin/python top_models/eval_all.py \
    --data-dir defect_classification_stack/runs/data_balanced \
    --output-dir top_models/eval_results
```

## Data

Raw dataset: `zeroq_cup_classification_scaffold/data/raw/`
Balanced split used for all runs: `defect_classification_stack/runs/data_balanced/`
(train 483 / val 69 / test 138, 50-50 balanced)
