#!/usr/bin/env python3
"""
Re-evaluate all top-10 checkpoints against the held-out test set.
Results are written to --output-dir/eval_results.csv.

Usage:
    cd /home/ubu/Desktop/Assessment
    .venv/bin/python top_models/eval_all.py \
        --data-dir defect_classification_stack/runs/data_balanced \
        --output-dir top_models/eval_results
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
STACK_ROOT = REPO_ROOT / "defect_classification_stack"
sys.path.insert(0, str(STACK_ROOT))

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.csv"

DEVICE_STR: str = ""

# ── Lazy imports (keep startup fast) ────────────────────────────────────────

def _device():
    import torch
    global DEVICE_STR
    d = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    DEVICE_STR = str(d)
    return d


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_yolo(ckpt: Path):
    from ultralytics import YOLO
    return ("yolo", YOLO(str(ckpt)), None)


def _load_mobilenet(ckpt: Path):
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    # Infer activation from directory name
    act_map = {
        "relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU,
        "selu": nn.SELU, "leaky_relu": nn.LeakyReLU,
    }
    act_name = "relu"
    for part in ckpt.parts:
        for k in act_map:
            if k in part.lower():
                act_name = k
                break
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    backbone.classifier = nn.Sequential(
        nn.Linear(backbone.last_channel, 256),
        nn.BatchNorm1d(256),
        act_map[act_name](),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state, strict=True)
    backbone.to(_device()).eval()
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return ("mobilenet", backbone, tf)


def _load_mlp(ckpt: Path):
    import ast
    import torch
    import torch.nn as nn
    from torchvision import transforms

    _ACT = {"relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU,
            "selu": nn.SELU, "leaky_relu": nn.LeakyReLU}

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    input_dim = int(state["net.1.weight"].shape[0])
    img_size  = round((input_dim / 3) ** 0.5)

    hp_path = ckpt.parent / "best_hyperparameters.json"
    if hp_path.exists():
        hp = json.loads(hp_path.read_text())
        hidden = ast.literal_eval(hp["hidden_sizes"])
        act    = hp["activation"]
        drop   = float(hp["dropout"])
    else:
        hidden, act, drop = (256,), "elu", 0.4

    layers = [nn.Flatten(), nn.BatchNorm1d(input_dim)]
    in_d = input_dim
    for h in hidden:
        layers += [nn.Linear(in_d, h), _ACT[act]()]
        if drop > 0:
            layers.append(nn.Dropout(drop))
        in_d = h
    layers.append(nn.Linear(in_d, 1))

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x).squeeze(1)

    model = _MLP()
    model.load_state_dict(state, strict=True)
    model.to(_device()).eval()

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return ("mlp", model, tf)


def _detect_type(ckpt: Path) -> str:
    try:
        from ultralytics import YOLO
        m = YOLO(str(ckpt))
        if getattr(m, "task", None) == "classify":
            return "yolo"
    except Exception:
        pass
    import torch
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    keys = list(state.keys())
    if any(k.startswith("net.") for k in keys):
        return "mlp"
    return "mobilenet"


# ── Inference ────────────────────────────────────────────────────────────────

def _eval_yolo(model, data_dir: Path) -> dict:
    import numpy as np
    correct = total = 0
    tp = fp = fn = 0
    for label_idx, label_name in enumerate(["defect", "non_defect"]):
        class_dir = data_dir / "test" / label_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            result   = model.predict(source=str(img_path), verbose=False)[0]
            top_idx  = int(result.probs.top1)
            pred_name = str(result.names[top_idx])
            pred_is_defect = "non" not in pred_name.lower() and "def" in pred_name.lower()
            true_is_defect = label_name == "defect"
            if pred_is_defect == true_is_defect:
                correct += 1
            if true_is_defect and pred_is_defect:
                tp += 1
            elif not true_is_defect and pred_is_defect:
                fp += 1
            elif true_is_defect and not pred_is_defect:
                fn += 1
            total += 1
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "accuracy": correct / max(total, 1),
        "precision": prec, "recall": rec, "f1": f1,
        "top1_accuracy": correct / max(total, 1),
        "total": total,
    }


def _eval_torch(model_tuple, data_dir: Path) -> dict:
    import torch
    import numpy as np
    import cv2

    model_type, model, tf = model_tuple
    preds_list, labels_list = [], []
    device = _device()

    for label_idx, label_name in enumerate(["defect", "non_defect"]):
        class_dir = data_dir / "test" / label_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = tf(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(tensor)
            pos_prob = float(torch.sigmoid(logit).squeeze().item())
            # pos_prob = P(non_defect); pred=0 means defect
            pred = 0 if pos_prob < 0.5 else 1
            preds_list.append(pred)
            labels_list.append(label_idx)  # defect=0, non_defect=1

    y_pred = np.array(preds_list)
    y_true = np.array(labels_list)
    tp = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 0) & (y_true == 1)).sum())
    fn = int(((y_pred == 1) & (y_true == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "accuracy":  float((y_pred == y_true).mean()),
        "precision": prec, "recall": rec, "f1": f1,
        "top1_accuracy": None,
        "total": len(y_pred),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # Read manifest
    rows = []
    with MANIFEST_PATH.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    results = []
    for entry in rows:
        rank = entry["rank"]
        name = entry["name"]
        ckpt = REPO_ROOT / entry["checkpoint_path"]
        print(f"\n[rank {rank}] {name}  →  {ckpt}")
        if not ckpt.exists():
            print(f"  SKIP: checkpoint not found")
            results.append({**entry, "eval_accuracy": "MISSING", "eval_f1": "MISSING"})
            continue

        mtype = _detect_type(ckpt)
        try:
            if mtype == "yolo":
                mt = _load_yolo(ckpt)
                metrics = _eval_yolo(mt[1], data_dir)
            else:
                mt = _load_mlp(ckpt) if mtype == "mlp" else _load_mobilenet(ckpt)
                metrics = _eval_torch(mt, data_dir)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({**entry, "eval_accuracy": "ERROR", "eval_f1": "ERROR"})
            continue

        acc = metrics["accuracy"]
        f1  = metrics["f1"]
        prec = metrics["precision"]
        rec  = metrics["recall"]
        print(f"  acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}  n={metrics['total']}")
        results.append({
            **entry,
            "eval_accuracy":  f"{acc:.4f}",
            "eval_precision": f"{prec:.4f}",
            "eval_recall":    f"{rec:.4f}",
            "eval_f1":        f"{f1:.4f}",
            "n_test":         metrics["total"],
        })

    out_csv = args.output_dir / "eval_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
