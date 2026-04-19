from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from common import ensure_dir, infer_class_names, set_seed


def find_metrics_csv(project_dir: Path) -> Path | None:
    candidates = sorted(project_dir.rglob("results.csv"))
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO26n in classification mode.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="yolo26n-cls.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    for split in ["train", "val", "test"]:
        split_dir = args.data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Required split directory not found: {split_dir}")

    class_names = infer_class_names(args.data_dir, split="train")
    if len(class_names) != 2:
        raise ValueError(f"Expected 2 classes, found {len(class_names)}: {class_names}")

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(output_dir),
        name="train",
        seed=args.seed,
        pretrained=True,
        verbose=True,
    )

    val_metrics = model.val(data=str(args.data_dir), split="test", imgsz=args.imgsz, batch=args.batch)

    summary = {
        "model_name": args.model,
        "classes": class_names,
        "top1": float(getattr(val_metrics, "top1", 0.0)),
        "top5": float(getattr(val_metrics, "top5", 0.0)),
        "fitness": float(getattr(val_metrics, "fitness", 0.0)),
        "save_dir": str(getattr(results, "save_dir", output_dir)),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    results_csv = find_metrics_csv(output_dir)
    if results_csv is not None:
        df = pd.read_csv(results_csv)
        df.to_csv(output_dir / "training_results.csv", index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
