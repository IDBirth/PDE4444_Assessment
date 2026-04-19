from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer

from common import ensure_dir, infer_class_names, set_seed


def find_metrics_csv(project_dir: Path) -> Path | None:
    candidates = sorted(project_dir.rglob("results.csv"))
    return candidates[0] if candidates else None


def resolve_best_checkpoint(save_dir: Path) -> Path:
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Expected trained checkpoint not found: {best_pt}")
    return best_pt


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        relative = path.relative_to(src)
        target = dst / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


class NoAugClassificationTrainer(ClassificationTrainer):
    """Force the training split onto the non-augmented classification transform path."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO26n in classification mode.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
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
    print("[INFO] Ultralytics online augmentation: disabled")
    results = model.train(
        trainer=NoAugClassificationTrainer,
        data=str(args.data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(output_dir),
        name="train",
        seed=args.seed,
        pretrained=True,
        verbose=True,
    )

    external_save_dir = Path(getattr(results, "save_dir", output_dir))
    local_save_dir = output_dir / external_save_dir.name
    if external_save_dir.exists() and external_save_dir.resolve() != local_save_dir.resolve():
        copy_tree(external_save_dir, local_save_dir)
        print(f"[INFO] copied YOLO artefacts to local run dir: {local_save_dir}")
    elif external_save_dir.exists():
        local_save_dir = external_save_dir

    best_checkpoint = resolve_best_checkpoint(local_save_dir)

    # Re-load the trained checkpoint and pin validation outputs to the local run
    # directory. Without this, Ultralytics may fall back to a global classify/val-*
    # path from user settings, which breaks reproducibility and can fail in
    # restricted environments.
    eval_model = YOLO(str(best_checkpoint))
    val_metrics = eval_model.val(
        data=str(args.data_dir),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(output_dir),
        name="val_test",
        exist_ok=True,
    )

    summary = {
        "model_name": args.model,
        "classes": class_names,
        "top1": float(getattr(val_metrics, "top1", 0.0)),
        "top5": float(getattr(val_metrics, "top5", 0.0)),
        "fitness": float(getattr(val_metrics, "fitness", 0.0)),
        "save_dir": str(local_save_dir),
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
