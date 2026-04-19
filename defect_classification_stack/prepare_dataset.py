from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from common import IMAGE_EXTENSIONS, ensure_dir, set_seed


def collect_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    files = sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
    if not files:
        raise ValueError(f"No images found in: {folder}")
    return files


def copy_files(paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for src in paths:
        dst = destination / src.name
        stem = src.stem
        suffix = src.suffix
        counter = 1
        while dst.exists():
            dst = destination / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/val/test folder splits from two raw class folders.")
    parser.add_argument("--defect-dir", required=True, type=Path)
    parser.add_argument("--pass-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction of full dataset used for validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", nargs=2, type=int, default=[224, 224], help="Stored in metadata for downstream scripts.")
    args = parser.parse_args()

    set_seed(args.seed)

    defect_images = collect_images(args.defect_dir)
    pass_images = collect_images(args.pass_dir)

    image_paths = defect_images + pass_images
    labels = [1] * len(defect_images) + [0] * len(pass_images)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths,
        labels,
        test_size=args.test_size,
        stratify=labels,
        random_state=args.seed,
    )

    val_ratio_within_trainval = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_within_trainval,
        stratify=y_train_val,
        random_state=args.seed,
    )

    output_dir = ensure_dir(args.output_dir)
    class_map = {0: "non_defect", 1: "defect"}

    for split_name, split_paths, split_labels in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        for label_value, class_name in class_map.items():
            selected = [p for p, y in zip(split_paths, split_labels) if y == label_value]
            copy_files(selected, output_dir / split_name / class_name)

    metadata = {
        "class_map": class_map,
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "img_size": args.img_size,
        "counts": {
            "train": dict(Counter(y_train)),
            "val": dict(Counter(y_val)),
            "test": dict(Counter(y_test)),
        },
    }

    with (output_dir / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))
    print(f"\nPrepared dataset written to: {output_dir}")


if __name__ == "__main__":
    main()
