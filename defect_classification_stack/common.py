from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Metrics:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])


def load_image(path: str | Path, size: tuple[int, int], grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    if grayscale:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        return image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return image


def infer_class_names(data_dir: str | Path, split: str = "train") -> list[str]:
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    classes = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    if not classes:
        raise ValueError(f"No classes found in: {split_dir}")
    return classes


def load_split_as_arrays(
    data_dir: str | Path,
    split: str,
    size: tuple[int, int],
    grayscale: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data_dir = Path(data_dir)
    class_names = infer_class_names(data_dir)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / split / class_name
        if not class_dir.exists():
            continue
        for image_path in list_images(class_dir):
            images.append(load_image(image_path, size=size, grayscale=grayscale))
            labels.append(class_idx)

    if not images:
        raise ValueError(f"No images found in split '{split}' under {data_dir}")

    return np.array(images), np.array(labels), class_names


def compute_binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    model_name: str,
) -> Metrics:
    return Metrics(
        model_name=model_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def save_metrics(metrics: Metrics, output_file: str | Path) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)


def save_predictions_csv(
    image_paths: Sequence[str | Path],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_file: str | Path,
) -> None:
    rows = []
    for image_path, true_idx, pred_idx in zip(image_paths, y_true, y_pred):
        rows.append(
            {
                "image_path": str(image_path),
                "y_true": int(true_idx),
                "y_pred": int(pred_idx),
                "true_label": class_names[int(true_idx)],
                "pred_label": class_names[int(pred_idx)],
                "correct": int(true_idx == pred_idx),
            }
        )
    pd.DataFrame(rows).to_csv(output_file, index=False)


def save_confusion_matrix_plot(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_file: str | Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200)
    plt.close(fig)


def save_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_file: str | Path,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(class_names),
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(output_file)


def get_test_image_paths(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    class_names = infer_class_names(data_dir, split="test")
    paths: list[Path] = []
    for class_name in class_names:
        paths.extend(list_images(data_dir / "test" / class_name))
    return paths
