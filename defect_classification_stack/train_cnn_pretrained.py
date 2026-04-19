"""
Transfer-Learning CNN  (Iteration 2 – Assessment Section 3)
============================================================
Uses a pretrained MobileNetV2 backbone (ImageNet) with a custom
classification head, comparing the same 5 activation functions.

Why this matters
----------------
Training from scratch on 483 images is insufficient for a deep CNN.
Transfer learning re-uses low-level features (edges, textures) learned
on ImageNet, massively reducing the data needed to converge.

Architecture
------------
  MobileNetV2 backbone (frozen)
  -> AdaptiveAvgPool2d(1) -> Flatten
  -> Linear(1280, 256) -> BatchNorm -> Activation -> Dropout(0.3)
  -> Linear(256, 1)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from common import (
    Metrics,
    ensure_dir,
    save_classification_report,
    save_confusion_matrix_plot,
    save_metrics,
    set_seed,
)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)

ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "relu":       nn.ReLU,
    "elu":        nn.ELU,
    "gelu":       nn.GELU,
    "selu":       nn.SELU,
    "leaky_relu": nn.LeakyReLU,
}


def build_model(activation_cls: type[nn.Module], unfreeze: bool) -> nn.Module:
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all backbone params unless fine-tuning
    for p in backbone.parameters():
        p.requires_grad = unfreeze

    # Always train the last backbone block
    for p in backbone.features[-3:].parameters():
        p.requires_grad = True

    in_features = backbone.last_channel  # 1280
    backbone.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        activation_cls(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    return backbone


def make_loaders(
    data_dir: Path, img_size: int, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.ImageFolder(str(data_dir / "train"), train_tf)
    val_ds   = datasets.ImageFolder(str(data_dir / "val"),   eval_tf)
    test_ds  = datasets.ImageFolder(str(data_dir / "test"),  eval_tf)
    kw = dict(num_workers=0, pin_memory=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
        train_ds.classes,
    )


def compute_pos_weight(loader: DataLoader) -> torch.Tensor:
    labels = torch.cat([y for _, y in loader])
    counts = torch.bincount(labels).float()
    return (counts[0] / counts[1]).unsqueeze(0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    training: bool,
) -> tuple[float, float]:
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE).float()
            logits = model(X).squeeze(1)
            loss = criterion(logits, y)
            if training and optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == y.long()).sum().item()
            total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def predict_all(
    model: nn.Module, loader: DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for X, y in loader:
        logits = model(X.to(DEVICE)).squeeze(1)
        preds.append((torch.sigmoid(logits) >= 0.5).cpu().long())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def train_activation(
    act_name: str,
    act_cls: type[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list[str],
    pos_weight: torch.Tensor,
    output_dir: Path,
    epochs: int,
    unfreeze: bool,
    seed: int,
) -> Metrics:
    torch.manual_seed(seed)
    model = build_model(act_cls, unfreeze).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    # Two-phase LR: low for frozen backbone, higher for head
    optimizer = optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" not in n and p.requires_grad], "lr": 5e-5},
        {"params": model.classifier.parameters(), "lr": 3e-4},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss, best_state = float("inf"), {}
    patience_left = 6
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, True
        )
        vl_loss, vl_acc = run_epoch(
            model, val_loader, criterion, None, False
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(
            f"  [{act_name}] E{epoch:02d}  "
            f"tr={tr_loss:.4f}/{tr_acc:.3f}  "
            f"vl={vl_loss:.4f}/{vl_acc:.3f}"
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_left = 6
        else:
            patience_left -= 1
            if patience_left == 0:
                print("  Early stop.")
                break

    model.load_state_dict(best_state)
    run_dir = ensure_dir(output_dir / act_name)

    y_pred, y_true = predict_all(model, test_loader)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)

    metrics = Metrics(
        model_name=f"mobilenet_{act_name}",
        accuracy=float((y_pred == y_true).mean()),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )
    save_metrics(metrics, run_dir / "metrics.json")
    save_confusion_matrix_plot(
        y_true, y_pred, class_names,
        run_dir / "confusion_matrix.png",
        f"MobileNetV2 {act_name}",
    )
    save_classification_report(
        y_true, y_pred, class_names,
        run_dir / "classification_report.csv",
    )
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    torch.save(model.state_dict(), run_dir / "model.pt")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer-learning CNN sweep (MobileNetV2, PyTorch)."
    )
    parser.add_argument("--data-dir",   required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size",   type=int, default=224)
    parser.add_argument("--unfreeze",   action="store_true",
                        help="Unfreeze full backbone (fine-tune).")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    train_loader, val_loader, test_loader, class_names = make_loaders(
        args.data_dir, args.img_size, args.batch_size
    )
    pos_weight = compute_pos_weight(train_loader)

    all_metrics: list[Metrics] = []
    histories: dict[str, dict] = {}

    for act_name, act_cls in ACTIVATION_MAP.items():
        print(f"\n{'='*55}")
        print(f"  MobileNetV2 + {act_name}  (device={DEVICE})")
        print(f"{'='*55}")
        metrics = train_activation(
            act_name, act_cls,
            train_loader, val_loader, test_loader, class_names,
            pos_weight, output_dir, args.epochs, args.unfreeze, args.seed,
        )
        all_metrics.append(metrics)
        histories[act_name] = pd.read_csv(
            output_dir / act_name / "training_history.csv"
        ).to_dict("list")

    # Convergence plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for act_name, h in histories.items():
        axes[0].plot(h["train_loss"], label=f"{act_name} train")
        axes[0].plot(h["val_loss"], linestyle="--", label=f"{act_name} val")
        axes[1].plot(h["val_acc"], label=act_name)

    axes[0].set_title("Transfer-Learning Loss (MobileNetV2)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(True)
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    fig.savefig(output_dir / "mobilenet_convergence.png", dpi=150)
    plt.close(fig)

    summary = pd.DataFrame(
        [m.to_dict() for m in all_metrics]
    ).sort_values("f1", ascending=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    print("\n=== MobileNetV2 Activation Sweep Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
