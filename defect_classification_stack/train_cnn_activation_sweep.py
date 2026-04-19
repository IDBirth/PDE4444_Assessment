"""
CNN Activation Function Sweep  (PyTorch)
=========================================
Assessment Section 3: Neural Network Design & Optimisation
- Compares activation functions: ReLU, ELU, GELU, SELU, LeakyReLU
- CNN with >= 3 hidden (convolutional) layers + dense head
- Saves per-run metrics, confusion matrices, training history, and
  a single combined convergence plot.
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
from torchvision import datasets, transforms

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

# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------
ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "relu":       nn.ReLU,
    "elu":        nn.ELU,
    "gelu":       nn.GELU,
    "selu":       nn.SELU,
    "leaky_relu": nn.LeakyReLU,
}


# ---------------------------------------------------------------------------
# Model definition  – 3 conv blocks (hidden layers) + dense classifier
# ---------------------------------------------------------------------------
class DefectCNN(nn.Module):
    """Fully convolutional backbone with >=3 hidden layers."""

    def __init__(self, activation_cls: type[nn.Module]) -> None:
        super().__init__()
        # --- Hidden layer 1 ---
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_cls(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        # --- Hidden layer 2 ---
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            activation_cls(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        # --- Hidden layer 3 ---
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            activation_cls(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        # Dense head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128),
            activation_cls(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x).squeeze(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def make_loaders(
    data_dir: Path,
    img_size: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    train_ds = datasets.ImageFolder(
        str(data_dir / "train"), transform=train_tf
    )
    val_ds = datasets.ImageFolder(
        str(data_dir / "val"), transform=eval_tf
    )
    test_ds = datasets.ImageFolder(
        str(data_dir / "test"), transform=eval_tf
    )

    kw = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, **kw
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **kw
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **kw
    )

    return train_loader, val_loader, test_loader, train_ds.classes


def compute_pos_weight(loader: DataLoader) -> torch.Tensor:
    labels = torch.cat([y for _, y in loader])
    counts = torch.bincount(labels).float()
    # pos_weight = n_negative / n_positive  (standard BCEWithLogitsLoss form)
    return (counts[0] / counts[1]).unsqueeze(0)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    training: bool,
) -> tuple[float, float]:
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(device), y.to(device).float()
            logits = model(X)
            loss = criterion(logits, y)
            if training and optimizer is not None:
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
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    for X, y in loader:
        logits = model(X.to(device))
        preds = (torch.sigmoid(logits) >= 0.5).cpu().long()
        all_preds.append(preds)
        all_labels.append(y)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CNN activation-function sweep (PyTorch)."
    )
    parser.add_argument("--data-dir",   required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size",   type=int, default=224)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    train_loader, val_loader, test_loader, class_names = make_loaders(
        args.data_dir, args.img_size, args.batch_size
    )
    pos_weight = compute_pos_weight(train_loader).to(DEVICE)

    all_metrics: list[Metrics] = []
    histories: dict[str, dict] = {}

    for act_name, act_cls in ACTIVATION_MAP.items():
        print(f"\n=== Activation: {act_name} ===")
        run_dir = ensure_dir(output_dir / act_name)

        model = DefectCNN(act_cls).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5
        )

        best_val_loss = float("inf")
        best_state: dict = {}
        patience_counter = 0
        patience = 4

        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [],  "val_acc": [],
        }

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_epoch(
                model, train_loader, criterion, optimizer, DEVICE, True
            )
            vl_loss, vl_acc = run_epoch(
                model, val_loader, criterion, None, DEVICE, False
            )
            scheduler.step(vl_loss)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(vl_acc)

            print(
                f"  Epoch {epoch:02d}  "
                f"train_loss={tr_loss:.4f}  "
                f"val_loss={vl_loss:.4f}  "
                f"val_acc={vl_acc:.4f}"
            )

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("  Early stopping triggered.")
                    break

        model.load_state_dict(best_state)
        histories[act_name] = history

        y_pred, y_true = predict_all(model, test_loader, DEVICE)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
        accuracy = float((y_pred == y_true).mean())

        metrics = Metrics(
            model_name=f"cnn_{act_name}",
            accuracy=accuracy,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
        )
        all_metrics.append(metrics)

        save_metrics(metrics, run_dir / "metrics.json")
        save_confusion_matrix_plot(
            y_true, y_pred, class_names,
            run_dir / "confusion_matrix.png",
            f"CNN {act_name} – Confusion Matrix",
        )
        save_classification_report(
            y_true, y_pred, class_names,
            run_dir / "classification_report.csv",
        )
        pd.DataFrame(history).to_csv(
            run_dir / "training_history.csv", index=False
        )
        torch.save(model.state_dict(), run_dir / "model.pt")

    # ---- Combined convergence plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for act_name, h in histories.items():
        axes[0].plot(h["train_loss"], label=f"{act_name} train")
        axes[0].plot(h["val_loss"], linestyle="--", label=f"{act_name} val")
        axes[1].plot(h["val_acc"], label=act_name)

    axes[0].set_title("Loss Convergence by Activation Function")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(True)

    axes[1].set_title("Validation Accuracy by Activation Function")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    fig.savefig(output_dir / "activation_convergence.png", dpi=150)
    plt.close(fig)

    summary = pd.DataFrame(
        [m.to_dict() for m in all_metrics]
    ).sort_values("f1", ascending=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print("\n=== Summary ===")
    print(summary)


if __name__ == "__main__":
    main()
