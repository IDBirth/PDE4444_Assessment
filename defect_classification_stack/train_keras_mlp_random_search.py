"""
PyTorch MLP Hyperparameter Search  (replaces Keras-Tuner version)
==================================================================
Performs a random search over MLP hyperparameters using PyTorch.
Covers Assessment Section 3 requirement for a deep fully-connected network
with hyperparameter tuning, without requiring TensorFlow/Keras.

Search space
------------
  hidden_sizes : [(128,), (256,), (128,64), (256,128), (128,64,32)]
  activation   : relu, elu, gelu, selu
  dropout      : 0.0, 0.2, 0.4
  lr           : 1e-4, 3e-4, 1e-3
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

ACTIVATION_FNS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "elu":  nn.ELU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
}

SEARCH_SPACE = {
    "hidden_sizes": [
        (128,),
        (256,),
        (128, 64),
        (256, 128),
        (128, 64, 32),
    ],
    "activation": list(ACTIVATION_FNS.keys()),
    "dropout":    [0.0, 0.2, 0.4],
    "lr":         [1e-4, 3e-4, 1e-3],
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Fully-connected MLP with configurable depth and activation."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...],
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        act_cls = ACTIVATION_FNS[activation]
        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.BatchNorm1d(input_dim),
        ]
        in_dim = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), act_cls()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_loaders(
    data_dir: Path,
    img_size: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], int]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.ImageFolder(str(data_dir / "train"), train_tf)
    val_ds = datasets.ImageFolder(str(data_dir / "val"), eval_tf)
    test_ds = datasets.ImageFolder(str(data_dir / "test"), eval_tf)
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
    # input_dim = C * H * W (after flatten)
    sample, _ = train_ds[0]
    input_dim = sample.numel()
    return train_loader, val_loader, test_loader, train_ds.classes, input_dim


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_config(
    input_dim: int,
    hp: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    pos_weight: torch.Tensor,
    seed: int,
) -> tuple[nn.Module, dict, float]:
    torch.manual_seed(seed)
    model = MLP(
        input_dim=input_dim,
        hidden_sizes=hp["hidden_sizes"],
        activation=hp["activation"],
        dropout=hp["dropout"],
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_val_loss = float("inf")
    best_state: dict = {}
    patience_left = 5
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": []
    }

    for _ in range(epochs):
        model.train()
        tr_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE).float()
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(y)
        tr_loss /= len(train_loader.dataset)  # type: ignore[arg-type]

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE).float()
                vl_loss += criterion(model(X), y).item() * len(y)
        vl_loss /= len(val_loader.dataset)  # type: ignore[arg-type]

        scheduler.step(vl_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_left = 5
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    return model, history, best_val_loss


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for X, y in loader:
        logits = model(X.to(DEVICE))
        preds.append((torch.sigmoid(logits) >= 0.5).cpu().long())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch MLP random hyperparameter search."
    )
    parser.add_argument("--data-dir",   required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs",     type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--img-size",   type=int, default=64,
                        help="Smaller image keeps MLP tractable.")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    random.seed(args.seed)

    train_loader, val_loader, test_loader, class_names, input_dim = \
        make_loaders(args.data_dir, args.img_size, args.batch_size)

    # pos_weight for class imbalance
    all_labels = torch.cat([y for _, y in train_loader])
    counts = torch.bincount(all_labels).float()
    pos_weight = (counts[0] / counts[1]).unsqueeze(0)

    # Build candidate list
    all_combos = list(itertools.product(*SEARCH_SPACE.values()))
    keys = list(SEARCH_SPACE.keys())
    candidates = [dict(zip(keys, c)) for c in all_combos]
    random.shuffle(candidates)
    candidates = candidates[: args.max_trials]

    print(
        f"Running {len(candidates)} trials  |  "
        f"input_dim={input_dim}  device={DEVICE}"
    )

    trial_results = []
    best_val_loss = float("inf")
    best_model: nn.Module | None = None
    best_hp: dict = {}
    best_history: dict = {}

    for i, hp in enumerate(candidates, 1):
        print(f"\n--- Trial {i}/{len(candidates)}: {hp}")
        model, history, val_loss = train_one_config(
            input_dim, hp, train_loader, val_loader,
            args.epochs, pos_weight, seed=args.seed + i,
        )
        trial_results.append({**hp, "best_val_loss": val_loss})
        print(f"    best_val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_hp = hp
            best_history = history

    assert best_model is not None
    print(f"\nBest HP: {best_hp}  (val_loss={best_val_loss:.4f})")

    # ---- Evaluate on test ----
    y_pred, y_true = predict(best_model, test_loader)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    metrics = Metrics(
        model_name="mlp_random_search",
        accuracy=float((y_pred == y_true).mean()),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )

    save_metrics(metrics, output_dir / "metrics.json")
    save_confusion_matrix_plot(
        y_true, y_pred, class_names,
        output_dir / "confusion_matrix.png",
        "MLP Random Search – Confusion Matrix",
    )
    save_classification_report(
        y_true, y_pred, class_names,
        output_dir / "classification_report.csv",
    )
    torch.save(best_model.state_dict(), output_dir / "best_model.pt")
    pd.DataFrame(best_history).to_csv(
        output_dir / "training_history.csv", index=False
    )
    pd.DataFrame(trial_results).sort_values("best_val_loss").to_csv(
        output_dir / "trial_results.csv", index=False
    )
    with (output_dir / "best_hyperparameters.json").open("w") as f:
        json.dump({k: str(v) for k, v in best_hp.items()}, f, indent=2)

    # Convergence plot for best trial
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(best_history["train_loss"], label="Train loss")
    ax.plot(best_history["val_loss"],   label="Val loss", linestyle="--")
    ax.set_title("MLP Best Trial – Loss Convergence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(output_dir / "convergence.png", dpi=150)
    plt.close(fig)

    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
