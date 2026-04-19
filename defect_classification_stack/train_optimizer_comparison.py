"""
Optimiser Comparison  (Assessment Section 3)
=============================================
Compares three classes of optimisation methods on the *same* small MLP
trained on HOG + PCA features (derived from the image dataset).

  Zero-order   – scipy Nelder-Mead (derivative-free, no gradients used)
  First-order  – SGD  (mini-batch gradient descent)
                 Adam (adaptive moment estimation)
  Second-order – L-BFGS (limited-memory quasi-Newton, via PyTorch)

All four optimisers are run on identical architecture and data so the
convergence plots are directly comparable.

Architecture
------------
  Input (n_pca=20) -> Dense(32) -> ReLU -> Dense(16) -> ReLU -> Dense(1)
  ~700 parameters – small enough for Nelder-Mead to converge.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import (
    Metrics,
    ensure_dir,
    infer_class_names,
    load_split_as_arrays,
    save_classification_report,
    save_confusion_matrix_plot,
    save_metrics,
    set_seed,
)

# ---------------------------------------------------------------------------
# HOG feature extraction (same as sklearn baseline)
# ---------------------------------------------------------------------------
HOG_SIZE = 128
N_PCA = 20       # reduce to keep Nelder-Mead tractable


def extract_hog(images: np.ndarray) -> np.ndarray:
    feats = []
    for img in images:
        u8 = (img * 255).astype(np.uint8)
        resized = cv2.resize(u8, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
        feat = hog(
            resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        feats.append(feat)
    return np.array(feats)


# ---------------------------------------------------------------------------
# Small MLP (shared architecture across all optimisers)
# ---------------------------------------------------------------------------
class SmallMLP(nn.Module):
    def __init__(self, n_input: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Helpers: flat weight <-> model parameter conversion  (for Nelder-Mead)
# ---------------------------------------------------------------------------
def get_flat_weights(model: nn.Module) -> np.ndarray:
    return np.concatenate(
        [p.detach().cpu().numpy().ravel() for p in model.parameters()]
    )


def set_flat_weights(model: nn.Module, flat: np.ndarray) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(
                torch.tensor(
                    flat[offset: offset + n],
                    dtype=p.dtype,
                ).reshape(p.shape)
            )
            offset += n


def bce_loss_np(
    flat: np.ndarray,
    model: nn.Module,
    X_t: torch.Tensor,
    y_t: torch.Tensor,
) -> float:
    set_flat_weights(model, flat)
    with torch.no_grad():
        logits = model(X_t)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y_t
        )
    return float(loss.item())


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------
def train_gradient(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    label: str,
) -> dict[str, list[float]]:
    """Mini-batch training for SGD / Adam (first-order)."""
    n = len(X_train)
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": []
    }
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = idx[start: start + batch_size]
            xb, yb = X_train[batch_idx], y_train[batch_idx]

            def closure():
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            epoch_loss += float(loss) * len(batch_idx)

        tr_loss = epoch_loss / n

        model.eval()
        with torch.no_grad():
            vl_loss = float(
                criterion(model(X_val), y_val).item()
            )

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        print(
            f"  [{label}] Epoch {epoch:03d}  "
            f"train={tr_loss:.4f}  val={vl_loss:.4f}"
        )

    return history


def train_lbfgs(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
) -> dict[str, list[float]]:
    """Full-batch L-BFGS (second-order quasi-Newton)."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=20,
        history_size=10,
        line_search_fn="strong_wolfe",
    )
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": []
    }

    for epoch in range(1, epochs + 1):
        model.train()

        def closure():
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            return loss

        optimizer.step(closure)

        model.eval()
        with torch.no_grad():
            tr_loss = float(criterion(model(X_train), y_train).item())
            vl_loss = float(criterion(model(X_val), y_val).item())

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        print(
            f"  [L-BFGS] Epoch {epoch:03d}  "
            f"train={tr_loss:.4f}  val={vl_loss:.4f}"
        )

    return history


def train_nelder_mead(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    max_iter: int,
) -> dict[str, list[float]]:
    """Zero-order optimisation via scipy Nelder-Mead (no gradients)."""
    criterion = nn.BCEWithLogitsLoss()
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": []
    }
    x0 = get_flat_weights(model)

    def callback(xk: np.ndarray) -> None:
        set_flat_weights(model, xk)
        with torch.no_grad():
            tr = float(criterion(model(X_train), y_train).item())
            vl = float(criterion(model(X_val), y_val).item())
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        if len(history["train_loss"]) % 50 == 0:
            print(
                f"  [Nelder-Mead] iter {len(history['train_loss']):4d}  "
                f"train={tr:.4f}  val={vl:.4f}"
            )

    minimize(
        fun=bce_loss_np,
        x0=x0,
        args=(model, X_train, y_train),
        method="Nelder-Mead",
        callback=callback,
        options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-4},
    )
    return history


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    model_name: str,
) -> Metrics:
    model.eval()
    probs = torch.sigmoid(model(X))
    preds = (probs >= 0.5).long().numpy()
    y_np = y.long().numpy()
    tp = int(((preds == 1) & (y_np == 1)).sum())
    fp = int(((preds == 1) & (y_np == 0)).sum())
    fn = int(((preds == 0) & (y_np == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    accuracy = float((preds == y_np).mean())
    return Metrics(
        model_name=model_name,
        accuracy=accuracy,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimiser comparison: zero / first / second order."
    )
    parser.add_argument("--data-dir",   required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs",     type=int, default=50,
                        help="Epochs for SGD/Adam/L-BFGS.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nm-iters",   type=int, default=800,
                        help="Max Nelder-Mead iterations.")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    # ---- Load & preprocess data ----
    print("Loading images and extracting HOG features...")
    X_tr_img, y_tr, class_names = load_split_as_arrays(
        args.data_dir, "train", size=(224, 224), grayscale=True
    )
    X_vl_img, y_vl, _ = load_split_as_arrays(
        args.data_dir, "val", size=(224, 224), grayscale=True
    )
    X_te_img, y_te, _ = load_split_as_arrays(
        args.data_dir, "test", size=(224, 224), grayscale=True
    )

    X_tr_hog = extract_hog(X_tr_img)
    X_vl_hog = extract_hog(X_vl_img)
    X_te_hog = extract_hog(X_te_img)

    # Scale -> PCA
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_hog)
    X_vl_s = scaler.transform(X_vl_hog)
    X_te_s = scaler.transform(X_te_hog)

    pca = PCA(n_components=N_PCA, random_state=args.seed)
    X_tr_p = pca.fit_transform(X_tr_s)
    X_vl_p = pca.transform(X_vl_s)
    X_te_p = pca.transform(X_te_s)

    X_tr_t = torch.tensor(X_tr_p, dtype=torch.float32)
    X_vl_t = torch.tensor(X_vl_p, dtype=torch.float32)
    X_te_t = torch.tensor(X_te_p, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    y_vl_t = torch.tensor(y_vl, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)

    print(
        f"HOG dims: {X_tr_hog.shape[1]}  "
        f"-> PCA dims: {N_PCA}  "
        f"(MLP params: ~{count_params(SmallMLP(N_PCA))})"
    )

    optimisers: dict[str, tuple[str, dict]] = {
        "nelder_mead":  ("zero-order  | Nelder-Mead (derivative-free)", {}),
        "sgd":          ("first-order | SGD (mini-batch gradient descent)", {}),
        "adam":         ("first-order | Adam (adaptive moments)", {}),
        "lbfgs":        ("second-order| L-BFGS (quasi-Newton)", {}),
    }

    all_metrics: list[Metrics] = []
    all_histories: dict[str, dict] = {}

    for opt_name, (desc, _) in optimisers.items():
        print(f"\n{'='*60}")
        print(f"  Optimiser: {opt_name}  ({desc})")
        print(f"{'='*60}")
        run_dir = ensure_dir(output_dir / opt_name)

        torch.manual_seed(args.seed)
        model = SmallMLP(N_PCA)

        if opt_name == "nelder_mead":
            history = train_nelder_mead(
                model, X_tr_t, y_tr_t, X_vl_t, y_vl_t,
                max_iter=args.nm_iters,
            )
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.01, momentum=0.9
            )
            history = train_gradient(
                model, X_tr_t, y_tr_t, X_vl_t, y_vl_t,
                optimizer, args.epochs, args.batch_size, "SGD",
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=1e-3
            )
            history = train_gradient(
                model, X_tr_t, y_tr_t, X_vl_t, y_vl_t,
                optimizer, args.epochs, args.batch_size, "Adam",
            )
        else:  # lbfgs
            history = train_lbfgs(
                model, X_tr_t, y_tr_t, X_vl_t, y_vl_t,
                epochs=args.epochs,
            )

        metrics = evaluate(model, X_te_t, y_te_t, model_name=opt_name)
        all_metrics.append(metrics)
        all_histories[opt_name] = history

        save_metrics(metrics, run_dir / "metrics.json")
        save_confusion_matrix_plot(
            y_te, (torch.sigmoid(model(X_te_t)) >= 0.5).long().numpy(),
            class_names,
            run_dir / "confusion_matrix.png",
            f"{opt_name} – Confusion Matrix",
        )
        save_classification_report(
            y_te,
            (torch.sigmoid(model(X_te_t)) >= 0.5).long().numpy(),
            class_names,
            run_dir / "classification_report.csv",
        )
        pd.DataFrame(history).to_csv(
            run_dir / "training_history.csv", index=False
        )
        torch.save(model.state_dict(), run_dir / "model.pt")

        with (run_dir / "description.txt").open("w") as f:
            f.write(f"Optimiser: {opt_name}\n")
            f.write(f"Description: {desc}\n")
            f.write(f"MLP params: {count_params(model)}\n")
            f.write(f"PCA components: {N_PCA}\n")

    # ---- Convergence plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = {
        "nelder_mead": "Nelder-Mead (zero-order)",
        "sgd":         "SGD (first-order)",
        "adam":        "Adam (first-order)",
        "lbfgs":       "L-BFGS (second-order)",
    }
    for opt_name, h in all_histories.items():
        iters = range(1, len(h["train_loss"]) + 1)
        lbl = labels[opt_name]
        axes[0].plot(iters, h["train_loss"], label=f"{lbl} train")
        axes[0].plot(
            iters, h["val_loss"], linestyle="--", label=f"{lbl} val"
        )
        axes[1].plot(iters, h["val_loss"], label=lbl)

    axes[0].set_title("Training Loss Convergence by Optimiser")
    axes[0].set_xlabel("Iteration / Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(True)

    axes[1].set_title("Validation Loss Convergence by Optimiser")
    axes[1].set_xlabel("Iteration / Epoch")
    axes[1].set_ylabel("BCE Loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    fig.savefig(output_dir / "optimizer_convergence.png", dpi=150)
    plt.close(fig)

    # ---- Bar chart: test metrics ----
    df = pd.DataFrame([m.to_dict() for m in all_metrics])
    df.to_csv(output_dir / "summary.csv", index=False)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    width = 0.2
    for i, col in enumerate(["accuracy", "precision", "recall", "f1"]):
        ax2.bar(x + i * width, df[col], width, label=col.capitalize())
    ax2.set_xticks(x + 1.5 * width)
    ax2.set_xticklabels(df["model_name"], rotation=20, ha="right")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Test Performance by Optimiser")
    ax2.legend()
    ax2.grid(axis="y")
    plt.tight_layout()
    fig2.savefig(output_dir / "optimizer_metrics_bar.png", dpi=150)
    plt.close(fig2)

    print("\n=== Optimiser Summary ===")
    print(df.sort_values("f1", ascending=False).to_string(index=False))

    # Save PCA explained variance info
    with (output_dir / "pca_info.json").open("w") as f:
        json.dump({
            "n_components": N_PCA,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "total_variance_explained": float(
                pca.explained_variance_ratio_.sum()
            ),
            "hog_feature_dim": int(X_tr_hog.shape[1]),
        }, f, indent=2)


if __name__ == "__main__":
    main()
