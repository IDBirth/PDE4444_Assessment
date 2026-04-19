"""
Cross-Validation  (Assessment Section 5: Experimental Rigor)
=============================================================
Applies Stratified K-Fold cross-validation on the combined
train+val data using HOG features with SVM and sklearn MLP.

Also produces:
  - per-fold metrics table
  - mean +/- std summary
  - overfitting analysis (train vs val loss per fold)
  - learning-curve plot (training size vs CV accuracy)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    learning_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (
    ensure_dir,
    load_split_as_arrays,
    save_classification_report,
    save_confusion_matrix_plot,
    set_seed,
)

HOG_SIZE = 128
N_FOLDS = 5


def extract_hog(images: np.ndarray) -> np.ndarray:
    feats = []
    for img in images:
        u8 = (img * 255).astype(np.uint8)
        resized = cv2.resize(
            u8, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA
        )
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


def build_svm() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            C=1.0, kernel="rbf", gamma="scale",
            probability=True, class_weight="balanced",
        )),
    ])


def build_mlp(seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )),
    ])


def run_cv(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    output_dir: Path,
    seed: int,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    scoring = ["accuracy", "precision", "recall", "f1"]
    results = cross_validate(
        estimator, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    records = []
    for fold in range(N_FOLDS):
        records.append({
            "model":          name,
            "fold":           fold + 1,
            "train_accuracy": results["train_accuracy"][fold],
            "val_accuracy":   results["test_accuracy"][fold],
            "val_precision":  results["test_precision"][fold],
            "val_recall":     results["test_recall"][fold],
            "val_f1":         results["test_f1"][fold],
            "overfit_gap":    (
                results["train_accuracy"][fold]
                - results["test_accuracy"][fold]
            ),
        })

    df = pd.DataFrame(records)

    # Summary row
    means = df[[c for c in df.columns if c not in ("model", "fold")]].mean()
    stds  = df[[c for c in df.columns if c not in ("model", "fold")]].std()
    summary_rows = [
        {"model": name, "fold": "mean"} | means.to_dict(),
        {"model": name, "fold": "std"}  | stds.to_dict(),
    ]
    summary_df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
    run_dir = ensure_dir(output_dir / name)
    summary_df.to_csv(run_dir / "cv_results.csv", index=False)

    print(f"\n[{name}] {N_FOLDS}-Fold CV Results:")
    print(df[["fold", "train_accuracy", "val_accuracy",
              "val_f1", "overfit_gap"]].to_string(index=False))
    print(f"  Mean val accuracy: {means['val_accuracy']:.4f} "
          f"+/- {stds['val_accuracy']:.4f}")
    print(f"  Mean val F1:       {means['val_f1']:.4f} "
          f"+/- {stds['val_f1']:.4f}")
    print(f"  Mean overfit gap:  {means['overfit_gap']:.4f}")

    return df


def plot_overfitting(
    df_svm: pd.DataFrame,
    df_mlp: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Bar chart: train vs val accuracy per fold (overfitting analysis)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, df, name in zip(axes, [df_svm, df_mlp], ["SVM", "MLP"]):
        x = np.arange(N_FOLDS)
        width = 0.35
        ax.bar(x - width / 2, df["train_accuracy"], width, label="Train")
        ax.bar(x + width / 2, df["val_accuracy"],   width, label="Val")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {i+1}" for i in range(N_FOLDS)])
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{name} – Train vs Val Accuracy (Overfitting Analysis)")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(axis="y")
    plt.tight_layout()
    fig.savefig(output_dir / "overfitting_analysis.png", dpi=150)
    plt.close(fig)


def plot_learning_curve(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    output_dir: Path,
    seed: int,
) -> None:
    """Learning curve: effect of training-set size on generalisation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1,
    )
    tr_mean = train_scores.mean(axis=1)
    tr_std  = train_scores.std(axis=1)
    te_mean = test_scores.mean(axis=1)
    te_std  = test_scores.std(axis=1)

    run_dir = ensure_dir(output_dir / name)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15)
    ax.fill_between(train_sizes, te_mean - te_std, te_mean + te_std, alpha=0.15)
    ax.plot(train_sizes, tr_mean, "o-", label="Train F1")
    ax.plot(train_sizes, te_mean, "s--", label="Val F1")
    ax.set_title(f"{name} – Learning Curve (F1 Score)")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("F1 Score")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(run_dir / "learning_curve.png", dpi=150)
    plt.close(fig)


def final_test_eval(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    name: str,
    output_dir: Path,
) -> None:
    """Retrain on full train set, evaluate on held-out test set."""
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    run_dir = ensure_dir(output_dir / name)
    save_confusion_matrix_plot(
        y_test, y_pred, class_names,
        run_dir / "confusion_matrix_test.png",
        f"{name} – Test Confusion Matrix",
    )
    save_classification_report(
        y_test, y_pred, class_names,
        run_dir / "classification_report_test.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified K-Fold cross-validation with overfitting analysis."
    )
    parser.add_argument("--data-dir",   required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

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

    # Pool train + val for CV (test remains completely unseen)
    X_cv = np.concatenate([X_tr_hog, X_vl_hog], axis=0)
    y_cv = np.concatenate([y_tr, y_vl], axis=0)

    print(
        f"CV pool: {len(X_cv)} samples  |  "
        f"Test: {len(X_te_hog)} samples  |  "
        f"HOG dims: {X_tr_hog.shape[1]}"
    )

    svm = build_svm()
    mlp = build_mlp(args.seed)

    df_svm = run_cv(svm, X_cv, y_cv, "hog_svm_cv", output_dir, args.seed)
    df_mlp = run_cv(mlp, X_cv, y_cv, "hog_mlp_cv", output_dir, args.seed)

    plot_overfitting(df_svm, df_mlp, output_dir)

    print("\nGenerating learning curves (this may take a moment)...")
    plot_learning_curve(
        build_svm(), X_cv, y_cv, "hog_svm_cv", output_dir, args.seed
    )
    plot_learning_curve(
        build_mlp(args.seed), X_cv, y_cv, "hog_mlp_cv", output_dir, args.seed
    )

    # Final hold-out test evaluation
    final_test_eval(
        build_svm(), X_cv, y_cv, X_te_hog, y_te,
        class_names, "hog_svm_cv", output_dir
    )
    final_test_eval(
        build_mlp(args.seed), X_cv, y_cv, X_te_hog, y_te,
        class_names, "hog_mlp_cv", output_dir
    )

    # Combined summary
    combined = pd.concat([df_svm, df_mlp], ignore_index=True)
    combined.to_csv(output_dir / "cv_combined.csv", index=False)
    print("\n=== Cross-Validation Complete ===")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
