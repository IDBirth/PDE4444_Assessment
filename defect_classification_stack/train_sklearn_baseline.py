from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


def extract_hog_features(images: np.ndarray) -> np.ndarray:
    features = []
    for image in images:
        image_u8 = (image * 255).astype(np.uint8)
        resized = cv2.resize(image_u8, (128, 128), interpolation=cv2.INTER_AREA)
        feat = hog(
            resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        features.append(feat)
    return np.array(features)


def train_with_predefined_split(
    estimator,
    param_distributions: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_iter: int,
    seed: int,
):
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    fold = np.array([-1] * len(X_train) + [0] * len(X_val))
    ps = PredefinedSplit(fold)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=ps,
        verbose=1,
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_combined, y_combined)
    return search


def evaluate_and_save(name: str, search, X_test, y_test, class_names, output_dir: Path) -> Metrics:
    y_pred = search.predict(X_test)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics = Metrics(
        model_name=name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
    )

    model_dir = ensure_dir(output_dir / name)
    save_metrics(metrics, model_dir / "metrics.json")
    save_confusion_matrix_plot(y_test, y_pred, class_names, model_dir / "confusion_matrix.png", f"{name} Confusion Matrix")
    save_classification_report(y_test, y_pred, class_names, model_dir / "classification_report.csv")

    with (model_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(search.best_params_, f, indent=2)

    pd.DataFrame(search.cv_results_).to_csv(model_dir / "search_results.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train scikit-learn baselines on the prepared dataset.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=12)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    X_train, y_train, class_names = load_split_as_arrays(args.data_dir, "train", size=(224, 224), grayscale=True)
    X_val, y_val, _ = load_split_as_arrays(args.data_dir, "val", size=(224, 224), grayscale=True)
    X_test, y_test, _ = load_split_as_arrays(args.data_dir, "test", size=(224, 224), grayscale=True)

    X_train_hog = extract_hog_features(X_train)
    X_val_hog = extract_hog_features(X_val)
    X_test_hog = extract_hog_features(X_test)

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, class_weight="balanced")),
    ])
    svm_params = {
        "clf__C": [0.1, 1.0, 5.0, 10.0, 25.0],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", "auto"],
    }
    svm_search = train_with_predefined_split(
        estimator=svm_pipeline,
        param_distributions=svm_params,
        X_train=X_train_hog,
        y_train=y_train,
        X_val=X_val_hog,
        y_val=y_val,
        n_iter=args.n_iter,
        seed=args.seed,
    )

    mlp_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            MLPClassifier(
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=args.seed,
            ),
        ),
    ])
    mlp_params = {
        "clf__hidden_layer_sizes": [(128,), (256,), (128, 64), (256, 128)],
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": [1e-5, 1e-4, 1e-3],
        "clf__learning_rate_init": [1e-4, 5e-4, 1e-3],
    }
    mlp_search = train_with_predefined_split(
        estimator=mlp_pipeline,
        param_distributions=mlp_params,
        X_train=X_train_hog,
        y_train=y_train,
        X_val=X_val_hog,
        y_val=y_val,
        n_iter=args.n_iter,
        seed=args.seed,
    )

    all_metrics = [
        evaluate_and_save("hog_svm", svm_search, X_test_hog, y_test, class_names, output_dir),
        evaluate_and_save("hog_mlp", mlp_search, X_test_hog, y_test, class_names, output_dir),
    ]

    pd.DataFrame([m.to_dict() for m in all_metrics]).sort_values("f1", ascending=False).to_csv(
        output_dir / "summary.csv", index=False
    )
    print(pd.DataFrame([m.to_dict() for m in all_metrics]).sort_values("f1", ascending=False))


if __name__ == "__main__":
    main()
