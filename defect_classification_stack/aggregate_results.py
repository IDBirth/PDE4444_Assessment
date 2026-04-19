"""
Aggregate all run results into one master comparison table + charts.
Usage:  python aggregate_results.py --runs-dir runs/ --output-dir runs/final_report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_metrics_json(path: Path) -> dict | None:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def collect_all(runs_dir: Path) -> pd.DataFrame:
    rows = []

    # --- sklearn baseline ---
    for sub in ["hog_svm", "hog_mlp"]:
        p = runs_dir / "iter1_sklearn" / sub / "metrics.json"
        m = load_metrics_json(p)
        if m:
            m["category"] = "Sklearn Baseline"
            rows.append(m)

    # --- CNN scratch activation sweep ---
    cnn_dir = runs_dir / "iter1_cnn_activation"
    for act in ["relu", "elu", "gelu", "selu", "leaky_relu"]:
        p = cnn_dir / act / "metrics.json"
        m = load_metrics_json(p)
        if m:
            m["category"] = "CNN Scratch"
            rows.append(m)

    # --- Optimizer comparison ---
    for opt in ["nelder_mead", "sgd", "adam", "lbfgs"]:
        p = runs_dir / "iter1_optimizer" / opt / "metrics.json"
        m = load_metrics_json(p)
        if m:
            m["model_name"] = f"optim_{opt}"
            m["category"] = "Optimiser Comparison"
            rows.append(m)

    # --- MLP hparam search ---
    # Match any iter*mlp* directory so variants like iter1_mlp_search and
    # iter1_mlp_boost are both picked up.
    for mlp_dir in sorted(runs_dir.glob("iter*mlp*")):
        p = mlp_dir / "metrics.json"
        m = load_metrics_json(p)
        if m:
            m.setdefault("model_name", mlp_dir.name)
            m["category"] = "MLP HParam Search"
            rows.append(m)

    # --- MobileNetV2 frozen ---
    mob_dir = runs_dir / "iter2_mobilenet"
    for act in ["relu", "elu", "gelu", "selu", "leaky_relu"]:
        p = mob_dir / act / "metrics.json"
        m = load_metrics_json(p)
        if m:
            m["category"] = "MobileNetV2 (frozen)"
            rows.append(m)

    # --- MobileNetV2 fine-tuned ---
    mob2_dir = runs_dir / "iter3_mobilenet_finetune"
    for act in ["relu", "elu", "gelu", "selu", "leaky_relu"]:
        p = mob2_dir / act / "metrics.json"
        m = load_metrics_json(p)
        if m:
            m["category"] = "MobileNetV2 (fine-tuned)"
            rows.append(m)

    # --- YOLO ---
    # YOLO reports top-1 classification accuracy, not binary F1.
    # top-1 is stored in the accuracy column only. F1/precision/recall are
    # left as None because YOLO does not report per-class binary metrics.
    # Match any iter*yolo* directory so 26n, 26s, and _320 variants are picked up.
    for yolo_dir in sorted(runs_dir.glob("iter*yolo*")):
        p = yolo_dir / "metrics.json"
        m = load_metrics_json(p)
        if not m:
            continue
        model_name = m.get("model_name") or yolo_dir.name
        rows.append({
            "model_name": model_name,
            "accuracy":   m.get("top1", None),
            "top1_acc":   m.get("top1", None),
            "fitness":    m.get("fitness", None),
            "precision":  None,
            "recall":     None,
            "f1":         None,   # top-1 != F1; left blank to avoid misleading ranking
            "category":   "YOLO (pretrained)",
        })

    df = pd.DataFrame(rows)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["f1"]       = pd.to_numeric(df.get("f1", None), errors="coerce")
    # Sort by F1 for models that have it; YOLO sorts by accuracy separately
    return df.sort_values("f1", ascending=False, na_position="last")


def plot_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Horizontal bar chart — F1 score per model, coloured by category."""
    plot_df = df.dropna(subset=["f1"]).copy()
    plot_df = plot_df.sort_values("f1")

    # Build the colour map from all categories (including YOLO rows that
    # have no F1), so the accuracy chart can look any of them up later.
    all_categories = df["category"].unique()
    categories     = plot_df["category"].unique()
    cmap = plt.cm.get_cmap("tab10", max(len(all_categories), 1))
    cat_color = {c: cmap(i) for i, c in enumerate(all_categories)}

    fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df) * 0.45)))
    bars = ax.barh(
        plot_df["model_name"],
        plot_df["f1"],
        color=[cat_color[c] for c in plot_df["category"]],
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1 Score (test set)")
    ax.set_title("Model Comparison – F1 Score")
    ax.grid(axis="x", alpha=0.4)

    from matplotlib.patches import Patch
    handles = [Patch(color=cat_color[c], label=c) for c in categories]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "model_comparison_f1.png", dpi=150)
    plt.close(fig)

    # Accuracy chart
    acc_df = df.dropna(subset=["accuracy"]).sort_values("accuracy")
    fig2, ax2 = plt.subplots(figsize=(11, max(6, len(acc_df) * 0.45)))
    bars2 = ax2.barh(
        acc_df["model_name"],
        acc_df["accuracy"],
        color=[cat_color[c] for c in acc_df["category"]],
    )
    ax2.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    ax2.set_xlim(0, 1.12)
    ax2.set_xlabel("Accuracy (test set)")
    ax2.set_title("Model Comparison – Test Accuracy")
    ax2.grid(axis="x", alpha=0.4)
    ax2.legend(handles=handles, loc="lower right", fontsize=8)
    plt.tight_layout()
    fig2.savefig(output_dir / "model_comparison_acc.png", dpi=150)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir",   required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = collect_all(args.runs_dir)

    out_csv = args.output_dir / "all_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    plot_comparison(df, args.output_dir)
    print(f"Saved charts to: {args.output_dir}")

    print("\n" + "="*70)
    print("  FULL MODEL COMPARISON  (sorted by F1 desc)")
    print("="*70)
    print(df[["model_name", "category", "accuracy", "precision",
              "recall", "f1"]].to_string(index=False))

    best_f1 = df.dropna(subset=["f1"]).iloc[0]
    best_acc = df.dropna(subset=["accuracy"]).sort_values("accuracy", ascending=False).iloc[0]
    print(f"\n>>> BEST by F1:  {best_f1['model_name']}  (F1={best_f1['f1']:.4f}  Acc={best_f1['accuracy']:.4f})")
    print(f">>> BEST by Acc: {best_acc['model_name']}  (Acc={best_acc['accuracy']:.4f})"
          + (f"  [top-1 accuracy, no F1]" if pd.isna(best_acc.get("f1", np.nan)) else ""))


if __name__ == "__main__":
    main()
