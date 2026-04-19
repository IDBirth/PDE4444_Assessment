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
    p = runs_dir / "iter1_mlp_search" / "metrics.json"
    m = load_metrics_json(p)
    if m:
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
    p = runs_dir / "iter1_yolo" / "metrics.json"
    m = load_metrics_json(p)
    if m:
        rows.append({
            "model_name": "yolo26n_cls",
            "accuracy":   m.get("top1", None),
            "precision":  None,
            "recall":     None,
            "f1":         m.get("top1", None),
            "category":   "YOLO (pretrained)",
        })

    df = pd.DataFrame(rows)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["f1"]       = pd.to_numeric(df.get("f1", None), errors="coerce")
    return df.sort_values("f1", ascending=False, na_position="last")


def plot_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Horizontal bar chart — F1 score per model, coloured by category."""
    plot_df = df.dropna(subset=["f1"]).copy()
    plot_df = plot_df.sort_values("f1")

    categories = plot_df["category"].unique()
    cmap = plt.cm.get_cmap("tab10", len(categories))
    cat_color = {c: cmap(i) for i, c in enumerate(categories)}

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

    best = df.dropna(subset=["f1"]).iloc[0]
    print(f"\n>>> BEST MODEL: {best['model_name']}  "
          f"(F1={best['f1']:.4f}  Acc={best['accuracy']:.4f})")


if __name__ == "__main__":
    main()
