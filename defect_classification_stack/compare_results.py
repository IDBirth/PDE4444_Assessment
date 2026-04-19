from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine metrics from all model families into one comparison CSV.")
    parser.add_argument("--sklearn-dir", required=True, type=Path)
    parser.add_argument("--keras-dir", required=True, type=Path)
    parser.add_argument("--cnn-dir", required=True, type=Path)
    parser.add_argument("--yolo-dir", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict] = []

    sklearn_summary = args.sklearn_dir / "summary.csv"
    if sklearn_summary.exists():
        rows.extend(pd.read_csv(sklearn_summary).to_dict(orient="records"))

    keras_metrics = args.keras_dir / "metrics.json"
    if keras_metrics.exists():
        rows.append(load_json(keras_metrics))

    cnn_summary = args.cnn_dir / "summary.csv"
    if cnn_summary.exists():
        rows.extend(pd.read_csv(cnn_summary).to_dict(orient="records"))

    yolo_metrics = args.yolo_dir / "metrics.json"
    if yolo_metrics.exists():
        yolo_row = load_json(yolo_metrics)
        rows.append(
            {
                "model_name": yolo_row.get("model_name", "yolo26n-cls.pt"),
                "accuracy": yolo_row.get("top1", None),
                "precision": None,
                "recall": None,
                "f1": None,
                "top5": yolo_row.get("top5", None),
                "fitness": yolo_row.get("fitness", None),
            }
        )

    if not rows:
        raise ValueError("No metric files were found.")

    df = pd.DataFrame(rows)
    sort_column = "f1" if "f1" in df.columns and df["f1"].notna().any() else "accuracy"
    df = df.sort_values(sort_column, ascending=False, na_position="last")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_file, index=False)
    print(df)


if __name__ == "__main__":
    main()
