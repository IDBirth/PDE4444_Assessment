#!/usr/bin/env python3
"""
End-to-end orchestration script — PDE4444 Defect Classification Pipeline.

Runs the complete workflow from raw images → balanced splits → trained models
→ aggregated report. Every artefact is written under --runs-dir and every
step is resumable (skipped if its output already exists and --force is not
set). All models are saved as PyTorch .pt checkpoints; no engine/ONNX/
TorchScript export is performed.

Usage:
    cd /home/ubu/Desktop/Assessment
    .venv/bin/python run_pipeline.py --runs-dir runs4

Steps executed:
     0. Build balanced train/val/test dataset
        (scaffold/raw → scaffold/interim/balanced → runs-dir/data_balanced)
     1. sklearn baselines (HOG+SVM, HOG+MLP)
     2. CNN activation sweep (5 activations, from scratch)
     3. Optimiser comparison (Adam / SGD / L-BFGS / Nelder-Mead)
     4. MLP hyperparameter random search (24 trials, 96px)
     5. MobileNetV2 frozen × 5 activations
     6. MobileNetV2 fine-tuned × 5 activations (patience=12, img=256)
     7. YOLO26n-cls (img=320, batch=64, epochs=50)
     8. YOLO26s-cls (img=320, batch=64, epochs=50)
     9. 5-fold cross-validation (HOG + SVM / MLP)
    10. Aggregate all results + comparison chart

Resuming: any step whose output directory exists and is non-empty is skipped.
Use --force to re-run, or --steps N M ... to run a subset.

Data dependency:
    The pipeline will auto-generate --data-dir if it does not exist, provided
    raw scaffold data is present at:
        zeroq_cup_classification_scaffold/data/raw/{defective,non_defective}/
    The balanced split is written to --data-dir (default: <runs-dir>/data_balanced).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parent
STACK_ROOT   = REPO_ROOT / "defect_classification_stack"
SCAFFOLD     = REPO_ROOT / "zeroq_cup_classification_scaffold"
VENV_PY      = REPO_ROOT / ".venv" / "bin" / "python"

RAW_DEFECTIVE      = SCAFFOLD / "data" / "raw" / "defective"
RAW_NON_DEFECTIVE  = SCAFFOLD / "data" / "raw" / "non_defective"
BALANCED_DIR       = SCAFFOLD / "data" / "interim" / "balanced"
BALANCE_SCRIPT     = SCAFFOLD / "scripts" / "balance_cleaned_dataset.py"
PREPARE_SCRIPT     = STACK_ROOT / "prepare_dataset.py"


def run(cmd: list[str], step_name: str, cwd: Path | None = None) -> bool:
    print(f"\n{'='*70}")
    print(f"  STEP: {step_name}")
    print(f"  CMD:  {' '.join(str(c) for c in cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, cwd=cwd or REPO_ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed (exit {result.returncode}).")
        return False
    return True


def skip_if_exists(out_dir: Path, force: bool, step_name: str) -> bool:
    if not force and out_dir.exists() and any(out_dir.iterdir()):
        print(f"  [SKIP] {step_name} — output already exists: {out_dir}")
        return True
    return False


def step_build_dataset(py: str, data_dir: Path, force: bool) -> bool:
    """Build <data_dir> with train/val/test of defect/non_defect class folders."""
    if skip_if_exists(data_dir, force, "build balanced dataset"):
        return True

    if not RAW_DEFECTIVE.exists() or not RAW_NON_DEFECTIVE.exists():
        print(f"[ERROR] Raw scaffold data not found at {SCAFFOLD / 'data/raw'}.")
        print("        Expected: raw/defective/ and raw/non_defective/ folders.")
        return False

    # 0a: balance raw defective/non_defective via undersampling the majority.
    balanced_defective = BALANCED_DIR / "defective"
    balanced_nondefective = BALANCED_DIR / "non_defective"
    need_balance = force or not balanced_defective.exists() or not balanced_nondefective.exists()
    if need_balance:
        if force and BALANCED_DIR.exists():
            shutil.rmtree(BALANCED_DIR)
        if not BALANCE_SCRIPT.exists():
            print(f"[ERROR] Balance script missing: {BALANCE_SCRIPT}")
            return False
        ok = run(
            [py, str(BALANCE_SCRIPT),
             "--input-root",  str(SCAFFOLD / "data" / "raw"),
             "--output-root", str(BALANCED_DIR),
             "--seed", "42"],
            "0a. balance raw dataset (undersample majority)",
        )
        if not ok:
            return False
    else:
        print(f"  [SKIP] balance step — {BALANCED_DIR} already exists.")

    # 0b: split balanced folders into train/val/test with defect/non_defect labels.
    if not PREPARE_SCRIPT.exists():
        print(f"[ERROR] Prepare script missing: {PREPARE_SCRIPT}")
        return False
    return run(
        [py, str(PREPARE_SCRIPT),
         "--defect-dir", str(balanced_defective),
         "--pass-dir",   str(balanced_nondefective),
         "--output-dir", str(data_dir),
         "--test-size", "0.2", "--val-size", "0.1", "--seed", "42"],
        "0b. split balanced data into train/val/test",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full PDE4444 pipeline — raw data to aggregated report."
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=REPO_ROOT / "runs_new",
        help="Root output directory for this run (default: runs_new/)."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Balanced train/val/test directory (default: <runs-dir>/data_balanced)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run steps even if output directory already exists."
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=list(range(0, 11)),
        metavar="N", help="Run only these step numbers (default: 0-10)."
    )
    args = parser.parse_args()

    py    = str(VENV_PY if VENV_PY.exists() else sys.executable)
    runs  = args.runs_dir
    force = args.force
    data  = args.data_dir if args.data_dir is not None else runs / "data_balanced"

    runs.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    # ── Step 0: build balanced dataset ──────────────────────────────────────
    if 0 in args.steps:
        ok = step_build_dataset(py, data, force)
        if not ok:
            errors.append("step 0 — build balanced dataset")
            print("\n[FATAL] Cannot continue without dataset.")
            sys.exit(1)

    if not data.exists():
        print(f"[ERROR] Data directory not found: {data}")
        print("Run step 0 first (or pass --steps 0 ...)")
        sys.exit(1)

    data_str = str(data)

    # ── Step 1: sklearn baselines ────────────────────────────────────────────
    if 1 in args.steps:
        out = runs / "iter1_sklearn"
        if not skip_if_exists(out, force, "sklearn baselines"):
            ok = run([py, str(STACK_ROOT / "train_sklearn_baseline.py"),
                      "--data-dir", data_str, "--output-dir", str(out)],
                     "1. sklearn baselines")
            if not ok:
                errors.append("step 1 — sklearn baselines")

    # ── Step 2: CNN activation sweep ────────────────────────────────────────
    if 2 in args.steps:
        out = runs / "iter1_cnn_activation"
        if not skip_if_exists(out, force, "CNN activation sweep"):
            ok = run([py, str(STACK_ROOT / "train_cnn_activation_sweep.py"),
                      "--data-dir", data_str, "--output-dir", str(out)],
                     "2. CNN activation sweep (scratch)")
            if not ok:
                errors.append("step 2 — CNN activation sweep")

    # ── Step 3: optimiser comparison ────────────────────────────────────────
    if 3 in args.steps:
        out = runs / "iter1_optimizer"
        if not skip_if_exists(out, force, "optimiser comparison"):
            ok = run([py, str(STACK_ROOT / "train_optimizer_comparison.py"),
                      "--data-dir", data_str, "--output-dir", str(out)],
                     "3. optimiser comparison")
            if not ok:
                errors.append("step 3 — optimiser comparison")

    # ── Step 4: MLP random search ────────────────────────────────────────────
    if 4 in args.steps:
        out = runs / "iter1_mlp_search"
        if not skip_if_exists(out, force, "MLP random search"):
            ok = run([py, str(STACK_ROOT / "train_keras_mlp_random_search.py"),
                      "--data-dir", data_str, "--output-dir", str(out),
                      "--max-trials", "24", "--img-size", "96", "--batch-size", "64",
                      "--epochs", "35"],
                     "4. MLP random search (24 trials, 96px)")
            if not ok:
                errors.append("step 4 — MLP random search")

    # ── Step 5: MobileNetV2 frozen ───────────────────────────────────────────
    if 5 in args.steps:
        out = runs / "iter2_mobilenet"
        if not skip_if_exists(out, force, "MobileNetV2 frozen"):
            ok = run([py, str(STACK_ROOT / "train_cnn_pretrained.py"),
                      "--data-dir", data_str, "--output-dir", str(out),
                      "--epochs", "40"],
                     "5. MobileNetV2 frozen × 5 activations")
            if not ok:
                errors.append("step 5 — MobileNetV2 frozen")

    # ── Step 6: MobileNetV2 fine-tuned ──────────────────────────────────────
    if 6 in args.steps:
        out = runs / "iter3_mobilenet_finetune"
        if not skip_if_exists(out, force, "MobileNetV2 fine-tuned"):
            ok = run([py, str(STACK_ROOT / "train_cnn_pretrained.py"),
                      "--data-dir", data_str, "--output-dir", str(out),
                      "--unfreeze",
                      "--epochs", "60", "--batch-size", "32",
                      "--img-size", "256", "--patience", "12"],
                     "6. MobileNetV2 fine-tuned × 5 activations")
            if not ok:
                errors.append("step 6 — MobileNetV2 fine-tuned")

    # ── Step 7: YOLO26n-cls ──────────────────────────────────────────────────
    if 7 in args.steps:
        out = runs / "iter1_yolo"
        if not skip_if_exists(out, force, "YOLO26n-cls"):
            ok = run([py, str(STACK_ROOT / "train_yolo26_cls.py"),
                      "--data-dir", data_str, "--output-dir", str(out),
                      "--imgsz", "320", "--batch", "64", "--epochs", "50"],
                     "7. YOLO26n-cls (img=320, batch=64, epochs=50)")
            if not ok:
                errors.append("step 7 — YOLO26n-cls")

    # ── Step 8: YOLO26s-cls ──────────────────────────────────────────────────
    if 8 in args.steps:
        out = runs / "iter1_yolo_s"
        if not skip_if_exists(out, force, "YOLO26s-cls"):
            ok = run([py, str(STACK_ROOT / "train_yolo26_cls.py"),
                      "--data-dir", data_str, "--output-dir", str(out),
                      "--model", "yolo26s-cls.pt",
                      "--imgsz", "320", "--batch", "64", "--epochs", "50"],
                     "8. YOLO26s-cls (img=320, batch=64, epochs=50)")
            if not ok:
                errors.append("step 8 — YOLO26s-cls")

    # ── Step 9: 5-fold cross-validation ─────────────────────────────────────
    if 9 in args.steps:
        out = runs / "iter1_crossval"
        if not skip_if_exists(out, force, "5-fold CV"):
            ok = run([py, str(STACK_ROOT / "train_cross_validation.py"),
                      "--data-dir", data_str, "--output-dir", str(out)],
                     "9. 5-fold cross-validation")
            if not ok:
                errors.append("step 9 — 5-fold CV")

    # ── Step 10: aggregate ───────────────────────────────────────────────────
    if 10 in args.steps:
        out = runs / "final_report"
        out.mkdir(parents=True, exist_ok=True)
        ok = run([py, str(STACK_ROOT / "aggregate_results.py"),
                  "--runs-dir", str(runs), "--output-dir", str(out)],
                 "10. aggregate results + comparison chart")
        if not ok:
            errors.append("step 10 — aggregate")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    if errors:
        print(f"  Pipeline finished with {len(errors)} error(s):")
        for e in errors:
            print(f"    x  {e}")
        sys.exit(1)
    else:
        print("  Pipeline completed successfully.")
        print(f"  Artefacts: {runs}/")
        print(f"  Report:    {runs}/final_report/all_results.csv")
    print("="*70)


if __name__ == "__main__":
    main()
