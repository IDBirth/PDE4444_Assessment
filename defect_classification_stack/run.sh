#!/usr/bin/env bash
# =============================================================================
# run.sh – Full pipeline for PDE4444 defect classification
# =============================================================================
# Usage:
#   bash run.sh [DATA_DIR] [RUNS_DIR]
#
# Defaults:
#   DATA_DIR  = ./data/processed
#   RUNS_DIR  = ./runs
#
# The script runs every model family required by the assessment brief and
# saves all artefacts (metrics, plots, model weights) under RUNS_DIR/.
# =============================================================================
set -euo pipefail

DATA_DIR="${1:-./data/processed}"
RUNS_DIR="${2:-./runs}"

PYTHON="${PYTHON:-python}"

echo "================================================================"
echo "  PDE4444 – Defect Classification Pipeline"
echo "  DATA_DIR : $DATA_DIR"
echo "  RUNS_DIR : $RUNS_DIR"
echo "================================================================"

# ---- Step 1: scikit-learn HOG baselines (SVM + MLP) ----------------------
echo ""
echo "[1/6] scikit-learn HOG baselines (SVM + sklearn MLP)..."
$PYTHON train_sklearn_baseline.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$RUNS_DIR/sklearn_baseline" \
    --n-iter 12

# ---- Step 2: CNN activation-function sweep (PyTorch) ----------------------
echo ""
echo "[2/6] CNN activation-function sweep (relu/elu/gelu/selu/leaky_relu)..."
$PYTHON train_cnn_activation_sweep.py \
    --data-dir   "$DATA_DIR" \
    --output-dir "$RUNS_DIR/cnn_activation_sweep" \
    --epochs 20 \
    --batch-size 32

# ---- Step 3: Optimiser comparison (zero / first / second order) -----------
echo ""
echo "[3/6] Optimiser comparison (Nelder-Mead / SGD / Adam / L-BFGS)..."
$PYTHON train_optimizer_comparison.py \
    --data-dir   "$DATA_DIR" \
    --output-dir "$RUNS_DIR/optimizer_comparison" \
    --epochs 50 \
    --nm-iters 800

# ---- Step 4: MLP hyperparameter random search (PyTorch) -------------------
echo ""
echo "[4/6] MLP hyperparameter random search..."
$PYTHON train_keras_mlp_random_search.py \
    --data-dir   "$DATA_DIR" \
    --output-dir "$RUNS_DIR/mlp_hparam_search" \
    --max-trials 12 \
    --epochs 25 \
    --img-size 64

# ---- Step 5: Cross-validation + overfitting analysis ----------------------
echo ""
echo "[5/6] Stratified K-Fold cross-validation (overfitting analysis)..."
$PYTHON train_cross_validation.py \
    --data-dir   "$DATA_DIR" \
    --output-dir "$RUNS_DIR/cross_validation"

# ---- Step 6: YOLO classification ------------------------------------------
echo ""
echo "[6/6] YOLO26n-cls training..."
$PYTHON train_yolo26_cls.py \
    --data-dir   "$DATA_DIR" \
    --output-dir "$RUNS_DIR/yolo26_cls" \
    --epochs 30 \
    --imgsz 224

# ---- Final: aggregate all model results -----------------------------------
echo ""
echo "[Final] Aggregating model comparison..."
$PYTHON compare_results.py \
    --sklearn-dir "$RUNS_DIR/sklearn_baseline" \
    --keras-dir   "$RUNS_DIR/mlp_hparam_search" \
    --cnn-dir     "$RUNS_DIR/cnn_activation_sweep" \
    --yolo-dir    "$RUNS_DIR/yolo26_cls" \
    --output-file "$RUNS_DIR/model_comparison.csv"

echo ""
echo "================================================================"
echo "  All done!  Results are in: $RUNS_DIR/"
echo "================================================================"
