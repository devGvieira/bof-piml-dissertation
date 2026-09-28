#!/bin/bash
# BOF PIML Pipeline — runs all 5 scripts in sequence with logging.
# Usage: bash scripts/run_pipeline.sh [from_step]
# Example: bash scripts/run_pipeline.sh 3   (skip to step 3)

set -e

FROM_STEP=${1:-1}
LOG="results/pipeline_$(date +%Y%m%d_%H%M%S).log"

mkdir -p data models results Images/results Images/shap

echo "=== BOF PIML Pipeline ===" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "Starting from step: $FROM_STEP" | tee -a "$LOG"

# Step 1: Generate synthetic data
if [ "$FROM_STEP" -le 1 ]; then
    echo "" | tee -a "$LOG"
    echo "[1/6] Generating synthetic BOF data (N=10000, seed=42)..." | tee -a "$LOG"
    python -m scripts.generate_bof_data 2>&1 | tee -a "$LOG"
    echo "  -> data/bof_synthetic.csv" | tee -a "$LOG"
fi

# Step 2: Feature engineering
if [ "$FROM_STEP" -le 2 ]; then
    echo "" | tee -a "$LOG"
    echo "[2/6] Computing physics-derived features..." | tee -a "$LOG"
    python -m scripts.feature_engineering 2>&1 | tee -a "$LOG"
    echo "  -> data/features_raw.csv, data/features_physics.csv" | tee -a "$LOG"
fi

# Step 3: Train models (longest step)
if [ "$FROM_STEP" -le 3 ]; then
    echo "" | tee -a "$LOG"
    echo "[3/6] Training models (Configs A, B, C) — this takes a while..." | tee -a "$LOG"
    python -m scripts.train_models 2>&1 | tee -a "$LOG"
    echo "  -> results/metrics.csv, models/*.joblib" | tee -a "$LOG"
fi

# Step 4: Evaluate and compare
if [ "$FROM_STEP" -le 4 ]; then
    echo "" | tee -a "$LOG"
    echo "[4/6] Evaluating and comparing configurations..." | tee -a "$LOG"
    python -m scripts.evaluate_compare 2>&1 | tee -a "$LOG"
    echo "  -> results/comparison_table.csv, results/extrapolation_results.csv" | tee -a "$LOG"
    echo "  -> Images/results/*.pdf" | tee -a "$LOG"
fi

# Step 5: SHAP analysis
if [ "$FROM_STEP" -le 5 ]; then
    echo "" | tee -a "$LOG"
    echo "[5/6] Running SHAP interpretability analysis..." | tee -a "$LOG"
    python -m scripts.shap_analysis 2>&1 | tee -a "$LOG"
    echo "  -> Images/shap/*.pdf, results/shap_importance.csv" | tee -a "$LOG"
fi

# Step 6: Circularity / mutual-information analysis
if [ "$FROM_STEP" -le 6 ]; then
    echo "" | tee -a "$LOG"
    echo "[6/6] Running circularity (mutual information) analysis..." | tee -a "$LOG"
    python -m scripts.analyze_circularity 2>&1 | tee -a "$LOG"
    echo "  -> results/mutual_info.csv" | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "Pipeline completed: $(date)" | tee -a "$LOG"
echo "Log saved to: $LOG"
echo ""
echo "Key outputs:"
echo "  results/metrics.csv          — MAE/RMSE/R2/within_tolerance for all models"
echo "  results/comparison_table.csv — A vs C delta comparison"
echo "  results/shap_importance.csv  — mean |SHAP| by feature/model/target"
echo "  Images/results/              — heatmaps and residual plots"
echo "  Images/shap/                 — SHAP summary and dependence plots"
