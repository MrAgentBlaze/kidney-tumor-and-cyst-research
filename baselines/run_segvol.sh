#!/bin/bash
# SegVol Zero-Shot Evaluation Pipeline for KiTS23
#
# Evaluates the pretrained SegVol model (BAAI/SegVol) on KiTS23 using
# text-prompted zero-shot inference. No training required.
#
# Environment: conda activate segvol
# -------------------------------------------------------------------

set -e

KITS23_DIR="/scratch2/salonso/medical/kits23/dataset"
GPU=1
RESULTS_DIR="baselines/results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

echo "============================================================"
echo " SegVol Zero-Shot Evaluation Pipeline — KiTS23"
echo "============================================================"
echo ""
echo "  KiTS23 data  : ${KITS23_DIR}"
echo "  GPU          : ${GPU}"
echo "  Results dir  : ${RESULTS_DIR}"
echo ""

mkdir -p "${RESULTS_DIR}"

# Step 1: Per-fold Dice (zoom-out only)
echo "--- Step 1: Per-fold evaluation (zoom-out only) ---"
python baselines/run_segvol.py \
    --gpu ${GPU} \
    --output "${RESULTS_DIR}/segvol_evaluation_zoomout.json"
echo ""

# Step 2: Per-fold Dice (zoom-out + zoom-in)
echo "--- Step 2: Per-fold evaluation (zoom-in/out) ---"
python baselines/run_segvol.py \
    --gpu ${GPU} \
    --use_zoom \
    --output "${RESULTS_DIR}/segvol_evaluation_zoomin.json"
echo ""

# Step 3: Computational benchmark
echo "--- Step 3: Computational benchmark ---"
python baselines/benchmark_segvol_inference.py \
    --gpu ${GPU} \
    --use_zoom \
    --output "${RESULTS_DIR}/segvol_benchmark.json"
echo ""

echo "============================================================"
echo " Pipeline complete."
echo ""
echo " Results:"
echo "   ${RESULTS_DIR}/segvol_evaluation_zoomout.json"
echo "   ${RESULTS_DIR}/segvol_evaluation_zoomin.json"
echo "   ${RESULTS_DIR}/segvol_benchmark.json"
echo "============================================================"
