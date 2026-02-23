#!/bin/bash
# nnU-Net v2 Training & Evaluation Pipeline for KiTS23
#
# End-to-end pipeline: data conversion, planning, 5-fold training,
# configuration search, and grouped Dice evaluation.
#
# Environment: conda activate convnextv2
# Estimated GPU time: ~2-3 days on a single A100 for all 5 folds.
# -------------------------------------------------------------------

set -e

KITS23_DIR="/scratch2/salonso/medical/kits23/dataset"
export nnUNet_raw="/scratch2/salonso/medical/kits23/nnunet_raw"
export nnUNet_preprocessed="/scratch2/salonso/medical/data_revision/nnunet_preprocessed"
export nnUNet_results="/scratch2/salonso/medical/data_revision/nnunet_results"

DATASET_ID=1
DATASET_NAME="Dataset001_KiTS23"
CONFIGURATION="3d_fullres"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " nnU-Net v2 Pipeline for KiTS23 (5-fold cross-validation)"
echo "============================================================"
echo ""
echo "  KiTS23 data   : ${KITS23_DIR}"
echo "  nnUNet raw     : ${nnUNet_raw}"
echo "  nnUNet preproc : ${nnUNet_preprocessed}"
echo "  nnUNet results : ${nnUNet_results}"
echo ""

# Step 0: Convert KiTS23 to nnU-Net format
echo "--- Step 0: Convert KiTS23 to nnU-Net format ---"
python "${SCRIPT_DIR}/convert_to_nnunet.py" \
    --kits23_dir "${KITS23_DIR}" \
    --output_dir "${nnUNet_raw}/${DATASET_NAME}" \
    --folds 5 \
    --random_state 42
echo ""

# Step 1: Plan and preprocess
echo "--- Step 1: Plan and preprocess ---"
nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity
echo ""

# Step 2: Copy fold splits into preprocessed dir
echo "--- Step 2: Copy fold splits ---"
SPLITS_SRC="${nnUNet_raw}/${DATASET_NAME}/splits_final.json"
SPLITS_DST="${nnUNet_preprocessed}/${DATASET_NAME}/splits_final.json"
if [ -f "${SPLITS_SRC}" ]; then
    cp "${SPLITS_SRC}" "${SPLITS_DST}"
    echo "Copied custom fold splits to ${SPLITS_DST}"
else
    echo "ERROR: splits_final.json not found at ${SPLITS_SRC}"
    exit 1
fi
echo ""

# Step 3: Train all 5 folds (1000 epochs each)
echo "--- Step 3: Training (5 folds x 1000 epochs) ---"
for FOLD in 0 1 2 3 4; do
    echo ""
    echo "=== Training Fold ${FOLD} ==="
    nnUNetv2_train ${DATASET_ID} ${CONFIGURATION} ${FOLD} --npz
done
echo ""

# Step 4: Find best configuration and produce cross-val predictions
echo "--- Step 4: Find best configuration ---"
nnUNetv2_find_best_configuration ${DATASET_ID} -c ${CONFIGURATION}
echo ""

# Step 5: Compute grouped Dice metrics for the paper table
echo "--- Step 5: Compute grouped Dice metrics ---"
NNUNET_RESULTS_DIR="${nnUNet_results}/${DATASET_NAME}/nnUNetTrainer__nnUNetPlans__${CONFIGURATION}"
python "${SCRIPT_DIR}/evaluate_nnunet.py" \
    --nnunet_results_dir "${NNUNET_RESULTS_DIR}" \
    --kits23_dir "${KITS23_DIR}" \
    --folds 5

echo ""
echo "============================================================"
echo " Pipeline complete."
echo "============================================================"
