#!/usr/bin/env python3
"""Compute per-fold grouped binary Dice from nnU-Net cross-validation predictions.

nnU-Net stores validation predictions in:
    <nnunet_results>/<dataset>/<trainer>__<plans>__<config>/fold_X/validation/

Computes three grouped Dice metrics matching the paper's table:
    Kidneys + masses : pred >= 1  vs  gt >= 1
    Tumour + cyst    : pred >= 2  vs  gt >= 2
    Tumour           : pred == 2  vs  gt == 2

Usage:
    conda activate convnextv2

    python baselines/evaluate_nnunet.py \
        --nnunet_results_dir /path/to/nnUNetTrainer__nnUNetPlans__3d_fullres \
        --kits23_dir /path/to/kits23/dataset \
        --folds 5
"""

import os, argparse, json
import numpy as np
import nibabel as nib
from glob import glob


def dice_binary(pred_mask, gt_mask):
    pred = pred_mask.astype(bool).ravel()
    gt = gt_mask.astype(bool).ravel()
    inter = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    return 2.0 * inter / union if union > 0 else float("nan")


def evaluate_fold(nnunet_results_dir, fold_idx, kits23_dir):
    val_dir = os.path.join(nnunet_results_dir, f"fold_{fold_idx}", "validation")
    if not os.path.isdir(val_dir):
        print(f"  WARNING: {val_dir} does not exist — skipping fold {fold_idx}")
        return []

    pred_files = sorted(glob(os.path.join(val_dir, "*.nii.gz")))
    if not pred_files:
        print(f"  WARNING: No .nii.gz files in {val_dir}")
        return []

    results = []
    for pred_path in pred_files:
        case_name = os.path.basename(pred_path).replace(".nii.gz", "")
        case_num = case_name.replace("KiTS23_", "")
        gt_path = os.path.join(kits23_dir, f"case_{case_num}",
                               "segmentation.nii.gz")
        if not os.path.exists(gt_path):
            continue

        pred = nib.load(pred_path).get_fdata().astype(np.int32)
        gt = nib.load(gt_path).get_fdata().astype(np.int32)

        if pred.shape != gt.shape:
            print(f"  WARNING: shape mismatch {case_name}: "
                  f"pred={pred.shape} gt={gt.shape} — skipping")
            continue

        km = dice_binary(pred >= 1, gt >= 1)
        tc = dice_binary(pred >= 2, gt >= 2)
        t  = dice_binary(pred == 2, gt == 2)

        results.append({
            "case_id": case_name, "fold": fold_idx,
            "kidneys_masses": km, "tumour_cyst": tc, "tumour": t,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-fold grouped Dice from nnU-Net cross-val.")
    parser.add_argument("--nnunet_results_dir", type=str, required=True,
                        help="Path to nnUNetTrainer__nnUNetPlans__3d_fullres")
    parser.add_argument("--kits23_dir", type=str,
                        default="/scratch2/salonso/medical/kits23/dataset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON results")
    args = parser.parse_args()

    print("=" * 64)
    print(" nnU-Net — Per-Fold Grouped Dice Evaluation")
    print("=" * 64)
    print(f"  Results dir : {args.nnunet_results_dir}")
    print(f"  KiTS23 dir  : {args.kits23_dir}")
    print(f"  Folds       : {args.folds}\n")

    fold_summaries = {}
    for fold_idx in range(args.folds):
        print(f"--- Fold {fold_idx + 1} ---")
        results = evaluate_fold(args.nnunet_results_dir, fold_idx, args.kits23_dir)
        if not results:
            print(f"  No results for fold {fold_idx}")
            continue

        km_vals = [r["kidneys_masses"] for r in results if not np.isnan(r["kidneys_masses"])]
        tc_vals = [r["tumour_cyst"] for r in results if not np.isnan(r["tumour_cyst"])]
        t_vals  = [r["tumour"] for r in results if not np.isnan(r["tumour"])]

        km_mean = np.mean(km_vals) if km_vals else float("nan")
        tc_mean = np.mean(tc_vals) if tc_vals else float("nan")
        t_mean  = np.mean(t_vals)  if t_vals  else float("nan")
        all_mean = np.nanmean([km_mean, tc_mean, t_mean])

        fold_summaries[fold_idx] = {
            "all": float(all_mean), "kidneys_masses": float(km_mean),
            "tumour_cyst": float(tc_mean), "tumour": float(t_mean),
            "n_cases": len(results),
        }
        print(f"  n={len(results):3d}  All={all_mean:.4f}  K+M={km_mean:.4f}  "
              f"T+C={tc_mean:.4f}  T={t_mean:.4f}")

    if fold_summaries:
        print("\n" + "=" * 64)
        avg_all = np.mean([v["all"] for v in fold_summaries.values()])
        avg_km  = np.mean([v["kidneys_masses"] for v in fold_summaries.values()])
        avg_tc  = np.mean([v["tumour_cyst"] for v in fold_summaries.values()])
        avg_t   = np.mean([v["tumour"] for v in fold_summaries.values()])
        print(f" Average (n_folds={len(fold_summaries)}):  "
              f"All={avg_all:.4f}  K+M={avg_km:.4f}  "
              f"T+C={avg_tc:.4f}  T={avg_t:.4f}")
        print("=" * 64)
        print(f"\n LaTeX row: nnU-Net & {avg_all:.4f} & {avg_km:.4f} "
              f"& {avg_tc:.4f} & {avg_t:.4f} \\\\")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out = {
            "per_fold": {str(k): v for k, v in fold_summaries.items()},
            "average": {
                "all": float(avg_all), "kidneys_masses": float(avg_km),
                "tumour_cyst": float(avg_tc), "tumour": float(avg_t),
            },
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n Results saved to {args.output}")


if __name__ == "__main__":
    main()
