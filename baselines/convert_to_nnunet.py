#!/usr/bin/env python3
"""Convert KiTS23 data to nnU-Net v2 format.

Creates the directory structure and naming convention expected by nnU-Net,
plus fold splits matching the paper (KFold with shuffle=True, random_state=42).

Usage:
    conda activate convnextv2

    python baselines/convert_to_nnunet.py \
        --kits23_dir /path/to/kits23/dataset \
        --output_dir /path/to/nnunet_raw/Dataset001_KiTS23

After conversion:
    nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
"""

import os, json, argparse
import numpy as np
from glob import glob
from sklearn.model_selection import KFold


def convert_kits23_to_nnunet(kits23_dir, output_dir, folds=5,
                              random_state=42):
    cases = sorted(glob(os.path.join(kits23_dir, "case_*")))
    valid_cases = [
        c for c in cases
        if (os.path.exists(os.path.join(c, "imaging.nii.gz"))
            and os.path.exists(os.path.join(c, "segmentation.nii.gz")))
    ]
    print(f"Found {len(valid_cases)} valid cases")

    images_dir = os.path.join(output_dir, "imagesTr")
    labels_dir = os.path.join(output_dir, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    case_ids = []
    for case_dir in valid_cases:
        case_num = os.path.basename(case_dir).replace("case_", "")
        case_id = f"KiTS23_{case_num}"
        case_ids.append(case_id)

        img_src = os.path.join(case_dir, "imaging.nii.gz")
        seg_src = os.path.join(case_dir, "segmentation.nii.gz")
        img_dst = os.path.join(images_dir, f"{case_id}_0000.nii.gz")
        seg_dst = os.path.join(labels_dir, f"{case_id}.nii.gz")

        if not os.path.exists(img_dst):
            os.symlink(os.path.abspath(img_src), img_dst)
        if not os.path.exists(seg_dst):
            os.symlink(os.path.abspath(seg_src), seg_dst)

    # dataset.json required by nnU-Net v2
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "kidney": 1, "tumor": 2, "cyst": 3},
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
        "name": "KiTS23",
        "description": "KiTS23 Kidney Tumour Segmentation Challenge",
        "reference": "https://kits-challenge.org/kits23/",
        "licence": "CC-BY-NC-SA 4.0",
    }
    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    print(f"Created dataset.json at {json_path}")

    # Fold splits
    if folds > 0:
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        splits = []
        indices = np.arange(len(case_ids))
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            splits.append({
                "train": [case_ids[i] for i in train_idx],
                "val": [case_ids[i] for i in val_idx],
            })
            print(f"  Fold {fold_idx}: {len(train_idx)} train, {len(val_idx)} val")

        splits_path = os.path.join(output_dir, "splits_final.json")
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=4)
        print(f"Created splits_final.json at {splits_path}")
        print(f"\nCopy to preprocessed dir after planning:")
        print(f"  cp {splits_path} $nnUNet_preprocessed/Dataset001_KiTS23/")

    print(f"\nConversion complete. Total cases: {len(case_ids)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert KiTS23 to nnU-Net v2 format.")
    parser.add_argument("--kits23_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="data/nnunet_raw/Dataset001_KiTS23")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    convert_kits23_to_nnunet(
        args.kits23_dir, args.output_dir, args.folds, args.random_state)


if __name__ == "__main__":
    main()
