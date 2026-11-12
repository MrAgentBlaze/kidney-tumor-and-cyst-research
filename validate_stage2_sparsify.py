
import os
import torch
import numpy as np

OUTPUT_DIR = "data/kits23_processed_highres_v2"
CASE_ID = "00008"

# Match the default HU filtering used by SparseDataset.
EMPTY_MIN = -300
EMPTY_MAX = 500

LABEL_NAMES = {
    0: "Background",
    1: "Kidney",
    2: "Tumor",
    3: "Cyst",
}


def validate_component(path):
    data = torch.load(path, map_location="cpu", weights_only=False)

    image = data["image"].as_tensor()
    label = data["label"].as_tensor()
    roi = data["roi"].as_tensor()

    roi_label = int(os.path.basename(path).split("_")[-1][:-3])

    # Reproduce the exact sparsify() selection logic.
    hu_mask = (image >= EMPTY_MIN) & (image <= EMPTY_MAX)
    roi_mask = roi == roi_label
    sparse_mask = hu_mask & roi_mask

    print(f"\n{'=' * 65}")
    print(f"Component: {os.path.basename(path)}")
    print(f"ROI label: {roi_label}")
    print(f"Image shape: {tuple(image.shape)}")

    print("\nVoxel selection:")
    print(f"  Total voxels: {image.numel():,}")
    print(f"  HU-selected voxels: {hu_mask.sum().item():,}")
    print(f"  ROI-selected voxels: {roi_mask.sum().item():,}")
    print(f"  Final sparse voxels: {sparse_mask.sum().item():,}")

    print("\nGround-truth coverage after sparsify():")
    print(
        f"{'Class':<15}"
        f"{'GT voxels':>15}"
        f"{'Selected':>15}"
        f"{'Excluded':>15}"
        f"{'Coverage':>12}"
    )

    for class_id, class_name in LABEL_NAMES.items():
        class_mask = label == class_id

        total = class_mask.sum().item()
        selected = (class_mask & sparse_mask).sum().item()
        excluded = total - selected
        coverage = selected / total if total > 0 else float("nan")

        print(
            f"{class_name:<15}"
            f"{total:>15,}"
            f"{selected:>15,}"
            f"{excluded:>15,}"
            f"{coverage:>11.4%}"
        )

    # Separate the effects of ROI filtering and HU filtering.
    print("\nForeground exclusion breakdown:")

    foreground = label > 0

    excluded_by_roi = (foreground & ~roi_mask).sum().item()
    excluded_by_hu_within_roi = (
        foreground & roi_mask & ~hu_mask
    ).sum().item()

    print(f"  Outside ROI: {excluded_by_roi:,}")
    print(f"  Outside HU range but inside ROI: {excluded_by_hu_within_roi:,}")
    print(f"  Total excluded foreground: {(foreground & ~sparse_mask).sum().item():,}")


def main():
    paths = [
        os.path.join(OUTPUT_DIR, f"{CASE_ID}_1.pt"),
        os.path.join(OUTPUT_DIR, f"{CASE_ID}_2.pt"),
    ]

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing component file: {path}")

    print("STAGE 2 SPARSIFY VALIDATION")
    print(f"Case: {CASE_ID}")
    print(f"HU range: [{EMPTY_MIN}, {EMPTY_MAX}]")

    for path in paths:
        validate_component(path)


if __name__ == "__main__":
    main()
