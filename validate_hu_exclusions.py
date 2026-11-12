import os
import torch
import numpy as np

OUTPUT_DIR = "data/kits23_processed_highres_v2"
CASE_ID = "00001"

EMPTY_MIN = -100
EMPTY_MAX = 200

LABEL_NAMES = {
    1: "Kidney",
    2: "Tumor",
    3: "Cyst",
}

PERCENTILES = [1, 5, 25, 50, 75, 95, 99]


def describe_hu(values):
    """Print HU statistics for a collection of voxels."""
    values = values.detach().cpu().numpy().astype(np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        print("    No voxels available.")
        return

    percentiles = np.percentile(values, PERCENTILES)

    print(f"    Count: {len(values):,}")
    print(f"    Min:   {values.min():.2f} HU")
    print(f"    Max:   {values.max():.2f} HU")
    print(f"    Mean:  {values.mean():.2f} HU")
    print("    Percentiles:")

    for percentile, value in zip(PERCENTILES, percentiles):
        print(f"      P{percentile}: {value:.2f} HU")


def validate_component(path):
    data = torch.load(path, map_location="cpu", weights_only=False)

    image = data["image"].as_tensor()[0]
    label = data["label"].as_tensor()[0]
    roi = data["roi"].as_tensor()[0]

    roi_label = int(os.path.basename(path).split("_")[-1][:-3])

    roi_mask = roi == roi_label
    hu_low = image < EMPTY_MIN
    hu_high = image > EMPTY_MAX
    hu_valid = (image >= EMPTY_MIN) & (image <= EMPTY_MAX)

    print("\n" + "=" * 70)
    print(f"Component: {os.path.basename(path)}")
    print(f"ROI label: {roi_label}")
    print(f"HU interval: [{EMPTY_MIN}, {EMPTY_MAX}]")
    print("=" * 70)

    for class_id, class_name in LABEL_NAMES.items():
        class_mask = label == class_id

        # All ground-truth voxels of this class in the crop.
        all_gt = class_mask

        # Ground-truth voxels inside the selected ROI.
        roi_gt = class_mask & roi_mask

        # Ground-truth voxels excluded by the HU filter.
        below = roi_gt & hu_low
        above = roi_gt & hu_high
        selected = roi_gt & hu_valid

        print(f"\n--- {class_name} (label {class_id}) ---")

        print("\n  All ground-truth voxels in crop:")
        describe_hu(image[all_gt])

        print("\n  Ground-truth voxels inside ROI:")
        describe_hu(image[roi_gt])

        print("\n  HU exclusion counts inside ROI:")
        print(f"    Below {EMPTY_MIN} HU: {below.sum().item():,}")
        print(f"    Above {EMPTY_MAX} HU: {above.sum().item():,}")
        print(f"    Within HU interval:  {selected.sum().item():,}")

        total_roi = roi_gt.sum().item()
        excluded = below.sum().item() + above.sum().item()

        if total_roi > 0:
            print(f"    Excluded percentage: {100 * excluded / total_roi:.4f}%")

        print("\n  HU distribution of excluded voxels:")
        print("    Below threshold:")
        describe_hu(image[below])

        print("    Above threshold:")
        describe_hu(image[above])


def main():
    print("HU EXCLUSION ANALYSIS")
    print(f"Case: {CASE_ID}")

    for component_id in (1, 2):
        path = os.path.join(
            OUTPUT_DIR,
            f"{CASE_ID}_{component_id}.pt",
        )

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        validate_component(path)


if __name__ == "__main__":
    main()
