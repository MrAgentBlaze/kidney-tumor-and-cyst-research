import os
import torch

OUTPUT_DIR = "data/kits23_processed_highres_v2"
CASE_ID = "00001"

# Candidate HU intervals.
HU_RANGES = [
    (-100, 200),
    (-100, 250),
    (-100, 300),
    (-100, 400),
    (None, None),  # No HU filtering.
]

LABEL_NAMES = {
    1: "Kidney",
    2: "Tumor",
    3: "Cyst",
}


def evaluate_component(path):
    data = torch.load(path, map_location="cpu", weights_only=False)

    image = data["image"].as_tensor()
    label = data["label"].as_tensor()
    roi = data["roi"].as_tensor()

    roi_label = int(os.path.basename(path).split("_")[-1][:-3])
    roi_mask = roi == roi_label

    print("\n" + "=" * 85)
    print(f"Component: {os.path.basename(path)}")
    print(f"ROI voxels: {roi_mask.sum().item():,}")
    print("=" * 85)

    print(
        f"\n{'HU interval':<18}"
        f"{'Sparse voxels':>16}"
        f"{'Kidney cov.':>14}"
        f"{'Tumor cov.':>14}"
        f"{'Cyst cov.':>14}"
    )

    for hu_min, hu_max in HU_RANGES:

        if hu_min is None:
            hu_mask = torch.ones_like(image, dtype=torch.bool)
            range_name = "No filtering"
        else:
            hu_mask = (image >= hu_min) & (image <= hu_max)
            range_name = f"[{hu_min}, {hu_max}]"

        sparse_mask = roi_mask & hu_mask
        sparse_count = sparse_mask.sum().item()

        coverages = []

        for class_id in LABEL_NAMES:
            class_mask = label == class_id
            total = class_mask.sum().item()
            selected = (class_mask & sparse_mask).sum().item()

            coverage = (
                100.0 * selected / total
                if total > 0 else float("nan")
            )

            coverages.append(coverage)

        print(
            f"{range_name:<18}"
            f"{sparse_count:>16,}"
            f"{coverages[0]:>13.3f}%"
            f"{coverages[1]:>13.3f}%"
            f"{coverages[2]:>13.3f}%"
        )


def main():
    print("HU THRESHOLD COMPARISON")
    print(f"Case: {CASE_ID}")

    for component_id in (1, 2):
        path = os.path.join(
            OUTPUT_DIR,
            f"{CASE_ID}_{component_id}.pt",
        )

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        evaluate_component(path)


if __name__ == "__main__":
    main()
