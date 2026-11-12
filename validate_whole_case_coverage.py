
import os
import torch
import numpy as np

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
)


# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

CASE_ID = "00001"

ORIGINAL_DATA_DIR = (
    "/mnt/d/Brian Lala/Research/"
    "2-step-segmentation-kidney-tumor-and-cyst/data/KiTS23"
)

CROP_DIR = "data/kits23_processed_highres_v2"

HIGHRES_SPACING = (0.78, 0.78, 0.78)


# --------------------------------------------------------
# Load original case
# --------------------------------------------------------

case_dir = os.path.join(
    ORIGINAL_DATA_DIR,
    f"case_{CASE_ID}",
)

image_path = os.path.join(case_dir, "imaging.nii.gz")
label_path = os.path.join(case_dir, "segmentation.nii.gz")

if not os.path.isfile(image_path):
    raise FileNotFoundError(image_path)

if not os.path.isfile(label_path):
    raise FileNotFoundError(label_path)

print("Loading original case:", CASE_ID)

loader = LoadImaged(
    keys=("image", "label"),
    image_only=False,
)

ensure_channel_first = EnsureChannelFirstd(
    keys=["image", "label"]
)

orientation = Orientationd(
    keys=["image", "label"],
    axcodes="PLI",
)

spacing_small = Spacingd(
    keys=["image", "label"],
    pixdim=HIGHRES_SPACING,
    mode=("bilinear", "nearest"),
)

data = loader({
    "image": image_path,
    "label": label_path,
})

data = ensure_channel_first(data)
data = orientation(data)
data = spacing_small(data)

full_gt = data["label"][0] > 0

full_shape = tuple(full_gt.shape)

print("High-resolution volume shape:", full_shape)
print(
    "Whole-case GT foreground voxels:",
    int(full_gt.sum().item()),
)


# --------------------------------------------------------
# Reconstruct combined ROI
# --------------------------------------------------------

full_roi = torch.zeros(
    full_shape,
    dtype=torch.bool,
)

component_ids = [1, 2]

for component_id in component_ids:

    path = os.path.join(
        CROP_DIR,
        f"{CASE_ID}_{component_id}.pt",
    )

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    crop = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    roi = crop["roi"][0] > 0
    label = crop["label"][0] > 0

    start = np.asarray(
        crop["foreground_start_coord"],
        dtype=np.int64,
    )

    end = np.asarray(
        crop["foreground_end_coord"],
        dtype=np.int64,
    )

    expected_shape = tuple((end - start).tolist())

    if tuple(roi.shape) != expected_shape:
        raise RuntimeError(
            f"Component {component_id}: "
            f"ROI shape {tuple(roi.shape)} "
            f"does not match coordinate range {expected_shape}"
        )

    if tuple(label.shape) != expected_shape:
        raise RuntimeError(
            f"Component {component_id}: "
            "Label shape does not match coordinate range."
        )

    if np.any(start < 0) or np.any(end > np.asarray(full_shape)):
        raise RuntimeError(
            f"Component {component_id}: "
            f"Crop coordinates exceed full volume bounds. "
            f"Start={start}, End={end}, Volume={full_shape}"
        )

    slices = tuple(
        slice(int(s), int(e))
        for s, e in zip(start, end)
    )

    # Restore this component's ROI to full-volume coordinates.
    full_roi[slices] |= roi

    print(f"\nComponent {component_id}")
    print("  Start:", start.tolist())
    print("  End:", end.tolist())
    print("  Shape:", tuple(roi.shape))
    print("  ROI voxels:", int(roi.sum().item()))
    print("  Crop GT foreground:", int(label.sum().item()))


# --------------------------------------------------------
# Whole-case evaluation
# --------------------------------------------------------

overlap = full_roi & full_gt
missed = full_gt & ~full_roi
extra = full_roi & ~full_gt

gt_count = int(full_gt.sum().item())
roi_count = int(full_roi.sum().item())
overlap_count = int(overlap.sum().item())
missed_count = int(missed.sum().item())
extra_count = int(extra.sum().item())

coverage = (
    overlap_count / gt_count
    if gt_count > 0 else float("nan")
)

dice = (
    2 * overlap_count / (gt_count + roi_count)
    if gt_count + roi_count > 0 else float("nan")
)

print("\n===== WHOLE-CASE ROI VALIDATION =====")

print("Volume shape:", full_shape)
print("GT foreground voxels:", gt_count)
print("Combined ROI voxels:", roi_count)
print("Overlap voxels:", overlap_count)
print("Missed foreground voxels:", missed_count)
print("Extra ROI voxels:", extra_count)

print(f"\nWhole-case foreground coverage: {coverage:.6f}")
print(f"Whole-case ROI Dice: {dice:.6f}")

print("\n===== RESULT =====")

if missed_count == 0:
    print("PASS: All ground-truth foreground voxels are covered.")
else:
    print(
        f"WARNING: {missed_count:,} ground-truth "
        "foreground voxels are outside the combined ROI."
    )
