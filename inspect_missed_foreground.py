
import os
import torch
import numpy as np

from scipy import ndimage
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
)


CASE_ID = "00001"

ORIGINAL_DATA_DIR = (
    "/mnt/d/Brian Lala/Research/"
    "2-step-segmentation-kidney-tumor-and-cyst/data/KiTS23"
)

CROP_DIR = "data/kits23_processed_highres_v2"

HIGHRES_SPACING = (0.78, 0.78, 0.78)


# --------------------------------------------------------
# Load and transform original segmentation
# --------------------------------------------------------

case_dir = os.path.join(
    ORIGINAL_DATA_DIR,
    f"case_{CASE_ID}",
)

data = LoadImaged(
    keys=("image", "label"),
    image_only=False,
)({
    "image": os.path.join(case_dir, "imaging.nii.gz"),
    "label": os.path.join(case_dir, "segmentation.nii.gz"),
})

data = EnsureChannelFirstd(
    keys=["image", "label"]
)(data)

data = Orientationd(
    keys=["image", "label"],
    axcodes="PLI",
)(data)

data = Spacingd(
    keys=["image", "label"],
    pixdim=HIGHRES_SPACING,
    mode=("bilinear", "nearest"),
)(data)

full_label = data["label"][0].cpu().numpy()
full_gt = full_label > 0

full_roi = np.zeros(
    full_gt.shape,
    dtype=bool,
)


# --------------------------------------------------------
# Reconstruct combined ROI
# --------------------------------------------------------

for component_id in [1, 2]:

    path = os.path.join(
        CROP_DIR,
        f"{CASE_ID}_{component_id}.pt",
    )

    crop = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    roi = crop["roi"][0].numpy() > 0

    start = np.asarray(
        crop["foreground_start_coord"],
        dtype=np.int64,
    )

    end = np.asarray(
        crop["foreground_end_coord"],
        dtype=np.int64,
    )

    slices = tuple(
        slice(int(s), int(e))
        for s, e in zip(start, end)
    )

    full_roi[slices] |= roi


# --------------------------------------------------------
# Analyze missed foreground
# --------------------------------------------------------

missed = full_gt & ~full_roi

print("\n===== MISSED FOREGROUND =====")
print("Total missed voxels:", int(missed.sum()))

classes, counts = np.unique(
    full_label[missed],
    return_counts=True,
)

print("\nMissed voxels by GT class:")

for cls, count in zip(classes, counts):
    print(f"  Class {int(cls)}: {int(count)}")


# --------------------------------------------------------
# Connected-component analysis
# --------------------------------------------------------

structure = ndimage.generate_binary_structure(
    rank=3,
    connectivity=3,
)

components, num_components = ndimage.label(
    missed,
    structure=structure,
)

component_sizes = np.bincount(
    components.ravel()
)[1:]

print("\n===== CONNECTED COMPONENTS =====")
print("Number of missed components:", num_components)

if num_components > 0:

    print(
        "Largest component sizes:",
        sorted(component_sizes.tolist(), reverse=True)[:20],
    )

    for component_id in np.argsort(component_sizes)[::-1][:10] + 1:

        coords = np.argwhere(
            components == component_id
        )

        print(f"\nComponent {component_id}")
        print("  Voxels:", len(coords))
        print("  Bounding box min:", coords.min(axis=0).tolist())
        print("  Bounding box max:", coords.max(axis=0).tolist())


# --------------------------------------------------------
# Distance to nearest ROI
# --------------------------------------------------------

distance = ndimage.distance_transform_edt(
    ~full_roi,
)

missed_distances = distance[missed]

print("\n===== DISTANCE TO ROI =====")

if missed_distances.size > 0:
    print(
        "Minimum distance (voxels):",
        float(missed_distances.min()),
    )
    print(
        "Maximum distance (voxels):",
        float(missed_distances.max()),
    )
    print(
        "Mean distance (voxels):",
        float(missed_distances.mean()),
    )
    print(
        "Maximum distance (mm):",
        float(missed_distances.max() * HIGHRES_SPACING[0]),
    )
