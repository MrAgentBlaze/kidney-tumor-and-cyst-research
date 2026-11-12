import os
import torch
import numpy as np
from scipy import ndimage

FILE_PATH = "data/kits23_processed_highres_v2/00002_1.pt"
ROI_LABEL = 1

def main():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(FILE_PATH)

    data = torch.load(
        FILE_PATH,
        map_location="cpu",
        weights_only=False
    )

    image = data["image"].as_tensor().squeeze(0).numpy()
    label = data["label"].as_tensor().squeeze(0).numpy()
    roi = data["roi"].as_tensor().squeeze(0).numpy()

    # Original ground-truth cyst label is 3.
    cyst_mask = label == 3
    roi_mask = roi == ROI_LABEL

    # Cysts excluded by the ROI, independent of HU filtering.
    missing_mask = cyst_mask & ~roi_mask

    total_cysts = int(cyst_mask.sum())
    inside_roi = int((cyst_mask & roi_mask).sum())
    missing = int(missing_mask.sum())

    print("=" * 65)
    print("MISSING CYST VOXEL INVESTIGATION")
    print("=" * 65)

    print(f"Image shape: {image.shape}")
    print(f"Total cyst voxels: {total_cysts:,}")
    print(f"Cyst voxels inside ROI: {inside_roi:,}")
    print(f"Cyst voxels outside ROI: {missing:,}")
    print(f"Cyst coverage: {inside_roi / total_cysts:.4%}")

    if missing == 0:
        print("\nNo missing cyst voxels.")
        return

    # Connected-component analysis using 26-connectivity.
    structure = np.ones((3, 3, 3), dtype=bool)

    components, num_components = ndimage.label(
        missing_mask,
        structure=structure
    )

    component_sizes = np.bincount(
        components.ravel()
    )[1:]

    print("\n" + "=" * 65)
    print("CONNECTED COMPONENT ANALYSIS")
    print("=" * 65)

    print(f"Number of missing components: {num_components}")
    print(f"Largest component: {component_sizes.max():,}")
    print(f"Smallest component: {component_sizes.min():,}")
    print(f"Mean component size: {component_sizes.mean():.2f}")

    component_order = np.argsort(component_sizes)[::-1]

    for rank, component_idx in enumerate(component_order, start=1):
        component_id = component_idx + 1
        size = int(component_sizes[component_idx])

        coords = np.argwhere(components == component_id)

        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)

        print(f"\nComponent {rank}")
        print(f"  Voxels: {size:,}")
        print(f"  Bounding box min: {bbox_min.tolist()}")
        print(f"  Bounding box max: {bbox_max.tolist()}")

    # Distance from missing voxels to nearest ROI voxel.
    distance_map = ndimage.distance_transform_edt(
        ~roi_mask,
        sampling=(0.78, 0.78, 0.78)
    )

    distances = distance_map[missing_mask]

    print("\n" + "=" * 65)
    print("DISTANCE TO NEAREST ROI VOXEL")
    print("=" * 65)

    print(f"Minimum: {distances.min():.3f} mm")
    print(f"Maximum: {distances.max():.3f} mm")
    print(f"Mean: {distances.mean():.3f} mm")
    print(f"Median: {np.median(distances):.3f} mm")
    print(f"95th percentile: {np.percentile(distances, 95):.3f} mm")

    print("\nDistance distribution:")

    for threshold in [0.78, 1.56, 2.34, 3.12, 5.0, 10.0]:
        count = int((distances <= threshold).sum())
        percentage = count / missing * 100

        print(
            f"  Within {threshold:5.2f} mm: "
            f"{count:6,} ({percentage:6.2f}%)"
        )

    # HU distribution of missing cyst voxels.
    missing_hu = image[missing_mask]

    print("\n" + "=" * 65)
    print("MISSING CYST HU DISTRIBUTION")
    print("=" * 65)

    print(f"Minimum HU: {missing_hu.min():.2f}")
    print(f"Maximum HU: {missing_hu.max():.2f}")
    print(f"Mean HU: {missing_hu.mean():.2f}")
    print(f"Median HU: {np.median(missing_hu):.2f}")

    print("\nInvestigation completed.")


if __name__ == "__main__":
    main()
