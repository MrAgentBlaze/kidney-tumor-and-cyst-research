"""
ROI processing pipeline for Stage 2 (v2).

Reconstructs regions of interest from Stage 1 sparse predictions,
maps them back to the original CT coordinate system, and generates
high-resolution crops for Stage 2 segmentation training.
"""

import argparse
import copy
import glob
import os
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import tqdm
import scipy.ndimage as ndi
import nibabel as nib

from monai.data import MetaTensor
from monai.transforms import ResampleToMatch

from monai.transforms import (
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Spacingd,
    RemoveSmallObjects,
    ToDeviced,
    CropForegroundd,
    ImageFilter,
)


# ============================================================
# CONFIGURATION
# ============================================================

ORIGINAL_DATA_DIR = (
    "/mnt/d/Brian Lala/Research/"
    "2-step-segmentation-kidney-tumor-and-cyst/data/KiTS23"
)

LARGE_DATA_DIR = "data/kits23_large_processed"
STAGE1_PREDICTIONS_DIR = "results/stage1"

OUTPUT_DIR = "data/kits23_processed_highres_v2"

ROI_THRESHOLD = 0.1
ROI_MARGIN = (35, 35, 10)
HIGHRES_SPACING = (0.78, 0.78, 0.78)
LOWRES_SPACING = (1.99, 1.99, 1.99)


# ============================================================
# METRICS
# ============================================================

def dice(prediction, reference):
    """Compute Dice similarity coefficient between two binary masks."""

    prediction = prediction.bool()
    reference = reference.bool()

    intersection = torch.count_nonzero(prediction & reference)
    numel_pred = torch.count_nonzero(prediction)
    numel_ref = torch.count_nonzero(reference)

    if numel_pred == 0 and numel_ref == 0:
        return float("nan")

    return (
        2.0 * intersection.item()
        / (numel_pred + numel_ref).item()
    )


def efficiency(prediction, reference):
    """Compute the fraction of reference voxels captured by prediction."""

    prediction = prediction.bool()
    reference = reference.bool()

    intersection = torch.count_nonzero(prediction & reference)
    numel_ref = torch.count_nonzero(reference)

    if numel_ref == 0:
        return float("nan")

    return intersection.item() / numel_ref.item()


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

def label_components(label):
    """Label connected components in a binary ROI mask."""

    labelled_mask, _ = ndi.label(
        label[0].detach().cpu().numpy(),
        structure=np.ones((3, 3, 3), dtype=int),
    )

    return (
        torch.as_tensor(labelled_mask, device=label.device)
        .unsqueeze(0)
    )


def dilate_multiclass_labels(labels, num_classes, dilation_filter):
    """Dilate each component independently, preserving label values."""

    dilated_labels = torch.zeros_like(labels)

    for class_idx in range(1, num_classes + 1):
        class_mask = labels == class_idx

        dilated_class_mask = dilation_filter(
            class_mask.float()
        ).bool()

        dilated_labels[dilated_class_mask] = class_idx

    return dilated_labels


# ============================================================
# DATA VALIDATION
# ============================================================

def discover_cases():
    """Build exact case-ID mappings and validate dataset coverage."""

    original_dir = Path(ORIGINAL_DATA_DIR)
    large_dir = Path(LARGE_DATA_DIR)
    prediction_dir = Path(STAGE1_PREDICTIONS_DIR)

    if not original_dir.is_dir():
        raise FileNotFoundError(
            f"Original dataset directory not found: {original_dir}"
        )

    if not large_dir.is_dir():
        raise FileNotFoundError(
            f"Processed dataset directory not found: {large_dir}"
        )

    if not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"Stage 1 prediction directory not found: {prediction_dir}"
        )

    original_cases = {}

    for case_dir in sorted(original_dir.glob("case_*")):
        if not case_dir.is_dir():
            continue

        case_id = case_dir.name.replace("case_", "")

        image_path = case_dir / "imaging.nii.gz"
        label_path = case_dir / "segmentation.nii.gz"

        if image_path.is_file() and label_path.is_file():
            original_cases[case_id] = {
                "image": str(image_path),
                "label": str(label_path),
            }

    prediction_cases = {
        path.stem: str(path)
        for path in prediction_dir.glob("*.pt")
        if path.is_file()
    }

    large_cases = {
        path.stem: str(path)
        for path in large_dir.glob("*.pkl")
        if path.is_file()
    }

    print("\n===== DATASET COVERAGE =====")
    print("Original cases:", len(original_cases))
    print("Stage 1 predictions:", len(prediction_cases))
    print("Processed cases:", len(large_cases))

    missing_predictions = set(original_cases) - set(prediction_cases)
    missing_processed = set(original_cases) - set(large_cases)

    extra_predictions = set(prediction_cases) - set(original_cases)
    extra_processed = set(large_cases) - set(original_cases)

    if (
        missing_predictions
        or missing_processed
        or extra_predictions
        or extra_processed
    ):
        raise RuntimeError(
            "Dataset case mismatch detected.\n"
            f"Missing predictions: {sorted(missing_predictions)[:10]}\n"
            f"Missing processed samples: {sorted(missing_processed)[:10]}\n"
            f"Extra predictions: {sorted(extra_predictions)[:10]}\n"
            f"Extra processed samples: {sorted(extra_processed)[:10]}"
        )

    print("All three datasets have matching case IDs.")

    return original_cases, prediction_cases, large_cases


def validate_affine(large_image, transformed_image, case_id):
    """Check whether the processed sample and transformed original share a grid."""

    large_affine = torch.as_tensor(
        large_image.affine, dtype=torch.float64
    )

    transformed_affine = torch.as_tensor(
        transformed_image.affine, dtype=torch.float64
    )

    if not torch.allclose(
        large_affine,
        transformed_affine,
        atol=1e-4,
        rtol=1e-5,
    ):
        raise RuntimeError(
            f"{case_id}: Affine mismatch between Stage 1 sample "
            "and transformed original image.\n"
            f"Stage 1 affine:\n{large_affine}\n"
            f"Original transformed affine:\n{transformed_affine}"
        )


def create_reference_grid(crop, reference_affine):
    """
    Create a local destination grid aligned with the global
    high-resolution reference grid.

    The destination covers the complete physical extent of
    the source crop.
    """

    crop_affine = crop.affine.detach().cpu().numpy()
    crop_shape = np.asarray(crop.shape[1:], dtype=np.int64)

    # Transform all eight voxel-center corners into
    # global reference-grid coordinates.
    corners = np.array([
        [i, j, k, 1.0]
        for i in (0, crop_shape[0] - 1)
        for j in (0, crop_shape[1] - 1)
        for k in (0, crop_shape[2] - 1)
    ]).T

    world_corners = crop_affine @ corners

    reference_coordinates = (
        np.linalg.inv(reference_affine) @ world_corners
    )[:3]

    # Include every source voxel center in the destination.
    grid_min = np.floor(
        reference_coordinates.min(axis=1)
    ).astype(np.int64)

    grid_max = np.ceil(
        reference_coordinates.max(axis=1)
    ).astype(np.int64)

    grid_shape = grid_max - grid_min + 1

    local_affine = reference_affine.copy()

    local_affine[:3, 3] = (
        reference_affine[:3, 3]
        + reference_affine[:3, :3] @ grid_min
    )

    reference_grid = MetaTensor(
        torch.zeros(
            (1, *grid_shape.tolist()),
            dtype=torch.float32,
        ),
        affine=local_affine,
    )

    return reference_grid, grid_min, grid_max

# ============================================================
# CASE PROCESSING
# ============================================================

def process_case(
    case_id,
    original_paths,
    prediction_path,
    large_path,
    output_path,
    transforms,
    device,
    overwrite=False,
):
    """Process one case and generate high-resolution ROI crops."""

    print(f"\n===== Processing case {case_id} =====")

    case_outputs = glob.glob(
        os.path.join(output_path, f"{case_id}_*.pt")
    )

    if case_outputs:
        if not overwrite:
            raise FileExistsError(
                f"Output files already exist for case {case_id}:\n"
                f"{case_outputs[:5]}\n"
                "Use --overwrite to replace them."
            )

        for existing_file in case_outputs:
            os.remove(existing_file)

        print("Removed existing outputs for this case.")

    # --------------------------------------------------------
    # Load Stage 1 prediction
    # --------------------------------------------------------

    roi = torch.load(
        prediction_path,
        map_location="cpu",
        weights_only=False,
    )

    if "coords" not in roi or "logits" not in roi:
        raise ValueError(
            f"{case_id}: Prediction must contain 'coords' and 'logits'."
        )

    coords = roi["coords"].long()
    probabilities = roi["logits"].float()

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"{case_id}: Invalid coordinate shape {tuple(coords.shape)}"
        )

    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(
            f"{case_id}: Invalid probability shape "
            f"{tuple(probabilities.shape)}"
        )

    if coords.shape[0] != probabilities.shape[0]:
        raise ValueError(
            f"{case_id}: Coordinate and probability counts differ."
        )

    if not torch.isfinite(probabilities).all():
        raise ValueError(
            f"{case_id}: Prediction contains NaN or infinite values."
        )

    if probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError(
            f"{case_id}: Expected probabilities between 0 and 1. "
            "The prediction file may contain raw logits."
        )

    # --------------------------------------------------------
    # Load Stage 1 processed sample
    # --------------------------------------------------------

    with open(large_path, "rb") as fd:
        large_event = pkl.load(fd)

    large_event = transforms["to_device"](large_event)

    image = large_event["image"]
    label = large_event["label"]

    spatial_shape = tuple(label.shape[1:])

    if tuple(image.shape) != tuple(label.shape):
        raise RuntimeError(
            f"{case_id}: Stage 1 image and label shapes differ."
        )

    if coords.numel() > 0:
        minimum = coords.min(dim=0).values
        maximum = coords.max(dim=0).values

        for axis, size in enumerate(spatial_shape):
            if minimum[axis] < 0 or maximum[axis] >= size:
                raise ValueError(
                    f"{case_id}: Coordinates outside spatial grid. "
                    f"Axis {axis}, min={minimum[axis].item()}, "
                    f"max={maximum[axis].item()}, size={size}."
                )

    coords = coords.to(device)
    probabilities = probabilities.to(device)

    # --------------------------------------------------------
    # Reconstruct ROI in Stage 1 coordinate space
    # --------------------------------------------------------

    roi_label = torch.zeros_like(label)

    if coords.shape[0] > 0:
        roi_label[
            0,
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
        ] = (
            probabilities.max(dim=1).values >= ROI_THRESHOLD
        ).float()

    if torch.count_nonzero(roi_label) == 0:
        print("WARNING: Empty Stage 1 ROI. Skipping case.")
        return 0

    # --------------------------------------------------------
    # Load original CT and segmentation
    # --------------------------------------------------------

    data_dict_ori = transforms["loader"]({
        "image": original_paths["image"],
        "label": original_paths["label"],
    })

    data_dict_ori = transforms["ensure_channel_first"](
        data_dict_ori
    )

    data_dict_ori = transforms["to_device"](data_dict_ori)

    # --------------------------------------------------------
    # Transform original data into Stage 1 coordinate space
    # --------------------------------------------------------

    data_dict_pli = transforms["orientation"](data_dict_ori)

    from monai.data.utils import zoom_affine, compute_shape_offset

    # Calculate the full-volume high-resolution reference grid
    # without allocating the resampled image.

    reference_image = data_dict_pli["image"]

    input_affine = (
        reference_image.affine.detach().cpu().numpy()
    )

    input_shape = tuple(reference_image.shape[1:])

    reference_affine = zoom_affine(
        input_affine,
        np.asarray(HIGHRES_SPACING, dtype=float),
        diagonal=False,
    )

    reference_shape, reference_offset = compute_shape_offset(
        input_shape,
        input_affine,
        reference_affine,
        scale_extent=False,
    )

    reference_affine[:3, 3] = reference_offset[:3]

    data_dict_spa = transforms["spacing_large"](data_dict_pli)

    if tuple(data_dict_spa["label"].shape) != tuple(roi_label.shape):
        raise RuntimeError(
            f"{case_id}: ROI shape mismatch.\n"
            f"Prediction ROI: {tuple(roi_label.shape)}\n"
            f"Transformed original: "
            f"{tuple(data_dict_spa['label'].shape)}"
        )

    validate_affine(
        image,
        data_dict_spa["image"],
        case_id,
    )

    # --------------------------------------------------------
    # Clean ROI and identify connected components
    # --------------------------------------------------------

    roi_label_clean = transforms["clean_objects"](roi_label)

    roi_comp = label_components(roi_label_clean)

    num_components = int(roi_comp.max().item())

    if num_components == 0:
        print("WARNING: No connected components remain after cleaning.")
        return 0

    print("Connected components:", num_components)

    roi_label_dilated = dilate_multiclass_labels(
        roi_comp,
        num_components,
        transforms["dilation_filter"],
    )

    # Replace the transformed label with the ROI component mask.
    data_dict_spa["label"][:] = roi_label_dilated[:]

    # --------------------------------------------------------
    # Reverse spacing and orientation transforms
    # --------------------------------------------------------

    data_dict_invspa = transforms["spacing_large"].inverse(
        data_dict_spa
    )

    data_dict_invpli = transforms["orientation"].inverse(
        data_dict_invspa
    )

    prediction = data_dict_invpli["label"]
    reference = data_dict_ori["label"]

    if prediction.shape != reference.shape:
        raise RuntimeError(
            f"{case_id}: Inverse-transformed ROI and original "
            f"segmentation shapes differ: {prediction.shape} "
            f"vs {reference.shape}"
        )

    # --------------------------------------------------------
    # Evaluate ROI quality
    # --------------------------------------------------------

    dice_val = dice(prediction, reference)
    efficiency_val = efficiency(prediction, reference)

    print(
        f"ROI evaluation: Dice={dice_val:.4f}, "
        f"Efficiency={efficiency_val:.4f}"
    )

    # --------------------------------------------------------
    # Crop components first, then resample each crop
    # --------------------------------------------------------

    # Attach the inverse-transformed ROI to the original CT.
    # Both are in the original CT coordinate space.
    data_dict_ori["roi"] = prediction

    # Determine original voxel spacing and orientation.
    original_affine = prediction.affine.detach().cpu().numpy()

    original_axis_codes = nib.aff2axcodes(original_affine)

    original_spacing = np.linalg.norm(
        original_affine[:3, :3],
        axis=0,
    )

    # Convert the existing high-resolution voxel margins
    # into physical millimeters.
    target_margin_mm = (
        np.asarray(ROI_MARGIN, dtype=float)
        * np.asarray(HIGHRES_SPACING, dtype=float)
    )

    # Axis groups:
    # Left/Right = 0, Posterior/Anterior = 1,
    # Inferior/Superior = 2.
    axis_group = {
        "L": 0,
        "R": 0,
        "P": 1,
        "A": 1,
        "I": 2,
        "S": 2,
    }

    # Target grid orientation is PLI:
    # P -> axis 0, L -> axis 1, I -> axis 2.
    target_axis_for_group = {
        1: 0,
        0: 1,
        2: 2,
    }

    margin_original = tuple(
        int(np.ceil(
            target_margin_mm[
                target_axis_for_group[axis_group[code]]
            ] / original_spacing[axis]
        )) + 2
        for axis, code in enumerate(original_axis_codes)
    )

    # Move the original CT, segmentation, and ROI to CPU.
    data_dict_ori_cpu = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in data_dict_ori.items()
    }

    # Release GPU references that are no longer needed.
    del data_dict_ori
    del data_dict_invspa
    del data_dict_invpli
    del data_dict_spa
    del data_dict_pli

    del large_event
    del image
    del label

    del roi_label
    del roi_label_clean
    del roi_comp
    del roi_label_dilated

    del prediction
    del reference
    del coords
    del probabilities

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Identify connected components in original CT space.
    component_ids = torch.unique(
        data_dict_ori_cpu["roi"]
    ).tolist()

    component_ids = [
        int(value)
        for value in component_ids
        if value > 0
    ]

    saved_count = 0

    # --------------------------------------------------------
    # Crop and resample one component at a time
    # --------------------------------------------------------

    # Resamplers for the original high-resolution reference grid.
    resample_image = ResampleToMatch(
        mode="bilinear",
        padding_mode="border",
    )

    resample_label = ResampleToMatch(
        mode="nearest",
        padding_mode="border",
    )

    resample_roi = ResampleToMatch(
        mode="nearest",
        padding_mode="border",
    )
    
    for component_id in component_ids:

        print(f"\nProcessing component {component_id}...")

        # 1. Crop this component in original CT space.
        original_crop_transform = CropForegroundd(
            keys=["image", "label", "roi"],
            source_key="roi",
            margin=margin_original,
            return_coords=True,
            allow_smaller=False,
            select_fn=lambda x, v=component_id: x == v,
        )

        # Final crop uses the original pipeline's high-resolution margin.
        final_crop_transform = CropForegroundd(
            keys=["image", "label", "roi"],
            source_key="roi",
            margin=ROI_MARGIN,
            return_coords=True,
            allow_smaller=False,
            select_fn=lambda x, v=component_id: x == v,
        )

        component_crop = original_crop_transform(
            data_dict_ori_cpu
        )

        if component_crop["image"].numel() == 0:
            print(
                f"WARNING: Component {component_id} "
                "produced an empty original-space crop."
            )
            continue

        # 2. Reorient to the original pipeline's PLI orientation.
        component_crop = transforms["orientation"](
            component_crop
        )

        # 3. Construct a local destination grid aligned with
        #    the global high-resolution reference grid.
        reference_grid, grid_min, grid_max = (
            create_reference_grid(
                component_crop["image"],
                reference_affine,
            )
        )

        # 4. Resample actual image data onto the reference grid.
        component_crop["image"] = resample_image(
            component_crop["image"],
            reference_grid,
        )

        component_crop["label"] = resample_label(
            component_crop["label"],
            reference_grid,
        )

        component_crop["roi"] = resample_roi(
            component_crop["roi"],
            reference_grid,
        )

        del reference_grid

        # 5. Apply the original high-resolution component crop.
        component_crop = final_crop_transform(
            component_crop
        )

        # 6. Validate the resulting tensors.
        image_shape = tuple(
            component_crop["image"].shape[1:]
        )
        label_shape = tuple(
            component_crop["label"].shape[1:]
        )
        roi_shape = tuple(
            component_crop["roi"].shape[1:]
        )

        if not (
            image_shape == label_shape == roi_shape
        ):
            raise RuntimeError(
                f"{case_id}, component {component_id}: "
                f"High-resolution shapes differ. "
                f"Image={image_shape}, "
                f"Label={label_shape}, ROI={roi_shape}"
            )

        # Verify that the final crop begins on integer
        # coordinates of the global reference grid.
        final_affine = (
            component_crop["image"]
            .affine.detach().cpu().numpy()
        )

        final_start = (
            np.linalg.inv(reference_affine) @ final_affine
        )[:3, 3]

        if not np.allclose(
            final_start,
            np.round(final_start),
            atol=1e-4,
        ):
            raise RuntimeError(
                f"{case_id}, component {component_id}: "
                "Final crop is not aligned with the "
                "global reference grid."
            )

        # 7. Save this component.
        out_file = os.path.join(
            output_path,
            f"{case_id}_{component_id}.pt",
        )

        torch.save(component_crop, out_file)

        saved_count += 1

        print(f"Saved component {component_id}: {out_file}")

        del component_crop
        del original_crop_transform

    # Release the original CPU volume.
    del data_dict_ori_cpu

    print(
        f"Case {case_id} completed. "
        f"Saved {saved_count} crop(s)."
    )

    return saved_count


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Stage 2 ROI preprocessing (v2)."
    )

    parser.add_argument(
        "--case_id",
        type=str,
        default=None,
        help="Process one case, e.g. 00000. Default: all cases.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory for high-resolution crops.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files for selected cases.",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # Discover and validate case IDs
    # --------------------------------------------------------

    original_cases, prediction_cases, large_cases = discover_cases()

    if args.case_id is not None:
        case_ids = [args.case_id]

        if args.case_id not in original_cases:
            raise FileNotFoundError(
                f"Original case {args.case_id} not found."
            )

        if args.case_id not in prediction_cases:
            raise FileNotFoundError(
                f"Stage 1 prediction for {args.case_id} not found."
            )

        if args.case_id not in large_cases:
            raise FileNotFoundError(
                f"Processed sample for {args.case_id} not found."
            )

    else:
        case_ids = sorted(original_cases.keys())

    # --------------------------------------------------------
    # Initialize transforms
    # --------------------------------------------------------

    loader = LoadImaged(
        keys=("image", "label"),
        image_only=False,
    )

    ensure_channel_first = EnsureChannelFirstd(
        keys=["image", "label"]
    )

    spacing_large = Spacingd(
        keys=["image", "label"],
        pixdim=LOWRES_SPACING,
        mode=("bilinear", "nearest"),
    )

    spacing_small = Spacingd(
        keys=["image", "label", "roi"],
        pixdim=HIGHRES_SPACING,
        mode=("bilinear", "nearest", "nearest"),
    )

    orientation = Orientationd(
        keys=["image", "label", "roi"],
        axcodes="PLI",
        allow_missing_keys=True,
    )

    dilation_filter = ImageFilter(
        filter="elliptical",
        filter_size=11,
    )

    clean_objects = RemoveSmallObjects(
        min_size=50,
        connectivity=1,
        by_measure=False,
    )

    to_device = ToDeviced(
        keys=["image", "label"],
        device=device,
    )

    transforms = {
        "loader": loader,
        "ensure_channel_first": ensure_channel_first,
        "spacing_large": spacing_large,
        "spacing_small": spacing_small,
        "orientation": orientation,
        "dilation_filter": dilation_filter,
        "clean_objects": clean_objects,
        "to_device": to_device,
    }

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Process cases
    # --------------------------------------------------------

    total_saved = 0

    for case_id in tqdm.tqdm(case_ids, desc="Processing cases"):

        saved = process_case(
            case_id=case_id,
            original_paths=original_cases[case_id],
            prediction_path=prediction_cases[case_id],
            large_path=large_cases[case_id],
            output_path=args.output_dir,
            transforms=transforms,
            device=device,
            overwrite=args.overwrite,
        )

        total_saved += saved

    print("\n===== PREPROCESSING SUMMARY =====")
    print("Cases processed:", len(case_ids))
    print("Total crops saved:", total_saved)
    print("Output directory:", args.output_dir)


if __name__ == "__main__":
    main()