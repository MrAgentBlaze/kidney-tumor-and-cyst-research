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

    print("Stage 1 spatial shape:", spatial_shape)
    print(
        "Predicted ROI voxels:",
        torch.count_nonzero(roi_label).item(),
    )

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
    # Resample original image and ROI to high resolution
    # --------------------------------------------------------

    small_ori = transforms["spacing_small"](data_dict_pli)

    small_roi = transforms["spacing_small"](data_dict_invspa)

    small_ori["roi"] = small_roi["label"]

    if small_ori["image"].shape[1:] != small_ori["roi"].shape[1:]:
        raise RuntimeError(
            f"{case_id}: High-resolution image and ROI shapes differ."
        )

    # --------------------------------------------------------
    # Generate one crop per connected component
    # --------------------------------------------------------

    component_ids = torch.unique(
        small_ori["roi"]
    ).detach().cpu().tolist()

    component_ids = [
        int(value)
        for value in component_ids
        if value > 0
    ]

    print("High-resolution components:", component_ids)

    saved_count = 0

    for component_id in component_ids:

        crop_transform = CropForegroundd(
            keys=["image", "label", "roi"],
            source_key="roi",
            margin=ROI_MARGIN,
            return_coords=True,
            allow_smaller=False,
            select_fn=lambda x, v=component_id: x == v,
        )

        small_ori_copy = copy.deepcopy(small_ori)
        small_ori_copy = crop_transform(small_ori_copy)

        if small_ori_copy["image"].numel() == 0:
            print(
                f"WARNING: Component {component_id} produced an empty crop."
            )
            continue

        # Move outputs to CPU before saving.
        for key in ("image", "label", "roi"):
            small_ori_copy[key] = small_ori_copy[key].cpu()

        out_file = os.path.join(
            output_path,
            f"{case_id}_{component_id}.pt",
        )

        torch.save(small_ori_copy, out_file)

        saved_count += 1

        print(
            f"Saved component {component_id}: "
            f"{out_file}"
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()

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
        keys=["image", "label"],
        pixdim=HIGHRES_SPACING,
        mode=("bilinear", "nearest"),
    )

    orientation = Orientationd(
        keys=["image", "label"],
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