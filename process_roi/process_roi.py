"""
ROI processing pipeline for Stage 2.

Takes low-resolution predictions from Stage 1 and crops the original
high-resolution scans to the detected regions of interest, producing
the input data for Stage 2 (full-resolution segmentation).
"""

import os
import copy
import glob
import pickle as pkl
import numpy as np
import torch
import tqdm
import scipy.ndimage as ndi

from monai.data import CacheDataset, DataLoader
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


def dice(prediction, reference):
    """Compute Dice similarity coefficient between two binary masks."""
    prediction = prediction.bool()
    reference = reference.bool()
    intersection = torch.count_nonzero(prediction & reference)
    numel_pred = torch.count_nonzero(prediction)
    numel_ref = torch.count_nonzero(reference)
    if numel_ref == 0 and numel_pred == 0:
        return float("nan")
    return 2 * intersection.item() / (numel_ref + numel_pred).item()


def efficiency(prediction, reference):
    """Compute fraction of reference voxels captured by the prediction."""
    prediction = prediction.bool()
    reference = reference.bool()
    intersection = torch.count_nonzero(prediction & reference)
    numel_ref = torch.count_nonzero(reference)
    numel_pred = torch.count_nonzero(prediction)
    if numel_ref == 0 and numel_pred == 0:
        return float("nan")
    return intersection.item() / numel_ref.item()


def label_components(label):
    """Label connected components in a binary mask."""
    labelled_mask, _ = ndi.label(
        label[0].cpu().numpy(), structure=np.ones((3, 3, 3), dtype=int)
    )
    return torch.tensor(labelled_mask).unsqueeze(0).to(label.device)


def dilate_multiclass_labels(labels, num_classes, dilation_filter):
    """Dilate each class independently, preserving label values."""
    dilated_labels = torch.zeros_like(labels)
    for class_idx in range(1, num_classes + 1):
        class_mask = (labels == class_idx)
        dilated_class_mask = dilation_filter(class_mask.float()).bool()
        dilated_labels[dilated_class_mask] = class_idx
    return dilated_labels


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    # Paths (update these for your setup)
    large_path = "data/kits23_large_processed"
    roi_results_path = "results/stage1"
    output_path = "data/kits23_processed_highres"
    data_dir = "data/kits23/dataset/"

    roi_results_filepaths = sorted(glob.glob(os.path.join(roi_results_path, "*")))

    # Load original data
    train_images = sorted(glob.glob(os.path.join(data_dir, "case_*", "imaging.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "case_*", "segmentation.nii.gz")))
    train_images = [
        x for x in train_images
        if any(x.split("/")[-2][-5:] in path for path in roi_results_filepaths)
    ]
    train_labels = [
        x for x in train_labels
        if any(x.split("/")[-2][-5:] in path for path in roi_results_filepaths)
    ]
    data_dicts = [
        {"image": img, "label": lbl}
        for img, lbl in zip(train_images, train_labels)
    ]

    # Transforms
    loader = LoadImaged(keys=("image", "label"), image_only=False)
    dataset = CacheDataset(data=data_dicts, transform=loader, cache_rate=0.0)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)

    ensure_channel_first = EnsureChannelFirstd(keys=["image", "label"])
    spacing_large = Spacingd(
        keys=["image", "label"], pixdim=(1.99, 1.99, 1.99),
        mode=("bilinear", "nearest"),
    )
    spacing_small = Spacingd(
        keys=["image", "label"], pixdim=(0.78, 0.78, 0.78),
        mode=("bilinear", "nearest"),
    )
    orientation = Orientationd(
        keys=["image", "label"], axcodes="PLI", allow_missing_keys=True,
    )
    dilation_filter = ImageFilter(filter="elliptical", filter_size=11)
    clean_objects = RemoveSmallObjects(
        min_size=50, connectivity=1, by_measure=False,
    )
    to_device = ToDeviced(keys=["image", "label"], device=device)

    # Crop transforms for each connected component
    crop_foregroundd = [
        CropForegroundd(
            keys=["image", "label", "roi"], source_key="roi",
            margin=(35, 35, 10), return_coords=True,
            allow_smaller=False, select_fn=lambda x, v=v: x == v,
        )
        for v in range(1, 6)
    ]

    roi_threshold = 0.1

    os.makedirs(output_path, exist_ok=True)

    t = tqdm.tqdm(
        enumerate(zip(roi_results_filepaths, data_loader)),
        total=len(roi_results_filepaths),
    )

    for i, (roi_result_path, data_dict_ori) in t:
        idx = roi_result_path.split("/")[-1][:-3]
        roi = torch.load(roi_result_path)
        with open(os.path.join(large_path, f"{idx}.pkl"), "rb") as fd:
            large_event = to_device(pkl.load(fd))

        image, label = large_event["image"], large_event["label"]
        coords = roi["coords"].long().to(device)
        logits = roi["logits"].max(dim=1)[0].to(device)
        roi_label = torch.zeros(label.shape).to(device)
        roi_label[0, coords[:, 0], coords[:, 1], coords[:, 2]] = (
            logits >= roi_threshold
        ).float()

        # Apply transforms to original data
        data_dict_ori = {
            "image": data_dict_ori["image"][0],
            "label": data_dict_ori["label"][0],
        }
        data_dict_ori = to_device(data_dict_ori)

        data_dict_cf = ensure_channel_first(data_dict_ori)
        data_dict_pli = orientation(data_dict_cf)
        data_dict_spa = spacing_large(data_dict_pli)

        # Clean and dilate ROI
        roi_label_clean = clean_objects(roi_label)
        roi_comp = label_components(roi_label_clean)
        roi_label_dilated = dilate_multiclass_labels(
            roi_comp, roi_comp.max(), dilation_filter,
        )
        data_dict_spa["label"][:] = roi_label_dilated[:]

        # Reverse transforms
        data_dict_invspa = spacing_large.inverse(data_dict_spa)
        data_dict_invpli = orientation.inverse(data_dict_invspa)

        # Evaluate ROI quality
        prediction, reference = data_dict_invpli["label"], data_dict_ori["label"]
        dice_val = dice(prediction, reference)
        eff_val = efficiency(prediction, reference)
        print(f"  {idx}: Dice={dice_val:.4f}, Efficiency={eff_val:.4f}")

        # Create high-resolution crops
        small_ori = spacing_small(data_dict_pli)
        small_roi = spacing_small(data_dict_invspa)
        small_ori["roi"] = small_roi["label"]

        nb_components = int(small_ori["roi"].max())
        for component in range(nb_components):
            small_ori_copy = copy.deepcopy(small_ori)
            small_ori_copy = crop_foregroundd[component](small_ori_copy)

            small_ori_copy["image"] = small_ori_copy["image"].to("cpu")
            small_ori_copy["label"] = small_ori_copy["label"].to("cpu")
            small_ori_copy["roi"] = small_ori_copy["roi"].to("cpu")

            if small_ori_copy["image"].numel() > 0:
                out_file = os.path.join(
                    output_path, f"{idx}_{component + 1}.pt"
                )
                torch.save(small_ori_copy, out_file)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
