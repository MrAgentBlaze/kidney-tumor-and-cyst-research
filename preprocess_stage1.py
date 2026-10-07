import os
import pickle as pkl

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
)
from tqdm import tqdm


SOURCE_DIR = "/mnt/d/Brian Lala/Research/2-step-segmentation-kidney-tumor-and-cyst/data/KiTS23"
OUTPUT_DIR = "data/kits23_large_processed"


def preprocess_case(case_id, loader, channel_first, orientation, spacing):
    data = {
        "image": os.path.join(SOURCE_DIR, case_id, "imaging.nii.gz"),
        "label": os.path.join(SOURCE_DIR, case_id, "segmentation.nii.gz"),
    }

    data = loader(data)
    data = channel_first(data)
    data = orientation(data)
    data = spacing(data)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{case_id.replace('case_', '')}.pkl",
    )

    with open(output_path, "wb") as fd:
        pkl.dump(data, fd)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    loader = LoadImaged(
        keys=("image", "label"),
        image_only=False,
    )

    channel_first = EnsureChannelFirstd(
        keys=["image", "label"],
    )

    orientation = Orientationd(
        keys=["image", "label"],
        axcodes="PLI",
        allow_missing_keys=True,
    )

    spacing = Spacingd(
        keys=["image", "label"],
        pixdim=(1.99, 1.99, 1.99),
        mode=("bilinear", "nearest"),
    )

    case_ids = [
        name
        for name in sorted(os.listdir(SOURCE_DIR))
        if name.startswith("case_")
        and os.path.isdir(os.path.join(SOURCE_DIR, name))
        and os.path.exists(os.path.join(SOURCE_DIR, name, "imaging.nii.gz"))
        and os.path.exists(os.path.join(SOURCE_DIR, name, "segmentation.nii.gz"))
    ]

    print(f"Found {len(case_ids)} KiTS23 cases.")
    print(f"Output directory: {OUTPUT_DIR}")

    for case_id in tqdm(case_ids, desc="Stage 1 preprocessing"):
        output_path = os.path.join(
            OUTPUT_DIR,
            f"{case_id.replace('case_', '')}.pkl",
        )

        if os.path.exists(output_path):
            continue

        preprocess_case(
            case_id,
            loader,
            channel_first,
            orientation,
            spacing,
        )

    print("Stage 1 preprocessing complete.")


if __name__ == "__main__":
    main()
