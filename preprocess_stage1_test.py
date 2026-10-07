import os
import pickle as pkl

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
)

CASE_ID = "case_00002"

SOURCE_DIR = "/mnt/d/Brian Lala/Research/2-step-segmentation-kidney-tumor-and-cyst/data/KiTS23"
OUTPUT_DIR = "data/kits23_large_processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = {
    "image": os.path.join(SOURCE_DIR, CASE_ID, "imaging.nii.gz"),
    "label": os.path.join(SOURCE_DIR, CASE_ID, "segmentation.nii.gz"),
}

data = LoadImaged(
    keys=("image", "label"),
    image_only=False,
)(data)

data = EnsureChannelFirstd(
    keys=["image", "label"],
)(data)

data = Orientationd(
    keys=["image", "label"],
    axcodes="PLI",
    allow_missing_keys=True,
)(data)

data = Spacingd(
    keys=["image", "label"],
    pixdim=(1.99, 1.99, 1.99),
    mode=("bilinear", "nearest"),
)(data)

output_path = os.path.join(OUTPUT_DIR, f"{CASE_ID}.pkl")

with open(output_path, "wb") as fd:
    pkl.dump(data, fd)

print("Saved:", output_path)
print("Image:", type(data["image"]).__name__, data["image"].shape)
print("Label:", type(data["label"]).__name__, data["label"].shape)
