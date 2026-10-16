import gc
import pickle
import torch
from pathlib import Path
from utils import augment

CASE_PATH = Path("data/kits23_large_processed/00223.pkl")
HU_MIN, HU_MAX = -53.4, 283.2

print("Loading case 00223 on CPU...", flush=True)
with CASE_PATH.open("rb") as f:
    data = pickle.load(f)

image = data["image"].as_tensor().cpu()
print(f"Image shape: {tuple(image.shape)}", flush=True)
print(f"Image dtype: {image.dtype}", flush=True)

def count_active(img):
    return int(((img >= HU_MIN) & (img <= HU_MAX)).sum().item())

original_count = count_active(image)
total_voxels = image[0].numel()
print(f"Original active voxels: {original_count:,}", flush=True)
print(f"Total voxels: {total_voxels:,}", flush=True)
print(f"Original active fraction: {original_count / total_voxels:.4%}", flush=True)

del image
gc.collect()

print("\nApplying one augmentation on CPU...", flush=True)
augmented = augment(data)
aug_image = augmented["image"].as_tensor().cpu()
augmented_count = count_active(aug_image)

print(f"Augmented active voxels: {augmented_count:,}", flush=True)
print(
    f"Change: {augmented_count - original_count:+,} voxels "
    f"({(augmented_count - original_count) / original_count:+.2%})",
    flush=True,
)
print("CPU augmentation diagnostic completed.", flush=True)
