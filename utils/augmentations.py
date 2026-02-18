import numpy as np
from monai.transforms import (
    Compose,
    RandAffined,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
)

# Affine transformation (rotation, scaling)
rand_affine = RandAffined(
    keys=["image", "label", "roi"],
    mode=("bilinear", "nearest", "nearest"),
    prob=0.9,
    rotate_range=(np.pi / 36, np.pi / 36, np.pi / 8),
    scale_range=(0.15, 0.15, 0.15),
    padding_mode="border",
    allow_missing_keys=True,
)

# Random flips along each axis
rand_flipx = RandFlipd(keys=["image", "label", "roi"], prob=0.3,
                        spatial_axis=0, allow_missing_keys=True)
rand_flipy = RandFlipd(keys=["image", "label", "roi"], prob=0.3,
                        spatial_axis=1, allow_missing_keys=True)
rand_flipz = RandFlipd(keys=["image", "label", "roi"], prob=0.3,
                        spatial_axis=2, allow_missing_keys=True)

# Intensity augmentations
rand_scale_intensity = RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3)
rand_shift_intensity = RandShiftIntensityd(keys=["image"], offsets=5, prob=0.3)

# Noise and smoothing
rand_noise = RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=1)
rand_blur = RandGaussianSmoothd(
    keys=["image"], prob=0.3,
    sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5),
)

# Full augmentation pipeline
transformations = Compose([
    rand_affine,
    rand_flipx,
    rand_flipy,
    rand_flipz,
    rand_scale_intensity,
    rand_shift_intensity,
    rand_noise,
    rand_blur,
])


def augment(event):
    """Apply data augmentation transforms to an event dictionary."""
    return transformations(event)
