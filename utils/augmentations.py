import numpy as np
from monai.transforms import (
    Compose,
    RandAffined,
    Rand3DElasticd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
)

# Affine transformation
rand_affine = RandAffined(
    keys=["image", "label"],
    mode=("bilinear", "nearest"),
    prob=0.9,
    #spatial_size=(300, 300, 500),
    #shear_range=(0.05, 0.05, 0.05),
    translate_range=(30, 30, 5),
    rotate_range=(np.pi / 36, np.pi / 36, np.pi / 8),
    scale_range=(0.15, 0.15, 0.15),
    padding_mode="border",
)

# Elastic deformation
rand_elastic = Rand3DElasticd(
    keys=["image", "label"],
    prob=0.2,
    sigma_range=(5, 8),
    magnitude_range=(100, 200),
    mode=("bilinear", "nearest"),
    padding_mode="border",
)

# Flip in all axes
rand_flipx = RandFlipd(
    keys=["image", "label"],
    prob=0.3,
    spatial_axis=0,
)

rand_flipy = RandFlipd(
    keys=["image", "label"],
    prob=0.3,
    spatial_axis=1,
)

rand_flipz = RandFlipd(
    keys=["image", "label"],
    prob=0.3,
    spatial_axis=2,
)

# Intensity scaling
rand_scale_intensity = RandScaleIntensityd(
    keys=["image"],
    factors=0.1,
    prob=0.3,
)

# Intensity shifting
rand_shift_intensity = RandShiftIntensityd(
    keys=["image"],
    offsets=5,
    prob=0.3,
)

# Add Gaussian noise
rand_noise = RandGaussianNoised(
    keys=["image"],
    prob=0.3,
    mean=0.0,
    std=1,
)

# Gaussian smoothing (blurring)
rand_blur = RandGaussianSmoothd(
    keys=["image"],
    prob=0.3,
    sigma_x=(0.25, 1.5),
    sigma_y=(0.25, 1.5),
    sigma_z=(0.25, 1.5),
)

# Pipeline of transformations
transformations = Compose([
    rand_affine,
    #rand_elastic,
    rand_flipx,
    rand_flipy,
    rand_flipz,
    rand_scale_intensity,
    rand_shift_intensity,
    rand_noise,
    rand_blur,
])


def augment(event):
    event_aug = transformations(event)
    return event_aug

