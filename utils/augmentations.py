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

# Affine transformation
rand_affine = RandAffined(
    keys=["image", "label"],
    mode=("bilinear", "nearest"),
    prob=1.0,
    #spatial_size=(300, 300, 50),
    translate_range=(20, 20, 2),
    rotate_range=(np.pi / 36, np.pi / 36, np.pi / 4),  # Reduced z-rotation
    scale_range=(0.15, 0.15, 0.15),  # Slightly reduced scaling
    padding_mode="border",
)

# Flip in all axes
rand_flip = RandFlipd(
    keys=["image", "label"],
    prob=0.5,
    spatial_axis=[0, 1, 2],
)

# Intensity scaling
rand_scale_intensity = RandScaleIntensityd(
    keys=["image"],
    factors=0.1,  # Scale intensity by a factor in [1-factors, 1+factors]
    prob=0.5,
)

# Intensity shifting
rand_shift_intensity = RandShiftIntensityd(
    keys=["image"],
    offsets=0.1,
    prob=0.5,
)

# Add Gaussian noise
rand_noise = RandGaussianNoised(
    keys=["image"],
    prob=0.5,
    mean=0.0,
    std=0.1,
)

# Gaussian smoothing (blurring)
rand_blur = RandGaussianSmoothd(
    keys=["image"],
    prob=0.5,
    sigma_x=(0.5, 1.5),
    sigma_y=(0.5, 1.5),
    sigma_z=(0.5, 1.5),
)

# Compose all transformations into one pipeline
transform = Compose([
    rand_affine,
    rand_flip,
    rand_scale_intensity,
    rand_shift_intensity,
    rand_noise,
    rand_blur,
])


def augment(event):
    return transform(event)

