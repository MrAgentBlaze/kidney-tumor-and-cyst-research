
import glob
import os
import pickle

import torch
from utils import sparsify

DATA_DIR = "data/kits23_large_processed"
MIN_HU = -53.4
MAX_HU = 283.2
CANDIDATES = [
    "00516", "00130", "00279", "00425", "00088",
    "00497", "00109", "00268", "00478", "00257",
    "00546", "00041", "00529", "00160", "00560",
]

files = {
    os.path.basename(path).split(".")[0]: path
    for path in glob.glob(os.path.join(DATA_DIR, "*.pkl"))
}

results = []

for case_id in CANDIDATES:
    path = files.get(case_id)

    if path is None:
        print(f"Case {case_id}: FILE NOT FOUND")
        continue

    with open(path, "rb") as f:
        sample = pickle.load(f)

    image = sample["image"].as_tensor()
    label = sample["label"].as_tensor()

    with torch.no_grad():
        coords, features, labels = sparsify(
            image,
            label,
            empty_min=MIN_HU,
            empty_max=MAX_HU,
        )

    dense_voxels = image[0].numel()
    sparse_voxels = coords.shape[0]
    results.append((case_id, sparse_voxels, dense_voxels))

    print(
        f"Case {case_id}: "
        f"sparse={sparse_voxels:,}, "
        f"dense={dense_voxels:,}, "
        f"sparsity={sparse_voxels / dense_voxels:.2%}, "
        f"features={tuple(features.shape)}, "
        f"labels={tuple(labels.shape)}"
    )

print("\nSorted by sparse voxel count (smallest first):")
for case_id, sparse_voxels, dense_voxels in sorted(
    results, key=lambda row: row[1]
):
    print(
        f"{case_id}: {sparse_voxels:,} sparse voxels "
        f"({sparse_voxels / dense_voxels:.2%})"
    )
