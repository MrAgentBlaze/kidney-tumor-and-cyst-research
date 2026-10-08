import re
import numpy as np
import torch
import pickle as pkl
from glob import glob
from torch.utils.data import Dataset
from utils import sparsify, augment


class SparseDataset(Dataset):
    """Dataset for loading and sparsifying CT volumes."""

    def __init__(self, args):
        self.root = args.dataset_path.format(args.dataset_name)
        self.stage2 = args.stage2
        self.data_files = self.processed_file_names
        self.hu_range = (args.min_hu, args.max_hu)
        self.source_range = (-1, 1)
        self.training = False

        if self.stage2:
            self.data_files = [
                (path, int(path.split("_")[-1][:-3]))
                for path in self.data_files
            ]

    @property
    def processed_dir(self):
        return f"{self.root}"

    @property
    def processed_file_names(self):
        ext = "pt" if self.stage2 else "pkl"
        return self._natural_sort(
            set(glob("{0}*.{1}".format(self.processed_dir, ext)))
        )

    def _natural_sort(self, l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        return sorted(l, key=alphanum_key)

    def __len__(self):
        return len(self.data_files)

    def read_file(self, idx):
        path, roi_label = (
            self.data_files[idx] if self.stage2
            else (self.data_files[idx], -1)
        )

        if self.stage2:
            data = torch.load(path)
        else:
            with open(path, "rb") as fd:
                data = pkl.load(fd)

        case_idx = "".join(filter(str.isdigit, path.split("/")[-1].split("_")[0]))
        return case_idx, roi_label, data

    def __getitem__(self, idx):
        case_idx, roi_label, data = self.read_file(idx)

        if self.training:
            if np.random.rand() > 0.01:
                data = augment(data)

        # Sparsify
        image = data["image"].as_tensor()
        label = data["label"].as_tensor()
        roi = data["roi"].as_tensor() if "roi" in data else None
        c, x, y = sparsify(
            image, label, roi=roi, roi_label=roi_label,
            empty_min=self.hu_range[0], empty_max=self.hu_range[1],
        )

        print("Dense voxels:", image[0].numel())
        print("Sparse voxels:", c.shape[0])
        print("Sparsity:", c.shape[0] / image[0].numel())

        # Remap labels: tumour (2) is more exclusive than cyst (3)
        mask_tumour, mask_cyst = y == 2, y == 3
        y[mask_tumour] = 3
        y[mask_cyst] = 2

        # Create multi-label targets
        targets = torch.zeros(size=(y.shape[0], 3))
        targets[:, 0] = y[:, 0] > 0   # kidneys + masses
        targets[:, 1] = y[:, 0] > 1   # tumour + cyst
        targets[:, 2] = y[:, 0] == 3   # tumour only
        y = targets

        # Standardise
        if self.stage2:
            x = (x - 57.802067) / 81.260414
        else:
            x = (x - 48.10007) / 62.645897

        return {
            "idx": case_idx,
            "x": x.float(),
            "c": c.float(),
            "y": y.float(),
        }
