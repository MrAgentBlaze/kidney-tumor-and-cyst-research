import re
import numpy as np
import torch
import pickle as pkl
from glob import glob
from torch.utils.data import Dataset
from utils import sparsify, augment


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


class SparseDataset(Dataset):
    def __init__(self, args):
        self.root = args.dataset_path.format(args.dataset_name)
        self.stage2 = args.stage2
        self.data_files = self.processed_file_names
        self.train = False
        self.total_events = self.__len__
        self.hu_range = (args.min_hu, args.max_hu)
        self.source_range = (-1, 1)
        self.training = False
        self.contrastive = args.contrastive 
        self.roi = args.roi

        if self.stage2:
            self.data_files = [(path, int(path.split("_")[-1][:-3])) for path in self.data_files]


    def set_training_mode(self, training=True):
        """Sets the split type dynamically."""
        print("Setting training mode to {}.".format(training))
        self.training = training


    @property
    def processed_dir(self):
        return f'{self.root}'
   

    @property
    def processed_file_names(self):
        return self._natural_sort(set(glob('{0}*.{1}'.format(self.processed_dir, 'pt' if self.stage2 else 'pkl'))))               
   

    def _natural_sort(self, l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key) 


    def __len__(self):
        return len(self.data_files)


    def __getitem__(self, idx):
        if self.stage2:
            path, roi_label = self.data_files[idx]
            data = torch.load(path)
        else:
            path, roi_label = self.data_files[idx], -1
            with open(path, 'rb') as fd:
                data = pkl.load(fd)

        idx = ''.join(char for char in path.split("/")[-1] if char.isdigit())

        if self.training:
            # augment
            if np.random.rand() > 0.01:
                data = augment(data)

        # sparsify
        image, label, roi = data["image"].as_tensor(), data["label"].as_tensor(), data["roi"].as_tensor() if "roi" in data else None
        mask = sparsify(image, label, roi=roi, roi_label=roi_label, empty_min=self.hu_range[0], empty_max=self.hu_range[1], return_mask=True) 
        #return image+1024, label, mask 

        c, x, y = sparsify(image, label, roi=roi, roi_label=roi_label, empty_min=self.hu_range[0], empty_max=self.hu_range[1])
        #return image, label, roi

        # rename labels (tumor==2 should be more exclusive than cyst==3 for the later labels)
        mask_tumor, mask_cyst = y == 2, y == 3
        y[mask_tumor] = 3
        y[mask_cyst] = 2

        if not self.contrastive:
            targets = torch.zeros(size=(y.shape[0], 3))
            targets[:, 0] = y[:, 0] > 0  # kidney_tumor_cyst 
            targets[:, 1] = y[:, 0] > 1  # tumor_cyst
            targets[:, 2] = y[:, 0] == 3  # tumor_only
            y = targets

            if self.roi:
                y = y[:, 0].reshape(-1, 1)

        # standardise
        if self.stage2:
            x = (x - 57.802067) / 81.260414
        else:
            x = (x - 48.10007) / 62.645897

        c = c.float()
        x = x.float()
        y = y.float()
 
        result = {
            'idx': idx,
            'x': x,
            'c': c,
            'y': y,
        }

        del data
        return result
        
