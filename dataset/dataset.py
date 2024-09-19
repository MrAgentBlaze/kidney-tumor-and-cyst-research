import re
import torch
import numpy as np
import MinkowskiEngine as ME
from glob import glob
from torch.utils.data import Dataset
from utils import random_rotation_saul

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


class SparseDataset(Dataset):
    def __init__(self, args):
        '''Initialiser for SparseEventProtoDUNE class'''
        
        self.root = args.dataset_path.format(args.dataset_name, args.min_hu, args.max_hu)
        self.data_files = self.processed_file_names
        self.train = False
        self.total_events = self.__len__
        self.hu_range = (args.min_hu, args.max_hu)
        self.source_range = (0, 1)
        self.training = False
   

    def set_training_mode(self, training=True):
        """Sets the split type dynamically."""
        print("Setting training mode to {}.".format(training))
        self.training = training


    def _augment(self, coords, feats, labels, height, width, nslices):
        # rotate
        coords = self._rotate(coords, height, width)
 
        # translate
        coords = self._translate(coords)
       
        # drop voxels
        #coords, feats, labels = self._drop(coords, feats, labels, p=0.1)

        # shift feature values
        feats = self._shift_hu(feats, max_scale_factor=0.1)

        # keep within limits
        coords, feats, labels = self._within_limits(coords, feats, labels, height, width, nslices)

        return coords, feats, labels


    def _rotate(self, coords, height, width):
        """Random rotation along Z axis"""
        first_point = np.array([height//2, width//2,  0])
        angle_limits = torch.tensor([
            [0, 0, -torch.pi/8],  # Min angles for X, Y, Z
            [0, 0,  torch.pi/8]   # Max angles for X, Y, Z
        ])
        return random_rotation_saul(coords=coords,
                                    angle_limits=angle_limits,
                                    origin=first_point)
 

    def _translate(self, coords):
        shift_x, shift_y = np.random.randint(low=-10, high=10, size=(2,))
        coords[:, 0] += shift_x
        coords[:, 1] += shift_y
        return coords


    def _drop(self, coords, feats, labels, p=0.1):
        mask = torch.rand(coords.shape[0]) > p
        #don't drop all coordinates
        if mask.sum() == 0:
            return coords, feats
        return coords[mask], feats[mask], labels[mask]


    def _shift_hu(self, feats, max_scale_factor=0.1):
        shift = 1 - np.random.rand(*feats.shape) * max_scale_factor
        return feats * shift


    def _within_limits(self, coords, feats, labels, height, width, nslices):
        mask = (coords[:, 0] >= 0) & (coords[:, 0] < height) & \
           (coords[:, 1] >= 0) & (coords[:, 1] < width) & \
           (coords[:, 2] >= 0) & (coords[:, 2] < nslices)
        return coords[mask], feats[mask], labels[mask]

       
    @property
    def processed_dir(self):
        return f'{self.root}'
    
    @property
    def processed_file_names(self):
        return self._natural_sort(set(glob(f'{self.processed_dir}*.npz')))               
   
    def _natural_sort(self, l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key) 

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        #print("idx: {}".format(idx))
        def quantize(c, x, y, quantization_size=1):
            quant_c, unique_indices, inverse_indices = ME.utils.sparse_quantize(
                coordinates=c,
                return_index=True,
                return_inverse=True,
                quantization_size=quantization_size,
            )

            def max_pool(features, shape, dtype):
                pooled = np.zeros(shape, dtype=dtype)
                np.maximum.at(pooled, inverse_indices, features)
                return pooled

            # Apply avg pooling if x is not None
            quant_x = max_pool(x, (len(quant_c), x.shape[1]), x.dtype)

            # Apply max pooling for y
            quant_y = max_pool(y, (len(quant_c), y.shape[1]), y.dtype)

            return quant_c, quant_x, quant_y

        # Load data
        data = np.load(self.data_files[idx])

        # Extract the index from the file name
        idx = ''.join(char for char in self.data_files[idx].split("/")[-1] if char.isdigit())
        
        # Extract data fields
        c = data['c'].copy()   # contiguous
        x = np.interp(data['x'].ravel(), self.hu_range, self.source_range).reshape(data['x'].shape)
        y = data['y']  # 0: background, 1: kidney, 2: tumor, 3: cyst
        height = int(data['height'])
        width = int(data['width'])
        nslices = int(data['nslices'])

        # Rename labels (tumor==2 should be exclusive than cyst==3 for the later labels)
        mask_tumor, mask_cyst = y == 2, y == 3
        y[mask_tumor] = 3
        y[mask_cyst] = 2

        # Random rotate if training
        if self.training:
            c, x, y = self._augment(c, x, y, height, width, nslices)

        # Quantise duplicated coordinates
        c, x, y = quantize(c, x, y)

        # Convert to torch tensors
        c = c.float()
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)   
 
        # Create the return dictionary
        result = {
            'idx': idx,
            'x': x,
            'c': c,
            'y': y,
            'width': width,
            'height': height,
            'nslices': nslices,
        }
        
        del data
        return result

