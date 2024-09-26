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
        self.root = args.dataset_path.format(args.dataset_name, int(args.min_hu), int(args.max_hu))
        self.data_files = self.processed_file_names
        self.train = False
        self.total_events = self.__len__
        self.hu_range = (args.min_hu, args.max_hu)
        self.source_range = (-1, 1)
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
        feats = self._shift_hu_gaussian(feats, std_dev=0.05)

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
        if (angle_limits==0).all():
            # no rotation at all
            return coords
        return random_rotation_saul(coords=coords,
                                    angle_limits=angle_limits,
                                    origin=first_point)
 

    def _translate(self, coords):
        shift_x, shift_y = np.random.randint(low=-10, high=10, size=(2,))
        coords[:, 0] += shift_x
        coords[:, 1] += shift_y
        return coords


    def _drop(self, coords, feats, labels, p=0.1):
        mask = np.random.rand(coords.shape[0]) > p
        #don't drop all coordinates
        if mask.sum() == 0:
            return coords, feats
        return coords[mask], feats[mask], labels[mask]


    def _shift_hu_uniform(self, feats, max_scale_factor=0.1):
        shift = 1 - np.random.rand(*feats.shape) * max_scale_factor
        return feats * shift


    def _shift_hu_gaussian(self, feats, std_dev=0.1):
        shift = 1 - np.random.randn(*feats.shape) * std_dev
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
 
        def trilinear_interpolation(voxels, feats, labels):
            """
            Perform vectorized trilinear interpolation on the rotated voxel coordinates
            and distribute the original features to surrounding voxels.

            Parameters:
            - voxels: an (N, 3) array of rotated (non-integer) voxel coordinates.
            - features: an (N, 1) array of features corresponding to the original voxels.
            - labels: an (N, 1) array of labels corresponding to the original voxels.
            Returns:
            - voxel_coords: a (M, 3) array of voxel coordinates (integer values).
            - aggregated_features: a (M,) array of corresponding interpolated feature values.
            - aggregated_labels: a (M,) array of corresponding interpolated label values.
            """
            offset = -1000 - self.hu_range[0]  # offset using min possible H.U. value (air)
            features = feats - offset

            lower_voxel_corners = np.floor(voxels).astype(int)  # Integer part (x0, y0, z0)
            fractional_part = voxels - lower_voxel_corners  # Fractional part (distances from lower corner)

            # Compute the weights for the 8 corners based on the fractional distances
            offsets = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
            xd, yd, zd = fractional_part.T
            weights = np.array([
                (1 - xd) * (1 - yd),  # (x0, y0)
                (1 - xd) * yd,        # (x0, y1)
                xd * (1 - yd),        # (x1, y0)
                xd * yd               # (x1, y1)
            ]).T  # Shape: (N, 4)

            # Distribute original features to the surrounding corners using the weights
            weighted_features = features * weights  # Shape: (N, 4)
            voxel_coords = lower_voxel_corners[:, np.newaxis, :] + offsets  # Shape: (N, 4, 3)
            voxel_coords = voxel_coords.reshape(-1, 3)  # Shape: (N*4, 3)
            features = weighted_features.flatten()  # Shape: (N*4,)
            labels = np.tile(labels, 4).flatten()
            weights = weights.flatten() 

            # Aggregate contributions to each voxel using unique voxel coordinates
            unique_coords, idx = np.unique(voxel_coords, axis=0, return_inverse=True)
            aggregated_features = np.bincount(idx, weights=features, minlength=len(unique_coords))
            aggregated_labels = np.full(len(unique_coords), -np.inf)
            np.maximum.at(aggregated_labels, idx, labels)
            aggregated_weights = np.bincount(idx, weights=weights, minlength=len(unique_coords))

            # Mask out the voxels that aren't sufficiently filled
            valid_mask = aggregated_weights > 0.5  # only > half-filled voxels are kept
            unique_coords = unique_coords[valid_mask]
            aggregated_features = aggregated_features[valid_mask] / aggregated_weights[valid_mask]
            aggregated_labels = aggregated_labels[valid_mask]
           
            # Add back the offset and filter based on the HU range
            aggregated_features += offset
            hu_mask = (aggregated_features >= self.hu_range[0] - 50) & (aggregated_features <= self.hu_range[1] + 50)
            unique_coords = unique_coords[hu_mask]
            aggregated_features = aggregated_features[hu_mask].reshape(-1, 1)
            aggregated_labels = aggregated_labels[hu_mask].reshape(-1, 1).round()

            return unique_coords, aggregated_features, aggregated_labels
        
        # Load data
        data = np.load(self.data_files[idx])

        # Extract the index from the file name
        idx = ''.join(char for char in self.data_files[idx].split("/")[-1] if char.isdigit())
        
        # Extract data fields
        c = data['c'].copy()   # contiguous
        x = data['x']
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
            if np.random.rand() > 0.05:
                # Augment 95% of the times during training
                c, x, y = self._augment(c, x, y, height, width, nslices)
                c, x, y = trilinear_interpolation(c, x, y)

        x = np.interp(x.ravel(), self.hu_range, self.source_range).reshape(x.shape)
        targets = np.zeros(shape=(y.shape[0], 3))
        targets[:, 0] = y[:, 0] > 0  # kidney_tumor_cyst 
        targets[:, 1] = y[:, 0] > 1  # tumor_cyst
        targets[:, 2] = y[:, 0] == 3  # tumor_only
        y = targets

        # Convert to torch tensors
        c = torch.from_numpy(c).float()
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
 
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

