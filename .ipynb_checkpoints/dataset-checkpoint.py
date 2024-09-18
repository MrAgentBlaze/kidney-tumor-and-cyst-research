# Imports
import os
import re
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import random

# Define pixel and signal ranges
ranges = [-30, 350]  # Max pixels 64, signal loss 2.08
source_range = [-1, 1]

def natural_sort(l):
    """Sort a list in a human-readable way."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def valid_coordinates(coordinates, height, width, slices):
    """Find indices of valid coordinates within bounds."""
    valid_indices = np.where(
        (0 <= coordinates[:, 0]) & (coordinates[:, 0] < height) &
        (0 <= coordinates[:, 1]) & (coordinates[:, 1] < width) &
        (0 <= coordinates[:, 2]) & (coordinates[:, 2] < slices)
    )[0]
    return valid_indices

def rotate_coordinates(sparse_coordinates, radians=5):
    """Rotate coordinates in the XY plane."""
    rotation_angle = np.radians(radians)
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                                [0, 0, 1]])
    rotated_coordinates = np.dot(sparse_coordinates[:, :2], rotation_matrix[:2, :2].T)
    rotated_coordinates = np.column_stack((rotated_coordinates, sparse_coordinates[:, 2]))
    return np.round(rotated_coordinates).astype(int)

def translate_coordinates(coordinates, translation):
    """
    Translate sparse 3D coordinates by the specified translation in both x and y directions.
    """
    tx, ty = translation
    translated_coordinates = coordinates.copy()
    translated_coordinates[:, 0] += tx
    translated_coordinates[:, 1] += ty
    return translated_coordinates

class SparseEvent(Dataset):
    def __init__(self, root, shuffle=False):
        """Initialize the SparseEvent dataset."""
        self.root = root
        self.data_files = self.processed_file_names
        self.train = False
        if shuffle:
            random.shuffle(self.data_files)
        self.total_events = self.__len__()

    @property
    def processed_dir(self):
        return f'{self.root}'

    @property
    def processed_file_names(self):
        return natural_sort(set(glob(f'{self.processed_dir}*.npz')))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        idx = ''.join(char for char in self.data_files[idx].split("/")[-1] if char.isdigit())
        
        c = data['c']
        x = data['x']
        y = data['y']
        height = int(data['height'])
        width = int(data['width'])
        nslices = int(data['nslices'])

        # Data augmentation
        if self.train:
            noise = np.random.normal(0, 1, size=x.shape)
            x += noise

            radians = np.random.randint(-5, 5)
            c = rotate_coordinates(c, radians=radians)
            
            translation = np.random.randint(-20, 20, 2)
            c = translate_coordinates(c, translation)

            valid_indexes = valid_coordinates(c, height, width, nslices)
            c = c[valid_indexes]
            x = x[valid_indexes]
            y = y[valid_indexes]

            if random.random() > 0.5:
                c[:, 0] = width - c[:, 0]

            c, x, y = ME.utils.sparse_quantize(coordinates=c, features=x.reshape(-1, 1), labels=y)

        # Normalise input
        x = np.interp(x.ravel(), ranges, source_range).reshape(x.shape)

        _, y2 = ME.utils.sparse_quantize(coordinates=c, features=None, labels=y, quantization_size=2)
        _, y4 = ME.utils.sparse_quantize(coordinates=c, features=None, labels=y, quantization_size=4)
        _, y8 = ME.utils.sparse_quantize(coordinates=c, features=None, labels=y, quantization_size=8)

        # Create binary masks for different scales
        kidney_tumor_cyst1, tumor_cyst1, tumor_only1 = y > 0, y > 1, y == 2
        kidney_tumor_cyst2, tumor_cyst2, tumor_only2 = y2 > 0, y2 > 1, y2 == 2
        kidney_tumor_cyst4, tumor_cyst4, tumor_only4 = y4 > 0, y4 > 1, y4 == 2
        kidney_tumor_cyst8, tumor_cyst8, tumor_only8 = y8 > 0, y8 > 1, y8 == 2

        c = torch.FloatTensor(c)
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)
        kidney_tumor_cyst1 = torch.FloatTensor(kidney_tumor_cyst1)
        kidney_tumor_cyst2 = torch.FloatTensor(kidney_tumor_cyst2)
        kidney_tumor_cyst4 = torch.FloatTensor(kidney_tumor_cyst4)
        kidney_tumor_cyst8 = torch.FloatTensor(kidney_tumor_cyst8)
        tumor_cyst1 = torch.FloatTensor(tumor_cyst1)
        tumor_cyst2 = torch.FloatTensor(tumor_cyst2)
        tumor_cyst4 = torch.FloatTensor(tumor_cyst4)
        tumor_cyst8 = torch.FloatTensor(tumor_cyst8)
        tumor_only1 = torch.FloatTensor(tumor_only1)
        tumor_only2 = torch.FloatTensor(tumor_only2)
        tumor_only4 = torch.FloatTensor(tumor_only4)
        tumor_only8 = torch.FloatTensor(tumor_only8)

        return {
            'idx': idx, 'x': x, 'c': c, 'y': y, 'width': width, 'height': height, 'nslices': nslices,
            'kidney_tumor_cyst1': kidney_tumor_cyst1, 'tumor_cyst1': tumor_cyst1, 'tumor_only1': tumor_only1,
            'kidney_tumor_cyst2': kidney_tumor_cyst2, 'tumor_cyst2': tumor_cyst2, 'tumor_only2': tumor_only2,
            'kidney_tumor_cyst4': kidney_tumor_cyst4, 'tumor_cyst4': tumor_cyst4, 'tumor_only4': tumor_only4,
            'kidney_tumor_cyst8': kidney_tumor_cyst8, 'tumor_cyst8': tumor_cyst8, 'tumor_only8': tumor_only8,
        }
