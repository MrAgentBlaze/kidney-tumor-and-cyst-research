import numpy as np
import matplotlib.pyplot as plt
import time
import torch

def print_slice(c, x, y, slice=0):
    """Plot a specified slice of coordinates, intensities, and labels."""
    
    # Convert to numpy arrays if inputs are PyTorch tensors
    if not isinstance(c, np.ndarray):
        c = c.numpy()
        x = x.numpy()
        y = y.numpy()

    # Calculate min and max for coordinates
    mins, maxs = c.min(axis=0), c.max(axis=0)
    
    # Define plot grid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    t0 = time.time()

    # Select coordinates for the specified slice
    indexes = c[:, 2] == slice
    c_x = c[indexes]
    x_x = x[indexes]
    y_x = y[indexes]
    trues = c_x[y_x.astype(bool).reshape(-1,)]
    
    t1 = time.time()
    print(f"time: {t1 - t0:.2f} s")

    # Define axis limits and titles
    ax1.set_xlim(mins[0], maxs[0])
    ax1.set_ylim(maxs[1], mins[1])
    ax2.set_xlim(mins[0], maxs[0])
    ax2.set_ylim(maxs[1], mins[1])

    # Plot background and true labels
    ax1.scatter(c_x[:, 0], c_x[:, 1], s=1, facecolors='none', edgecolors='lime', marker='o', alpha=0.5, zorder=0, label="background")
    ax1.scatter(trues[:, 0], trues[:, 1], c='red', marker='o', s=1, alpha=1.0, zorder=5, label="true kidney/cyst/tumor")
    ax1.legend()

    # Plot intensities
    ax2.scatter(c_x[:, 0], c_x[:, 1], s=1, c=x_x, marker='o', alpha=0.5, zorder=0, label="intensity")

    plt.show()

def print_slice2(c, x, y, slice, height, width, nslices):
    if not isinstance(c, np.ndarray):
        c = c.numpy().astype(int)
        x = x.numpy()
        y = y.numpy()

    vol = np.full(shape=(height, width, nslices), fill_value=-50.)
    seg = np.full(shape=(height, width, nslices), fill_value=-50.)

    vol[c[:, 0], c[:, 1], c[:, 2]] = x[:]
    seg[c[:, 0], c[:, 1], c[:, 2]] = y[:]

    vol_slice = vol[:, :, slice]
    seg_slice = seg[:, :, slice]

    vol_slice[vol_slice<-30] = np.nan
    seg_slice[seg_slice<-30] = np.nan
    
    # define grid
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(10)

    t0 = time.time()
    
    t1 = time.time()
    print("time: {} s".format(t1-t0))

    # define lims and subplot titles
    ax1.set_xlim(0, height)
    ax1.set_ylim(width, 0)
    ax2.set_xlim(0, height)
    ax2.set_ylim(width, 0)

    # plot
    ax1.imshow(vol_slice.T, vmin=-1, vmax=1, alpha=1.0, cmap="Blues",)
    ax2.imshow(seg_slice.T, vmin=0, vmax=3, alpha=1.0, cmap="hsv",)

    plt.show()

# collate function
def collate_sparse_minkowski(batch):
    coords = [d['c'].int() for d in batch]
    feats = torch.cat([d['x'] for d in batch])
    y = torch.cat([d['y'] for d in batch])

    kidney_tumor_cyst1 = torch.cat([d['kidney_tumor_cyst1'] for d in batch])
    kidney_tumor_cyst2 = torch.cat([d['kidney_tumor_cyst2'] for d in batch])
    kidney_tumor_cyst4 = torch.cat([d['kidney_tumor_cyst4'] for d in batch])
    kidney_tumor_cyst8 = torch.cat([d['kidney_tumor_cyst8'] for d in batch])
    tumor_cyst1 = torch.cat([d['tumor_cyst1'] for d in batch])
    tumor_cyst2 = torch.cat([d['tumor_cyst2'] for d in batch])
    tumor_cyst4 = torch.cat([d['tumor_cyst4'] for d in batch])
    tumor_cyst8 = torch.cat([d['tumor_cyst8'] for d in batch])
    tumor_only1 = torch.cat([d['tumor_only1'] for d in batch])
    tumor_only2 = torch.cat([d['tumor_only2'] for d in batch])
    tumor_only4 = torch.cat([d['tumor_only4'] for d in batch])
    tumor_only8 = torch.cat([d['tumor_only8'] for d in batch])
    
    ret = { 'f': feats, 'c': coords, 'y': y,
            'kidney_tumor_cyst1': kidney_tumor_cyst1, 'tumor_cyst1': tumor_cyst1, 'tumor_only1': tumor_only1,
            'kidney_tumor_cyst2': kidney_tumor_cyst2, 'tumor_cyst2': tumor_cyst2, 'tumor_only2': tumor_only2,
            'kidney_tumor_cyst4': kidney_tumor_cyst4, 'tumor_cyst4': tumor_cyst4, 'tumor_only4': tumor_only4,
            'kidney_tumor_cyst8': kidney_tumor_cyst8, 'tumor_cyst8': tumor_cyst8, 'tumor_only8': tumor_only8,
          }
    return ret
