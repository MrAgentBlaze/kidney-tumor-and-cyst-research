import torch
from utils.funcs import argsort_sparse_tensor

N = 5_120_278
print("Device: CPU")
print(f"Coordinate rows: {N:,}")

# Synthetic coordinates with a batch column and three spatial columns.
coords = torch.empty((N, 4), dtype=torch.int32)
coords[:, 0] = 0
coords[:, 1] = torch.randint(0, 245, (N,), dtype=torch.int32)
coords[:, 2] = torch.randint(0, 245, (N,), dtype=torch.int32)
coords[:, 3] = torch.randint(0, 722, (N,), dtype=torch.int32)

class SparseLike:
    def __init__(self, coordinates):
        self.coordinates = coordinates

tensor = SparseLike(coords)

print("Input coordinate bytes:", coords.numel() * coords.element_size())
indices = argsort_sparse_tensor(tensor)

print("Index dtype:", indices.dtype)
print("Index count:", indices.numel())
print("Index bytes:", indices.numel() * indices.element_size())
print("Sorting completed successfully.")
