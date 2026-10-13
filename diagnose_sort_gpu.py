import gc
import torch
from utils.funcs import argsort_sparse_tensor

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

device = torch.device("cuda")
N = 5_120_278

def report(label):
    torch.cuda.synchronize()
    print(
        f"{label}: "
        f"allocated={torch.cuda.memory_allocated() / 1024**2:.1f} MiB, "
        f"peak={torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB, "
        f"free/total={torch.cuda.mem_get_info()[0] / 1024**2:.1f}/"
        f"{torch.cuda.mem_get_info()[1] / 1024**2:.1f} MiB",
        flush=True,
    )

print("GPU:", torch.cuda.get_device_name(0))
report("Before test")

coords = torch.empty((N, 4), dtype=torch.int32, device=device)
coords[:, 0] = 0
coords[:, 1] = torch.randint(0, 245, (N,), dtype=torch.int32, device=device)
coords[:, 2] = torch.randint(0, 245, (N,), dtype=torch.int32, device=device)
coords[:, 3] = torch.randint(0, 722, (N,), dtype=torch.int32, device=device)

class SparseLike:
    def __init__(self, coordinates):
        self.coordinates = coordinates

tensor = SparseLike(coords)

torch.cuda.reset_peak_memory_stats()
report("Before sorting")

indices = argsort_sparse_tensor(tensor)
report("After sorting")

print("Sorted indices:", indices.numel())
print("Sorting completed successfully.")

del indices, tensor, coords
gc.collect()
torch.cuda.empty_cache()
report("After cleanup")
