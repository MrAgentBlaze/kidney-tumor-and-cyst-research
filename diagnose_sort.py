import os
import pickle
import numpy as np

path = os.path.join("data", "kits23_large_processed", "00223.pkl")

with open(path, "rb") as f:
    data = pickle.load(f)

for key in ("image", "label"):
    arr = data[key]
    print(f"\n{key}:")
    print("  type:", type(arr))
    print("  shape:", getattr(arr, "shape", None))
    print("  dtype:", getattr(arr, "dtype", None))

    if isinstance(arr, np.ndarray):
        print("  min:", arr.min())
        print("  max:", arr.max())