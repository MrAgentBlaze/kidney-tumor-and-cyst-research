import torch

# Simulate 3D sparse coordinates: batch index + x, y, z.
coords = torch.tensor([
    [0, 244, 244, 721],
    [0,   0,   0,   0],
    [0, 100, 200, 300],
    [0, 244, 244, 720],
    [0,   1,   0,   0],
], dtype=torch.int32)

max_val = coords.max() + 1

multipliers = torch.tensor(
    [max_val ** i for i in reversed(range(coords.shape[1]))],
    device=coords.device,
)

encoded_coords = (coords * multipliers).sum(dim=1)
indices = torch.argsort(encoded_coords)

print("Coordinate dtype:", coords.dtype)
print("Multiplier dtype:", multipliers.dtype)
print("Encoded-key dtype:", encoded_coords.dtype)
print("Multipliers:", multipliers.tolist())
print("Encoded keys:", encoded_coords.tolist())
print("Sorted indices:", indices.tolist())
print("Sorted coordinates:")
print(coords[indices])

# Compare with an independent lexicographic sort.
expected = sorted(
    range(len(coords)),
    key=lambda i: tuple(coords[i].tolist())
)
print("Expected indices:", expected)
print("Sorting correct:", indices.tolist() == expected)
