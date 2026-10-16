import torch


def check_sort(coords):
    coords = coords.to(torch.int64)
    max_val = coords.max() + 1

    multipliers = torch.tensor(
        [max_val ** i for i in reversed(range(coords.shape[1]))],
        device=coords.device,
        dtype=torch.int64,
    )

    encoded = (coords * multipliers).sum(dim=1)
    indices = torch.argsort(encoded)

    sorted_coords = coords[indices]
    lexicographic = sorted(
        range(len(coords)),
        key=lambda i: tuple(coords[i].tolist()),
    )

    print("Device:", coords.device)
    print("Coordinate shape:", tuple(coords.shape))
    print("Max coordinate:", coords.max().item())
    print("Keys unique:", encoded.unique().numel() == encoded.numel())
    print("Lexicographic order correct:", indices.tolist() == lexicographic)


# Simulate a large set of 3D sparse coordinates plus batch index.
torch.manual_seed(42)
coords = torch.randint(0, 722, (100_000, 4), dtype=torch.int32)
coords[:, 0] = torch.randint(0, 2, (100_000,), dtype=torch.int32)

check_sort(coords)