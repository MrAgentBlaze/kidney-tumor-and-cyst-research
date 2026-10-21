import torch
import MinkowskiEngine as ME
from torch.utils.checkpoint import checkpoint

from model.minkunet_convnextv2 import Block

device = torch.device("cuda")
torch.manual_seed(42)

coords = torch.tensor([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
], dtype=torch.int32, device=device)

features = torch.randn(8, 4, device=device, requires_grad=True)
x = ME.SparseTensor(
    features=features,
    coordinates=coords,
    device=device,
)

block = Block(dim=4, kernel_size=3, D=3).to(device)
block.train()

def run_block(f):
    sparse_input = ME.SparseTensor(
        features=f,
        coordinate_map_key=x.coordinate_map_key,
        coordinate_manager=x.coordinate_manager,
    )
    return block(sparse_input).F

try:
    out = checkpoint(run_block, x.F, use_reentrant=False)
    loss = out.float().square().mean()
    loss.backward()

    print("PASS: checkpointed sparse block completed.")
    print("Output shape:", tuple(out.shape))
    print("Input gradient exists:", x.F.grad is not None)
    print(
        "Block parameter gradient exists:",
        any(p.grad is not None for p in block.parameters()),
    )
except Exception as exc:
    print("FAIL:", type(exc).__name__, str(exc))
    raise
