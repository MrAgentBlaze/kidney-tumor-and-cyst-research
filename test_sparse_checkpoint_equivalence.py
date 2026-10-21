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

base_features = torch.randn(8, 4, device=device)

block_normal = Block(dim=4, kernel_size=3, D=3).to(device)
block_checkpointed = Block(dim=4, kernel_size=3, D=3).to(device)

# Copy parameter values without deep-copying MinkowskiEngine objects.
with torch.no_grad():
    for p_normal, p_checkpointed in zip(
        block_normal.parameters(), block_checkpointed.parameters()
    ):
        p_checkpointed.copy_(p_normal)

block_normal.train()
block_checkpointed.train()

def make_sparse(features):
    return ME.SparseTensor(
        features=features,
        coordinates=coords,
        device=device,
    )

# Normal execution
features_normal = base_features.clone().detach().requires_grad_(True)
out_normal = block_normal(make_sparse(features_normal)).F
out_normal.float().square().mean().backward()

# Checkpointed execution
features_checkpointed = base_features.clone().detach().requires_grad_(True)

def run_block(features):
    return block_checkpointed(make_sparse(features)).F

out_checkpointed = checkpoint(
    run_block,
    features_checkpointed,
    use_reentrant=False,
)
out_checkpointed.float().square().mean().backward()

output_diff = (out_normal - out_checkpointed).abs().max().item()
input_grad_diff = (
    features_normal.grad - features_checkpointed.grad
).abs().max().item()

param_grad_diffs = [
    (p_normal.grad - p_checkpointed.grad).abs().max().item()
    for p_normal, p_checkpointed in zip(
        block_normal.parameters(), block_checkpointed.parameters()
    )
    if p_normal.grad is not None and p_checkpointed.grad is not None
]
max_param_grad_diff = max(param_grad_diffs, default=0.0)

print(f"Maximum output difference: {output_diff:.8g}")
print(f"Maximum input-gradient difference: {input_grad_diff:.8g}")
print(f"Maximum parameter-gradient difference: {max_param_grad_diff:.8g}")

tol = 1e-4
if max(output_diff, input_grad_diff, max_param_grad_diff) <= tol:
    print("PASS: checkpointed and normal results agree within tolerance.")
else:
    print("CHECK: differences exceed tolerance; inspect before proceeding.")
