import gc
import glob
import os
import pickle
import torch
import MinkowskiEngine as ME
from functools import partial
from types import SimpleNamespace

from utils import ini_argparse, arrange_sparse_minkowski, arrange_truth, dice_loss
from dataset import SparseDataset
from model import MinkUNetConvNeXtV2, SparseLightningModel


def report_memory(label):
    torch.cuda.synchronize()
    print(
        f"{label}: allocated={torch.cuda.memory_allocated() / 1024**2:.1f} MiB, "
        f"reserved={torch.cuda.memory_reserved() / 1024**2:.1f} MiB, "
        f"peak={torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB",
        flush=True,
    )


args = ini_argparse().parse_args([
    "--dataset_name", "kits23_large_processed",
    "--dataset_path", "data/{}/*",
    "--num_workers", "0",
    "--ds_steps", "3",
    "--losses", "dice",
    "--lr", "5e-4",
    "--warmup_steps", "0",
    "--cosine_annealing_steps", "1",
    "--accum_grad_batches", "1",
    "--weight_decay", "1e-5",
    "--beta1", "0.9",
    "--beta2", "0.95",
])
args.start_cosine_step = 0
args.scheduler_steps = 1

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

print("PyTorch:", torch.__version__, flush=True)
print("MinkowskiEngine:", ME.__version__, flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)

# Load case 00223 without training augmentation.
dataset = SparseDataset(args)
target_file = os.path.abspath(os.path.join("data", "kits23_large_processed", "00223.pkl"))
matches = [
    i for i, file_path in enumerate(dataset.data_files)
    if os.path.abspath(file_path) == os.path.abspath(target_file)
]
if not matches:
    raise FileNotFoundError(f"Could not find processed file: {target_file}")

print("Loading original KiTS23 case 00223 from 00223.pkl...", flush=True)
sample = dataset[matches[0]]
batch = {
    "idx": [sample["idx"]],
    "f": sample["x"],
    "c": [sample["c"]],
    "y": sample["y"],
}

model = None
lightning_model = None
x = None
y = None
outputs = None
targets = None
loss = None
part_losses = None

try:
    print("Initializing original model...", flush=True)
    model = MinkUNetConvNeXtV2(
        in_channels=1, out_channels=3, D=3, args=args
    ).cuda()
    model.train()

    loss_fn = [
        partial(dice_loss, sigmoid=args.sigmoid, reduction="mean")
    ]
    lightning_model = SparseLightningModel(
        model=model, loss_fn=loss_fn, args=args
    ).cuda()
    lightning_model.train()

    # Construct sparse input and targets.
    print("Building sparse tensors...", flush=True)
    x = arrange_sparse_minkowski(batch, torch.device("cuda"))
    y = arrange_truth(batch, torch.device("cuda"))
    report_memory("After sparse tensor construction")
    torch.cuda.reset_peak_memory_stats()

    # Forward pass WITH gradients enabled.
    print("PHASE 1: Forward pass with gradients enabled...", flush=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs, targets = lightning_model(x, y)
    report_memory("After forward")

    # Compute the same loss used by the training code.
    print("PHASE 2: Actual loss calculation...", flush=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, part_losses = lightning_model.compute_losses(outputs, targets)
    print("Loss:", loss.detach().float().item(), flush=True)
    report_memory("After loss")

    if not loss.requires_grad:
        raise RuntimeError("Loss has no gradient graph; backward test cannot proceed.")

    # No optimizer is created or stepped. This only computes gradients.
    print("PHASE 3: Backward pass (NO optimizer step)...", flush=True)
    loss.backward()
    torch.cuda.synchronize()
    report_memory("After backward")

    grads = sum(p.grad is not None for p in model.parameters())
    total_params = sum(1 for p in model.parameters())
    print(f"Parameters with gradients: {grads}/{total_params}", flush=True)

    print("PHASE 4: AdamW optimizer step...", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        betas=(0.9, 0.95),
        weight_decay=1e-5,
    )
    report_memory("Before optimizer step")

    optimizer.step()
    torch.cuda.synchronize()
    report_memory("After optimizer step")

    print("OPTIMIZER STEP TEST SUCCEEDED.", flush=True)
    del optimizer

except torch.cuda.OutOfMemoryError as exc:
    print("\nCUDA OUT OF MEMORY during diagnostic:", str(exc), flush=True)
    print(
        "Peak allocated before cleanup:",
        f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB",
        flush=True,
    )
    raise

finally:
    # Release diagnostic tensors and gradients; no weights were updated.
    del part_losses, loss, targets, outputs, y, x
    del lightning_model, model, batch, sample, dataset
    gc.collect()
    torch.cuda.empty_cache()
    print("Diagnostic cleanup complete.", flush=True)
