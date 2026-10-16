import os
import gc
import traceback

import torch
import pytorch_lightning as pl

from functools import partial
from torch.utils.data import Subset

from utils import (
    ini_argparse,
    get_k_fold_data_loaders,
    dice_loss,
)
from dataset import SparseDataset
from model import MinkUNetConvNeXtV2, SparseLightningModel


def memory_report(tag):
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA unavailable", flush=True)
        return

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"[{tag}] CUDA allocated={allocated:.2f} GiB, "
        f"reserved={reserved:.2f} GiB, peak={peak:.2f} GiB",
        flush=True,
    )



class DiagnosticLightningModel(SparseLightningModel):
    def training_step(self, batch, batch_idx):
        case_ids = batch.get("idx", ["unknown"])
        print(
            f"\n[TRAIN BATCH {batch_idx}] case_ids={case_ids}",
            flush=True,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            memory_report(f"batch {batch_idx} before forward")

        try:
            loss = super().training_step(batch, batch_idx)
            memory_report(f"batch {batch_idx} after forward/loss")
            return loss
        except torch.cuda.OutOfMemoryError:
            print(
                f"\n[CUDA OOM during training_step] "
                f"batch={batch_idx}, case_ids={case_ids}",
                flush=True,
            )
            # Avoid synchronizing here: the CUDA context may be in an
            # error state after an asynchronous CUDA failure.
            if torch.cuda.is_available():
                print(
                    "Allocated GiB:",
                    torch.cuda.memory_allocated() / 1024**3,
                    flush=True,
                )
                print(
                    "Reserved GiB:",
                    torch.cuda.memory_reserved() / 1024**3,
                    flush=True,
                )
            raise

    def on_before_backward(self, loss):
        if torch.cuda.is_available():
            print(
                "[before backward] "
                f"allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GiB, "
                f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GiB, "
                f"peak={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
                flush=True,
            )

    def on_after_backward(self):
        if torch.cuda.is_available():
            # Avoid synchronizing here; report allocator counters directly.
            print(
                "[after backward] "
                f"allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GiB, "
                f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GiB, "
                f"peak={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
                flush=True,
            )


def main():
    parser = ini_argparse()
    args = parser.parse_args([
        "--target", "-1",
        "--dataset_name", "kits23_large_processed",
        "--dataset_path", "data/{}/*",
        "--min_hu", "-53.4",
        "--max_hu", "283.2",
        "--ds_steps", "3",
        "--batch_size", "1",
        "--num_workers", "8",
        "--folds", "2",
        "--epochs", "1",
        "--accum_grad_batches", "8",
        "--lr", "5e-4",
        "--warmup_steps", "1",
        "--cosine_annealing_steps", "400",
        "--weight_decay", "1e-5",
        "--beta1", "0.9",
        "--beta2", "0.95",
        "--losses", "dice",
        "--gpus", "0",
    ])

    # Use the same split seed as the main training script.
    torch.multiprocessing.set_sharing_strategy("file_system")

    dataset = SparseDataset(args)
    fold_loaders = get_k_fold_data_loaders(
        dataset, args=args, shuffle=True, random_state=42
    )
    train_loader, _ = fold_loaders[0]

    # Bound the diagnostic to a small number of batches.
    max_batches = 5
    diagnostic_loader = train_loader

    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Lightning: {pl.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"Diagnostic batch limit: {max_batches}", flush=True)
    print("Training augmentation is enabled.", flush=True)
    print("Gradient accumulation: 8", flush=True)

    loss_fn = [
        partial(dice_loss, sigmoid=args.sigmoid, reduction="mean")
    ]

    args.scheduler_steps = 400
    args.start_cosine_step = -399

    model = MinkUNetConvNeXtV2(
        in_channels=1, out_channels=3, D=3, args=args
    )
    lightning_model = DiagnosticLightningModel(
        model=model, loss_fn=loss_fn, args=args
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=1,
        limit_train_batches=max_batches,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accumulate_grad_batches=8,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    try:
        trainer.fit(lightning_model, train_dataloaders=diagnostic_loader)
        print("\nDIAGNOSTIC COMPLETED", flush=True)
    except Exception:
        print("\nDIAGNOSTIC FAILED:", flush=True)
        traceback.print_exc()
        raise
    finally:
        del trainer, lightning_model, model, diagnostic_loader
        del train_loader, fold_loaders, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Diagnostic cleanup complete.", flush=True)


if __name__ == "__main__":
    main()