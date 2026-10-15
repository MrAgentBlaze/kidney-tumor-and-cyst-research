import os
import gc
import torch
import pytorch_lightning as pl

from functools import partial
from torch.utils.data import DataLoader, Subset

from utils import (
    ini_argparse,
    collate_sparse_minkowski,
    dice_loss,
)
from dataset import SparseDataset
from model import MinkUNetConvNeXtV2, SparseLightningModel


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
        "--num_workers", "0",
        "--lr", "5e-4",
        "--warmup_steps", "1",
        "--cosine_annealing_steps", "400",
        "--weight_decay", "1e-5",
        "--beta1", "0.9",
        "--beta2", "0.95",
        "--losses", "dice",
        "--gpus", "0",
    ])

    # Scheduler values for a one-batch diagnostic, not a training run.
    args.scheduler_steps = 400
    args.start_cosine_step = -399

    print("PyTorch:", torch.__version__)
    print("Lightning:", pl.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Loading fixed case 00295; augmentation disabled.")

    dataset = SparseDataset(args)
    matches = [
        i for i, path in enumerate(dataset.data_files)
        if os.path.basename(path) == "00223.pkl"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one 00295.pkl, found {len(matches)}"
        )

    loader = DataLoader(
        Subset(dataset, matches),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_sparse_minkowski,
    )

    loss_fn = [
        partial(dice_loss, sigmoid=args.sigmoid, reduction="mean")
    ]

    model = MinkUNetConvNeXtV2(
        in_channels=1, out_channels=3, D=3, args=args
    )
    lightning_model = SparseLightningModel(
        model=model, loss_fn=loss_fn, args=args
    )

    print("Starting one-batch Lightning test.")
    print("No validation, checkpointing, or external loggers.")
    print("Gradient accumulation: 1")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accumulate_grad_batches=1,
        num_nodes=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        deterministic=False,
    )

    try:
        trainer.fit(lightning_model, train_dataloaders=loader)
        print("\nDIAGNOSTIC RESULT: Lightning completed one batch.")
    except Exception:
        print("\nDIAGNOSTIC RESULT: Lightning test failed.")
        raise
    finally:
        del trainer, lightning_model, model, loader, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Diagnostic cleanup complete.")


if __name__ == "__main__":
    main()
