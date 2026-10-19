
import os
import glob
import shutil
import pickle
import gc
from functools import partial

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import ini_argparse, collate_sparse_minkowski, dice_loss
from dataset import SparseDataset
from model import MinkUNetConvNeXtV2, SparseLightningModel


CASE_ID = "00130"
RUN_NAME = "case00130_memory_test"
CHECKPOINT_DIR = "checkpoints_diagnostic_case00130_memory"
LOG_DIR = "logs_diagnostic_case00130_memory"


class OneCaseDataset(Dataset):
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.sample


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    # Avoid accidentally reusing a checkpoint from an earlier smoke test.
    if os.path.exists(CHECKPOINT_DIR):
        raise FileExistsError(
            f"{CHECKPOINT_DIR} already exists. Rename it or inspect it first."
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
        "--batch_size", "1",
        "--epochs", "1",
        "--save_top_k", "0",
        "--log_every_n_steps", "1",
        "--gpus", "0",
    ])

    args.scheduler_steps = 1
    args.start_cosine_step = 0
    args.warmup_steps = 0

    print("PyTorch:", torch.__version__, flush=True)
    print("Lightning:", pl.__version__, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    dataset = SparseDataset(args)
    target_path = os.path.abspath(
        os.path.join("data", "kits23_large_processed", f"{CASE_ID}.pkl")
    )

    matches = [
        i for i, path in enumerate(dataset.data_files)
        if os.path.abspath(path) == target_path
    ]
    if not matches:
        raise FileNotFoundError(target_path)

    print(f"Loading only case {CASE_ID}...", flush=True)
    sample = dataset[matches[0]]
    train_loader = DataLoader(
        OneCaseDataset(sample),
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

    os.makedirs(LOG_DIR, exist_ok=True)

    logger = CSVLogger(save_dir=LOG_DIR, name=RUN_NAME)
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        save_last=False,
        save_top_k=0,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=logger,
        log_every_n_steps=1,
        deterministic=True,
        accumulate_grad_batches=1,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        trainer.fit(
            model=lightning_model,
            train_dataloaders=train_loader,
        )

        torch.cuda.synchronize()

        print("\n=== GPU memory report ===", flush=True)
        print(
            f"Peak allocated: "
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
            flush=True,
        )
        print(
            f"Peak reserved: "
            f"{torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB",
            flush=True,
        )
        print("LIGHTNING ONE-CASE TEST SUCCEEDED.", flush=True)

    finally:
        del trainer, lightning_model, model, train_loader, sample, dataset
        gc.collect()
        torch.cuda.empty_cache()
        print("Diagnostic cleanup complete.", flush=True)


if __name__ == "__main__":
    main()
