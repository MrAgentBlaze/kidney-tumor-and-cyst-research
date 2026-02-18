import os
import torch
import glob
import pytorch_lightning as pl
from functools import partial
from utils import (
    collate_sparse_minkowski,
    ini_argparse,
    ce_loss,
    focal_loss,
    dice_loss,
)
from dataset import SparseDataset
from model import MinkUNetConvNeXtV2, SparseLightningModel
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import Subset, DataLoader


torch.set_float32_matmul_precision("medium")
pl_major = int(pl.__version__.split(".")[0])


class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ascii = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ascii = True
        return bar


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = ini_argparse()
    args = parser.parse_args()

    print("\n- Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    nb_gpus = len(args.gpus)
    gpus = ",".join(map(str, args.gpus)) if nb_gpus > 1 else str(args.gpus[0])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    dataset = SparseDataset(args)
    dataset = Subset(dataset, range(len(dataset)))
    dataset.dataset.training = True

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
        collate_fn=collate_sparse_minkowski, shuffle=True,
    )
    val_loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
        collate_fn=collate_sparse_minkowski, shuffle=False,
    )

    # Build loss functions
    loss_fn = []
    for loss in args.losses:
        if loss == "ce":
            loss_fn.append(partial(ce_loss, sigmoid=args.sigmoid, reduction="mean"))
        elif loss == "focal":
            loss_fn.append(partial(focal_loss, sigmoid=args.sigmoid, reduction="mean"))
        elif loss == "dice":
            loss_fn.append(partial(dice_loss, sigmoid=args.sigmoid, reduction="mean"))
        else:
            raise ValueError(f"Unknown loss: {loss}")

    warmup_steps = args.warmup_steps

    # Compute scheduler parameters
    nb_batches = len(train_loader)
    args.scheduler_steps = nb_batches * args.cosine_annealing_steps // (args.accum_grad_batches * nb_gpus)
    args.warmup_steps = nb_batches * warmup_steps // (args.accum_grad_batches * nb_gpus)
    args.start_cosine_step = (nb_batches * args.epochs // (args.accum_grad_batches * nb_gpus)) - args.scheduler_steps

    out_channels = 1 if (args.roi or args.target != -1) else 3
    model = MinkUNetConvNeXtV2(
        in_channels=1, out_channels=out_channels, D=3, args=args,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    lightning_model = SparseLightningModel(
        model=model, loss_fn=loss_fn, args=args,
    )

    # Loggers and callbacks
    logger = CSVLogger(
        save_dir=os.path.join(args.save_dir, "logs"), name=args.name,
    )
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(args.save_dir, "tb_logs"), name=args.name,
    )
    ckpt_dir = os.path.join(args.checkpoint_path, args.checkpoint_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir, save_last=True,
        save_top_k=args.save_top_k, monitor="loss/val_total",
    )
    progress_bar = CustomProgressBar()

    # Resume from checkpoint if available
    if args.load_checkpoint is None:
        ckpt_path = glob.glob(os.path.join(ckpt_dir, "last*ckpt"))
        if ckpt_path:
            args.load_checkpoint = sorted(ckpt_path)[-1]

    logger.log_hyperparams(vars(args))
    tb_logger.log_hyperparams(vars(args))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, progress_bar],
        precision="bf16-mixed" if pl_major >= 2 else 32,
        accelerator="gpu",
        devices=nb_gpus,
        strategy="ddp" if nb_gpus > 1 else "auto",
        logger=[logger, tb_logger],
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True,
        accumulate_grad_batches=args.accum_grad_batches,
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.load_checkpoint if args.load_checkpoint else None,
    )


if __name__ == "__main__":
    main()
