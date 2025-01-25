"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 09.24

Description: Training script.
"""


import os
import torch
import pytorch_lightning as pl
from functools import partial
from utils import CustomFinetuningReversed, ini_argparse, get_k_fold_data_loaders, supervised_pixel_contrastive_loss, ce_loss, focal_loss, dice_loss
from dataset import SparseDataset
from model import MinkUNetConvNeXtV2, SparseLightningModel
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

#torch.set_float32_matmul_precision('high')

class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ascii = True  # Ensure ASCII characters are used
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ascii = True  # Ensure ASCII characters are used for validation
        return bar


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = ini_argparse()
    args = parser.parse_args()

    print("\n- Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    nb_gpus = len(args.gpus)
    gpus = [int(gpu) for gpu in args.gpus]

    # Dataset
    dataset = SparseDataset(args)

    # K folds for cross validation
    fold_loaders = get_k_fold_data_loaders(dataset, 
                                           args=args, 
                                           shuffle=True, 
                                           random_state=42)

    # Define loss functions
    if args.contrastive and not args.finetuning:
        loss_fn = supervised_pixel_contrastive_loss
    else: 
        loss_fn = []
        for loss in args.losses:
            if loss == "ce":
                loss_fn.append(partial(ce_loss, sigmoid=args.sigmoid, reduction="mean"))
            elif loss == "focal":
                loss_fn.append(partial(focal_loss, sigmoid=args.sigmoid, reduction="mean"))
            elif loss == "dice":
                loss_fn.append(partial(dice_loss, sigmoid=args.sigmoid, reduction="mean"))
            else:
                raise ValueError("Wrong loss")

    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        if fold == 0:
            continue

        print(f'Fold {fold + 1}')
    
        # Calculate arguments for scheduler
        nb_batches = len(train_loader)
        args.scheduler_steps = nb_batches * (args.epochs - args.warmup_steps) // (args.accum_grad_batches * nb_gpus)
        args.warmup_steps = nb_batches * args.warmup_steps // (args.accum_grad_batches * nb_gpus)

        # Initialize the model
        model = MinkUNetConvNeXtV2(in_channels=1, out_channels=1 if (args.roi or args.target != -1) else 3, D=3, args=args)
        #print(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total trainable params model (total): {}".format(total_params))

        lightning_model = SparseLightningModel(model=model,
                                               loss_fn=loss_fn,
                                               args=args)

        # Define logger and checkpoint
        logger = CSVLogger(save_dir=args.save_dir + "/logs", name=args.name + "_{}".format(str(fold)))
        tb_logger = TensorBoardLogger(save_dir=args.save_dir + "/tb_logs", name=args.name + "_{}".format(str(fold)))
        checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_path + "/" + args.checkpoint_name + "_{}".format(str(fold)),
                                              save_last=True, save_top_k=args.save_top_k, monitor="loss/val_total")
        progress_bar = CustomProgressBar()

        # Log the hyperparameters
        logger.log_hyperparams(vars(args))
        tb_logger.log_hyperparams(vars(args))
 
        # Initialize PyTorch Lightning trainer
        trainer = pl.Trainer(
            #num_sanity_val_steps=0,
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback, progress_bar],
            precision=16,
            accelerator="gpu",
            devices=gpus,
            strategy="ddp" if nb_gpus > 1 else None,
            logger=[logger, tb_logger],
            log_every_n_steps=args.log_every_n_steps,
            deterministic=True,
            accumulate_grad_batches=args.accum_grad_batches,
        )

        # Train and validate the model for this fold
        trainer.fit(model=lightning_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path="/scratch/salonso/sparse-nns/medical_ai/ai_cancer_research/checkpoints_stage_ind/dice_optuna_1Kepochs_1/last.ckpt" if fold==1 else None)


if __name__ == "__main__":
    main()

