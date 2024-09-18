import os
import torch
import pytorch_lightning as pl
from functools import partial
from utils import ini_argparse, get_k_fold_data_loaders, sigmoid_focal_loss, dice_loss
from dataset import SparseDataset
from model import MinkUNetConvNeXtV2, SparseLightningModel
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# manually specify the GPUs to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = ini_argparse()
args = parser.parse_args([])

print("\n- Arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
nb_gpus = len(args.gpus)
gpus = [int(gpu) for gpu in args.gpus]

# Dataset
dataset = SparseDataset(args)

# K folds for cross validation
fold_loaders = get_k_fold_data_loaders(dataset, 
                                       num_folds=args.folds, 
                                       batch_size=args.batch_size,
                                       shuffle=True, 
                                       random_state=42)

# Store validation metrics for each fold
fold_results = []

loss_fn = [partial(sigmoid_focal_loss, gamma=1., reduction="mean"), dice_loss]

for fold, (train_loader, val_loader) in enumerate(fold_loaders):
    print(f'Fold {fold + 1}')
    
    # Calculate arguments for scheduler
    nb_batches = len(train_loader)
    args.scheduler_steps = nb_batches * (args.epochs - args.warmup_steps) // (args.accum_grad_batches * nb_gpus)
    args.warmup_steps = nb_batches * args.warmup_steps // (args.accum_grad_batches * nb_gpus)

    # Initialize the model
    model = MinkUNetConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
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

    # Log the hyperparameters
    logger.log_hyperparams(vars(args))
    tb_logger.log_hyperparams(vars(args))
 
    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        logger=[logger, tb_logger],
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True,
        accumulate_grad_batches=args.accum_grad_batches,
    )

    # Train and validate the model for this fold
    trainer.fit(lightning_model, train_loader, val_loader)


