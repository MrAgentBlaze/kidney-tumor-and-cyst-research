import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from utils import arrange_sparse_minkowski, arrange_truth, argsort_sparse_tensor, CustomLambdaLR, CombinedScheduler
from pytorch_lightning.trainer.supporters import CombinedDataset


class SparseLightningModel(pl.LightningModule):
    def __init__(self, model, loss_fn, args):
        super(SparseLightningModel, self).__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.warmup_steps = args.warmup_steps
        self.cosine_annealing_steps = args.scheduler_steps
        self.lr = args.lr
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = args.weight_decay
        self.eps = args.eps


    def on_train_start(self):
        "Fixing bug: https://github.com/Lightning-AI/pytorch-lightning/issues/17296#issuecomment-1726715614"
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups
 

    def on_train_epoch_start(self):
        """Hook to be called at the start of each training epoch."""
        train_loader = self.trainer.train_dataloader
        if isinstance(train_loader.dataset, CombinedDataset):
            train_loader.dataset.datasets.dataset.set_training_mode(True)
        else:
            train_loader.dataset.set_training_mode(True)


    def on_validation_epoch_start(self):
        """Hook to be called at the start of each validation epoch."""
        val_loader = self.trainer.val_dataloaders[0]
        val_loader.dataset.dataset.set_training_mode(False)


    def on_test_epoch_start(self):
        """Hook to be called at the start of each test epoch."""
        test_loader = self.trainer.test_dataloaders[0]
        test_loader.dataset.dataset.set_training_mode(False)


    def forward(self, x, y):
        return self.model(x, y)


    def _arrange_batch(self, batch):
        batch_input = arrange_sparse_minkowski(batch, self.device)
        batch_target = arrange_truth(batch, self.device)
        return batch_input, batch_target


    def compute_losses(self, batch_output, batch_target):
        losses = [0., 0.]
        for step, (output, target) in enumerate(zip(batch_output, batch_target)):
            weight = 1/(2**step)

            sorted_indices_output = argsort_sparse_tensor(output)
            sorted_indices_target = argsort_sparse_tensor(target)
            
            sorted_coords_output = output.coordinates[sorted_indices_output]
            sorted_coords_target = target.coordinates[sorted_indices_target]

            # Now the coordinates should be aligned; you can sum the features
            assert (sorted_coords_output == sorted_coords_target).all(), "Coordinates are still not aligned!"

            sorted_feats_output = output.F[sorted_indices_output]
            sorted_feats_target = target.F[sorted_indices_target].view(-1)
           
            # Retrieve independent targets
            kidney_tumor_cyst = (sorted_feats_target > 0).float()
            tumor_cyst = (sorted_feats_target > 1).float()
            tumor_only = (sorted_feats_target == 2).float() 

            # Compute losses
            for i, loss_fn in enumerate(self.loss_fn):
                curr_loss = loss_fn(sorted_feats_output[:, 0], kidney_tumor_cyst)
                curr_loss += loss_fn(sorted_feats_output[:, 1], tumor_cyst)
                curr_loss += loss_fn(sorted_feats_output[:, 2], tumor_only)
                losses[i] += weight * curr_loss 
        
        total_loss = losses[0] + losses[1]

        return total_loss, losses 


    def common_step(self, batch):
        batch_size = len(batch["c"])
        batch_input, batch_target = self._arrange_batch(batch)

        # Forward pass
        batch_output, batch_target = self.forward(batch_input, batch_target)

        # Compute loss
        loss, (loss1, loss2) = self.compute_losses(batch_output, batch_target)
  
        # Retrieve current learning rate
        lr = self.optimizers().param_groups[0]['lr']

        return loss, loss1, loss2, batch_size, lr


    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, loss1, loss2, batch_size, lr = self.common_step(batch)

        self.log(f"loss/train_total", loss.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)
        self.log(f"loss/train1", loss1.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)
        self.log(f"loss/train2", loss2.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)
        self.log(f"lr", lr, batch_size=batch_size, prog_bar=True, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, loss1, loss2, batch_size, lr = self.common_step(batch)

        self.log(f"loss/val_total", loss.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)
        self.log(f"loss/val1", loss1.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)
        self.log(f"loss/val2", loss2.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)

        return loss


    def configure_optimizers(self):
        """Configure and initialize the optimizer and learning rate scheduler."""
        # Optimiser
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )

        # Cosine annealing scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.cosine_annealing_steps,
            eta_min=0
        )

        if self.warmup_steps > 0:
            # Warm-up scheduler
            warmup_scheduler = CustomLambdaLR(optimizer, self.warmup_steps)
        
            # Combine both schedulers
            combined_scheduler = CombinedScheduler(
                optimizer=optimizer,
                scheduler1=warmup_scheduler,
                scheduler2=cosine_scheduler,
                warmup_steps=self.warmup_steps,
                lr_decay=1.0
            )
        else:
            # No warm-up
            combined_scheduler = cosine_scheduler

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': combined_scheduler, 'interval': 'step'}}


    def lr_scheduler_step(self, scheduler, *args):
        """Perform a learning rate scheduler step."""
        scheduler.step()

