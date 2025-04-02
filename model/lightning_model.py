import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from utils import arrange_sparse_minkowski, arrange_truth, argsort_sparse_tensor, CustomLambdaLR, CombinedScheduler
from packaging import version


pl_version = pl.__version__


class SparseLightningModel(pl.LightningModule):
    def __init__(self, model, loss_fn, args):
        super(SparseLightningModel, self).__init__()

        self.model = model
        self.target = args.target
        self.losses = args.losses
        self.loss_fn = loss_fn
        self.warmup_steps = args.warmup_steps
        self.start_cosine_step = args.start_cosine_step
        self.cosine_annealing_steps = args.scheduler_steps
        self.lr = args.lr
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = args.weight_decay
        self.eps = args.eps
        self.roi = args.roi


    def on_train_start(self):
        "Fixing bug: https://github.com/Lightning-AI/pytorch-lightning/issues/17296#issuecomment-1726715614"
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups
 

    def forward(self, x, y):
        return self.model(x, y)


    def _arrange_batch(self, batch):
        batch_input = arrange_sparse_minkowski(batch, self.device)
        batch_target = arrange_truth(batch, self.device)
        return batch_input, batch_target


    def compute_losses(self, batch_output, batch_target):
        losses = [0. for _ in range(len(self.losses))]
        part_losses = [[] for _ in range(len(self.losses))]
        for step, (output, target) in enumerate(zip(batch_output, batch_target)):
            #weight = 1 / (2 ** step)  # inspired by https://arxiv.org/pdf/2310.04110
            weight = 1 / (2 ** (3 * step))
           
            # Align output and target
            sorted_indices_output = argsort_sparse_tensor(output)
            sorted_indices_target = argsort_sparse_tensor(target)
            sorted_coords_output = output.coordinates[sorted_indices_output]
            sorted_coords_target = target.coordinates[sorted_indices_target]

            # Now the coordinates should be aligned; you can sum the features
            assert (sorted_coords_output == sorted_coords_target).all(), "Coordinates are still not aligned!"

            sorted_feats_output = output.F[sorted_indices_output]
            sorted_feats_target = target.F[sorted_indices_target]

            decom_feats_output = output.decomposed_features
            decom_feats_target = target.decomposed_features

            # Compute losses
            for i, (loss, loss_fn) in enumerate(zip(self.losses, self.loss_fn)):
                extra_args = {}
                if loss == "focal":
                    extra_args["gamma"] = 2.0
                    extra_args["alpha"] = 0.9

                if self.roi:
                    all_masses_loss = loss_fn(decom_feats_output, decom_feats_target, **extra_args)
                    part_losses[i].append([all_masses_loss])
                    curr_loss = all_masses_loss
                else:
                    if self.target == -1:
                        kidney_masses_loss = loss_fn(sorted_feats_output[:, 0], sorted_feats_target[:, 0], **extra_args)
                        masses_loss = loss_fn(sorted_feats_output[:, 1], sorted_feats_target[:, 1], **extra_args)
                        tumor_only_loss = loss_fn(sorted_feats_output[:, 2], sorted_feats_target[:, 2], **extra_args) 
                        part_losses[i].append([kidney_masses_loss, masses_loss, tumor_only_loss])
                        curr_loss = kidney_masses_loss + masses_loss + tumor_only_loss
                    else:
                        curr_loss = loss_fn(sorted_feats_output[:, 0], sorted_feats_target[:, self.target], **extra_args)
                        part_losses[i].append([curr_loss])
                
                losses[i] += weight * curr_loss 

        part_losses = torch.tensor(part_losses)
        total_loss = sum(losses)

        return total_loss, part_losses
   

    def common_step(self, batch):
        batch_size = len(batch["c"])
        batch_input, batch_target = self._arrange_batch(batch)

        # Forward pass
        batch_output, batch_target = self.forward(batch_input, batch_target)
        loss, part_losses = self.compute_losses(batch_output, batch_target)
  
        # Retrieve current learning rate
        lr = self.optimizers().param_groups[0]['lr']

        return loss, part_losses, batch_size, lr


    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        #print(batch["idx"], batch["f"].shape)

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(f"loss/train_total", loss.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)

        for i in range(part_losses.shape[0]):  # Loop over N
            total_sum_over_mk = part_losses[i, :, :].sum().item()  # Sum over M and K for the i-th N
            self.log("loss/train_{}".format(self.losses[i]), total_sum_over_mk, batch_size=batch_size, prog_bar=True, sync_dist=True)

        for j in range(part_losses.shape[1]):  # Loop over M
            total_sum_over_nk = part_losses[:, j, :].sum().item()  # Sum over N and K for the j-th M
            self.log(f"loss/train_step{j}", total_sum_over_nk, batch_size=batch_size, prog_bar=False, sync_dist=True)

        for k in range(part_losses.shape[2]):  # Loop over K
            total_sum_over_nm = part_losses[:, :, k].sum().item()  # Sum over N and M for the k-th K
            self.log(f"loss/train_o{k}", total_sum_over_nm, batch_size=batch_size, prog_bar=False, sync_dist=True)

        for i in range(part_losses.shape[0]):
            for j in range(part_losses.shape[1]):
                for k in range(part_losses.shape[2]):
                    output = part_losses[i, j, k].item()
                    self.log("loss/train_{}_o{}_step{}".format(self.losses[i], k, j), output, batch_size=batch_size, prog_bar=False, sync_dist=True)
        
        self.log(f"lr", lr, batch_size=batch_size, prog_bar=True, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(f"loss/val_total", loss.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)

        for i in range(part_losses.shape[0]):  # Loop over N
            total_sum_over_mk = part_losses[i, :, :].sum().item()  # Sum over M and K for the i-th N
            self.log("loss/val_{}".format(self.losses[i]), total_sum_over_mk, batch_size=batch_size, prog_bar=False, sync_dist=True)

        for j in range(part_losses.shape[1]):  # Loop over M
            total_sum_over_nk = part_losses[:, j, :].sum().item()  # Sum over N and K for the j-th M
            self.log(f"loss/val_step{j}", total_sum_over_nk, batch_size=batch_size, prog_bar=False, sync_dist=True)

        for k in range(part_losses.shape[2]):  # Loop over K
            total_sum_over_nm = part_losses[:, :, k].sum().item()  # Sum over N and M for the k-th K
            self.log(f"loss/val_o{k}", total_sum_over_nm, batch_size=batch_size, prog_bar=False, sync_dist=True)

        for i in range(part_losses.shape[0]):
            for j in range(part_losses.shape[1]):
                for k in range(part_losses.shape[2]):
                    output = part_losses[i, j, k].item()
                    self.log("loss/val_{}_o{}_step{}".format(self.losses[i], k, j), output, batch_size=batch_size, prog_bar=False, sync_dist=True)

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

        # Warm-up scheduler
        warmup_scheduler = CustomLambdaLR(optimizer, self.warmup_steps)
        
        # Combine both schedulers
        combined_scheduler = CombinedScheduler(
            optimizer=optimizer,
            scheduler1=warmup_scheduler,
            scheduler2=cosine_scheduler,
            warmup_steps=self.warmup_steps,
            start_cosine_step=self.start_cosine_step,
            lr_decay=1.0
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': combined_scheduler, 'interval': 'step'}}


    def lr_scheduler_step(self, scheduler, *args):
        """Perform a learning rate scheduler step."""
        scheduler.step()

