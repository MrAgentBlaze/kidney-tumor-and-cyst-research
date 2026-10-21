import torch
import pytorch_lightning as pl
import timm.optim.optim_factory as optim_factory
from utils import (
    arrange_sparse_minkowski,
    arrange_truth,
    argsort_sparse_tensor,
    CustomLambdaLR,
    CombinedScheduler,
)


class SparseLightningModel(pl.LightningModule):
    def __init__(self, model, loss_fn, args):
        super().__init__()
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
        self._memory_diagnostic = True

    def on_train_start(self):
        # Fix: https://github.com/Lightning-AI/pytorch-lightning/issues/17296
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    def forward(self, x, y):
        return self.model(x, y)
    

    def _memory_snapshot(self, stage, batch_idx):
        if not torch.cuda.is_available() or not getattr(
            self, "_memory_diagnostic", False
        ):
            return

        device = torch.cuda.current_device()
        torch.cuda.synchronize(device)

        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)

        print(
            f"[GPU-MEM] batch={batch_idx} stage={stage} "
            f"allocated={allocated:.0f}MiB "
            f"reserved={reserved:.0f}MiB "
            f"peak_allocated={peak:.0f}MiB "
            f"device_free={free_bytes / (1024 ** 2):.0f}MiB "
            f"device_total={total_bytes / (1024 ** 2):.0f}MiB",
            flush=True,
        )

    def on_before_backward(self, loss):
        if getattr(self, "_memory_diagnostic", False):
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                self._memory_snapshot(
                    "before_backward",
                    getattr(self, "_memory_batch_idx", self.global_step),
                )

    def on_after_backward(self):
        if getattr(self, "_memory_diagnostic", False):
            self._memory_snapshot(
                "after_backward",
                getattr(self, "_memory_batch_idx", self.global_step),
            )

    def _arrange_batch(self, batch):
        batch_input = arrange_sparse_minkowski(batch, self.device)
        batch_target = arrange_truth(batch, self.device)
        return batch_input, batch_target

    def compute_losses(self, batch_output, batch_target):
        losses = [0.0 for _ in range(len(self.losses))]
        part_losses = [[] for _ in range(len(self.losses))]
        for step, (output, target) in enumerate(zip(batch_output, batch_target)):
            weight = 1 / (2 ** step)

            if not getattr(self, "_coord_diag_done", False):
                print("\n=== Coordinate alignment diagnostic ===")
                print("Number of output stages:", len(batch_output))
                print("Number of target stages:", len(batch_target))

                for stage, (out, tgt) in enumerate(zip(batch_output, batch_target)):
                    print(f"Stage {stage}:")
                    print("  Output shape:", tuple(out.coordinates.shape))
                    print("  Target shape:", tuple(tgt.coordinates.shape))
                    print("  Same coordinate map:",
                        out.coordinate_map_key == tgt.coordinate_map_key)

                    if out.coordinates.shape == tgt.coordinates.shape:
                        print("  Same coordinate order:",
                            torch.equal(out.coordinates, tgt.coordinates))
                    else:
                        print("  Same coordinate order: False (different shapes)")

                self._coord_diag_done = True
                print("=== End diagnostic ===\n")
            
            # Align output and target by sorting coordinates
            sorted_indices_output = argsort_sparse_tensor(output)
            sorted_indices_target = argsort_sparse_tensor(target)
            sorted_coords_output = output.coordinates[sorted_indices_output]
            sorted_coords_target = target.coordinates[sorted_indices_target]
            assert (sorted_coords_output == sorted_coords_target).all(), \
                "Coordinates are not aligned"

            sorted_feats_output = output.F[sorted_indices_output]
            sorted_feats_target = target.F[sorted_indices_target]

            decom_feats_output = output.decomposed_features
            decom_feats_target = target.decomposed_features

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
                        kidney_masses_loss = loss_fn(
                            sorted_feats_output[:, 0], sorted_feats_target[:, 0], **extra_args)
                        masses_loss = loss_fn(
                            sorted_feats_output[:, 1], sorted_feats_target[:, 1], **extra_args)
                        tumour_only_loss = loss_fn(
                            sorted_feats_output[:, 2], sorted_feats_target[:, 2], **extra_args)
                        part_losses[i].append([kidney_masses_loss, masses_loss, tumour_only_loss])
                        curr_loss = kidney_masses_loss + masses_loss + tumour_only_loss
                    else:
                        curr_loss = loss_fn(
                            sorted_feats_output[:, 0], sorted_feats_target[:, self.target], **extra_args)
                        part_losses[i].append([curr_loss])

                losses[i] += weight * curr_loss

        part_losses = torch.tensor(
            [
                [
                    [value.detach().item() if isinstance(value, torch.Tensor) else value
                    for value in step_losses]
                    for step_losses in loss_steps
                ]
                for loss_steps in part_losses
            ],
            dtype=torch.float32,
        )
        total_loss = sum(losses)
        return total_loss, part_losses


    def common_step(self, batch):
        batch_size = len(batch["c"])
        batch_input, batch_target = self._arrange_batch(batch)

        diagnostic = getattr(self, "_memory_diagnostic", False)
        batch_idx = getattr(self, "_memory_batch_idx", self.global_step)

        if diagnostic and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self._memory_snapshot("before_forward", batch_idx)

        batch_output, batch_target = self.forward(batch_input, batch_target)

        if diagnostic:
            self._memory_snapshot("after_forward", batch_idx)
            torch.cuda.reset_peak_memory_stats()

        loss, part_losses = self.compute_losses(batch_output, batch_target)

        if diagnostic:
            self._memory_snapshot("after_loss", batch_idx)

        lr = self.optimizers().param_groups[0]["lr"]
        return loss, part_losses, batch_size, lr


    def training_step(self, batch, batch_idx):
        self._memory_batch_idx = batch_idx
        torch.cuda.empty_cache()
        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log("loss/train_total", loss.item(), batch_size=batch_size,
                 prog_bar=True, sync_dist=True)

        for i in range(part_losses.shape[0]):
            total = part_losses[i, :, :].sum().item()
            self.log(f"loss/train_{self.losses[i]}", total,
                     batch_size=batch_size, sync_dist=True)

        for j in range(part_losses.shape[1]):
            total = part_losses[:, j, :].sum().item()
            self.log(f"loss/train_step{j}", total, batch_size=batch_size, sync_dist=True)

        for k in range(part_losses.shape[2]):
            total = part_losses[:, :, k].sum().item()
            self.log(f"loss/train_o{k}", total, batch_size=batch_size, sync_dist=True)

        self.log("lr", lr, batch_size=batch_size, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._memory_batch_idx = batch_idx
        torch.cuda.empty_cache()
        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log("loss/val_total", loss.item(), batch_size=batch_size,
                 prog_bar=True, sync_dist=True)

        for i in range(part_losses.shape[0]):
            total = part_losses[i, :, :].sum().item()
            self.log(f"loss/val_{self.losses[i]}", total,
                     batch_size=batch_size, sync_dist=True)

        for j in range(part_losses.shape[1]):
            total = part_losses[:, j, :].sum().item()
            self.log(f"loss/val_step{j}", total, batch_size=batch_size, sync_dist=True)

        for k in range(part_losses.shape[2]):
            total = part_losses[:, :, k].sum().item()
            self.log(f"loss/val_o{k}", total, batch_size=batch_size, sync_dist=True)

        return loss

    def configure_optimizers(self):
        no_weight_decay_list = [
            name for name, _ in self.model.named_parameters()
            if name.endswith(".gamma") or name.endswith(".beta")
        ]
        param_groups = optim_factory.param_groups_weight_decay(
            self.model, self.weight_decay, no_weight_decay_list=no_weight_decay_list,
        )

        optimiser = torch.optim.AdamW(
            param_groups, lr=self.lr, betas=self.betas, eps=self.eps,
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimiser, T_max=self.cosine_annealing_steps, eta_min=0,
        )

        warmup_scheduler = CustomLambdaLR(optimiser, self.warmup_steps)

        combined_scheduler = CombinedScheduler(
            optimizer=optimiser,
            scheduler1=warmup_scheduler,
            scheduler2=cosine_scheduler,
            warmup_steps=self.warmup_steps,
            start_cosine_step=self.start_cosine_step,
            lr_decay=1.0,
        )

        return {
            "optimizer": optimiser,
            "lr_scheduler": {"scheduler": combined_scheduler, "interval": "step"},
        }

    def lr_scheduler_step(self, scheduler, *args):
        scheduler.step()
