import torch
import copy
import MinkowskiEngine as ME
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def sparsify(dense_tensor, label_tensor, roi=None, roi_label=-1,
             empty_min=-100, empty_max=200, return_mask=False):
    """Convert a dense tensor to sparse representation by thresholding."""
    mask = (dense_tensor >= empty_min) & (dense_tensor <= empty_max)
    if roi is not None:
        mask = mask & (roi == roi_label)
    if return_mask:
        return mask
    coords = torch.argwhere(mask[0])
    feats = dense_tensor[mask].unsqueeze(1)
    labels = label_tensor[mask].unsqueeze(1)
    return coords, feats, labels


def replace_depthwise_with_channelwise(model):
    """Replace MinkowskiDepthwiseConvolution with MinkowskiChannelwiseConvolution.

    Required for CPU inference, as the custom CUDA kernel for depthwise
    convolutions does not support CPU execution.
    """
    for name, module in model.named_modules():
        if isinstance(module, ME.MinkowskiDepthwiseConvolution):
            in_channels = module.in_channels
            kernel_size = module.kernel_generator.kernel_size
            stride = module.kernel_generator.kernel_stride
            dilation = module.kernel_generator.kernel_dilation
            bias = module.bias is not None
            dimension = module.dimension

            new_conv = ME.MinkowskiChannelwiseConvolution(
                in_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                dimension=dimension,
            )
            new_conv.kernel = module.kernel
            if bias:
                new_conv.bias = module.bias

            parent, attr = _get_parent_module(model, name)
            setattr(parent, attr, new_conv)
    return model


def _get_parent_module(model, layer_name):
    """Get the parent module and attribute name for a nested layer."""
    components = layer_name.split(".")
    parent = model
    for comp in components[:-1]:
        parent = getattr(parent, comp)
    return parent, components[-1]


def load_model_from_lightning(model, checkpoint_path, device, strict=False):
    """Load model weights from a PyTorch Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {
        key.replace("model.", ""): value
        for key, value in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict, strict=strict)
    if device.type == "cpu":
        model = replace_depthwise_with_channelwise(model)
    else:
        model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Checkpoint loaded. Total trainable parameters: {total_params}")
    return model


def get_k_fold_data_loaders(dataset, args, shuffle=True, random_state=None):
    """Split dataset into K folds and return DataLoaders for each fold."""
    kfold = KFold(n_splits=args.folds, shuffle=shuffle, random_state=random_state)
    data_loaders = []

    for train_indices, val_indices in kfold.split(dataset):
        train_subset = Subset(copy.deepcopy(dataset), train_indices)
        train_subset.dataset.training = True
        val_subset = Subset(copy.deepcopy(dataset), val_indices)

        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            collate_fn=collate_sparse_minkowski, shuffle=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            collate_fn=collate_sparse_minkowski, shuffle=False,
        )
        data_loaders.append((train_loader, val_loader))
    return data_loaders


def collate_sparse_minkowski(batch):
    """Collate function for batching sparse data for MinkowskiEngine."""
    return {
        "idx": [d["idx"] for d in batch],
        "f": torch.cat([d["x"] for d in batch]),
        "c": [d["c"] for d in batch],
        "y": torch.cat([d["y"] for d in batch]),
    }


def arrange_sparse_minkowski(data, device):
    """Create a MinkowskiEngine SparseTensor from batched input data."""
    return ME.SparseTensor(
        features=data["f"],
        coordinates=ME.utils.batched_coordinates(data["c"], dtype=torch.int),
        device=device,
    )


def arrange_truth(data, device):
    """Create a MinkowskiEngine SparseTensor from batched target data."""
    return ME.SparseTensor(
        features=data["y"],
        coordinates=ME.utils.batched_coordinates(data["c"], dtype=torch.int),
        device=device,
    )


def argsort_sparse_tensor(tensor):
    """Return indices that sort a sparse tensor by its coordinates."""
    max_val = tensor.coordinates.max() + 1
    multipliers = torch.tensor(
        [max_val ** i for i in reversed(range(tensor.coordinates.shape[1]))],
        device=tensor.coordinates.device,
    )
    encoded_coords = (tensor.coordinates * multipliers).sum(dim=1)
    return torch.argsort(encoded_coords)


class CustomLambdaLR(LambdaLR):
    """Linear warm-up learning rate scheduler."""

    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, lr_lambda=self.lr_lambda)

    def lr_lambda(self, step):
        return float(step) / max(1, self.warmup_steps)


class CombinedScheduler(_LRScheduler):
    """Scheduler combining linear warm-up with cosine annealing."""

    def __init__(self, optimizer, scheduler1, scheduler2,
                 lr_decay=1.0, warmup_steps=100, start_cosine_step=100):
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.warmup_steps = warmup_steps
        self.start_cosine_step = start_cosine_step
        self.step_num = 0
        self.lr_decay = lr_decay

    def step(self):
        if self.step_num < self.warmup_steps:
            self.scheduler1.step()
        elif self.step_num >= self.start_cosine_step:
            self.scheduler2.step()
            if self.lr_decay < 1.0 and (self.scheduler2.T_cur + 1 == self.scheduler2.T_i):
                self.scheduler2.base_lrs[0] *= self.lr_decay
        self.step_num += 1

    def state_dict(self):
        return {
            "warmup_steps": self.warmup_steps,
            "start_cosine_step": self.start_cosine_step,
            "step_num": self.step_num,
            "lr_decay": self.lr_decay,
            "scheduler1": self.scheduler1.state_dict(),
            "scheduler2": self.scheduler2.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.warmup_steps = state_dict["warmup_steps"]
        self.start_cosine_step = state_dict["start_cosine_step"]
        self.step_num = state_dict["step_num"]
        self.lr_decay = state_dict["lr_decay"]
        self.scheduler1.load_state_dict(state_dict["scheduler1"])
        self.scheduler2.load_state_dict(state_dict["scheduler2"])
