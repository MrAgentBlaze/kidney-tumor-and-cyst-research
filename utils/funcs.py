import torch
import MinkowskiEngine as ME
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def sparsify(dense_tensor, label_tensor, roi=None, roi_label=-1, empty_min = -100, empty_max=200, return_mask=False):
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
    for name, module in model.named_modules():
        if isinstance(module, ME.MinkowskiDepthwiseConvolution):
            in_channels = module.in_channels
            kernel_size = module.kernel_generator.kernel_size
            stride = module.kernel_generator.kernel_stride
            dilation = module.kernel_generator.kernel_dilation
            bias = module.bias is not None
            dimension = module.dimension

            # create a new MinkowskiChannelwiseConvolution with the same parameters
            new_conv = ME.MinkowskiChannelwiseConvolution(
                in_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                dimension=dimension
            )

            # copy the weights and bias from old depthwise convolution
            new_conv.kernel = module.kernel
            if bias:
                new_conv.bias = module.bias

            parent_module, attr_name = _get_parent_module(model, name)
            setattr(parent_module, attr_name, new_conv)

    return model


def _get_parent_module(model, layer_name):
    components = layer_name.split('.')
    parent = model
    for comp in components[:-1]:
        parent = getattr(parent, comp)
    return parent, components[-1]


def load_model_from_lightning(model, checkpoint_path, device, strict=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Remove the "model." prefix from the keys in the state_dict
    state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict, strict=strict)
    model = replace_depthwise_with_channelwise(model) if device.type == "cpu" else model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Checkpoint loaded. Total trainable params model (total): {}".format(total_params))

    return model


def get_k_fold_data_loaders(dataset, args, shuffle=True, random_state=None):
    """
    Splits a dataset into K folds and returns DataLoaders for training and validation sets for each fold.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset to be split.
        num_folds (int): The number of folds for cross-validation. Default is 5.
        batch_size (int): Batch size for the DataLoaders. Default is 32.
        shuffle (bool): Whether to shuffle the dataset before splitting into folds. Default is True.
        random_state (int, optional): Seed for reproducibility. Default is None.
    
    Returns:
        List of tuples: Each tuple contains (train_loader, val_loader) for a fold.
    """
    kfold = KFold(n_splits=args.folds, shuffle=shuffle, random_state=random_state)
    data_loaders = []

    for train_indices, val_indices in kfold.split(dataset):
        # Create training and validation subsets
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        # Create DataLoaders for the subsets
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, num_workers=args.num_workers,
                pin_memory=True, persistent_workers=True if args.num_workers > 0 else False,
                collate_fn=collate_sparse_minkowski, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, num_workers=args.num_workers,
                pin_memory=True, persistent_workers=True if args.num_workers > 0 else False,
                collate_fn=collate_sparse_minkowski, shuffle=False)
        
        # Append the loaders as a tuple to the list
        data_loaders.append((train_loader, val_loader))
    
    return data_loaders


def collate_sparse_minkowski(batch):
    idx = [d['idx'] for d in batch]
    coords = [d['c'] for d in batch]
    feats = torch.cat([d['x'] for d in batch])
    y = torch.cat([d['y'] for d in batch])
    
    # Create the return dictionary
    ret = {
        'idx': idx,
        'f': feats,
        'c': coords,
        'y': y,
    }
    
    return ret


def arrange_sparse_minkowski(data, device):
    tensor = ME.SparseTensor(
        features=data['f'],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device)

    return tensor


def arrange_truth(data, device):
    return ME.SparseTensor(
        features=data['y'],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device)


def argsort_sparse_tensor(tensor):
    # Assume coordinates are integers. Create a large enough multiplier to uniquely represent each dimension.
    # Multiply coordinates by powers of a large number to encode them uniquely into one tensor
    max_val = tensor.coordinates.max() + 1
    multipliers = torch.tensor([max_val**i for i in reversed(range(tensor.coordinates.shape[1]))], device=tensor.coordinates.device)
    
    # Create a single sortable tensor
    #encoded_coords = torch.matmul(tensor.coordinates.float(), multipliers.float())
    encoded_coords = (tensor.coordinates * multipliers).sum(dim=1)

    # Sort based on the encoded coordinates
    sorted_indices = torch.argsort(encoded_coords)
    
    return sorted_indices


class CustomLambdaLR(LambdaLR):
    def __init__(self, optimizer, warmup_steps):
        """
        Initialise a custom LambdaLR learning rate scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which the learning rate will be scheduled.
            warmup_steps (int): number of iterations for warm-up.
            lr_func (callable): A function to calculate the learning rate lambda.
        """
        self.warmup_steps = warmup_steps
        super(CustomLambdaLR, self).__init__(optimizer, lr_lambda=self.lr_lambda)

    def lr_lambda(self, step):
        """
        Calculate the learning rate lambda based on the current step and warm-up steps.

        Args:
            step (int): The current step in training.

        Returns:
            float: The learning rate lambda.
        """
        return float(step) / max(1, self.warmup_steps)


class CombinedScheduler(_LRScheduler):
    def __init__(self, optimizer, scheduler1, scheduler2, lr_decay=1.0, warmup_steps=100, start_cosine_step=100):
        """
        Initialize the CombinedScheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimiser for which the learning rate will be scheduled.
            scheduler1 (_LRScheduler): The first scheduler for the warm-up phase.
            scheduler2 (_LRScheduler): The second scheduler for the main phase.
            lr_decay (float): The factor by which the learning rate is decayed after each restart (default: 1.0).
            warmup_steps (int): The number of steps for the warm-up phase (default: 100).
            start_cosine_step (int): The step to start cosine annealing scheduling.
        """
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.warmup_steps = warmup_steps
        self.start_cosine_step = start_cosine_step
        self.step_num = 0  # current scheduler step
        self.lr_decay = lr_decay  # decrease of lr after every restart

    def step(self):
        """
        Update the learning rate based on the current step and the selected scheduler.
        This method alternates between the two provided schedulers based on the current step number.
        After the warm-up phase, it switches to the second scheduler and optionally decays the learning
        rate after each restart.
        """
        if self.step_num < self.warmup_steps:
            self.scheduler1.step()
        elif self.step_num >= self.start_cosine_step:
            self.scheduler2.step()
            if self.lr_decay < 1.0 and (self.scheduler2.T_cur+1 == self.scheduler2.T_i):
                # Reduce the learning rate after every restart
                self.scheduler2.base_lrs[0] *= self.lr_decay
        self.step_num += 1

    def state_dict(self):
        """Return the state of the scheduler."""
        return {
            'warmup_steps': self.warmup_steps,
            'start_cosine_step': self.start_cosine_step,
            'step_num': self.step_num,
            'lr_decay': self.lr_decay,
            'scheduler1': self.scheduler1.state_dict(),
            'scheduler2': self.scheduler2.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Load the scheduler state."""
        self.warmup_steps = state_dict['warmup_steps']
        self.start_cosine_step = state_dict['start_cosine_step']
        self.step_num = state_dict['step_num']
        self.lr_decay = state_dict['lr_decay']
        self.scheduler1.load_state_dict(state_dict['scheduler1'])
        self.scheduler2.load_state_dict(state_dict['scheduler2'])


def replace_depthwise_with_channelwise(model):
    for name, module in model.named_modules():
        if isinstance(module, ME.MinkowskiDepthwiseConvolution):
            in_channels = module.in_channels
            kernel_size = module.kernel_generator.kernel_size
            stride = module.kernel_generator.kernel_stride
            dilation = module.kernel_generator.kernel_dilation
            bias = module.bias is not None
            dimension = module.dimension
            
            # create a new MinkowskiChannelwiseConvolution with the same parameters
            new_conv = ME.MinkowskiChannelwiseConvolution(
                in_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                dimension=dimension
            )
            
            # copy the weights and bias from old depthwise convolution
            new_conv.kernel = module.kernel
            if bias:
                new_conv.bias = module.bias
            
            parent_module, attr_name = _get_parent_module(model, name)
            setattr(parent_module, attr_name, new_conv)
    
    return model


def _get_parent_module(model, layer_name):
    components = layer_name.split('.')
    parent = model
    for comp in components[:-1]:
        parent = getattr(parent, comp)
    return parent, components[-1]

