import torch
import MinkowskiEngine as ME
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


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
    coords = [d['c'] for d in batch]
    feats = torch.cat([d['x'] for d in batch])
    y = torch.cat([d['y'] for d in batch])

    # Create the return dictionary
    ret = {
        'f': feats,
        'c': coords,
        'y': y,
    }
    
    return ret


def arrange_sparse_minkowski(data, device):
    return ME.SparseTensor(
        features=data['f'],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        #quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE, 
        device=device)


def arrange_truth(data, device):
    return ME.SparseTensor(
        features=data['y'],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        #quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE, 
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
    def __init__(self, optimizer, scheduler1, scheduler2, lr_decay=1.0, warmup_steps=100):
        """
        Initialize the CombinedScheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimiser for which the learning rate will be scheduled.
            scheduler1 (_LRScheduler): The first scheduler for the warm-up phase.
            scheduler2 (_LRScheduler): The second scheduler for the main phase.
            lr_decay (float): The factor by which the learning rate is decayed after each restart (default: 1.0).
            warmup_steps (int): The number of steps for the warm-up phase (default: 100).
        """
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.warmup_steps = warmup_steps
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
        else:
            self.scheduler2.step()
            if self.lr_decay < 1.0 and (self.scheduler2.T_cur+1 == self.scheduler2.T_i):
                # Reduce the learning rate after every restart
                self.scheduler2.base_lrs[0] *= self.lr_decay
        self.step_num += 1



