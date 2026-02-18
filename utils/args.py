import argparse


def ini_argparse():
    parser = argparse.ArgumentParser(
        description="Sparse 3D U-Net for kidney and tumour segmentation in CT."
    )

    # Task configuration
    parser.add_argument("--target", type=int, default=-1,
                        help="Target label (0, 1, 2) or -1 for all")
    parser.add_argument("--train", action="store_true", default=True,
                        help="Training mode")
    parser.add_argument("--test", action="store_false", dest="train",
                        help="Testing mode")
    parser.add_argument("--stage2", action="store_true", default=False,
                        help="Stage 2 (high-resolution segmentation)")
    parser.add_argument("--sigmoid", action="store_true", default=True,
                        help="Sigmoid activation (binary)")
    parser.add_argument("--softmax", action="store_false", dest="sigmoid",
                        help="Softmax activation (multi-class)")
    parser.add_argument("--sparse", action="store_true", default=True,
                        help="Use sparse representation")
    parser.add_argument("--dense", action="store_false", dest="sparse",
                        help="Use dense representation")
    parser.add_argument("--roi", action="store_true", default=False,
                        help="ROI mode")

    # Dataset
    parser.add_argument("--dataset_name", type=str, default="kits23_large_processed",
                        help="Dataset name")
    parser.add_argument("-d", "--dataset_path", type=str,
                        default="data/{}/*",
                        help="Dataset path template")
    parser.add_argument("--min_hu", type=float, default=-53.4,
                        help="Minimum Hounsfield units threshold")
    parser.add_argument("--max_hu", type=float, default=283.2,
                        help="Maximum Hounsfield units threshold")

    # Model
    parser.add_argument("--ds_steps", type=int, default=3,
                        help="Deep supervision steps")
    parser.add_argument("--eps", type=float, default=1e-12,
                        help="Smoothing constant to prevent division by zero")

    # Training
    parser.add_argument("-b", "--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of folds for K-fold cross-validation")
    parser.add_argument("-e", "--epochs", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=8,
                        help="Number of data loader workers")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("-ag", "--accum_grad_batches", type=int, default=8,
                        help="Gradient accumulation batches")
    parser.add_argument("-ws", "--warmup_steps", type=int, default=1,
                        help="Number of warm-up epochs")
    parser.add_argument("--cosine_annealing_steps", type=int, default=400,
                        help="Number of cosine annealing epochs")
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("-b1", "--beta1", type=float, default=0.9,
                        help="AdamW beta1")
    parser.add_argument("-b2", "--beta2", type=float, default=0.95,
                        help="AdamW beta2")
    parser.add_argument("--losses", nargs="*", default=["dice"],
                        help='Loss functions (options: "focal", "dice", "ce")')

    # Logging and checkpointing
    parser.add_argument("--save_dir", type=str, default="logs",
                        help="Log save directory")
    parser.add_argument("--name", type=str, default="v1",
                        help="Experiment name")
    parser.add_argument("--log_every_n_steps", type=int, default=5,
                        help="Steps between logs")
    parser.add_argument("--save_top_k", type=int, default=1,
                        help="Number of top checkpoints to save")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--checkpoint_name", type=str, default="v1",
                        help="Checkpoint name")
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--gpus", nargs="*", default=[0],
                        help="GPU device IDs")

    return parser
