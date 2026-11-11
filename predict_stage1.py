import argparse
from pathlib import Path

import torch
import MinkowskiEngine as ME
from sklearn.model_selection import KFold

from utils import ini_argparse, arrange_sparse_minkowski, arrange_truth
from dataset import SparseDataset
from model import MinkUNetConvNeXtV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1,
                        choices=[1, 2, 3, 4, 5])
    parser.add_argument("--val_position", type=int, default=0)
    parser.add_argument("--output_dir", default="results/stage1")
    cli_args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This test expects your GPU.")

    device = torch.device("cuda")

    # Reconstruct the exact Stage 1 dataset configuration.
    args = ini_argparse().parse_args([])
    args.dataset_name = "kits23_large_processed"
    args.dataset_path = "data/{}/*"
    args.min_hu = -53.4
    args.max_hu = 283.2
    args.ds_steps = 3
    args.target = -1
    args.stage2 = False

    dataset = SparseDataset(args)
    dataset.training = False

    if len(dataset) != 489:
        raise RuntimeError(
            f"Expected 489 Stage 1 cases, found {len(dataset)}."
        )

    # Reconstruct the original five-fold split.
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kfold.split(range(len(dataset))))

    fold_index = cli_args.fold - 1
    _, val_indices = splits[fold_index]

    if cli_args.val_position >= len(val_indices):
        raise IndexError(
            f"Validation position {cli_args.val_position} is out of range "
            f"for Fold {cli_args.fold} ({len(val_indices)} cases)."
        )

    dataset_index = int(val_indices[cli_args.val_position])

    print(f"Fold: {cli_args.fold}")
    print(f"Validation position: {cli_args.val_position}")
    print(f"Dataset index: {dataset_index}")

    # Load one validation case.
    sample = dataset[dataset_index]
    case_id = sample["idx"]

    print(f"Case ID: {case_id}")

    # Locate the best checkpoint for this fold.
    checkpoint_dir = Path(f"checkpoints_stage1/v1_{cli_args.fold}")
    checkpoints = sorted(checkpoint_dir.glob("epoch=*.ckpt"))

    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one best checkpoint in {checkpoint_dir}, "
            f"found {len(checkpoints)}."
        )

    checkpoint_path = checkpoints[0]
    print(f"Checkpoint: {checkpoint_path}")

    # Construct the Stage 1 model.
    model = MinkUNetConvNeXtV2(
        in_channels=1,
        out_channels=3,
        D=3,
        args=args,
    )

    # Load the Lightning checkpoint strictly.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
    }

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    model.activation_checkpointing = False

    print("Checkpoint loaded successfully.")

    # Construct sparse input and target tensors.
    batch = {
        "f": sample["x"],
        "c": [sample["c"]],
        "y": sample["y"],
    }

    sparse_input = arrange_sparse_minkowski(batch, device)
    sparse_target = arrange_truth(batch, device)

    print(f"Input sparse voxels: {sparse_input.F.shape[0]}")

    # Run inference.
    with torch.inference_mode():
        outputs, _ = model(sparse_input, sparse_target)

        if not outputs:
            raise RuntimeError("The model returned no prediction outputs.")

        prediction = outputs[0]

        if prediction.F.ndim != 2 or prediction.F.shape[1] != 3:
            raise RuntimeError(
                f"Expected prediction shape [N, 3], "
                f"received {tuple(prediction.F.shape)}."
            )

        # The model returns raw logits. Convert them to probabilities.
        probabilities = torch.sigmoid(prediction.F)

        # MinkowskiEngine coordinates contain [batch, x, y, z].
        coordinates = prediction.coordinates

        if coordinates.ndim != 2 or coordinates.shape[1] != 4:
            raise RuntimeError(
                f"Expected coordinates with four columns, "
                f"received {tuple(coordinates.shape)}."
            )

        # Remove the batch-index column for process_roi.py.
        coordinates = coordinates[:, 1:]

        if coordinates.shape[0] != probabilities.shape[0]:
            raise RuntimeError("Coordinate and prediction counts differ.")

        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Predictions contain NaN or infinity.")

        coordinates = coordinates.cpu().to(torch.int32)
        probabilities = probabilities.cpu().float()

    # Save using the key names expected by process_roi.py.
    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{case_id}.pt"

    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. "
            "Refusing to overwrite an existing prediction."
        )

    torch.save(
        {
            "coords": coordinates,
            "logits": probabilities,
        },
        output_path,
    )

    print("\n===== INFERENCE RESULT =====")
    print(f"Saved: {output_path}")
    print(f"Coordinates: {tuple(coordinates.shape)}")
    print(f"Probabilities: {tuple(probabilities.shape)}")
    print(f"Probability range: "
          f"{probabilities.min().item():.6f} - "
          f"{probabilities.max().item():.6f}")

    for channel in range(3):
        count = (probabilities[:, channel] > 0.1).sum().item()
        print(f"Channel {channel}: {count:,} voxels above 0.1")

    print("\nSingle-case inference completed successfully.")


if __name__ == "__main__":
    main()
