import argparse
from pathlib import Path

import torch
from sklearn.model_selection import KFold

from utils import ini_argparse, arrange_sparse_minkowski, arrange_truth
from dataset import SparseDataset
from model import MinkUNetConvNeXtV2


def load_fold_model(fold, args, device):
    """Load the best Stage 1 checkpoint for one fold."""

    checkpoint_dir = Path(f"checkpoints_stage1/v1_{fold}")
    checkpoints = sorted(checkpoint_dir.glob("epoch=*.ckpt"))

    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one best checkpoint in {checkpoint_dir}, "
            f"found {len(checkpoints)}."
        )

    checkpoint_path = checkpoints[0]
    print(f"\nLoading Fold {fold} checkpoint: {checkpoint_path}")

    model = MinkUNetConvNeXtV2(
        in_channels=1,
        out_channels=3,
        D=3,
        args=args,
    )

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

    return model


def validate_existing_prediction(path, expected_coordinates):
    """Check an existing prediction before deciding to skip it."""

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid prediction file: {path}")

    if "coords" not in data or "logits" not in data:
        raise RuntimeError(
            f"Prediction file is missing required keys: {path}"
        )

    coords = data["coords"]
    probabilities = data["logits"]

    if not isinstance(coords, torch.Tensor):
        raise RuntimeError(f"Invalid coords tensor in {path}")

    if not isinstance(probabilities, torch.Tensor):
        raise RuntimeError(f"Invalid logits tensor in {path}")

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError(f"Invalid coordinate shape in {path}")

    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise RuntimeError(f"Invalid probability shape in {path}")

    if coords.shape[0] != probabilities.shape[0]:
        raise RuntimeError(f"Coordinate/probability count mismatch in {path}")

    if not torch.equal(
        coords.to(torch.int32),
        expected_coordinates.to(torch.int32),
    ):
        raise RuntimeError(
            f"Existing prediction coordinates do not match the dataset: {path}"
        )

    if not torch.isfinite(probabilities).all():
        raise RuntimeError(f"Non-finite probabilities found in {path}")

    if probabilities.numel() > 0:
        if probabilities.min().item() < 0 or probabilities.max().item() > 1:
            raise RuntimeError(
                f"Values outside probability range [0, 1] in {path}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Run complete five-fold Stage 1 inference for KiTS23."
    )

    parser.add_argument(
        "--output_dir",
        default="results/stage1",
        help="Directory for Stage 1 prediction files.",
    )

    cli_args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. GPU inference is required.")

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

    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct the original five-fold split.
    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    splits = list(kfold.split(range(len(dataset))))

    # Verify that the validation folds partition the complete dataset.
    all_validation_indices = [
        int(index)
        for _, validation_indices in splits
        for index in validation_indices
    ]

    if (
        len(all_validation_indices) != len(dataset)
        or sorted(all_validation_indices) != list(range(len(dataset)))
    ):
        raise RuntimeError(
            "The five validation folds do not cover the dataset exactly once."
        )

    print("===== STAGE 1 FULL INFERENCE =====")
    print(f"Dataset cases: {len(dataset)}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")

    for fold, (_, validation_indices) in enumerate(splits, start=1):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold}/5")
        print(f"Validation cases: {len(validation_indices)}")
        print(f"{'=' * 60}")

        model = load_fold_model(fold, args, device)

        fold_saved = 0
        fold_skipped = 0

        for position, dataset_index in enumerate(validation_indices):
            dataset_index = int(dataset_index)

            # Load one validation case.
            sample = dataset[dataset_index]
            case_id = str(sample["idx"])

            output_path = output_dir / f"{case_id}.pt"

            print(
                f"\nFold {fold} | "
                f"Case {case_id} | "
                f"Validation {position + 1}/{len(validation_indices)}"
            )

            # Construct sparse input.
            batch = {
                "f": sample["x"],
                "c": [sample["c"]],
                "y": sample["y"],
            }

            sparse_input = arrange_sparse_minkowski(batch, device)

            expected_coordinates = sparse_input.coordinates[:, 1:].cpu()

            print(
                f"Input sparse voxels: {sparse_input.F.shape[0]:,}"
            )

            # Resume safely if a valid prediction already exists.
            if output_path.exists():
                validate_existing_prediction(
                    output_path,
                    expected_coordinates,
                )

                print(f"Existing valid prediction found. Skipping: {output_path}")
                fold_skipped += 1

                del sparse_input, batch, sample
                continue

            # Construct target tensor required by the model forward method.
            sparse_target = arrange_truth(batch, device)

            # Run inference.
            with torch.inference_mode():
                outputs, _ = model(sparse_input, sparse_target)

                if not outputs:
                    raise RuntimeError(
                        f"Model returned no predictions for case {case_id}."
                    )

                prediction = outputs[0]

                if (
                    prediction.F.ndim != 2
                    or prediction.F.shape[1] != 3
                ):
                    raise RuntimeError(
                        f"Unexpected prediction shape for case {case_id}: "
                        f"{tuple(prediction.F.shape)}"
                    )

                # Verify exact spatial coordinate alignment.
                if not torch.equal(
                    prediction.coordinates,
                    sparse_input.coordinates,
                ):
                    raise RuntimeError(
                        f"Prediction coordinates differ from input "
                        f"coordinates for case {case_id}."
                    )

                # Convert raw logits to sigmoid probabilities.
                probabilities = torch.sigmoid(prediction.F)

                coordinates = prediction.coordinates[:, 1:]

                if coordinates.shape[0] != probabilities.shape[0]:
                    raise RuntimeError(
                        f"Coordinate/probability count mismatch for {case_id}."
                    )

                if not torch.isfinite(probabilities).all():
                    raise RuntimeError(
                        f"Non-finite predictions found for case {case_id}."
                    )

                coordinates = coordinates.cpu().to(torch.int32)
                probabilities = probabilities.cpu().float()

            # Save using the format expected by process_roi.py.
            torch.save(
                {
                    "coords": coordinates,
                    "logits": probabilities,
                },
                output_path,
            )

            print(f"Saved: {output_path}")
            print(f"Coordinates: {tuple(coordinates.shape)}")
            print(f"Probabilities: {tuple(probabilities.shape)}")

            for channel in range(3):
                count = (
                    probabilities[:, channel] > 0.1
                ).sum().item()

                print(
                    f"Channel {channel} above 0.1: {count:,}"
                )

            fold_saved += 1

            del (
                sparse_input,
                sparse_target,
                prediction,
                outputs,
                probabilities,
                coordinates,
                batch,
                sample,
            )

        del model
        torch.cuda.empty_cache()

        print(f"\nFold {fold} completed.")
        print(f"New predictions saved: {fold_saved}")
        print(f"Existing predictions skipped: {fold_skipped}")

    # Final verification of all expected prediction files.
    print(f"\n{'=' * 60}")
    print("FINAL DATASET VERIFICATION")
    print(f"{'=' * 60}")

    missing_files = []
    invalid_case_ids = []

    for dataset_index in range(len(dataset)):
        sample = dataset[dataset_index]
        case_id = str(sample["idx"])
        prediction_path = output_dir / f"{case_id}.pt"

        if not prediction_path.exists():
            missing_files.append(case_id)

        del sample

    prediction_files = sorted(output_dir.glob("*.pt"))

    print(f"Expected dataset cases: {len(dataset)}")
    print(f"Prediction files found: {len(prediction_files)}")
    print(f"Missing predictions: {len(missing_files)}")

    if missing_files:
        print("Missing case IDs:", missing_files)

    if len(prediction_files) != len(dataset):
        print(
            "Warning: prediction file count differs from dataset size. "
            "Check for unexpected or duplicate files."
        )

    if missing_files or len(prediction_files) != len(dataset):
        raise RuntimeError(
            "Full inference verification failed. "
            "Do not proceed to ROI processing."
        )

    print("\nAll 489 dataset cases have prediction files.")
    print("Stage 1 full inference completed successfully.")


if __name__ == "__main__":
    main()
