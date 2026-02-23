#!/usr/bin/env python3
"""Benchmark nnU-Net inference time and peak VRAM on KiTS23.

Loads the trained nnU-Net model and runs sliding-window inference on all
489 KiTS23 volumes using their real preprocessed shapes.

nnU-Net auto-configured parameters for KiTS23:
    Patch size : [128, 128, 128]
    Spacing    : [1.0, 0.78125, 0.78125] mm
    Overlap    : 0.5

Usage:
    conda activate convnextv2

    python baselines/benchmark_nnunet_inference.py --gpu 0
    python baselines/benchmark_nnunet_inference.py --gpu 0 --mirror
    python baselines/benchmark_nnunet_inference.py --gpu 0 --max_cases 20
"""

import os, sys, gc, glob, json, time, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from monai.inferers import sliding_window_inference

NNUNET_RESULTS_DIR = (
    "/scratch2/salonso/medical/data_revision/nnunet_results/"
    "Dataset001_KiTS23/nnUNetTrainer__nnUNetPlans__3d_fullres"
)
NNUNET_PREPROCESSED_DIR = (
    "/scratch2/salonso/medical/data_revision/nnunet_preprocessed/"
    "Dataset001_KiTS23"
)
PLANS_FILE = os.path.join(NNUNET_PREPROCESSED_DIR, "nnUNetPlans.json")
PREPROC_3D = os.path.join(NNUNET_PREPROCESSED_DIR, "nnUNetPlans_3d_fullres")

PATCH_SIZE = (128, 128, 128)
NUM_CLASSES = 4


def build_nnunet(deep_supervision=True):
    """Build PlainConvUNet matching nnU-Net's auto-config for KiTS23."""
    return PlainConvUNet(
        input_channels=1, n_stages=6,
        features_per_stage=(32, 64, 128, 256, 320, 320),
        conv_op=nn.Conv3d,
        kernel_sizes=((3, 3, 3),) * 6,
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2),
                 (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        n_conv_per_stage=(2, 2, 2, 2, 2, 2),
        n_conv_per_stage_decoder=(2, 2, 2, 2, 2),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
        num_classes=NUM_CLASSES,
    )


def load_trained_weights(model, fold=0):
    ckpt_path = os.path.join(
        NNUNET_RESULTS_DIR, f"fold_{fold}", "checkpoint_final.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["network_weights"])
    print(f"  Loaded weights from fold {fold}: {ckpt_path}")
    return model


def get_resampled_shapes():
    """Compute volume shapes at nnU-Net's target spacing for each case."""
    with open(PLANS_FILE) as f:
        plans = json.load(f)
    target_spacing = np.array(
        plans["configurations"]["3d_fullres"]["spacing"])

    pkls = sorted(glob.glob(os.path.join(PREPROC_3D, "*.pkl")))
    shapes = {}
    for pkl_path in pkls:
        case_id = os.path.basename(pkl_path).replace(".pkl", "")
        with open(pkl_path, "rb") as f:
            props = pickle.load(f)
        orig_spacing = np.array(props["spacing"])
        orig_shape = np.array(props["shape_after_cropping_and_before_resampling"])
        new_shape = np.round(
            orig_shape * orig_spacing[::-1] / target_spacing).astype(int)
        new_shape = np.maximum(new_shape, PATCH_SIZE)
        shapes[case_id] = tuple(new_shape.tolist())
    return shapes


def run_benchmark(model, shapes, device, use_gpu, mirror=False,
                  warmup=5, overlap=0.5):
    model.eval()

    def predictor(x):
        out = model(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    case_ids = sorted(shapes.keys())

    print(f"\n  Warm-up ({warmup} iterations)...")
    warmup_shape = shapes[case_ids[len(case_ids) // 2]]
    for _ in range(warmup):
        dummy = torch.randn(1, 1, *warmup_shape, device=device)
        with torch.no_grad():
            if use_gpu:
                with torch.amp.autocast("cuda"):
                    _ = sliding_window_inference(
                        dummy, PATCH_SIZE, 1, predictor, overlap=overlap)
                torch.cuda.synchronize(device)
            else:
                _ = sliding_window_inference(
                    dummy, PATCH_SIZE, 1, predictor, overlap=overlap)
        del dummy
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

    print(f"  Benchmarking {len(case_ids)} cases...")
    results = []

    for i, case_id in enumerate(case_ids):
        shape = shapes[case_id]
        dummy = torch.randn(1, 1, *shape, device=device)

        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                if use_gpu:
                    with torch.amp.autocast("cuda"):
                        logits = sliding_window_inference(
                            dummy, PATCH_SIZE, 1, predictor, overlap=overlap)
                        if mirror:
                            for axes in [(2,), (3,), (4,),
                                         (2, 3), (2, 4), (3, 4),
                                         (2, 3, 4)]:
                                flipped = torch.flip(dummy, axes)
                                out = sliding_window_inference(
                                    flipped, PATCH_SIZE, 1, predictor,
                                    overlap=overlap)
                                logits += torch.flip(out, axes)
                            logits /= 8.0
                    torch.cuda.synchronize(device)
                else:
                    logits = sliding_window_inference(
                        dummy, PATCH_SIZE, 1, predictor, overlap=overlap)
                    if mirror:
                        for axes in [(2,), (3,), (4,),
                                     (2, 3), (2, 4), (3, 4),
                                     (2, 3, 4)]:
                            flipped = torch.flip(dummy, axes)
                            out = sliding_window_inference(
                                flipped, PATCH_SIZE, 1, predictor,
                                overlap=overlap)
                            logits += torch.flip(out, axes)
                        logits /= 8.0

            elapsed = time.perf_counter() - t0
            peak_vram = (torch.cuda.max_memory_allocated(device) / 1024**3
                         if use_gpu else 0.0)
            results.append({
                "case_id": case_id, "shape": shape,
                "time_s": elapsed, "peak_vram_gb": peak_vram, "oom": False})

        except torch.cuda.OutOfMemoryError:
            elapsed = time.perf_counter() - t0
            results.append({
                "case_id": case_id, "shape": shape,
                "time_s": float("nan"), "peak_vram_gb": float("nan"), "oom": True})
            gc.collect()
            torch.cuda.empty_cache()

        del dummy
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

        if (i + 1) % 25 == 0 or (i + 1) == len(case_ids):
            r = results[-1]
            status = ("OOM" if r["oom"]
                      else f"{r['time_s']:.2f}s"
                           + (f"  VRAM={r['peak_vram_gb']:.2f}GB"
                              if use_gpu else ""))
            print(f"    [{i+1:3d}/{len(case_ids)}] {case_id} "
                  f"shape={shape}  {status}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark nnU-Net inference time and VRAM on KiTS23.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gpu", type=int, help="GPU index")
    group.add_argument("--cpu", action="store_true", help="Run on CPU")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold to load weights from (default: 0)")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--mirror", action="store_true",
                        help="Test-time mirroring (8x forward passes)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    use_gpu = not args.cpu
    if use_gpu:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    print("=" * 64)
    print("  nnU-Net Inference Benchmark — KiTS23")
    print("=" * 64)
    if use_gpu:
        print(f"  GPU          : {torch.cuda.get_device_name(device)}")
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"  GPU VRAM     : {total_gb:.1f} GB")
    else:
        print("  Device       : CPU")
    print(f"  Patch size   : {PATCH_SIZE}")
    print(f"  Overlap      : {args.overlap}")
    print(f"  Mirroring    : {'Yes (8x)' if args.mirror else 'No'}")

    model = build_nnunet(deep_supervision=True)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters   : {num_params:,}")
    model = load_trained_weights(model, fold=args.fold)
    model.to(device)

    shapes = get_resampled_shapes()
    if args.max_cases:
        case_ids = sorted(shapes.keys())[:args.max_cases]
        shapes = {k: shapes[k] for k in case_ids}
    print(f"  Cases        : {len(shapes)}")

    results = run_benchmark(model, shapes, device, use_gpu,
                            mirror=args.mirror, warmup=args.warmup,
                            overlap=args.overlap)

    valid = [r for r in results if not r["oom"]]
    oom_count = sum(1 for r in results if r["oom"])

    print("\n" + "=" * 64)
    print("  Results")
    print("=" * 64)
    if valid:
        times = np.array([r["time_s"] for r in valid])
        print(f"  Inference time : {times.mean():.2f} +/- {times.std():.2f} s")
        if use_gpu:
            vrams = np.array([r["peak_vram_gb"] for r in valid])
            print(f"  Peak VRAM      : {vrams.mean():.2f} +/- {vrams.std():.2f} GB")
    if oom_count:
        print(f"  OOM            : {oom_count}/{len(results)} cases")
    print(f"  Parameters     : {num_params:,}")

    if valid:
        t_mean, t_std = times.mean(), times.std()
        if use_gpu:
            v_mean, v_std = vrams.mean(), vrams.std()
            print(f"\n  LaTeX (time) : ${t_mean:.2f} \\pm {t_std:.2f}$")
            print(f"  LaTeX (VRAM) : ${v_mean:.2f} \\pm {v_std:.2f}$")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "device": ("cpu" if args.cpu
                           else f"cuda:{args.gpu} ({torch.cuda.get_device_name(device)})"),
                "patch_size": list(PATCH_SIZE),
                "overlap": args.overlap,
                "mirror": args.mirror,
                "num_params": num_params,
                "num_cases": len(results),
                "num_oom": oom_count,
                "mean_time_s": float(times.mean()) if valid else None,
                "std_time_s": float(times.std()) if valid else None,
                "mean_vram_gb": float(vrams.mean()) if valid and use_gpu else None,
                "std_vram_gb": float(vrams.std()) if valid and use_gpu else None,
                "per_case": results,
            }, f, indent=2, default=str)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
