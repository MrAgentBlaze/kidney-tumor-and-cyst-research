#!/usr/bin/env python3
"""Benchmark SegVol inference time and peak VRAM on KiTS23.

Loads pretrained SegVol and runs text-prompted inference on all 489 KiTS23
volumes using their real spatial dimensions.

Usage:
    conda activate segvol

    python baselines/benchmark_segvol_inference.py --gpu 0
    python baselines/benchmark_segvol_inference.py --gpu 0 --use_zoom
    python baselines/benchmark_segvol_inference.py --gpu 0 --max_cases 20
"""

import os, sys, gc, json, time, argparse, importlib
import numpy as np
import torch
import torch.nn.functional as F

KITS23_DIR = "/scratch2/salonso/medical/kits23/dataset"


def _load_segvol_model(device):
    """Load SegVol with pretrained weights from HuggingFace."""
    from huggingface_hub import hf_hub_download
    from transformers import CLIPTokenizer

    snapshot_dir = os.path.dirname(
        hf_hub_download("BAAI/SegVol", "model_segvol_single.py"))
    weights_path = hf_hub_download("BAAI/SegVol", "pytorch_model.bin")

    spec = importlib.util.spec_from_file_location(
        "segvol_model",
        os.path.join(snapshot_dir, "model_segvol_single.py"))
    segvol_module = importlib.util.module_from_spec(spec)
    sys.modules["segvol_model"] = segvol_module
    spec.loader.exec_module(segvol_module)

    spatial_size = [32, 256, 256]
    patch_size = [4, 16, 16]

    sam_model = segvol_module._build_sam(
        image_encoder_type="vit", embed_dim=768, patch_size=patch_size,
        checkpoint=None, image_size=spatial_size)
    model = segvol_module.SegVol(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        roi_size=spatial_size, patch_size=patch_size, test_mode=True)

    state_dict = torch.load(weights_path, map_location="cpu")
    new_sd = {(k[len("model."):] if k.startswith("model.") else k): v
              for k, v in state_dict.items()}
    model.load_state_dict(new_sd, strict=False)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model.text_encoder.tokenizer = tokenizer
    model = model.to(device).eval()

    processor = segvol_module.SegVolProcessor(spatial_size=spatial_size)
    _orig_loader = processor.img_loader
    def _safe_load(path):
        data = _orig_loader(path)
        return data[0] if isinstance(data, tuple) else data
    processor.img_loader = _safe_load

    logits2roi_coor = segvol_module.logits2roi_coor
    sw_inference = segvol_module.sliding_window_inference
    return model, processor, spatial_size, logits2roi_coor, sw_inference


def get_all_case_shapes():
    """Get original volume shapes of all KiTS23 cases from NIfTI headers."""
    import nibabel as nib
    from glob import glob
    cases = sorted(glob(os.path.join(KITS23_DIR, "case_*")))
    shapes = {}
    for case_dir in cases:
        case_id = os.path.basename(case_dir)
        img_path = os.path.join(case_dir, "imaging.nii.gz")
        if not os.path.exists(img_path):
            continue
        img = nib.load(img_path)
        shapes[case_id] = tuple(img.header.get_data_shape())
    return shapes


def run_benchmark(model, processor, spatial_size, shapes, device, use_gpu,
                  use_zoom=False, logits2roi_coor_fn=None,
                  sw_inference_fn=None, warmup=5, max_cases=None):
    """Run SegVol inference on all volumes and measure time + VRAM."""
    model.eval()
    categories = ["kidney", "kidney tumor", "kidney cyst"]
    zoom_shape = tuple(spatial_size)

    case_ids = sorted(shapes.keys())
    if max_cases:
        case_ids = case_ids[:max_cases]
    if not case_ids:
        return []
    if use_zoom and (logits2roi_coor_fn is None or sw_inference_fn is None):
        raise ValueError("Zoom mode requires logits2roi_coor_fn and sw_inference_fn.")

    # Warm-up
    print(f"\n  Warm-up ({warmup} iterations)...")
    warmup_shape = shapes[case_ids[len(case_ids) // 2]]
    for _ in range(warmup):
        dummy = torch.randn(1, 1, *zoom_shape, device=device)
        image_full = (torch.randn(1, 1, *warmup_shape, device=device)
                      if use_zoom else None)
        with torch.no_grad():
            for text_prompt in categories:
                logits = model(dummy, text=[text_prompt])
                if use_zoom:
                    logits_up = F.interpolate(logits.cpu(), size=warmup_shape,
                                              mode="nearest")
                    min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor_fn(
                        spatial_size, logits_up[0][0])
                    if min_d is not None:
                        image_cropped = image_full[:, :, min_d:max_d+1,
                                                   min_h:max_h+1, min_w:max_w+1]
                        global_preds_crop = (
                            torch.sigmoid(logits_up[:, :, min_d:max_d+1,
                                                    min_h:max_h+1,
                                                    min_w:max_w+1]) > 0.5).long()
                        prompt_reflection = (global_preds_crop.float().to(device),
                                             global_preds_crop.to(device))
                        logits_cropped = sw_inference_fn(
                            image_cropped.to(device), prompt_reflection,
                            spatial_size, 1, model, 0.5,
                            text=[text_prompt], use_box=True, use_point=False)
                        del logits_cropped, image_cropped, global_preds_crop, prompt_reflection
                    del logits_up
                del logits
        if use_gpu:
            torch.cuda.synchronize(device)
        del dummy, image_full
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

    # Timed runs
    print(f"  Benchmarking {len(case_ids)} cases...")
    results = []

    for i, case_id in enumerate(case_ids):
        original_shape = shapes[case_id]
        zoomed_input = torch.randn(1, 1, *zoom_shape, device=device)
        image_full = (torch.randn(1, 1, *original_shape, device=device)
                      if use_zoom else None)

        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                for text_prompt in categories:
                    logits_global = model(zoomed_input, text=[text_prompt])
                    logits_global = F.interpolate(
                        logits_global.cpu(), size=original_shape, mode="nearest")

                    if use_zoom:
                        min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor_fn(
                            spatial_size, logits_global[0][0])
                        if min_d is not None:
                            image_cropped = image_full[:, :, min_d:max_d+1,
                                                       min_h:max_h+1, min_w:max_w+1]
                            global_preds_crop = (
                                torch.sigmoid(logits_global[:, :, min_d:max_d+1,
                                                            min_h:max_h+1,
                                                            min_w:max_w+1]) > 0.5).long()
                            prompt_reflection = (global_preds_crop.float().to(device),
                                                 global_preds_crop.to(device))
                            logits_cropped = sw_inference_fn(
                                image_cropped.to(device), prompt_reflection,
                                spatial_size, 1, model, 0.5,
                                text=[text_prompt], use_box=True, use_point=False)
                            logits_cropped = logits_cropped.cpu().squeeze()
                            logits_global[:, :, min_d:max_d+1,
                                          min_h:max_h+1, min_w:max_w+1] = logits_cropped
                            del logits_cropped, image_cropped, global_preds_crop, prompt_reflection

                    pred = (torch.sigmoid(logits_global.squeeze()) > 0.5)
                    del logits_global, pred

            if use_gpu:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            peak_vram = (torch.cuda.max_memory_allocated(device) / 1024**3
                         if use_gpu else 0.0)
            results.append({
                "case_id": case_id, "shape": original_shape,
                "time_s": elapsed, "peak_vram_gb": peak_vram, "oom": False})

        except torch.cuda.OutOfMemoryError:
            elapsed = time.perf_counter() - t0
            results.append({
                "case_id": case_id, "shape": original_shape,
                "time_s": float("nan"), "peak_vram_gb": float("nan"), "oom": True})
            gc.collect()
            if use_gpu:
                torch.cuda.empty_cache()

        del zoomed_input
        if image_full is not None:
            del image_full
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
                  f"shape={original_shape}  {status}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SegVol inference time and VRAM on KiTS23.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gpu", type=int, help="GPU index")
    group.add_argument("--cpu", action="store_true", help="Run on CPU")
    parser.add_argument("--use_zoom", action="store_true",
                        help="Enable zoom-out + zoom-in pipeline")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON")
    args = parser.parse_args()

    use_gpu = not args.cpu
    if use_gpu:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    print("=" * 64)
    print("  SegVol Inference Benchmark — KiTS23")
    print("=" * 64)
    if use_gpu:
        print(f"  GPU          : {torch.cuda.get_device_name(device)}")
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"  GPU VRAM     : {total_gb:.1f} GB")
    else:
        print("  Device       : CPU")
    print(f"  Zoom-out res : [32, 256, 256]")
    print(f"  Zoom-in      : {'Yes' if args.use_zoom else 'No (zoom-out only)'}")
    print(f"  Text prompts : kidney, kidney tumor, kidney cyst (3 passes)")

    print("\nLoading SegVol from HuggingFace (BAAI/SegVol)...")
    (model, processor, spatial_size,
     logits2roi_coor_fn, sw_inference_fn) = _load_segvol_model(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters   : {num_params:,}")

    print("\nCollecting volume shapes...")
    shapes = get_all_case_shapes()
    if args.max_cases:
        case_ids = sorted(shapes.keys())[:args.max_cases]
        shapes = {k: shapes[k] for k in case_ids}
    print(f"  Cases        : {len(shapes)}")

    results = run_benchmark(model, processor, spatial_size, shapes,
                            device, use_gpu, use_zoom=args.use_zoom,
                            logits2roi_coor_fn=logits2roi_coor_fn,
                            sw_inference_fn=sw_inference_fn,
                            warmup=args.warmup, max_cases=args.max_cases)

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
                "use_zoom": args.use_zoom,
                "zoom_out_resolution": list(spatial_size),
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
