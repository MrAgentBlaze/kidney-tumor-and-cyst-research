#!/usr/bin/env python3
"""Evaluate SegVol (zero-shot, text-prompted) on KiTS23 with per-fold grouped Dice.

Uses the pretrained SegVol model from HuggingFace (BAAI/SegVol) with its
zoom-out / zoom-in inference strategy and text prompts matching the M3D-Seg
vocabulary: "kidney", "kidney tumor", "kidney cyst".

Usage:
    conda activate segvol

    # All folds (zoom-out only)
    python baselines/run_segvol.py

    # All folds with zoom-in refinement
    python baselines/run_segvol.py --use_zoom

    # Single fold, quick test
    python baselines/run_segvol.py --fold 0 --max_cases 5
"""

import os, sys, json, gc, time, argparse, importlib
import numpy as np
import torch
import torch.nn.functional as F

KITS23_DIR = "/scratch2/salonso/medical/kits23/dataset"
SPLITS_JSON = (
    "/scratch2/salonso/medical/data_revision/nnunet_preprocessed/"
    "Dataset001_KiTS23/splits_final.json"
)


def _load_segvol_model(device):
    """Load SegVol by instantiating the core nn.Module and loading
    pretrained weights, bypassing the transformers wrapper."""
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
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 3: {missing[:3]})")
    print(f"  Loaded weights ({len(new_sd)} keys, {len(unexpected)} unexpected)")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model.text_encoder.tokenizer = tokenizer
    model = model.to(device).eval()

    # Processor with MONAI 1.2 compatibility patch
    processor = segvol_module.SegVolProcessor(spatial_size=spatial_size)
    _orig_loader = processor.img_loader
    def _safe_load(path):
        data = _orig_loader(path)
        return data[0] if isinstance(data, tuple) else data
    processor.img_loader = _safe_load

    logits2roi_coor = segvol_module.logits2roi_coor
    sw_inference = segvol_module.sliding_window_inference
    return model, processor, spatial_size, logits2roi_coor, sw_inference


def preprocess_case(processor, ct_path, gt_path, categories, spatial_size):
    """Load and preprocess a CT + GT pair.

    DimTranspose is intentionally skipped: it swaps axes -1 and -3 for
    M3D-Seg .npy files stored in (W,H,D) order. NIfTI volumes loaded via
    MONAI are already in (D,H,W) so the transpose would incorrectly map
    the axial direction into the 32-voxel zoom-out depth.
    """
    import monai.transforms as T

    ct_voxel = np.array(processor.img_loader(ct_path)).squeeze()
    ct_shape = ct_voxel.shape
    ct_voxel = processor.ForegroundNorm(np.expand_dims(ct_voxel, 0))

    gt_voxel = np.array(processor.img_loader(gt_path)).squeeze()
    present = np.unique(gt_voxel)
    gt_masks = []
    for cls_idx in range(len(categories)):
        cls = cls_idx + 1
        gt_masks.append(
            (gt_voxel == cls).astype(np.float32) if cls in present
            else np.zeros(ct_shape))
    gt_stacked = np.stack(gt_masks, axis=0).astype(np.int32)

    item = {"image": ct_voxel, "label": gt_stacked}

    # MinMaxNorm
    item["image"] = item["image"] - item["image"].min()
    item["image"] = item["image"] / np.clip(item["image"].max(), 1e-8, None)

    crop_fg = T.CropForegroundd(keys=["image", "label"], source_key="image")
    to_tensor = T.ToTensord(keys=["image", "label"])
    item = crop_fg(item)
    item = to_tensor(item)

    resize = T.Resized(keys=["image", "label"], spatial_size=spatial_size,
                       mode="nearest-exact")
    item_zoom_out = resize(dict(item))
    item["zoom_out_image"] = item_zoom_out["image"]
    item["zoom_out_label"] = item_zoom_out["label"]
    return item


def dice(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    return 2.0 * inter / union if union > 0 else float("nan")


def case_id_to_dir(case_id):
    num = case_id.replace("KiTS23_", "")
    return os.path.join(KITS23_DIR, f"case_{num}")


def infer_zoom_out_only(model, zoomed_image, volume_shape, categories, device):
    preds = {}
    for text_prompt in categories:
        with torch.no_grad():
            logits = model(zoomed_image, text=[text_prompt])
            logits_up = F.interpolate(logits.cpu(), size=volume_shape,
                                      mode="nearest")
        preds[text_prompt] = (
            (torch.sigmoid(logits_up.squeeze()) > 0.5).numpy().astype(np.uint8))
    return preds


def infer_zoom_in_out(model, image, zoomed_image, volume_shape,
                      spatial_size, categories, device,
                      logits2roi_coor_fn, sw_inference_fn):
    """Full zoom-out / zoom-in pipeline."""
    preds = {}
    for text_prompt in categories:
        with torch.no_grad():
            logits_global = model(zoomed_image, text=[text_prompt])
        logits_global = F.interpolate(logits_global.cpu(),
                                      size=volume_shape, mode="nearest")

        min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor_fn(
            spatial_size, logits_global[0][0])

        if min_d is None:
            preds[text_prompt] = (
                (torch.sigmoid(logits_global.squeeze()) > 0.5)
                .numpy().astype(np.uint8))
            continue

        image_cropped = image[:, :, min_d:max_d+1,
                              min_h:max_h+1, min_w:max_w+1]
        global_preds_crop = (
            torch.sigmoid(logits_global[:, :, min_d:max_d+1,
                                        min_h:max_h+1,
                                        min_w:max_w+1]) > 0.5).long()
        prompt_reflection = (global_preds_crop.float().to(device),
                             global_preds_crop.to(device))

        with torch.no_grad():
            logits_cropped = sw_inference_fn(
                image_cropped.to(device), prompt_reflection,
                spatial_size, 1, model, 0.5,
                text=[text_prompt], use_box=True, use_point=False)
            logits_cropped = logits_cropped.cpu().squeeze()

        logits_global[:, :, min_d:max_d+1,
                      min_h:max_h+1, min_w:max_w+1] = logits_cropped
        preds[text_prompt] = (
            (torch.sigmoid(logits_global.squeeze()) > 0.5)
            .numpy().astype(np.uint8))
    return preds


def run_segvol_fold(model, processor, fold_idx, splits, device,
                    use_zoom, spatial_size, logits2roi_coor_fn,
                    sw_inference_fn, max_cases=None):
    val_cases = splits[fold_idx]["val"]
    if max_cases is not None:
        val_cases = val_cases[:max_cases]
    categories = ["kidney", "kidney tumor", "kidney cyst"]

    km_dices, tc_dices, t_dices = [], [], []
    times, vrams = [], []

    for i, case_id in enumerate(val_cases):
        case_dir = case_id_to_dir(case_id)
        ct_path = os.path.join(case_dir, "imaging.nii.gz")
        gt_path = os.path.join(case_dir, "segmentation.nii.gz")
        if not os.path.exists(ct_path) or not os.path.exists(gt_path):
            print(f"  SKIP {case_id}: files not found", flush=True)
            continue

        data_item = preprocess_case(processor, ct_path, gt_path,
                                    categories, spatial_size)
        image = data_item["image"].unsqueeze(0).to(device)
        zoomed_image = data_item["zoom_out_image"].unsqueeze(0).to(device)
        gt_label_full = data_item["label"]
        volume_shape = image[0][0].shape

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        if use_zoom:
            preds = infer_zoom_in_out(
                model, image, zoomed_image, volume_shape,
                spatial_size, categories, device,
                logits2roi_coor_fn, sw_inference_fn)
        else:
            preds = infer_zoom_out_only(
                model, zoomed_image, volume_shape, categories, device)

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        times.append(elapsed)
        vrams.append(peak_vram)

        pred_seg = np.zeros_like(preds["kidney"], dtype=np.uint8)
        pred_seg[preds["kidney"] > 0] = 1
        pred_seg[preds["kidney tumor"] > 0] = 2
        pred_seg[preds["kidney cyst"] > 0] = 3

        gt_np = (gt_label_full.numpy() if hasattr(gt_label_full, "numpy")
                 else np.array(gt_label_full))
        gt_seg = np.zeros(gt_np.shape[1:], dtype=np.uint8)
        gt_seg[gt_np[0] > 0] = 1
        gt_seg[gt_np[1] > 0] = 2
        gt_seg[gt_np[2] > 0] = 3

        km = dice(pred_seg >= 1, gt_seg >= 1)
        tc = dice(pred_seg >= 2, gt_seg >= 2)
        t = dice(pred_seg == 2, gt_seg == 2)
        km_dices.append(km)
        tc_dices.append(tc)
        t_dices.append(t)

        if (i + 1) % 10 == 0 or (i + 1) == len(val_cases):
            print(f"  Fold {fold_idx}: {i+1}/{len(val_cases)} "
                  f"(last: K+M={km:.4f} T+C={tc:.4f} T={t:.4f} "
                  f"time={elapsed:.1f}s vram={peak_vram:.0f}MB)", flush=True)

        del image, zoomed_image, data_item, preds
        gc.collect()
        torch.cuda.empty_cache()

    km_avg = np.nanmean(km_dices)
    tc_avg = np.nanmean(tc_dices)
    t_avg = np.nanmean(t_dices)
    all_avg = np.nanmean([km_avg, tc_avg, t_avg])
    return {
        "all": float(all_avg),
        "kidneys_masses": float(km_avg),
        "tumour_cyst": float(tc_avg),
        "tumour": float(t_avg),
        "n_cases": len(km_dices),
        "mean_time": float(np.mean(times)) if times else 0,
        "mean_vram": float(np.mean(vrams)) if vrams else 0,
        "per_case_km": [float(x) for x in km_dices],
        "per_case_tc": [float(x) for x in tc_dices],
        "per_case_t": [float(x) for x in t_dices],
    }


def main():
    parser = argparse.ArgumentParser(
        description="SegVol zero-shot evaluation on KiTS23.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Single fold to evaluate (default: all 5)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_zoom", action="store_true",
                        help="Enable zoom-in refinement")
    parser.add_argument("--max_cases", type=int, default=None,
                        help="Max cases per fold (for quick testing)")
    parser.add_argument("--output", type=str,
                        default="baselines/results/segvol_evaluation.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for SegVol evaluation.")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print("=" * 64)
    print("  SegVol Zero-Shot Evaluation on KiTS23")
    print("=" * 64)
    print(f"  GPU:     {torch.cuda.get_device_name(device)}")
    print(f"  VRAM:    "
          f"{torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    print(f"  Zoom-in: {'Yes' if args.use_zoom else 'No (zoom-out only)'}")

    print("\nLoading SegVol from HuggingFace (BAAI/SegVol)...")
    (model, processor, spatial_size,
     logits2roi_coor_fn, sw_inference_fn) = _load_segvol_model(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    folds = [args.fold] if args.fold is not None else list(range(5))
    results = {}
    for fold_idx in folds:
        print(f"\n--- Fold {fold_idx + 1} ---", flush=True)
        r = run_segvol_fold(
            model, processor, fold_idx, splits, device,
            use_zoom=args.use_zoom, spatial_size=spatial_size,
            logits2roi_coor_fn=logits2roi_coor_fn,
            sw_inference_fn=sw_inference_fn, max_cases=args.max_cases)
        results[fold_idx] = r
        print(f"  Fold {fold_idx+1} (n={r['n_cases']}): "
              f"All={r['all']:.4f}  K+M={r['kidneys_masses']:.4f}  "
              f"T+C={r['tumour_cyst']:.4f}  T={r['tumour']:.4f}  "
              f"time={r['mean_time']:.1f}s  vram={r['mean_vram']:.0f}MB",
              flush=True)

    if len(results) > 1:
        print(f"\n{'=' * 64}")
        avg_all = np.mean([v["all"] for v in results.values()])
        avg_km = np.mean([v["kidneys_masses"] for v in results.values()])
        avg_tc = np.mean([v["tumour_cyst"] for v in results.values()])
        avg_t = np.mean([v["tumour"] for v in results.values()])
        print(f"  Average (n_folds={len(results)}): "
              f"All={avg_all:.4f}  K+M={avg_km:.4f}  "
              f"T+C={avg_tc:.4f}  T={avg_t:.4f}")
        print(f"\n  LaTeX table row:")
        print(f"  SegVol & {avg_all:.4f} & {avg_km:.4f} "
              f"& {avg_tc:.4f} & {avg_t:.4f} \\\\")
        print("=" * 64)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out = {
        "num_params": num_params,
        "use_zoom": args.use_zoom,
        "per_fold": {str(k): v for k, v in results.items()},
    }
    if len(results) > 1:
        out["average"] = {
            "all": float(avg_all), "kidneys_masses": float(avg_km),
            "tumour_cyst": float(avg_tc), "tumour": float(avg_t),
        }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
