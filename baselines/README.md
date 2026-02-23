# Baseline Models — KiTS23 Evaluation

Scripts used to reproduce the baseline comparisons reported in the paper.
Two baselines are evaluated on KiTS23 using the same 5-fold cross-validation
splits: **nnU-Net** (supervised) and **SegVol** (zero-shot).

## nnU-Net

Fully supervised 3D segmentation using
[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) with the `3d_fullres`
configuration (auto-configured by the nnU-Net planner).

| Script | Description |
|---|---|
| `convert_to_nnunet.py` | Convert KiTS23 to nnU-Net raw format and generate fold splits |
| `run_nnunet.sh` | End-to-end pipeline: conversion → planning → training → evaluation |
| `evaluate_nnunet.py` | Compute grouped Dice from nnU-Net cross-validation predictions |
| `benchmark_nnunet_inference.py` | Measure inference time and peak VRAM |

**Environment:** requires `nnunetv2`, `monai`,
`dynamic-network-architectures`.

```bash
bash baselines/run_nnunet.sh
```

## SegVol

Zero-shot text-prompted 3D segmentation using
[SegVol](https://huggingface.co/BAAI/SegVol) (Du et al., 2024).
No training is required; the pretrained model is downloaded from HuggingFace.

| Script | Description |
|---|---|
| `run_segvol.py` | Per-fold Dice evaluation with zoom-out or zoom-in/out |
| `run_segvol.sh` | Full evaluation pipeline (zoom-out, zoom-in, benchmark) |
| `benchmark_segvol_inference.py` | Measure inference time and peak VRAM |

**Environment:** requires `monai==1.2`, `transformers==4.35.2`.

```bash
bash baselines/run_segvol.sh
```

## Metrics

Both baselines are evaluated with grouped binary Dice matching the paper:

| Metric | Definition |
|---|---|
| Kidneys + masses | `pred ≥ 1` vs `gt ≥ 1` |
| Tumour + cyst | `pred ≥ 2` vs `gt ≥ 2` |
| Tumour | `pred == 2` vs `gt == 2` |
| All | Mean of the above |
