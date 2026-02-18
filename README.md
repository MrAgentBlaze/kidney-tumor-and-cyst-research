# Sparse 3D U-Net for Kidney and Tumour Segmentation in CT

Implementation of the paper: **"Submanifold Sparse Convolutional Networks for Automated 3D Segmentation of Kidneys and Kidney Tumours in Computed Tomography"** ([arXiv:2511.04334](https://arxiv.org/abs/2511.04334)).

This repository provides a sparse voxel-based 3D U-Net architecture with ConvNeXtV2 blocks for segmenting kidneys, kidney tumours, and cysts in CT images, using [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) for submanifold sparse convolutions.

## Method Overview

The method follows a two-stage approach:

1. **Stage 1 (ROI Detection):** A low-resolution sparsified CT scan is processed by the sparse 3D U-Net to detect regions of interest (ROIs) containing kidneys and masses.
2. **Stage 2 (Full Segmentation):** The identified ROIs are cropped from the original high-resolution scan and fed into a second sparse 3D U-Net for full-resolution segmentation.

Both stages use the same network architecture with deep supervision. Voxel sparsification reduces computational cost by discarding voxels outside a relevant Hounsfield Unit (HU) range.

## Architecture

The encoder consists of six hierarchical stages with feature dimensions (16, 32, 64, 128, 256, 512) and block depths (2, 4, 4, 8, 8, 8). All convolutions use 3×3×3 kernels. The decoder mirrors the encoder with five stages of two ConvNeXtV2 blocks each. Skip connections are implemented via element-wise summation. Classification heads at multiple decoder stages enable deep supervision.

The network has 27,473,696 trainable parameters.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- MinkowskiEngine >= 0.5.4 with depthwise convolution support
  - Install from: https://github.com/shwoo93/MinkowskiEngine

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

This implementation uses the [KiTS23 challenge](https://kits-challenge.org/kits23/) dataset. Download it via the [official repository](https://github.com/neheller/kits23).

### Data Preparation

1. **Low-resolution resampling:** Resample scans to isotropic 1.99 mm voxel spacing for Stage 1.
2. **High-resolution resampling:** Resample scans to isotropic 0.78 mm voxel spacing for Stage 2.
3. **Sparsification:** Voxels outside the HU range (−53.4, 283.2) are discarded in Stage 1. In Stage 2, only the ROI from Stage 1 is retained.

The dataset should be stored in pickle format (`.pkl`) for Stage 1, and PyTorch format (`.pt`) for Stage 2. Each file should contain a dictionary with keys `image` and `label` (and optionally `roi` for Stage 2).

Update the `--dataset_path` argument to point to your processed data directory.

## Training

### Stage 1: ROI Detection

Train the low-resolution ROI detection network with 5-fold cross-validation:

```bash
bash run_stage1.sh
```

### ROI Processing

After Stage 1 training, run the ROI processing script to crop high-resolution scans:

```bash
python -m process_roi.process_roi
```

Update the paths in `process_roi/process_roi.py` to match your data directories.

### Stage 2: Full Segmentation

Train the high-resolution segmentation network:

```bash
bash run_stage2.sh
```

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--stage2` | `False` | Enable Stage 2 (high-resolution) mode |
| `--target` | `-1` | Target label: 0 (kidneys+masses), 1 (tumour+cyst), 2 (tumour), -1 (all) |
| `--dataset_path` | `data/{}/*` | Path template for the dataset directory |
| `--dataset_name` | `kits23_large_processed` | Dataset folder name |
| `--min_hu` | `-53.4` | Minimum HU threshold for sparsification |
| `--max_hu` | `283.2` | Maximum HU threshold for sparsification |
| `--ds_steps` | `3` | Number of deep supervision steps |
| `--batch_size` | `1` | Batch size |
| `--accum_grad_batches` | `8` | Gradient accumulation steps |
| `--epochs` | `500` | Number of training epochs |
| `--lr` | `5e-4` | Learning rate |
| `--weight_decay` | `1e-5` | Weight decay |
| `--beta1` | `0.9` | AdamW β₁ |
| `--beta2` | `0.95` | AdamW β₂ |
| `--warmup_steps` | `1` | Number of warm-up epochs |
| `--cosine_annealing_steps` | `400` | Number of cosine annealing epochs |
| `--losses` | `dice` | Loss function(s): `dice`, `focal`, `ce` |
| `--folds` | `5` | Number of cross-validation folds |
| `--gpus` | `0` | GPU device ID(s) |
| `--sparse` / `--dense` | `sparse` | Use sparse or dense representation |

See `utils/args.py` for the full list of arguments.

## Performance Testing

To benchmark inference time and VRAM usage:

```bash
# Sparse model
python -m performance.test --sparse --gpus 0

# Dense model (for comparison)
python -m performance.test --dense --gpus 0
```

## Project Structure

```
├── model/
│   ├── minkunet_convnextv2.py   # Sparse 3D U-Net architecture
│   ├── denseunet_convnextv2.py  # Dense equivalent (for benchmarking)
│   ├── lightning_model.py       # PyTorch Lightning training module
│   └── utils.py                 # Sparse layer utilities (GRN, DropPath, LayerNorm)
├── dataset/
│   └── dataset.py               # Dataset loading and sparsification
├── utils/
│   ├── args.py                  # Command-line arguments
│   ├── augmentations.py         # MONAI-based data augmentations
│   ├── callbacks.py             # Fine-tuning callbacks
│   ├── funcs.py                 # Utility functions and schedulers
│   └── losses.py                # Loss functions (Dice, focal, cross-entropy)
├── train/
│   ├── train.py                 # K-fold cross-validation training
│   └── train_all.py             # Training on the full dataset
├── process_roi/
│   └── process_roi.py           # ROI processing pipeline (Stage 1 → Stage 2)
├── performance/
│   └── test.py                  # Inference benchmarking
├── run_stage1.sh                # Stage 1 training script
├── run_stage2.sh                # Stage 2 training script
├── requirements.txt
└── README.md
```

## Citation

If you use this code, please cite:

```bibtex
@article{alonsomonsalve2025sparse,
  title={Submanifold Sparse Convolutional Networks for Automated 3D Segmentation of Kidneys and Kidney Tumours in Computed Tomography},
  author={Alonso-Monsalve, Sa{\'u}l and Whitehead, Leigh H. and Aurisano, Adam and Escudero Sanchez, Lorena},
  journal={arXiv preprint arXiv:2511.04334},
  year={2025}
}
```

## Licence

This project is licensed under the [MIT License](LICENSE).
