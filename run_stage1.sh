#!/bin/bash
# Stage 1: Low-resolution ROI detection with 5-fold cross-validation.
# See paper Section "Stage 1: ROI finder (detection)" for details.

python -m train.train \
    --train \
    --target -1 \
    --dataset_name kits23_large_processed \
    --dataset_path "data/{}/*" \
    --min_hu -53.4 \
    --max_hu 283.2 \
    --ds_steps 3 \
    --batch_size 1 \
    --folds 5 \
    --epochs 500 \
    --num_workers 8 \
    --lr 5e-4 \
    --accum_grad_batches 8 \
    --warmup_steps 1 \
    --cosine_annealing_steps 400 \
    --weight_decay 1e-5 \
    --beta1 0.9 \
    --beta2 0.95 \
    --losses dice \
    --save_dir logs_stage1 \
    --name v1 \
    --log_every_n_steps 5 \
    --save_top_k 1 \
    --checkpoint_path checkpoints_stage1 \
    --checkpoint_name v1 \
    --gpus 0
