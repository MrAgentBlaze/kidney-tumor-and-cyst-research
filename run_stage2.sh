#!/bin/bash
# Stage 2: High-resolution segmentation with 5-fold cross-validation.
# See paper Section "Stage 2: Full segmentation" for details.

python -m train.train \
    --train \
    --stage2 \
    --target -1 \
    --dataset_name kits23_processed_highres \
    --dataset_path "data/{}/*" \
    --min_hu -300 \
    --max_hu 500 \
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
    --save_dir logs_stage2 \
    --name v1 \
    --log_every_n_steps 5 \
    --save_top_k 1 \
    --checkpoint_path checkpoints_stage2 \
    --checkpoint_name v1 \
    --gpus 0
