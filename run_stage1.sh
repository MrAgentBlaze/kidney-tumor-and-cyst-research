#!/bin/bash

# Default arguments
dataset_name="kits23_large_processed"
dataset_path="/scratch/salonso/sparse-nns/medical_data/{}/*"
min_hu=-53.4
max_hu=283.2
ds_steps=4
eps=1e-12
batch_size=1
folds=5
epochs=100
num_workers=8
lr=5e-4
accum_grad_batches=8
warmup_steps=0
weight_decay=1e-4
beta1=0.9
beta2=0.999
losses=("dice" "focal")
save_dir="/scratch/salonso/sparse-nns/medical_ai/ai_cancer_research/logs_stage1"
name="dicefocal_aug_newarch_monai_ds4_b8_pos_sym"
log_every_n_steps=5
save_top_k=1
checkpoint_path="/scratch/salonso/sparse-nns/medical_ai/ai_cancer_research/checkpoints_stage1"
checkpoint_name="dicefocal_aug_newarch_monai_ds4_b8_pos_sym"
load_checkpoint=None
gpus=(0)

python -m train.train \
    --train \
    --dataset_name $dataset_name \
    --dataset_path $dataset_path \
    --min_hu $min_hu \
    --max_hu $max_hu \
    --ds_steps $ds_steps \
    --eps $eps \
    --batch_size $batch_size \
    --folds $folds \
    --epochs $epochs \
    --num_workers $num_workers \
    --lr $lr \
    --accum_grad_batches $accum_grad_batches \
    --warmup_steps $warmup_steps \
    --weight_decay $weight_decay \
    --beta1 $beta1 \
    --beta2 $beta2 \
    --losses "${losses[@]}" \
    --save_dir $save_dir \
    --name $name \
    --log_every_n_steps $log_every_n_steps \
    --save_top_k $save_top_k \
    --checkpoint_path $checkpoint_path \
    --checkpoint_name $checkpoint_name \
    --load_checkpoint $load_checkpoint \
    --gpus "${gpus[@]}"

