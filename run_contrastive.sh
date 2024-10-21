#!/bin/bash

# Default arguments
dataset_name="large_iso"
dataset_path="/scratch/salonso/sparse-nns/medical_data/data_kits23_{}_{}_{}_good/*"
chunk_size=512
min_hu=-30
max_hu=350
ds_steps=1
eps=1e-12
batch_size=1
folds=5
epochs=50
num_workers=32
lr=1e-3
accum_grad_batches=1
warmup_steps=0
weight_decay=1e-5
beta1=0.9
beta2=0.999
losses=("focal" "dice")
save_dir="/scratch/salonso/sparse-nns/medical_ai/ai_cancer_research/logs_contrastive"
name="v1"
log_every_n_steps=25
save_top_k=1
checkpoint_path="/scratch/salonso/sparse-nns/medical_ai/ai_cancer_research/checkpoints_contrastive"
checkpoint_name="v1"
load_checkpoint=None
gpus=(0)

python -m train.train \
    --train \
    --contrastive \
    --chunk_size $chunk_size \
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

