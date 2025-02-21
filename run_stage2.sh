#!/bin/bash

# Default arguments
dataset_name="kits23_processed_highres11_all"
dataset_path="/scratch/salonso/sparse-nns/medical_data/{}/*"
#min_hu=-78.3
#max_hu=531.4
min_hu=-300
max_hu=500
ds_steps=3
eps=1e-12
batch_size=1
folds=5
epochs=500
num_workers=8
lr=5e-4
accum_grad_batches=8
warmup_steps=1
cosine_annealing_steps=400
weight_decay=1e-5
beta1=0.9
beta2=0.95
losses=("dice")
save_dir="/scratch/salonso/sparse-nns/medical_ai/ai_cancer_research/logs_stage2"
name="dice_v5"
log_every_n_steps=5
save_top_k=1
checkpoint_path="/scratch/salonso/sparse-nns/medical_ai/ai_cancer_research/checkpoints_stage2"
checkpoint_name="dice_v5"
load_checkpoint=None
gpus=(1)

python -m train.train \
    --train \
    --stage2 \
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
    --cosine_annealing_steps $cosine_annealing_steps \
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
    --gpus "${gpus[@]}"

