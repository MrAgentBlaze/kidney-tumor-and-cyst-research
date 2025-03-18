#!/bin/bash

# Default arguments
dataset_path="/raid/monsals/ai_cancer_research/{}/*"
gpus=(0)

python -m performance.test \
    --train \
    --sparse \
    --dataset_path $dataset_path \
    --gpus "${gpus[@]}"

