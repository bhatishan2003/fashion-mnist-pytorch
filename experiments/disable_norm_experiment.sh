#!/bin/bash

seeds=(0 1 2)

for seed in "${seeds[@]}"; do
    python main.py --mode train \
                   --learning_rate 0.01 \
                   --optimizer adam \
                   --seed "$seed" \
                   --num_epochs 25 \
                   --use_wandb \
                   --disable_normalization
done
