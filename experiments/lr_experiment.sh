#!/bin/bash

learning_rates=(0.1 0.01 0.001)
seeds=(0 1 2)

for lr in "${learning_rates[@]}"; do
    for seed in "${seeds[@]}"; do
        python main.py --mode train \
                       --learning_rate "$lr" \
                       --optimizer sgd \
                       --seed "$seed" \
                       --num_epochs 25 \
                       --use_wandb
    done
done
