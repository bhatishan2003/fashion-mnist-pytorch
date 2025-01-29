#!/bin/bash
for seed in 0 1 2; do
    python main.py --mode train \
                   --learning_rate 0.01 \
                   --optimizer sgd \
                   --seed "$seed" \
                   --num_epochs 25 \
                   --use_wandb \
                   --model_type cnn
done
