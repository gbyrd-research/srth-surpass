#!/bin/bash

# Example script for multi-dataset co-training
# This trains a model on multiple datasets simultaneously

# Example 1: Train on two datasets with equal weighting (automatic)
# The model will automatically balance sampling between datasets based on their sizes
python imitate_episodes.py \
    --task_name cnh_exvivo suturing_final_map \
    --ckpt_dir ckpt_dir_multi_cnh_suturing \
    --policy_class SRT \
    --kl_weight 10 \
    --chunk_size 60 \
    --hidden_dim 512 \
    --batch_size 12 \
    --dim_feedforward 3200 \
    --num_epochs 20000 \
    --lr 1e-5 \
    --seed 0 \
    --use_language \
    --language_encoder distilbert \
    --image_encoder efficientnet_b3film \
    --gpu 0 \
    --policy_level low

# Example 2: Train on two datasets with custom weights
# Weight the first dataset 2x more than the second
# python imitate_episodes.py \
#     --task_name cnh_exvivo suturing_final_map \
#     --dataset_weights 2.0 1.0 \
#     --ckpt_dir ckpt_dir_multi_weighted \
#     --policy_class SRT \
#     --kl_weight 10 \
#     --chunk_size 60 \
#     --hidden_dim 512 \
#     --batch_size 12 \
#     --dim_feedforward 3200 \
#     --num_epochs 20000 \
#     --lr 1e-5 \
#     --seed 0 \
#     --use_language \
#     --language_encoder distilbert \
#     --image_encoder efficientnet_b3film \
#     --gpu 0 \
#     --policy_level low

# Example 3: Train on three datasets
# python imitate_episodes.py \
#     --task_name cnh_exvivo suturing_final_map invivo_test \
#     --ckpt_dir ckpt_dir_multi_three \
#     --policy_class SRT \
#     --kl_weight 10 \
#     --chunk_size 60 \
#     --hidden_dim 512 \
#     --batch_size 12 \
#     --dim_feedforward 3200 \
#     --num_epochs 20000 \
#     --lr 1e-5 \
#     --seed 0 \
#     --use_language \
#     --language_encoder distilbert \
#     --image_encoder efficientnet_b3film \
#     --gpu 0 \
#     --policy_level low
