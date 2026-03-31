#!/bin/bash
# Activate the correct conda environment
source /home/iulian/anaconda3/bin/activate aloha

# Run training
python imitate_episodes.py \
    --task_name cnh_exvivo \
    --ckpt_dir ckpt_dir_srth_cnh \
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
    --gpu 1 \
    --policy_level low
