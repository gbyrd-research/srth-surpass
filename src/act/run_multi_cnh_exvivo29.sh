#!/bin/bash
# Training script for multi-dataset training: cnh_exvivo + exvivo_29_auto_label_srt
#
# This trains a single policy on both datasets simultaneously with automatic balancing
#
# Usage:
#   ./run_multi_cnh_exvivo29.sh           # Use defaults
#   ./run_multi_cnh_exvivo29.sh 1         # Use GPU 1
#   ./run_multi_cnh_exvivo29.sh 0 2.0 1.0 # GPU 0, custom weights (cnh_exvivo 2x)

GPU=${1:-0}           # Default to GPU 0
WEIGHT_1=${2:-}       # Optional: weight for cnh_exvivo
WEIGHT_2=${3:-}       # Optional: weight for exvivo_29_auto_label_srt

echo "=================================="
echo "Multi-Dataset Training"
echo "=================================="
echo "Datasets: cnh_exvivo + exvivo_29_auto_label_srt"
echo "GPU: $GPU"

if [ -n "$WEIGHT_1" ] && [ -n "$WEIGHT_2" ]; then
    echo "Custom weights: [$WEIGHT_1, $WEIGHT_2]"
    WEIGHT_ARGS="--dataset_weights $WEIGHT_1 $WEIGHT_2"
else
    echo "Weighting: Auto-balanced"
    WEIGHT_ARGS=""
fi

echo "=================================="
echo ""

# Run training
python train_multi_dataset_cnh_exvivo29.py \
    --gpu $GPU \
    --ckpt_dir ckpt_dir_multi_cnh_exvivo29 \
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
    --policy_level low \
    --eval_every 100 \
    --save_every 500 \
    $WEIGHT_ARGS \
    2>&1 | tee training_multi_cnh_exvivo29.log

echo ""
echo "=================================="
echo "Training finished!"
echo "Log saved to: training_multi_cnh_exvivo29.log"
echo "Checkpoints in: ckpt_dir_multi_cnh_exvivo29/"
echo "=================================="
