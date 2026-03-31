#!/usr/bin/env python3
"""
Multi-dataset training script for 'cnh_exvivo' + 'exvivo_29_auto_label_srt'

This script trains a single policy on multiple datasets simultaneously using
weighted sampling to balance contributions from each dataset.

Usage:
    python train_multi_dataset_cnh_exvivo29.py --gpu 0 --batch_size 12
    
    # With custom dataset weights (e.g., weight cnh_exvivo 2x more):
    python train_multi_dataset_cnh_exvivo29.py --gpu 0 --dataset_weights 2.0 1.0
"""

import os
import sys
import argparse
import torch
import numpy as np
from copy import deepcopy

sys.path.append(os.path.dirname(__file__))

from dvrk_scripts.constants_dvrk import TASK_CONFIGS
from utils import load_data_dvrk_multi_dataset
from policy import SRTPolicy
from imitate_episodes import make_policy, make_optimizer, forward_pass, train_bc

def main():
    parser = argparse.ArgumentParser()
    
    # Multi-dataset configuration
    parser.add_argument('--dataset_weights', nargs='+', type=float, default=None,
                        help='Custom weights for each dataset (default: auto-balance). '
                        'Example: --dataset_weights 2.0 1.0 to weight first dataset 2x more')
    
    # Model configuration (matching your existing training)
    parser.add_argument('--policy_class', type=str, default='SRT', 
                        choices=['SRT', 'ACT', 'Diffusion'],
                        help='Policy class to use')
    parser.add_argument('--kl_weight', type=float, default=10.0,
                        help='KL divergence weight')
    parser.add_argument('--chunk_size', type=int, default=60,
                        help='Action chunk size')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='Batch size for training')
    parser.add_argument('--dim_feedforward', type=int, default=3200,
                        help='Feedforward dimension')
    parser.add_argument('--num_epochs', type=int, default=20000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    
    # Language and vision
    parser.add_argument('--use_language', action='store_true', default=True,
                        help='Use language conditioning')
    parser.add_argument('--language_encoder', type=str, default='distilbert',
                        choices=['distilbert', 'clip'],
                        help='Language encoder to use')
    parser.add_argument('--image_encoder', type=str, default='efficientnet_b3film',
                        help='Image encoder to use')
    
    # Training configuration
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device to use')
    parser.add_argument('--policy_level', type=str, default='low',
                        choices=['low', 'high'],
                        help='Policy level')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt_dir_multi_cnh_exvivo29',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval_every', type=int, default=100,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=500,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("MULTI-DATASET TRAINING: cnh_exvivo + exvivo_29_auto_label_srt")
    print("="*80)
    
    # Define the two datasets to train on
    task_names = ['cnh_exvivo', 'exvivo_29_auto_label_srt']
    
    # Verify tasks exist in TASK_CONFIGS
    for task_name in task_names:
        if task_name not in TASK_CONFIGS:
            print(f"ERROR: Task '{task_name}' not found in TASK_CONFIGS")
            print(f"Available tasks: {list(TASK_CONFIGS.keys())}")
            return 1
    
    # Prepare dataset configurations
    dataset_dirs = []
    num_episodes_list = []
    task_configs_list = []
    
    print("\nDataset Configuration:")
    for i, task_name in enumerate(task_names, 1):
        task_config = TASK_CONFIGS[task_name]
        dataset_dirs.append(task_config['dataset_dir'])
        num_episodes_list.append(task_config['num_episodes'])
        task_configs_list.append(task_config)
        
        weight_str = f" (weight: {args.dataset_weights[i-1]})" if args.dataset_weights else " (auto-balanced)"
        print(f"  {i}. {task_name}")
        print(f"     Path: {task_config['dataset_dir']}")
        print(f"     Episodes: {task_config['num_episodes']}")
        print(f"     Weighting: {weight_str}")
    
    # Validate dataset weights if provided
    if args.dataset_weights is not None:
        if len(args.dataset_weights) != len(task_names):
            print(f"\nERROR: Number of weights ({len(args.dataset_weights)}) "
                  f"must match number of datasets ({len(task_names)})")
            return 1
        print(f"\nUsing custom dataset weights: {args.dataset_weights}")
    else:
        print(f"\nUsing automatic dataset balancing")
    
    # Load data
    print("\n" + "-"*80)
    print("Loading datasets...")
    print("-"*80)
    
    # Use camera names from first task config (should be same for all)
    camera_names = task_configs_list[0]['camera_names']
    
    train_dataloader, val_dataloader, stats, is_sim = load_data_dvrk_multi_dataset(
        dataset_dirs=dataset_dirs,
        num_episodes_list=num_episodes_list,
        camera_names=camera_names,
        batch_size_train=args.batch_size,
        batch_size_val=args.batch_size,
        task_configs=task_configs_list,
        chunk_size=args.chunk_size,
        use_language=args.use_language,
        dataset_weights=args.dataset_weights
    )
    
    print(f"\n✓ Datasets loaded successfully!")
    print(f"  Train batches: {len(train_dataloader)}")
    print(f"  Val batches: {len(val_dataloader) if val_dataloader else 0}")
    print(f"  Cameras: {camera_names}")
    
    # Create policy
    print("\n" + "-"*80)
    print("Creating policy...")
    print("-"*80)
    
    # Use the task config structure from first dataset (they should be compatible)
    base_task_config = task_configs_list[0]
    
    policy_config = {
        'lr': args.lr,
        'camera_names': camera_names,
        'action_dim': 16,  # Standard DVRK action dimension
        'kl_weight': args.kl_weight,
        'hidden_dim': args.hidden_dim,
        'dim_feedforward': args.dim_feedforward,
        'chunk_size': args.chunk_size,
        'use_language': args.use_language,
        'language_encoder': args.language_encoder,
        'image_encoder': args.image_encoder,
        'policy_level': args.policy_level,
    }
    
    print(f"Policy: {args.policy_class}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  KL weight: {args.kl_weight}")
    print(f"  Image encoder: {args.image_encoder}")
    print(f"  Language: {args.language_encoder if args.use_language else 'disabled'}")
    
    policy = make_policy(args.policy_class, policy_config)
    policy.to(device)
    
    # Create optimizer
    optimizer = make_optimizer(args.policy_class, policy)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_ckpt:
        print(f"\nLoading checkpoint from {args.resume_ckpt}")
        checkpoint = torch.load(args.resume_ckpt)
        policy.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Create checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.ckpt_dir, 'training_config.txt')
    with open(config_path, 'w') as f:
        f.write("Multi-Dataset Training Configuration\n")
        f.write("="*50 + "\n\n")
        f.write(f"Datasets: {task_names}\n")
        f.write(f"Dataset Weights: {args.dataset_weights if args.dataset_weights else 'Auto-balanced'}\n")
        f.write(f"\nDataset Details:\n")
        for i, (task_name, num_eps) in enumerate(zip(task_names, num_episodes_list), 1):
            f.write(f"  {i}. {task_name}: {num_eps} episodes\n")
        f.write(f"\nModel Configuration:\n")
        for key, value in policy_config.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nTraining Parameters:\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Num epochs: {args.num_epochs}\n")
        f.write(f"  Seed: {args.seed}\n")
    
    print(f"\nConfiguration saved to {config_path}")
    
    # Training
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    train_bc(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        policy=policy,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        ckpt_dir=args.ckpt_dir,
        seed=args.seed,
        policy_class=args.policy_class,
        eval_every=args.eval_every,
        save_every=args.save_every,
        start_epoch=start_epoch
    )
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Checkpoints saved to: {args.ckpt_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
