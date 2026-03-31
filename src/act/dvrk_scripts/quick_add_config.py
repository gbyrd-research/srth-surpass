#!/usr/bin/env python3
"""
Quick command-line tool to add a task configuration from a JSON statistics file.

Usage:
    python quick_add_config.py <task_name> <json_path> [options]

Example:
    python quick_add_config.py my_task ./std_mean_invivo.json \\
        --dataset-dir /path/to/data \\
        --tissue-ids 1 2 3 \\
        --num-episodes 1000
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dvrk_scripts.constants_dvrk import (
    add_task_config_from_json,
    load_statistics_from_json,
    TASK_CONFIGS
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Add a task configuration from JSON statistics file',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('task_name', type=str,
                        help='Name for the new task configuration')
    parser.add_argument('json_path', type=str,
                        help='Path to JSON file containing statistics')
    
    # Dataset configuration
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--tissue-ids', type=int, nargs='+', required=True,
                        help='Tissue sample IDs for training (e.g., 1 2 3)')
    parser.add_argument('--tissue-ids-val', type=int, nargs='+', default=[],
                        help='Tissue sample IDs for validation')
    
    # Episode configuration
    parser.add_argument('--num-episodes', type=int, default=None,
                        help='Number of training episodes')
    parser.add_argument('--num-episodes-val', type=int, default=0,
                        help='Number of validation episodes')
    parser.add_argument('--episode-len', type=int, default=500,
                        help='Length of episodes')
    
    # Task configuration
    parser.add_argument('--phantom', action='store_true',
                        help='Use phantom data')
    parser.add_argument('--use-auto-label', action='store_true',
                        help='Use auto labeling')
    parser.add_argument('--no-qpos', action='store_true',
                        help='No qpos (True for SRT, False for ACT)')
    parser.add_argument('--goal-condition-style', type=str, choices=['map', 'mask', 'dot'],
                        help='Goal conditioning style')
    
    # Camera configuration
    parser.add_argument('--camera-names', type=str, nargs='+',
                        default=['left', 'left_wrist', 'right_wrist'],
                        help='Camera names')
    parser.add_argument('--camera-file-suffixes', type=str, nargs='+',
                        default=['_left.jpg', '_psm1.jpg', '_psm2.jpg'],
                        help='Camera file suffixes')
    
    # Other parameters
    parser.add_argument('--cutting-action-pad-size', type=int, default=0,
                        help='Padding size for cutting actions')
    parser.add_argument('--recovery-ratio', type=float, default=1.0,
                        help='Recovery ratio')
    parser.add_argument('--norm-scheme', type=str, default='std',
                        help='Normalization scheme')
    parser.add_argument('--save-frequency', type=int, default=50,
                        help='Checkpoint save frequency')
    parser.add_argument('--merging-subtasks', action='store_true',
                        help='Merge subtasks')
    
    # Actions
    parser.add_argument('--list-configs', action='store_true',
                        help='List all existing task configurations')
    parser.add_argument('--show-stats', action='store_true',
                        help='Show statistics from JSON file')
    
    return parser.parse_args()


def list_existing_configs():
    """List all existing task configurations"""
    print("\n" + "=" * 80)
    print("EXISTING TASK CONFIGURATIONS")
    print("=" * 80)
    
    for i, (name, config) in enumerate(TASK_CONFIGS.items(), 1):
        print(f"\n{i}. {name}")
        print(f"   Dataset: {config.get('dataset_dir', 'N/A')}")
        print(f"   Tissue IDs: {config.get('tissue_samples_ids', 'N/A')}")
        print(f"   Episodes: {config.get('num_episodes', 'N/A')}")
    
    print("\n" + "=" * 80)


def show_statistics(json_path):
    """Show statistics from JSON file"""
    print("\n" + "=" * 80)
    print(f"STATISTICS FROM: {json_path}")
    print("=" * 80)
    
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        return
    
    stats = load_statistics_from_json(json_path)
    
    print(f"\nMean (shape {stats['mean'].shape}):")
    print(stats['mean'])
    print(f"\nStd (shape {stats['std'].shape}):")
    print(stats['std'])
    print(f"\nMin (shape {stats['min'].shape}):")
    print(stats['min'])
    print(f"\nMax (shape {stats['max'].shape}):")
    print(stats['max'])
    
    print("\n" + "=" * 80)


def main():
    args = parse_args()
    
    # Handle list configs action
    if args.list_configs:
        list_existing_configs()
        return
    
    # Handle show stats action
    if args.show_stats:
        show_statistics(args.json_path)
        return
    
    # Validate JSON file exists
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found: {args.json_path}")
        sys.exit(1)
    
    # Check if task name already exists
    if args.task_name in TASK_CONFIGS:
        print(f"Warning: Task config '{args.task_name}' already exists!")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    print(f"\nCreating task configuration '{args.task_name}' from {args.json_path}...")
    
    # Add the task config
    try:
        config = add_task_config_from_json(
            task_name=args.task_name,
            json_path=args.json_path,
            dataset_dir=args.dataset_dir,
            tissue_ids=args.tissue_ids,
            tissue_ids_val=args.tissue_ids_val,
            num_episodes=args.num_episodes,
            num_episodes_val=args.num_episodes_val,
            phantom=args.phantom,
            use_auto_label=args.use_auto_label,
            goal_condition_style=args.goal_condition_style,
            no_qpos=args.no_qpos,
            camera_file_suffixes=args.camera_file_suffixes,
            episode_len=args.episode_len,
            cutting_action_pad_size=args.cutting_action_pad_size,
            recovery_ratio=args.recovery_ratio,
            norm_scheme=args.norm_scheme,
            save_frequency=args.save_frequency,
            camera_names=args.camera_names,
            merging_subtasks=args.merging_subtasks
        )
        
        print("\n" + "=" * 80)
        print("SUCCESS! Task configuration created:")
        print("=" * 80)
        print(f"Task name: {args.task_name}")
        print(f"Dataset: {config['dataset_dir']}")
        print(f"Tissue IDs (train): {config['tissue_samples_ids']}")
        print(f"Tissue IDs (val): {config['tissue_samples_ids_val']}")
        print(f"Episodes (train): {config.get('num_episodes', 'Not set')}")
        print(f"Episodes (val): {config.get('num_episodes_val', 0)}")
        print(f"Camera names: {config['camera_names']}")
        print("=" * 80)
        
        print("\nTo use this configuration in your code:")
        print(f"    from constants_dvrk import TASK_CONFIGS")
        print(f"    config = TASK_CONFIGS['{args.task_name}']")
        
    except Exception as e:
        print(f"\nError creating task configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
