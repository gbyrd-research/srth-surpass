#!/usr/bin/env python3
"""
Example script demonstrating how to load statistics from JSON
and create new task configurations.

Usage:
    python example_load_json_config.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import constants_dvrk
sys.path.insert(0, str(Path(__file__).parent.parent))

from dvrk_scripts.constants_dvrk import (
    load_statistics_from_json,
    create_task_config_from_json,
    add_task_config_from_json,
    TASK_CONFIGS
)


def example_load_statistics():
    """Example: Load statistics from JSON file"""
    print("=" * 80)
    print("Example 1: Loading Statistics from JSON")
    print("=" * 80)
    
    json_path = "../std_mean_invivo.json"
    
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        print("Please run calculate_std_mean.py first to generate the JSON file.")
        return None
    
    stats = load_statistics_from_json(json_path)
    
    print(f"Loaded statistics from: {json_path}")
    print(f"\nMean shape: {stats['mean'].shape}")
    print(f"Mean values: {stats['mean']}")
    print(f"\nStd shape: {stats['std'].shape}")
    print(f"Std values: {stats['std']}")
    print(f"\nMin shape: {stats['min'].shape}")
    print(f"Min values: {stats['min']}")
    print(f"\nMax shape: {stats['max'].shape}")
    print(f"Max values: {stats['max']}")
    
    return stats


def example_create_task_config():
    """Example: Create a task config from JSON"""
    print("\n" + "=" * 80)
    print("Example 2: Creating Task Config from JSON")
    print("=" * 80)
    
    json_path = "../std_mean_invivo.json"
    
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return None
    
    config = create_task_config_from_json(
        task_name='invivo_from_json',
        json_path=json_path,
        dataset_dir="/home/iulian/chole_ws/data/cnh_exvivo_chole",
        tissue_ids=[1, 2, 3],
        tissue_ids_val=[4],
        num_episodes=853,
        num_episodes_val=50,
        phantom=False,
        use_auto_label=False,
        no_qpos=False,
        camera_file_suffixes=["_left.jpg", "_psm2.jpg", "_psm1.jpg"],
        episode_len=500,
        camera_names=['left', 'left_wrist', 'right_wrist']
    )
    
    print(f"Created task config: 'invivo_from_json'")
    print(f"\nConfig keys: {list(config.keys())}")
    print(f"Dataset dir: {config['dataset_dir']}")
    print(f"Tissue IDs: {config['tissue_samples_ids']}")
    print(f"Action mode: {config['action_mode'][0]}")
    print(f"Statistics loaded: mean, std, min, max")
    
    return config


def example_add_task_config():
    """Example: Add a task config to TASK_CONFIGS"""
    print("\n" + "=" * 80)
    print("Example 3: Adding Task Config to TASK_CONFIGS")
    print("=" * 80)
    
    json_path = "../std_mean_invivo.json"
    
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return
    
    print(f"Number of task configs before: {len(TASK_CONFIGS)}")
    
    add_task_config_from_json(
        task_name='my_new_invivo_task',
        json_path=json_path,
        dataset_dir="/home/iulian/chole_ws/data/cnh_exvivo_chole",
        tissue_ids=[1, 2, 3],
        num_episodes=1000,
        camera_names=['left', 'left_wrist', 'right_wrist']
    )
    
    print(f"Number of task configs after: {len(TASK_CONFIGS)}")
    print(f"New task config available as: TASK_CONFIGS['my_new_invivo_task']")
    
    # Verify it was added
    if 'my_new_invivo_task' in TASK_CONFIGS:
        config = TASK_CONFIGS['my_new_invivo_task']
        print(f"\nVerification - Dataset: {config['dataset_dir']}")
        print(f"Verification - Tissue IDs: {config['tissue_samples_ids']}")


def example_compare_configs():
    """Example: Compare manually created vs JSON-loaded configs"""
    print("\n" + "=" * 80)
    print("Example 4: Comparing Configs")
    print("=" * 80)
    
    json_path = "../std_mean_invivo.json"
    
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return
    
    # Check if there's an existing invivo config
    if 'invivo_test' in TASK_CONFIGS:
        manual_config = TASK_CONFIGS['invivo_test']
        print("Found existing 'invivo_test' config (manually created)")
        
        # Load from JSON
        stats = load_statistics_from_json(json_path)
        
        print("\nComparing statistics:")
        print(f"Manual mean (first 5): {manual_config['action_mode'][1]['mean'][:5]}")
        print(f"JSON mean (first 5): {stats['mean'][:5]}")
        print(f"\nManual std (first 5): {manual_config['action_mode'][1]['std'][:5]}")
        print(f"JSON std (first 5): {stats['std'][:5]}")
    else:
        print("No 'invivo_test' config found for comparison")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("TASK CONFIG JSON LOADER - EXAMPLES")
    print("=" * 80 + "\n")
    
    try:
        example_load_statistics()
        example_create_task_config()
        example_add_task_config()
        example_compare_configs()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
