#!/usr/bin/env python3
"""
Quick test script for multi-dataset training functionality.
This doesn't actually train, just tests the data loading pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from dvrk_scripts.constants_dvrk import TASK_CONFIGS
from utils import load_data_dvrk_multi_dataset, load_data_dvrk

def test_single_dataset():
    """Test single dataset loading (baseline)"""
    print("\n" + "="*60)
    print("TEST 1: Single Dataset Loading")
    print("="*60)
    
    task_name = 'cnh_exvivo'
    task_config = TASK_CONFIGS[task_name]
    
    try:
        train_loader, val_loader, stats, is_sim = load_data_dvrk(
            dataset_dir=task_config['dataset_dir'],
            num_episodes=task_config['num_episodes'],
            camera_names=task_config['camera_names'],
            batch_size_train=4,
            batch_size_val=4,
            task_config=task_config,
            chunk_size=60,
            use_language=True
        )
        
        print(f"✓ Successfully loaded single dataset")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader) if val_loader else 0}")
        
        # Test fetching one batch
        batch = next(iter(train_loader))
        print(f"  Batch structure: {len(batch)} items")
        print(f"  Image shape: {batch[0].shape}")
        print(f"  Action shape: {batch[1].shape if len(batch) > 3 else batch[2].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_dataset_auto_weights():
    """Test multi-dataset loading with automatic weighting"""
    print("\n" + "="*60)
    print("TEST 2: Multi-Dataset Loading (Auto Weights)")
    print("="*60)
    
    # Use two datasets that should be available
    task_names = ['cnh_exvivo', 'exvivo_29_auto_label_srt']

    dataset_dirs = []
    num_episodes_list = []
    task_configs_list = []
    
    for task_name in task_names:
        if task_name not in TASK_CONFIGS:
            print(f"⚠ Skipping test: {task_name} not found in TASK_CONFIGS")
            return None
        
        task_config = TASK_CONFIGS[task_name]
        dataset_dirs.append(task_config['dataset_dir'])
        num_episodes_list.append(task_config['num_episodes'])
        task_configs_list.append(task_config)
    
    try:
        train_loader, val_loader, stats, is_sim = load_data_dvrk_multi_dataset(
            dataset_dirs=dataset_dirs,
            num_episodes_list=num_episodes_list,
            camera_names=task_configs_list[0]['camera_names'],
            batch_size_train=4,
            batch_size_val=4,
            task_configs=task_configs_list,
            chunk_size=60,
            use_language=True,
            dataset_weights=None  # Automatic weighting
        )
        
        print(f"✓ Successfully loaded {len(task_names)} datasets")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader) if val_loader else 0}")
        
        # Test fetching batches
        batch = next(iter(train_loader))
        print(f"  Batch structure: {len(batch)} items")
        print(f"  Image shape: {batch[0].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_dataset_custom_weights():
    """Test multi-dataset loading with custom weights"""
    print("\n" + "="*60)
    print("TEST 3: Multi-Dataset Loading (Custom Weights)")
    print("="*60)
    
    task_names = ['cnh_exvivo', 'exvivo_29_auto_label_srt']
    custom_weights = [2.0, 1.0]  # Weight first dataset 2x more
    
    dataset_dirs = []
    num_episodes_list = []
    task_configs_list = []
    
    for task_name in task_names:
        if task_name not in TASK_CONFIGS:
            print(f"⚠ Skipping test: {task_name} not found in TASK_CONFIGS")
            return None
        
        task_config = TASK_CONFIGS[task_name]
        dataset_dirs.append(task_config['dataset_dir'])
        num_episodes_list.append(task_config['num_episodes'])
        task_configs_list.append(task_config)
    
    try:
        train_loader, val_loader, stats, is_sim = load_data_dvrk_multi_dataset(
            dataset_dirs=dataset_dirs,
            num_episodes_list=num_episodes_list,
            camera_names=task_configs_list[0]['camera_names'],
            batch_size_train=4,
            batch_size_val=4,
            task_configs=task_configs_list,
            chunk_size=60,
            use_language=True,
            dataset_weights=custom_weights
        )
        
        print(f"✓ Successfully loaded {len(task_names)} datasets with custom weights")
        print(f"  Weights used: {custom_weights}")
        print(f"  Train batches: {len(train_loader)}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# Multi-Dataset Training Test Suite")
    print("#"*60)
    
    results = {}
    
    # Test 1: Single dataset (baseline)
    results['single'] = test_single_dataset()
    
    # Test 2: Multi-dataset with auto weights
    results['multi_auto'] = test_multi_dataset_auto_weights()
    
    # Test 3: Multi-dataset with custom weights
    results['multi_custom'] = test_multi_dataset_custom_weights()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⚠ SKIPPED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(r in [True, None] for r in results.values())
    
    if all_passed:
        print("\n✓ All tests passed or skipped!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
