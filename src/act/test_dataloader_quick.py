#!/usr/bin/env python3
"""
Quick test to verify the new camera-synchronized dataloader works
"""

import numpy as np
import os
import sys

# Add path to dvrk_scripts
path_to_yay_robot = os.getenv('PATH_TO_SKAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))

from generic_dataset_invivo import EpisodicDatasetDvrkGeneric

def test_dataloader():
    """Test basic dataloader functionality"""
    
    print("=" * 70)
    print("TESTING NEW CAMERA-SYNCHRONIZED DATALOADER")
    print("=" * 70)
    
    # Setup paths
    path_to_dataset = os.getenv("PATH_TO_DATASET")
    if not path_to_dataset:
        print("❌ PATH_TO_DATASET environment variable not set")
        return False
    
    dataset_dir = os.path.join(path_to_dataset, "cnh_exvivo_chole")
    
    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    print(f"✅ Dataset directory found: {dataset_dir}\n")
    
    # Load task config
    sys.path.append('/home/iulian/chole_ws/src/skay_jhu_private_new/src')
    from dvrk_scripts.constants_dvrk import TASK_CONFIGS
    
    task_config = TASK_CONFIGS['cnh_exvivo']
    camera_names = task_config['camera_names']
    tissue_samples_ids = task_config["tissue_samples_ids"][:2]  # Test with first 2 tissues
    num_episodes = 10  # Test with small subset
    camera_file_suffixes = task_config['camera_file_suffixes']
    
    print(f"Camera names: {camera_names}")
    print(f"Camera suffixes: {camera_file_suffixes}")
    print(f"Testing with tissues: {tissue_samples_ids}")
    print(f"Number of test samples: {num_episodes}\n")
    
    # Create dataset
    try:
        episode_ids = [i for i in range(num_episodes)]
        dataset = EpisodicDatasetDvrkGeneric(
            episode_ids,
            tissue_samples_ids,
            dataset_dir,
            camera_names,
            camera_file_suffixes,
            task_config,
            chunk_size=60,
            use_language=True
        )
        print(f"✅ Dataset created successfully")
        print(f"   Total samples available: {len(dataset.all_samples)}\n")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return False
    
    # Test loading samples
    print("Testing sample loading...")
    print("-" * 70)
    
    success_count = 0
    fail_count = 0
    
    for i in range(min(5, len(dataset))):
        try:
            # Load sample
            result = dataset[i]
            
            # Unpack based on configuration
            no_qpos = task_config.get('no_qpos', False)
            if no_qpos:
                image_data, action_data, is_pad, command_embedding = result
            else:
                image_data, qpos_data, action_data, is_pad, command_embedding = result
            
            # Verify shapes
            print(f"Sample {i}:")
            print(f"  ✅ Image shape: {image_data.shape}")
            print(f"  ✅ Action shape: {action_data.shape}")
            print(f"  ✅ Is_pad shape: {is_pad.shape}")
            print(f"  ✅ Command embedding shape: {command_embedding.shape}")
            
            # Verify data types
            assert image_data.dtype == np.float32 or image_data.dtype == np.float64
            assert action_data.dtype == np.float32
            
            # Verify number of cameras
            assert image_data.shape[0] == len(camera_names), \
                f"Expected {len(camera_names)} cameras, got {image_data.shape[0]}"
            
            success_count += 1
            print(f"  ✅ Sample {i} loaded successfully\n")
            
        except Exception as e:
            print(f"  ❌ Sample {i} failed: {e}\n")
            fail_count += 1
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Successful loads: {success_count}/{success_count + fail_count}")
    print(f"❌ Failed loads: {fail_count}/{success_count + fail_count}")
    
    if fail_count == 0:
        print("\n🎉 ALL TESTS PASSED! Dataloader working correctly.")
        return True
    else:
        print(f"\n⚠️  {fail_count} tests failed. Please check errors above.")
        return False


if __name__ == "__main__":
    success = test_dataloader()
    sys.exit(0 if success else 1)
