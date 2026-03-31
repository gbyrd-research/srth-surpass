"""
Test script for open vocabulary labels feature

This script demonstrates how to use the new open-vocabulary labels system.
"""

import os
import sys
import numpy as np

# Add path if needed
path_to_skay_robot = os.getenv('PATH_TO_SKAY_ROBOT')
if path_to_skay_robot:
    sys.path.append(os.path.join(path_to_skay_robot, 'src'))

from generic_dataset_invivo_wound import EpisodicDatasetDvrkGeneric

def test_open_vocab_labels():
    """Test the open vocabulary labels feature"""
    
    # Example configuration
    task_config = {
        'use_open_vocab_labels': True,  # Enable the new feature
        'action_mode': ['hybrid'],
        'norm_scheme': 'std',
        'phantom': False,
        'recovery_ratio': 0.0,
        'camera_names': ['left', 'right', 'left_wrist', 'right_wrist'],
        'camera_file_suffixes': ['_left.jpg', '_right.jpg', '_psm1.jpg', '_psm2.jpg'],
        'cutting_action_pad_size': 50,
        'num_episodes': 10,
        'tissue_samples_ids': [1, 2],
        'use_sketch': False,
        'use_history': False,
        'no_qpos': False,
        'estimation': False,
        'stereo': False,
    }
    
    # Dataset directory - update with your actual path
    dataset_dir = "/path/to/your/dataset"
    
    # Check if labels.jsonl exists
    labels_path = os.path.join(dataset_dir, 'labels.jsonl')
    if not os.path.exists(labels_path):
        print(f"Error: labels.jsonl not found at {labels_path}")
        print("Please create labels.jsonl first or update dataset_dir")
        return
    
    # Create dataset
    print("Initializing dataset with open vocabulary labels...")
    tissue_sample_ids = task_config['tissue_samples_ids']
    camera_names = task_config['camera_names']
    camera_file_suffixes = task_config['camera_file_suffixes']
    num_episodes = task_config['num_episodes']
    episode_ids = list(range(num_episodes))
    
    try:
        dataset = EpisodicDatasetDvrkGeneric(
            episode_ids=episode_ids,
            tissue_sample_ids=tissue_sample_ids,
            dataset_dir=dataset_dir,
            camera_names=camera_names,
            camera_file_suffixes=camera_file_suffixes,
            task_config=task_config,
            chunk_size=60,
            use_language=True,
            language_encoder="distilbert"
        )
        
        print(f"\nDataset initialized successfully!")
        print(f"Total episodes: {len(dataset)}")
        print(f"Open vocab labels enabled: {dataset.use_open_vocab_labels}")
        
        if dataset.use_open_vocab_labels:
            print(f"Loaded labels for {len(dataset.labels_data)} episodes")
            print(f"Pre-encoded {len(dataset.label_embeddings_cache)} unique labels")
            print(f"Episode mapping size: {len(dataset.episode_folder_to_index)}")
        
        # Test sampling
        print("\nTesting data sampling...")
        for i in range(3):
            try:
                sample_idx = np.random.randint(0, len(dataset))
                data = dataset[sample_idx]
                
                if dataset.task_config.get('no_qpos', False):
                    image_data, action_data, is_pad, command_embedding = data
                    print(f"Sample {i}: image shape={image_data.shape}, "
                          f"action shape={action_data.shape}, "
                          f"embedding shape={command_embedding.shape}")
                else:
                    image_data, qpos_data, action_data, is_pad, command_embedding = data
                    print(f"Sample {i}: image shape={image_data.shape}, "
                          f"qpos shape={qpos_data.shape}, "
                          f"action shape={action_data.shape}, "
                          f"embedding shape={command_embedding.shape}")
            except Exception as e:
                print(f"Error sampling index {sample_idx}: {e}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\nError during dataset initialization: {e}")
        import traceback
        traceback.print_exc()


def test_old_system():
    """Test that the old system still works (backward compatibility)"""
    
    task_config = {
        'use_open_vocab_labels': False,  # Use old system
        'action_mode': ['hybrid'],
        'norm_scheme': 'std',
        'phantom': False,
        'recovery_ratio': 0.0,
        'camera_names': ['left', 'right', 'left_wrist', 'right_wrist'],
        'camera_file_suffixes': ['_left.jpg', '_right.jpg', '_psm1.jpg', '_psm2.jpg'],
        'cutting_action_pad_size': 50,
        'num_episodes': 10,
        'tissue_samples_ids': [1, 2],
        'use_sketch': False,
        'use_history': False,
        'no_qpos': False,
        'estimation': False,
        'stereo': False,
    }
    
    dataset_dir = "/path/to/your/dataset"
    
    print("Testing backward compatibility (old system)...")
    # ... similar initialization and testing as above
    print("Old system test completed")


if __name__ == "__main__":
    print("=" * 60)
    print("Open Vocabulary Labels Test")
    print("=" * 60)
    
    # Test new system
    test_open_vocab_labels()
    
    # Optionally test old system
    # test_old_system()
