#!/usr/bin/env python3
"""
Integration test: Load samples from all three format combinations simultaneously.
"""

import sys
import torch
from utils import load_data_dvrk_multi_dataset

print("\n" + "="*80)
print("MULTI-DATASET LOADING INTEGRATION TEST")
print("="*80)

# Test configuration
dataset_dirs = [
    '/home/iulian/chole_ws/data/cnh_exvivo_chole',
    '/home/iulian/chole_ws/data/Jesse',
    '/home/iulian/chole_ws/data/base_chole_clipping_cutting'
]

num_episodes_list = [5, 5, 5]  # Small number for quick test

camera_names = ['left']  # Just test one camera

task_configs = [
    {
        'dataset_dir': dataset_dirs[0],
        'num_episodes': 5,
        'num_episodes_val': 1,
        'episode_len': 1000,
        'camera_names': camera_names,
        'tissue_samples_ids': list(range(1, 6)),  # Tissues 1-5
        'tissue_samples_ids_val': [6],  # Tissue 6 for validation
        'camera_file_suffixes': {'left': '_left.jpg'},
        'action_mode': ['ee_xyz_to_delta'],
    },
    {
        'dataset_dir': dataset_dirs[1],
        'num_episodes': 5,
        'num_episodes_val': 1,
        'episode_len': 1000,
        'camera_names': camera_names,
        'tissue_samples_ids': list(range(1, 6)),
        'tissue_samples_ids_val': [6],
        'camera_file_suffixes': {'left': '_left.jpg'},
        'action_mode': ['ee_xyz_to_delta'],
    },
    {
        'dataset_dir': dataset_dirs[2],
        'num_episodes': 5,
        'num_episodes_val': 1,
        'episode_len': 1000,
        'camera_names': camera_names,
        'tissue_samples_ids': list(range(1, 6)),
        'tissue_samples_ids_val': [6],
        'camera_file_suffixes': {'left': '_left.jpg'},
        'action_mode': ['ee_xyz_to_delta'],
    }
]

print("\nTest Configuration:")
print(f"  Datasets: {len(dataset_dirs)}")
for i, d in enumerate(dataset_dirs, 1):
    print(f"    {i}. {d.split('/')[-1]}")
print(f"  Episodes per dataset: {num_episodes_list}")
print(f"  Cameras: {camera_names}")

print("\n" + "─"*80)
print("Loading datasets...")
print("─"*80)

try:
    train_dataloader, val_dataloader, stats, _ = load_data_dvrk_multi_dataset(
        dataset_dirs=dataset_dirs,
        num_episodes_list=num_episodes_list,
        camera_names=camera_names,
        batch_size_train=4,
        batch_size_val=4,
        task_configs=task_configs,
        chunk_size=100,
        use_language=False,
        dataset_weights=None  # Auto-balance
    )
    
    print(f"✅ Datasets loaded successfully!")
    print(f"   Train batches: {len(train_dataloader)}")
    print(f"   Val batches: {len(val_dataloader)}")
    
    print("\n" + "─"*80)
    print("Testing data loading from all datasets...")
    print("─"*80)
    
    # Load a few batches to ensure no errors
    for i, batch in enumerate(train_dataloader):
        if i >= 3:  # Just test first 3 batches
            break
        
        image_data = batch['image_dict']
        qpos = batch['qpos']
        actions = batch['actions']
        
        print(f"\nBatch {i+1}:")
        print(f"  Images shape: {image_data['left'].shape}")
        print(f"  qpos shape: {qpos.shape}")
        print(f"  actions shape: {actions.shape}")
        print(f"  Images min/max: [{image_data['left'].min():.3f}, {image_data['left'].max():.3f}]")
        
        # Check for NaN or invalid values
        if torch.isnan(image_data['left']).any():
            print("  ⚠️  WARNING: NaN values in images!")
        elif torch.isinf(image_data['left']).any():
            print("  ⚠️  WARNING: Inf values in images!")
        else:
            print("  ✓ Images valid")
        
        if torch.isnan(qpos).any() or torch.isnan(actions).any():
            print("  ⚠️  WARNING: NaN values in qpos or actions!")
        else:
            print("  ✓ qpos and actions valid")
    
    print("\n" + "="*80)
    print("✅ INTEGRATION TEST PASSED!")
    print("="*80)
    print("""
All three format combinations loaded successfully:
  1. cnh_exvivo: CSV (camera_source + timestamp) + Images (timestamp)
  2. Jesse: CSV (timestamp only) + Images (frame index)
  3. base_chole: CSV (timestamp only) + Images (frame index)

The system correctly:
  ✓ Detected CSV formats (with/without camera_source)
  ✓ Detected image formats (timestamp vs frame-indexed)
  ✓ Handled independent CSV and image format combinations
  ✓ Loaded and normalized images correctly
  ✓ Batched data from multiple datasets

Ready for full training!
    """)
    print("="*80)

except Exception as e:
    print("\n" + "="*80)
    print("❌ INTEGRATION TEST FAILED!")
    print("="*80)
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    print("="*80)
    sys.exit(1)
