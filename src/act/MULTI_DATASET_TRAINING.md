# Multi-Dataset Co-Training

This document explains how to train a model on multiple datasets simultaneously.

## Overview

The multi-dataset training feature allows you to:
- Train on multiple datasets at once (e.g., cnh_exvivo + suturing + invivo)
- Automatically balance sampling across datasets
- Optionally specify custom weights for each dataset
- Handle datasets with different task distributions

## Quick Start

### Basic Usage (Automatic Weighting)

To train on multiple datasets with automatic weighting:

```bash
python imitate_episodes.py \
    --task_name dataset1 dataset2 dataset3 \
    --ckpt_dir ckpt_dir_multi \
    --policy_class SRT \
    --chunk_size 60 \
    --hidden_dim 512 \
    --batch_size 12 \
    --num_epochs 20000 \
    --lr 1e-5 \
    --seed 0 \
    --use_language \
    --language_encoder distilbert \
    --image_encoder efficientnet_b3film \
    --gpu 0 \
    --policy_level low
```

### Custom Dataset Weights

To specify custom sampling weights for each dataset:

```bash
python imitate_episodes.py \
    --task_name dataset1 dataset2 \
    --dataset_weights 2.0 1.0 \
    --ckpt_dir ckpt_dir_multi_weighted \
    # ... other args
```

This will sample from dataset1 twice as often as dataset2.

## How It Works

### 1. Automatic Weighting (Default)

When no `--dataset_weights` are provided, the system uses **inverse frequency weighting**:
- Smaller datasets get higher sampling weights
- Larger datasets get lower sampling weights
- This ensures balanced representation from all datasets

Formula: `weight_i = total_samples / dataset_i_samples`

### 2. Custom Weighting

When `--dataset_weights` are provided:
- You specify the relative importance of each dataset
- Weights are combined with per-task balancing within each dataset
- Higher weight = more samples from that dataset during training

### 3. Two-Level Balancing

The system performs balancing at two levels:

**Level 1: Dataset-level balancing**
- Balances sampling between different datasets
- Uses inverse frequency or custom weights

**Level 2: Task-level balancing (within each dataset)**
- Balances sampling between different tasks/phases within each dataset
- Uses inverse frequency based on task labels (e.g., phase_1, phase_2, phase_3)

**Combined formula:**
```
sample_weight = dataset_weight * (1.0 / task_count_in_dataset)
```

## Examples

### Example 1: Two Datasets (Automatic)

```bash
python imitate_episodes.py \
    --task_name cnh_exvivo suturing_final_map \
    --ckpt_dir ckpt_dir_cnh_suturing \
    --policy_class SRT \
    --chunk_size 60 \
    --hidden_dim 512 \
    --batch_size 12 \
    --num_epochs 20000 \
    --lr 1e-5 \
    --seed 0 \
    --use_language \
    --language_encoder distilbert \
    --image_encoder efficientnet_b3film \
    --gpu 0 \
    --policy_level low
```

**Result:**
- If cnh_exvivo has 1000 samples and suturing has 500 samples
- Auto weights: cnh=0.67, suturing=1.33
- Suturing samples will be upsampled to match cnh frequency

### Example 2: Two Datasets (Custom Weights)

```bash
python imitate_episodes.py \
    --task_name cnh_exvivo suturing_final_map \
    --dataset_weights 3.0 1.0 \
    --ckpt_dir ckpt_dir_cnh_weighted \
    # ... other args
```

**Result:**
- cnh_exvivo samples will appear 3x more frequently in training batches
- Use this when one dataset is more important or has higher quality data

### Example 3: Three Datasets

```bash
python imitate_episodes.py \
    --task_name cnh_exvivo suturing_final_map invivo_test \
    --dataset_weights 2.0 1.5 1.0 \
    --ckpt_dir ckpt_dir_three_datasets \
    # ... other args
```

**Result:**
- Relative sampling: cnh=2x, suturing=1.5x, invivo=1x
- All three datasets will be used in each epoch

## Requirements

### Dataset Compatibility

For multi-dataset training to work, datasets must have:
- ✅ Same camera configuration (camera_names must match)
- ✅ Same action space (20-dim for low-level policy)
- ✅ Same normalization scheme (or None for separate per-dataset normalization)
- ✅ Compatible image sizes (resized to same dimensions)
- ⚠️ Language embeddings must be from the same encoder

### Task Configuration

Each dataset must have a valid `TASK_CONFIGS` entry in `dvrk_scripts/constants_dvrk.py`:

```python
TASK_CONFIGS = {
    'your_dataset': {
        'dataset_dir': '/path/to/data',
        'num_episodes': 1000,
        'num_episodes_val': 100,
        'tissue_samples_ids': [1, 2, 3],
        'tissue_samples_ids_val': [4],
        'camera_names': ['left', 'left_wrist', 'right_wrist'],
        'camera_file_suffixes': ['_left.jpg', '_psm1.jpg', '_psm2.jpg'],
        'episode_len': 500,
        'action_mode': ['hybrid', {...}],
        'norm_scheme': 'std',
        'no_qpos': True,
        'use_language': True,
        # ... other config
    }
}
```

## Output and Logging

During training, you'll see:

```
=== Loading 2 datasets for co-training ===

--- Dataset 1/2: /path/to/dataset1 ---
Loading training data from /path/to/dataset1
  Training samples: 1500
  Validation samples: 100

--- Dataset 2/2: /path/to/dataset2 ---
Loading training data from /path/to/dataset2
  Training samples: 800
  Validation samples: 50

=== Combining datasets ===
Total training samples: 2300
Total validation samples: 150

Dataset sizes: [1500, 800]
Auto-computed dataset weights (inverse frequency): ['0.77', '1.44']
Dataset 1 task counts: Counter({'1': 500, '2': 500, '3': 500})
Dataset 2 task counts: Counter({'1': 400, '2': 400})

=== Multi-dataset loading complete ===
```

## Best Practices

### 1. Start with Automatic Weighting
- Let the system balance datasets automatically first
- Only use custom weights if you have a specific reason

### 2. Monitor Training Metrics
- Watch validation loss for each dataset separately (if using wandb)
- Check if one dataset is dominating the training

### 3. Dataset Weights Guidelines
- **Equal importance**: Use automatic weighting
- **Quality difference**: Weight higher-quality dataset more (e.g., 2.0 vs 1.0)
- **Domain preference**: Weight target domain more (e.g., 3.0 for target, 1.0 for source)
- **Curriculum learning**: Start with simpler dataset weighted higher, then adjust

### 4. Validation Strategy
- Each dataset contributes validation samples
- Combined validation dataloader samples from all datasets
- Monitor per-dataset performance if possible

## Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce batch size or use fewer datasets

### Issue: One dataset dominates
**Solution:** Adjust custom weights to balance contributions

### Issue: Camera names don't match
**Solution:** Ensure all datasets use the same camera configuration

### Issue: Normalization mismatch
**Solution:** Use 'std' or 'min_max' consistently, or set to None

### Issue: Language embeddings incompatible
**Solution:** Ensure all datasets use the same language encoder (distilbert or clip)

## Advanced: Understanding the Sampler

The `WeightedRandomSampler` performs:

1. **Sample weight calculation**: For each sample in the combined dataset
   ```python
   weight = dataset_weight * (1.0 / task_count_in_dataset)
   ```

2. **Normalization**: Weights are normalized to sum to `len(dataset)`

3. **Sampling**: In each epoch, samples are drawn with replacement according to weights

This ensures:
- Balanced dataset representation
- Balanced task representation within each dataset
- No dataset or task is ignored during training

## Implementation Details

Key files:
- `utils.py`: `load_data_dvrk_multi_dataset()` - Main loading function
- `imitate_episodes.py`: Multi-dataset training logic
- `generic_dataset_invivo.py`: Individual dataset loader

The implementation uses PyTorch's `ConcatDataset` and `WeightedRandomSampler` for efficient multi-dataset training.
