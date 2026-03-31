# Multi-Dataset Training Guide

This guide explains how to train on multiple datasets simultaneously using the new multi-dataset co-training functionality.

## Overview

The multi-dataset training system allows you to:
- Train a single policy on 2+ datasets simultaneously
- Automatically balance contributions from each dataset
- Apply custom weights to prioritize specific datasets
- Handle different dataset formats transparently

## Quick Start: Train on cnh_exvivo + exvivo_29_auto_label_srt

### Option 1: Using the Shell Script (Easiest)

```bash
# Default: Auto-balanced training on GPU 0
./run_multi_cnh_exvivo29.sh

# Use GPU 1
./run_multi_cnh_exvivo29.sh 1

# Custom weights: Weight cnh_exvivo 2x more than exvivo_29
./run_multi_cnh_exvivo29.sh 0 2.0 1.0
```

### Option 2: Using Python Script Directly

```bash
# Auto-balanced (default)
python train_multi_dataset_cnh_exvivo29.py --gpu 0

# Custom weights
python train_multi_dataset_cnh_exvivo29.py \
    --gpu 0 \
    --dataset_weights 2.0 1.0 \
    --batch_size 12 \
    --num_epochs 20000
```

### Option 3: Using imitate_episodes.py (General Purpose)

For any combination of datasets:

```bash
python imitate_episodes.py \
    --task_name cnh_exvivo exvivo_29_auto_label_srt base_chole_clipping_cutting \
    --dataset_weights 2.0 1.0 1.0 \
    --ckpt_dir ckpt_dir_multi_custom \
    --policy_class SRT \
    --batch_size 12 \
    --gpu 0 \
    # ... other args
```

**Note:** When you provide multiple `--task_name` arguments, the system automatically switches to multi-dataset mode.

## Dataset Weighting

### Automatic Balancing (Default)

When no weights are specified, the system automatically balances datasets:
- Each sample has weight = `dataset_weight × (1.0 / task_count_in_dataset)`
- Dataset weight is set to balance total samples from each dataset

**Example:**
- Dataset A: 1000 samples → weight = 1.0
- Dataset B: 500 samples → weight = 2.0 (to equalize)
- Result: Equal probability of sampling from A or B

### Custom Weights

You can specify custom weights to prioritize certain datasets:

```bash
--dataset_weights 2.0 1.0  # Sample from first dataset 2x more
```

**Example:**
- Dataset A: weight = 2.0
- Dataset B: weight = 1.0
- Result: Dataset A samples appear ~2x more often in batches

### When to Use Custom Weights

1. **Quality prioritization**: Weight higher-quality datasets more
2. **Task importance**: Weight more critical tasks higher
3. **Dataset size mismatch**: Manually adjust if auto-balance isn't ideal
4. **Transfer learning**: Weight target domain higher

## Supported Dataset Format Combinations

The system automatically detects and handles these format combinations:

| Dataset | CSV Format | Image Format | Example |
|---------|-----------|--------------|---------|
| cnh_exvivo | camera_source + timestamp | Timestamp | `1768581678498812246_left.jpg` |
| Jesse/suturing | timestamp only | Frame-indexed | `frame000000_left.jpg` |
| base_chole | timestamp only | Frame-indexed | `frame000000_left.jpg` |
| exvivo_29 | timestamp only | Frame-indexed | `frame000000_left.jpg` |

**Key Features:**
- ✅ CSV and image formats are checked **independently**
- ✅ Handles mixed format combinations in the same training run
- ✅ No manual configuration needed - fully automatic

## Training on Other Dataset Combinations

To train on different datasets, you have two options:

### Option A: Modify the Script

Edit `train_multi_dataset_cnh_exvivo29.py`:

```python
# Change line 60
task_names = ['cnh_exvivo', 'exvivo_29_auto_label_srt']

# To your desired datasets
task_names = ['dataset1', 'dataset2', 'dataset3']
```

### Option B: Use imitate_episodes.py

For maximum flexibility without creating new scripts:

```bash
python imitate_episodes.py \
    --task_name dataset1 dataset2 dataset3 \
    --dataset_weights 1.0 1.5 0.5 \
    --ckpt_dir ckpt_dir_my_multi_training \
    # ... other training args
```

## Monitoring Training

### Check Training Progress

```bash
# View live log
tail -f training_multi_cnh_exvivo29.log

# Check tensorboard (if configured)
tensorboard --logdir ckpt_dir_multi_cnh_exvivo29
```

### Verify Dataset Balancing

The training script prints:
```
Dataset Configuration:
  1. cnh_exvivo (weight: 1.0)
     Path: /path/to/cnh_exvivo
     Episodes: 250
  2. exvivo_29_auto_label_srt (weight: 1.0)
     Path: /path/to/exvivo_29
     Episodes: 180
```

### Check Batch Composition

To verify samples from all datasets are being used, check the data loading output.

## Resume Training

To resume from a checkpoint:

```bash
python train_multi_dataset_cnh_exvivo29.py \
    --gpu 0 \
    --resume_ckpt ckpt_dir_multi_cnh_exvivo29/policy_epoch_5000_seed_0.ckpt
```

## Testing Before Training

Use the test script to verify data loading without training:

```bash
python test_multidataset.py
```

This runs three tests:
1. Single dataset loading (baseline)
2. Multi-dataset with auto weights
3. Multi-dataset with custom weights

## Troubleshooting

### Error: "KeyError: 'task_name'"

**Cause:** Task name not found in `TASK_CONFIGS`

**Solution:** Check available tasks:
```python
from dvrk_scripts.constants_dvrk import TASK_CONFIGS
print(list(TASK_CONFIGS.keys()))
```

### Error: "Using fallback image" with large timestamp differences

**Cause:** Mismatch between CSV format and image format (should be fixed now)

**Solution:** This should be automatically handled. If you still see this:
1. Check that `generic_dataset_invivo.py` has the latest fixes
2. Verify images exist in the directory
3. Check image naming format matches CSV format

### Warning: "Number of weights doesn't match number of datasets"

**Cause:** Provided weights list length doesn't match number of datasets

**Solution:**
```bash
# Wrong: 3 datasets, 2 weights
--task_name ds1 ds2 ds3 --dataset_weights 1.0 1.0

# Correct: 3 datasets, 3 weights
--task_name ds1 ds2 ds3 --dataset_weights 1.0 1.0 1.0
```

### Out of Memory (OOM) Error

**Solutions:**
1. Reduce batch size: `--batch_size 8`
2. Use fewer cameras: modify `camera_names` in config
3. Reduce chunk size: `--chunk_size 30`
4. Use gradient accumulation (modify training code)

## Advanced Configuration

### Different Episode Counts per Dataset

By default, uses episode count from each task's config. To customize:

Edit the script and modify:
```python
num_episodes_list = [100, 150]  # Custom episode counts
```

### Different Validation Splits

Each dataset can have different validation splits defined in `TASK_CONFIGS`.

### Language Conditioning

The system supports language conditioning across multiple datasets:

```bash
--use_language \
--language_encoder distilbert  # or 'clip'
```

Each dataset's task descriptions will be encoded and used for conditioning.

## Performance Tips

1. **Start with auto-balancing** - only add custom weights if needed
2. **Monitor validation loss** per dataset if possible
3. **Use similar datasets** for best results (same robot, similar tasks)
4. **Larger batch sizes** generally work better for multi-dataset training
5. **Longer training** may be needed when combining datasets

## Example Training Runs

### Balanced Training (Default)
```bash
./run_multi_cnh_exvivo29.sh 0
# Equal sampling from both datasets
```

### Prioritize cnh_exvivo
```bash
./run_multi_cnh_exvivo29.sh 0 3.0 1.0
# 75% samples from cnh_exvivo, 25% from exvivo_29
```

### Prioritize exvivo_29
```bash
./run_multi_cnh_exvivo29.sh 0 1.0 3.0
# 25% samples from cnh_exvivo, 75% from exvivo_29
```

### Three Datasets with Custom Weights
```bash
python imitate_episodes.py \
    --task_name cnh_exvivo exvivo_29_auto_label_srt base_chole_clipping_cutting \
    --dataset_weights 2.0 1.5 1.0 \
    --ckpt_dir ckpt_dir_multi_three \
    --gpu 1 \
    --batch_size 16
```

## Files Created During Multi-Dataset Training

```
ckpt_dir_multi_cnh_exvivo29/
├── training_config.txt           # Full training configuration
├── policy_epoch_500_seed_0.ckpt  # Checkpoint every 500 epochs
├── policy_epoch_1000_seed_0.ckpt
└── ...

training_multi_cnh_exvivo29.log   # Full training log
```

## Next Steps

After training:
1. **Evaluate** the policy on validation sets from both datasets
2. **Compare** to single-dataset baselines
3. **Fine-tune** dataset weights if performance is imbalanced
4. **Test** on real robot or additional test episodes

## Support

For issues or questions:
1. Check the test scripts: `test_multidataset.py`, `test_all_format_combinations.py`
2. Review documentation: `MULTI_DATASET_FIX_SUMMARY.md`
3. Check the implementation in `utils.py` (line ~350) and `generic_dataset_invivo.py`
