# Multi-Dataset Training Implementation Summary

## What Was Implemented

I've added complete support for training on multiple datasets simultaneously (co-training). This allows you to combine different surgical datasets to train a single model.

## Key Features

### 1. **Automatic Dataset Balancing**
   - Smaller datasets are automatically upsampled
   - Larger datasets are automatically downsampled
   - Ensures all datasets contribute equally to training

### 2. **Custom Dataset Weighting**
   - Optional: specify relative importance of each dataset
   - Useful when one dataset has higher quality or is more relevant

### 3. **Two-Level Balancing**
   - **Level 1**: Balance between datasets
   - **Level 2**: Balance between tasks/phases within each dataset
   - Prevents any single task or dataset from dominating

### 4. **Easy to Use**
   - Just provide multiple task names: `--task_name dataset1 dataset2 dataset3`
   - Optionally add weights: `--dataset_weights 2.0 1.5 1.0`

## Files Modified/Created

### Modified Files:
1. **`utils.py`**
   - Added `load_data_dvrk_multi_dataset()` function
   - Handles combining multiple datasets with `ConcatDataset`
   - Implements two-level weighted sampling

2. **`imitate_episodes.py`**
   - Modified to detect multiple task names
   - Automatically switches between single and multi-dataset mode
   - Added `--dataset_weights` argument

3. **`generic_dataset_invivo.py`**
   - No changes needed - already compatible!

### New Files:
1. **`run_multidataset_training.sh`**
   - Example training script with multiple configurations
   - Shows how to use the feature

2. **`MULTI_DATASET_TRAINING.md`**
   - Comprehensive documentation
   - Examples, best practices, troubleshooting

3. **`test_multidataset.py`**
   - Test suite to verify functionality
   - Tests single dataset, multi with auto weights, multi with custom weights

## How to Use

### Basic Usage (2 datasets, automatic weighting):

```bash
python imitate_episodes.py \
    --task_name cnh_exvivo suturing_final_map \
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

### With Custom Weights:

```bash
python imitate_episodes.py \
    --task_name cnh_exvivo suturing_final_map \
    --dataset_weights 2.0 1.0 \
    --ckpt_dir ckpt_dir_weighted \
    # ... other args
```

This will sample from `cnh_exvivo` 2x more often than `suturing_final_map`.

### Three or More Datasets:

```bash
python imitate_episodes.py \
    --task_name dataset1 dataset2 dataset3 \
    --dataset_weights 3.0 2.0 1.0 \
    # ... other args
```

## Example Output

When training starts, you'll see:

```
=== Loading 2 datasets for co-training ===

--- Dataset 1/2: /path/to/cnh_exvivo ---
Loading training data...
  Training samples: 1500
  Validation samples: 100

--- Dataset 2/2: /path/to/suturing ---
Loading training data...
  Training samples: 800
  Validation samples: 50

=== Combining datasets ===
Total training samples: 2300
Total validation samples: 150

Dataset sizes: [1500, 800]
Auto-computed dataset weights: ['0.77', '1.44']
Dataset 1 task counts: Counter({'1': 500, '2': 500, '3': 500})
Dataset 2 task counts: Counter({'1': 400, '2': 400})

=== Multi-dataset loading complete ===
```

## Testing

Run the test suite to verify everything works:

```bash
cd /home/iulian/chole_ws/src/skay_jhu_private_new/src/act
python test_multidataset.py
```

Or use the example script:

```bash
bash run_multidataset_training.sh
```

## Requirements

All datasets must have:
- ✅ Same camera configuration
- ✅ Same action dimensions (20-dim for low-level)
- ✅ Same language encoder (if using language)
- ✅ Valid entries in `TASK_CONFIGS`

## Benefits

1. **Better Generalization**: Model learns from diverse data
2. **Data Efficiency**: Use all available data at once
3. **Transfer Learning**: Combine source and target domains
4. **Curriculum Learning**: Weight simpler tasks higher initially

## Next Steps

1. **Test with your datasets**:
   ```bash
   python test_multidataset.py
   ```

2. **Start training on 2 datasets**:
   ```bash
   # Edit run_multidataset_training.sh with your dataset names
   bash run_multidataset_training.sh
   ```

3. **Monitor training**:
   - Check if both datasets are being used
   - Watch validation loss
   - Adjust weights if one dataset dominates

4. **Experiment with weights**:
   - Start with automatic (no weights specified)
   - If results are imbalanced, add custom weights
   - Higher weight = more samples from that dataset

## Troubleshooting

**Q: Training is slow**
- Reduce batch size or use fewer datasets

**Q: One dataset dominates**
- Adjust weights to balance contributions
- Check dataset sizes and task distributions

**Q: Camera names don't match**
- Ensure all datasets use same camera setup
- Modify `TASK_CONFIGS` to align camera names

**Q: Out of memory**
- Reduce batch size
- Reduce number of workers
- Use fewer datasets at once

## Documentation

- Full guide: `MULTI_DATASET_TRAINING.md`
- Example script: `run_multidataset_training.sh`
- Test suite: `test_multidataset.py`

## Technical Details

The implementation uses:
- `torch.utils.data.ConcatDataset` to combine datasets
- `torch.utils.data.WeightedRandomSampler` for balanced sampling
- Two-level weighting: dataset-level × task-level
- Efficient caching and parallel data loading

Sample weight formula:
```python
weight = dataset_weight * (1.0 / task_count_in_dataset)
```

All weights are normalized to ensure proper probability distribution.
