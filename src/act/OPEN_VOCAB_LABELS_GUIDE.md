# Open Vocabulary Labels Guide

## Overview
The dataset loader now supports open-vocabulary labels loaded from `labels.jsonl`. This allows you to use custom, frame-level annotations instead of pre-generated phase-level embeddings.

## Features
- **Dynamic label loading**: Labels are loaded from `labels.jsonl` files in each tissue folder
- **Timestamp-based lookup**: Labels are matched to frames based on timestamps
- **Pre-encoded embeddings**: All unique labels are pre-encoded during initialization for efficiency
- **Default label for unlabeled frames**: Frames without corresponding labels use "wound closure" as the default
- **Backward compatible**: Supports both the old (pre-generated embeddings) and new (open-vocab) approaches

## How to Use

### 1. Prepare labels.jsonl
Place a `labels.jsonl` file in **each tissue folder** (e.g., `tissue_1/labels.jsonl`, `tissue_2/labels.jsonl`, etc.) with the following format:

```jsonl
{"episode_index": 0, "start_frame": 0, "end_frame": 233, "start_timestamp": 0.0, "end_timestamp": 7.7667, "label": "Pick up long suture end with the left gripper."}
{"episode_index": 0, "start_frame": 233, "end_frame": 604, "start_timestamp": 7.7667, "end_timestamp": 20.1333, "label": "Double-wrap the suture around the right gripper."}
{"episode_index": 0, "episode_done": true}
{"episode_index": 1, "start_frame": 0, "end_frame": 188, "start_timestamp": 0.0, "end_timestamp": 6.2667, "label": "Use the left gripper to pick up the needle near the tip."}
...
```

**Important notes:**
- `episode_index` is 0-based **within each tissue folder**
- Each tissue folder can have its own independent episode indexing (e.g., both `tissue_1` and `tissue_2` can have episodes 0, 1, 2, ...)
- `start_timestamp` and `end_timestamp` are in seconds, starting from 0 for each episode
- Episode folders (e.g., `20250424-111109-544182`) are sorted alphabetically **within each tissue** to determine their index
- Include an `episode_done` marker at the end of each episode's labels

### 2. Enable in Task Config
Add the following to your task configuration:

```python
task_config = {
    'use_open_vocab_labels': True,  # Enable open vocabulary labels
    'use_language': True,            # Must be True to use language embeddings
    'language_encoder': 'distilbert', # Or your preferred encoder
    # ... other config options
}
```

### 3. Dataset Initialization
The dataset will automatically:
1. Search for `labels.jsonl` in each tissue folder (e.g., `tissue_1/`, `tissue_2/`, etc.)
2. Load and merge labels from all tissue folders
3. Build episode folder → episode index mapping
4. Pre-encode all unique labels using the specified language encoder
5. Create lookup structures for efficient timestamp-based retrieval

**Note:** It's okay if some tissue folders don't have `labels.jsonl` - the system will use "wound closure" as the default label for unlabeled episodes.

### 4. Runtime Behavior
During training:
1. A frame is sampled from an episode
2. The episode folder name is mapped to its episode index
3. The frame's timestamp is converted to relative seconds (from episode start)
4. The appropriate label is looked up based on the timestamp
5. The pre-encoded embedding for that label is returned

## Technical Details

### Default Label for Unlabeled Frames
The system uses **"wound closure"** as the default label in the following cases:
- Episodes from tissue folders that don't have `labels.jsonl`
- Episodes that are not present in their tissue folder's `labels.jsonl`
- Frames that fall outside any labeled time segment (gaps in labeling)
- Empty episodes with no label segments

This ensures training can proceed smoothly even with partially labeled datasets. The default label is always pre-encoded during initialization, so no runtime overhead is incurred.

### Episode Index Mapping
- Episodes are indexed **per-tissue** (0-based within each tissue folder)
- Episode folders within each tissue are sorted alphabetically (e.g., `20250424-111109-544182`)
- The oldest (lexicographically first) folder in each tissue gets index 0 within that tissue
- This mapping is built once during initialization
- **Example:**
  - `tissue_1`: episodes 0, 1, 2, ... (sorted folders within tissue_1)
  - `tissue_2`: episodes 0, 1, 2, ... (sorted folders within tissue_2)
  - `tissue_3`: episodes 0, 1, 2, ... (sorted folders within tissue_3)

### Timestamp Conversion
- CSV timestamps are in nanoseconds (absolute)
- labels.jsonl timestamps are in seconds (relative to episode start)
- Conversion: `relative_time_sec = (abs_timestamp_ns - episode_start_ns) / 1e9`

### Label Lookup
- Binary search is used for efficient timestamp → label matching
- A label is selected if: `start_timestamp ≤ frame_timestamp < end_timestamp`
- **Default label**: Unlabeled episodes and frames with no corresponding label return "wound closure"
- This handles both completely unlabeled episodes and gaps within labeled episodes

### Embedding Cache
- All unique labels are pre-encoded during `__init__`
- The default label "wound closure" is always pre-encoded, even if not in labels.jsonl
- Uses the same `encode_text()` function as the old system
- Stored as tensors in `self.label_embeddings_cache`

## Example Output
When loading with open vocabulary labels enabled:

```
Loading open vocabulary labels from labels.jsonl in tissue folders...

Loading labels from tissue_1/labels.jsonl...
Loading labels from tissue_2/labels.jsonl...
No labels.jsonl found in tissue_3/ (skipping)

Built episode index mapping for 20 episodes across 3 tissues

Pre-encoding unique labels...
Encoding labels: 100%|██████████| 245/245 [00:15<00:00, 15.89it/s]
Pre-encoded 245 unique labels
```

## Fallback Behavior
If `use_open_vocab_labels=True` but no `labels.jsonl` is found in any tissue folder:
- A warning is printed
- The system automatically falls back to the old pre-generated embeddings approach
- Training can continue without interruption

**Partial labeling is supported:** If only some tissue folders have `labels.jsonl`, the system will load those and use the default "wound closure" label for unlabeled episodes.

## Backward Compatibility
To use the old system (pre-generated embeddings):
- Set `use_open_vocab_labels=False` in task config (or omit it entirely)
- The dataset will load embeddings from `candidate_embeddings_{encoder}.json` as before

## Performance Considerations
- **Startup time**: Pre-encoding all labels takes ~15-30 seconds for 200-300 unique labels
- **Training time**: No overhead during training (embeddings are cached)
- **Memory**: Minimal (embeddings are small compared to image data)

## Troubleshooting

### Unlabeled Episodes
**Behavior**: Episodes not present in labels.jsonl will use "wound closure" as the default label
- This is expected behavior for partially labeled datasets
- No error will be raised
- Training will continue normally

### Error: "Episode folder not found in episode index mapping"
- Check that your episode folder names match the expected format
- Verify they are present in the dataset directory structure

### Error: "Label not found in pre-encoded embeddings cache"
- This shouldn't happen if initialization completed successfully (default "wound closure" label is always pre-encoded)
- Try re-running to regenerate the cache
