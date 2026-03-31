# Number of Episodes in JSON - Feature Update

## Overview
The `calculate_std_mean.py` script now automatically includes the **number of episodes** (CSV files processed) in the generated JSON file.

## What Changed

### 1. `calculate_std_mean.py`
- `compute_diffs()` now returns an additional value: `num_episodes`
- `num_episodes` is calculated as the total count of CSV files found
- This value is included in the JSON output under `metadata.num_episodes`
- Also displayed in the console output and text summary

### 2. `constants_dvrk.py`
- `load_statistics_from_json()` now returns `num_episodes` if available
- `create_task_config_from_json()` automatically uses `num_episodes` from JSON if not manually specified
- This makes config creation even easier!

## JSON Output Format

```json
{
  "metadata": {
    "tissue_ids": [1, 2, 3],
    "data_directory": "/path/to/data",
    "num_episodes": 853,           ← NEW!
    "computation_time_seconds": 45.67,
    "timestamp": "2026-01-24 10:30:00"
  },
  "statistics": {
    "mean": [...],
    "std": [...],
    "min": [...],
    "max": [...]
  }
}
```

## Usage Examples

### Example 1: Automatic num_episodes (Recommended)

```python
from dvrk_scripts.constants_dvrk import add_task_config_from_json

# num_episodes is automatically loaded from JSON!
add_task_config_from_json(
    'my_task',
    './std_mean_invivo.json',
    dataset_dir="/path/to/data",
    tissue_ids=[1, 2, 3]
    # No need to specify num_episodes!
)
```

### Example 2: Override num_episodes

```python
# You can still override it if needed
add_task_config_from_json(
    'my_task',
    './std_mean_invivo.json',
    dataset_dir="/path/to/data",
    tissue_ids=[1, 2, 3],
    num_episodes=1000  # Override the value from JSON
)
```

### Example 3: Check num_episodes from JSON

```python
from dvrk_scripts.constants_dvrk import load_statistics_from_json

stats = load_statistics_from_json('./std_mean_invivo.json')
print(f"Number of episodes: {stats.get('num_episodes', 'Not available')}")
print(f"Mean shape: {stats['mean'].shape}")
```

## Command-Line Tools

The command-line tools also benefit from this feature:

```bash
# Without specifying num_episodes - it's loaded automatically!
python dvrk_scripts/quick_add_config.py my_task ./std_mean_invivo.json \
    --dataset-dir /path/to/data \
    --tissue-ids 1 2 3
    # num_episodes is automatically pulled from JSON

# You can still override if needed
python dvrk_scripts/quick_add_config.py my_task ./std_mean_invivo.json \
    --dataset-dir /path/to/data \
    --tissue-ids 1 2 3 \
    --num-episodes 1000
```

## Benefits

1. **No Manual Counting**: Don't need to manually count episodes anymore
2. **Accuracy**: Guaranteed to match the actual data used for statistics
3. **Convenience**: One less parameter to specify when creating configs
4. **Traceability**: Know exactly how many episodes were in the original computation
5. **Backward Compatible**: Old JSON files without `num_episodes` still work

## Console Output Example

When you run `calculate_std_mean.py`, you'll now see:

```
Scanning /home/iulian/chole_ws/data/cnh_exvivo_chole/tissue_1
Scanning /home/iulian/chole_ws/data/cnh_exvivo_chole/tissue_2
Scanning /home/iulian/chole_ws/data/cnh_exvivo_chole/tissue_3
Total CSV files found: 853
Using multiprocessing with 8 workers
Total episodes (CSV files): 853        ← NEW!
Total diff arrays: 853
Total samples: 426500
Computation completed in 32.45 seconds

================================================================================
RESULTS SUMMARY
================================================================================
Number of Episodes: 853                 ← NEW!
Mean: [0.001 -0.002 0.001 ...]
Std: [0.01 0.015 0.01 ...]
Min: [-0.1 -0.08 -0.03 ...]
Max: [0.09 0.093 0.12 ...]
================================================================================

Results written to ./std_mean_invivo.json
Summary written to ./std_mean_invivo_summary.txt
```

## Text Summary File

The text summary also includes the number of episodes:

```
Tissue IDs: [1, 2, 3]
Number of Episodes: 853                 ← NEW!
Computation Time: 32.45 seconds
Timestamp: 2026-01-24 10:30:00
Mean: 0.001, -0.002, 0.001, ...
Std: 0.01, 0.015, 0.01, ...
Min: -0.1, -0.08, -0.03, ...
Max: 0.09, 0.093, 0.12, ...
```

## Workflow

### Step 1: Generate Statistics (includes num_episodes automatically)
```bash
python calculate_std_mean.py
```

### Step 2: Create Config (num_episodes loaded automatically)
```python
from dvrk_scripts.constants_dvrk import add_task_config_from_json

add_task_config_from_json(
    'my_task',
    './std_mean_invivo.json',
    dataset_dir="/path/to/data",
    tissue_ids=[1, 2, 3]
)
# num_episodes is automatically 853 from JSON!
```

### Step 3: Verify
```python
from dvrk_scripts.constants_dvrk import TASK_CONFIGS

config = TASK_CONFIGS['my_task']
print(f"Episodes: {config['num_episodes']}")  # Output: Episodes: 853
```

## Notes

- **What counts as an episode**: Each CSV file (typically `ee_csv.csv`) found in the dataset directories
- **Automatic detection**: The script scans all tissue directories and phases to find all CSV files
- **Corrections folder**: Special handling for "Corrections" subdirectories is maintained
- **Validation**: If no CSV files are found, the script will raise an error

## Backward Compatibility

✅ Old JSON files without `num_episodes` continue to work
✅ You can still manually specify `num_episodes` to override
✅ No breaking changes to existing code

## See Also

- `calculate_std_mean.py` - Generate statistics with num_episodes
- `constants_dvrk.py` - Load statistics and create configs
- `README_JSON_CONFIG.md` - Full documentation on JSON config system
