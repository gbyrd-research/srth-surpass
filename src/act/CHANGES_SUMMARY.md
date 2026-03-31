# Summary of Changes

## Overview
Enhanced the constants_dvrk.py module to support loading statistics from JSON files and dynamically creating task configurations.

## Files Modified

### 1. `/home/iulian/chole_ws/src/skay_jhu_private_new/src/act/calculate_std_mean.py`
**Changes:**
- Added parallel processing using multiprocessing for faster computation
- Replaced text output with structured JSON output
- Added timing and metadata tracking
- Optimized data processing with vectorized operations
- Still generates a summary text file for backward compatibility

**New Features:**
- Processes CSV files in parallel across multiple CPU cores
- Outputs results to `std_mean_invivo.json` with metadata
- Creates `std_mean_invivo_summary.txt` for quick reference
- Displays computation time and progress indicators

### 2. `/home/iulian/chole_ws/src/skay_jhu_private_new/src/act/dvrk_scripts/constants_dvrk.py`
**Changes:**
- Added `import json` for JSON file handling
- Added three new utility functions at the beginning of the file
- Added comprehensive documentation and examples

**New Functions:**
1. `load_statistics_from_json(json_path)`
   - Loads mean, std, min, max from JSON file
   - Returns dictionary with numpy arrays

2. `create_task_config_from_json(...)`
   - Creates a complete task configuration dictionary
   - Loads statistics from JSON
   - Accepts all standard task config parameters

3. `add_task_config_from_json(...)`
   - Creates task config and adds it to TASK_CONFIGS
   - Simplifies the process of adding new configs

## Files Created

### 3. `/home/iulian/chole_ws/src/skay_jhu_private_new/src/act/dvrk_scripts/example_load_json_config.py`
**Purpose:** Demonstration script showing how to use the new functionality

**Features:**
- Example 1: Loading statistics from JSON
- Example 2: Creating task config from JSON
- Example 3: Adding task config to TASK_CONFIGS
- Example 4: Comparing manual vs JSON-loaded configs

**Usage:**
```bash
cd /home/iulian/chole_ws/src/skay_jhu_private_new/src/act
python dvrk_scripts/example_load_json_config.py
```

### 4. `/home/iulian/chole_ws/src/skay_jhu_private_new/src/act/dvrk_scripts/quick_add_config.py`
**Purpose:** Command-line tool for quickly adding task configurations

**Features:**
- Add task configs from command line
- List all existing configurations
- Show statistics from JSON file
- Comprehensive argument parsing
- Validation and error checking

**Usage:**
```bash
python quick_add_config.py my_task ./std_mean_invivo.json \
    --dataset-dir /path/to/data \
    --tissue-ids 1 2 3 \
    --num-episodes 1000

# List existing configs
python quick_add_config.py --list-configs

# Show statistics
python quick_add_config.py --show-stats ./std_mean_invivo.json
```

**⚠️ IMPORTANT NOTE:**
The `quick_add_config.py` tool demonstrates how to create configs but does NOT permanently 
modify `constants_dvrk.py`. To use the config in your actual code, you need to:

**Option 1:** Add the function call in your Python script:
```python
from dvrk_scripts.constants_dvrk import add_task_config_from_json
add_task_config_from_json('my_task', './std_mean_invivo.json', 
                          dataset_dir="/path", tissue_ids=[1,2,3])
```

**Option 2:** Manually add the config to `constants_dvrk.py` in the `TASK_CONFIGS` dict

### 5. `/home/iulian/chole_ws/src/skay_jhu_private_new/src/act/dvrk_scripts/README_JSON_CONFIG.md`
**Purpose:** Comprehensive documentation

**Contents:**
- Overview of new functionality
- Quick start guide
- Function reference with examples
- Complete workflow documentation
- Command-line tool options
- Troubleshooting guide

## Usage Workflow

### Step 1: Generate Statistics
```bash
cd /home/iulian/chole_ws/src/skay_jhu_private_new/src/act
python calculate_std_mean.py
```
**Output:** 
- `std_mean_invivo.json` (JSON format)
- `std_mean_invivo_summary.txt` (text summary)

### Step 2: Create Task Config (Option A - Python)
```python
from dvrk_scripts.constants_dvrk import add_task_config_from_json

add_task_config_from_json(
    'my_task',
    './std_mean_invivo.json',
    dataset_dir="/home/iulian/chole_ws/data/cnh_exvivo_chole",
    tissue_ids=[1, 2, 3],
    num_episodes=1000
)
```

### Step 2: Create Task Config (Option B - Command Line)
```bash
cd dvrk_scripts
python quick_add_config.py my_task ../std_mean_invivo.json \
    --dataset-dir /home/iulian/chole_ws/data/cnh_exvivo_chole \
    --tissue-ids 1 2 3 \
    --num-episodes 1000
```

### Step 3: Use the Configuration
```python
from dvrk_scripts.constants_dvrk import TASK_CONFIGS

config = TASK_CONFIGS['my_task']
# Use config in your training/inference code
```

## Key Benefits

1. **Performance**: Parallel processing significantly speeds up statistics computation
2. **Automation**: No manual copy-paste of statistics arrays needed
3. **Reproducibility**: JSON files include metadata (timestamp, computation time, etc.)
4. **Flexibility**: Easy to create multiple configs with different parameters
5. **Version Control**: JSON files can be tracked in git
6. **Documentation**: Complete metadata and examples provided
7. **Usability**: Both Python API and command-line interface available

## Example Outputs

### JSON Statistics File Structure
```json
{
  "metadata": {
    "tissue_ids": [1, 2, 3],
    "data_directory": "/home/iulian/chole_ws/data/cnh_exvivo_chole",
    "computation_time_seconds": 45.67,
    "timestamp": "2026-01-24 10:30:00"
  },
  "statistics": {
    "mean": [0.001, -0.002, 0.001, ...],
    "std": [0.01, 0.015, 0.01, ...],
    "min": [-0.1, -0.08, -0.03, ...],
    "max": [0.09, 0.093, 0.12, ...]
  }
}
```

### Task Configuration Structure
The created task configs follow the same structure as existing configs in TASK_CONFIGS:
```python
{
    'dataset_dir': '/path/to/data',
    'tissue_samples_ids': [1, 2, 3],
    'num_episodes': 1000,
    'action_mode': ['hybrid', {
        'mean': np.array([...]),
        'std': np.array([...]),
        'min_': np.array([...]),
        'max_': np.array([...])
    }],
    'camera_names': ['left', 'left_wrist', 'right_wrist'],
    ...
}
```

## Testing

To test the new functionality:

1. Run the calculation script:
   ```bash
   python calculate_std_mean.py
   ```

2. Run the example demonstrations:
   ```bash
   python dvrk_scripts/example_load_json_config.py
   ```

3. Try the command-line tool:
   ```bash
   python dvrk_scripts/quick_add_config.py test_task ./std_mean_invivo.json \
       --dataset-dir /tmp/test \
       --tissue-ids 1 2 3 \
       --num-episodes 100
   ```

## Backward Compatibility

All changes are backward compatible:
- Existing TASK_CONFIGS entries remain unchanged
- New functions are additions, not modifications
- Text summary file still generated by calculate_std_mean.py
- No breaking changes to existing code

## Next Steps

1. Run `calculate_std_mean.py` to generate your first JSON statistics file
2. Try the examples in `example_load_json_config.py`
3. Use `quick_add_config.py` to add new task configurations
4. Integrate into your training pipelines as needed
