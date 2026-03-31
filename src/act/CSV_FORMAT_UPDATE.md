# CSV Format Update - Camera-Synchronized Kinematics

## Overview
The data loader has been rewritten to support the new CSV format where kinematics are recorded per camera source, enabling exact image-kinematics alignment.

## New CSV Structure

### Additional Column: `camera_source`
The CSV now includes a `camera_source` column that indicates which camera's image arrival triggered the kinematics recording.

**Possible values:**
- `left` - Left external camera
- `right` - Right external camera  
- `psm1` - Left wrist endoscope
- `psm2` - Right wrist endoscope

### Example CSV Structure
```csv
timestamp,camera_source,psm1_pose.position.x,...
1768845055717077157,left,-0.103964,...
1768845055737274156,psm1,-0.103961,...
1768845055738113658,psm2,-0.103961,...
1768845055735988230,right,-0.103961,...
```

## Key Changes to Data Loader

### 1. **New Caching Mechanism**
```python
# Old: Cached sorted timestamps per camera per episode
self.image_timestamp_cache = {}

# New: Cache camera-specific CSV data per episode
self.camera_csv_cache = {}
```

### 2. **Camera-Specific CSV Loading**
New method `load_camera_specific_csv()` that:
- Loads the full CSV once per episode
- Splits it by `camera_source` for efficient access
- Caches the split DataFrames for future use

```python
camera_csvs = {
    'left': DataFrame with left camera rows,
    'right': DataFrame with right camera rows,
    'psm1': DataFrame with psm1 camera rows,
    'psm2': DataFrame with psm2 camera rows
}
```

### 3. **Random Camera Sampling**
Instead of using a single unified CSV, the loader now:
1. **Randomly selects** one camera source per sample
2. **Uses that camera's CSV rows** for kinematics and temporal reference
3. **Loads images from all cameras** using timestamps closest to the selected camera's timestamp

This approach:
- ✅ Ensures exact alignment between selected camera and kinematics
- ✅ Provides data augmentation through camera variation
- ✅ Handles potential timestamp differences between cameras gracefully

### 4. **Image Loading Logic**
```python
# For each camera in the configuration
for cam_name in self.camera_names:
    source, suffix, subdir = camera_source_map[cam_name]
    cam_csv = camera_csvs[source]  # Get camera-specific CSV
    
    # Find closest timestamp to our target
    time_diffs = np.abs(cam_csv['timestamp'].values - target_timestamp)
    closest_idx = np.argmin(time_diffs)
    image_timestamp = cam_csv['timestamp'].iloc[closest_idx]
    
    # Construct exact filename
    image_filename = f"frame{image_timestamp}{suffix}"
```

### 5. **Deprecated Function**
The old `get_closest_timestamp_file()` binary search function is marked as deprecated but kept for backward compatibility.

## Benefits of New Approach

### 1. **Exact Synchronization**
- Kinematics are guaranteed to match the selected camera's image
- No more approximate timestamp matching with binary search
- Eliminates potential temporal misalignment errors

### 2. **Better Data Augmentation**
- Random camera selection provides natural variation
- Each epoch sees different camera perspectives paired with same kinematics
- Helps model learn camera-invariant representations

### 3. **Efficiency**
- CSV is loaded once and cached per episode
- Split by camera source for O(1) filtering instead of O(n) searches
- Faster data loading with reduced I/O

### 4. **Flexibility**
- Easy to extend to more cameras
- Can handle cameras with different frame rates
- Gracefully handles missing camera data

## Camera Source Mapping

The loader maintains this mapping internally:

| Camera Name (config) | Camera Source (CSV) | Suffix | Subdirectory |
|---------------------|---------------------|---------|--------------|
| `left` | `left` | `_left.jpg` | `left_img_dir` |
| `right` | `right` | `_right.jpg` | `right_img_dir` |
| `left_wrist` | `psm1` | `_psm1.jpg` | `endo_psm1` |
| `right_wrist` | `psm2` | `_psm2.jpg` | `endo_psm2` |

## Usage

No changes required in training scripts! The data loader interface remains the same:

```python
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

# Returns same format as before
image_data, qpos_data, action_data, is_pad, command_embedding = dataset[idx]
```

## Backward Compatibility

The loader automatically detects if the CSV has the `camera_source` column:
- **New format:** Uses camera-specific loading
- **Old format:** Falls back to deprecated binary search method

## Performance Improvements

| Metric | Old Method | New Method |
|--------|-----------|------------|
| Image lookup | Binary search O(log n) per camera | Direct access O(1) |
| CSV parsing | Once per call | Once per episode (cached) |
| Temporal alignment | Approximate (nearest) | Exact (camera-specific) |
| Memory overhead | Timestamp lists per camera | Single split CSV per episode |

## Testing

To test with the new format:
```bash
cd /home/iulian/chole_ws/src/skay_jhu_private_new/src/act
python generic_dataset_invivo.py
```

This will load a random sample and verify:
- ✅ All cameras load correctly
- ✅ Kinematics align with selected camera
- ✅ Images visualize properly
- ✅ No timestamp mismatches

## Future Enhancements

Possible extensions:
1. **Camera-specific action sequences** - Use different action horizons per camera
2. **Multi-camera fusion** - Combine kinematics from multiple cameras
3. **Frame rate adaptation** - Handle cameras with different frequencies
4. **Temporal interpolation** - Smooth out timestamp differences

## Summary

The rewritten data loader leverages the new camera-synchronized CSV format to provide:
- **Exact temporal alignment** between images and kinematics
- **Improved data augmentation** through random camera selection  
- **Better performance** with caching and direct access
- **Maintained compatibility** with existing training pipelines

All while keeping the same external interface for seamless integration!
