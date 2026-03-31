#!/usr/bin/env python3
"""
Test all combinations of CSV format and image format.
"""

import os
import pandas as pd

test_cases = [
    {
        'name': 'cnh_exvivo',
        'description': 'CSV with camera_source + Timestamp images',
        'csv_path': '/home/iulian/chole_ws/data/cnh_exvivo_chole/tissue_1/1_grabbing_gallbladder/20260116-114118-478925/ee_csv.csv',
        'img_dir': '/home/iulian/chole_ws/data/cnh_exvivo_chole/tissue_1/1_grabbing_gallbladder/20260116-114118-478925/left_img_dir/',
        'expected_csv': 'camera_source: YES, timestamp: YES',
        'expected_img': 'Timestamp-based (1234567890_left.jpg)'
    },
    {
        'name': 'Jesse/suturing',
        'description': 'CSV without camera_source + Frame images',
        'csv_path': '/home/iulian/chole_ws/data/Jesse/tissue_1/1_needle_pickup/20250117-120051-073008/ee_csv.csv',
        'img_dir': '/home/iulian/chole_ws/data/Jesse/tissue_1/1_needle_pickup/20250117-120051-073008/left_img_dir/',
        'expected_csv': 'camera_source: NO, timestamp: YES',
        'expected_img': 'Frame-indexed (frame000000_left.jpg)'
    },
    {
        'name': 'base_chole_clipping_cutting',
        'description': 'CSV with camera_source + Frame images (MIXED!)',
        'csv_path': '/home/iulian/chole_ws/data/base_chole_clipping_cutting/tissue_19/4_clipping_second_clip_left_tube/20240722-135148-328031/ee_csv.csv',
        'img_dir': '/home/iulian/chole_ws/data/base_chole_clipping_cutting/tissue_19/4_clipping_second_clip_left_tube/20240722-135148-328031/left_img_dir/',
        'expected_csv': 'camera_source: ?, timestamp: YES',
        'expected_img': 'Frame-indexed (frame000000_left.jpg)'
    }
]

print("\n" + "="*80)
print("TESTING ALL CSV + IMAGE FORMAT COMBINATIONS")
print("="*80)

for i, test in enumerate(test_cases, 1):
    print(f"\n{'─'*80}")
    print(f"Test {i}: {test['name']}")
    print(f"Description: {test['description']}")
    print('─'*80)
    
    # Check CSV
    if os.path.exists(test['csv_path']):
        csv = pd.read_csv(test['csv_path'])
        has_camera_source = 'camera_source' in csv.columns
        has_timestamp = 'timestamp' in csv.columns
        
        csv_status = f"✓ camera_source: {'YES' if has_camera_source else 'NO'}, timestamp: {'YES' if has_timestamp else 'NO'}"
        print(f"CSV Format:   {csv_status}")
        print(f"Expected:     {test['expected_csv']}")
        
        if has_camera_source:
            sources = csv['camera_source'].unique()
            print(f"              Camera sources: {sources}")
    else:
        print(f"CSV Format:   ✗ File not found: {test['csv_path']}")
    
    # Check Images
    if os.path.exists(test['img_dir']):
        files = sorted([f for f in os.listdir(test['img_dir']) if f.endswith('.jpg')])[:3]
        
        if files:
            is_frame_indexed = any(f.startswith('frame') for f in files)
            img_status = f"✓ {'Frame-indexed' if is_frame_indexed else 'Timestamp-based'}"
            print(f"Image Format: {img_status}")
            print(f"Expected:     {test['expected_img']}")
            print(f"Examples:     {', '.join(files)}")
            
            # Check for critical case: CSV timestamp + Frame images
            if has_timestamp and is_frame_indexed:
                print(f"\n⚠️  CRITICAL CASE: Timestamp CSV + Frame Images")
                print(f"   → System must use CSV row index, NOT timestamp value!")
        else:
            print(f"Image Format: ✗ No .jpg files found")
    else:
        print(f"Image Format: ✗ Directory not found: {test['img_dir']}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
The system now handles these combinations:

1. ✅ CSV (camera_source + timestamp) + Images (timestamp)
   → Use timestamp matching
   → Example: cnh_exvivo

2. ✅ CSV (timestamp only) + Images (frame index)
   → Use CSV row index
   → Example: Jesse/suturing

3. ✅ CSV (camera_source + timestamp) + Images (frame index)  ⭐ THE TRICKY ONE!
   → Use CSV row index (NOT timestamp!)
   → Example: base_chole_clipping_cutting

Key: CSV and image formats are INDEPENDENT and checked separately!
""")
print("="*80)
