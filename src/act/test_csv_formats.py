#!/usr/bin/env python3
"""
Quick test to check if CSV format detection works for both old and new formats.
"""

import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))

# Test with sample CSV paths
test_cases = [
    {
        'name': 'cnh_exvivo (new format with camera_source)',
        'path': '/home/iulian/chole_ws/data/cnh_exvivo_chole/tissue_1/1_grabbing_gallbladder/20260116-114118-478925/ee_csv.csv'
    },
    {
        'name': 'suturing (old format without camera_source)', 
        'path': '/home/iulian/chole_ws/data/Jesse/tissue_1/1_needle_pickup/20250117-120051-073008/ee_csv.csv'
    }
]

for test in test_cases:
    print(f"\nTesting: {test['name']}")
    print(f"Path: {test['path']}")
    
    if not os.path.exists(test['path']):
        print(f"  ⚠ File not found, skipping")
        continue
    
    try:
        csv = pd.read_csv(test['path'])
        print(f"  ✓ CSV loaded successfully")
        print(f"  Columns: {list(csv.columns)[:10]}...")  # Show first 10 columns
        
        if 'camera_source' in csv.columns:
            print(f"  ✓ Has 'camera_source' column (NEW FORMAT)")
            unique_sources = csv['camera_source'].unique()
            print(f"    Camera sources: {unique_sources}")
        else:
            print(f"  ✓ No 'camera_source' column (OLD FORMAT)")
            print(f"    Will use shared timestamps for all cameras")
        
        if 'timestamp' in csv.columns:
            print(f"  ✓ Has 'timestamp' column")
        else:
            print(f"  ✗ No 'timestamp' column (will use row indices)")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("CSV format detection test complete")
print("="*60)
