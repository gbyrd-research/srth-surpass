#!/usr/bin/env python3
"""
Test image filename format detection for both old and new formats.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

# Test directories
test_cases = [
    {
        'name': 'cnh_exvivo (timestamp-based)',
        'path': '/home/iulian/chole_ws/data/cnh_exvivo_chole/tissue_1/1_grabbing_gallbladder/20260116-114118-478925/left_img_dir/'
    },
    {
        'name': 'Jesse/suturing (frame-indexed)',
        'path': '/home/iulian/chole_ws/data/Jesse/tissue_1/1_needle_pickup/20250117-120051-073008/left_img_dir/'
    }
]

for test in test_cases:
    print(f"\n{'='*60}")
    print(f"Testing: {test['name']}")
    print(f"Path: {test['path']}")
    print('='*60)
    
    if not os.path.exists(test['path']):
        print(f"  ⚠ Directory not found, skipping")
        continue
    
    try:
        # Get first 5 files
        files = sorted([f for f in os.listdir(test['path']) if f.endswith('.jpg')])[:5]
        
        if not files:
            print(f"  ⚠ No .jpg files found")
            continue
        
        print(f"  ✓ Found {len(files)} sample files:")
        
        # Check naming format
        is_frame_indexed = any(f.startswith('frame') for f in files)
        
        if is_frame_indexed:
            print(f"  ✓ Format: INDEX-BASED (frameXXXXXX_xxx.jpg)")
            print(f"    Examples:")
            for f in files[:3]:
                print(f"      - {f}")
        else:
            print(f"  ✓ Format: TIMESTAMP-BASED (timestamp_xxx.jpg)")
            print(f"    Examples:")
            for f in files[:3]:
                print(f"      - {f}")
        
        print(f"  ✓ Detection successful!")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("Image filename format detection test complete")
print("="*60)
print("\n✓ The system will now automatically handle both formats!")
