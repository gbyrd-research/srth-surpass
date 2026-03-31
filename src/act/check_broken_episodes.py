#!/usr/bin/env python3
"""
Check for broken episodes with timestamp misalignment or missing images.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def check_episode(episode_path):
    """
    Check a single episode for issues.
    Returns dict with diagnostic info.
    """
    issues = []
    episode_name = os.path.basename(episode_path)
    
    # Check if CSV exists
    csv_path = os.path.join(episode_path, "ee_csv.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(episode_path, "ee_estimate.csv")
        if not os.path.exists(csv_path):
            return {
                'episode': episode_name,
                'status': 'BROKEN',
                'issues': ['No CSV file found'],
                'image_counts': {},
                'timestamp_range': None
            }
    
    # Load CSV
    try:
        csv = pd.read_csv(csv_path)
    except Exception as e:
        return {
            'episode': episode_name,
            'status': 'BROKEN',
            'issues': [f'Failed to load CSV: {e}'],
            'image_counts': {},
            'timestamp_range': None
        }
    
    # Get CSV timestamp range
    has_timestamp = 'timestamp' in csv.columns
    if has_timestamp:
        csv_min_ts = csv['timestamp'].min()
        csv_max_ts = csv['timestamp'].max()
        csv_timestamp_range = (csv_min_ts, csv_max_ts)
    else:
        csv_timestamp_range = None
    
    # Check image directories
    camera_dirs = {
        'endo_psm1': '_psm1.jpg',
        'endo_psm2': '_psm2.jpg',
        'left_img_dir': '_left.jpg',
        'right_img_dir': '_right.jpg'
    }
    
    image_counts = {}
    image_timestamp_ranges = {}
    
    for dir_name, suffix in camera_dirs.items():
        camera_dir = os.path.join(episode_path, dir_name)
        
        if not os.path.exists(camera_dir):
            issues.append(f'{dir_name} directory missing')
            image_counts[dir_name] = 0
            continue
        
        # Count images
        images = [f for f in os.listdir(camera_dir) if f.endswith(suffix)]
        
        # Try alternate suffix (some episodes have swapped naming)
        if not images:
            alternate_suffix = '_psm2.jpg' if suffix == '_psm1.jpg' else '_psm1.jpg' if suffix == '_psm2.jpg' else suffix
            images = [f for f in os.listdir(camera_dir) if f.endswith(alternate_suffix)]
            if images:
                suffix = alternate_suffix
        
        image_counts[dir_name] = len(images)
        
        if len(images) == 0:
            issues.append(f'{dir_name} has no images')
            continue
        
        if len(images) < 10:
            issues.append(f'{dir_name} has only {len(images)} images (likely incomplete recording)')
        
        # Extract timestamps from first and last few images
        try:
            # Sort images
            images.sort()
            
            # Parse timestamps from filenames
            def parse_timestamp(filename):
                # Remove 'frame' prefix if present and suffix
                ts_str = filename.replace('frame', '').replace(suffix, '')
                return int(ts_str)
            
            first_ts = parse_timestamp(images[0])
            last_ts = parse_timestamp(images[-1])
            image_timestamp_ranges[dir_name] = (first_ts, last_ts)
            
            # Check if timestamps look like frame indices (< 1 million) or actual timestamps
            if first_ts < 1_000_000:
                # Frame-indexed, skip timestamp comparison
                pass
            elif has_timestamp:
                # Compare with CSV timestamp range
                # Images should overlap with CSV timestamps
                csv_range_ns = csv_max_ts - csv_min_ts
                img_range_ns = last_ts - first_ts
                
                # Check if image timestamps are completely outside CSV range
                if first_ts > csv_max_ts:
                    time_diff_s = (first_ts - csv_max_ts) / 1e9
                    issues.append(f'{dir_name}: Images start {time_diff_s:.1f}s AFTER CSV ends (recording desync)')
                elif last_ts < csv_min_ts:
                    time_diff_s = (csv_min_ts - last_ts) / 1e9
                    issues.append(f'{dir_name}: Images end {time_diff_s:.1f}s BEFORE CSV starts (recording desync)')
                else:
                    # Check for partial overlap with large offset
                    # Use middle timestamp from CSV
                    csv_mid_ts = csv_min_ts + (csv_max_ts - csv_min_ts) // 2
                    
                    # Find time difference from image range to middle of CSV
                    if csv_mid_ts < first_ts:
                        time_diff_s = (first_ts - csv_mid_ts) / 1e9
                        if time_diff_s > 60:  # More than 1 minute offset
                            issues.append(f'{dir_name}: Images start {time_diff_s:.1f}s after CSV midpoint')
                    elif csv_mid_ts > last_ts:
                        time_diff_s = (csv_mid_ts - last_ts) / 1e9
                        if time_diff_s > 60:
                            issues.append(f'{dir_name}: Images end {time_diff_s:.1f}s before CSV midpoint')
                
        except Exception as e:
            issues.append(f'{dir_name}: Failed to parse timestamps - {e}')
    
    # Determine overall status
    if not issues:
        status = 'OK'
    elif any('desync' in issue or 'only' in issue for issue in issues):
        status = 'BROKEN'
    else:
        status = 'WARNING'
    
    return {
        'episode': episode_name,
        'status': status,
        'issues': issues,
        'image_counts': image_counts,
        'csv_timestamp_range': csv_timestamp_range,
        'image_timestamp_ranges': image_timestamp_ranges
    }

def main():
    base_dir = "/media/iulian/T7 Shield/Wound_Closure/Processing/tissue_1/1_wound_closure"
    
    if not os.path.exists(base_dir):
        print(f"ERROR: Directory not found: {base_dir}")
        return
    
    # Get all episode directories
    episodes = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d)) 
                and not d.startswith('.')]
    
    episodes.sort()
    
    print(f"Checking {len(episodes)} episodes in: {base_dir}\n")
    print("=" * 100)
    
    broken_episodes = []
    
    for episode in episodes:
        episode_path = os.path.join(base_dir, episode)
        result = check_episode(episode_path)
        
        if result['status'] == 'BROKEN':
            broken_episodes.append(result)
            
            # Print broken episode details
            print(f"\n❌ BROKEN: {result['episode']}")
            
            # Print image counts
            if result['image_counts']:
                print(f"  Image counts:")
                for cam, count in result['image_counts'].items():
                    print(f"    {cam}: {count} images")
            
            # Print timestamp info
            if result['csv_timestamp_range']:
                csv_min, csv_max = result['csv_timestamp_range']
                csv_duration_s = (csv_max - csv_min) / 1e9
                print(f"  CSV timestamp range: {csv_min} to {csv_max} ({csv_duration_s:.1f}s duration)")
            
            if result['image_timestamp_ranges']:
                print(f"  Image timestamp ranges:")
                for cam, (img_min, img_max) in result['image_timestamp_ranges'].items():
                    if img_min < 1_000_000:
                        print(f"    {cam}: frame {img_min} to {img_max} (frame-indexed)")
                    else:
                        img_duration_s = (img_max - img_min) / 1e9
                        print(f"    {cam}: {img_min} to {img_max} ({img_duration_s:.1f}s duration)")
            
            # Print issues
            if result['issues']:
                print(f"  Issues:")
                for issue in result['issues']:
                    print(f"    - {issue}")
    
    # Summary
    print("\n" + "=" * 100)
    print(f"\nSUMMARY:")
    print(f"  Total episodes checked: {len(episodes)}")
    print(f"  ❌ Broken episodes: {len(broken_episodes)}")
    print(f"  ✓ OK episodes: {len(episodes) - len(broken_episodes)}")
    
    # Save to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "broken_episodes.csv")
    try:
        import csv
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['episode_name', 'issue_summary', 'endo_psm1_count', 'endo_psm2_count', 
                         'left_img_count', 'right_img_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in broken_episodes:
                issue_summary = '; '.join(result['issues'][:3])  # First 3 issues
                if len(result['issues']) > 3:
                    issue_summary += f" (+{len(result['issues'])-3} more)"
                
                writer.writerow({
                    'episode_name': result['episode'],
                    'issue_summary': issue_summary,
                    'endo_psm1_count': result['image_counts'].get('endo_psm1', 0),
                    'endo_psm2_count': result['image_counts'].get('endo_psm2', 0),
                    'left_img_count': result['image_counts'].get('left_img_dir', 0),
                    'right_img_count': result['image_counts'].get('right_img_dir', 0),
                })
        
        print(f"\n✓ Broken episodes list saved to: {csv_path}")
    except Exception as e:
        print(f"\n⚠️  Failed to save CSV: {e}")
    
    if broken_episodes:
        print(f"\n❌ BROKEN EPISODES LIST:")
        for result in broken_episodes:
            print(f"  - {result['episode']}")

if __name__ == "__main__":
    main()
