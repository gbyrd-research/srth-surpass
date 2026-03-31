#!/usr/bin/env python3
"""
Script to PERMANENTLY add a task configuration to constants_dvrk.py.

This script will modify the constants_dvrk.py file directly, adding your
new task configuration to the TASK_CONFIGS dictionary.

WARNING: This modifies the constants_dvrk.py file. Make sure you have a backup
or that the file is under version control.

Usage:
    python add_config_permanently.py <task_name> <json_path> [options]

Example:
    python add_config_permanently.py my_task ./std_mean_invivo.json \\
        --dataset-dir /path/to/data \\
        --tissue-ids 1 2 3 \\
        --num-episodes 1000
"""

import argparse
import sys
import os
from pathlib import Path
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dvrk_scripts.constants_dvrk import (
    create_task_config_from_json,
    load_statistics_from_json
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Permanently add a task configuration to constants_dvrk.py',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('task_name', type=str,
                        help='Name for the new task configuration')
    parser.add_argument('json_path', type=str,
                        help='Path to JSON file containing statistics')
    
    # Dataset configuration
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--tissue-ids', type=int, nargs='+', required=True,
                        help='Tissue sample IDs for training (e.g., 1 2 3)')
    parser.add_argument('--tissue-ids-val', type=int, nargs='+', default=[],
                        help='Tissue sample IDs for validation')
    
    # Episode configuration
    parser.add_argument('--num-episodes', type=int, default=None,
                        help='Number of training episodes')
    parser.add_argument('--num-episodes-val', type=int, default=0,
                        help='Number of validation episodes')
    parser.add_argument('--episode-len', type=int, default=500,
                        help='Length of episodes')
    
    # Task configuration
    parser.add_argument('--phantom', action='store_true',
                        help='Use phantom data')
    parser.add_argument('--use-auto-label', action='store_true',
                        help='Use auto labeling')
    parser.add_argument('--no-qpos', action='store_true',
                        help='No qpos (True for SRT, False for ACT)')
    parser.add_argument('--goal-condition-style', type=str, choices=['map', 'mask', 'dot'],
                        help='Goal conditioning style')
    
    # Camera configuration
    parser.add_argument('--camera-names', type=str, nargs='+',
                        default=['left', 'left_wrist', 'right_wrist'],
                        help='Camera names')
    parser.add_argument('--camera-file-suffixes', type=str, nargs='+',
                        default=['_left.jpg', '_psm1.jpg', '_psm2.jpg'],
                        help='Camera file suffixes')
    
    # Other parameters
    parser.add_argument('--cutting-action-pad-size', type=int, default=0,
                        help='Padding size for cutting actions')
    parser.add_argument('--recovery-ratio', type=float, default=1.0,
                        help='Recovery ratio')
    parser.add_argument('--norm-scheme', type=str, default='std',
                        help='Normalization scheme')
    parser.add_argument('--save-frequency', type=int, default=50,
                        help='Checkpoint save frequency')
    parser.add_argument('--merging-subtasks', action='store_true',
                        help='Merge subtasks')
    
    # Options
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be added without modifying the file')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create a backup of constants_dvrk.py')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug information during processing')
    
    return parser.parse_args()


def format_numpy_array(arr, indent=12):
    """Format numpy array for insertion into Python file"""
    arr_str = np.array2string(arr, separator=', ', max_line_width=120)
    # Add proper indentation
    lines = arr_str.split('\n')
    if len(lines) > 1:
        formatted = lines[0] + '\n'
        for line in lines[1:]:
            formatted += ' ' * indent + line + '\n'
        return formatted.rstrip()
    return arr_str


def generate_config_code(task_name, config, stats):
    """Generate the Python code for the task configuration"""
    
    code = f"\n    '{task_name}': {{\n"
    code += f"        'dataset_dir': \"{config['dataset_dir']}\",\n"
    code += f"        'phantom': {config['phantom']},\n"
    code += f"        'use_auto_label': {config['use_auto_label']},\n"
    
    if 'goal_condition_style' in config and config['goal_condition_style']:
        code += f"        'goal_condition_style': '{config['goal_condition_style']}',\n"
    
    code += f"        'no_qpos': {config['no_qpos']},\n"
    
    if 'num_episodes' in config and config['num_episodes']:
        code += f"        'num_episodes': {config['num_episodes']},\n"
    
    code += f"        'num_episodes_val': {config['num_episodes_val']},\n"
    code += f"        'tissue_samples_ids': {config['tissue_samples_ids']},\n"
    code += f"        'tissue_samples_ids_val': {config['tissue_samples_ids_val']},\n"
    code += f"        'camera_file_suffixes': {config['camera_file_suffixes']},\n"
    code += f"        'episode_len': {config['episode_len']},\n"
    code += f"        'cutting_action_pad_size': {config['cutting_action_pad_size']},\n"
    code += f"        'recovery_ratio': {config['recovery_ratio']},\n"
    
    # Action mode with statistics
    code += f"        'action_mode': ['hybrid',\n"
    code += f"        {{'max_': np.array({format_numpy_array(stats['max'])}), \n\n"
    code += f"        'min_': np.array({format_numpy_array(stats['min'])}),\n\n"
    code += f"        'mean': np.array({format_numpy_array(stats['mean'])}),\n\n"
    code += f"        'std': np.array({format_numpy_array(stats['std'])}) }}],\n"
    
    code += f"        \n"
    code += f"        'norm_scheme': '{config['norm_scheme']}',\n"
    code += f"        'save_frequency': {config['save_frequency']},\n"
    code += f"        'camera_names': {config['camera_names']},\n"
    code += f"        'merging_subtasks': {config['merging_subtasks']},\n"
    code += f"    }},\n"
    
    return code


def add_config_to_file(constants_file, task_name, config_code, no_backup=False, debug=False):
    """Add the configuration to constants_dvrk.py file"""
    
    # Read the file
    with open(constants_file, 'r') as f:
        content = f.read()
    
    if debug:
        print(f"[DEBUG] File size: {len(content)} characters")
    
    # Create backup unless disabled
    if not no_backup:
        backup_file = str(constants_file) + '.backup'
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"✓ Backup created: {backup_file}")
    
    # Find the closing brace of TASK_CONFIGS
    import re
    
    # Look for TASK_CONFIGS = { ... }
    # Find the closing brace of the TASK_CONFIGS dictionary
    # Pattern: last entry's closing brace and comma, then the dict's closing brace
    
    # Try multiple patterns to find where to insert
    patterns = [
        # Pattern 1: },\n}\n\n (most common)
        (r"(        },\n    },\n)(\}\n\n)", 1),
        # Pattern 2: },\n}\n (without extra newline)
        (r"(        },\n    },\n)(\})", 1),
        # Pattern 3: Look for the literal closing of TASK_CONFIGS
        (r"(    },\n)(\}\n\n### Example Usage)", 1),
        # Pattern 4: Just before closing brace
        (r"(        },\n)(\s*\}\n)", 1),
    ]
    
    match = None
    pattern_used = None
    insert_offset = 0
    
    for i, (pattern, offset) in enumerate(patterns):
        match = re.search(pattern, content)
        if match:
            pattern_used = i + 1
            insert_offset = offset
            if debug:
                print(f"[DEBUG] Matched pattern {pattern_used}: {pattern}")
                print(f"[DEBUG] Match position: {match.start()}-{match.end()}")
            break
    
    if not match:
        if debug:
            print("[DEBUG] No regex pattern matched, using brace counting fallback")
        
        # Fallback: find TASK_CONFIGS closing brace manually
        # Find "TASK_CONFIGS = {" and then find its matching closing brace
        task_configs_start = content.find('TASK_CONFIGS = {')
        if task_configs_start == -1:
            print("Error: Could not find 'TASK_CONFIGS = {' in file")
            return False
        
        if debug:
            print(f"[DEBUG] TASK_CONFIGS starts at position {task_configs_start}")
        
        # Find the matching closing brace by counting braces
        brace_count = 0
        in_task_configs = False
        closing_pos = -1
        
        for i in range(task_configs_start, len(content)):
            if content[i] == '{':
                brace_count += 1
                in_task_configs = True
            elif content[i] == '}':
                brace_count -= 1
                if in_task_configs and brace_count == 0:
                    closing_pos = i
                    break
        
        if closing_pos == -1:
            print("Error: Could not find closing brace for TASK_CONFIGS")
            return False
        
        if debug:
            print(f"[DEBUG] TASK_CONFIGS closes at position {closing_pos}")
        
        # Insert before the closing brace
        # Find the last },\n before the closing brace
        search_start = max(0, closing_pos - 200)
        last_entry_end = content.rfind('},\n', search_start, closing_pos)
        
        if last_entry_end == -1:
            print("Error: Could not find insertion point")
            if debug:
                print(f"[DEBUG] Searched from {search_start} to {closing_pos}")
                print(f"[DEBUG] Content snippet: {content[search_start:closing_pos]}")
            return False
        
        if debug:
            print(f"[DEBUG] Last entry ends at position {last_entry_end}")
        
        # Insert after the last entry's closing brace and comma
        insert_pos = last_entry_end + 3  # After "},\n"
        new_content = content[:insert_pos] + config_code + content[insert_pos:]
    else:
        # Insert using the matched pattern
        insert_pos = match.start() + len(match.group(insert_offset))
        if debug:
            print(f"[DEBUG] Inserting at position {insert_pos}")
        new_content = content[:insert_pos] + config_code + content[insert_pos:]
    
    # Write back
    with open(constants_file, 'w') as f:
        f.write(new_content)
    
    if debug:
        print(f"[DEBUG] New file size: {len(new_content)} characters")
    
    return True


def main():
    args = parse_args()
    
    # Validate JSON file exists
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found: {args.json_path}")
        sys.exit(1)
    
    # Load statistics
    print(f"Loading statistics from: {args.json_path}")
    stats = load_statistics_from_json(args.json_path)
    
    # Create config
    print(f"Creating task configuration: '{args.task_name}'")
    config = create_task_config_from_json(
        task_name=args.task_name,
        json_path=args.json_path,
        dataset_dir=args.dataset_dir,
        tissue_ids=args.tissue_ids,
        tissue_ids_val=args.tissue_ids_val,
        num_episodes=args.num_episodes,
        num_episodes_val=args.num_episodes_val,
        phantom=args.phantom,
        use_auto_label=args.use_auto_label,
        goal_condition_style=args.goal_condition_style,
        no_qpos=args.no_qpos,
        camera_file_suffixes=args.camera_file_suffixes,
        episode_len=args.episode_len,
        cutting_action_pad_size=args.cutting_action_pad_size,
        recovery_ratio=args.recovery_ratio,
        norm_scheme=args.norm_scheme,
        save_frequency=args.save_frequency,
        camera_names=args.camera_names,
        merging_subtasks=args.merging_subtasks
    )
    
    # Generate code
    config_code = generate_config_code(args.task_name, config, stats)
    
    # Show what will be added
    print("\n" + "=" * 80)
    print("GENERATED CONFIGURATION CODE:")
    print("=" * 80)
    print(config_code)
    print("=" * 80)
    
    if args.dry_run:
        print("\n[DRY RUN] No changes made to constants_dvrk.py")
        sys.exit(0)
    
    # Confirm with user
    response = input("\nAdd this configuration to constants_dvrk.py? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        sys.exit(0)
    
    # Find constants_dvrk.py
    constants_file = Path(__file__).parent / 'constants_dvrk.py'
    
    if not constants_file.exists():
        print(f"Error: constants_dvrk.py not found at {constants_file}")
        sys.exit(1)
    
    # Add to file
    print(f"\nAdding configuration to: {constants_file}")
    
    if add_config_to_file(constants_file, args.task_name, config_code, args.no_backup, args.debug):
        print("\n" + "=" * 80)
        print("✓ SUCCESS! Configuration added to constants_dvrk.py")
        print("=" * 80)
        print(f"\nYou can now use it in your code:")
        print(f"    from dvrk_scripts.constants_dvrk import TASK_CONFIGS")
        print(f"    config = TASK_CONFIGS['{args.task_name}']")
    else:
        print("\n✗ Failed to add configuration")
        if not args.debug:
            print("\nTip: Try running with --debug flag for more information:")
            print(f"    python {sys.argv[0]} ... --debug")
        sys.exit(1)


if __name__ == "__main__":
    main()
