#!/usr/bin/env python3
"""
Script to count total frames in endo_psm2 folders across all episodes in all tissue folders.
"""

import os
from pathlib import Path


def count_endo_psm2_frames(base_dir):
    """
    Count total frames in endo_psm2 folders across all tissue folders and episodes.
    
    Args:
        base_dir: Base directory containing tissue folders
    
    Returns:
        Dictionary with counts per tissue and total count
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist!")
        return None
    
    total_frames = 0
    tissue_counts = {}
    episode_details = []
    
    # Find all tissue folders
    tissue_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('tissue')])
    
    if not tissue_folders:
        print(f"No tissue folders found in {base_dir}")
        return None
    
    print(f"Found {len(tissue_folders)} tissue folder(s)")
    print("=" * 80)
    
    for tissue_folder in tissue_folders:
        tissue_name = tissue_folder.name
        tissue_frame_count = 0
        tissue_episode_count = 0
        
        print(f"\nProcessing {tissue_name}...")
        
        # Find all episode/phase folders within tissue folder
        episode_folders = sorted([d for d in tissue_folder.iterdir() if d.is_dir()])
        
        for episode_folder in episode_folders:
            episode_name = episode_folder.name
            
            # Find all demo folders (timestamp folders) within episode
            demo_folders = sorted([d for d in episode_folder.iterdir() if d.is_dir()])
            
            for demo_folder in demo_folders:
                demo_name = demo_folder.name
                endo_psm2_path = demo_folder / 'endo_psm2'
                
                if endo_psm2_path.exists() and endo_psm2_path.is_dir():
                    # Count files in endo_psm2 folder
                    frame_files = [f for f in endo_psm2_path.iterdir() if f.is_file()]
                    frame_count = len(frame_files)
                    
                    total_frames += frame_count
                    tissue_frame_count += frame_count
                    tissue_episode_count += 1
                    
                    episode_details.append({
                        'tissue': tissue_name,
                        'episode': episode_name,
                        'demo': demo_name,
                        'frames': frame_count
                    })
                    
                    print(f"  {episode_name}/{demo_name}: {frame_count} frames")
        
        tissue_counts[tissue_name] = {
            'episodes': tissue_episode_count,
            'frames': tissue_frame_count
        }
        
        print(f"\n{tissue_name} Summary: {tissue_episode_count} episodes, {tissue_frame_count} frames")
    
    return {
        'total_frames': total_frames,
        'tissue_counts': tissue_counts,
        'episode_details': episode_details
    }


def main():
    base_dir = "/home/iulian/chole_ws/data/cnh_exvivo_chole"
    frame_rate = 30  # Hz
    
    print("Counting endo_psm2 frames...")
    print(f"Base directory: {base_dir}")
    print(f"Frame rate: {frame_rate} Hz")
    print("=" * 80)
    
    results = count_endo_psm2_frames(base_dir)
    
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        for tissue_name, counts in sorted(results['tissue_counts'].items()):
            print(f"{tissue_name}: {counts['episodes']} episodes, {counts['frames']} frames")
        
        print("=" * 80)
        print(f"TOTAL FRAMES: {results['total_frames']}")
        print(f"TOTAL EPISODES: {sum(c['episodes'] for c in results['tissue_counts'].values())}")
        
        # Calculate time duration
        total_seconds = results['total_frames'] / frame_rate
        total_minutes = total_seconds / 60
        total_hours = total_minutes / 60
        
        print(f"\nTIME DURATION (at {frame_rate} Hz):")
        print(f"  Total seconds: {total_seconds:.2f}")
        print(f"  Total minutes: {total_minutes:.2f}")
        print(f"  Total hours: {total_hours:.4f}")
        print("=" * 80)
        
        # Calculate average frames per episode
        total_episodes = sum(c['episodes'] for c in results['tissue_counts'].values())
        if total_episodes > 0:
            avg_frames = results['total_frames'] / total_episodes
            avg_seconds = avg_frames / frame_rate
            print(f"Average frames per episode: {avg_frames:.2f}")
            print(f"Average duration per episode: {avg_seconds:.2f} seconds ({avg_seconds/60:.2f} minutes)")
        
        # Optionally save detailed results to a file
        save_details = input("\nSave detailed results to file? (y/n): ").strip().lower()
        if save_details == 'y':
            output_file = Path(__file__).parent / "endo_psm2_frame_counts.txt"
            with open(output_file, 'w') as f:
                f.write("Detailed Frame Counts by Episode\n")
                f.write("=" * 80 + "\n\n")
                
                for detail in results['episode_details']:
                    f.write(f"{detail['tissue']}/{detail['episode']}/{detail['demo']}: {detail['frames']} frames\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("SUMMARY\n")
                f.write("=" * 80 + "\n")
                
                for tissue_name, counts in sorted(results['tissue_counts'].items()):
                    f.write(f"{tissue_name}: {counts['episodes']} episodes, {counts['frames']} frames\n")
                
                f.write("=" * 80 + "\n")
                f.write(f"TOTAL FRAMES: {results['total_frames']}\n")
                f.write(f"TOTAL EPISODES: {total_episodes}\n")
                f.write(f"\nTIME DURATION (at {frame_rate} Hz):\n")
                f.write(f"  Total seconds: {total_seconds:.2f}\n")
                f.write(f"  Total minutes: {total_minutes:.2f}\n")
                f.write(f"  Total hours: {total_hours:.4f}\n")
                f.write(f"\nAverage frames per episode: {avg_frames:.2f}\n")
                f.write(f"Average duration per episode: {avg_seconds:.2f} seconds ({avg_seconds/60:.2f} minutes)\n")
                f.write("=" * 80 + "\n")
            
            print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
