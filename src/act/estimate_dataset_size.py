#!/usr/bin/env python3
"""
Estimate dataset size and required training epochs for CNH ex-vivo chole data.

Expected layout:
  <base_dir>/tissue_*/<subtask_dir>/<episode_dir>/left_img_dir/*.jpg

Example:
  /home/iulian/chole_ws/data/cnh_exvivo_chole/tissue_1/
    1_grabbing_gallbladder/20260116-114118-478925/left_img_dir/*.jpg
    ...
    17_go_back_from_the_cut_right_tube/...
"""

import argparse
import csv
import math
import os
from collections import defaultdict

from tqdm import tqdm


def count_frames_in_episode(episode_path):
    left_img_dir = os.path.join(episode_path, "left_img_dir")

    if not os.path.isdir(left_img_dir):
        return 0

    try:
        return sum(1 for f in os.listdir(left_img_dir) if f.lower().endswith(".jpg"))
    except Exception as exc:
        print(f"Error reading {left_img_dir}: {exc}")
        return 0


def safe_avg(total, count):
    return total / count if count > 0 else 0


def estimate_required_epochs(avg_frames_per_episode, target_coverage=0.95):
    if avg_frames_per_episode <= 0:
        return 0

    if target_coverage >= 0.99:
        return int(avg_frames_per_episode * math.log(max(2, avg_frames_per_episode)))

    return int(avg_frames_per_episode * math.log(1.0 / (1.0 - target_coverage)))


def discover_dataset(base_dir):
    tissue_dirs = []
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and name.startswith("tissue_"):
            tissue_dirs.append(path)

    if not tissue_dirs:
        print(f"⚠️  No tissue_* folders found under: {base_dir}")

    return tissue_dirs


def analyze_dataset(base_dir, fps):
    tissue_dirs = discover_dataset(base_dir)
    episode_records = []

    for tissue_path in tissue_dirs:
        tissue_name = os.path.basename(tissue_path)

        subtask_dirs = []
        for name in sorted(os.listdir(tissue_path)):
            path = os.path.join(tissue_path, name)
            if os.path.isdir(path) and name[0:1].isdigit():
                subtask_dirs.append(path)

        for subtask_path in subtask_dirs:
            subtask_name = os.path.basename(subtask_path)
            episode_dirs = []

            for name in sorted(os.listdir(subtask_path)):
                path = os.path.join(subtask_path, name)
                if not os.path.isdir(path):
                    continue
                if name.startswith("."):
                    continue
                episode_dirs.append(path)

            for episode_path in tqdm(
                episode_dirs,
                desc=f"{tissue_name}/{subtask_name}",
                leave=False,
            ):
                episode_name = os.path.basename(episode_path)
                frame_count = count_frames_in_episode(episode_path)

                episode_records.append(
                    {
                        "tissue": tissue_name,
                        "subtask": subtask_name,
                        "episode": episode_name,
                        "frames": frame_count,
                    }
                )

    by_tissue = defaultdict(lambda: {"episodes": 0, "frames": 0})
    by_subtask = defaultdict(lambda: {"episodes": 0, "frames": 0})

    for rec in episode_records:
        by_tissue[rec["tissue"]]["episodes"] += 1
        by_tissue[rec["tissue"]]["frames"] += rec["frames"]

        by_subtask[rec["subtask"]]["episodes"] += 1
        by_subtask[rec["subtask"]]["frames"] += rec["frames"]

    total_episodes = len(episode_records)
    total_frames = sum(r["frames"] for r in episode_records)
    total_seconds = total_frames / fps if fps > 0 else 0
    total_hours = total_seconds / 3600.0

    return {
        "episode_records": episode_records,
        "by_tissue": by_tissue,
        "by_subtask": by_subtask,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_seconds": total_seconds,
        "total_hours": total_hours,
    }


def print_report(stats, fps):
    total_episodes = stats["total_episodes"]
    total_frames = stats["total_frames"]
    total_hours = stats["total_hours"]
    avg_frames_per_episode = safe_avg(total_frames, total_episodes)

    print("=" * 90)
    print("CNH EX-VIVO CHOLE DATASET SIZE ESTIMATION")
    print("=" * 90)

    print("\nOVERALL SUMMARY")
    print("-" * 90)
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {total_frames:,}")
    print(f"Total duration (@ {fps:.1f} Hz): {total_hours:.2f} hours ({total_hours * 60:.1f} minutes)")
    print(f"Average frames per episode: {avg_frames_per_episode:.1f}")

    print("\nBY TISSUE")
    print("-" * 90)
    for tissue_name in sorted(stats["by_tissue"].keys()):
        tissue_stats = stats["by_tissue"][tissue_name]
        tissue_hours = tissue_stats["frames"] / fps / 3600.0 if fps > 0 else 0
        print(
            f"{tissue_name}: episodes={tissue_stats['episodes']}, "
            f"frames={tissue_stats['frames']:,}, hours={tissue_hours:.2f}"
        )

    print("\nBY SUBTASK")
    print("-" * 90)
    for subtask_name in sorted(stats["by_subtask"].keys(), key=lambda s: int(s.split("_", 1)[0])):
        subtask_stats = stats["by_subtask"][subtask_name]
        subtask_hours = subtask_stats["frames"] / fps / 3600.0 if fps > 0 else 0
        print(
            f"{subtask_name}: episodes={subtask_stats['episodes']}, "
            f"frames={subtask_stats['frames']:,}, hours={subtask_hours:.2f}"
        )

    print("\nTRAINING EPOCH RECOMMENDATIONS")
    print("-" * 90)
    print("Assuming each epoch samples 1 random frame per episode:")
    print(f"  • Average frames/episode: {avg_frames_per_episode:.1f}")
    for coverage, label in [
        (0.63, "63% coverage (1 - 1/e)"),
        (0.86, "86% coverage"),
        (0.95, "95% coverage"),
        (0.99, "99% coverage"),
    ]:
        epochs = estimate_required_epochs(avg_frames_per_episode, coverage)
        print(f"  • {label}: ~{epochs} epochs")

    if total_frames > 0:
        print("\nCoverage ratio per epoch:")
        print(f"  • {(total_episodes / total_frames) * 100:.2f}% of all frames seen per epoch")
    else:
        print("\n⚠️  No frames found. Check directory structure and image extensions.")


def write_csv(stats, fps, csv_path):
    total_episodes = stats["total_episodes"]
    total_frames = stats["total_frames"]
    total_hours = stats["total_hours"]

    with open(csv_path, "w", newline="") as csv_file:
        fieldnames = [
            "level",
            "name",
            "episodes",
            "frames",
            "hours",
            "avg_frames_per_episode",
            "avg_duration_per_episode_sec",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for tissue_name in sorted(stats["by_tissue"].keys()):
            tissue_stats = stats["by_tissue"][tissue_name]
            writer.writerow(
                {
                    "level": "tissue",
                    "name": tissue_name,
                    "episodes": tissue_stats["episodes"],
                    "frames": tissue_stats["frames"],
                    "hours": f"{(tissue_stats['frames'] / fps / 3600.0):.3f}" if fps > 0 else "0.000",
                    "avg_frames_per_episode": f"{safe_avg(tissue_stats['frames'], tissue_stats['episodes']):.1f}",
                    "avg_duration_per_episode_sec": f"{safe_avg(tissue_stats['frames'], tissue_stats['episodes']) / fps:.1f}" if fps > 0 else "0.0",
                }
            )

        for subtask_name in sorted(stats["by_subtask"].keys(), key=lambda s: int(s.split("_", 1)[0])):
            subtask_stats = stats["by_subtask"][subtask_name]
            writer.writerow(
                {
                    "level": "subtask",
                    "name": subtask_name,
                    "episodes": subtask_stats["episodes"],
                    "frames": subtask_stats["frames"],
                    "hours": f"{(subtask_stats['frames'] / fps / 3600.0):.3f}" if fps > 0 else "0.000",
                    "avg_frames_per_episode": f"{safe_avg(subtask_stats['frames'], subtask_stats['episodes']):.1f}",
                    "avg_duration_per_episode_sec": f"{safe_avg(subtask_stats['frames'], subtask_stats['episodes']) / fps:.1f}" if fps > 0 else "0.0",
                }
            )

        writer.writerow(
            {
                "level": "total",
                "name": "ALL",
                "episodes": total_episodes,
                "frames": total_frames,
                "hours": f"{total_hours:.3f}",
                "avg_frames_per_episode": f"{safe_avg(total_frames, total_episodes):.1f}",
                "avg_duration_per_episode_sec": f"{safe_avg(total_frames, total_episodes) / fps:.1f}" if fps > 0 else "0.0",
            }
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate dataset size for CNH ex-vivo chole dataset.")
    parser.add_argument(
        "--base-dir",
        default="/home/iulian/chole_ws/data/cnh_exvivo_chole",
        help="Dataset root containing tissue_* folders.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate used to estimate duration.",
    )
    parser.add_argument(
        "--csv-path",
        default=os.path.join(os.path.dirname(__file__), "dataset_size_estimation.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.base_dir):
        raise FileNotFoundError(f"Base directory not found: {args.base_dir}")

    stats = analyze_dataset(args.base_dir, args.fps)
    print_report(stats, args.fps)
    write_csv(stats, args.fps, args.csv_path)
    print(f"\n✓ Results saved to: {args.csv_path}")


if __name__ == "__main__":
    main()
