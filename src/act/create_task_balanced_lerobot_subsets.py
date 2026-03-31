#!/usr/bin/env python3
"""Create reproducible LeRobot subsets for data-efficiency ablations.

This utility is designed for datasets whose episodes each belong to a single task
label in `meta/tasks.jsonl` and where a higher-level source split such as
`perfect` versus `recovery` should be preserved proportionally in sampled
subsets.

Sampling behavior:
1. Episodes are grouped by `(source split, task)`.
2. Each bucket is shuffled independently with a deterministic random seed.
3. Requested subset percentages first preserve the relative size of the source
   splits (for example `perfect` versus `recovery`).
4. Within each source split, task counts are kept proportional to the original
   task frequencies by taking per-task prefixes from the shuffled buckets.
5. Smaller subsets are nested inside larger subsets for a fixed seed.

The script reuses `sample_lerobot_dataset.py` for LeRobot-safe episode copying,
video copying, and metadata regeneration, then rewrites the sampled dataset's
split metadata so `perfect` and `recovery` remain valid while `train` spans the
entire sampled dataset.

Example:
    /lustre/fsw/portfolios/healthcareeng/users/nigeln/cosmos-predict1/miniconda/envs/6_lerobot/bin/python \
        examples/dVRK/utils/create_task_balanced_lerobot_subsets.py \
        --input /lustre/fsw/portfolios/healthcareeng/users/nigeln/cache/huggingface/lerobot/nigeln/suturebot_points_t4_to_t10 \
        --output-root /lustre/fsw/portfolios/healthcareeng/users/nigeln/cache/huggingface/lerobot/nigeln/subsets \
        --percentages 33 66 100 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Any

try:
    from sample_lerobot_dataset import load_episodes, load_info, load_tasks, sample_dataset
except ImportError:  # pragma: no cover - fallback for module-style execution
    from examples.dVRK.utils.sample_lerobot_dataset import (  # type: ignore
        load_episodes,
        load_info,
        load_tasks,
        sample_dataset,
    )


DEFAULT_PERCENTAGES = [33.0, 66.0, 100.0]
DEFAULT_STRATIFY_SPLITS = ["perfect", "recovery"]
OPTIONAL_META_FILES_TO_COPY = ["percentile_stats.json", "relative_stats.json"]


def _normalize_percentages(raw_percentages: list[float]) -> list[float]:
    """Normalize percentage inputs to fractions in `(0, 1]`.

    Args:
        raw_percentages: Percentage-like values from the CLI. Values greater than
            `1` are interpreted as percentages (for example `33 -> 0.33`), while
            values in `(0, 1]` are treated as already normalized fractions.

    Returns:
        Normalized fractions in the same order they were provided, with duplicate
        values removed after normalization.

    Raises:
        ValueError: If any normalized percentage falls outside `(0, 1]`.
    """

    fractions: list[float] = []
    seen: set[float] = set()

    for raw_value in raw_percentages:
        fraction = raw_value / 100.0 if raw_value > 1 else raw_value
        if fraction <= 0 or fraction > 1:
            raise ValueError(
                f"Invalid percentage '{raw_value}'. Use values in (0, 1] or (0, 100]."
            )

        rounded_fraction = round(fraction, 8)
        if rounded_fraction not in seen:
            seen.add(rounded_fraction)
            fractions.append(fraction)

    return fractions


def _load_split_ranges(info: dict[str, Any], split_names: list[str]) -> dict[str, tuple[int, int]]:
    """Load split ranges from `meta/info.json`.

    Args:
        info: Parsed dataset `meta/info.json`.
        split_names: Source split names that should be preserved proportionally in
            the sampled subsets.

    Returns:
        Mapping from split name to `(start_episode_index, end_episode_index)`.

    Raises:
        ValueError: If any requested split is missing or malformed.
    """

    split_ranges: dict[str, tuple[int, int]] = {}
    for split_name in split_names:
        split_range = info.get("splits", {}).get(split_name)
        if split_range is None:
            raise ValueError(
                f"Source dataset does not define split '{split_name}' in meta/info.json."
            )

        try:
            start_str, end_str = split_range.split(":")
            start_idx = int(start_str)
            end_idx = int(end_str)
        except ValueError as exc:
            raise ValueError(
                f"Split '{split_name}' has malformed range '{split_range}'. Expected 'start:end'."
            ) from exc

        split_ranges[split_name] = (start_idx, end_idx)

    return split_ranges


def _episode_task_name(episode: dict[str, Any], episode_index: int) -> str:
    """Extract the single task label associated with an episode.

    Args:
        episode: Episode metadata entry from `meta/episodes.jsonl`.
        episode_index: Episode index used for error reporting.

    Returns:
        The episode's task label.

    Raises:
        ValueError: If the episode does not contain exactly one task label.
    """

    tasks = episode.get("tasks", [])
    if len(tasks) != 1:
        raise ValueError(
            f"Episode {episode_index} has tasks={tasks}. Expected exactly one task label per episode."
        )
    return tasks[0]


def _collect_episode_buckets(
    episodes: list[dict[str, Any]],
    task_order: list[str],
    split_ranges: dict[str, tuple[int, int]],
) -> dict[str, dict[str, list[int]]]:
    """Group episode indices by source split and task label.

    Args:
        episodes: Parsed `meta/episodes.jsonl` entries.
        task_order: Ordered task labels from `meta/tasks.jsonl`.
        split_ranges: Source split ranges to preserve proportionally.

    Returns:
        Nested mapping `split_name -> task_name -> [episode_indices]`.

    Raises:
        ValueError: If the requested source splits overlap, skip episodes, or an
            episode uses an unknown task label.
    """

    ordered_split_names = sorted(split_ranges, key=lambda name: split_ranges[name][0])
    split_buckets = {
        split_name: {task_name: [] for task_name in task_order}
        for split_name in ordered_split_names
    }

    for episode_index, episode in enumerate(episodes):
        task_name = _episode_task_name(episode, episode_index)
        if task_name not in task_order:
            raise ValueError(
                f"Episode {episode_index} uses task '{task_name}', which is missing from meta/tasks.jsonl."
            )

        matching_splits = [
            split_name
            for split_name, (start_idx, end_idx) in split_ranges.items()
            if start_idx <= episode_index < end_idx
        ]
        if len(matching_splits) != 1:
            raise ValueError(
                f"Episode {episode_index} matched source splits {matching_splits}. "
                "Requested stratification splits must be disjoint and cover the dataset exactly once."
            )

        split_buckets[matching_splits[0]][task_name].append(episode_index)

    return split_buckets


def _shuffle_task_buckets(
    split_task_buckets: dict[str, list[int]],
    task_order: list[str],
    rng: random.Random,
) -> dict[str, list[int]]:
    """Shuffle per-task episode buckets for one source split.

    Args:
        split_task_buckets: Mapping from task name to all episode indices for one
            source split.
        task_order: Ordered task labels from `meta/tasks.jsonl`.
        rng: Deterministic random number generator used to shuffle each task bucket.

    Returns:
        Copy of the input buckets with each task bucket shuffled in place.
    """

    shuffled_buckets = {
        task_name: episode_indices.copy()
        for task_name, episode_indices in split_task_buckets.items()
    }
    for task_name in task_order:
        rng.shuffle(shuffled_buckets[task_name])

    return shuffled_buckets


def _apportion_targets(
    total_target: int,
    group_sizes: dict[str, int],
    group_order: list[str],
) -> dict[str, int]:
    """Allocate an exact total across groups while preserving source proportions.

    Args:
        total_target: Total number of episodes to sample across all groups.
        group_sizes: Available episode count per group.
        group_order: Stable ordering used for deterministic tie-breaking.

    Returns:
        Mapping from group name to allocated episode count.

    Raises:
        ValueError: If `total_target` is larger than the total available episodes.
    """

    total_available = sum(group_sizes.values())
    if total_target > total_available:
        raise ValueError(
            f"Requested {total_target} episodes but only {total_available} are available."
        )

    if total_available == 0:
        return {group_name: 0 for group_name in group_order}

    quotas = {
        group_name: total_target * group_sizes[group_name] / total_available
        for group_name in group_order
    }
    allocations = {
        group_name: min(group_sizes[group_name], math.floor(quotas[group_name]))
        for group_name in group_order
    }

    remainder = total_target - sum(allocations.values())
    group_ranks = {group_name: rank for rank, group_name in enumerate(group_order)}
    remainders = sorted(
        group_order,
        key=lambda group_name: (
            -(quotas[group_name] - allocations[group_name]),
            group_ranks[group_name],
        ),
    )

    while remainder > 0:
        allocated_in_pass = False
        for group_name in remainders:
            if allocations[group_name] >= group_sizes[group_name]:
                continue

            allocations[group_name] += 1
            remainder -= 1
            allocated_in_pass = True
            if remainder == 0:
                break

        if not allocated_in_pass:
            break

    return allocations


def _select_task_proportional_subset(
    shuffled_task_buckets: dict[str, list[int]],
    task_order: list[str],
    total_target: int,
) -> list[int]:
    """Select a split-local subset while preserving source task proportions.

    Args:
        shuffled_task_buckets: Deterministically shuffled task buckets for one
            source split.
        task_order: Ordered task labels from `meta/tasks.jsonl`.
        total_target: Number of episodes to select from this source split.

    Returns:
        Episode indices selected from this split. The caller may sort them later
        if source order should be restored.
    """

    task_sizes = {
        task_name: len(shuffled_task_buckets[task_name])
        for task_name in task_order
    }
    task_targets = _apportion_targets(total_target, task_sizes, task_order)

    selected_episode_indices: list[int] = []
    for task_name in task_order:
        selected_episode_indices.extend(
            shuffled_task_buckets[task_name][: task_targets[task_name]]
        )

    return selected_episode_indices


def _compute_preserved_splits(
    selected_episode_indices: list[int],
    split_ranges: dict[str, tuple[int, int]],
) -> dict[str, str]:
    """Map source split ranges onto the sampled dataset's new episode numbering.

    Args:
        selected_episode_indices: Original source episode indices included in the
            sampled dataset. The caller may pass them in any order.
        split_ranges: Source split ranges that should remain valid after sampling.

    Returns:
        Split range mapping suitable for `meta/info.json`.
    """

    sorted_selected_indices = sorted(selected_episode_indices)
    new_splits: dict[str, str] = {}

    for split_name, (start_idx, end_idx) in sorted(
        split_ranges.items(),
        key=lambda item: item[1][0],
    ):
        new_start = bisect_left(sorted_selected_indices, start_idx)
        new_end = bisect_left(sorted_selected_indices, end_idx)
        new_splits[split_name] = f"{new_start}:{new_end}"

    return new_splits


def _count_selected_tasks(
    selected_episode_indices: list[int],
    episodes: list[dict[str, Any]],
    split_ranges: dict[str, tuple[int, int]],
    task_order: list[str],
) -> dict[str, dict[str, int]]:
    """Count selected episodes by preserved source split and task label.

    Args:
        selected_episode_indices: Original episode indices that were selected.
        episodes: Parsed `meta/episodes.jsonl` entries from the source dataset.
        split_ranges: Source split ranges used for stratification.
        task_order: Ordered task labels from `meta/tasks.jsonl`.

    Returns:
        Nested mapping `split_name -> task_name -> count`.
    """

    split_counts = {
        split_name: {task_name: 0 for task_name in task_order}
        for split_name in split_ranges
    }

    for episode_index in selected_episode_indices:
        task_name = _episode_task_name(episodes[episode_index], episode_index)
        for split_name, (start_idx, end_idx) in split_ranges.items():
            if start_idx <= episode_index < end_idx:
                split_counts[split_name][task_name] += 1
                break

    return split_counts


def _copy_optional_meta_files(input_path: Path, output_path: Path) -> list[str]:
    """Copy optional metadata files that downstream training often expects.

    Args:
        input_path: Source dataset path.
        output_path: Sampled dataset path.

    Returns:
        List of optional metadata filenames copied into the sampled dataset.
    """

    copied_files: list[str] = []
    for filename in OPTIONAL_META_FILES_TO_COPY:
        source_file = input_path / "meta" / filename
        if not source_file.exists():
            continue

        shutil.copy2(source_file, output_path / "meta" / filename)
        copied_files.append(filename)

    return copied_files


def _rewrite_output_info(
    output_path: Path,
    selected_episode_indices: list[int],
    split_ranges: dict[str, tuple[int, int]],
) -> None:
    """Rewrite `meta/info.json` to keep preserved split metadata meaningful.

    Args:
        output_path: Materialized sampled dataset path.
        selected_episode_indices: Original episode indices used to build the subset.
        split_ranges: Source split ranges that should remain valid after sampling.
    """

    info_path = output_path / "meta" / "info.json"
    with open(info_path) as info_file:
        info = json.load(info_file)

    preserved_splits = _compute_preserved_splits(selected_episode_indices, split_ranges)
    info["splits"] = {
        "train": f"0:{len(selected_episode_indices)}",
        **preserved_splits,
    }

    with open(info_path, "w") as info_file:
        json.dump(info, info_file, indent=4)


def _format_percentage_label(fraction: float) -> str:
    """Format a fraction as a zero-padded percentage label."""

    return f"{int(round(fraction * 100)):03d}"


def _build_output_path(output_root: Path, dataset_name: str, fraction: float) -> Path:
    """Build the output directory name for one subset percentage."""

    return output_root / f"{dataset_name}_{_format_percentage_label(fraction)}pct"


def _materialize_subset(
    input_path: Path,
    output_path: Path,
    selected_episode_indices: list[int],
    split_ranges: dict[str, tuple[int, int]],
    verbose: bool,
) -> list[str]:
    """Create one sampled dataset on disk.

    Args:
        input_path: Source dataset root.
        output_path: Destination directory for the sampled dataset.
        selected_episode_indices: Original episode indices selected for this subset.
        split_ranges: Source split ranges that should remain valid after sampling.
        verbose: Whether to print progress messages.

    Returns:
        List of optional metadata filenames that were copied into the sampled
        dataset after materialization.

    Raises:
        FileExistsError: If the destination already exists.
    """

    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")

    sample_dataset(
        input_path=input_path,
        output_path=output_path,
        episode_indices=selected_episode_indices,
        verbose=verbose,
    )

    copied_meta_files = _copy_optional_meta_files(input_path, output_path)
    _rewrite_output_info(output_path, selected_episode_indices, split_ranges)
    return copied_meta_files


def _write_manifest(output_root: Path, dataset_name: str, seed: int, manifest: dict[str, Any]) -> Path:
    """Write the combined subset manifest for reproducibility and auditing.

    Args:
        output_root: Root directory containing the generated subsets.
        dataset_name: Source dataset basename used in the manifest filename.
        seed: Sampling seed used to generate all subsets.
        manifest: JSON-serializable manifest content.

    Returns:
        Path to the written manifest JSON file.
    """

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"{dataset_name}_task_proportional_sampling_seed{seed}.json"
    with open(manifest_path, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=4)
    return manifest_path


def _build_argument_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for this utility."""

    parser = argparse.ArgumentParser(
        description="Create split- and task-proportional LeRobot subsets for data-efficiency ablations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to the source LeRobot dataset.",
    )
    parser.add_argument(
        "--output-root",
        "-o",
        type=Path,
        required=True,
        help="Directory where subset datasets and the manifest will be written.",
    )
    parser.add_argument(
        "--percentages",
        "-p",
        type=float,
        nargs="+",
        default=DEFAULT_PERCENTAGES,
        help=(
            "Subset percentages to create. Values > 1 are interpreted as percentages "
            "(for example 33 66 100), while values in (0, 1] are treated as fractions."
        ),
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed used to shuffle each `(split, task)` bucket. Default: 42.",
    )
    parser.add_argument(
        "--stratify-splits",
        nargs="+",
        default=DEFAULT_STRATIFY_SPLITS,
        help=(
            "Source split names whose relative sizes should be preserved in each subset. "
            "Default: perfect recovery."
        ),
    )
    parser.add_argument(
        "--materialize-100",
        action="store_true",
        help=(
            "Also copy the 100 percent subset into `--output-root`. By default the "
            "100 percent entry in the manifest just points at `--input`."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned subsets and manifest summary without writing any files.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce progress logging.",
    )

    return parser


def main() -> None:
    """Parse CLI arguments and create split- and task-proportional subsets."""

    parser = _build_argument_parser()
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input dataset does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.input == args.output_root:
        print("Error: --output-root must differ from --input.", file=sys.stderr)
        sys.exit(1)

    try:
        fractions = _normalize_percentages(args.percentages)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    info = load_info(args.input)
    episodes = load_episodes(args.input)
    tasks_by_index = load_tasks(args.input)
    task_order = [tasks_by_index[idx]["task"] for idx in sorted(tasks_by_index)]

    try:
        split_ranges = _load_split_ranges(info, args.stratify_splits)
        split_buckets = _collect_episode_buckets(episodes, task_order, split_ranges)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    split_order = sorted(split_ranges, key=lambda split_name: split_ranges[split_name][0])
    shuffled_split_task_buckets = {
        split_name: _shuffle_task_buckets(split_buckets[split_name], task_order, rng)
        for split_name in split_order
    }
    split_sizes = {
        split_name: sum(
            len(shuffled_split_task_buckets[split_name][task_name])
            for task_name in task_order
        )
        for split_name in split_order
    }

    manifest: dict[str, Any] = {
        "source_dataset": str(args.input.resolve()),
        "output_root": str(args.output_root.resolve()),
        "seed": args.seed,
        "task_order": task_order,
        "stratify_splits": split_order,
        "task_sampling_mode": "split_proportional",
        "source_split_sizes": split_sizes,
        "subsets": [],
    }

    if not args.quiet:
        print(f"Source dataset: {args.input}")
        print(f"Task order: {task_order}")
        print(f"Preserved source splits: {split_sizes}")

    for fraction in fractions:
        total_target = int(round(info["total_episodes"] * fraction))
        split_targets = _apportion_targets(total_target, split_sizes, split_order)

        selected_episode_indices: list[int] = []
        for split_name in split_order:
            selected_episode_indices.extend(
                _select_task_proportional_subset(
                    shuffled_task_buckets=shuffled_split_task_buckets[split_name],
                    task_order=task_order,
                    total_target=split_targets[split_name],
                )
            )

        selected_episode_indices = sorted(selected_episode_indices)
        subset_task_counts = _count_selected_tasks(
            selected_episode_indices=selected_episode_indices,
            episodes=episodes,
            split_ranges=split_ranges,
            task_order=task_order,
        )
        total_task_counts = {
            task_name: sum(
                subset_task_counts[split_name][task_name] for split_name in split_order
            )
            for task_name in task_order
        }

        should_materialize = not args.dry_run and (
            fraction < 1.0 or args.materialize_100
        )
        output_path = _build_output_path(args.output_root, args.input.name, fraction)
        copied_meta_files: list[str] = []

        if should_materialize:
            copied_meta_files = _materialize_subset(
                input_path=args.input,
                output_path=output_path,
                selected_episode_indices=selected_episode_indices,
                split_ranges=split_ranges,
                verbose=not args.quiet,
            )

        resolved_output_path = (
            str(output_path.resolve()) if should_materialize else str(args.input.resolve())
        )

        subset_summary = {
            "fraction": fraction,
            "percentage": round(fraction * 100, 4),
            "episodes": len(selected_episode_indices),
            "materialized": should_materialize,
            "output_path": resolved_output_path,
            "split_targets": split_targets,
            "task_counts_by_split": subset_task_counts,
            "task_counts_total": total_task_counts,
            "selected_episode_indices": selected_episode_indices,
            "copied_optional_meta_files": copied_meta_files,
        }
        manifest["subsets"].append(subset_summary)

        if not args.quiet:
            print(
                f"{round(fraction * 100, 2):>6.2f}% -> {len(selected_episode_indices):4d} episodes "
                f"(split targets: {split_targets})"
            )
            print(f"         tasks: {total_task_counts}")
            if should_materialize:
                print(f"         output: {output_path}")
            else:
                print(f"         output: source dataset ({args.input})")

    if args.dry_run:
        if not args.quiet:
            print("Dry run complete. No files were written.")
        return

    manifest_path = _write_manifest(args.output_root, args.input.name, args.seed, manifest)
    if not args.quiet:
        print(f"Manifest written to: {manifest_path}")
        print(
            "Note: copied percentile/relative stats reflect the source dataset. "
            "Regenerate subset-specific stats if the ablation requires them."
        )


if __name__ == "__main__":
    main()
