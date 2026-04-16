from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset


REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from aloha_pro.aloha_scripts.utils import encode_text, initialize_model_and_tokenizer
from dvrk_scripts.constants_dvrk import TASK_CONFIGS
from policy import ACTPolicy


MANIFEST_FILENAME = "manifest.json"


class DeterministicComparisonDataset(Dataset):
    """Lazy loader for the saved deterministic debug dataset."""

    def __init__(self, root_dir: str | Path, split: str, decode_images: bool = True) -> None:
        super().__init__()
        self.root_dir = Path(root_dir).expanduser()
        if not self.root_dir.is_absolute():
            self.root_dir = self.root_dir.resolve()
        self.decode_images = decode_images

        manifest_path = self.root_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as file:
            self.manifest = json.load(file)

        if split not in self.manifest["splits"]:
            available = ", ".join(sorted(self.manifest["splits"].keys()))
            raise ValueError(f"Unknown split '{split}'. Available splits: {available}")

        self.split = split
        self.entries = list(self.manifest["splits"][split])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        sample_path = self.root_dir / entry["relative_path"]
        sample = torch.load(sample_path, map_location="cpu")
        if self.decode_images and sample["image_data"].dtype == torch.uint8:
            sample["image_data"] = sample["image_data"].float() / 255.0
        sample["relative_path"] = entry["relative_path"]
        return sample


def _build_policy_config(task_name: str, chunk_size: int) -> dict[str, Any]:
    task_config = TASK_CONFIGS[task_name]
    return {
        "lr": 1e-5,
        "num_queries": chunk_size,
        "action_dim": 20,
        "kl_weight": 10,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "lr_backbone": 1e-5,
        "backbone": "efficientnet_b3film",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": task_config["camera_names"],
        "multi_gpu": False,
        "use_language": True,
    }


def _load_policy(
    ckpt_path: str | Path,
    policy_class: str,
    task_name: str,
    seed: int,
    num_epochs: int,
    chunk_size: int,
    device: torch.device,
) -> ACTPolicy:
    original_argv = sys.argv[:]
    sys.argv = [
        original_argv[0],
        "--ckpt_dir",
        str(ckpt_path),
        "--policy_class",
        policy_class,
        "--task_name",
        task_name,
        "--seed",
        str(seed),
        "--num_epochs",
        str(num_epochs),
    ]
    try:
        policy = ACTPolicy(_build_policy_config(task_name, chunk_size))
    finally:
        sys.argv = original_argv
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    load_result = policy.deserialize(state_dict)
    print(load_result)
    policy.to(device)
    policy.eval()
    return policy


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return array / norms


def _convert_delta_6d_to_taskspace_quat(
    all_actions: np.ndarray,
    all_actions_converted: np.ndarray,
    qpos: np.ndarray,
) -> np.ndarray:
    c1 = _normalize_rows(all_actions[:, 3:6])
    c2 = all_actions[:, 6:9]
    c2 = _normalize_rows(c2 - np.sum(c1 * c2, axis=1, keepdims=True) * c1)
    c3 = np.cross(c1, c2)
    r_mat = np.dstack((c1, c2, c3))
    rots = R.from_matrix(r_mat)
    rot_init = R.from_quat(qpos[3:7])
    all_actions_converted[:, 3:7] = (rot_init * rots).as_quat()
    return all_actions_converted


def _convert_delta_6d_to_taskspace_quat_relative_endo(
    all_actions: np.ndarray,
    all_actions_converted: np.ndarray,
) -> np.ndarray:
    c1 = _normalize_rows(all_actions[:, 3:6])
    c2 = all_actions[:, 6:9]
    c2 = _normalize_rows(c2 - np.sum(c1 * c2, axis=1, keepdims=True) * c1)
    c3 = np.cross(c1, c2)
    r_mat = np.dstack((c1, c2, c3))
    all_actions_converted[:, 3:7] = R.from_matrix(r_mat).as_quat()
    return all_actions_converted


def _denormalize_policy_actions(
    normalized_actions: torch.Tensor,
    task_name: str,
) -> np.ndarray:
    task_config = TASK_CONFIGS[task_name]
    norm_scheme = task_config["norm_scheme"]
    stats = task_config["action_mode"][1]

    normalized_np = normalized_actions.detach().cpu().numpy()
    if norm_scheme == "std":
        denormalized = normalized_np * stats["std"] + stats["mean"]
        denormalized[..., 3:9] = normalized_np[..., 3:9]
        denormalized[..., 13:19] = normalized_np[..., 13:19]
        return denormalized.astype(np.float32)
    if norm_scheme == "min_max":
        denormalized = (normalized_np + 1.0) / 2.0 * (stats["max_"] - stats["min_"]) + stats["min_"]
        denormalized[..., 3:9] = normalized_np[..., 3:9]
        denormalized[..., 13:19] = normalized_np[..., 13:19]
        return denormalized.astype(np.float32)
    raise NotImplementedError(f"Unsupported norm scheme: {norm_scheme}")


def _policy_actions_to_absolute_actions(
    policy_actions: np.ndarray,
    current_pose: np.ndarray,
    task_name: str,
) -> np.ndarray:
    task_config = TASK_CONFIGS[task_name]
    action_mode = task_config["action_mode"][0]
    qpos_psm1 = current_pose[:8]
    qpos_psm2 = current_pose[8:16]
    chunk_size = policy_actions.shape[0]

    if action_mode == "hybrid":
        actions_psm1 = np.zeros((chunk_size, 8), dtype=np.float32)
        actions_psm1[:, 0:3] = qpos_psm1[0:3] + policy_actions[:, 0:3]
        actions_psm1 = _convert_delta_6d_to_taskspace_quat(policy_actions[:, 0:10], actions_psm1, qpos_psm1)
        actions_psm1[:, 7] = np.clip(policy_actions[:, 9], -0.698, 0.698)

        actions_psm2 = np.zeros((chunk_size, 8), dtype=np.float32)
        actions_psm2[:, 0:3] = qpos_psm2[0:3] + policy_actions[:, 10:13]
        actions_psm2 = _convert_delta_6d_to_taskspace_quat(policy_actions[:, 10:20], actions_psm2, qpos_psm2)
        actions_psm2[:, 7] = np.clip(policy_actions[:, 19], -0.698, 0.698)
    elif action_mode == "relative_endoscope":
        actions_psm1 = np.zeros((chunk_size, 8), dtype=np.float32)
        actions_psm1[:, 0:3] = qpos_psm1[0:3] + policy_actions[:, 0:3]
        actions_psm1 = _convert_delta_6d_to_taskspace_quat_relative_endo(policy_actions[:, 0:10], actions_psm1)
        actions_psm1[:, 7] = np.clip(policy_actions[:, 9], -0.698, 0.698)

        actions_psm2 = np.zeros((chunk_size, 8), dtype=np.float32)
        actions_psm2[:, 0:3] = qpos_psm2[0:3] + policy_actions[:, 10:13]
        actions_psm2 = _convert_delta_6d_to_taskspace_quat_relative_endo(policy_actions[:, 10:20], actions_psm2)
        actions_psm2[:, 7] = np.clip(policy_actions[:, 19], -0.698, 0.698)
    else:
        raise NotImplementedError(
            f"Unsupported action_mode '{action_mode}' for deterministic comparison."
        )

    return np.column_stack((actions_psm1, actions_psm2)).astype(np.float32)


def _collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_data": torch.stack([sample["image_data"] for sample in samples], dim=0),
        "current_pose_data": torch.stack([sample["current_pose_data"] for sample in samples], dim=0),
        "raw_action_data": torch.stack([sample["raw_action_data"] for sample in samples], dim=0),
        "is_pad": torch.stack([sample["is_pad"] for sample in samples], dim=0),
        "normalized_action_data": torch.stack(
            [sample["normalized_action_data"] for sample in samples], dim=0
        ),
        "normalized_action_predictions": torch.stack(
            [sample["normalized_action_predictions"] for sample in samples], dim=0
        ),
        "raw_action_predictions": torch.stack(
            [sample["raw_action_predictions"] for sample in samples], dim=0
        ),
        "command_text": [sample["command_text"] for sample in samples],
        "relative_path": [sample["relative_path"] for sample in samples],
        "epoch_index": [int(sample["epoch_index"]) for sample in samples],
        "dataset_index": [int(sample["dataset_index"]) for sample in samples],
        "seed": [int(sample["seed"]) for sample in samples],
    }


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> float:
    if not bool(valid_mask.any()):
        return 0.0
    return float((pred[valid_mask] - target[valid_mask]).abs().mean().item())


def _masked_max_abs(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> float:
    if not bool(valid_mask.any()):
        return 0.0
    return float((pred[valid_mask] - target[valid_mask]).abs().max().item())


def _mean_stacked_vectors(vectors: list[torch.Tensor], action_dim: int) -> np.ndarray:
    if not vectors:
        return np.zeros(action_dim, dtype=np.float64)
    return torch.stack(vectors, dim=0).mean(dim=0).cpu().numpy()


def _mean_concatenated_rows(rows: list[torch.Tensor], action_dim: int) -> np.ndarray:
    if not rows:
        return np.zeros(action_dim, dtype=np.float64)
    return torch.cat(rows, dim=0).mean(dim=0).cpu().numpy()


def _format_vector(values: np.ndarray) -> str:
    return np.array2string(np.asarray(values, dtype=np.float64), precision=6, separator=", ")


def _encode_command_embeddings(
    command_texts: list[str],
    language_encoder: str,
    tokenizer,
    language_model,
    device: torch.device,
    cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    embeddings = []
    for text in command_texts:
        if text not in cache:
            embedding = torch.tensor(
                encode_text(text, language_encoder, tokenizer, language_model),
                dtype=torch.float32,
            ).flatten()
            cache[text] = embedding.cpu()
        embeddings.append(cache[text])
    return torch.stack(embeddings, dim=0).to(device)


def _compare_split(
    dataset: DeterministicComparisonDataset,
    policy: ACTPolicy,
    args: argparse.Namespace,
    device: torch.device,
    tokenizer,
    language_model,
    split
):
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_samples,
    )
    command_cache: dict[str, torch.Tensor] = {}
    per_sample_rows: list[dict[str, Any]] = []

    for batch in dataloader:
        image_data = batch["image_data"].to(device=device, dtype=torch.float32)
        batch_size = image_data.shape[0]
        qpos_zero = torch.zeros((batch_size, 20), device=device, dtype=torch.float32)
        command_embedding = _encode_command_embeddings(
            batch["command_text"],
            args.language_encoder,
            tokenizer,
            language_model,
            device,
            command_cache,
        )

        with torch.inference_mode():
            legacy_normalized_predictions = policy(
                qpos_zero,
                image_data,
                command_embedding=command_embedding,
            ).detach().cpu()

        legacy_denormalized_policy_actions = _denormalize_policy_actions(
            legacy_normalized_predictions,
            args.task_name,
        )
        legacy_raw_predictions = np.stack(
            [
                _policy_actions_to_absolute_actions(policy_action, current_pose.numpy(), args.task_name)
                for policy_action, current_pose in zip(
                    legacy_denormalized_policy_actions,
                    batch["current_pose_data"],
                )
            ],
            axis=0,
        )

        gt_action_data = batch["raw_action_data"]
        new_raw_predictions = batch["raw_action_predictions"]
        valid_mask = ~batch["is_pad"]

        new_l1_first_pred_list = list()
        new_l1_total_list = list()
        old_l1_first_pred_list = list()
        old_l1_total_list = list()
        compare_l1_first_pred_list = list()
        compare_l1_total_list = list()
        new_l1_first_pred_per_dim_list = list()
        new_l1_all_steps_per_dim_rows = list()
        old_l1_first_pred_per_dim_list = list()
        old_l1_all_steps_per_dim_rows = list()
        compare_l1_first_pred_per_dim_list = list()
        compare_l1_all_steps_per_dim_rows = list()

        for batch_idx in range(valid_mask.shape[0]):
            old_pred_raw_valid = torch.tensor(legacy_raw_predictions[batch_idx, valid_mask[batch_idx]])
            new_pred_raw_valid = new_raw_predictions[batch_idx, valid_mask[batch_idx].cpu()].cpu()
            gt_raw_valid = gt_action_data[batch_idx, valid_mask[batch_idx].cpu()].cpu()

            new_first_abs_diff = (new_pred_raw_valid[0] - gt_raw_valid[0]).abs()
            new_all_abs_diff = (new_pred_raw_valid - gt_raw_valid).abs()
            old_first_abs_diff = (old_pred_raw_valid[0] - gt_raw_valid[0]).abs()
            old_all_abs_diff = (old_pred_raw_valid - gt_raw_valid).abs()
            compare_first_abs_diff = (old_pred_raw_valid[0] - new_pred_raw_valid[0]).abs()
            compare_all_abs_diff = (old_pred_raw_valid - new_pred_raw_valid).abs()

            new_l1_first_pred_list.append(float(new_first_abs_diff.sum()))
            new_l1_total_list.append(float(new_all_abs_diff.sum()))
            new_l1_first_pred_per_dim_list.append(new_first_abs_diff)
            new_l1_all_steps_per_dim_rows.append(new_all_abs_diff)

            old_l1_first_pred_list.append(float(old_first_abs_diff.sum()))
            old_l1_total_list.append(float(old_all_abs_diff.sum()))
            old_l1_first_pred_per_dim_list.append(old_first_abs_diff)
            old_l1_all_steps_per_dim_rows.append(old_all_abs_diff)

            compare_l1_first_pred_list.append(float(compare_first_abs_diff.sum()))
            compare_l1_total_list.append(float(compare_all_abs_diff.sum()))
            compare_l1_first_pred_per_dim_list.append(compare_first_abs_diff)
            compare_l1_all_steps_per_dim_rows.append(compare_all_abs_diff)

        action_dim = gt_action_data.shape[-1]
        new_l1_first_pred_avg = (sum(new_l1_first_pred_list) / len(new_l1_first_pred_list)) / action_dim
        new_l1_total_avg = (sum(new_l1_total_list) / len(new_l1_total_list)) / action_dim
        new_l1_first_pred_per_dim_avg = _mean_stacked_vectors(new_l1_first_pred_per_dim_list, action_dim)
        new_l1_all_steps_per_dim_avg = _mean_concatenated_rows(new_l1_all_steps_per_dim_rows, action_dim)

        old_l1_first_pred_avg = (sum(old_l1_first_pred_list) / len(old_l1_first_pred_list)) / action_dim
        old_l1_total_avg = (sum(old_l1_total_list) / len(old_l1_total_list)) / action_dim
        old_l1_first_pred_per_dim_avg = _mean_stacked_vectors(old_l1_first_pred_per_dim_list, action_dim)
        old_l1_all_steps_per_dim_avg = _mean_concatenated_rows(old_l1_all_steps_per_dim_rows, action_dim)

        compare_l1_first_pred_avg = (sum(compare_l1_first_pred_list) / len(compare_l1_first_pred_list)) / action_dim
        compare_l1_total_avg = (sum(compare_l1_total_list) / len(compare_l1_total_list)) / action_dim
        compare_l1_first_pred_per_dim_avg = _mean_stacked_vectors(compare_l1_first_pred_per_dim_list, action_dim)
        compare_l1_all_steps_per_dim_avg = _mean_concatenated_rows(compare_l1_all_steps_per_dim_rows, action_dim)

        print(
            "%s raw action debug - | "
            "NEW: First Pred: %.6f Total: %.6f | "
            "OLD: First Pred: %.6f Total: %.6f | "
            "COMPARE: First Pred: %.6f Total: %.6f"
            % (
                split,
                new_l1_first_pred_avg,
                new_l1_total_avg,
                old_l1_first_pred_avg,
                old_l1_total_avg,
                compare_l1_first_pred_avg,
                compare_l1_total_avg,
            )
        )
        print(f"{split} raw action per-dim L1 avg -")
        print(f"NEW: First Pred: {_format_vector(new_l1_first_pred_per_dim_avg)}")
        print(f"NEW: All Timesteps Mean: {_format_vector(new_l1_all_steps_per_dim_avg)}")
        print(f"OLD: First Pred: {_format_vector(old_l1_first_pred_per_dim_avg)}")
        print(f"OLD: All Timesteps Mean: {_format_vector(old_l1_all_steps_per_dim_avg)}")
        print(f"COMPARE: First Pred: {_format_vector(compare_l1_first_pred_per_dim_avg)}")
        print(f"COMPARE: All Timesteps Mean: {_format_vector(compare_l1_all_steps_per_dim_avg)}")

        return


    #     legacy_raw_predictions = torch.from_numpy(legacy_raw_predictions)

    #     for sample_idx in range(batch_size):
    #         sample_valid_mask = valid_mask[sample_idx]
    #         per_sample_rows.append(
    #             {
    #                 "relative_path": batch["relative_path"][sample_idx],
    #                 "epoch_index": batch["epoch_index"][sample_idx],
    #                 "dataset_index": batch["dataset_index"][sample_idx],
    #                 "seed": batch["seed"][sample_idx],
    #                 "command_text": batch["command_text"][sample_idx],
    #                 "num_valid_steps": int(sample_valid_mask.sum().item()),
    #                 "legacy_vs_saved_norm_mae": _masked_mae(
    #                     legacy_normalized_predictions[sample_idx],
    #                     batch["normalized_action_predictions"][sample_idx],
    #                     sample_valid_mask,
    #                 ),
    #                 "legacy_vs_saved_norm_max_abs": _masked_max_abs(
    #                     legacy_normalized_predictions[sample_idx],
    #                     batch["normalized_action_predictions"][sample_idx],
    #                     sample_valid_mask,
    #                 ),
    #                 "legacy_vs_saved_raw_mae": _masked_mae(
    #                     legacy_raw_predictions[sample_idx],
    #                     batch["raw_action_predictions"][sample_idx],
    #                     sample_valid_mask,
    #                 ),
    #                 "legacy_vs_saved_raw_max_abs": _masked_max_abs(
    #                     legacy_raw_predictions[sample_idx],
    #                     batch["raw_action_predictions"][sample_idx],
    #                     sample_valid_mask,
    #                 ),
    #                 "legacy_vs_target_norm_mae": _masked_mae(
    #                     legacy_normalized_predictions[sample_idx],
    #                     batch["normalized_action_data"][sample_idx],
    #                     sample_valid_mask,
    #                 ),
    #                 "legacy_vs_target_raw_mae": _masked_mae(
    #                     legacy_raw_predictions[sample_idx],
    #                     batch["raw_action_data"][sample_idx],
    #                     sample_valid_mask,
    #                 ),
    #                 "saved_vs_target_norm_mae": _masked_mae(
    #                     batch["normalized_action_predictions"][sample_idx],
    #                     batch["normalized_action_data"][sample_idx],
    #                     sample_valid_mask,
    #                 ),
    #                 "saved_vs_target_raw_mae": _masked_mae(
    #                     batch["raw_action_predictions"][sample_idx],
    #                     batch["raw_action_data"][sample_idx],
    #                     sample_valid_mask,
    #                 ),
    #             }
    #         )

    #         if args.max_samples is not None and len(per_sample_rows) >= args.max_samples:
    #             break
    #     if args.max_samples is not None and len(per_sample_rows) >= args.max_samples:
    #         break

    # metric_names = [
    #     "legacy_vs_saved_norm_mae",
    #     "legacy_vs_saved_raw_mae",
    #     "legacy_vs_target_norm_mae",
    #     "legacy_vs_target_raw_mae",
    #     "saved_vs_target_norm_mae",
    #     "saved_vs_target_raw_mae",
    # ]
    # aggregate_metrics = {}
    # if per_sample_rows:
    #     for metric_name in metric_names:
    #         values = [row[metric_name] for row in per_sample_rows]
    #         aggregate_metrics[metric_name] = float(np.mean(values))
    #         aggregate_metrics[f"{metric_name}_median"] = float(np.median(values))
    #         aggregate_metrics[f"{metric_name}_max"] = float(np.max(values))

    # worst_by_raw = sorted(
    #     per_sample_rows,
    #     key=lambda row: row["legacy_vs_saved_raw_mae"],
    #     reverse=True,
    # )[: args.top_k]
    # worst_by_norm = sorted(
    #     per_sample_rows,
    #     key=lambda row: row["legacy_vs_saved_norm_mae"],
    #     reverse=True,
    # )[: args.top_k]

    # return {
    #     "num_samples_compared": len(per_sample_rows),
    #     "aggregate_metrics": aggregate_metrics,
    #     "worst_samples_by_legacy_vs_saved_raw_mae": worst_by_raw,
    #     "worst_samples_by_legacy_vs_saved_norm_mae": worst_by_norm,
    # }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the legacy ACT inference pipeline on a deterministic dataset "
            "and compare its normalized and raw outputs against the saved dataset outputs."
        )
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the deterministic comparison dataset root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "all"],
        help="Dataset split to compare.",
    )
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--policy_class", type=str, default="ACT")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=60)
    parser.add_argument("--language_encoder", type=str, default="distilbert")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path for the comparison summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.policy_class != "ACT":
        raise NotImplementedError(
            f"Only ACT checkpoints are supported right now, got: {args.policy_class}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    policy = _load_policy(
        args.ckpt_dir,
        args.policy_class,
        args.task_name,
        args.seed,
        args.num_epochs,
        args.chunk_size,
        device,
    )
    tokenizer, language_model = initialize_model_and_tokenizer(args.language_encoder)
    language_model.eval()

    splits = ["train", "val"] if args.split == "all" else [args.split]
    for split in splits:
        print(f"\nComparing split: {split}")
        dataset = DeterministicComparisonDataset(args.root_dir, split, decode_images=True)
        _compare_split(dataset, policy, args, device, tokenizer, language_model, split)


if __name__ == "__main__":
    main()
