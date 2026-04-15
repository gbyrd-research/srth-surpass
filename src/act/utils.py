import numpy as np
import torch
import os
import random
# import h5py
import torch.utils.data
from torch.utils.data import DataLoader, ConcatDataset, Sampler, WeightedRandomSampler
from collections import Counter
import cv2
import json
from torchvision import transforms
import sys
from torch.utils.data import Dataset
path_to_yay_robot = os.getenv('PATH_TO_SKAY_ROBOT')

if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
from aloha_pro.aloha_scripts.utils import crop_resize
# from auto_label.auto_label_func import get_auto_label
# from generic_dataset import *
from generic_dataset_invivo import *
CROP_TOP = True  # hardcode
FILTER_MISTAKES = True  # Filter out mistakes from the dataset even if not use_language


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        max_len=None,
        command_list=None,
        use_language=False,
        language_encoder=None,
        policy_class=None,
    ):
        super().__init__()
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0]
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.max_len = max_len
        self.command_list = [cmd.strip("'\"") for cmd in command_list]
        self.use_language = use_language
        self.language_encoder = language_encoder
        self.policy_class = policy_class
        self.transformations = None

        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        max_len = self.max_len

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")

        if self.use_language or FILTER_MISTAKES:
            json_name = f"episode_{episode_id}_encoded_{self.language_encoder}.json"
            encoded_json_path = os.path.join(self.dataset_dir, json_name)

            with open(encoded_json_path, "r") as f:
                episode_data = json.load(f)

        if len(self.command_list) > 0:
            # If command_list is provided, use the JSON file to determine the relevant timesteps
            matching_segments = []

            for segment in episode_data:
                if segment["command"] in self.command_list:
                    current_idx = episode_data.index(segment)
                    if (
                        current_idx + 1 < len(episode_data)
                        and episode_data[current_idx + 1]["type"] == "correction"
                    ):
                        continue
                    else:
                        matching_segments.append(segment)

            # Choose a segment randomly among the matching segments
            chosen_segment = random.choice(matching_segments)

            segment_start, segment_end = (
                chosen_segment["start_timestep"],
                chosen_segment["end_timestep"],
            )
            if self.use_language:
                command_embedding = torch.tensor(chosen_segment["embedding"]).squeeze()

            if segment_start is None or segment_end is None:
                raise ValueError(f"Command segment not found for episode {episode_id}")

        elif self.use_language or FILTER_MISTAKES:
            while True:
                # Randomly sample a segment
                segment = np.random.choice(episode_data)
                current_idx = episode_data.index(segment)
                if (
                    current_idx + 1 < len(episode_data)
                    and episode_data[current_idx + 1]["type"] == "correction"
                ):
                    continue
                segment_start, segment_end = (
                    segment["start_timestep"],
                    segment["end_timestep"],
                )
                # if end and start are too close, skip
                if segment_end - segment_start + 1 < 20:
                    continue
                command_embedding = torch.tensor(segment["embedding"]).squeeze()
                break

        with h5py.File(dataset_path, "r") as root:
            is_sim = root.attrs["sim"]
            self.is_sim = is_sim
            compressed = root.attrs.get("compress", False)
            original_action_shape = root["/action"].shape

            if len(self.command_list) > 0 or self.use_language:
                # Sample within the segment boundaries
                start_ts = np.random.randint(segment_start, segment_end)
                end_ts = min(segment_end, start_ts + max_len - 2)
            else:
                start_ts = np.random.choice(original_action_shape[0])
                end_ts = original_action_shape[0] - 1

            # Get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            # qvel = root['/observations/qvel'][start_ts]

            # Construct the image dictionary for the desired timestep
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][
                    start_ts
                ]

            # Decompress images if they're compressed
            if compressed:
                for cam_name in image_dict.keys():
                    decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                    image_dict[cam_name] = np.array(decompressed_image)

                    # Check for 'cam_high' and apply transformation
                    if CROP_TOP and cam_name == "cam_high":
                        image_dict[cam_name] = crop_resize(image_dict[cam_name])

            # Swap BGR to RGB
            for cam_name in image_dict.keys():
                image_dict[cam_name] = cv2.cvtColor(
                    image_dict[cam_name], cv2.COLOR_BGR2RGB
                )

            # Adjusting action loading
            if is_sim:
                action = root["/action"][start_ts : end_ts + 1]
                action_len = end_ts - start_ts + 1
            else:
                # hack, to make timesteps more aligned
                action = root["/action"][max(0, start_ts - 1) : end_ts + 1]
                action_len = end_ts - max(0, start_ts - 1) + 1

            # Adjusting the padded action and padding flags
            padded_action = np.zeros(
                (max_len,) + original_action_shape[1:], dtype=np.float32
            )
            padded_action[:action_len] = action
            is_pad = np.zeros(max_len)
            is_pad[action_len:] = 1

            # Constructing the image data for all cameras
            all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
            all_cam_images = np.stack(all_cam_images, axis=0)

            # Constructing the observations
            image_data = torch.from_numpy(all_cam_images)
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            # Adjusting channel
            image_data = torch.einsum("k h w c -> k c h w", image_data)

            # Augmentation
            if self.transformations is None:
                print("Initializing transformations")
                original_size = image_data.shape[2:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(
                        size=[
                            int(original_size[0] * ratio),
                            int(original_size[1] * ratio),
                        ]
                    ),
                    transforms.Resize(original_size, antialias=True),
                ]
                if self.policy_class == "Diffusion":
                    self.transformations.extend(
                        [
                            transforms.RandomRotation(
                                degrees=[-5.0, 5.0], expand=False
                            ),
                            transforms.ColorJitter(
                                brightness=0.3, contrast=0.4, saturation=0.5
                            ),
                        ]
                    )  # , hue=0.08

            for transform in self.transformations:
                image_data = transform(image_data)

            # Normalize image data and adjust data types
            image_data = image_data / 255.0
            if self.policy_class == "Diffusion":
                # normalize to [-1, 1]
                action_data = (
                    (action_data - self.norm_stats["action_min"])
                    / (self.norm_stats["action_max"] - self.norm_stats["action_min"])
                ) * 2 - 1
            else:
                # normalize to mean 0 std 1
                action_data = (
                    action_data - self.norm_stats["action_mean"]
                ) / self.norm_stats["action_std"]
            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
                "qpos_std"
            ]

            if self.use_language:
                return image_data, qpos_data, action_data, is_pad, command_embedding
            else:
                return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dirs, num_episodes_list):
    all_qpos_data = []
    all_action_data = []

    # Iterate over each directory and the corresponding number of episodes
    for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list):
        for episode_idx in range(num_episodes):
            dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
            with h5py.File(dataset_path, "r") as root:
                qpos = root["/observations/qpos"][()]
                action = root["/action"][()]
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))

    # Concatenate data from all directories
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # Normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)

    # Normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()
    eps = 0.0001

    stats = {
        "action_mean": action_mean.numpy(),
        "action_std": action_std.numpy(),
        "action_min": action_min.numpy() - eps,
        "action_max": action_max.numpy() + eps,
        "qpos_mean": qpos_mean.numpy(),
        "qpos_std": qpos_std.numpy(),
        "example_qpos": all_qpos_data[-1].numpy(),
    }  # example from the last loaded qpos

    return stats

def load_data_dvrk(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    task_config,
    chunk_size,
    use_language=False):
    
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    num_episodes_val = task_config['num_episodes_val']
    train_indices = np.random.permutation(num_episodes)
    val_indices = np.random.permutation(num_episodes_val)

    # obtain normalization stats for qpos and action
    # norm_stats = get_norm_stats(dataset_dir, num_episodes)
    norm_stats = None
    action_mode = task_config['action_mode'][0]
    dataset_path = task_config['dataset_dir']
    tissue_samples_ids = task_config['tissue_samples_ids']
    tissue_samples_ids_val = task_config['tissue_samples_ids_val']
    camera_file_suffixes = task_config['camera_file_suffixes']

    print("\n-------------loading training data-------------\n")

    train_datasets = EpisodicDatasetDvrkGeneric(
            train_indices,
            tissue_samples_ids,
            dataset_dir,
            camera_names,
            camera_file_suffixes, 
            task_config,
            chunk_size=chunk_size,
            use_language=use_language,
        )
    print("\n-------------loading validation data-------------\n")
    
    val_datasets = EpisodicDatasetDvrkGeneric(
            val_indices,
            tissue_samples_ids_val,
            dataset_dir,
            camera_names,
            camera_file_suffixes, 
            task_config,
            use_language=use_language,
        )
    # merged_train_dataset = ConcatDataset(train_datasets)
    # merged_val_dataset = ConcatDataset(val_datasets)
    # train_dataset = EpisodicDatasetDvrkGeneric(train_indices, dataset_path, camera_names, norm_stats, task_config)
    # val_dataset = EpisodicDatasetDvrkGeneric(val_indices, dataset_path, camera_names, norm_stats, task_config)

    train_datasets[0]

    # Get task labels for all samples
    task_labels = train_datasets.sample_task_labels
    task_counts = Counter(task_labels)  # e.g., {'1': 500, '2': 200, '3': 500}
    print(f"task count:{task_counts}")
    total_samples = len(task_labels)

    # Compute weights
    weights = [1.0 / task_counts[task] for task in task_labels]

    # Create sampler
    sampler = WeightedRandomSampler(weights, num_samples=total_samples, replacement=True)

    train_dataloader = DataLoader(train_datasets, batch_size=batch_size_train, sampler=sampler,
                              pin_memory=True, num_workers=16, prefetch_factor=4, persistent_workers=True)

    # train_dataloader = DataLoader(train_datasets, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=16, prefetch_factor=4, persistent_workers=True)
    val_dataloader = DataLoader(val_datasets, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=16, prefetch_factor=4, persistent_workers=True)

    return train_dataloader, val_dataloader, norm_stats, train_datasets.is_sim

def save_dataloader(dataloader, save_path):
    all_batches = []

    for batch_idx, batch in enumerate(dataloader):
        # batch is typically a tuple of 5 tensors (already batched)
        if isinstance(batch, (list, tuple)):
            frozen = tuple(t.clone() for t in batch)
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        all_batches.append(frozen)

    torch.save(all_batches, save_path)
    print(f"Saved {len(all_batches)} batches to {save_path}")


def load_data_dvrk_multi_dataset(
    dataset_dirs,
    num_episodes_list,
    camera_names,
    batch_size_train,
    batch_size_val,
    task_configs,
    chunk_size,
    use_language=False,
    dataset_weights=None):
    """
    Load data from multiple datasets and combine them for co-training.
    
    Args:
        dataset_dirs: List of dataset directory paths
        num_episodes_list: List of episode counts for each dataset
        camera_names: List of camera names (should be consistent across datasets)
        batch_size_train: Training batch size
        batch_size_val: Validation batch size
        task_configs: List of task configs, one for each dataset
        chunk_size: Action chunk size
        use_language: Whether to use language conditioning
        dataset_weights: Optional list of weights for each dataset (for sampling).
                        If None, datasets are weighted equally by number of samples.
    
    Returns:
        train_dataloader, val_dataloader, norm_stats, is_sim
    """
    
    print(f'\n=== Loading {len(dataset_dirs)} datasets for co-training ===\n')
    
    train_datasets_list = []
    val_datasets_list = []
    
    # Load each dataset
    for idx, (dataset_dir, num_episodes, task_config) in enumerate(
        zip(dataset_dirs, num_episodes_list, task_configs)):
        
        print(f'\n--- Dataset {idx+1}/{len(dataset_dirs)}: {dataset_dir} ---')
        
        # obtain train test split
        num_episodes_val = task_config['num_episodes_val']
        train_indices = np.random.permutation(num_episodes)
        val_indices = np.random.permutation(num_episodes_val)
        
        dataset_path = task_config['dataset_dir']
        tissue_samples_ids = task_config['tissue_samples_ids']
        tissue_samples_ids_val = task_config['tissue_samples_ids_val']
        camera_file_suffixes = task_config['camera_file_suffixes']
        
        print(f"Loading training data from {dataset_path}")
        train_dataset = EpisodicDatasetDvrkGeneric(
            train_indices,
            tissue_samples_ids,
            dataset_dir,
            camera_names,
            camera_file_suffixes,
            task_config,
            chunk_size=chunk_size,
            use_language=use_language,
        )
        train_datasets_list.append(train_dataset)
        print(f"  Training samples: {len(train_dataset.all_samples)}")
        
        if num_episodes_val > 0:
            print(f"Loading validation data from {dataset_path}")
            val_dataset = EpisodicDatasetDvrkGeneric(
                val_indices,
                tissue_samples_ids_val,
                dataset_dir,
                camera_names,
                camera_file_suffixes,
                task_config,
                use_language=use_language,
            )
            val_datasets_list.append(val_dataset)
            print(f"  Validation samples: {len(val_dataset.all_samples)}")
    
    # Combine datasets
    print(f'\n=== Combining datasets ===')
    merged_train_dataset = ConcatDataset(train_datasets_list)
    print(f"Total training samples: {len(merged_train_dataset)}")
    
    if len(val_datasets_list) > 0:
        merged_val_dataset = ConcatDataset(val_datasets_list)
        print(f"Total validation samples: {len(merged_val_dataset)}")
    else:
        # Create empty validation dataset if none exist
        merged_val_dataset = None
        print("No validation data")
    
    # Create weighted sampler for balanced sampling across datasets
    if dataset_weights is None:
        # Weight by dataset size (inverse frequency)
        dataset_sizes = [len(ds.all_samples) for ds in train_datasets_list]
        total_size = sum(dataset_sizes)
        dataset_weights = [total_size / size for size in dataset_sizes]
        print(f"\nDataset sizes: {dataset_sizes}")
        print(f"Auto-computed dataset weights (inverse frequency): {[f'{w:.2f}' for w in dataset_weights]}")
    else:
        print(f"\nUsing provided dataset weights: {dataset_weights}")
    
    # Build sample weights for the merged dataset
    sample_weights = []
    for dataset_idx, train_dataset in enumerate(train_datasets_list):
        # Get task labels within this dataset
        task_labels = train_dataset.sample_task_labels
        task_counts = Counter(task_labels)
        
        # Compute per-task weights within this dataset
        task_weights = {task: 1.0 / count for task, count in task_counts.items()}
        
        # Apply dataset-level weight
        dataset_weight = dataset_weights[dataset_idx]
        
        # Combine: dataset weight * task weight
        for task_label in task_labels:
            combined_weight = dataset_weight * task_weights[task_label]
            sample_weights.append(combined_weight)
        
        print(f"Dataset {dataset_idx+1} task counts: {task_counts}")
    
    # Normalize weights
    sample_weights = np.array(sample_weights)
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    # Create sampler
    sampler = WeightedRandomSampler(
        sample_weights.tolist(),
        num_samples=len(merged_train_dataset),
        replacement=True
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        merged_train_dataset,
        batch_size=batch_size_train,
        sampler=sampler,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=False
    )
    
    if merged_val_dataset is not None:
        val_dataloader = DataLoader(
            merged_val_dataset,
            batch_size=batch_size_val,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=2,
            persistent_workers=False
        )
    else:
        val_dataloader = None
    
    # Return norm_stats as None (each dataset may have different stats)
    # is_sim from first dataset
    norm_stats = None
    is_sim = train_datasets_list[0].is_sim
    
    print(f'\n=== Multi-dataset loading complete ===\n')
    
    return train_dataloader, val_dataloader, norm_stats, is_sim


def load_mid_level_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    task_config,
    chunk_size,
    use_language=False):
    
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 1
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    # val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    # norm_stats = get_norm_stats(dataset_dir, num_episodes)
    norm_stats = None
    action_mode = task_config['action_mode'][0]
    dataset_path = task_config['dataset_dir']
    tissue_samples_ids = task_config['tissue_samples_ids']
    camera_file_suffixes = task_config['camera_file_suffixes']

    print("\n-------------loading training data-------------\n")

    train_datasets = EpisodicDatasetMidLevel(
            train_indices,
            tissue_samples_ids,
            dataset_dir,
            camera_names,
            camera_file_suffixes, 
            task_config,
            chunk_size=chunk_size,
            use_language=use_language,
        )


    train_dataloader = DataLoader(train_datasets, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=0)

    return train_dataloader, norm_stats, train_datasets.is_sim

### Merge multiple datasets
def load_merged_data(
    dataset_dirs,
    num_episodes_list,
    camera_names,
    batch_size_train,
    max_len=None,
    command_list=None,
    use_language=False,
    language_encoder=None,
    dagger_ratio=None,
    policy_class=None,
):
    assert len(dataset_dirs) == len(
        num_episodes_list
    ), "Length of dataset_dirs and num_episodes_list must be the same."
    if dagger_ratio is not None:
        assert 0 <= dagger_ratio <= 1, "dagger_ratio must be between 0 and 1."

    all_filtered_indices = []
    last_dataset_indices = []

    for i, (dataset_dir, num_episodes) in enumerate(
        zip(dataset_dirs, num_episodes_list)
    ):
        print(f"\nData from: {dataset_dir}\n")

        # Filter episodes based on command list if provided
        filtered_indices = []

        if len(command_list) > 0:
            cleaned_commands = [cmd.strip("'\"") for cmd in command_list]

            for episode_id in range(num_episodes):
                json_path = os.path.join(dataset_dir, f"episode_{episode_id}.json")
                with open(json_path, "r") as f:
                    instruction_data = json.load(f)

                # Check for valid command segments
                for segment in instruction_data:
                    if segment["command"] in cleaned_commands:
                        current_idx = instruction_data.index(segment)
                        if (
                            current_idx + 1 < len(instruction_data)
                            and instruction_data[current_idx + 1]["type"]
                            == "correction"
                        ):
                            continue
                        else:
                            filtered_indices.append((dataset_dir, episode_id))
                            break
        else:
            filtered_indices = [(dataset_dir, i) for i in range(num_episodes)]

        if i == len(dataset_dirs) - 1:  # Last dataset
            last_dataset_indices.extend(filtered_indices)
        all_filtered_indices.extend(filtered_indices)

    print(f"Total number of episodes across datasets: {len(all_filtered_indices)}")

    # Obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dirs, num_episodes_list)

    # Construct dataset and dataloader for each dataset dir and merge them
    train_datasets = [
        EpisodicDataset(
            [idx for d, idx in all_filtered_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            norm_stats,
            max_len,
            command_list,
            use_language,
            language_encoder,
            policy_class,
        )
        for dataset_dir in dataset_dirs
    ]
    merged_train_dataset = ConcatDataset(train_datasets)

    if dagger_ratio is not None:
        dataset_sizes = {
            dataset_dir: num_episodes
            for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list)
        }
        dagger_sampler = DAggerSampler(
            all_filtered_indices,
            last_dataset_indices,
            batch_size_train,
            dagger_ratio,
            dataset_sizes,
        )
        train_dataloader = DataLoader(
            merged_train_dataset,
            batch_sampler=dagger_sampler,
            pin_memory=True,
            num_workers=24,
            prefetch_factor=4,
            persistent_workers=True,
        )
    else:
        # Use default shuffling if dagger_ratio is not provided
        train_dataloader = DataLoader(
            merged_train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            pin_memory=True,
            num_workers=24,
            prefetch_factor=4,
            persistent_workers=True,
        )

    return train_dataloader, norm_stats, train_datasets[-1].is_sim


### For DAgger
class DAggerSampler(Sampler):
    def __init__(
        self, all_indices, last_dataset_indices, batch_size, dagger_ratio, dataset_sizes
    ):
        self.other_indices, self.last_dataset_indices = self._flatten_indices(
            all_indices, last_dataset_indices, dataset_sizes
        )
        print(
            f"Len of data from the last dataset: {len(self.last_dataset_indices)}, Len of data from other datasets: {len(self.other_indices)}"
        )
        self.batch_size = batch_size
        self.dagger_ratio = dagger_ratio
        self.num_batches = len(all_indices) // self.batch_size

    @staticmethod
    def _flatten_indices(all_indices, last_dataset_indices, dataset_sizes):
        flat_other_indices = []
        flat_last_dataset_indices = []
        cumulative_size = 0

        for dataset_dir, size in dataset_sizes.items():
            for idx in range(size):
                if (dataset_dir, idx) in last_dataset_indices:
                    flat_last_dataset_indices.append(cumulative_size + idx)
                elif (dataset_dir, idx) in all_indices:
                    flat_other_indices.append(cumulative_size + idx)
            cumulative_size += size

        return flat_other_indices, flat_last_dataset_indices

    def __iter__(self):
        num_samples_last = int(self.batch_size * self.dagger_ratio)
        num_samples_other = self.batch_size - num_samples_last

        for _ in range(self.num_batches):
            batch_indices = []

            if num_samples_last > 0 and self.last_dataset_indices:
                batch_indices.extend(
                    np.random.choice(
                        self.last_dataset_indices, num_samples_last, replace=True
                    )
                )

            if num_samples_other > 0 and self.other_indices:
                batch_indices.extend(
                    np.random.choice(
                        self.other_indices, num_samples_other, replace=True
                    )
                )

            np.random.shuffle(batch_indices)  # shuffle within each batch
            yield batch_indices

    def __len__(self):
        return self.num_batches


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach().cpu()
    return new_d


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def number_to_one_hot(number, size=501):
    one_hot_array = np.zeros(size)
    one_hot_array[number] = 1
    return one_hot_array
