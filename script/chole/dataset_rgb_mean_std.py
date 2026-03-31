import os
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import time
import json
from torchvision import transforms

def compute_mean_std_gpu(subdatasets, camera_type, batch_size=32, downsampling_resolution=(224, 224), image_step_size=1, output_json="mean_std.json", wrist_images_rel_width=0.75):
    """
    Compute mean and standard deviation for images in the given subdatasets and camera type using GPU with batch processing.
    
    Args:
        subdatasets (list of str): List of paths to subdataset folders.
        camera_type (str): Camera type folder (e.g., "left_img_dir", "endo_psm2").
        batch_size (int): Number of images to process in a batch.
        downsampling_resolution (tuple): Resolution to downsample the images to.
        image_step_size (int): Interval for selecting every nth image.
        output_json (str): Path to the output JSON file to store mean and std values.
    
    Returns:
        tuple: Mean and standard deviation of the dataset.
    """
    
    valid_camera_folders = ["endo_psm2", "left_img_dir", "right_img_dir", "endo_psm1"]
    if camera_type not in valid_camera_folders:
        raise ValueError(f"Invalid camera type. Valid options are: {valid_camera_folders}")
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    resize_transform = transforms.Resize(downsampling_resolution, antialias=True)
    
    # Load or create JSON structure for results
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            tissue_stats = json.load(f)
    else:
        tissue_stats = {}

    # Loop through each subdataset
    for subdataset in subdatasets:
        # Loop through tissue folders
        tissue_folders = [tissue_folder for tissue_folder in os.listdir(subdataset) if os.path.isdir(os.path.join(subdataset, tissue_folder)) and len(tissue_folder.split('_')) == 2 and tissue_folder.split('_')[1].isdigit()]
        tissue_folders_sorted = sorted(tissue_folders, key=lambda x: int(x.split('_')[1]))
        for tissue_folder in tissue_folders_sorted:
            tissue_path = os.path.join(subdataset, tissue_folder)
            tissue_start_time = time.time()

            pixel_sum = torch.zeros(3, dtype=torch.float32, device=device)
            pixel_sum_sq = torch.zeros(3, dtype=torch.float32, device=device)
            tissue_total_pixels = 0

            # Skip processing if tissue is already in the JSON file
            if tissue_folder in tissue_stats:
                print(f"Skipping {tissue_folder}, already processed with mean: {tissue_stats[tissue_folder]['mean']}, std: {tissue_stats[tissue_folder]['std']}")
                continue

            # Loop through phase folders
            phase_folders = [phase_folder for phase_folder in os.listdir(tissue_path) if os.path.isdir(os.path.join(tissue_path, phase_folder)) and phase_folder.split('_')[0].isdigit()]
            phase_folders_sorted = sorted(phase_folders, key=lambda x: int(x.split('_')[0]))
            for phase_folder in phase_folders_sorted:
                phase_path = os.path.join(tissue_path, phase_folder)

                print(f"Processing phase folder: {phase_folder} ({tissue_folder}, {subdataset})")

                # Loop through demo folders
                demo_folders = [demo_folder for demo_folder in os.listdir(phase_path) if os.path.isdir(os.path.join(phase_path, demo_folder)) and demo_folder[8] == "-"]
                for demo_folder in demo_folders:
                    demo_path = os.path.join(phase_path, demo_folder)

                    # Get the camera folder
                    camera_folder = os.path.join(demo_path, camera_type)
                    if not os.path.exists(camera_folder):
                        continue

                    # Collect every nth image path from the camera folder
                    image_names_sorted = sorted(os.listdir(camera_folder))
                    image_paths = [os.path.join(camera_folder, image_name) for image_name in image_names_sorted][::image_step_size]

                    # Process images in batches
                    for frame_idx in range(0, len(image_paths), batch_size):
                        batch_image_paths = image_paths[frame_idx:frame_idx + batch_size]
                        batch_images = []

                        # Load batch of images
                        for image_path in batch_image_paths:
                            img = Image.open(image_path).convert('RGB')
                            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)  # Convert to CHW format
                            # Crop wrist image
                            if wrist_images_rel_width and wrist_images_rel_width > 0 and wrist_images_rel_width < 1.0:
                                if camera_type == "endo_psm2":
                                    split_idx = int(img_tensor.shape[2] * wrist_images_rel_width)
                                    img_tensor = img_tensor[:, :, -split_idx:]
                                elif camera_type == "endo_psm1":
                                    split_idx = int(img_tensor.shape[2] * wrist_images_rel_width)
                                    img_tensor = img_tensor[:, :, :split_idx]
                            # Resize image
                            if downsampling_resolution:
                                img_tensor = resize_transform(img_tensor)
                            img_tensor = img_tensor.to(dtype=torch.float32).to(device) / 255.0
                            batch_images.append(img_tensor)

                        # Stack batch images for efficient processing
                        batch_images = torch.stack(batch_images)

                        # Accumulate pixel sums and squared sums for the tissue
                        pixel_sum += torch.sum(batch_images, dim=(0, 2, 3))  # Sum over height and width
                        pixel_sum_sq += torch.sum(batch_images ** 2, dim=(0, 2, 3))
                        tissue_total_pixels += batch_images.shape[0] * batch_images.shape[2] * batch_images.shape[3]  # Batch size * Height * Width

            # Compute tissue-level mean and std
            tissue_mean = pixel_sum / tissue_total_pixels
            tissue_variance = (pixel_sum_sq / tissue_total_pixels) - tissue_mean ** 2
            tissue_std = torch.sqrt(tissue_variance)

            # Store tissue mean and std in JSON structure
            tissue_stats[tissue_folder] = {
                "mean": tissue_mean.cpu().tolist(),
                "std": tissue_std.cpu().tolist()
            }

            # Write tissue mean and std to JSON file immediately
            with open(output_json, "w") as f:
                json.dump(tissue_stats, f, indent=4)

            tissue_end_time = time.time()
            time_elapsed = tissue_end_time - tissue_start_time
            print(f"Finished processing tissue folder: {tissue_folder} ({subdataset}) in {time_elapsed:.2f} seconds.")
            print(f"Tissue {tissue_folder} Mean: {tissue_mean.cpu().numpy()}, Std: {tissue_std.cpu().numpy()}")

    # Calculate dataset-wide mean and standard deviation - based on the tissue means and variances (as we gonna sample each tissue equally and otherwise tissues with more data weight more in the total dataset stats)
    tissue_means = torch.stack([torch.tensor(tissue_data["mean"], dtype=torch.float32, device=device) for tissue_data in tissue_stats.values()])
    all_tissues_mean = torch.mean(tissue_means, dim=0)
    tissue_stds = torch.stack([torch.tensor(tissue_data["std"], dtype=torch.float32, device=device) for tissue_data in tissue_stats.values()])
    tissue_variances = tissue_stds ** 2
    mean_variance = torch.mean(tissue_variances, dim=0)
    all_tissues_std = torch.sqrt(mean_variance)

    # Add the total dataset stats to the JSON structure
    tissue_stats["total_dataset"] = {
        "mean": all_tissues_mean.cpu().tolist(),
        "std": all_tissues_std.cpu().tolist()
    }

    # Write final total dataset stats to the JSON file
    with open(output_json, "w") as f:
        json.dump(tissue_stats, f, indent=4)

    print(f"\nDataset Mean: {all_tissues_mean.cpu().numpy()}, Std: {all_tissues_std.cpu().numpy()}")
    return all_tissues_mean.cpu().numpy(), all_tissues_std.cpu().numpy()

# Example usage
dataset_names = ["base_chole_clipping_cutting", "base_chole_clipping_cutting_amos"]  # List of subdataset folders
path_to_datasets = os.getenv("PATH_TO_DATASET")
dataset_paths = [os.path.join(path_to_datasets, name) for name in dataset_names]
camera_type = "endo_psm2"  # Specify the camera type - ["endo_psm2", "left_img_dir", "right_img_dir", "endo_psm1"]
downsampling_resolution = (224, 224)  # Specify the downsampling resolution
image_step_size = 5 # Take every nth image
wrist_images_rel_width = 0.75  # Relative width of wrist images
if camera_type in ["endo_psm2", "endo_psm1"]:
    output_json_name = f"dataset_mean_std_{camera_type=}_{image_step_size=}_{wrist_images_rel_width=}.json"  # Output JSON file
else:
    output_json_name = f"dataset_mean_std_{camera_type=}_{image_step_size=}.json"  # Output JSON file
instructor_folder_path = os.path.join(os.getenv('PATH_TO_YAY_ROBOT'), "src", "instructor")
output_json_path = os.path.join(instructor_folder_path, output_json_name)
mean, std = compute_mean_std_gpu(dataset_paths, camera_type, batch_size=32, downsampling_resolution=downsampling_resolution, image_step_size=image_step_size, output_json=output_json_path, wrist_images_rel_width=wrist_images_rel_width)

print(f"\nTotal Dataset Mean: {mean}")
print(f"Total Dataset Standard Deviation: {std}")
