from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform, util
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib
matplotlib.use('Agg')  # or 'Qt5Agg' if installed
import matplotlib.pyplot as plt
import time
import copy
import torch.nn.functional as F
import tqdm
from mpl_toolkits.mplot3d import Axes3D
import albumentations as abm
import cv2
import torchvision.transforms.functional as TF
# from torchvision.transforms import v2
# from torch.utils.tensorboard import SummaryWriter
import natsort
from natsort import natsorted
import scipy
import copy
from einops import rearrange

import warnings; warnings.simplefilter('ignore')
import random 
import shutil
from PIL import Image

import csv

def create_offset_map_with_gradient(image_shape, insert_point, exit_point, normalize_size=224.0, device='cpu', eps=1e-6):
    """
    Returns a 3-channel offset map:
      - Channel 0: dx to insertion point
      - Channel 1: dy to insertion point
      - Channel 2: scalar heatmap (1 at insertion, 0 at exit)

    Args:
        image_shape: (H, W)
        insert_point: (x, y)
        exit_point: (x, y)
        normalize_size: reference image size for normalization
        device: 'cpu' or 'cuda'
    """
    H, W = image_shape
    normalizing_constant = 250.0 * (min(H, W) / normalize_size)

    y_coords = torch.arange(H, device=device)
    x_coords = torch.arange(W, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Offsets to insertion point (dy, dx)
    dx = (x_grid - insert_point[0]) / normalizing_constant
    dy = (y_grid - insert_point[1]) / normalizing_constant

    # Gradient heatmap: insertion → 1.0, exit → 0.0
    d_insert = torch.sqrt((x_grid - insert_point[0]) ** 2 + (y_grid - insert_point[1]) ** 2)
    d_exit = torch.sqrt((x_grid - exit_point[0]) ** 2 + (y_grid - exit_point[1]) ** 2)
    heat = d_exit / (d_insert + d_exit + eps)  # in [0, 1]

    # Stack to shape (3, H, W)
    offset_map = torch.stack([dx, dy, heat], dim=0)
    return offset_map.clamp(-1.0, 1.0)  # Optional clamp

def offset_map_to_rgb_visual(offset_map):
    """
    Converts a (3, H, W) offset map (dx, dy, heat) to a uint8 RGB image for visualization.
    - Red = dx
    - Green = dy
    - Blue = heat
    """
    if torch.is_tensor(offset_map):
        offset_map = offset_map.detach().cpu().numpy()

    # Normalize each channel to [0, 1]
    def normalize(x):
        x = x - np.min(x)
        x = x / (np.max(x) + 1e-6)
        return x

    dx_norm = normalize(offset_map[0])
    dy_norm = normalize(offset_map[1])
    heat_norm = normalize(offset_map[2])

    rgb_image = np.stack([
        dx_norm,     # R
        dy_norm,     # G
        heat_norm    # B
    ], axis=-1)  # (H, W, 3)

    rgb_uint8 = (rgb_image * 255).astype(np.uint8)
    return rgb_uint8

def save_mask(image, img_path, clicked_points):
    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    if image_cv.ndim == 3 and image_cv.shape[2] == 3:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    height, width = image_cv.shape[:2]

    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Mark the clicked points in the mask
    for x, y in clicked_points:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(mask, (cx, cy), radius=5, color=255, thickness=-1)

    # Save binary mask
    cv2.imwrite(os.path.join(img_path, "clicked_point_mask.png"), mask)

    # # Generate heatmap from mask (apply Gaussian blur + colormap)
    # heatmap = cv2.GaussianBlur(mask, (0, 0), sigmaX=15, sigmaY=15)
    # heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # # Optional: blend with original image for visualization
    # heatmap_overlay = cv2.addWeighted(image_cv, 0.6, heatmap_colored, 0.4, 0)

    # # Save heatmap
    # cv2.imwrite(os.path.join(img_path, "clicked_point_heatmap.png"), heatmap_overlay)

    print("Saved binary mask and heatmap to", img_path)

def draw_heat_map(image, clicked_points):
    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    if image_cv.ndim == 3 and image_cv.shape[2] == 3:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    height, width = image_cv.shape[:2]

    # Create a distance map initialized to a large value
    distance_map = np.full((height, width), np.inf, dtype=np.float32)

    # Compute the distance of each pixel to the nearest clicked point
    for x, y in clicked_points:
        cx, cy = int(round(x)), int(round(y))
        y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        distances = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
        distance_map = np.minimum(distance_map, distances)

    # Normalize the distance map to [0, 255] for visualization
    normalized_map = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX)
    normalized_map = normalized_map.astype(np.uint8)

    # Apply a colormap to the normalized distance map
    heatmap_colored = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)

    # Optional: blend with original image for visualization
    heatmap_overlay = cv2.addWeighted(image_cv, 0.6, heatmap_colored, 0.4, 0)

    return heatmap_overlay

def draw_heat_map_new(image, clicked_points):
    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    if image_cv.ndim == 3 and image_cv.shape[2] == 3:
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    height, width = image_cv.shape[:2]

    # Create separate distance maps for x and y axes
    x_distance_map = np.zeros((height, width), dtype=np.float32)
    y_distance_map = np.zeros((height, width), dtype=np.float32)

    # Compute the distance of each pixel to the clicked points along x and y axes
    for x, y in clicked_points:
        cx, cy = int(round(x)), int(round(y))
        y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        x_distances = np.abs(x_indices - cx)
        y_distances = np.abs(y_indices - cy)
        x_distance_map = np.maximum(x_distance_map, x_distances)
        y_distance_map = np.maximum(y_distance_map, y_distances)

    # Normalize the distance maps to [0, 255] for visualization
    x_normalized_map = cv2.normalize(x_distance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    y_normalized_map = cv2.normalize(y_distance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create a 3-channel heatmap
    heatmap = np.zeros((height, width, 3), dtype=np.uint8)
    heatmap[..., 0] = x_normalized_map  # Red channel for x-axis distances
    heatmap[..., 1] = y_normalized_map  # Green channel for y-axis distances
    heatmap[..., 2] = 0  # Blue channel is black

    # Optional: blend with original image for visualization
    # heatmap_overlay = cv2.addWeighted(image_cv, 0.6, heatmap, 0.4, 0)

    return heatmap


## ----------------- Main ----------------- ##

data_dir = os.getenv('PATH_TO_DATASET')
tissue_ids = [1,2,3,4,5,6,7,8,9]     ## change this according to your dataset

crop_coords = []

for tissue_id in tissue_ids:
    ## calculate time taken for each tissue
    tissue_start_t = time.time()
    root = os.path.join(data_dir, f"tissue_{tissue_id}")
    dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
    dirlist = natsorted(dirlist)
    for dir in dirlist:
        phase_start_t = time.time()
        # if dir.startswith("3"):
        phase = os.path.join(root, dir)
        # print("Processing", dir)
        if dir.startswith("2_needle_throw"):
            for item in os.listdir(phase):
                img_dir = os.path.join(root, phase, item)
                # img_dir = data_dir + "/tissue_5/4_clipping_second_clip_left_tube/20240710-184558-875384"
                print(img_dir)

                ## if the labelled images are already present and is not empty, skip the processing
                ## check if the txt file is present
                if os.path.exists(os.path.join(img_dir, "clicked_point.csv")):
                    print("clicked points already present: ", os.path.join(img_dir, "clicked_point.csv"))
                    img_path = os.path.join(img_dir, "left_img_dir")
                    images = [item for item in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, item))]
                    images = natsorted(images)

                    ## show the first, the middle and the last image
                    for i in range(3):
                        if i == 0:
                            last_image_path = images[len(images)//2 - 150]
                        elif i == 1:
                            last_image_path = images[len(images)//2 - 50]
                        else:
                            last_image_path = images[-1]                        
                        # Show the current frame
                        plt.figure(figsize=(12, 8))
                        print("reading", os.path.join(img_path, last_image_path))
                        image = Image.open(os.path.join(img_path, last_image_path))
                        plt.imshow(image)
                        ## plot the clicked points
                        clicked = pd.read_csv(os.path.join(img_dir, "clicked_point.csv"))
                        insert_x = int(clicked.iloc[0, 0])
                        insert_y = int(clicked.iloc[0, 1])
                        exit_x = int(clicked.iloc[1, 0])
                        exit_y = int(clicked.iloc[1, 1])
                        insert_point = (insert_x, insert_y)
                        exit_point = (exit_x, exit_y)
                        w, h = image.size

                        # Create offset map
                        offset_map = create_offset_map_with_gradient(
                            image_shape=(h, w),
                            insert_point=insert_point,
                            exit_point=exit_point,
                            device='cpu'
                        )

                        rgb_offset_viz = offset_map_to_rgb_visual(offset_map)

                        clicked_points_mask = np.zeros((h, w, 3), dtype=np.uint8)

                        # Draw insertion point (first point) as red
                        insert_x, insert_y = int(clicked.iloc[0]['x']), int(clicked.iloc[0]['y'])
                        cv2.circle(clicked_points_mask, (insert_x, insert_y), radius=10, color=(0, 0, 255), thickness=-1)  # Red in BGR

                        # Draw exit point (second point) as green
                        exit_x, exit_y = int(clicked.iloc[1]['x']), int(clicked.iloc[1]['y'])
                        cv2.circle(clicked_points_mask, (exit_x, exit_y), radius=10, color=(0, 255, 0), thickness=-1)  # Green in BGR

                        ## overlay the mask on the image
                        # Only blend where the mask has non-zero content
                        image_np = np.array(image)

                        nonzero_mask = np.any(clicked_points_mask != 0, axis=-1)
                        overlay = image_np.copy()
                        overlay[nonzero_mask] = cv2.addWeighted(
                            image_np, 0.5, clicked_points_mask, 0.5, 0
                        )[nonzero_mask]
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                        rgb_image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                        clicked_points_mask = cv2.cvtColor(clicked_points_mask, cv2.COLOR_BGR2RGB)


                        cv2.imwrite(f'./distance_map.png', rgb_offset_viz)
                        cv2.imwrite(f'./dot.png', rgb_image)
                        cv2.imwrite(f'./mask.png', clicked_points_mask)
                        cv2.imwrite(f'./ori.png', image_np)

                        input("continue")

                else:
                    # Get the last image    
                    img_path = os.path.join(img_dir, "left_img_dir")
                    images = [item for item in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, item))]
                    images = natsorted(images)

                    last_image_path = images[-1]
                    # img = cv2.imread(os.path.join(img_path, last_image_path))

                    # Variable to store click position
                    clicked_points = []
                    # Show the current frame
                    plt.figure(figsize=(12, 8))
                    print("reading", os.path.join(img_path, last_image_path))
                    image = Image.open(os.path.join(img_path, last_image_path))
                    plt.imshow(image)

                    # Get the points from the user
                    clicked_points = plt.ginput(n=2, timeout=0, show_clicks=True) 
                    plt.close()


                    print("Click on the image to select a point. Press 'q' to quit and save.")

                    if len(clicked_points) == 2:
                        save_path = os.path.join(img_dir, "clicked_point.csv")
                        with open(save_path, "w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(["x", "y"])  # Optional: header
                            for point in clicked_points:
                                writer.writerow([f"{point[0]:.2f}", f"{point[1]:.2f}"])
                        print(f"Saved 2 points to: {save_path}")
                    else:
                        print("Expected 2 points, but got:", len(clicked_points))

                    save_mask(image, img_dir, clicked_points)
                    # input("Press Enter to continue...")

            phase_time_taken = time.time() - phase_start_t
            print(f"Time taken for phase {dir}: {phase_time_taken} seconds")

    tissue_time_taken = time.time() - tissue_start_t
    print(f"Time taken for tissue {tissue_id}: {tissue_time_taken} seconds")
                


