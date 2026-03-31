import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# ----------------------- Sam2 Setup -----------------------

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor

sam2_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sam2_checkpoint = os.path.join(sam2_dir, "checkpoints", "sam2_hiera_large.pt")
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, "cuda")

# ----------------------- Util functions -----------------------

def show_mask(mask, ax, obj_id=None, color=None):
    if color is None:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def convert_bbox_to_yolo_format(crop_coords, img_width, img_height):
    """
    Converts bounding boxes to YOLO format.
    
    :param crop_coords: List of bounding boxes, where each bbox is a tuple (x, y, width, height).
    :param img_width: Width of the image.
    :param img_height: Height of the image.
    :return: List of labels in YOLO format.
    """
    labels = []
    for bbox in crop_coords:
        x, y, w, h, obj_id = bbox
        
        # Compute center coordinates
        x_center = x + w / 2
        y_center = y + h / 2
        
        # Normalize coordinates
        x_center /= img_width
        y_center /= img_height
        w /= img_width
        h /= img_height
        
        # Assume class 0 for all objects; modify if you have multiple classes
        class_id = 0 if obj_id == 1 else 1
        
        labels.append(f"{class_id} {x_center} {y_center} {w} {h}")
    
    return labels


def convert_contour_to_yolo_format(contour, image_width, image_height):
    # Flatten the contour array and normalize coordinates
    contour = contour.reshape(-1, 2)
    x_coords, y_coords = contour[:, 0], contour[:, 1]
    x_coords = x_coords / image_width
    y_coords = y_coords / image_height
    return list(zip(x_coords, y_coords))


def save_bb_labels_to_txt_file(labels, image_name, output_dir):
    """
    Saves labels to a text file.
    
    :param labels: List of labels in YOLO format.
    :param image_name: Name of the image file.
    :param output_dir: Directory to save the labels in.
    """
    if labels:
        # Create a txt file with the same name as the image file
        txt_file_name = os.path.splitext(image_name)[0] + '.txt'
        txt_file_path = os.path.join(output_dir, txt_file_name)
        
        with open(txt_file_path, 'w') as f:
            for label in labels:
                f.write(label + '\n')


def postprocess_mask(mask_image, min_size=500, kernel_size=(5, 5)):
    """
    Post-process the binary mask image by filling small holes and removing small objects.
    
    Args:
        mask_image (np.ndarray): The binary mask image (2D array) where the object is represented by non-zero pixels.
        min_size (int): The minimum size of connected components to keep. Smaller components will be removed.
        kernel_size (tuple): The size of the kernel used for morphological closing to fill small holes.
        
    Returns:
        np.ndarray: The processed binary mask image.
    """
    
    # Fill small holes using morphological closing
    kernel = np.ones(kernel_size, np.uint8)
    mask_filled = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel)

    # Remove small objects using connected component analysis
    num_labels, labels_im = cv2.connectedComponents(mask_filled)
    
    # Create an empty mask to hold the filtered result
    final_mask = np.zeros_like(mask_filled)
    
    # Iterate through the labeled components and keep only large enough components
    for label in range(1, num_labels):  # Skip the background label 0
        component = (labels_im == label)
        if np.sum(component) > min_size:
            final_mask[component] = 255            
    
    return final_mask

# ----------------------- Parameters + init frames to apply Sam2 on -----------------------

# Parameters
dataset_dir =  os.path.join(os.getenv("PATH_TO_DATASET"), "base_chole_clipping_cutting")
tissue_name = "tissue_8" # exclude tissue 1,5,23, and other tissues that are not complete
phase_idx = 1 
frame_stride = 5
take_existing_demos = True
postprocesing_flag = True # Set to True if SAM2 postprocessing fails
num_demos = 6 # Will be ignored when take_existing_demos is True 
ann_frame_idx = 0  # the frame index we interact with
cmap = plt.get_cmap("tab10") # Color map to use

# Set all input and output directories
home_dir = os.path.expanduser('~')
tissue_dir_path = os.path.join(dataset_dir, tissue_name)
phase_start_name = f"{phase_idx}_*[^recovery]" # Exclude recovery phases 
phase_dir_path = list(Path(tissue_dir_path).glob(phase_start_name))[0]
phase_name = phase_dir_path.name

phase_tissue_dataset_dir = os.path.join(home_dir, "yolo", "datasets", "clips_tubes_flaps_segmentations", tissue_name, phase_name)
input_images_dir = os.path.join(phase_tissue_dataset_dir, "images")
output_labels_dir = os.path.join(phase_tissue_dataset_dir, "labels")
output_seg_masks_dir = os.path.join(phase_tissue_dataset_dir, "seg_masks")
output_contours_on_images_dir = os.path.join(phase_tissue_dataset_dir, "contours_on_image")
for dir in [input_images_dir, output_labels_dir, output_seg_masks_dir, output_contours_on_images_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Add metadata file to the dataset directory (if it doesn't exist)
metadata_file_path = os.path.join(phase_tissue_dataset_dir, "metadata.json")
import json
if not os.path.exists(metadata_file_path):
    metadata = {
        "tissue_name": tissue_name,
        "phase_name": phase_name,
        "frame_stride": frame_stride,
    }
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f)
    existing_demos = []
else:
    with open(metadata_file_path, 'r') as f:
        metadata = json.load(f)
        existing_demos = metadata.get("existing_demos", [])

# ----------------- Copy together the demo frames to prepare the video_dir for Sam2 ------------------

frame_cnt = 0
if phase_idx % 2 == 1:
    reversed_flag = True # Do annotation in reverse order for going back (as more visible)
else:
    reversed_flag = False

# TODO: Incorporate reversed order
# TODO: Incorporate the merging of the demos
if not (take_existing_demos and existing_demos):
    all_demos = os.listdir(phase_dir_path)
    all_demos_w_existing_demos = [demo for demo in all_demos if demo not in existing_demos]
    selected_demos = np.random.choice(all_demos_w_existing_demos, num_demos-len(existing_demos), replace=False).tolist() # Choose n demos randomly (excluding the existing ones)
    for demo in selected_demos:
        demo_dir = os.path.join(phase_dir_path, demo, "left_img_dir")
        num_frames = len(os.listdir(demo_dir))
        for frame_idx in range(0, num_frames, frame_stride):
            frame_file_name = f"frame{str(frame_idx).zfill(6)}_left.jpg"
            frame_path = os.path.join(demo_dir, frame_file_name)
            new_frame_path = os.path.join(input_images_dir, f"{frame_cnt}.jpg")
            shutil.copy(frame_path, new_frame_path)
            frame_cnt += 1

    # Add the selecte demos to the metadata file
    with open(metadata_file_path, 'w') as f:
        demos_in_dataset = existing_demos + selected_demos
        metadata["existing_demos"] = demos_in_dataset
        json.dump(metadata, f)

# Sleep to allow the files to be copied
time.sleep(2)

# Get the frame names (sorted based on the frame file name which is just the frame index)
frame_names = sorted(os.listdir(input_images_dir), key=lambda x: int(x.split(".")[0]))

### -----------  Select points for each class to segment ----------------

# Initialize the Sam2 predictor
inference_state = predictor.init_state(video_path=input_images_dir)

# Mapping of classes to the phases in which they appear 
classes_in_which_phases_mapping = {
    # "clip_1_left": list(range(1,18)),
    # "clip_2_left": list(range(4,18)),
    # "clip_3_left": list(range(6,18)),
    # "clip_1_right": list(range(10,18)),
    # "clip_2_right": list(range(12,18)),
    # "clip_3_right": list(range(14,18)),
    "clip": list(range(1,18)),
    "left_tube": list(range(1,9)),
    "right_tube": list(range(1,17)),
    "flap": [1]
}
classes = list(classes_in_which_phases_mapping.keys())
phase_idx_to_class_name_mapping = defaultdict(list)
for class_name, phase_idxs in classes_in_which_phases_mapping.items():
    for tmp_phase_idx in phase_idxs:
        phase_idx_to_class_name_mapping[tmp_phase_idx].append(class_name)

# Note: Choose the points that belong to the class (as 1), don't belong (as 0) to the class
# --> Each value: (num_pos_points, num_neg_points), obj_id
class_names_labels_obj_id_dict = {
    # "clip_1_left": ([1,1,0,0,0,0], 0),
    # "clip_2_left": ([1,1,1,1,1,0,0,0,0,0,0], 1),
    # "clip_3_left": ([1,1,1,1], 2),
    # "clip_1_right": ([1,1,1,1], 3), 
    # "clip_2_right": ([1,1,1,1], 4),
    # "clip_3_right": ([1,1,1,1], 5),
    "clip": ((3, 4), 0),
    "left_tube": ((3, 4), 1),
    "right_tube": ((3, 4), 2),
    "flap": ((3, 4), 3)
}

classes_to_consider = phase_idx_to_class_name_mapping[phase_idx]
prompts = {}  # hold all the clicks we add for visualization
for class_name in classes_to_consider:
    labels, ann_obj_id = class_names_labels_obj_id_dict[class_name]

    # Show the current frame
    plt.figure(figsize=(12, 8))
    num_pos, num_neg = labels
    plt.title(f"Phase idx {phase_idx}: Set points for {class_name} with {num_pos} positive and {num_neg} negative points")
    image = Image.open(os.path.join(input_images_dir, frame_names[ann_frame_idx]))
    plt.imshow(image)

    # Get the points from the user
    input_points = plt.ginput(n=len(labels), timeout=0, show_clicks=True) 
    plt.close()
    
    # Save the points and labels for the current object in the prompts dictionary
    points = np.array(input_points)
    print(f"Selected points: {points}")
    labels_array = np.array([1]*labels[0]+[0]*labels[1], np.int32)
    prompts[ann_obj_id] = points, labels_array

    # `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels_array,
    )

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"Tissue {tissue_name} - phase {tmp_phase_idx} - frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(input_images_dir, frame_names[ann_frame_idx])))
for class_idx, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    class_color = np.array([*cmap(out_obj_id)[:3], 0.6])
    show_mask((out_mask_logits[class_idx] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id, color=class_color)
save_path = os.path.join(phase_tissue_dataset_dir, f"{tissue_name=}_{tmp_phase_idx=}_frame_{ann_frame_idx}.jpg")
plt.savefig(save_path)
plt.show()

# Wait for decision of continuing or not (based on the segmentation quality)
continue_flag = input("\nDo you want to continue? (y/n): ")
if continue_flag == "n":
    exit()

# ----------------------- Propagate the segmentation on the full batch -----------------------

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
count = 0
for out_frame_idx in range(0, len(frame_names)):
    frame_name = frame_names[out_frame_idx].split(".")[0]
    count += 1
    orginal_image = cv2.imread(os.path.join(input_images_dir, frame_names[out_frame_idx]))
    class_name = classes[out_obj_id] 
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        # Get the mask in correct format
        mask_image = out_mask.squeeze()
        mask_image = mask_image.astype(np.uint8) * 255
        h, w = mask_image.shape
        
        # Post processing - filling small holes, removing small objects, etc.
        if postprocesing_flag:
            mask_image = postprocess_mask(mask_image, min_size=500, kernel_size=(5, 5))

        # Save the segmentation mask to a file
        seg_mask_file_name = f"frame_{frame_name}_obj_{out_obj_id}.jpg"
        seg_mask_path = os.path.join(output_seg_masks_dir, seg_mask_file_name)
        cv2.imwrite(seg_mask_path, mask_image)

        # Find contours in the mask - for YOLO format
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Write YOLO format annotations to text file
        output_txt_path = os.path.join(output_labels_dir, f"{frame_name}.txt") 
        write_mode = 'w' if out_obj_id == 0 else 'a'
        with open(output_txt_path, write_mode) as file:
            for contour in contours:
                yolo_points = convert_contour_to_yolo_format(contour, w, h) 
                # Format the annotation for YOLO
                annotation = f"{out_obj_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in yolo_points)
                file.write(annotation + '\n')
                
        # Draw the contours on the original image - check if this works
        class_color = np.array(cmap(out_obj_id)[:3])  # Extract only RGB values
        class_color_bgr = tuple((class_color * 255).astype(int)[::-1].tolist())  # Convert to BGR and ensure it's a tuple
        cv2.drawContours(orginal_image, contours, -1, class_color_bgr, 2)
    contour_file_name = f"frame_{frame_name}_contours.jpg"
    cv2.imwrite(os.path.join(output_contours_on_images_dir, contour_file_name), orginal_image)

print("finished count:", count) 
