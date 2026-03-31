import argparse
import time
from datetime import datetime
import os
from collections import defaultdict, deque
import contextlib
import sys
import signal
import json

import cv2
import pandas as pd
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String, Bool
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import rospy
from sklearn.metrics import f1_score, accuracy_score

# Import the necessary modules from this package
PATH_TO_YAY_ROBOT = os.getenv('PATH_TO_YAY_ROBOT')
if PATH_TO_YAY_ROBOT:
    sys.path.append(os.path.join(PATH_TO_YAY_ROBOT, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")

from instructor.train_daVinci import build_instructor, log_confusion_matrix, log_combined_image
from instructor.constants_daVinci import DATASET_CONFIGS, INSTRUMENT_CLOSED_THRESHOLD
from instructor.dataset_daVinci import get_valid_demo_start_end_indices, SequenceDataset
from auto_label.auto_label import get_all_auto_labels_list

# Context manager to measure the execution time of a code block
@contextlib.contextmanager
def measure_execution_time(label, execution_times_dict):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times_dict[label].append(execution_time)

# Triggering smooth stopping of the instructor pipeline when pressing Ctrl+c
exit_flag = False
started_main_loop_flag = False
def signal_handler(sig, frame):
    if not started_main_loop_flag:
        sys.exit(0)
    else:
        global exit_flag
        print('\nStopping the instructor pipeline...\n')
        exit_flag = True
signal.signal(signal.SIGINT, signal_handler)

# -------------- ROS Subscriber Callbacks --------------

# Init the global variables for the camera images
image_left_seg_mask = image_left = image_right = image_psm1_wrist = image_psm2_wrist = psm1_jaw = psm2_jaw = user_correction_instruction = None
new_user_correction_flag = pause_robot_flag = False

ros_cv2_bridge = CvBridge() # Initialize the CvBridge

# Callback function for the left camera
def left_camera_callback(data):
    global image_left
    image_left = ros_cv2_bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')
    
# Callback function for the right camera
def right_camera_callback(data):
    global image_right
    image_right = ros_cv2_bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')
    
# Callback function for the PSM1 wrist camera
def psm1_wrist_camera_callback(data):
    global image_psm1_wrist
    image_psm1_wrist = ros_cv2_bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')
    
def psm2_wrist_camera_callback(data):
    global image_psm2_wrist
    image_psm2_wrist = ros_cv2_bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')

def left_camera_seg_mask_callback(data):
    global image_left_seg_mask
    image_left_seg_mask = ros_cv2_bridge.imgmsg_to_cv2(data, desired_encoding = 'passthrough')    

def psm1_jaw_callback(data):
    global psm1_jaw
    psm1_jaw = data.position[0]
    
def psm2_jaw_callback(data):
    global psm2_jaw
    psm2_jaw = data.position[0]
    
def pause_robot_callback(data):
    global pause_robot_flag
    pause_robot_flag = data.data
    
    if pause_robot_flag:
        print("Robot paused. Waiting for the robot to be unpaused...")
    else:
        print("Robot unpaused. Resuming the instructor pipeline...")
    
def user_correction_instruction_callback(data):
    global user_correction_instruction
    global new_user_correction_flag
    user_correction_instruction = data.data
    new_user_correction_flag = True
    
# --------------- Utils function ---------------

def create_random_chole_episode(dataset_dir, selected_tissue_sample, camera_names, camera_name_file_suffix_dict, image_dim, phase_to_instruction_mapping=None, wrist_camera_rel_width=None,
                                selected_multitasks=None, use_seg_masks_input_flag=False, seg_mask_objs=None, merge_seg_masks_flag=False, add_center_crop_view_flag=False, distance_from_border_y=0.1, 
                                distance_from_border_x=0.25, y_offset = - 0.1):
    # Put together the stitched episode based on randomly getting a tissue id and a random index for a demo for each phase. Then sample a random timestep and get the corresponding image sequence and command embedding
       
    print("\nGenerating a random episode sequence...\n")
       
    # Init the episode sequence and the ground truth instruction sequence
    episode_jaw_values_sequence = []
    episode_frame_sequence = []
    episode_gt_instruction_sequence = []
       
    # Check within the dataset directory for tissue characteristics (e.g., before and after phase offset)
    dataset_name = os.path.basename(dataset_dir)
    before_phase_offset = DATASET_CONFIGS[dataset_name]["before_phase_offset"]
    after_phase_offset = DATASET_CONFIGS[dataset_name]["after_phase_offset"]
    
    # Go through the phases in fixed order of execution
    tissue_sample_dir_path = os.path.join(dataset_dir, selected_tissue_sample)
    phases_folder_names = [file_name for file_name in os.listdir(tissue_sample_dir_path) if os.path.isdir(os.path.join(tissue_sample_dir_path, file_name))]
    phases_folder_names = [phase_folder_name for phase_folder_name in phases_folder_names if "recovery" not in phase_folder_name]
    sorted_phases = sorted(phases_folder_names, key=lambda x: int(x.split('_')[0]))
    phase_start_idx_dict = {}
    curr_episode_len = 0
    episode_multitask_gt_labels_dict = None
    for phase_folder_name in sorted_phases:
        # Select a random demo for the current phase
        files_in_phase_folder = os.listdir(os.path.join(dataset_dir, selected_tissue_sample, phase_folder_name))
        demos_folder_names = [demo_sample for demo_sample in files_in_phase_folder if demo_sample[8] == "-"]
        selected_phase_demo_folder_name = np.random.choice(demos_folder_names)
        
        # Load the start and end indices for the current demo as the valid range of the demo
        selected_demo_folder_path = os.path.join(dataset_dir, selected_tissue_sample, phase_folder_name, selected_phase_demo_folder_name)
        start_idx, end_idx, num_frames = get_valid_demo_start_end_indices(selected_demo_folder_path, before_phase_offset, after_phase_offset, use_kinematic_indices_flag=False)
        
        # Extract the phase command from the folder name (removing the phase idx and the "_" in between the words) 
        phase_instruction = " ".join(phase_folder_name.split("_")[1:]) 
        if phase_to_instruction_mapping:
            phase_instruction = phase_to_instruction_mapping[phase_folder_name]
        episode_gt_instruction_sequence += [phase_instruction]*num_frames # Add the instruction for the current demo
        
        # Store the start index for the current phase
        phase_start_idx_dict[phase_folder_name] = curr_episode_len
        curr_episode_len = curr_episode_len + num_frames
        
        # Get the jaw values for the current demo
        kinematics_csv_path = os.path.join(selected_demo_folder_path, 'ee_csv.csv')
        demo_kinematics = pd.read_csv(kinematics_csv_path)
        valid_jaw_values = torch.tensor(demo_kinematics.iloc[start_idx:end_idx + 1][["psm2_jaw", "psm1_jaw"]].values)
        episode_jaw_values_sequence.append(valid_jaw_values)
        
        # Append the frames of the selected demo
        for ts_demo_frame_idx in range(start_idx, end_idx + 1):
            all_cam_images = []
            for camera_name in camera_names:
                camera_file_suffix = camera_name_file_suffix_dict[camera_name]
                demo_folder_path = os.path.join(dataset_dir, selected_tissue_sample, phase_folder_name, selected_phase_demo_folder_name)
                camera_folder_name = os.path.join(demo_folder_path, camera_name)
                frame_path = os.path.join(camera_folder_name, f"frame{str(ts_demo_frame_idx).zfill(6)}{camera_file_suffix}")
                camera_frame = torch.tensor(cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                
                # Add segmentation mask input for the left wrist cam as additional input channel (if desired)
                if camera_name == "left_img_dir" and use_seg_masks_input_flag:
                    if args.disable_seg_masks_input_flag:
                        masked_seg_masks = torch.zeros(len(seg_mask_objs), camera_frame.shape[1], camera_frame.shape[2]).to(dtype=torch.uint8)
                        camera_frame = torch.cat([camera_frame, masked_seg_masks], dim=0)
                    else:
                        seg_masks_folder_path = os.path.join(demo_folder_path, "seg_masks")
                        camera_frame_and_seg_masks_list = [camera_frame]  
                        for seg_mask_obj in seg_mask_objs:    
                            seg_mask_path = os.path.join(seg_masks_folder_path, f"frame{str(ts_demo_frame_idx).zfill(6)}_{seg_mask_obj}.jpg")
                            if not os.path.exists(seg_mask_path):
                                seg_mask = torch.zeros(1, camera_frame.shape[1], camera_frame.shape[2]).to(dtype=torch.uint8)
                            else:
                                seg_mask = torch.tensor(cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)).unsqueeze(0) / 255
                            seg_mask = torch.where(seg_mask > 0.5, torch.tensor(1), torch.tensor(0)).to(dtype=torch.uint8) # Map between 0 and 1
                            camera_frame_and_seg_masks_list.append(seg_mask)
                        camera_frame = torch.cat(camera_frame_and_seg_masks_list, dim=0)
            
                # Add the image to the image dictionary
                if camera_name in ["left_img_dir", "right_img_dir"]:
                    # Add global image
                    global_img_resized = transforms.Resize(image_dim)(camera_frame)
                    all_cam_images.append(global_img_resized)
                    
                    if add_center_crop_view_flag: 
                        # Add center crop image
                        center_crop_y_min, center_crop_y_max = int(camera_frame.shape[1] * (distance_from_border_y - y_offset)), int(camera_frame.shape[1] * (1-distance_from_border_y-y_offset)) 
                        center_crop_x_min, center_crop_x_max = int(camera_frame.shape[2] * distance_from_border_x), int(camera_frame.shape[2] * (1-distance_from_border_x))
                        camera_frame_cropped = camera_frame[:, center_crop_y_min:center_crop_y_max, center_crop_x_min:center_crop_x_max]
                        center_part_of_img = transforms.Resize(image_dim)(camera_frame_cropped)
                        all_cam_images.append(center_part_of_img)                       
                elif camera_name in ["endo_psm2", "endo_psm1"]: 
                    # Resize the image and crop the wrist camera if needed
                    if wrist_camera_rel_width and wrist_camera_rel_width < 1:
                        split_idx = int(camera_frame.shape[2] * wrist_camera_rel_width)
                        if camera_name == "endo_psm2":
                            camera_frame = camera_frame[:, :, -split_idx:]
                        elif camera_name == "endo_psm1":
                            camera_frame = camera_frame[:, :, :split_idx]
                    frame_resized = transforms.Resize(image_dim)(camera_frame)
                    all_cam_images.append(frame_resized)
                    
            # Stack the camera frames together
            all_cam_images_tensor = torch.stack(all_cam_images, dim=0)
            episode_frame_sequence.append(all_cam_images_tensor)
            
            # Get the multitask ground truth labels for the current frame
            if selected_multitasks:
                curr_jaw_values = valid_jaw_values[ts_demo_frame_idx - start_idx]
                multitask_labels_indices_dict = SequenceDataset.get_multitask_labels_dict(selected_multitasks, ts_demo_frame_idx, phase_folder_name, start_idx, end_idx, curr_jaw_values, selected_tissue_sample,
                                                                                          selected_phase_demo_folder_name, ee_csv_data=demo_kinematics, ts_demo_frame_idx=ts_demo_frame_idx,
                                                                                          corrections_dict={})
                
                multitask_gt_labels_dict = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(multitask_labels_indices_dict)
                if not episode_multitask_gt_labels_dict:
                    episode_multitask_gt_labels_dict = {task_name: [task_label] for task_name, task_label in multitask_gt_labels_dict.items()}
                else:
                    for task_name, task_label in multitask_gt_labels_dict.items():
                        episode_multitask_gt_labels_dict[task_name].append(task_label)
    
    # Stack the episode frame sequence and the ground truth instruction sequence
    episode_frame_sequence_tensor = torch.stack(episode_frame_sequence, dim=0).to(dtype=torch.float32)
    episode_frame_sequence_tensor[:, :, :3, :, :] = episode_frame_sequence_tensor[:, :, :3, :, :] / 255.0
    
    # If desired, merge the segmentation masks for to just one additional channel
    if use_seg_masks_input_flag and merge_seg_masks_flag:
        num_ts, num_cam_patches, num_channels, h, w = episode_frame_sequence_tensor.shape
        merged_seg_mask = torch.zeros(num_ts, num_cam_patches, 1, h, w).to(dtype=torch.uint8)
        if not args.disable_seg_masks_input_flag:
            seg_mask_channels = episode_frame_sequence_tensor[:, :, 3:, :, :]
            for class_id in range(1, num_channels-3+1): 
                seg_mask = seg_mask_channels[:, :, class_id-1, :, :].unsqueeze(2) * class_id
                # Merge the segmentation masks based on the priority of the objects (lower the higher, except for 0)
                merged_seg_mask = torch.where((merged_seg_mask == 0) & (seg_mask != 0), seg_mask, merged_seg_mask)
                merged_seg_mask = torch.where((merged_seg_mask != 0) & (seg_mask != 0) & (seg_mask < merged_seg_mask), seg_mask, merged_seg_mask)
        episode_frame_sequence_tensor = torch.cat([episode_frame_sequence_tensor[:,:,:3], merged_seg_mask], dim=2)   
    
    episode_jaw_value_sequence_tensor = torch.concatenate(episode_jaw_values_sequence, dim=0).to(dtype=torch.float32)
    return episode_frame_sequence_tensor, episode_jaw_value_sequence_tensor, episode_gt_instruction_sequence, phase_start_idx_dict, episode_multitask_gt_labels_dict # Jaw values shape should be: num_frames, 2


def get_current_jaw_values(input_type, frame_idx=None, random_episode_jaw_value_sequence=None):
    # Based on the input type (live, random) get the current jaw values (+ if successful - indicating end of the episode for offline data)
    
    if input_type == "live":
        # Access the ROS subscribers and get the current jaw values
        if psm1_jaw is None or psm2_jaw is None:
            return False, None
        else:
            return True, torch.tensor([psm2_jaw, psm1_jaw])
    else:
        # Access the random generated episode jaw values
        if frame_idx >= len(random_episode_jaw_value_sequence):
            return False, None
        else:
            return True, random_episode_jaw_value_sequence[frame_idx]


def get_current_frames(args, image_dim, frame_idx=None, random_episode_frame_sequence=None, camera_names=None, wrist_camera_rel_width=None, 
                       use_seg_masks_input_flag=False, merge_seg_masks_flag=False, seg_mask_objs=None, video_capture=None, video_path=None,
                       add_center_crop_view_flag=False, distance_from_border_y=0.1, distance_from_border_x=0.25, y_offset = - 0.1):
    # Based on the input type (live, random) get the current frames stacked together
    
    if args.input_type == "live":
        # Access the ROS subscribers and get the current frames - do a copy of it + apply transformations - size + color
        camera_frames_dict = {}
        if "left_img_dir" in camera_names:
            camera_frames_dict["left_img_dir"] = image_left.copy()
        if "right_img_dir" in camera_names:
            camera_frames_dict["right_img_dir"] = image_right.copy()
        if "endo_psm1" in camera_names:
            camera_frames_dict["endo_psm1"] = image_psm1_wrist.copy()
        if "endo_psm2" in camera_names:
            camera_frames_dict["endo_psm2"] = image_psm2_wrist.copy()
        # Sort the camera frames based on the camera names
        camera_frames = [camera_frames_dict[camera_name] for camera_name in camera_names]
        
        # Apply transformations (resize, color, ..)
        current_frames = []
        for camera_frame, camera_name in zip(camera_frames, camera_names):
            if camera_frame is None:
                return False, None
            else:                
                # Add segmentation mask input for the left wrist cam as additional input channel (if desired)
                if not args.disable_seg_masks_input_flag and (camera_name == "left_img_dir" and use_seg_masks_input_flag):
                    if image_left_seg_mask is not None:
                        seg_mask = image_left_seg_mask.copy()
                        selected_seg_masks = []
                        # Extract all the segmentation masks for the different classes from the 1D segmentation mask
                        for seg_mask_class_name in seg_mask_objs:
                            # Assuming seg_mask_index_mapping is a dictionary mapping class names to their respective class IDs
                            class_id = args.seg_mask_index_mapping[seg_mask_class_name]
                            # Create a binary mask where the class ID matches
                            class_seg_mask = np.where(seg_mask == class_id, 1, 0).astype(np.uint8)
                            class_seg_mask = np.expand_dims(class_seg_mask, axis=2)
                            selected_seg_masks.append(class_seg_mask)
                        
                        # Stack the original image frame with the selected segmentation masks along the first dimension
                        camera_frame_seg_masks_list = [camera_frame] + selected_seg_masks
                        camera_frame = np.concatenate(camera_frame_seg_masks_list, axis=2)
                    else:
                        # Create a zero mask if no segmentation masks are available
                        masked_seg_masks = np.zeros((len(seg_mask_objs), camera_frame.shape[1], camera_frame.shape[2]), dtype=np.uint8)
                        camera_frame = np.concatenate([camera_frame, masked_seg_masks], axis=0)
                
                # Add the image to the image dictionary
                if camera_name in ["left_img_dir", "right_img_dir"]:
                    # Add global image
                    global_img_resized = cv2.resize(camera_frame, (image_dim[1], image_dim[0])) # As cv2 uses width x height 
                    current_frames.append(torch.tensor(global_img_resized))
                    
                    if add_center_crop_view_flag: 
                        # Add center crop image
                        center_crop_y_min, center_crop_y_max = int(camera_frame.shape[0] * (distance_from_border_y - y_offset)), int(camera_frame.shape[0] * (1-distance_from_border_y-y_offset)) 
                        center_crop_x_min, center_crop_x_max = int(camera_frame.shape[1] * distance_from_border_x), int(camera_frame.shape[1] * (1-distance_from_border_x))
                        camera_frame_cropped = camera_frame[center_crop_y_min:center_crop_y_max, center_crop_x_min:center_crop_x_max, :]
                        center_part_of_img = cv2.resize(camera_frame_cropped, (image_dim[1], image_dim[0])) # As cv2 uses width x height
                        current_frames.append(torch.tensor(center_part_of_img))                       
                elif camera_name in ["endo_psm2", "endo_psm1"]: 
                    # Resize the image and crop the wrist camera if needed
                    if wrist_camera_rel_width and wrist_camera_rel_width < 1:
                        split_idx = int(camera_frame.shape[1] * wrist_camera_rel_width)
                        if camera_name == "endo_psm2":
                            camera_frame = camera_frame[:, -split_idx:, :]
                        elif camera_name == "endo_psm1":
                            camera_frame = camera_frame[:, :split_idx, :]
                    camera_frame = cv2.resize(camera_frame, (image_dim[1] ,image_dim[0])) # As cv2 uses width x height
                    camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    current_frames.append(torch.tensor(camera_frame)) 
      
        # Transform to correct shape and format
        current_frames_transformed_tensor = torch.stack(current_frames, dim=0).permute(0, 3, 1, 2).to(torch.float32) # Shape: cam, c, h, w
        current_frames_transformed_tensor[:, :3, :, :] = current_frames_transformed_tensor[:, :3, :, :] / 255.0 # Shape: cam, c, h, w

        # If desired, merge the segmentation masks for to just one additional channel
        if use_seg_masks_input_flag and merge_seg_masks_flag:
            num_cam_patches, num_channels, h, w = current_frames_transformed_tensor.shape
            merged_seg_mask = torch.zeros(num_cam_patches, 1, h, w).to(dtype=torch.uint8)
            if not args.disable_seg_masks_input_flag:
                seg_mask_channels = current_frames_transformed_tensor[:, 3:, :, :]
                for class_id in range(1, num_channels-3+1): 
                    seg_mask = seg_mask_channels[:, class_id-1, :, :].unsqueeze(1) * class_id
                    # Merge the segmentation masks based on the priority of the objects (lower the higher, except for 0)
                    merged_seg_mask = torch.where((merged_seg_mask == 0) & (seg_mask != 0), seg_mask, merged_seg_mask)
                    merged_seg_mask = torch.where((merged_seg_mask != 0) & (seg_mask != 0) & (seg_mask < merged_seg_mask), seg_mask, merged_seg_mask)
            current_frames_transformed_tensor = torch.cat([current_frames_transformed_tensor[:,:3], merged_seg_mask], dim=1)      

        return True, current_frames_transformed_tensor # Shape: cam, c, h, w 
        
    elif args.input_type == "random":
        # Access the random generated episode frames
        if frame_idx >= len(random_episode_frame_sequence):
            return False, None
        else:
            current_frames = random_episode_frame_sequence[frame_idx]  # Shape: cam, c, h, w
            return True, current_frames
        
    elif args.input_type == "video":
        # Access the video capture and get the current frames
        ret, camera_frame = video_capture.read()
        if not ret:
            return False, None
        else:
            if use_seg_masks_input_flag:
                height, width = camera_frame.shape[:2]
                if 2*height != width:
                    raise ValueError("Currently only accepting square image dimensions for YOLO video mode evaluation.")
                
                # Split the camera frame into the camera frame and the segmentation masks and resize them
                seg_mask = camera_frame[:, height:]
                camera_frame = camera_frame[:, :height]
                seg_mask = cv2.resize(seg_mask, (image_dim[1], image_dim[0]))
                camera_frame = cv2.resize(camera_frame, (image_dim[1], image_dim[0]))
                camera_frame = torch.tensor(cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).to(torch.float32) / 255.0
                
                if merge_seg_masks_flag:
                    seg_mask = torch.tensor(seg_mask.transpose(2, 0, 1)).to(torch.float32) / 255.0
                    seg_mask = torch.where(seg_mask > 0.5, torch.tensor(1), torch.tensor(0)) # Map between 0 and 1
                    merged_seg_mask = torch.zeros((1, image_dim[0], image_dim[1])).to(torch.float32)
                    for class_id, class_seg_mask in enumerate(seg_mask, start=1): 
                        class_seg_mask_reshaped = class_seg_mask.unsqueeze(0) * class_id
                        # Merge the segmentation masks based on the priority of the objects (lower the higher, except for 0)
                        merged_seg_mask = torch.where((merged_seg_mask == 0) & (class_seg_mask_reshaped != 0), class_seg_mask_reshaped, merged_seg_mask)
                        merged_seg_mask = torch.where((merged_seg_mask != 0) & (class_seg_mask_reshaped != 0) & (class_seg_mask_reshaped < merged_seg_mask), class_seg_mask_reshaped, merged_seg_mask)
                    seg_mask = merged_seg_mask
                else:
                    seg_mask = torch.tensor(seg_mask.transpose(2, 0, 1)).to(torch.float32) / 255.0
                    seg_mask = torch.where(seg_mask > 0.5, torch.tensor(1), torch.tensor(0)) # Map between 0 and 1
                
                # Stack the camera frame and the segmentation masks    
                camera_frame_tensor = torch.cat([camera_frame, seg_mask], dim=0).unsqueeze(0) # Shape: cam, c, h, w
            else:
                if "yolo." in video_path:
                    # Crop the leftmost part of the frame to a square size (height x height)
                    height = camera_frame.shape[0]  
                    camera_frame = camera_frame[:, :height] 
                camera_frame = cv2.resize(camera_frame, (image_dim[1] ,image_dim[0])) # As cv2 uses width x height
                camera_frame_tensor = torch.tensor(cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).to(torch.float32).unsqueeze(0) / 255.0 # Shape: c, h, w
            return True, camera_frame_tensor # Shape: cam, c, h, w
    
    
def trigger_offline_phase_transition_waiting_scheme(curr_phase_idx, phases_idx_to_folder_name_dict, phase_start_idx_dict, episode_gt_instruction_sequence, phase_to_instruction_mapping=None):
    # Setting all the variables for the offline evaluation scheme (waiting for the end of the phase to predict the next phase)
    current_phase_name = phases_idx_to_folder_name_dict[curr_phase_idx]
    next_phase_name_to_predict = phases_idx_to_folder_name_dict[curr_phase_idx+1] if curr_phase_idx < list(phases_idx_to_folder_name_dict.keys())[-1] else None
    if next_phase_name_to_predict:
        next_phase_instruction_to_predict = phase_to_instruction_mapping[next_phase_name_to_predict] if phase_to_instruction_mapping else " ".join(next_phase_name_to_predict.split("_")[1:])
    else:
        next_phase_instruction_to_predict = None
    
    # If current phase is go to cut, already provide the first frame of the next phase as the end index (as end of cut does not provide visually that the tube is cut at the end of the demo)
    if "cutting" in current_phase_name:
        curr_phase_abs_end_idx = phase_start_idx_dict[next_phase_name_to_predict]
    elif next_phase_name_to_predict is None:
        curr_phase_abs_end_idx = len(episode_gt_instruction_sequence) - 1
    else:
        curr_phase_abs_end_idx = phase_start_idx_dict[next_phase_name_to_predict] - 1
    wait_counter = 0
    
    return curr_phase_abs_end_idx, next_phase_instruction_to_predict, wait_counter, current_phase_name, next_phase_name_to_predict


def apply_logits_lp_filter(last_n_logits, filter_mode="average", smoothing_factor=2):
    # Apply a low pass filter on the logits (e.g., average, ema (exponential moving average), ...)
    
    if filter_mode == "average":
        smoothed_logits = torch.mean(last_n_logits, dim=0).unsqueeze(0)
        return smoothed_logits
    elif filter_mode == "ema":
        # Determine the smoothing period based on the length of last_n_logits
        smoothing_period = len(last_n_logits)
        
        # Calculate alpha based on the smoothing period and smoothing factor
        alpha = smoothing_factor / (1 + smoothing_period)
        
        # Initialize EMA with the first logit value
        ema = last_n_logits[0]
        
        # Calculate EMA using the adjusted formula
        for t in range(1, smoothing_period):
            ema = (
                last_n_logits[t] * alpha +
                ema * (1 - alpha)
            )
        
        smoothed_logits = ema.unsqueeze(0)
        return smoothed_logits
    else:
        raise ValueError(f"Filter mode {filter_mode} is not implemented yet")


def apply_ensemble_mode(logits_list, multitask_logits_dict_list, ensemble_mode="average"):
    # Apply the ensemble mode on the logits (e.g., average, (majority) voting, ...)
        
    # TODO: Check if this works 
    if ensemble_mode == "average":
        # Average the logits
        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        avg_multitask_logits_dict = {}
        stacked_multitask_logits_dict = defaultdict(list)
        # Stack the multitask logits for all ensemble predictions
        for multitask_logits_dict in multitask_logits_dict_list:
            for task_name in multitask_logits_dict:
                stacked_multitask_logits_dict[task_name].append(multitask_logits_dict[task_name])
        # Average the stacked multitask logits
        for task_name in stacked_multitask_logits_dict:
            avg_multitask_logits_dict[task_name] = torch.mean(torch.stack(stacked_multitask_logits_dict[task_name], dim=0), dim=0)
        return avg_logits, avg_multitask_logits_dict
    elif ensemble_mode == "max":
        # Take the maximum value of the logits
        max_logits = torch.max(torch.stack(logits_list), dim=0).values
        max_multitask_logits_dict = {}
        for multitask_logits_dict in multitask_logits_dict_list:
            for task_name in multitask_logits_dict:
                max_multitask_logits_dict[task_name] = torch.max(torch.stack(multitask_logits_dict[task_name]), dim=0).values
        return max_logits, max_multitask_logits_dict
    elif ensemble_mode == "voting":
        # Take the majority vote by checking for the logits with the highest value and then average their logits to display the merged probability (if there is a tie, then average the tied logits)
        # TODO: Implement this
        raise NotImplementedError("Voting ensemble mode is not implemented yet")
    else:
        raise ValueError(f"Ensemble mode {ensemble_mode} is not implemented yet")
        
        


def visualize_current_frames(current_frames, language_instruction_prediction, predicted_instruction_prob=None, language_instruction_ground_truth=None, upscaling_factor=2, 
                             visualization_flag=True, current_jaw_values=None, phase_history=None, candidate_texts=None, multitask_preds=None, 
                             multitask_probs=None, multitask_gts=None, additional_width = 500, do_not_predict_flag=False, seg_mask_objs=["clips", "left_tube", "right_tube"], cmap=plt.get_cmap("tab10")):
    
    # Use the segmentation mask as another camera
    num_channels = current_frames.shape[1] 
    if num_channels > 3:
        additional_channels = num_channels - 3
        if additional_channels == 3:
            current_frames = torch.cat([current_frames[:, :3], current_frames[:, 3:]], dim=0)
        elif additional_channels == 1 and len(seg_mask_objs) == 3:
            seg_mask_list = []
            for class_id in range(1, len(seg_mask_objs)+1):
                seg_mask = (current_frames[:, 3] == class_id).float()
                seg_mask_list.append(seg_mask)
            seg_masks = torch.stack(seg_mask_list, dim=1)
            current_frames = torch.cat([current_frames[:, :3], seg_masks], dim=0)
        else:
            current_frames = current_frames[:, :3] # Note: Currently only possible to work with 3 seg masks (as currently only required)
    
    # Concatenate the frames together
    current_frames_concatenated = (torch.cat(list(current_frames), dim=2).detach().to("cpu").numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    
    # Convert to BGR and upscale
    current_frames_concatenated_bgr = cv2.cvtColor(current_frames_concatenated, cv2.COLOR_RGB2BGR)
    current_frames_concatenated_bgr_upscaled = cv2.resize(current_frames_concatenated_bgr, 
                                                          (int(current_frames_concatenated_bgr.shape[1] * upscaling_factor), 
                                                           int(current_frames_concatenated_bgr.shape[0] * upscaling_factor)))
    
    # Determine the new image dimensions with space on the right for additional text
    new_width = current_frames_concatenated_bgr_upscaled.shape[1] + additional_width
    new_height = current_frames_concatenated_bgr_upscaled.shape[0]
    
    # Create a new black image with additional space on the right
    annotated_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    annotated_image[:current_frames_concatenated_bgr_upscaled.shape[0], :current_frames_concatenated_bgr_upscaled.shape[1]] = current_frames_concatenated_bgr_upscaled
    
    # Font settings
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale, thickness = 0.7, 2
    
    # Define the starting positions for the text
    text_start_x = current_frames_concatenated_bgr_upscaled.shape[1] + 10  # Start drawing text in the black space
    text_start_y = 30
    line_spacing = 30
    
    # Add the predicted language instruction (and GT if available)
    if do_not_predict_flag:
        prediction_text = f"User correction: {language_instruction_prediction}"
    elif predicted_instruction_prob:
        prediction_text = f"Prediction: {language_instruction_prediction} ({predicted_instruction_prob*100:.1f}%)"
    else:
        prediction_text = f"Prediction: {language_instruction_prediction}"
    if language_instruction_ground_truth:
        prediction_text_color = (0, 255, 0) if language_instruction_prediction == language_instruction_ground_truth else (0, 0, 255)
    else:
        prediction_text_color = (255, 255, 255)
    cv2.putText(annotated_image, prediction_text, (text_start_x, text_start_y), font, font_scale, prediction_text_color, thickness, cv2.LINE_AA)
    text_start_y += line_spacing
    
    if language_instruction_ground_truth:
        gt_text = f"GT: {language_instruction_ground_truth}"
        if multitask_gts and "dominant_moving_direction" in multitask_gts: 
            gt_text = f"{gt_text} ({multitask_gts['dominant_moving_direction']})"
        cv2.putText(annotated_image, gt_text, (text_start_x, text_start_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        text_start_y += line_spacing*2  # Move down for next section
    else:
        text_start_y += line_spacing
    
    # Add jaw values
    if current_jaw_values is not None:
        jaw_text = f"Jaw Values: PSM2 = {current_jaw_values[0]:.2f}, PSM1 = {current_jaw_values[1]:.2f}"
        cv2.putText(annotated_image, jaw_text, (text_start_x, text_start_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        text_start_y += line_spacing*2
    
    # Add phase history
    if phase_history is not None:
        phase_history_text = f"Phase History:"
        cv2.putText(annotated_image, phase_history_text, (text_start_x, text_start_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        text_start_y += line_spacing
        for phase_idx in phase_history:
            if candidate_texts:
                phase_history_text = f"{candidate_texts[phase_idx-1]}" if phase_idx != 0 else None
            else:
                phase_history_text = f"{phase_idx}"
            if phase_history_text:
                cv2.putText(annotated_image, phase_history_text, (text_start_x, text_start_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                text_start_y += line_spacing
        text_start_y += line_spacing
    
    # Add multitask predictions and probabilities
    if not do_not_predict_flag and multitask_preds is not None and multitask_probs is not None:
        task_text = "Multitask Predictions:"
        cv2.putText(annotated_image, task_text, (text_start_x, text_start_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        text_start_y += line_spacing  # Add extra space before multitask section
        for task_name in multitask_preds:
            task_pred = multitask_preds[task_name][0]
            task_prob = multitask_probs[task_name]
            task_gt = multitask_gts[task_name] if multitask_gts else None
            
            # Determine the text color based on GT match
            if task_gt is not None:
                task_text_color = (0, 255, 0) if task_pred == task_gt else (0, 0, 255)
            else:
                task_text_color = (255, 255, 255)
                
            task_text = f"{task_name}: {task_pred} ({task_prob*100:.1f}%)" if task_prob is not None else f"{task_name}: {task_pred}"
            cv2.putText(annotated_image, task_text, (text_start_x, text_start_y), font, font_scale, task_text_color, thickness, cv2.LINE_AA)
            text_start_y += line_spacing
    
    # Show the annotated image
    if visualization_flag:
        cv2.imshow("HL Policy Prediction", annotated_image)
    
    return annotated_image  # Return the annotated frame for saving the video


def evaluate_instruction_prediction(predictions, ground_truths, all_phases, timestamp, save_folder_path):
    # Init metrics dict
    metrics_dict = {}
    
    # Compute the accuracy, f1 score, confusion matrix
    metrics_dict["accuracy"] = accuracy_score(ground_truths, predictions)
    metrics_dict["f1_score"] = f1_score(ground_truths, predictions, average="macro")
    
    # Save the confusion matrix function from the training script 
    save_path = os.path.join(save_folder_path, f"confusion_matrix_{timestamp}.png")
    log_confusion_matrix("phase", ground_truths, predictions, all_phases, save_path=save_path, log_wandb_flag=False)

    return metrics_dict


def evaluate_multitask_prediction(multitask_preds_eval_dict, multitask_gts_eval_dict):
    # Init multitask metrics dict
    multitask_metrics_dict = {}
    
    # Compute the accuracy, f1 score, confusion matrix for each multitask
    for task_name in multitask_preds_eval_dict:
        task_preds = multitask_preds_eval_dict[task_name]
        task_gts = multitask_gts_eval_dict[task_name]
        
        # Compute the accuracy, f1 score, confusion matrix
        multitask_metrics_dict[task_name] = {}
        multitask_metrics_dict[task_name]["accuracy"] = accuracy_score(task_gts, task_preds)
        multitask_metrics_dict[task_name]["f1_score"] = f1_score(task_gts, task_preds, average="macro")

    return multitask_metrics_dict

# ------------------------------------- Main function --------------------------------------

def instructor_pipeline(args):
    # ------------- Access the command line parameters -------------

    # Output the command line parameters
    print("\n----------------------- Command line parameters ----------------------------\n")
    for arg in vars(args):
        if args.input_type == "live" and (arg == "tissue_name" or arg == "dataset_dir"): # Skip the tissue name and dataset dir for live input
            continue
        print(f"{arg}: {getattr(args, arg)}")
    print("\n-----------------------------------------------------------------------------\n")
    if args.starting_phase_idx is None or args.starting_phase_idx < 1:
        args.starting_phase_idx = 1 # Default starting phase index is 1
    if args.print_n_highest_probabilities is None or args.print_n_highest_probabilities < 1:
        args.print_n_highest_probabilities = 1

    # Define the output folder
    timestamp = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    ckpt_output_folder_name = None
    for ckpt_path in args.ckpt_paths:
        ckpt_folder_name = os.path.basename(os.path.dirname(ckpt_path)).split(".")[0]
        ckpt_epoch = os.path.basename(ckpt_path).split(".")[0].split("=")[-1]
        if ckpt_output_folder_name:
            ckpt_output_folder_name += f"_{ckpt_folder_name}_epoch={ckpt_epoch}"
        else:
            ckpt_output_folder_name = f"{ckpt_folder_name}_epoch={ckpt_epoch}"
    if args.input_type == "random":
        output_folder_path = os.path.join(PATH_TO_YAY_ROBOT, "evaluation", "hl_policy_pipeline", args.input_type, ckpt_output_folder_name, f"{args.tissue_name}_{timestamp=}")
    else:
        output_folder_path = os.path.join(PATH_TO_YAY_ROBOT, "evaluation", "hl_policy_pipeline", args.input_type, ckpt_output_folder_name, timestamp)
        
    # Create the recordings folder if it does not exist
    if not os.path.exists(output_folder_path) and (args.save_video_flag or args.input_type == "random"): 
        os.makedirs(output_folder_path)
    
    # -------------- Initialize the instructor model --------------

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Init instructor model + load parameters and weigths from checkpoint
    instructor_models = []
    # history_step_size_list, history_len_list, model_camera_names_list, phase_history_flag_set_list = [], [], [] # TODO: To be added later - ensemble model that can handle models from different configs
    for ckpt_path in args.ckpt_paths:
        checkpoint = torch.load(ckpt_path, map_location=device)
        history_len = checkpoint.history_len 
        candidate_texts = checkpoint.candidate_texts 
        candidate_embeddings = checkpoint.candidate_embeddings
        prediction_offset = checkpoint.prediction_offset 
        history_step_size = checkpoint.history_step_size 
        one_hot_flag = checkpoint.one_hot_flag
        model_camera_names = checkpoint.camera_names
        backbone_model_name = checkpoint.backbone_model_name
        model_init_weights = "imagenet" # checkpoint.model_init_weights # TODO: Check how to solve this as it does not need any model init weights - or important for preprocessing, or feature extraction
        freeze_backbone_until = checkpoint.freeze_backbone_until
        use_jaw_values_flag = checkpoint.use_jaw_values_flag
        use_phase_history_flag = checkpoint.use_phase_history_flag
        phase_history_len = checkpoint.phase_history_len if hasattr(checkpoint, "phase_history_len") else 0
        phase_history_only_phase_switches_flag = checkpoint.phase_history_only_phase_switches_flag if hasattr(checkpoint, "phase_history_only_phase_switches_flag") else True
        if use_phase_history_flag and not phase_history_only_phase_switches_flag:
            raise Exception("Pipeline currently only works with models trained with phase_history_only_phase_switches_flag=True.")
        image_dim = checkpoint.image_dim if hasattr(checkpoint, "image_dim") else (224, 224)
        if isinstance(image_dim, int):
            image_dim = (image_dim, image_dim)
        wrist_camera_rel_width = checkpoint.wrist_images_rel_width if hasattr(checkpoint, "wrist_images_rel_width") else 1 
        selected_multitasks = checkpoint.selected_multitasks if hasattr(checkpoint, "selected_multitasks") else []
        phase_to_instruction_mapping = checkpoint.phase_to_instruction_mapping if hasattr(checkpoint, "phase_to_instruction_mapping") else None
        use_seg_masks_input_flag = checkpoint.use_seg_masks_input_flag if hasattr(checkpoint, "use_seg_masks_input_flag") else False
        merge_seg_masks_flag = checkpoint.merge_seg_masks_flag if hasattr(checkpoint, "merge_seg_masks_flag") else False
        seg_mask_objs = checkpoint.seg_mask_objs if hasattr(checkpoint, "seg_mask_objs") else []
        add_center_crop_view_flag = checkpoint.add_center_crop_view_flag if hasattr(checkpoint, "add_center_crop_view_flag") else False
        merge_global_and_center_embs_flag = checkpoint.merge_global_and_center_embs_flag if hasattr(checkpoint, "merge_global_and_center_embs_flag") else False
        distance_from_border_y = checkpoint.distance_from_border_y if hasattr(checkpoint, "distance_from_border_y") else 0.1
        distance_from_border_x = checkpoint.distance_from_border_x if hasattr(checkpoint, "distance_from_border_x") else 0.25
        y_offset = checkpoint.y_offset if hasattr(checkpoint, "y_offset") else -0.1
        model_camera_patch_names = checkpoint.camera_patch_names if hasattr(checkpoint, "camera_patch_names") else model_camera_names
        use_phase_history_for_moving_direction_and_corr_pred_flag = checkpoint.use_phase_history_for_moving_direction_and_corr_pred_flag if hasattr(checkpoint, "use_phase_history_for_moving_direction_and_corr_pred_flag") else False
        moving_direction_and_corr_history_len = checkpoint.moving_direction_and_corr_history_len if hasattr(checkpoint, "moving_direction_and_corr_history_len") else history_len
        use_separate_backbones_flag = checkpoint.use_separate_backbones_flag if hasattr(checkpoint, "use_separate_backbones_flag") else False
        dataset_mean_std_camera_dict = checkpoint.dataset_mean_std_camera_dict if hasattr(checkpoint, "dataset_mean_std_camera_dict") else None 
        temporal_mode = checkpoint.temporal_mode if hasattr(checkpoint, "temporal_mode") else None
        global_pool_image_features_flag = checkpoint.global_pool_image_features_flag if hasattr(checkpoint, "global_pool_image_features_flag") else False
        use_complexer_multitask_mlp_head_flag = checkpoint.use_complexer_multitask_mlp_head_flag if hasattr(checkpoint, "use_complexer_multitask_mlp_head_flag") else False
        num_transformer_heads = checkpoint.num_heads if hasattr(checkpoint, "num_heads") else 4
        num_transformer_layers = checkpoint.num_layers if hasattr(checkpoint, "num_layers") else 2
        use_transformer_for_language_corrections_flag = checkpoint.use_transformer_for_language_corrections_flag if hasattr(checkpoint, "use_transformer_for_language_corrections_flag") else False
        add_multitask_queries_flag = checkpoint.add_multitask_queries_flag if hasattr(checkpoint, "add_multitask_queries_flag") else False
        instructor_model = build_instructor(history_len, history_step_size, prediction_offset, candidate_embeddings, candidate_texts, device, one_hot_flag, 
                                            model_camera_names, backbone_model_name, model_init_weights, freeze_backbone_until, global_pool_image_features_flag, use_jaw_values_flag, 
                                            use_phase_history_flag, phase_history_len, temporal_mode, phase_to_instruction_mapping, phase_history_only_phase_switches_flag,
                                            image_dim=image_dim, selected_multitasks=selected_multitasks, use_seg_masks_input_flag=use_seg_masks_input_flag, 
                                            merge_seg_masks_flag=merge_seg_masks_flag, seg_mask_objs=seg_mask_objs, add_center_crop_view_flag=add_center_crop_view_flag,
                                            merge_global_and_center_embs_flag=merge_global_and_center_embs_flag, distance_from_border_y=distance_from_border_y,
                                            distance_from_border_x=distance_from_border_x, y_offset=y_offset, use_phase_history_for_moving_direction_and_corr_pred_flag=use_phase_history_for_moving_direction_and_corr_pred_flag,
                                            moving_direction_and_corr_history_len=moving_direction_and_corr_history_len, use_separate_backbones_flag=use_separate_backbones_flag,
                                            dataset_mean_std_camera_dict=dataset_mean_std_camera_dict, use_complexer_multitask_mlp_head_flag=use_complexer_multitask_mlp_head_flag, 
                                            use_transformer_for_language_corrections_flag=use_transformer_for_language_corrections_flag, add_multitask_queries_flag=add_multitask_queries_flag,
                                            num_transformer_heads=num_transformer_heads, num_transformer_layers=num_transformer_layers)
                                            
        
        # Load the model weights
        instructor_model.load_state_dict(checkpoint.state_dict())    
        instructor_model.to(device)
        del checkpoint # Free up memory

        # Set the model to evaluation mode
        instructor_model.eval()
        
        instructor_models.append(instructor_model)
    
    # If model has no history len th
    if history_len == 0:
        history_step_size = prediction_stride = args.prediction_stride
        print(f"Model has no history length, setting history_step_size to prediction stride: {prediction_stride}\n")
    else:
        # Set the prediction stride - that it only predicts after getting a having the newest frame
        stride_factor = history_step_size * args.ll_policy_slowness_factor
        prediction_stride = ((args.prediction_stride + stride_factor - 1) // stride_factor) * stride_factor
        if prediction_stride != args.prediction_stride:
            print(f"Adjusted prediction stride to: {prediction_stride} as multiple of history step size (for more efficient sampling)\n")
    
    if args.input_type == "video" and add_center_crop_view_flag:
        raise ValueError("Center crop view is not yet supported for video input.")
    
    # ------------- Initialize ROS communication -------------
    
    if args.input_type == "live":        
        rospy.init_node('hl_policy_pipepline', anonymous=True)
    
        # Set the rate of execution
        rate = rospy.Rate(args.fps)
        
        # Instructor publisher for the language instruction prediction and direction prediction
        instruction_publisher = rospy.Publisher("/instructor_prediction", String, queue_size=args.publisher_queue_size) 
        if "dominant_moving_direction" in selected_multitasks:
            direction_instruction_publisher = rospy.Publisher("/direction_instruction", String, queue_size=args.publisher_queue_size)
        if "is_correction" in selected_multitasks:
            is_correction_publisher = rospy.Publisher("/is_correction", Bool, queue_size=args.publisher_queue_size)
        if "clip_loading_tool_switching_required" in selected_multitasks:
            clip_loading_tool_switching_required_publisher = rospy.Publisher("/clip_loading_tool_switching_required", Bool, queue_size=args.publisher_queue_size)
            
        # Wrist camera subs
        if "endo_psm1" in model_camera_names:
            endo_psm1_sub = rospy.Subscriber("/PSM1/endoscope_img", Image, psm1_wrist_camera_callback, queue_size=args.subscriber_queue_size)
        if "endo_psm2" in model_camera_names:
            endo_psm2_sub = rospy.Subscriber("/PSM2/endoscope_img", Image, psm2_wrist_camera_callback, queue_size=args.subscriber_queue_size)
        
        # Endoscope imgs
        if "left_img_dir" in model_camera_names:
            left_img_dir_sub = rospy.Subscriber("/jhu_daVinci/left/image_raw", Image, left_camera_callback, queue_size=args.subscriber_queue_size)
        if "right_img_dir" in model_camera_names:
            right_img_dir_sub = rospy.Subscriber("/jhu_daVinci/right/image_raw", Image, right_camera_callback, queue_size=args.subscriber_queue_size)
            
        # Get the segmentation mask via this ROS topic
        if use_seg_masks_input_flag and "left_img_dir" in model_camera_names:
            left_img_dir_seg_mask_sub = rospy.Subscriber("/yolo_segmentation_masks", Image, left_camera_seg_mask_callback, queue_size=args.subscriber_queue_size)
            
        # Jaw values
        if use_jaw_values_flag:
            psm1_jaw_sub = rospy.Subscriber("/PSM1/jaw/measured_js", JointState, psm1_jaw_callback, queue_size=args.subscriber_queue_size)
            psm2_jaw_sub = rospy.Subscriber("/PSM2/jaw/measured_js", JointState, psm2_jaw_callback, queue_size=args.subscriber_queue_size)
         
        # Pause robot flag
        pause_robot_sub = rospy.Subscriber("/pause_hl", Bool, pause_robot_callback, queue_size=args.subscriber_queue_size)
        
        # User correction instruction
        if args.incorporate_hl_user_corrections_flag:
            user_correction_instruction_sub = rospy.Subscriber("/hl_policy_correction_phase_instruction", String, user_correction_instruction_callback, queue_size=args.subscriber_queue_size)
            user_direction_correction_instruction_sub = rospy.Subscriber("/direction_instruction_user", String, user_correction_instruction_callback, queue_size=args.subscriber_queue_size)

        time.sleep(1) # Wait for the subscribers to be initialized
        
    # -------------- Init the recorded episode data (if random) --------------
    
    if args.input_type == "random":
        # Generate a random episode
        frame_episode_sequence, jaw_values_episode_sequence, episode_gt_instruction_sequence, phase_start_idx_dict, episode_multitask_gt_labels_dict = create_random_chole_episode(args.dataset_dir, args.tissue_name, model_camera_names, args.camera_name_file_suffix_dict, 
                                                                                                                        image_dim, phase_to_instruction_mapping, wrist_camera_rel_width, selected_multitasks, use_seg_masks_input_flag,
                                                                                                                        seg_mask_objs, merge_seg_masks_flag, add_center_crop_view_flag=add_center_crop_view_flag, distance_from_border_x=distance_from_border_x,
                                                                                                                        distance_from_border_y=distance_from_border_y, y_offset=y_offset) 

        # Check if the gt instructions are within the learned instructions of the model
        gt_instructions_set = set(episode_gt_instruction_sequence)
        candidate_texts_set = set(candidate_texts)
        if not gt_instructions_set.issubset(candidate_texts_set):
            raise ValueError(f"The ground truth instructions are not within the learned instructions of the model. Missing instructions: {gt_instructions_set - candidate_texts_set}")
    
    # -------------- Init the recorded video (e.g., live failure modes) --------------
    
    if args.input_type == "video":
        if use_jaw_values_flag or use_phase_history_flag:
            raise ValueError("Jaw values and phase history are not supported for video input.")
        if use_seg_masks_input_flag and not "yolo." in args.video_path:
            raise ValueError("Segmentation mask model can only be applied when having video with segmentation masks.")
        video_capture = cv2.VideoCapture(args.video_path)
        if not video_capture.isOpened():
            raise ValueError(f"Error opening video file {args.video_path}")
    
    # -------------- Initialize evaluation variables --------------
    
    # Initialize the language instruction prediction and ground truth lists
    instruction_pred_list, instruction_gt_list = [], []
    
    # Init the execution times dict
    execution_times_dict = defaultdict(list)
    
    # Initialize the multitask predictions and ground truths dictionaries
    if selected_multitasks:
        multitask_preds_eval_dict = {task_name: [] for task_name in selected_multitasks}
        multitask_gts_eval_dict = {task_name: [] for task_name in selected_multitasks}        
    
    # ---------------------- Init phase history and jaw values (if used) ----------------------
    
    if args.input_type == "random" and args.wait_offline_eval_scheme_flag:
        phase_indices = [int(phase_folder_name.split("_")[0]) for phase_folder_name in phase_start_idx_dict]
        phases_idx_to_folder_name_dict = dict(zip(phase_indices, list(phase_start_idx_dict.keys())))
    
    if args.input_type == "random":
        starting_phase_folder = phases_idx_to_folder_name_dict[args.starting_phase_idx] 
    
    # Init phase history list (with the phase indices up to the start phase)    
    if use_phase_history_flag:
        if args.input_type == "random":
            first_phase_idx = [int(phase_folder_name.split("_")[0]) for phase_folder_name in phase_start_idx_dict][0]
            if first_phase_idx > args.starting_phase_idx: # If the first phase in the episode is after the given starting phase index, then set the starting phase index to the first phase index
                args.starting_phase_idx = first_phase_idx
                print(f"Starting phase index is set to {first_phase_idx} as the first recorded phase in the episode (for {args.tissue_name}) is after the given starting phase index.")
        
        if args.starting_phase_idx == 1:
            phase_history = [0]*phase_history_len 
        else:
            phase_history = [0]*(max(0, phase_history_len-args.starting_phase_idx)) + list(range(args.starting_phase_idx))[-phase_history_len:]
        
        # Get the starting phase based on the starting phase index 
        phase_history_commands = [candidate_texts[phase_idx-1] for phase_idx in phase_history if phase_idx != 0]
        starting_phase = candidate_texts[args.starting_phase_idx-1] 
        print(f"Starting phase: {starting_phase} - Phase history: {phase_history_commands}")
        model_input_phase_history = torch.tensor(phase_history).to(device).unsqueeze(0) # Shape: batch_size (=1), phase_history_len
    else:
        model_input_phase_history = None
        phase_history = None
    
    # Keeping the last frames (for the history length)
    model_input_frames = deque(maxlen=history_len+1)
    if use_jaw_values_flag:
        model_input_jaw_values = deque(maxlen=history_len+1)
    else:
        model_input_jaw_values_tensor = None
    
    # ------------- Init the offline evaluation scheme variables -------------
    
    if args.input_type == "random" and args.wait_offline_eval_scheme_flag:
        curr_phase_idx = args.starting_phase_idx
        max_wait_counter = (history_len * history_step_size) * args.low_pass_buffer_size # No ll slowness filter as offline data is only expert data
        predicted_on_repeated_end_frame_flag = now_predict_on_repeated_end_frame_only_flag = False
        stuck_in_phase_transition_counter = 0
        stuck_in_phase_transitions_list = []
        last_phase_idx = list(phases_idx_to_folder_name_dict.keys())[-1]
        curr_phase_abs_end_idx, next_phase_instruction_to_predict, wait_counter, current_phase_name, next_phase_name = trigger_offline_phase_transition_waiting_scheme(curr_phase_idx, phases_idx_to_folder_name_dict, phase_start_idx_dict,
                                                                                                                                          episode_gt_instruction_sequence, phase_to_instruction_mapping)
    else:
        wait_counter = 0 # To be ignored when not using the offline evaluation scheme
    
    # -------------- Init the video saving -------------- 
    
    if args.save_video_flag:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create the video file name
        if args.input_type == "random":
            additional_info = args.tissue_name
            video_file_path = os.path.join(output_folder_path, f'{args.input_type}_{additional_info}_{timestamp=}.mp4')
        else:
            video_file_path = os.path.join(output_folder_path, f'{args.input_type}_{timestamp=}.mp4')            

        img_dim_upscaled = (int(image_dim[1]*args.upscaling_factor), int(image_dim[0]*args.upscaling_factor)) # Args given in (H, W) -> for cv2 it is (W, H)
        additional_width = 750  # Space for text on the right (for prediction/gt, .. texts)
        additional_seg_cam = 1 if use_seg_masks_input_flag else 0
        video_dim = (img_dim_upscaled[0]*(len(model_camera_patch_names)+additional_seg_cam)+additional_width, img_dim_upscaled[1])
        out = cv2.VideoWriter(video_file_path, fourcc, args.fps, video_dim)
        print(f"Saving the video to {video_file_path}")
    
    # ------------- Main loop -------------
    
    # Init output buffer
    logits_buffer = deque(maxlen=args.low_pass_buffer_size)
    
    # Init the frame index and the frame index offset (if starting phase index is given)
    frame_idx = 0
    frame_idx_offset = 0 if args.input_type in ["live", "video"] else phase_start_idx_dict[starting_phase_folder]
    new_pred_flag = False
    
    # Init multitask variables
    multitask_preds = multitask_probs = multitask_gts = None
    
    # Init phase pred variables
    predicted_instruction_prob = None
    
    # HL user correction variables
    do_not_predict_flag = False
    do_not_predict_counter = 0
    global new_user_correction_flag
    
    global started_main_loop_flag
    started_main_loop_flag = True
    while not exit_flag:
        # -------------- HL user corrections check --------------
        
        if do_not_predict_flag:
            do_not_predict_counter -= 1
            if do_not_predict_counter == 0:
                do_not_predict_flag = False
                
                # Reinitialize logit output buffer
                logits_buffer = deque(maxlen=args.low_pass_buffer_size)
        
        if args.input_type == "live" and args.incorporate_hl_user_corrections_flag:
            # Check if a new instruction got published
            if new_user_correction_flag:
                # Publish the new instruction - if actual phase instruction - otherwise just wait until direction instruction executed (via fixed time)
                if user_correction_instruction not in get_all_auto_labels_list(): # TODO: Check if works
                    print(f"User correction instruction received: {user_correction_instruction}.")
                    instruction_publisher.publish(user_correction_instruction)
                
                    # Update the model input phase history (will then be added to the phase history list below)
                    new_pred_flag = True
                    predicted_instruction = user_correction_instruction
                else:
                    print(f"User direction instruction received: {user_correction_instruction}")
                    
                # Stop predicting for the next n frames
                do_not_predict_flag = True
                do_not_predict_counter = args.do_not_predict_counter
                    
                # Reset the new user correction flag
                new_user_correction_flag = False
                
           
        # -------------- HL Policy Loading and Inference --------------   
         
        if args.input_type == "live" and pause_robot_flag:
            rate.sleep()
            continue
            
        frame_idx_abs = frame_idx + frame_idx_offset
        with measure_execution_time("Total inference time", execution_times_dict):
            # -------------- Load current jaw values + camera frames --------------
 
            if args.input_type == "video" or args.visualization_flag or args.save_video_flag or ((frame_idx + wait_counter) % history_step_size*args.ll_policy_slowness_factor) == 0: # Only load the frames if they are needed (for visualization, saving video, or for the model input)
                with measure_execution_time("Loading jaw values time", execution_times_dict):
                    if args.input_type == "random":
                        success, current_jaw_values = get_current_jaw_values(args.input_type, frame_idx=frame_idx_abs, random_episode_jaw_value_sequence=jaw_values_episode_sequence)
                    elif use_jaw_values_flag and args.input_type == "live":
                        success, current_jaw_values = get_current_jaw_values(args.input_type)
                    else:
                        current_jaw_values = None
                        success = True 
                    # Break out of the loop if the jaw values could not be loaded (e.g., end of the episode)
                    if not success:
                        print(f"Current jaw values could not be loaded. End of the episode reached or could not load current jaw values.")
                        break

                with measure_execution_time("Loading video frame time", execution_times_dict):
                    if args.input_type == "random":
                        success, current_frames = get_current_frames(args, image_dim, frame_idx=frame_idx_abs, random_episode_frame_sequence=frame_episode_sequence, add_center_crop_view_flag=add_center_crop_view_flag,
                                                                                 distance_from_border_y=distance_from_border_y, distance_from_border_x=distance_from_border_x, y_offset=y_offset)
                    elif args.input_type == "live":
                        success, current_frames = get_current_frames(args, image_dim, camera_names=model_camera_names, wrist_camera_rel_width=wrist_camera_rel_width, 
                                                                     use_seg_masks_input_flag=use_seg_masks_input_flag, merge_seg_masks_flag=merge_seg_masks_flag, seg_mask_objs=seg_mask_objs, add_center_crop_view_flag=add_center_crop_view_flag,
                                                                    distance_from_border_y=distance_from_border_y, distance_from_border_x=distance_from_border_x, y_offset=y_offset)
                    elif args.input_type == "video":
                        success, current_frames = get_current_frames(args, image_dim, video_capture=video_capture, video_path=args.video_path, use_seg_masks_input_flag=use_seg_masks_input_flag, merge_seg_masks_flag=merge_seg_masks_flag, seg_mask_objs=seg_mask_objs)
                    # Break out of the loop if the image could not be loaded (e.g., frame could not be loaded)
                    if not success:
                        print(f"Current frames could not be loaded. End of the episode reached or could not load current frames.")
                        break   # Stop the loop and save the video up to here (if desired)
                    
                    # Add frames and jaw values to model input based on the history step size (and dependent of the slower speed of the low level policy)
                    if (frame_idx + wait_counter) % (history_step_size*args.ll_policy_slowness_factor) == 0:
                        if frame_idx == 0:
                            for _ in range(history_len+1):
                                model_input_frames.append(current_frames)
                        else:
                            model_input_frames.append(current_frames) # model_input_frames has not the full length, then add current frame repeatedly
                        model_input_frames_tensor = torch.stack(list(model_input_frames), dim=0).to(device).unsqueeze(0) # Shape: batch_size (=1), history_len, cam, c, h, w 
                
                        if use_jaw_values_flag:
                            model_input_jaw_values.append(current_jaw_values)
                            model_input_jaw_values_tensor = torch.stack(list(model_input_jaw_values), dim=0).to(device).unsqueeze(0) # Shape: batch_size (=1), history_len, 2
            
            # -------------- Predict (+publish) the language instruction --------------
            
            # Start with predictions after the history length is reached and predict every x frames and not at the beginning (as starting with the first phase)
            if not do_not_predict_flag and (frame_idx + wait_counter) % prediction_stride == 0 and len(model_input_frames) == history_len+1:
                with measure_execution_time("Instructor_inference_time", execution_times_dict):
                    # Apply the model on the current frames
                    ensemble_logits_list, ensemble_multitask_logits_dict_list = [], []
                    for instructor_model in instructor_models:
                        logits, _, multitask_logits_dict, temperature = instructor_model(model_input_frames_tensor, model_input_jaw_values_tensor, model_input_phase_history)
                        ensemble_logits_list.append(logits)
                        ensemble_multitask_logits_dict_list.append(multitask_logits_dict)
                    # TODO: Apply ensemble mode on each model output
                    ensemble_logits, ensemble_multitask_logits_dict = apply_ensemble_mode(ensemble_logits_list, ensemble_multitask_logits_dict_list, ensemble_mode=args.ensemble_mode)
                    
                    
                    # Apply the low pass filter on the logits (if desired)
                    if args.low_pass_buffer_size > 1:
                        logits_buffer.append(ensemble_logits)
                        stacked_logits = torch.stack(list(logits_buffer)).squeeze(1)
                        smoothed_logits = apply_logits_lp_filter(stacked_logits, filter_mode=args.lp_filter_mode, smoothing_factor=args.lp_ema_smoothing_factor) 
                    else:
                        smoothed_logits = ensemble_logits
                    # Decode the model output to the language instruction
                    predicted_instruction = instructor_model.decode_logits(smoothed_logits, temperature)[0]
                    instruction_pred_list.append(predicted_instruction)
                    new_pred_flag = True
                    
                    # Not predicting for the next 2s when the user corrects the prediction
                    if predicted_instruction == "go back":
                        do_not_predict_flag = True
                        do_not_predict_counter = 60 
                    
                    # Trigger prediction on repeated end frame only (if offline evaluation scheme is used)
                    if args.input_type == "random" and args.wait_offline_eval_scheme_flag and now_predict_on_repeated_end_frame_only_flag:
                        predicted_on_repeated_end_frame_flag = True
                        now_predict_on_repeated_end_frame_only_flag = False
                    
                    # Log the model input and the predicted instruction
                    print("\n----------------------------------")
                    if use_jaw_values_flag and use_phase_history_flag:
                        print(f"\nFrame Idx: {frame_idx_abs} - Jaw values (PSM2, PSM1): {model_input_jaw_values_tensor}, Phase history: {model_input_phase_history}")
                    elif use_jaw_values_flag:
                        print(f"\nFrame Idx: {frame_idx_abs} - Jaw values (PSM2, PSM1): {model_input_jaw_values_tensor}")
                    elif use_phase_history_flag:
                        print(f"\nFrame Idx: {frame_idx_abs} - Phase history: {model_input_phase_history}")
                    
                    # Get the probabilities of the smoothed logits
                    smoothed_probabilities = torch.nn.functional.softmax(smoothed_logits.squeeze(), dim=0)
                    mapped_smoothed_probabilities_dict = {candidate_texts[phase_idx]: smoothed_probabilities[phase_idx].item() for phase_idx in range(len(candidate_texts))}
                    # Output the three highest probabilities
                    sorted_probabilities = sorted(mapped_smoothed_probabilities_dict.items(), key=lambda x: x[1], reverse=True)
                    for prob_idx in range(args.print_n_highest_probabilities):
                        if prob_idx == 0:
                            predicted_instruction_prob = sorted_probabilities[prob_idx][1]
                        print(f"Prediction {prob_idx+1}: {sorted_probabilities[prob_idx][0]} - Probability: {sorted_probabilities[prob_idx][1]*100:.2f}%")
                        
                    
                    # Publish the predicted language instruction
                    if args.input_type in ["live", "video"]:
                        print(f"----> {frame_idx_abs} - Predicted instruction: {predicted_instruction}")
                        if args.input_type == "live":
                            # Publish the predicted language instruction
                            instruction_publisher.publish(predicted_instruction)
                        
                        if selected_multitasks:
                            multitask_preds = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(ensemble_multitask_logits_dict, batch_wise=True, logits_flag=True,
                                                                                                                ignore_do_not_move=args.ignore_do_not_move_direction_flag)
                            
                            if "dominant_moving_direction" in selected_multitasks:
                                predicted_direction_instruction = multitask_preds["dominant_moving_direction"][0]
                                direction_instruction_publisher.publish(predicted_direction_instruction)
                                
                            if "is_correction" in selected_multitasks:
                                is_correction = multitask_preds["is_correction"][0]
                                is_correction_bool = True if is_correction == "correction" else False
                                is_correction_publisher.publish(is_correction_bool)
                                
                            if "clip_loading_tool_switching_required" in selected_multitasks:
                                clip_loading_tool_switching_required = multitask_preds["clip_loading_tool_switching_required"][0]
                                clip_loading_tool_switching_required_bool = True if clip_loading_tool_switching_required == "required" else False
                                clip_loading_tool_switching_required_publisher.publish(clip_loading_tool_switching_required_bool)
                            
                            multitask_probs = {multitask: torch.nn.functional.softmax(multitask_logits, dim=1).squeeze()[torch.argmax(torch.nn.functional.softmax(multitask_logits, dim=1).squeeze()).item()].item() for multitask, multitask_logits in ensemble_multitask_logits_dict.items()} if ensemble_multitask_logits_dict else None
                            print("")
                            for multitask_name in ensemble_multitask_logits_dict:
                                print(f"Multitask: {multitask_name} - Prediction: {multitask_preds[multitask_name][0]} - Probability: {multitask_probs[multitask_name]*100:.2f}%")
                            print("")
                    
                    # # Debugging: Save the frame with the predicted instruction
                    # save_path = os.path.join(output_folder_path, f"frame_{frame_idx_abs}.png")
                    # log_combined_image(model_input_frames_tensor[0], predicted_instruction, predicted_instruction, pred_prob=predicted_instruction_prob, save_path=save_path,
                    #                    multitask_preds=multitask_preds, multitask_probs=multitask_probs, psm2_psm1_jaw_values=model_input_jaw_values_tensor,
                    #                     phase_history=model_input_phase_history)
                    
                    # Evaluate the prediction (if gt instruction is available --> in offline case)
                    if args.input_type == "random":
                        psm1_not_closed_criterium = bool(("cutting" in current_phase_name or "clipping" in current_phase_name) and current_jaw_values[1] >= INSTRUMENT_CLOSED_THRESHOLD)
                        psm2_not_closed_criterium = bool("grabbing" in current_phase_name and current_jaw_values[0] >= INSTRUMENT_CLOSED_THRESHOLD)
                        episode_end_flag = frame_idx_abs + prediction_offset >= len(episode_gt_instruction_sequence)
                        if psm1_not_closed_criterium or psm2_not_closed_criterium or episode_end_flag:
                            gt_instruction = episode_gt_instruction_sequence[frame_idx_abs]
                        else:
                            gt_instruction = episode_gt_instruction_sequence[frame_idx_abs + prediction_offset]
                        
                        instruction_gt_list.append(gt_instruction)
                        
                        if gt_instruction == predicted_instruction:
                            print(f"--> {frame_idx_abs} - GT instruction: {gt_instruction} - Predicted instruction: {predicted_instruction}")
                        else:
                            print(f"----> {frame_idx_abs} - GT instruction: {gt_instruction} - Predicted instruction: {predicted_instruction} --> Wrong prediction")
                        
                        if selected_multitasks:
                            multitask_preds = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(ensemble_multitask_logits_dict, batch_wise=True, logits_flag=True,
                                                                                                                ignore_do_not_move=args.ignore_do_not_move_direction_flag)
                            multitask_probs = {multitask: torch.nn.functional.softmax(multitask_logits, dim=1).squeeze()[torch.argmax(torch.nn.functional.softmax(multitask_logits, dim=1).squeeze()).item()].item() for multitask, multitask_logits in ensemble_multitask_logits_dict.items()} if ensemble_multitask_logits_dict else None
                            multitask_gts = {multitask: episode_multitask_gt_labels_dict[multitask][frame_idx_abs] for multitask in selected_multitasks}
                            print("")
                            for multitask_name in ensemble_multitask_logits_dict:
                                multitask_gt, multitask_pred, multitask_prob = multitask_gts[multitask_name], multitask_preds[multitask_name][0], multitask_probs[multitask_name]
                                if multitask_gt == multitask_pred:
                                    print(f"Multitask: {multitask_name} - GT - {multitask_gt} Prediction: {multitask_pred} - Probability: {multitask_prob*100:.2f}%")
                                else: 
                                    print(f"Multitask: {multitask_name} - GT - {multitask_gt} Prediction: {multitask_pred} - Probability: {multitask_prob*100:.2f}% --> Wrong prediction")
                            print("")
                            
                            # Add prediction and gt to the evaluation dictionaries
                            for task_name in selected_multitasks:
                                multitask_preds_eval_dict[task_name].append(multitask_preds[task_name][0])
                                multitask_gts_eval_dict[task_name].append(multitask_gts[task_name])

                        # # Debugging: Save the frame with the predicted instruction
                        # save_path = os.path.join(output_folder_path, f"frame_{frame_idx_abs}.png")
                        # log_combined_image(model_input_frames_tensor[0], predicted_instruction, predicted_instruction, pred_prob=predicted_instruction_prob, save_path=save_path,
                        #                 multitask_preds=multitask_preds, multitask_probs=multitask_probs, psm2_psm1_jaw_values=model_input_jaw_values_tensor,
                        #                 phase_history=model_input_phase_history)  
            
            # Update the phase history list
            if use_phase_history_flag and new_pred_flag and predicted_instruction != "go back":
                predicted_instruction_idx = candidate_texts.index(predicted_instruction) + 1 # Because of padding index
                # If the predicted instruction is different from the last instruction, add it to the phase history
                if predicted_instruction_idx != phase_history[-1]:
                    phase_history.append(predicted_instruction_idx)
                    # Keep the phase history list at the desired length
                    phase_history = phase_history[1:]
                    model_input_phase_history = torch.tensor(phase_history).to(device).unsqueeze(0) # Shape: batch_size (=1), phase_history_len
                new_pred_flag = False # Reset the new prediction flag
            
            # -------------- Visualize the frame --------------
            
            if args.visualization_flag or args.save_video_flag:
                with measure_execution_time("Visualizing frame time", execution_times_dict):
                    # Visualization of the predicted language instruction
                    if args.input_type == "random":
                        psm1_not_closed_criterium = bool(("cutting" in current_phase_name or "clipping" in current_phase_name) and current_jaw_values[1] >= INSTRUMENT_CLOSED_THRESHOLD)
                        psm2_not_closed_criterium = bool("grabbing" in current_phase_name and current_jaw_values[0] >= INSTRUMENT_CLOSED_THRESHOLD)
                        episode_end_flag = frame_idx_abs + prediction_offset >= len(episode_gt_instruction_sequence)
                        if psm1_not_closed_criterium or psm2_not_closed_criterium or episode_end_flag:
                            curr_gt_instruction = episode_gt_instruction_sequence[frame_idx_abs]
                        else:
                            curr_gt_instruction = episode_gt_instruction_sequence[frame_idx_abs + prediction_offset]
                        # Get here the current gt multitask labels
                        if selected_multitasks:
                            curr_multitask_gts = {multitask: episode_multitask_gt_labels_dict[multitask][frame_idx_abs] for multitask in selected_multitasks}
                        else:
                            curr_multitask_gts = None
                        annotated_frame = visualize_current_frames(current_frames, predicted_instruction, predicted_instruction_prob, curr_gt_instruction, upscaling_factor=args.upscaling_factor,
                                                                   visualization_flag=args.visualization_flag, current_jaw_values=current_jaw_values, phase_history=phase_history, candidate_texts=candidate_texts,
                                                                   multitask_preds=multitask_preds, multitask_probs=multitask_probs, multitask_gts=curr_multitask_gts, additional_width=additional_width, 
                                                                   seg_mask_objs=seg_mask_objs)
                    else:
                        curr_multitask_gts = None
                        annotated_frame = visualize_current_frames(current_frames, predicted_instruction, predicted_instruction_prob, upscaling_factor=args.upscaling_factor,
                                                                   visualization_flag=args.visualization_flag, current_jaw_values=current_jaw_values, phase_history=phase_history, candidate_texts=candidate_texts,
                                                                   multitask_preds=multitask_preds, multitask_probs=multitask_probs, multitask_gts=curr_multitask_gts, additional_width=additional_width, do_not_predict_flag=do_not_predict_flag,
                                                                   seg_mask_objs=seg_mask_objs)
            
            if args.save_video_flag:
                # Write the annotated frame directly to the video file
                out.write(annotated_frame)

        # -------------- Update + log execution times --------------  
            
        if args.log_execution_times_during_execution_flag:
            # Log the average execution times of the last n samples
            print("") # Add empty line for better readability
            last_n_samples = 100
            num_samples_time_eval = min(len(execution_times_dict["Total inference time"]), last_n_samples)
            for key, value in execution_times_dict.items():
                if key != "Total inference time":
                    print(f"{key} (last {num_samples_time_eval} samples): {np.mean(value[-last_n_samples:])*1000:.2f} ms")
            # For better visibility (+ with infos on the number of frames - as higher at the beginning at start of the process)
            # Note: The average total inference time might be lower than the individual components as not every frame is processed (by the model)
            print(f"--> Total inference time (last {num_samples_time_eval} samples): {np.mean(execution_times_dict['Total inference time'][-last_n_samples:])*1000:.2f} ms")

            print("\n------------------------------------------------------")

        # Exit the pipeline if the end of the episode is reached
        if args.input_type == "random" and frame_idx_abs >= len(frame_episode_sequence) - 1:
            print(f"\nEnd of the episode reached - Stop the pipeline.")
            break

        if args.input_type == "random" and args.wait_offline_eval_scheme_flag:    
            if frame_idx_abs + wait_counter >= curr_phase_abs_end_idx and predicted_instruction == next_phase_instruction_to_predict:
                print(f"Next phase {current_phase_name} triggered by the HL policy after finishing the current phase.")
                if curr_phase_idx < last_phase_idx:
                    curr_phase_idx = curr_phase_idx + 1
                    frame_idx += 1
                curr_phase_abs_end_idx, next_phase_instruction_to_predict, wait_counter, current_phase_name, next_phase_name = trigger_offline_phase_transition_waiting_scheme(curr_phase_idx, phases_idx_to_folder_name_dict, phase_start_idx_dict, 
                                                                                                                                                  episode_gt_instruction_sequence, phase_to_instruction_mapping)
            elif frame_idx_abs + wait_counter >= curr_phase_abs_end_idx + max_wait_counter and predicted_on_repeated_end_frame_flag:
                print(f"End (+ tolerance) of phase {current_phase_name} reached - HL policy has not triggered to the next phase, trigger automatically.")
                if curr_phase_idx < last_phase_idx:
                    curr_phase_idx = curr_phase_idx + 1
                    frame_idx += 1
                    predicted_on_repeated_end_frame_flag = False
                    stuck_in_phase_transition_counter += 1
                    stuck_in_phase_transitions_list.append(f"{current_phase_name} -> {next_phase_name}")
                curr_phase_abs_end_idx, next_phase_instruction_to_predict, wait_counter, current_phase_name, next_phase_name = trigger_offline_phase_transition_waiting_scheme(curr_phase_idx, phases_idx_to_folder_name_dict, phase_start_idx_dict,
                                                                                                                                                  episode_gt_instruction_sequence, phase_to_instruction_mapping)
            elif frame_idx_abs + wait_counter >= curr_phase_abs_end_idx + max_wait_counter:
                print(f"Now all model input frames are the last frame of the phase {current_phase_name} - Waiting for the next phase to be triggered by the HL policy ({wait_counter=}).")
                if not now_predict_on_repeated_end_frame_only_flag:
                    now_predict_on_repeated_end_frame_only_flag = True # Wait until the model had the chance to predict on the repeated end frames only
                wait_counter += 1
            elif frame_idx_abs + wait_counter >= curr_phase_abs_end_idx:
                print(f"End of phase {current_phase_name} reached - Waiting for the next phase to be triggered by the HL policy ({wait_counter=}).")
                wait_counter += 1              
            else:
                frame_idx += 1 # Increase the frame index
        else:
            frame_idx += 1

        # -------------- Stop the pipeline + clean up --------------

        # Break out of the loop if the user pressed q
        if args.visualization_flag and cv2.waitKey(1) == ord('q'):
            break
        
        # Sleep to ensure given rate
        if args.input_type == "live":
            rate.sleep()
        else:
            time_to_sleep = max(1/args.fps - execution_times_dict["Total inference time"][-1], 0) # Sleep for the remaining time of the desired rate
            time.sleep(time_to_sleep)

    if args.visualization_flag:
        cv2.destroyAllWindows()
    
    if args.save_video_flag:
        out.release()
        print(f"\nSaved the video to {video_file_path}")

    # ----------------------- Evaluation ----------------------------

    print("\n----------------------- Evaluation ----------------------------")

    # Evaluate if it got stuck in a phase transition
    if args.input_type == "random" and args.wait_offline_eval_scheme_flag:
        print(f"\nStuck in phase transition counter: {stuck_in_phase_transition_counter}")
        print(f"Stuck in phase transitions: {stuck_in_phase_transitions_list}")

    # Evaluate the language instruction prediction
    if args.input_type == "random":
        metrics_dict = evaluate_instruction_prediction(instruction_pred_list, instruction_gt_list, candidate_texts, timestamp, output_folder_path)

        # Phase prediction evaluation
        print("\nPhase prediction evaluation:")
        print(f"\nAccuracy: {metrics_dict['accuracy']*100:.2f}")
        print(f"F1 score: {metrics_dict['f1_score']*100:.2f}")
        
        print("\n-----------------------\n")
        
        if selected_multitasks:
            multitask_metrics_dict = evaluate_multitask_prediction(multitask_preds_eval_dict, multitask_gts_eval_dict)
            for multitask_name, multitask_metrics in multitask_metrics_dict.items():
                print(f"\nMultitask: {multitask_name}")
                print(f"Accuracy: {multitask_metrics['accuracy']*100:.2f}")
                print(f"F1 score: {multitask_metrics['f1_score']*100:.2f}")
    
        print("\n-----------------------")
    
    # Print average timings (for all different components, with keys in the dict + the total inference time)
    print("\nAverage timings:")
    for key, value in execution_times_dict.items():
        if key != "Total inference time":
            print(f"{key}: {np.mean(value)*1000:.2f} ms")
    print(f"Total inference time: {np.mean(execution_times_dict['Total inference time'])*1000:.2f} ms") 
    

# -------------------------------- Create parser --------------------------------

def parse_pipeline_args():
    """
    Define a parser that reads out the command line arguments and returns the parsed arguments.
    """
    
    parser = argparse.ArgumentParser(description='Surgical Tool Tracking Pipeline')

    # --------------------------- Instruction model parameters ---------------------------
    
    # Instructor model path - Note: Current only working with models that are trained with the same history len, step size -> so e.g. use models trained with same parameter but different train/val split are different vision backbone
    default_ckpt_folder_names =  ["q_star_hl_4_hsz30_second_extra_corrs_pause_prediction"]
    default_ckpt_folder_paths = [os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model_ckpts", "hl", default_ckpt_folder_name) for default_ckpt_folder_name in default_ckpt_folder_names]
    default_ckpt_file_names = ["best_val_acc_epoch=348"] #, "best_val_loss_epoch=150"] 
    default_ckpt_file_paths = [os.path.join(default_ckpt_folder_path, f"{default_ckpt_file_name}.ckpt") for default_ckpt_folder_path, default_ckpt_file_name in zip(default_ckpt_folder_paths, default_ckpt_file_names)]
    parser.add_argument('--ckpt_paths', type=str, default=default_ckpt_file_paths, nargs="+", help="Path to the instructor model")

    # Ensemble mode (e.g., majority, average, max)
    parser.add_argument('--ensemble_mode', type=str, default="average", help="Ensemble mode for the predictions (e.g., voting, average, max)")

    # Prediction stride value to only predict every x frames
    parser.add_argument('--prediction_stride', type=int, default=90, help="Prediction stride value (e.g., predict every x frames). Should be a multiple of the history step size, if not it will be increased to the next multiple.")

    # Output low pass buffer size 
    parser.add_argument('--low_pass_buffer_size', type=int, default=1, help="Low pass buffer size for the logits")

    # Output low pass filter mode
    parser.add_argument('--lp_filter_mode', type=str, default="average", help="Low pass filter mode for the logits (e.g., average, ema)")

    # Output low pass filter alpha value (for ema)
    parser.add_argument('--lp_ema_smoothing_factor', type=float, default=2, help="Low pass filter smoothing factor for the logits (only for ema -> exponential moving average)")

    # Skip do no move direction instruction flag
    parser.add_argument('--ignore_do_not_move_direction_flag', action='store_true', default=True, help="Flag to skip the 'do not move' direction instruction")

    # GPU to run inference on
    parser.add_argument('--gpu', type=int, default=0, help="GPU id to use")

    # ---------------------------------- Data parameters -------------------------------------
    
    # Input type (testing it with live data, random generated episodes
    parser.add_argument('--input_type', type=str, default="live",
                        help="Can be either 'video', 'live' or 'random' (for random generated episode)")
    
    default_video_name = "left_right_failure_mode_yolo.mp4"
    default_video_path = os.path.join(PATH_TO_YAY_ROBOT, "failure_mode_videos", default_video_name)
    parser.add_argument('--video_path', type=str, default=default_video_path, help="Path to the live video stream")
    
    # Camera name file suffix dict
    default_camera_name_file_suffix_dict = {"endo_psm2": "_psm2.jpg", "left_img_dir": "_left.jpg", "right_img_dir": "_right.jpg", "endo_psm1": "_psm1.jpg"}
    parser.add_argument('--camera_name_file_suffix_dict', type=dict, default=default_camera_name_file_suffix_dict, help="Dictionary with the camera names and their corresponding file suffixes")
    
    # Starting phase 
    parser.add_argument('--starting_phase_idx', type=int, default=1, help="Starting phase index for the random generated episode (1 when starting from the first phase)")
    
    # Low level policy speed ratio (as the low level policy is slower than the high level policy) - set when ll policy will be used
    parser.add_argument('--ll_policy_slowness_factor', type=int, default=3, help="Speed ratio of the low level policy compared to the high level policy")
    
    # Add dataset directory
    default_dataset_name = "base_chole_clipping_cutting"
    local_dataset_path = os.getenv("PATH_TO_DATASET")
    if local_dataset_path:
        default_dataset_dir = os.path.join(local_dataset_path, default_dataset_name)
    else:
        default_dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chole_data", default_dataset_name)
    parser.add_argument('--dataset_dir', type=str, default=default_dataset_dir, help="Path to the dataset directory")
    
    # Add tissue id
    default_tissue_sample = "tissue_40"
    parser.add_argument('--tissue_name', type=str, default=default_tissue_sample, help="Tissue id for the random generated episode")
    
    # Add correction option 
    parser.add_argument('--incorporate_hl_user_corrections_flag', action='store_true', default=True, help="Flag to incorporate corrections")
    
    # Add do_not_predict_counter -> to not predict after a user correction got issued
    default_number_sec_not_predict = 4
    parser.add_argument('--do_not_predict_counter', type=int, default=30*default_number_sec_not_predict, help="Counter to not predict after a new instruction got published")
    
    # --------------------------- Visualization & Evaluation parameters ---------------------------
    
    # Upscaling factor for the visualization
    parser.add_argument('--upscaling_factor', type=int, default=2.5, help="Upscaling factor for the visualization of the frames")
    
    # Add save video flag
    parser.add_argument('--save_video_flag', action='store_true', default=True,
                    help="Flag to save the phase prediction recording")
    
    # Add visualization flag
    parser.add_argument('--visualization_flag', action='store_true', default=False,
                    help="Flag to visualize phase prediction during execution.")
    
    # Add log execution times flag
    parser.add_argument('--log_execution_times_during_execution_flag', action='store_true', default=False,
                    help="Flag to log the execution times of the different components")

    # Print the n highest probabilities
    parser.add_argument('--print_n_highest_probabilities', type=int, default=2, help="Print the n highest probabilities of the smoothed logits (>= 1)")

    # Wait offline evaluation scheme flag
    parser.add_argument('--wait_offline_eval_scheme_flag', action='store_true', default=True,
                    help="Flag to wait for the offline evaluation scheme (e.g., only switch to the next phase after the previous phase is completed or after robot triggered the next phase)")

    # Disable segmentation masks input flag (for testing the model without segmentation masks)
    parser.add_argument('--disable_seg_masks_input_flag', action='store_true', default=False, help="Flag to disable the segmentation masks input if model trained on segmentations masks. To check ability to predict without segmentation masks.")

    # Segmentation mask class value mapping
    default_seg_mask_index_mapping = {"clips": 1, "left_tube": 2, "right_tube": 3, "flap": 4}
    parser.add_argument('--seg_mask_index_mapping', type=dict, default=default_seg_mask_index_mapping, help="Dictionary with the segmentation mask class names and their corresponding index values")

    # --------------------------- ROS communication parameters ---------------------------
    
    # Fps (for ros communication)
    parser.add_argument('--fps', type=int, default=30, help="FPS of the ROS node")
    
    # Subscriber queue size    
    parser.add_argument('--subscriber_queue_size', type=int, default=1, help="Queue size for the ROS subscriber")
    
    # Publisher queue size
    parser.add_argument('--publisher_queue_size', type=int, default=1, help="Queue size for the ROS publisher")

    # ----------------------------------------------------------------------------------------------

    # Parse the command line arguments
    args = parser.parse_args()
    
    return args  

if __name__ == '__main__':
    args = parse_pipeline_args()
    instructor_pipeline(args)  