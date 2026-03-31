import os
import json
import numpy as np
import pandas as pd

def calculate_absolute_movement(values):
    """Calculate the absolute movement in 1D or 3D space."""
    diffs = np.diff(values, axis=0)
    
    # Check if diffs is 1D or 2D
    if diffs.ndim == 1:
        distances = np.abs(diffs)  # For 1D data, just take the absolute value of the differences
    else:
        distances = np.linalg.norm(diffs, axis=1)  # For 2D or 3D data, calculate the norm along axis 1
    
    distances = np.insert(distances, 0, 0)  # Insert zero at the beginning
    return distances

def find_end_idx(values, threshold):
    """Find the first index where values drop below the threshold and stay below it."""
    below_threshold = values < threshold
    for idx in range(len(below_threshold)):
        if below_threshold[idx] and np.all(below_threshold[idx:]):
            return idx
    return len(values) - 1  # If no such point is found, return the last index

def find_start_idx(values, threshold):
    """Find the first index exceeding the threshold and where the previous indices were all below the threshold."""
    below_threshold = values < threshold
    for idx in range(len(below_threshold)):
        if not below_threshold[idx] and np.all(below_threshold[:idx]):
            return idx
    return len(values) - 1  # If no such point is found, return the last index

def compute_and_save_end_idx(base_path, movement_threshold=0.01, jaw_threshold=-0.3, jaw_movement_threshold=0.01, gallbladder_pulling_num_frames_offset=15, potential_image_kinematics_delay_offset=10, overwrite_flag=False):
    """Compute and save the end index for each demo."""
    tissue_folders = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder)) and folder.startswith('tissue_')]

    if not tissue_folders:
        raise ValueError(f"No tissue folders found in the specified dataset path: {base_path}")

    for tissue_folder_name in tissue_folders:
        tissue_folder_path = os.path.join(base_path, tissue_folder_name)
        print(f"Processing tissue folder: {tissue_folder_name}")

        for phase_folder_name in os.listdir(tissue_folder_path):            
            if not phase_folder_name.split('_')[0].isdigit():
                continue
            phase_idx = int(phase_folder_name.split('_')[0])
            
            phase_folder_path = os.path.join(tissue_folder_path, phase_folder_name)
            print(f"Processing phase folder: {phase_folder_name}")

            date_folders = [folder for folder in os.listdir(phase_folder_path) if os.path.isdir(os.path.join(phase_folder_path, folder))]
            if not date_folders:
                print(f"No demo folders found for tissue {tissue_folder_name} and phase {phase_folder_name}\n")
                continue

            for demo_folder_name in date_folders:
                demo_folder_path = os.path.join(phase_folder_path, demo_folder_name)
                print(f"Processing demo folder: {demo_folder_name}")

                indices_curated_path = os.path.join(demo_folder_path, 'indices_curated.json')
                
                # Check if the file already exists and contains the necessary entries
                if os.path.exists(indices_curated_path):
                    with open(indices_curated_path, 'r') as f:
                        curated_indices = json.load(f)
                    if all(key in curated_indices for key in ['movement_end_idx_psm1', 'movement_end_idx_psm2', 'movement_end_idx']):
                        if not overwrite_flag:
                            print(f"Skipping demo '{demo_folder_name}' as it already contains the necessary entries.")
                            continue
                        else:
                            print(f"Overwriting demo '{demo_folder_name}' entries due to overwrite flag.")
                else:
                    curated_indices = {}

                ee_csv_path = os.path.join(demo_folder_path, 'ee_csv.csv')
                if not os.path.exists(ee_csv_path):
                    print(f"No kinematics csv file found at {ee_csv_path}\n")
                    continue

                ee_df = pd.read_csv(ee_csv_path)

                psm1_positions = ee_df[['psm1_pose.position.x', 'psm1_pose.position.y', 'psm1_pose.position.z']].values
                psm2_positions = ee_df[['psm2_pose.position.x', 'psm2_pose.position.y', 'psm2_pose.position.z']].values
                psm1_jaw = ee_df['psm1_jaw'].values
                psm2_jaw = ee_df['psm2_jaw'].values

                psm1_movement = calculate_absolute_movement(psm1_positions)
                psm2_movement = calculate_absolute_movement(psm2_positions)
                
                psm1_jaw_movement = calculate_absolute_movement(psm1_jaw)
                psm2_jaw_movement = calculate_absolute_movement(psm2_jaw)

                # ---------------------- End index conditions (phase dependent) ----------------------
                # Skip phases with "cutting" in their name
                if not "cutting" in phase_folder_name:
                    if phase_idx == 1: # Handle phase 1
                        movement_end_idx_psm2 = find_end_idx(psm2_movement, movement_threshold)
                        psm2_jaw_end_idx = min(len(psm1_movement)-1, find_end_idx(psm2_jaw, jaw_threshold) + gallbladder_pulling_num_frames_offset)
                        movement_end_idx = psm2_jaw_end_idx
                        
                        curated_indices.update({
                            'movement_end_idx_psm2': int(movement_end_idx_psm2),
                            'psm2_jaw_end_idx': int(psm2_jaw_end_idx),
                            'movement_end_idx': int(movement_end_idx)
                            })
                    elif phase_idx % 2 != 0 and not phase_folder_name.startswith("1_"):  # Handle odd phases (except phase 1)
                        movement_end_idx_psm1 = find_end_idx(psm1_movement, movement_threshold)
                        movement_end_idx_psm2 = find_end_idx(psm2_movement, movement_threshold)
                        movement_end_idx = max(movement_end_idx_psm1, movement_end_idx_psm2)
                        
                        curated_indices.update({
                            'movement_end_idx_psm1': int(movement_end_idx_psm1),
                            'movement_end_idx_psm2': int(movement_end_idx_psm2),
                            'movement_end_idx': int(movement_end_idx)
                            })
                    else:  # Handle even phases
                        psm1_jaw_end_idx = min(len(psm1_movement)-1, find_end_idx(psm1_jaw, jaw_threshold) + potential_image_kinematics_delay_offset)
                        movement_end_idx_psm1 = find_end_idx(psm1_movement, movement_threshold)
                        movement_end_idx = max(movement_end_idx_psm1, psm1_jaw_end_idx)

                        curated_indices.update({
                            'movement_end_idx_psm1': int(movement_end_idx_psm1),
                            'movement_end_idx': int(movement_end_idx)
                        })
                
                # ---------------------- Start index conditions (phase dependent) ----------------------
                
                # Add start index conditions
                movement_start_idx_psm1 = find_start_idx(psm1_movement, movement_threshold)
                movement_start_idx_psm2 = find_start_idx(psm2_movement, movement_threshold)
                movement_start_idx_jaw_psm1 = find_start_idx(psm1_jaw_movement, jaw_movement_threshold)
                movement_start_idx_jaw_psm2 = find_start_idx(psm2_jaw_movement, jaw_movement_threshold)
                movement_start_idx = min(movement_start_idx_psm1, movement_start_idx_psm2, movement_start_idx_jaw_psm1, movement_start_idx_jaw_psm2)
                
                curated_indices.update({
                    'movement_start_idx_psm1': int(movement_start_idx_psm1),
                    'movement_start_idx_psm2': int(movement_start_idx_psm2),
                    'movement_start_idx_jaw_psm1': int(movement_start_idx_jaw_psm1),
                    'movement_start_idx_jaw_psm2': int(movement_start_idx_jaw_psm2),
                    'movement_start_idx': int(movement_start_idx)
                })
                        
                # ------------------------- Save the indices -------------------------

                with open(indices_curated_path, 'w') as f:
                    json.dump(curated_indices, f, indent=4)
                print(f"Saved indices to {indices_curated_path}\n")

if __name__ == "__main__":
    base_path = os.path.join(os.getenv("PATH_TO_DATASET"), "experiments")
    movement_threshold = 1.5e-4 
    jaw_threshold = -0.3  # Set the jaw threshold to -0.3
    jaw_movement_threshold = 0.05  # Set the jaw movement threshold
    gallbladder_pulling_num_frames_offset = 30
    potential_image_kinematics_delay_offset = 20  # Set the image offset to 10 (default)
    overwrite_flag = True  # Set to True to overwrite existing entries
    compute_and_save_end_idx(base_path, movement_threshold=movement_threshold, jaw_threshold=jaw_threshold, jaw_movement_threshold=jaw_movement_threshold, 
                             gallbladder_pulling_num_frames_offset=gallbladder_pulling_num_frames_offset,
                             potential_image_kinematics_delay_offset=potential_image_kinematics_delay_offset, overwrite_flag=overwrite_flag)
