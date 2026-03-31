
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def visualize_robot_trajectory(qpos):

    factor = 1000
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    cutting_start = 208  #145
    cutting_end = 213  #149
    # ax.scatter(actions_psm1[:, 0], actions_psm1[:, 1], actions_psm1[:, 2], c ='r')
    # ax.scatter(actions_psm2[:, 0]*factor, actions_psm2[:, 1]*factor, actions_psm2[:, 2]*factor, c ='g', label = 'Generated trajectory')
    # ax.scatter(qpos_psm1[0], qpos_psm1[1], qpos_psm1[2], c = 'b')
    ax.scatter(qpos[0:cutting_start, 0]*factor, qpos[0:cutting_start, 1]*factor, qpos[0:cutting_start, 2]*factor, c = 'b', label = 'PSM1 position')
    ax.scatter(qpos[cutting_start:cutting_end, 0]*factor, qpos[cutting_start:cutting_end, 1]*factor, qpos[cutting_start:cutting_end, 2]*factor, c = 'r', label = 'PSM1 cutting')
    ax.scatter(qpos[cutting_end:, 0]*factor, qpos[cutting_end:, 1]*factor, qpos[cutting_end:, 2]*factor, c = 'k', label = 'PSM1 after cutting')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    n_bins = 7
    ax.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(n_bins))
    ax.yaxis.set_major_locator(plt.MaxNLocator(n_bins))
    ax.zaxis.set_major_locator(plt.MaxNLocator(n_bins))
    plt.show()

def generate_cutting_motion(csv_df, filename, repeating_num):
    # Define the constants
    closing_angle = -0.349096
    closing_rate = 0.5
    if csv_df.empty:
        print("The DataFrame is empty. Please check the CSV file.")
        return
    
    # Check if cutting motion has already been generated
    # by checking if the last psm1_jaw_sp is already at the closing angle
    last_jaw_sp = csv_df['psm1_jaw_sp'].iloc[-1]
    if abs(last_jaw_sp - closing_angle) <= 0.05:  # Tolerance of 0.05
        print(f"Cutting motion already generated (last psm1_jaw_sp = {last_jaw_sp:.4f}). Skipping...")
        return
    
    # Create a backup of the original CSV file if it doesn't exist
    backup_filename = filename.replace('.csv', '_original.csv')
    if not os.path.exists(backup_filename):
        csv_df.to_csv(backup_filename, index=False)
        print(f"Created backup: {backup_filename}")
    
    # Work on a copy of the dataframe
    csv_df_copy = csv_df.copy()
    
    # Get all unique camera sources
    camera_sources = csv_df_copy['camera_source'].unique()
    
    # Get the last set of rows (one for each camera source)
    last_rows_by_camera = {}
    for camera in camera_sources:
        camera_df = csv_df_copy[csv_df_copy['camera_source'] == camera]
        if not camera_df.empty:
            last_rows_by_camera[camera] = camera_df.iloc[-1].to_dict()
    
    # Create a list to hold new rows
    new_rows = []
    
    # Generate cutting motion for each camera source
    for i in range(repeating_num):
        for camera in camera_sources:
            if camera not in last_rows_by_camera:
                continue
                
            row = deepcopy(last_rows_by_camera[camera])
            jaw_angle = row["psm1_jaw"]

            if jaw_angle > 0:
                row["psm1_jaw"] = max(jaw_angle - closing_rate * (i + 1), closing_angle)
            else:
                row["psm1_jaw"] = closing_angle
            
            new_rows.append(row.copy())

    # Convert new rows to a DataFrame
    new_rows_df = pd.DataFrame(new_rows)
    
    # Concatenate the copied data and new rows
    csv_df_copy = pd.concat([csv_df_copy, new_rows_df], ignore_index=True)
    csv_df_copy.to_csv(filename, index=False)

    # input("enter to continue...")
def generate_cutting_motion_sp(csv_df, filename, repeating_num):
    # Define the constants
    closing_angle = -0.349096
    closing_rate = 0.5
    if csv_df.empty:
        print("The DataFrame is empty. Please check the CSV file.", filename)
        return
    
    # Check if cutting motion has already been generated
    # by checking if the last psm1_jaw_sp is already at the closing angle
    last_jaw_sp = csv_df['psm1_jaw_sp'].iloc[-1]
    if abs(last_jaw_sp - closing_angle) < 0.05:  # Tolerance of 0.05
        print(f"Cutting motion SP already generated (last psm1_jaw_sp = {last_jaw_sp:.4f}). Skipping...")
        return
    
    # Create a backup of the original CSV file if it doesn't exist
    backup_filename = filename.replace('.csv', '_original.csv')
    if not os.path.exists(backup_filename):
        csv_df.to_csv(backup_filename, index=False)
        print(f"Created backup: {backup_filename}")
    
    # Work on a copy of the dataframe
    csv_df_copy = csv_df.copy()
    
    # Get all unique camera sources
    camera_sources = csv_df_copy['camera_source'].unique()
    
    # Update the psm1_jaw_sp for the last repeating_num rows for each camera source
    for camera in camera_sources:
        camera_mask = csv_df_copy['camera_source'] == camera
        camera_indices = csv_df_copy[camera_mask].index
        
        if len(camera_indices) < repeating_num:
            continue
            
        # Get the last repeating_num indices for this camera
        last_indices = camera_indices[-repeating_num:]
        
        # Update psm1_jaw_sp for these rows
        for i, idx in enumerate(last_indices):
            # Get the reference jaw angle from the previous row or current row
            if i == 0:
                # For the first row, reference the jaw angle from the same row
                ref_jaw = csv_df_copy.loc[idx, 'psm1_jaw']
            else:
                # For subsequent rows, reference from the previous updated row
                ref_jaw = csv_df_copy.loc[last_indices[i-1], 'psm1_jaw']
            
            csv_df_copy.loc[idx, 'psm1_jaw_sp'] = max(ref_jaw - 0.1, closing_angle)

    # Save the updated DataFrame to the CSV file
    csv_df_copy.to_csv(filename, index=False)

def remove_extra_row(csv_df, filename, sample_path, repeating_num):
    image_dir = os.path.join(sample_path, "left_img_dir")
    image_list = os.listdir(image_dir)
    
    # Count the number of images
    num_images = len(image_list)
    
    # Create a backup of the original CSV file if it doesn't exist
    backup_filename = filename.replace('.csv', '_original.csv')
    if not os.path.exists(backup_filename):
        csv_df.to_csv(backup_filename, index=False)
        print(f"Created backup: {backup_filename}")
    
    # Work on a copy of the dataframe
    csv_df_copy = csv_df.copy()
    
    # Get the number of unique camera sources
    camera_sources = csv_df_copy['camera_source'].unique()
    num_camera_sources = len(camera_sources)
    
    # Calculate the threshold for removing redundant rows
    # Each image corresponds to one set of camera sources, plus the generated cutting motion rows
    threshold = num_images * num_camera_sources + repeating_num * num_camera_sources
    
    # If the number of rows in the DataFrame exceeds the threshold, remove the redundant rows
    if len(csv_df_copy) > threshold:
        print(f"Removing redundant rows from {filename}, csv length: {len(csv_df_copy)}, num of img: {num_images}, num of cameras: {num_camera_sources}, threshold: {threshold}")
        csv_df_copy = csv_df_copy.iloc[:threshold]

    csv_df_copy.to_csv(filename, index=False)


if __name__ == "__main__":
    tissue_ids = [4]
    # dataset_path = "/home/imerse/chole_ws/data/phantom_chole/phantom_1/ACTUAL_CUTTING_right"
    phases = ["8_go_to_the_cutting_position_left_tube",
              "16_go_to_the_cutting_position_right_tube"]
            #   "8_go_to_the_cutting_position_left_tube_recovery", 
            #   "16_go_to_the_cutting_position_right_tube_recovery"]

    # path_to_dataset = os.getenv('PATH_TO_DATASET')
    path_to_dataset = "/home/iulian/chole_ws/data/cnh_exvivo_chole"

    def process_samples(tissue_ids, phases, path_to_dataset, process_func, process_name):
        """Helper function to process samples with a given function."""
        for tissue_id in tissue_ids:
            for phase in phases:
                dataset_path = f"{path_to_dataset}/tissue_{tissue_id}/{phase}"
                if not os.path.exists(dataset_path):
                    print(f"dataset path not found in {dataset_path}")
                    continue
                
                samples = os.listdir(dataset_path)
                for sample in samples:
                    sample_dir = os.path.join(dataset_path, sample)
                    csv_path = os.path.join(sample_dir, "ee_csv.csv")
                    
                    if not os.path.exists(csv_path):
                        print(f"ee state csv file not found in {sample_dir}")
                        continue
                    
                    csv = pd.read_csv(csv_path)
                    
                    if process_func == remove_extra_row:
                        process_func(csv, csv_path, sample_dir, 10)
                    else:
                        process_func(csv, csv_path, 10)
                
                print(f"Done {process_name} for tissue {tissue_id} in phase {phase}")
                # input("enter to continue...")


    # Execute the three processing steps
    process_samples(tissue_ids, phases, path_to_dataset, generate_cutting_motion, "generating cutting")
    # process_samples(tissue_ids, phases, path_to_dataset, remove_extra_row, "removing extra rows")
    process_samples(tissue_ids, phases, path_to_dataset, generate_cutting_motion_sp, "generating sp")
    print("job finished")
