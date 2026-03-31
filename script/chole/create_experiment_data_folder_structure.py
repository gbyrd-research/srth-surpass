import os
import shutil
import pandas as pd
from datetime import datetime, timedelta
import cv2

# ----------------------------- Configuration -----------------------------

# Mapping from phase instructions to folder names
phase_instruction_to_folder_name_mapping = { 
    'grabbing gallbladder': '1_grabbing_gallbladder',
    'grabbing gallbladder recovery': '1_grabbing_gallbladder_recovery',
    'clipping first clip left tube': '2_clipping_first_clip_left_tube',
    'clipping first clip left tube recovery': '2_clipping_first_clip_left_tube_recovery',
    'going back first clip left tube': '3_going_back_first_clip_left_tube',
    'clipping second clip left tube': '4_clipping_second_clip_left_tube',
    'clipping second clip left tube recovery': '4_clipping_second_clip_left_tube_recovery',
    'going back second clip left tube': '5_going_back_second_clip_left_tube',
    'clipping third clip left tube': '6_clipping_third_clip_left_tube',
    'clipping third clip left tube recovery': '6_clipping_third_clip_left_tube_recovery',
    'going back third clip left tube': '7_going_back_third_clip_left_tube',
    'go to the cutting position left tube': '8_go_to_the_cutting_position_left_tube',
    'go to the cutting position left tube recovery': '8_go_to_the_cutting_position_left_tube_recovery',
    'go back from the cut left tube': '9_go_back_from_the_cut_left_tube',
    'clipping first clip right tube': '10_clipping_first_clip_right_tube',
    'clipping first clip right tube recovery': '10_clipping_first_clip_right_tube_recovery',
    'going back first clip right tube': '11_going_back_first_clip_right_tube',
    'clipping second clip right tube': '12_clipping_second_clip_right_tube',
    'clipping second clip right tube recovery': '12_clipping_second_clip_right_tube_recovery',
    'going back second clip right tube': '13_going_back_second_clip_right_tube',
    'clipping third clip right tube': '14_clipping_third_clip_right_tube',
    'clipping third clip right tube recovery': '14_clipping_third_clip_right_tube_recovery',
    'going back third clip right tube': '15_going_back_third_clip_right_tube',
    'go to the cutting position right tube': '16_go_to_the_cutting_position_right_tube',
    'go to the cutting position right tube recovery': '16_go_to_the_cutting_position_right_tube_recovery',
    'go back from the cut right tube': '17_go_back_from_the_cut_right_tube'
}

# Constants
EXPERIMENTS_START_TISSUE_NUMBER = 100

# ----------------------------- Helper Functions -----------------------------

def find_or_select_tissue_folder(output_dataset_path, tissue_number=None):
    if tissue_number is not None:
        tissue_folder_path = os.path.join(output_dataset_path, f"tissue_{tissue_number}")
        if os.path.exists(tissue_folder_path):
            return tissue_folder_path
        else:
            raise ValueError(f"Tissue folder tissue_{tissue_number} does not exist. Please specify an existing folder.")
    else:
        x = EXPERIMENTS_START_TISSUE_NUMBER
        while True:
            tissue_folder_path = os.path.join(output_dataset_path, f"tissue_{x}")
            if not os.path.exists(tissue_folder_path):
                return tissue_folder_path
            x += 1

def parse_labels(labels_file_path, phase_instruction_to_folder_name_mapping):
    labels = []
    with open(labels_file_path, 'r') as f:
        for line in f:
            # Expected format: "HH:MM:SS:ms, Instruction"
            parts = line.strip().split(', ')
            if len(parts) >= 3:
                # First is timestamp, second is instruction, subsequent parts are actions
                timestamp_str, instruction = parts[0], parts[1]
                actions = parts[2:]  # List of actions
                instruction = instruction.strip() # Remove trailing or leading spaces from instruction

                if instruction not in phase_instruction_to_folder_name_mapping and instruction != "pause":
                    print(f"Instruction '{instruction}' not found in the mapping. Skipping.")
                    raise ValueError("Postprocess the labels.txt file first before calling this script. Check that all instructions are valid")
                
                # Add recovery to the instruction name if it is not already there (to assign to correct folder)
                if "recovery" not in instruction:
                    instruction += " recovery"
                
                # Convert timestamp to timedelta
                time_parts = list(map(int, timestamp_str.split(':')))
                timestamp = timedelta(hours=time_parts[0], 
                                      minutes=time_parts[1],
                                      seconds=time_parts[2],
                                      milliseconds=time_parts[3])
                
                # Check that current timestamp is greater than the previous timestamp
                if labels and timestamp < labels[-1][0]:
                    raise ValueError(f"Timestamp {timestamp} for {instruction} is smaller than the previous timestamp {labels[-1][0]} for {instruction}. Exiting.")
                
                # Add each action with the same timestamp
                for action in actions:
                    action_corrected = action.replace("direction correction: ", "") # Remove the prefix from first action
                    action = action_corrected.strip() # Remove trailing or leading spaces
                    action = action.replace(" ", "_") # Replace spaces with underscores -> align with folder name convention
                    labels.append((timestamp, instruction, action))
            elif len(parts) >= 2:
                timestamp_str, instruction = parts[0], parts[1]
                instruction = instruction.strip() # Remove trailing or leading spaces from instruction
                if instruction not in phase_instruction_to_folder_name_mapping and instruction != "pause":
                    print(f"Instruction '{instruction}' not found in the mapping. Skipping.")
                    raise ValueError("Postprocess the labels.txt file first before calling this script. Check that all instructions are valid")
                # Convert timestamp to timedelta
                time_parts = list(map(int, timestamp_str.split(':')))
                timestamp = timedelta(hours=time_parts[0], 
                                      minutes=time_parts[1],
                                      seconds=time_parts[2],
                                      milliseconds=time_parts[3])
                
                # Check that current timestamp is greater than the previous timestamp
                if labels and timestamp < labels[-1][0]:
                    raise ValueError(f"Timestamp {timestamp} for {instruction} is smaller than the previous timestamp {labels[-1][0]} for {instruction}. Exiting.")
                
                labels.append((timestamp, instruction, None))
    return labels

def extract_demo_frames_from_video(source_video_path, demo_folder_output_folder_path, start_time, end_time, camera_suffix_dict, ll_policy_slowness_factor=3,
                                   presampling_duration=4, max_presampling_duration=20):
    """
    Extracts frames from the video between the given start and end times at exactly 30 FPS intervals.
    """
    
    # Create the camera folder - overwrite when already exists
    camera_name = os.path.basename(source_video_path).split('.')[0]
    output_camera_folder_path = os.path.join(demo_folder_output_folder_path, camera_name)
    if os.path.exists(output_camera_folder_path):
        shutil.rmtree(output_camera_folder_path)
        print(f"Overwriting existing camera folder at {output_camera_folder_path}")
    else:
        os.makedirs(output_camera_folder_path, exist_ok=True)
    camera_suffix = camera_suffix_dict[camera_name]
    
    # Initialize the video capture object
    cap = cv2.VideoCapture(source_video_path) 
    fps = cap.get(cv2.CAP_PROP_FPS) # Should be 30 FPS
    
    # Calculate the start frame based on the start time and FPS - also add the presampling frames (if it is a recovery phase) 
    start_frame_before_corrected = int(start_time.total_seconds() * fps)
    phase_name = os.path.basename(os.path.dirname(demo_folder_output_folder_path)) 
    if "recovery" in phase_name:
        pre_frames = min(presampling_duration * ll_policy_slowness_factor, max_presampling_duration) * fps # Calculate the number of pre frames to skip
        pre_frames = int(min(pre_frames, start_frame_before_corrected)) # Ensure that the pre_frames don't exceed the start frame
    else:
        pre_frames = 0
    start_frame = start_frame_before_corrected - pre_frames # will be > 0
    if end_time is None:
        # Set end frame to the last frame of the video
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        end_frame = int(end_time.total_seconds() * fps)

    # Set the initial position of the video to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract frames from the video and save them to the output folder
    current_rel_video_frame = num_camera_frames = 0
    current_video_frame = current_rel_video_frame + start_frame
    pre_frames_to_extract = pre_frames // ll_policy_slowness_factor
    while current_video_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video ({source_video_path}) reached at frame {current_video_frame}. Exiting.")
            break

        # Save the frame only if it meets the slowness factor condition
        if current_rel_video_frame % ll_policy_slowness_factor == 0:
            if num_camera_frames < pre_frames_to_extract:
                pre_frame_idx = abs(num_camera_frames - pre_frames_to_extract)
                frame_filename = f"pre_frame{str(pre_frame_idx).zfill(6)}{camera_suffix}"
            else:
                frame_idx = num_camera_frames - pre_frames_to_extract
                frame_filename = f"frame{str(frame_idx).zfill(6)}{camera_suffix}" 
            frame_filepath = os.path.join(output_camera_folder_path, frame_filename)
            cv2.imwrite(frame_filepath, frame)
            num_camera_frames += 1
        
        # Move to the next frame
        current_rel_video_frame += 1
        current_video_frame = current_rel_video_frame + start_frame

    cap.release()
    
    print(f"Frames extracted from {source_video_path} to {output_camera_folder_path}")
    
    return num_camera_frames # Return number of camera samples

def split_kinematics(source_csv_path, demo_folder_output_folder_path, start_time, end_time, num_camera_samples, ll_policy_slowness_factor=3,
                     presampling_duration=4, max_presampling_duration=20):
    """
    Splits the kinematic data based on the split timestamps.
    `splits` is a list of tuples: (start_time, end_time)
    """
    df = pd.read_csv(source_csv_path)
    kinematics_start_timestamp = df['timestamp'].iloc[0]
    df['rel_timestamp'] = pd.to_timedelta(df['timestamp'] - kinematics_start_timestamp)
    
    # Filter the dataframe based on the relative timestamp - considering the pre period 
    phase_name = os.path.basename(os.path.dirname(demo_folder_output_folder_path))
    if "recovery" in phase_name:
        presampling_duration_to_apply = min(presampling_duration * ll_policy_slowness_factor, max_presampling_duration)
    else:
        presampling_duration_to_apply = 0
    start_time_corrected = max(start_time - timedelta(seconds=presampling_duration_to_apply), timedelta(0)) # Ensure that the start time is not negative
    phase_df = df[(df['rel_timestamp'] >= start_time_corrected) & (df['rel_timestamp'] < end_time)]
    phase_df = phase_df.iloc[::ll_policy_slowness_factor] # Apply the slowness factor
    num_kinematic_samples = len(phase_df)
    
    # Check if the number of kinematic samples matches the number of camera samples and adjust accordingly
    if num_kinematic_samples > num_camera_samples:
        # Skip the last kinematic samples
        phase_df = phase_df.iloc[:num_camera_samples]
        print(f"Kinematic samples ({num_kinematic_samples}) exceed the number of camera samples ({num_camera_samples}). Skipping the last kinematic samples.")
    elif num_kinematic_samples < num_camera_samples:
        # Repeat the last kinematic sample
        last_kinematic_sample = phase_df.iloc[-1:]
        num_repeats = num_camera_samples - num_kinematic_samples
        repeated_samples = pd.concat([last_kinematic_sample] * num_repeats, ignore_index=True)
        phase_df = pd.concat([phase_df, repeated_samples], ignore_index=True)
        print(f"Kinematic samples ({num_kinematic_samples}) are less than the number of camera samples ({num_camera_samples}). Repeating the last kinematic sample.")

    # Save to the output folder
    phase_df.to_csv(os.path.join(demo_folder_output_folder_path, 'ee_csv.csv'), index=False)
    print(f"Kinematics data saved to {demo_folder_output_folder_path}")

# ----------------------------- Main Function -----------------------------

def reorganize_data(source_folder_path, output_dataset_path, camera_file_name_suffix_dict, cameras_to_consider, 
                    ll_policy_slowness_factor, phase_instruction_to_folder_name_mapping, tissue_number, presampling_duration, max_presampling_duration):
    # Ensure the output dataset folder exists
    os.makedirs(output_dataset_path, exist_ok=True)
    
    # Find or create the tissue_x folder
    tissue_folder = find_or_select_tissue_folder(output_dataset_path, tissue_number)
    os.makedirs(tissue_folder, exist_ok=True)
    
    # Define paths to source files
    labels_file = os.path.join(source_folder_path, 'labels.txt')
    kinematics_file_path = os.path.join(source_folder_path, 'ee_csv.csv')
    video_files = [video_name for video_name in os.listdir(source_folder_path) if video_name.endswith('.mp4') and video_name.split(".")[0] in cameras_to_consider]
    
    # Parse labels
    labels = parse_labels(labels_file, phase_instruction_to_folder_name_mapping)
    if not labels:
        print("No labels found. Exiting.")
        return
    
    # Sort labels by timestamp
    labels.sort(key=lambda x: x[0])
    
    # Define splits based on labels
    splits = []
    for label_idx in range(len(labels)):
        instruction = labels[label_idx][1]
        if instruction == "pause":
            continue
        start_time = labels[label_idx][0]
        action = labels[label_idx][2]
        phase_folder_name = phase_instruction_to_folder_name_mapping[instruction]
        if action:
            phase_folder_name += f"-{action}"
        # Search for the next label with a different timestamp - as some labels have the same timestamps as they got multiple actions assigned
        end_time = None
        for next_idx in range(label_idx + 1, len(labels)):
            if labels[next_idx][0] != start_time:
                end_time = labels[next_idx][0]
                break
        
        splits.append((start_time, end_time, phase_folder_name))
    
    # Process each split
    for start_time, end_time, phase_folder_name in splits:
        phase_folder_path = os.path.join(tissue_folder, phase_folder_name)
        os.makedirs(phase_folder_path, exist_ok=True)
        curr_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        demo_folder_name = f"{curr_timestamp}_recovery" if "recovery" in phase_folder_name else curr_timestamp
        demo_folder_output_folder_path = os.path.join(phase_folder_path, demo_folder_name)
        os.makedirs(demo_folder_output_folder_path, exist_ok=True)
        
        # Split videos 
        num_camera_samples_list = []
        for video_name in video_files:
            video_path = os.path.join(source_folder_path, video_name)
            if os.path.exists(video_path):
                num_camera_samples = extract_demo_frames_from_video(video_path, demo_folder_output_folder_path, start_time, end_time, camera_file_name_suffix_dict, ll_policy_slowness_factor,
                                                                    presampling_duration, max_presampling_duration)
            else:
                raise ValueError(f"Video file not found at {video_path}. Exiting.")
        # Compare also number of frames across cameras
        if num_camera_samples_list and num_camera_samples != num_camera_samples_list[-1]:
            raise ValueError(f"Number of camera samples ({num_camera_samples}) does not match the previous number of camera samples ({num_camera_samples_list[-1] if num_camera_samples_list else None}). Exiting.")
        else:
            num_camera_samples_list.append(num_camera_samples)
        
        # Split kinematics
        if os.path.exists(kinematics_file_path):
            split_kinematics(kinematics_file_path, demo_folder_output_folder_path, start_time, end_time, num_camera_samples, ll_policy_slowness_factor,
                            presampling_duration, max_presampling_duration)
        else:
            raise ValueError(f"Kinematics file not found at {kinematics_file_path}. Exiting.")
        
        print(f"Data reorganized successfully into {demo_folder_output_folder_path}\n")
    
    print(f"\nData reorganized successfully into {tissue_folder}")

# ----------------------------- Execution -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reorganize Recorded Data into Structured Folders")
    
    # Experiment source folder
    default_experiment_name = "experiment_20240903-194630-415301"
    PATH_TO_YAY_ROBOT = os.getenv('PATH_TO_YAY_ROBOT')
    default_source_folder = os.path.join(PATH_TO_YAY_ROBOT, "experiment_recordings", default_experiment_name)
    parser.add_argument('--source', type=str, default=default_source_folder, help='Path to the source folder containing recordings')
    
    # Output dataset folder
    default_dataset_name = "experiments"
    default_dataset_dir = os.path.join(os.getenv("PATH_TO_DATASET"), default_dataset_name) 
    parser.add_argument('--output', type=str, default=default_dataset_dir, help='Path to the output dataset folder')
    
    # Camera suffix dictionary
    default_camera_file_name_suffix_dict = {"endo_psm2": "_psm2.jpg", "left_img_dir": "_left.jpg", "endo_psm1": "_psm1.jpg"}
    parser.add_argument('--camera_file_name_suffix_dict', type=dict, default=default_camera_file_name_suffix_dict, help='Dictionary mapping camera names to suffixes')
    
    # Cameras to consider
    default_cameras_to_consider = ["left_img_dir"] # ["endo_psm2", "left_img_dir", "endo_psm1"]
    parser.add_argument('--cameras_to_consider', type=list, default=default_cameras_to_consider, help='List of camera names to consider')
    
    # Add the slowness factor argument to the argparse section
    parser.add_argument('--ll_policy_slowness_factor', type=int, default=5, help='Factor by which to slow down the processing by taking every nth frame/kinematic point')
    
    # Add argument to specify the tissue number
    parser.add_argument('--tissue_number', type=int, default=None, help='Specify an existing tissue number to save new recordings without overwriting')
    
    # Add presampling duration argument
    parser.add_argument('--presampling_duration', type=int, default=4, help='Duration in seconds to presample before the start of the phase. Will be multiplied times the ll slowness factor')
    
    # Add max presampling duration argument
    parser.add_argument('--max_presampling_duration', type=int, default=20, help='Maximum duration in seconds (without ll slowness factor) to presample before the start of the phase')
    
    
    args = parser.parse_args()
    
    reorganize_data(args.source, args.output, args.camera_file_name_suffix_dict, args.cameras_to_consider, args.ll_policy_slowness_factor, 
                    phase_instruction_to_folder_name_mapping, args.tissue_number, args.presampling_duration, args.max_presampling_duration)
