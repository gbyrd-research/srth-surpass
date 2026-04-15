import os
import random
# import h5py
import json
import sys
from collections import defaultdict, deque
import math
import copy
import json

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
import albumentations as A
from torch.utils.data import DataLoader, ConcatDataset

# import src code
path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
else:
    raise EnvironmentError("Environment variable PATH_TO_YAY_ROBOT is not set")
from aloha_pro.aloha_scripts.utils import initialize_model_and_tokenizer, encode_text
from instructor.constants_daVinci import DATASET_CONFIGS, INSTRUMENT_CLOSED_THRESHOLD # get task parameters
from instructor.utils import randintgaussian, rotate_image, shift_image
from auto_label.auto_label import get_auto_label, get_all_auto_labels_list
    
def generate_command_embeddings(unique_phase_folder_names, encoder, tokenizer, model, reduced_base_instruction_set_flag):
    # Returns a dictionary containing the phase command as key and a tuple of the phase command and phase embedding as value
    phase_command_embeddings_dict = {}
    try: 
        unique_phase_folder_names_sorted = sorted(unique_phase_folder_names, key=lambda x: int(x.split('_')[0]))
    except:
        unique_phase_folder_names_sorted = unique_phase_folder_names
    
    if reduced_base_instruction_set_flag:
        # instruction_to_phase_idx_mapping = {
        #     "Apply first clip left tube": [1,2,3],
        #     "Apply second clip left tube": [4,5],
        #     "Apply third clip left tube": [6,7],
        #     "Cut left tube": [8,9],
        #     "Apply first clip right tube": [10,11],
        #     "Apply second clip right tube": [12,13],
        #     "Apply third clip right tube": [14,15],
        #     "Cut right tube": [16,17],
        # }
        instruction_to_phase_idx_mapping = {
            "Go back": [3,5,7,9,11,13,15,17],
        }
        phase_idx_to_instruction_mapping = {str(phase_idx): instruction for instruction, phase_idx_list in instruction_to_phase_idx_mapping.items() for phase_idx in phase_idx_list}
    for phase_folder_name in unique_phase_folder_names_sorted:
        # Extract the phase command from the folder name (removing the phase idx and the "_" in between the words) - No extra command for recovery phases
        phase_idx, phase_command = phase_folder_name.split("_")[0], " ".join(phase_folder_name.replace("_recovery", "").split("_")[1:])

        if reduced_base_instruction_set_flag:
            # Reduce base instruction set (keep finetuining instructions)
            if phase_idx in phase_idx_to_instruction_mapping:
                phase_command = phase_idx_to_instruction_mapping[phase_idx]
                
        embedding = encode_text(phase_command, encoder, tokenizer, model)
        phase_command_embeddings_dict[phase_folder_name]= (phase_command, embedding)

    return phase_command_embeddings_dict

def extract_candidate_embeddings_and_commands(command_embeddings_dict):
    # Extract the candidate embeddings and commands
    candidate_embeddings = []
    candidate_texts = []
    for _, (phase_command, phase_embedding) in command_embeddings_dict.items():
        if phase_command not in candidate_texts: # Only add unique commands
            candidate_texts.append(phase_command)
            candidate_embeddings.append(torch.tensor(phase_embedding).squeeze())
        
    return torch.stack(candidate_embeddings), candidate_texts

def extract_phase_idx_to_instruction_mapping(command_embeddings_dict):
    # Extract the instruction to phase index mapping
    instruction_to_phase_idx_mapping = defaultdict(list)
    for phase_idx, (phase_command, _) in command_embeddings_dict.items():
        instruction_to_phase_idx_mapping[phase_command].append(phase_idx)
    # Get the phase index to instruction mapping
    phase_to_instruction_mapping = {str(phase_idx): instruction for instruction, phase_idx_list in instruction_to_phase_idx_mapping.items() for phase_idx in phase_idx_list}
    
    return phase_to_instruction_mapping

def get_valid_demo_start_end_indices(demo_folder_path, before_phase_offset, after_phase_offset, use_kinematic_indices_flag=True):
    # Load the start and end indices for the current demo as the valid range of the demo
    demo_num_frames_total = len(os.listdir(os.path.join(demo_folder_path, "left_img_dir")))
    start, end = 0, demo_num_frames_total - 1
    indices_curated_file_path = os.path.join(demo_folder_path, "indices_curated.json")
    if os.path.exists(indices_curated_file_path):
        with open(indices_curated_file_path, 'r') as indices_curated_file:
            try:
                indices_curated_dict = json.load(indices_curated_file)
            except json.JSONDecodeError:
                print(f"Error reading indices_curated.json for {demo_folder_path}. Continue with max recording range.")
            if use_kinematic_indices_flag and "movement_start_idx" in indices_curated_dict:
                start = max(indices_curated_dict['movement_start_idx'] - before_phase_offset, start)
            elif "start" in indices_curated_dict:
                start = max(indices_curated_dict['start'] - before_phase_offset, start)
            if use_kinematic_indices_flag and "movement_end_idx" in indices_curated_dict:
                end = min(indices_curated_dict['movement_end_idx'] + after_phase_offset, end)
            elif "end" in indices_curated_dict:
                end = min(indices_curated_dict['end'] + after_phase_offset, end)
    
    demo_num_frames_valid = end - start + 1
    
    return start, end, demo_num_frames_valid

def get_all_recovery_samples_dict(dataset_dir):
    # Get all recovery samples - ordered by tissue and recovery phase
    recovery_samples_dict = {}
    tissue_folders = [file_name for file_name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, file_name))]
    for tissue_sample_name in tissue_folders:
        tissue_id = tissue_sample_name.split("_")[1]
        recovery_samples_dict[tissue_id] = {}
        tissue_sample_dir_path = os.path.join(dataset_dir, tissue_sample_name)
        recovery_phases = [file_name for file_name in os.listdir(tissue_sample_dir_path) if os.path.isdir(os.path.join(tissue_sample_dir_path, file_name)) and file_name.split('_')[0].isdigit() and "recovery" in file_name]
        for recovery_phase in recovery_phases:
            recovery_phase_dir_path = os.path.join(tissue_sample_dir_path, recovery_phase)
            demos = [demo_sample for demo_sample in os.listdir(recovery_phase_dir_path) if demo_sample[8] == "-"]
            if demos:    
                recovery_samples_dict[tissue_id][recovery_phase] = []
                for demo_sample in demos:
                    recovery_samples_dict[tissue_id][recovery_phase].append(demo_sample)
    return recovery_samples_dict

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_name,
        tissue_sample_names,
        dataset_dir,
        camera_names,
        camera_file_suffixes,
        history_len=4,
        prediction_offset=15,
        history_step_size=30,
        num_episodes=200,
        input_transforms=None,
        reduced_base_instruction_set_flag=False,
        use_phase_history_flag=False,
        use_jaw_values_flag=False,
        phase_history_len=6,
        prediction_step_size=30,
        recovery_probability=0.2,
        phase_history_only_phase_switches_flag=True,
        image_dim = (224, 224),
        llava_anyres_flag=False,
        no_llava_anyres_global_image_flag=False,
        wrist_images_rel_width = 3/4,
        llava_anyres_rel_width = 1/2,
        uniform_sampling_flag=False,
        selected_multitasks=[], # NOTE: Add here new multitasks: Possible multitasks: ["is_correction", "dominant_moving_direction", "psm1_instrument", "curr_tube", "total_number_clips", "clip_loaded", "gallbladder_grabbed", "clips_left_tube", "clips_right_tube", "psm2_instrument_closed", "psm1_instrument_closed"]
        use_seg_masks_input_flag=False,
        seg_mask_objs = ["clips", "left_tube", "right_tube"], # Possible segmentation mask objects: ["clips", "left_tube", "right_tube", "flap"]. NOTE: When merging order is expressing the priority, so what will be chosen when two masks overlap.
        merge_seg_masks_flag=False,
        use_kinematic_indices_flag=True,
        future_frame_prediction_flag=False,
        future_frame_delta_t=30,
        record_debug_episode_video_flag=False,
        extra_corrections_sampling_flag=False,
        extra_corrections_sampling_probability=0.1,
        extra_repeated_phase_last_frame_sampling_flag=False,
        extra_repeated_phase_last_frame_sampling_probability=0.1,
        add_center_crop_view_flag=False,
        distance_from_border_y=0.1,
        distance_from_border_x=0.25,
        y_offset=-0.1,
        rel_corr_idx_threshold=0.45, 
        verbose=False,
    ):
        super().__init__()
        
        if len(tissue_sample_names) == 0:
            raise ValueError("No tissue samples found in the dataset directory.")
        
        if llava_anyres_flag and llava_anyres_rel_width < 0.5:
            raise ValueError("The relative width of the llava anyres images must be at least 0.5.")
        
        if use_seg_masks_input_flag and camera_names != ["left_img_dir"]:
            raise ValueError("The segmentation mask input flag can only be used with the left image directory (as segmentation masks are only available for the left camera).")
        
        if add_center_crop_view_flag and llava_anyres_flag:
            print("Center crop can only be added for non-llava anyres images. So continuing without llava anyres.")
        
        if (add_center_crop_view_flag or llava_anyres_flag) and use_seg_masks_input_flag:
            raise ValueError("The center crop view does not work with seg masks yet.")
        
        self.split_name = split_name
        self.dataset_dir = dataset_dir
        self.image_dim = image_dim
        self.camera_names = camera_names
        self.camera_file_suffixes = camera_file_suffixes
        self.history_len = history_len
        self.prediction_offset = prediction_offset
        self.history_step_size = history_step_size
        self.num_episodes = num_episodes
        self.input_transforms = input_transforms
        self.reduced_base_instruction_set_flag = reduced_base_instruction_set_flag
        self.use_phase_history_flag = use_phase_history_flag
        self.use_jaw_values_flag = use_jaw_values_flag
        self.phase_history_len = phase_history_len
        self.prediction_step_size = prediction_step_size
        self.recovery_probability = recovery_probability
        self.phase_history_only_phase_switches_flag = phase_history_only_phase_switches_flag
        self.llava_anyres_flag = llava_anyres_flag if not add_center_crop_view_flag else False
        self.no_llava_anyres_global_image_flag = no_llava_anyres_global_image_flag
        self.wrist_images_rel_width = wrist_images_rel_width
        self.llava_anyres_rel_width = llava_anyres_rel_width
        self.uniform_sampling_flag = uniform_sampling_flag
        self.selected_multitasks = selected_multitasks
        self.use_segmentation_mask_input_flag = use_seg_masks_input_flag
        self.seg_mask_objs = seg_mask_objs
        self.merge_seg_masks_flag = merge_seg_masks_flag
        self.use_kinematic_indices_flag = use_kinematic_indices_flag
        self.future_frame_prediction_flag = future_frame_prediction_flag
        self.future_frame_delta_t = future_frame_delta_t
        self.record_debug_episode_video_flag = record_debug_episode_video_flag
        self.extra_corrections_sampling_flag = extra_corrections_sampling_flag
        self.extra_corrections_sampling_probability = extra_corrections_sampling_probability
        self.extra_repeated_phase_last_frame_sampling_flag = extra_repeated_phase_last_frame_sampling_flag
        if extra_corrections_sampling_flag and extra_repeated_phase_last_frame_sampling_flag:
            # Adjust the probability based on on the go back correction probability - to obtain the requested sampling likelihood (implementation reasons - see getitem)
            self.extra_repeated_phase_last_frame_sampling_probability = extra_repeated_phase_last_frame_sampling_probability / (1 - extra_corrections_sampling_probability) 
        else:
            self.extra_repeated_phase_last_frame_sampling_probability = extra_repeated_phase_last_frame_sampling_probability
        self.add_center_crop_view_flag = add_center_crop_view_flag
        self.distance_from_border_y = distance_from_border_y
        self.distance_from_border_x = distance_from_border_x
        self.y_offset = y_offset
        self.verbose = verbose
        self.rel_corr_idx_threshold = rel_corr_idx_threshold
        
        # Set the before_phase_offset and after_phase_offset
        dataset_name = os.path.basename(dataset_dir)
        dataset_config = DATASET_CONFIGS[dataset_name]
        self.before_phase_offset = dataset_config["before_phase_offset"]
        self.after_phase_offset = dataset_config["after_phase_offset"]
        self.correct_psm1_rotation_tissues = dataset_config["correct_psm1_rotation_tissues"] if "correct_psm1_rotation_tissues" in dataset_config else []
        self.incomplete_demos_flag = dataset_config["incomplete_demos_flag"]
        self.tissue_samples_old_grab_pull_separation = dataset_config["tissue_samples_old_grab_pull_separation"] if "tissue_samples_old_grab_pull_separation" in dataset_config else []
 
        # Initialize the phase_len_dict with defaultdict
        phase_len_dict = defaultdict(list)

        if extra_corrections_sampling_flag or "is_correction" in selected_multitasks:
            # List of correction demos
            corrections_json_file_path = os.path.join(self.dataset_dir, "corrections.json")
            if os.path.exists(corrections_json_file_path):
                with open(corrections_json_file_path, 'r') as file:
                    self.corrections_dict = json.load(file)
            else: # Use all recovery samples as corrections samples
                self.corrections_dict = get_all_recovery_samples_dict(self.dataset_dir)
                
        elif selected_multitasks:
            self.corrections_dict = None


        # Initialize tissue_phase_demo_dict
        self.tissue_phase_demo_dict = {}

        for tissue_sample_name in tissue_sample_names:
            tissue_sample_dir_path = os.path.join(dataset_dir, tissue_sample_name)
            phases = [file_name for file_name in os.listdir(tissue_sample_dir_path) if os.path.isdir(os.path.join(tissue_sample_dir_path, file_name)) and file_name.split('_')[0].isdigit()]
            phases_ordered = sorted(phases, key=lambda x: int(x.split('_')[0]))
            self.tissue_phase_demo_dict[tissue_sample_name] = {}
            for phase_sample in phases_ordered:
                files_in_phase_folder = os.listdir(os.path.join(tissue_sample_dir_path, phase_sample))
                # demo_samples = [demo_sample for demo_sample in files_in_phase_folder if demo_sample[8] == "-"]  
                demo_samples = files_in_phase_folder              
                self.tissue_phase_demo_dict[tissue_sample_name][phase_sample] = demo_samples
                # Add the length of the phase for current demo to phase_len_dict
                for demo_sample in demo_samples:
                    demo_num_frames_valid = get_valid_demo_start_end_indices(os.path.join(tissue_sample_dir_path, phase_sample, demo_sample), self.before_phase_offset, self.after_phase_offset, use_kinematic_indices_flag)[2]
                    phase_len_dict[phase_sample].append(demo_num_frames_valid)               
            
        # Generate the embeddings for all phase commands
        encoder_name = "distilbert"
        tokenizer, model = initialize_model_and_tokenizer(encoder_name)
        unique_phase_folder_names = np.unique([phase_folder_name for tissue_sample in self.tissue_phase_demo_dict.values() for phase_folder_name in tissue_sample.keys()]).tolist()
        self.command_embeddings_dict = generate_command_embeddings(unique_phase_folder_names, encoder_name, tokenizer, model, reduced_base_instruction_set_flag) 
        del tokenizer, model
        
        # Compute the dataset statistics
        self.ds_statistics_dict = self.compute_dataset_statistics(phase_len_dict)
        
        # Add resize transform
        self.resize = transforms.Resize(self.image_dim, antialias=True)
        
        # Add camera patch names
        self.camera_patch_names = self.get_camera_patch_names(camera_names, add_center_crop_view_flag, llava_anyres_flag, no_llava_anyres_global_image_flag)
    
    
    def __len__(self):
        # Here this means the number of randomly generated stitched episodes
        return self.num_episodes

    def compute_dataset_statistics(self, phase_len_dict):
        # Compute the statistics of the dataset
        ds_statistics_dict = {}
        for phase_name, phase_len_list in phase_len_dict.items():
            ds_statistics_dict[phase_name] = {
                "min": min(phase_len_list),
                "max": max(phase_len_list),
                "mean": sum(phase_len_list) / len(phase_len_list),
                "std": np.std(phase_len_list),
                "num_demos": len(phase_len_list),
            }
            
        # Compute statistics for merged phases
        if self.reduced_base_instruction_set_flag:
            # Merge the phases from the self.command_embeddings_dict
            for phase_name, (phase_command, _) in self.command_embeddings_dict.items():
                if phase_command in ds_statistics_dict:
                    ds_statistics_dict[phase_command]["min"] = ds_statistics_dict[phase_command]["min"] + ds_statistics_dict[phase_name]["min"]
                    ds_statistics_dict[phase_command]["max"] = ds_statistics_dict[phase_command]["max"] + ds_statistics_dict[phase_name]["max"]
                    ds_statistics_dict[phase_command]["mean"] = ds_statistics_dict[phase_command]["mean"] + ds_statistics_dict[phase_name]["mean"]
                    ds_statistics_dict[phase_command]["std"].append(ds_statistics_dict[phase_name]["std"])
                    ds_statistics_dict[phase_command]["num_demos"] *= ds_statistics_dict[phase_name]["num_demos"] 
                else:
                    ds_statistics_dict[phase_command] = ds_statistics_dict[phase_name].copy()
                    ds_statistics_dict[phase_command]["std"] = [ds_statistics_dict[phase_name]["std"]]
            
            phase_commands_list = [phase_command for phase_command, _ in self.command_embeddings_dict.values()]
            for phase_command in phase_commands_list:
                ds_statistics_dict[phase_command]["std"] = np.sqrt(np.sum(np.square(np.array(ds_statistics_dict[phase_command]["std"])))) # Std of Gaussian distribution is sqrt of sum of squares of stds
            
        return ds_statistics_dict

    @staticmethod
    def get_camera_patch_names(camera_names, add_center_crop_view_flag, llava_anyres_flag, no_llava_anyres_global_image_flag):
        # Returns the camera patch names that are extracted from the camera images
        if add_center_crop_view_flag:
            camera_patch_names = []
            for camera_name in camera_names:
                if camera_name in ["endo_psm2", "endo_psm1"]:
                    camera_patch_names.append(camera_name)
                else:
                    camera_patch_names.append(f"{camera_name}_global")
                    camera_patch_names.append(f"{camera_name}_center")
        elif not llava_anyres_flag:
            camera_patch_names = camera_names
        else:
            camera_patch_names = []
            for camera_name in camera_names:
                if camera_name in ["endo_psm2", "endo_psm1"]:
                    camera_patch_names.append(camera_name)
                else:
                    if not no_llava_anyres_global_image_flag:
                        curr_camera_patch_names = [f"{camera_name}_global", f"{camera_name}_left", f"{camera_name}_right"]
                    else:
                        curr_camera_patch_names = [f"{camera_name}_left", f"{camera_name}_right"]
                    camera_patch_names.extend(curr_camera_patch_names)
        return camera_patch_names
    
    @staticmethod
    def get_num_patches(camera_names, add_center_crop_view_flag, llava_anyres_flag, no_llava_anyres_global_image_flag):
        # Returns the number of patches that are extracted from the camera images
        return len(SequenceDataset.get_camera_patch_names(camera_names, add_center_crop_view_flag, llava_anyres_flag, no_llava_anyres_global_image_flag))
        
    def get_embedding_command_phase_for_ts(self, selected_phase_demo_dict, target_ts):
        # Returns the command embedding and the command for the target timestep
        for phase_segment in selected_phase_demo_dict.values():
            if phase_segment["phases_demo_start_idx"] <= target_ts <= phase_segment["phases_demo_end_idx"]:
                return torch.tensor(phase_segment["embedding"]).squeeze(), phase_segment["command"], phase_segment["phase_folder_name"]
        else:
            return None, None, None

    def get_current_phase_demo_folder_and_demo_frame_idx(self, selected_phase_demo_dict, target_ts):
        # Returns the phase and the demo frame index for the target timestep
        for phase_segment in selected_phase_demo_dict.values():
            if phase_segment["phases_demo_start_idx"] <= target_ts <= phase_segment["phases_demo_end_idx"]:
                demo_frame_idx = target_ts - phase_segment["phases_demo_start_idx"] + phase_segment["demo_rel_start_idx"]
                return phase_segment["phase_folder_name"], phase_segment["demo_folder_name"], demo_frame_idx
        else:
            raise ValueError(f"Could not find phase and demo frame index for target_ts {target_ts}.")

    def create_phase_demo_metadata_dict(self, selected_tissue_sample, selected_phases, selected_phase_demos):
        # Returns a dictionary containing the selected phase demos and the start and end timestep, and further metadata   
        selected_phase_demo_dict = {}
        episode_num_frames = 0
        next_phase_start_idx_counter = 0
        for phase, selected_phase_demo in zip(selected_phases, selected_phase_demos):
            # Add the selected phase demo to the selected_phase_demo_dict
            selected_phase_demo_dict[phase] = {}
            selected_phase_demo_dict[phase]["phase_folder_name"] = phase
            selected_phase_demo_dict[phase]["demo_folder_name"] = selected_phase_demo
            selected_phase_demo_dict[phase]["phases_demo_start_idx"] = next_phase_start_idx_counter
            selected_phase_demo_dict[phase]["command"], selected_phase_demo_dict[phase]["embedding"] = self.command_embeddings_dict[phase]
            
            # Load the start and end indices for the current demo as the valid range of the demo
            selected_demo_folder_path = os.path.join(self.dataset_dir, selected_tissue_sample, phase, selected_phase_demo)
            start, _, demo_num_frames_valid = get_valid_demo_start_end_indices(selected_demo_folder_path, self.before_phase_offset, self.after_phase_offset, self.use_kinematic_indices_flag)
            selected_phase_demo_dict[phase]["demo_rel_start_idx"] = start
            
            # Count the number of valid frames for the current demo
            episode_num_frames += demo_num_frames_valid
            next_phase_start_idx_counter += demo_num_frames_valid
            selected_phase_demo_dict[phase]["phases_demo_end_idx"] = next_phase_start_idx_counter - 1 # -1 because the phases_demo_end_idx is inclusive

        return selected_phase_demo_dict

    def get_jaw_psm2_psm1_data_sequence(self, selected_tissue_sample, selected_phase_demo_dict, start_ts, curr_ts, history_step_size=None):
        # Returns the jaw data sequence for psm2 and psm1 for time steps from start_ts to curr_ts
        
        if not history_step_size:
            history_step_size = self.history_step_size
        
        # Get a mapping which value needs to be loaded from which csv file
        ts_kinematics_file_dame_frame_idx_dict = defaultdict(list)
        for ts in range(start_ts, curr_ts + 1, history_step_size):
            ts_corrected = max(0, ts) # If the episode is not long enough replicate the last frame/yaw value for the first recorded demo
            ts_phase_folder, ts_demo_folder, ts_demo_frame_idx = self.get_current_phase_demo_folder_and_demo_frame_idx(selected_phase_demo_dict, ts_corrected)
            kinematics_file_path = os.path.join(self.dataset_dir, selected_tissue_sample, ts_phase_folder, ts_demo_folder, "ee_csv.csv")
            ts_kinematics_file_dame_frame_idx_dict[kinematics_file_path].append(ts_demo_frame_idx)
        
        # Load the jaw data for the desired timesteps (from the corresponding csv files)
        jaw_psm2_data_sequence_list, jaw_psm1_data_sequence_list = [], []
        for kinematics_file_path, frame_indices in ts_kinematics_file_dame_frame_idx_dict.items():
            kinematics_data = pd.read_csv(kinematics_file_path)
            jaw_psm2_data_sequence_list.append(torch.tensor(kinematics_data.loc[frame_indices, "psm2_jaw"].values))
            jaw_psm1_data_sequence_list.append(torch.tensor(kinematics_data.loc[frame_indices, "psm1_jaw"].values))
        kinematics_data_curr_ts = kinematics_data
        
        jaw_psm2_data_sequence = torch.concatenate(jaw_psm2_data_sequence_list)
        jaw_psm1_data_sequence = torch.concatenate(jaw_psm1_data_sequence_list)
        jaw_psm2_psm1_data_sequence = torch.stack((jaw_psm2_data_sequence, jaw_psm1_data_sequence), dim=1).to(dtype=torch.float32)
        
        return jaw_psm2_psm1_data_sequence, kinematics_data_curr_ts
            
    def get_phase_history(self, selected_phase_demo_dict, sorted_phases, curr_ts, correction_flag=False):
        # Returns the phase history for the last six performed phases or phase predictions (with padding if needed)
        
        # Compute the last prediction timestep (until which the history will be considered)
        last_pred_ts = curr_ts - self.prediction_step_size if not correction_flag else curr_ts
        
        # Get the last n performed phases from the last prediction timestep on
        if self.phase_history_only_phase_switches_flag:
            # Get the last n performed phases from the current phase on
            last_pred_ts_phase = self.get_embedding_command_phase_for_ts(selected_phase_demo_dict, last_pred_ts)[2]
            if last_pred_ts_phase:
                last_pred_ts_phase_idx = sorted_phases.index(last_pred_ts_phase.replace("_recovery", ""))
            else: 
                curr_ts_phase = self.get_embedding_command_phase_for_ts(selected_phase_demo_dict, curr_ts)[2]
                last_pred_ts_phase_idx = sorted_phases.index(curr_ts_phase.replace("_recovery", "")) if curr_ts_phase else 0 
            phase_history_phase_folder_names = sorted_phases[max(0, last_pred_ts_phase_idx - self.phase_history_len + 1):last_pred_ts_phase_idx+1]
            phase_history = [] # Transform to the commands corresponding to the phase folder names
            for phase_folder_name in phase_history_phase_folder_names:
                if phase_folder_name not in self.command_embeddings_dict:
                    recovery_phase_folder_name = f"{phase_folder_name}_recovery"
                    phase_command = self.command_embeddings_dict[recovery_phase_folder_name][0]
                else:
                    phase_command = self.command_embeddings_dict[phase_folder_name][0]
                if not phase_history or phase_command != phase_history[-1]:
                    phase_history.append(phase_command)
            # Remove replicant merged phases
            if self.reduced_base_instruction_set_flag:
                phase_history = [phase for idx, phase in enumerate(phase_history) if idx == 0 or phase != phase_history[idx-1]]
            
            # Add padding (if needed)
            if len(phase_history) < self.phase_history_len:
                phase_history = ["padding"] * (self.phase_history_len - len(phase_history)) + phase_history
        else:            
            # Check first if we need to add further sample number information of precessor phases to phase demo dict (if not everything covered)
            first_phase_partially_stitched_episode = list(selected_phase_demo_dict.keys())[0].replace("_recovery", "")
            min_episode_idx = 0 # NOTE: Will be < 0 for the new precessor phases
            remaining_precessor_phases_reversed = sorted_phases[:sorted_phases.index(first_phase_partially_stitched_episode)][::-1]
            selected_phase_demo_dict_with_pre_phases = selected_phase_demo_dict.copy()
            for precessor_phase in remaining_precessor_phases_reversed:
                num_samples_til_pred_ts = last_pred_ts - min_episode_idx
                if num_samples_til_pred_ts >= self.prediction_step_size*self.phase_history_len:
                    break
                # Sample the length of the precessor phase
                precessor_phase_len_min = self.ds_statistics_dict[precessor_phase]["min"]
                precessor_phase_len_max = self.ds_statistics_dict[precessor_phase]["max"]
                precessor_phase_len_mean = self.ds_statistics_dict[precessor_phase]["mean"]
                precessor_phase_len_std = self.ds_statistics_dict[precessor_phase]["std"]
                precessor_phase_len = randintgaussian(precessor_phase_len_min, precessor_phase_len_max, mean=precessor_phase_len_mean, std_dev=precessor_phase_len_std) 
                min_episode_idx -= precessor_phase_len
                
                # Add the precessor phase to the phase demo dict
                selected_phase_demo_dict_with_pre_phases[precessor_phase] = {}
                selected_phase_demo_dict_with_pre_phases[precessor_phase]["phase_folder_name"] = precessor_phase
                selected_phase_demo_dict_with_pre_phases[precessor_phase]["phases_demo_start_idx"] = min_episode_idx
                selected_phase_demo_dict_with_pre_phases[precessor_phase]["phases_demo_end_idx"] = min_episode_idx + precessor_phase_len - 1
                selected_phase_demo_dict_with_pre_phases[precessor_phase]["command"], selected_phase_demo_dict_with_pre_phases[precessor_phase]["embedding"] = self.command_embeddings_dict[precessor_phase]
            
            # Get the last n performed phases from the current phase on
            phase_history = []
            pred_ts_limit = last_pred_ts - self.prediction_step_size*self.phase_history_len
            for pred_ts in range(last_pred_ts, pred_ts_limit, -self.prediction_step_size):
                pred_ts_command = self.get_embedding_command_phase_for_ts(selected_phase_demo_dict_with_pre_phases, pred_ts)[1]
                if pred_ts_command:
                    phase_history.append(pred_ts_command)
                else: 
                    phase_history.append("padding")
            # Reverse the phase history order at the end (as newest predictions should be last)
            phase_history = phase_history[::-1]
                            
        return phase_history

    def get_image_sequence(self, selected_tissue_sample, selected_phase_demo_dict, image_timesteps):
        # Construct the image sequences for the desired timesteps
        image_sequence = []
        for ts in image_timesteps:
            ts_corrected = max(0, ts) # If the episode is not long enough replicate the last frame/yaw value for the first recorded demo
            image_dict = {}
            ts_phase_folder, ts_demo_folder, ts_demo_frame_idx = self.get_current_phase_demo_folder_and_demo_frame_idx(selected_phase_demo_dict, ts_corrected)
            for cam_name, cam_file_suffix in zip(self.camera_names, self.camera_file_suffixes):
                demo_folder_path = os.path.join(self.dataset_dir, selected_tissue_sample, ts_phase_folder, ts_demo_folder)
                cam_folder = os.path.join(demo_folder_path, cam_name)
                frame_path = os.path.join(cam_folder, f"frame{str(ts_demo_frame_idx).zfill(6)}{cam_file_suffix}")
                img = torch.tensor(cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                
                # Add segmentation mask input for the left wrist cam as additional input channel (if desired)
                if cam_name == "left_img_dir" and self.use_segmentation_mask_input_flag:
                    seg_masks_folder_path = os.path.join(demo_folder_path, "seg_masks")
                    img_and_seg_masks_list = [img]  
                    for seg_mask_obj in self.seg_mask_objs:    
                        seg_mask_path = os.path.join(seg_masks_folder_path, f"frame{str(ts_demo_frame_idx).zfill(6)}_{seg_mask_obj}.jpg")
                        if not os.path.exists(seg_mask_path):
                            seg_mask = torch.zeros(1, img.shape[1], img.shape[2]).to(dtype=torch.uint8)
                        else:
                            seg_mask = torch.tensor(cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)).unsqueeze(0) / 255
                        seg_mask = torch.where(seg_mask > 0.5, torch.tensor(1), torch.tensor(0)).to(dtype=torch.uint8) # Map between 0 and 1
                        img_and_seg_masks_list.append(seg_mask)
                    img = torch.cat(img_and_seg_masks_list, dim=0)
                
                # Apply wrist rotation function on certain tissues of PSM1 - where instrument is not centered
                if cam_name == "endo_psm1" and selected_tissue_sample in self.correct_psm1_rotation_tissues:
                    # Rectify rotation of the right wrist cam
                    angle = -52.0
                    img_reshaped = img.permute(1, 2, 0).numpy()
                    img = rotate_image(img_reshaped, angle)
                    shift_x, shift_y = 10, 0 
                    img = torch.tensor(shift_image(img, shift_x, shift_y)).permute(2, 0, 1)
                
                # Add the image to the image dictionary
                if self.add_center_crop_view_flag or self.llava_anyres_flag:
                    img_patches_list = []
                if self.add_center_crop_view_flag and cam_name not in ["endo_psm1", "endo_psm2"]:                     
                    # Add global image
                    global_img_resized = self.resize(img) 
                    img_patches_list.append(global_img_resized)
                    
                    # Add center crop image
                    center_crop_y_min, center_crop_y_max = int(img.shape[1] * (self.distance_from_border_y - self.y_offset)), int(img.shape[1] * (1-self.distance_from_border_y-self.y_offset)) 
                    center_crop_x_min, center_crop_x_max = int(img.shape[2] * self.distance_from_border_x), int(img.shape[2] * (1-self.distance_from_border_x))
                    center_part_of_img = self.resize(img[:, center_crop_y_min:center_crop_y_max, center_crop_x_min:center_crop_x_max])
                    img_patches_list.append(center_part_of_img)
                elif self.llava_anyres_flag and not self.add_center_crop_view_flag and cam_name not in ["endo_psm2", "endo_psm1"]:                    
                    # Add global image (if desired - only for DaVinci camera)
                    if not self.no_llava_anyres_global_image_flag: 
                        global_img_resized = self.resize(img) 
                        img_patches_list.append(global_img_resized)
                    
                    # If llava_anyres_rel_width > 0.5 then having overlapping patches 
                    split_idx = int(img.shape[2] * self.llava_anyres_rel_width)
                    img_left_part = self.resize(img[:, :, :split_idx])
                    img_right_part = self.resize(img[:, :, -split_idx:])
                    img_patches_list.append(img_left_part)
                    img_patches_list.append(img_right_part)                           
                else:
                    if cam_name == "endo_psm2":
                        split_idx = int(img.shape[2] * self.wrist_images_rel_width)
                        img = img[:, :, -split_idx:]
                        if self.add_center_crop_view_flag or self.llava_anyres_flag:
                            img_patches_list.append(self.resize(img))
                    elif cam_name == "endo_psm1":
                        split_idx = int(img.shape[2] * self.wrist_images_rel_width)
                        img = img[:, :, :split_idx]
                        if self.add_center_crop_view_flag or self.llava_anyres_flag:
                            img_patches_list.append(self.resize(img))
                    
                if self.add_center_crop_view_flag or self.llava_anyres_flag:
                    final_img = torch.stack(img_patches_list, dim=0) # Shape: num_patches, c, h, w
                else:
                    # Resize the image to desired image size
                    final_img = self.resize(img) # Shape: c, h, w
                
                image_dict[cam_name] = final_img
                
            all_cam_images = [
                image_dict[cam_name] for cam_name in self.camera_names
            ]
            if self.llava_anyres_flag or self.add_center_crop_view_flag:
                all_cam_images = torch.concat(all_cam_images, dim=0)
            else:
                all_cam_images = torch.stack(all_cam_images, dim=0)
            image_sequence.append(all_cam_images)

        # Apply the same transform for all camera images 
        image_sequence = torch.stack(image_sequence, dim=0) # Shape: ts, cam (+ its patches), c, h, w
        if self.split_name == "train" and self.input_transforms["torch"] is not None and self.input_transforms["torch"] is not None:
            image_sequence = image_sequence.reshape(-1, *image_sequence.shape[-3:]) # Reshape to (ts*cam (+ its patches), c, h, w) for applying the same transform to all camera images
            if self.input_transforms["torch"] is not None:
                image_sequence = self.input_transforms["torch"](image_sequence)
            if self.input_transforms["torch_color"] is not None:
                # Apply color augmentations only on the RGB images
                image_sequence[:, :3, :, :] = self.input_transforms["torch_color"](image_sequence[:, :3, :, :])
            if self.input_transforms["albumentations"] is not None:
                images_dict = dict(zip(["image"]+[f"image{img_idx}" for img_idx in range(image_sequence.shape[0]-1)], list(np.array(image_sequence.permute(0, 2, 3, 1)))))
                images_transformed_dict = self.input_transforms["albumentations"](**images_dict)
                image_sequence = torch.tensor(np.stack(list(images_transformed_dict.values()), axis=0)).permute(0, 3, 1, 2)
            image_sequence = image_sequence.reshape(-1, len(self.camera_patch_names), *image_sequence.shape[-3:]) # Reshape back to (ts, cam (+ its patches), c, h, w)
            
        # Scale the RGB images to [0, 1]
        image_sequence = image_sequence.to(dtype=torch.float32)
        image_sequence[:, :, :3, :, :] = image_sequence[:, :, :3, :, :] / 255.0

        # If desired, merge the segmentation masks for to just one additional channel
        if self.use_segmentation_mask_input_flag and self.merge_seg_masks_flag:
            batch_size, num_cam_patches, num_channels, h, w = image_sequence.shape
            merged_seg_mask = torch.zeros(batch_size, num_cam_patches, 1, h, w).to(dtype=torch.uint8)
            seg_mask_channels = image_sequence[:, :, 3:, :, :]
            num_seg_channels = seg_mask_channels.shape[2]
            for class_id in range(1, num_seg_channels+1): 
                seg_mask = seg_mask_channels[:, :, class_id-1, :, :].unsqueeze(2) * class_id
                # Merge the segmentation masks based on the priority of the objects (lower the higher, except for 0)
                merged_seg_mask = torch.where((merged_seg_mask == 0) & (seg_mask != 0), seg_mask, merged_seg_mask)
                merged_seg_mask = torch.where((merged_seg_mask != 0) & (seg_mask != 0) & (seg_mask < merged_seg_mask), seg_mask, merged_seg_mask)
            image_sequence = torch.cat([image_sequence[:,:,:3], merged_seg_mask], dim=2)   
            
        return image_sequence                

    # -------------- Multitask Label Generation --------------

    @staticmethod
    def get_psm1_instrument_label(curr_phase):
        # Returns the label for the psm1 instrument for the current timestep
        # Clip applier: 0, curved scissor: 1
        if "cut" in curr_phase:
            return 1
        else: # so for clipping + gallbladder
            return 0

    @staticmethod
    def get_curr_tube_label(curr_phase):
        # Returns the label for the current tube (that is being worked on) for the current timestep
        # Left tube: 0, right tube: 1
        if "right" in curr_phase:
            return 1
        else: # also for grabbing gallbladder
            return 0
    
    @staticmethod
    def get_total_number_clips_label(curr_phase, tisssue_sample_name, demo_sample_name):
        num_clips_to_phase_idx_mapping = {1: [1,2,3], 2: [4,5], 3: [6,7,8,9], 4: [10,11], 5: [12,13], 6: [14,15,16,17]}
        phase_idx_to_num_clips_mapping = {phase_idx: num_clips for num_clips, phase_idx_list in num_clips_to_phase_idx_mapping.items() for phase_idx in phase_idx_list}
        curr_phase_idx = int(curr_phase.split("_")[0])
        if tisssue_sample_name == "tissue_5" and curr_phase_idx == 1 and not demo_sample_name == "20240710-180855-456630":
            return 0 # Exception for tissue 5 (except of last grabbing gallbladder phase demo)
        else:
            return phase_idx_to_num_clips_mapping[curr_phase_idx]
    
    @staticmethod
    def get_clip_loaded_label(curr_phase, tisssue_sample_name, demo_sample_name, curr_jaw_values=None, jaw_closed_open_thresh=0):
        # Returns the label for the clip loaded for the current timestep
        curr_phase_idx = int(curr_phase.split("_")[0])
        if "clipping" in curr_phase: # Always loaded when in clipping phase
            return 1
        elif curr_phase_idx == 1 and not (tisssue_sample_name == "tissue_5" and not demo_sample_name == "20240710-180855-456630"):
            return 1
        elif "going back" in curr_phase and "clip" in curr_phase and curr_jaw_values[1] <= jaw_closed_open_thresh: # Partially also loaded in going back until the jaw is openend
            return 1 
        else:
            return 0
       
    @staticmethod 
    def get_gallbladder_grabbed_label(curr_phase, abs_phase_start_idx, abs_phase_end_idx, curr_ts, curr_jaw_values, jaw_closed_open_thresh=0, rel_phase_ts_idx_thresh=0.3):
        # Returns the label for the gallbladder grabbed for the current timestep
        if "gallbladder" in curr_phase:
            phase_ts_idx = curr_ts - abs_phase_start_idx
            phase_end_idx = abs_phase_end_idx - abs_phase_start_idx
            rel_phase_ts_idx = phase_ts_idx / phase_end_idx
        
        if not "gallbladder" in curr_phase:
            return 1
        elif "gallbladder" in curr_phase and curr_jaw_values[0] <= jaw_closed_open_thresh and rel_phase_ts_idx > rel_phase_ts_idx_thresh:
            return 1 # Gallbladder grabbed - in gallbladder grabbing phase - relative phase ts idx to filter out recovery data starting from closed jaw
        else:
            return 0
       
    @staticmethod 
    def get_clips_on_tube_label(tube_side, curr_phase, curr_ts, abs_phase_start_idx, abs_phase_end_idx, curr_jaw_values=None, jaw_closed_open_thresh=0, rel_phase_ts_idx_thresh=0.5):
        # Returns the label for the clips on the tube for the current timestep
        
        # If recovery treat the kinematic data differently - only look at the second part (ignore closed starting position)
        if "recovery" in curr_phase:
            phase_ts_idx = curr_ts - abs_phase_start_idx
            phase_end_idx = abs_phase_end_idx - abs_phase_start_idx
            rel_phase_ts_idx = phase_ts_idx / phase_end_idx
        
        if tube_side == "left":
            initial_num_clips_on_left_tube_to_phase_idx_mapping = {0: [1,2], 1: [3,4], 2: [5,6], 3: [7,8,9,10,11,12,13,14,15,16,17]}
            phase_idx_to_initial_num_clips_on_left_tube_mapping = {phase_idx: num_clips for num_clips, phase_idx_list in initial_num_clips_on_left_tube_to_phase_idx_mapping.items() for phase_idx in phase_idx_list}
            curr_phase_idx = int(curr_phase.split("_")[0])
            if "clipping" in curr_phase and "left" in curr_phase and curr_jaw_values[1] <= jaw_closed_open_thresh and ((not "recovery" in curr_phase) or ("recovery" in curr_phase and rel_phase_ts_idx > rel_phase_ts_idx_thresh)):
                return phase_idx_to_initial_num_clips_on_left_tube_mapping[curr_phase_idx] + 1 # When clipped then one more clip on the tube, then at the beginning of clipping phase
            else:
                return phase_idx_to_initial_num_clips_on_left_tube_mapping[curr_phase_idx]
        elif tube_side == "right":
            initial_num_clips_on_right_tube_to_phase_idx_mapping = {0: [1,2,3,4,5,6,7,8,9,10], 1: [11,12], 2: [13,14], 3: [15,16,17]}
            phase_idx_to_initial_num_clips_on_right_tube_mapping = {phase_idx: num_clips for num_clips, phase_idx_list in initial_num_clips_on_right_tube_to_phase_idx_mapping.items() for phase_idx in phase_idx_list}
            curr_phase_idx = int(curr_phase.split("_")[0])
            if "clipping" in curr_phase and "right" in curr_phase and curr_jaw_values[1] <= jaw_closed_open_thresh and ((not "recovery" in curr_phase) or ("recovery" in curr_phase and rel_phase_ts_idx > rel_phase_ts_idx_thresh)):
                return phase_idx_to_initial_num_clips_on_right_tube_mapping[curr_phase_idx] + 1
            else:
                return phase_idx_to_initial_num_clips_on_right_tube_mapping[curr_phase_idx]
        else:
            raise ValueError(f"Tube side {tube_side} not supported.")
         
    @staticmethod
    def get_dominant_moving_direction_label(ee_csv_data, ts_demo_frame_idx, curr_jaw_values, curr_phase, curr_ts, abs_phase_start_idx, 
                                            abs_phase_end_idx, cutting_chunk_size, image_delay_offset, jaw_closing_opening_thresh):        
        # Returns the label for the dominant moving direction for the current timestep
        
        # Get the mapping from the auto labels to the label indices
        all_auto_labels_list = get_all_auto_labels_list()
        dominant_moving_direction_to_idx_mapping = dict(zip(all_auto_labels_list, range(len(all_auto_labels_list))))
        
        # Get the dominant moving direction for cutting (as the final kinematics for the cutting are missing)
        indices_to_end = abs_phase_end_idx - curr_ts
        if "cutting" in curr_phase and indices_to_end <= cutting_chunk_size:
            dominant_moving_direction = "close right gripper"
            dominant_moving_direction_idx = dominant_moving_direction_to_idx_mapping[dominant_moving_direction]
            return dominant_moving_direction_idx
        
        # Returns the dominant direction that it will move next to
        phase_ts_idx = curr_ts - abs_phase_start_idx
        phase_end_idx = abs_phase_end_idx - abs_phase_start_idx
        rel_phase_ts_idx = phase_ts_idx / phase_end_idx
            
        # Add negative image delay offset
        ts_demo_frame_idx = max(0, ts_demo_frame_idx - image_delay_offset)
            
        # Get the dominant moving direction
        base_chunk_size = 10
        for chunk_size_factor in range(1, 5):
            chunk_size = base_chunk_size * chunk_size_factor
            dominant_moving_direction = get_auto_label(ee_csv_data, ts_demo_frame_idx, chunk_size=chunk_size, jaw_threshold=jaw_closing_opening_thresh)
            if dominant_moving_direction != "do not move":
                break
        dominant_moving_direction_idx = dominant_moving_direction_to_idx_mapping[dominant_moving_direction]
        
        return dominant_moving_direction_idx
            
    @staticmethod
    def get_is_correction_label(corrections_dict, tissue_sample_name, curr_phase, demo_sample_name, curr_ts, abs_phase_start_idx, abs_phase_end_idx, 
                                rel_corr_idx_thresh=0.45):
        # Returns a label expressing if this is a correction at the current timestep
        
        # Compute the relative phase timestep index
        phase_ts_idx = curr_ts - abs_phase_start_idx
        phase_end_idx = abs_phase_end_idx - abs_phase_start_idx
        rel_phase_ts_idx = phase_ts_idx / phase_end_idx
        
        tissue_idx = tissue_sample_name.split("_")[-1]
        tissue_phase_correction_samples = corrections_dict.get(tissue_idx, {}).get(curr_phase, [])
        # Check if the current timestep is a correction
        if demo_sample_name in tissue_phase_correction_samples and rel_phase_ts_idx <= rel_corr_idx_thresh:
            return 1
        else:
            return 0
    
    @staticmethod
    def get_multitask_labels_dict(selected_multitasks, curr_ts, curr_phase, abs_phase_start_idx, abs_phase_end_idx, curr_jaw_values, selected_tisssue_name, selected_demo_name, 
                                  jaw_closed_open_thresh=0, rel_phase_ts_idx_thresh=0.3, rel_corr_idx_thresh=0.45, ee_csv_data=None, ts_demo_frame_idx=None, corrections_dict=None,
                                  cutting_chunk_size=20, image_delay_offset=3, jaw_closing_opening_thresh=0.2): 
        # Returns the multitask labels for the selected multitasks
        multitask_labels_dict = {}
        if "psm1_instrument" in selected_multitasks:
            multitask_labels_dict["psm1_instrument"] = SequenceDataset.get_psm1_instrument_label(curr_phase)
        if "curr_tube" in selected_multitasks:
            multitask_labels_dict["curr_tube"] = SequenceDataset.get_curr_tube_label(curr_phase)
        if "total_number_clips" in selected_multitasks:
            multitask_labels_dict["total_number_clips"] = SequenceDataset.get_total_number_clips_label(curr_phase, selected_tisssue_name, selected_demo_name)
        if "clip_loaded" in selected_multitasks:
            multitask_labels_dict["clip_loaded"] = SequenceDataset.get_clip_loaded_label(curr_phase, selected_tisssue_name, selected_demo_name, curr_jaw_values, jaw_closed_open_thresh)
        if "gallbladder_grabbed" in selected_multitasks:
            multitask_labels_dict["gallbladder_grabbed"] = SequenceDataset.get_gallbladder_grabbed_label(curr_phase, abs_phase_start_idx, abs_phase_end_idx, curr_ts, curr_jaw_values, 
                                                                                                        jaw_closed_open_thresh, rel_phase_ts_idx_thresh)
        if "clips_left_tube" in selected_multitasks:
            multitask_labels_dict["clips_left_tube"] = SequenceDataset.get_clips_on_tube_label("left", curr_phase, curr_ts, abs_phase_start_idx, abs_phase_end_idx, curr_jaw_values, jaw_closed_open_thresh, rel_phase_ts_idx_thresh)
        if "clips_right_tube" in selected_multitasks:
            multitask_labels_dict["clips_right_tube"] = SequenceDataset.get_clips_on_tube_label("right", curr_phase, curr_ts, abs_phase_start_idx, abs_phase_end_idx, curr_jaw_values, jaw_closed_open_thresh, rel_phase_ts_idx_thresh)
        if "psm2_instrument_closed" in selected_multitasks:
            multitask_labels_dict["psm2_instrument_closed"] = 1 if curr_jaw_values[0] <= jaw_closed_open_thresh else 0
        if "psm1_instrument_closed" in selected_multitasks:
            multitask_labels_dict["psm1_instrument_closed"] = 1 if curr_jaw_values[1] <= jaw_closed_open_thresh else 0
        if "is_correction" in selected_multitasks:
            multitask_labels_dict["is_correction"] =  SequenceDataset.get_is_correction_label(corrections_dict, selected_tisssue_name, curr_phase, selected_demo_name, curr_ts, abs_phase_start_idx, abs_phase_end_idx, rel_corr_idx_thresh)
        if "dominant_moving_direction" in selected_multitasks:
            multitask_labels_dict["dominant_moving_direction"] = SequenceDataset.get_dominant_moving_direction_label(ee_csv_data, ts_demo_frame_idx, curr_jaw_values, curr_phase, curr_ts, abs_phase_start_idx, abs_phase_end_idx,
                                                                                                                     cutting_chunk_size=cutting_chunk_size, jaw_closing_opening_thresh=jaw_closing_opening_thresh, 
                                                                                                                     image_delay_offset=image_delay_offset)
            	
        return multitask_labels_dict
           
    @staticmethod     
    def get_multitask_labels_output_dim_dict(selected_multitasks):
        # NOTE: Update list when adding new tasks
        multitasks_labels_dict = {
            "psm1_instrument": 2,
            "curr_tube": 2,
            "total_number_clips": 7,
            "clip_loaded": 2,
            "gallbladder_grabbed": 2,
            "clips_left_tube": 4,
            "clips_right_tube": 4,
            "psm2_instrument_closed": 2,
            "psm1_instrument_closed": 2,
            "dominant_moving_direction": len(get_all_auto_labels_list()), # initially 17
            "is_correction": 2
        }
        selected_multitasks_dict = {multitask: multitasks_labels_dict[multitask] for multitask in selected_multitasks}
        return selected_multitasks_dict
        
    @staticmethod
    def get_multitask_labels_from_label_indices_or_logits(multitask_labels_indices_dict, batch_wise=False, logits_flag=False, img_idx=None, ignore_do_not_move=False):
        # Returns the multitask labels from the label indices
        
        # NOTE: Update list when adding new tasks
        multitask_label_idx_to_label_dict = {
            "psm1_instrument": {0: "clip applier", 1: "curved scissor"},
            "curr_tube": {0: "left tube", 1: "right tube"},
            "total_number_clips": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
            "clip_loaded": {0: "not loaded", 1: "loaded"},
            "gallbladder_grabbed": {0: "not grabbed", 1: "grabbed"},
            "clips_left_tube": {0: 0, 1: 1, 2: 2, 3: 3},
            "clips_right_tube": {0: 0, 1: 1, 2: 2, 3: 3},
            "psm2_instrument_closed": {0: "open", 1: "closed"},
            "psm1_instrument_closed": {0: "open", 1: "closed"},
            "dominant_moving_direction": dict(zip(range(len(get_all_auto_labels_list())), get_all_auto_labels_list())),
            "is_correction": {0: "no correction", 1: "correction"}
        }
        
        multitask_labels_dict = {}
        if not batch_wise:
            for multitask, label_indices in multitask_labels_indices_dict.items():
                    multitask_labels_dict[multitask] = multitask_label_idx_to_label_dict[multitask][label_indices]
        else:
            if img_idx is not None:
                for multitask, label_indices in multitask_labels_indices_dict.items():
                    if label_indices[img_idx].dim() == 0:
                        label_idx = label_indices[img_idx].item()
                    else:
                        label_idx = torch.argmax(label_indices[img_idx]).item()
                    multitask_labels_dict[multitask] = multitask_label_idx_to_label_dict[multitask][label_idx]
            else:
                for multitask, label_indices in multitask_labels_indices_dict.items():
                    batch_label_list = []
                    for label_idx in label_indices:
                        if logits_flag:
                            real_label_idx = torch.argmax(label_idx).item() # in this case label_idx would be the label logits for that subtask
                            predicted_multitask_label = multitask_label_idx_to_label_dict[multitask][real_label_idx]
                            # Take the second most likely prediction if the dominant moving direction is do not move
                            if multitask == "dominant_moving_direction" and ignore_do_not_move and predicted_multitask_label == "do not move":
                                # Get the second most likely prediction
                                label_idx_ignore_do_not_move = label_idx.clone()
                                label_idx_ignore_do_not_move[real_label_idx] = -float("inf")
                                alternative_label_idx = torch.argmax(label_idx_ignore_do_not_move).item()
                                alternative_multitask_label = multitask_label_idx_to_label_dict[multitask][alternative_label_idx]
                                batch_label_list.append(alternative_multitask_label)
                            else:
                                batch_label_list.append(predicted_multitask_label)
                        else:
                            batch_label_list.append(multitask_label_idx_to_label_dict[multitask][label_idx.item()])
                    multitask_labels_dict[multitask] = batch_label_list

        return multitask_labels_dict
        
    @staticmethod
    def get_all_multitask_names():
        # Returns all multitask names
        # NOTE: Update list when adding new tasks
        selected_multitasks = ["psm1_instrument", "curr_tube", "total_number_clips", "clip_loaded", "gallbladder_grabbed", "clips_left_tube",
                               "clips_right_tube", "psm2_instrument_closed", "psm1_instrument_closed", "dominant_moving_direction", "is_correction"]
        return selected_multitasks
    
    @staticmethod
    def get_all_multitask_labels(multitask):
        # Returns the multitask labels for a specific multitask
        # NOTE: Update list when adding new tasks
        multitask_label_idx_to_label_dict = {
            "psm1_instrument": ["clip applier", "curved scissor"],
            "curr_tube": ["left tube", "right tube"],
            "total_number_clips": ["0", "1", "2", "3", "4", "5", "6"],
            "clip_loaded": ["not loaded", "loaded"],
            "gallbladder_grabbed": ["not grabbed", "grabbed"],
            "clips_left_tube": ["0", "1", "2", "3"],
            "clips_right_tube": ["0", "1", "2", "3"],
            "psm2_instrument_closed": ["open", "closed"],
            "psm1_instrument_closed": ["open", "closed"],
            "dominant_moving_direction": get_all_auto_labels_list(),  # Assuming get_all_auto_labels_list() returns a list
            "is_correction": ["no correction", "correction"]
        }
        multitask_labels = multitask_label_idx_to_label_dict[multitask]
        return multitask_labels
    
    @staticmethod
    def get_multitask_index(multitask, label):
        # Returns the index for each multitask label for the specific multitask
        # NOTE: Update list when adding new tasks
        multitask_label_to_idx_mapping = {
            "psm1_instrument": {"clip applier": 0, "curved scissor": 1},
            "curr_tube": {"left tube": 0, "right tube": 1},
            "total_number_clips": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6},
            "clip_loaded": {"not loaded": 0, "loaded": 1},
            "gallbladder_grabbed": {"not grabbed": 0, "grabbed": 1},
            "clips_left_tube": {"0": 0, "1": 1, "2": 2, "3": 3},
            "clips_right_tube": {"0": 0, "1": 1, "2": 2, "3": 3},
            "psm2_instrument_closed": {"open": 0, "closed": 1},
            "psm1_instrument_closed": {"open": 0, "closed": 1},
            "dominant_moving_direction": dict(zip(get_all_auto_labels_list(), range(len(get_all_auto_labels_list())))),
            "is_correction": {"no correction": 0, "correction": 1}
        }
        multitask_label_idx = multitask_label_to_idx_mapping[multitask][label]
        return multitask_label_idx      
        
    def save_debug_episode_video(self, selected_tissue_sample, selected_phase_demo_dict, start_ts, curr_ts, output_dir, instruction, break_time=2):
        # Initialize the video writer
        phases = "_".join(selected_phase_demo_dict.keys())
        video_name = f"{selected_tissue_sample=}_{phases=}.avi"
        video_path = os.path.join(output_dir, video_name)

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'mp4v' or 'MJPG'
        fps = 30  # Define the frames per second
        video_dim = (self.image_dim[1] * len(self.camera_names), self.image_dim[0])
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, video_dim)
        
        last_phase = list(selected_phase_demo_dict.keys())[-1]
        last_frame = selected_phase_demo_dict[last_phase]["phases_demo_end_idx"]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2 / abs(5 - len(self.camera_names))
        font_color = (255, 255, 255)  # White color in BGR
        font_thickness = 1

        for ts in range(0, last_frame + 1):
            ts_corrected = max(0, ts)
            ts_phase_folder, ts_demo_folder, ts_demo_frame_idx = self.get_current_phase_demo_folder_and_demo_frame_idx(selected_phase_demo_dict, ts_corrected)
            concat_img = []
            for cam_name, cam_file_suffix in zip(self.camera_names, self.camera_file_suffixes):
                demo_folder_path = os.path.join(self.dataset_dir, selected_tissue_sample, ts_phase_folder, ts_demo_folder)
                cam_folder = os.path.join(demo_folder_path, cam_name)
                frame_path = os.path.join(cam_folder, f"frame{str(ts_demo_frame_idx).zfill(6)}{cam_file_suffix}")
                img = cv2.imread(frame_path)
                if img is None:
                    raise ValueError(f"Could not load image from {frame_path}.")
                
                if cam_name == "endo_psm2":
                    split_idx = int(img.shape[1] * self.wrist_images_rel_width)
                    img = img[:, -split_idx:, :]
                elif cam_name == "endo_psm1":
                    split_idx = int(img.shape[1] * self.wrist_images_rel_width)
                    img = img[:, :split_idx, :]
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.image_dim[1], self.image_dim[0]))
                
                # Add the red recording dot only for the left_img_dir camera during the important part of the video
                if cam_name == "left_img_dir" and start_ts <= ts <= curr_ts:
                    dot_radius = 10
                    dot_center = (img.shape[1] - dot_radius - 10, dot_radius + 10)
                    cv2.circle(img, dot_center, dot_radius, (255, 0, 0), -1)  # Red color in BGR
                
                concat_img.append(img)
            
            concat_img = np.concatenate(concat_img, axis=1)
            
            # Calculate the position to center the text
            text_size = cv2.getTextSize(instruction, font, font_scale, font_thickness)[0]
            text_y = self.image_dim[0] - 20
            text_x = (concat_img.shape[1] - text_size[0]) // 2

            # Add the instruction text to the image
            cv2.putText(concat_img, instruction, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # Write the image to the video
            video_writer.write(cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR))  # Convert back to BGR before writing to video
            
            # Add a break at the curr ts for `break_time` seconds (showing the point to predict on)
            if ts == curr_ts:
                for _ in range(int(fps * break_time)):
                    video_writer.write(cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR))

        # Release the video writer
        video_writer.release()
        print(f"\nSaved debug episode video to {video_path}.")

        
    # TODO: Check if this works
    def get_future_frame(self, selected_tissue_sample, selected_phase_demo_dict, future_ts):
        ts_phase_folder, ts_demo_folder, ts_demo_frame_idx = self.get_current_phase_demo_folder_and_demo_frame_idx(selected_phase_demo_dict, future_ts)
        future_frame_path = os.path.join(self.dataset_dir, selected_tissue_sample, ts_phase_folder, ts_demo_folder, "left_img_dir", f"frame{str(ts_demo_frame_idx).zfill(6)}_left.jpg")
        future_frame = torch.tensor(cv2.cvtColor(cv2.imread(future_frame_path), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        return future_frame
            
    def __getitem__(self, index):
        # Put together the stitched episode based on randomly getting a tissue id and a random index for a demo for each phase. Then sample a random timestep and get the corresponding image sequence and command embedding
        
        choose_correction_sample_flag = np.random.rand() <= self.extra_corrections_sampling_probability
        choose_repeated_phase_last_frame_sample_flag = np.random.rand() <= self.extra_repeated_phase_last_frame_sampling_probability
        if self.extra_corrections_sampling_flag and choose_correction_sample_flag:
            # Select a go back correction demo from the predefined go back correction demos
            split_tissue_samples = [tissue_sample.split("_")[1] for tissue_sample in self.tissue_phase_demo_dict]
            dataset_split_tissue_samples = list(set(self.corrections_dict.keys()) & set(split_tissue_samples))
            selected_tissue_sample_idx = np.random.choice(dataset_split_tissue_samples)
            selected_tissue_sample = f"tissue_{selected_tissue_sample_idx}"
            available_phases = list(self.corrections_dict[selected_tissue_sample_idx].keys())
            if "1_grabbing_gallbladder_recovery" in available_phases:
                available_phases += ["1_grabbing_gallbladder_recovery"] # Increase the probability of selecting the grabbing gallbladder recovery phase
            selected_phase = np.random.choice(list(available_phases))
            selected_demo_name = np.random.choice(self.corrections_dict[selected_tissue_sample_idx][selected_phase])
            demo_folder_path = os.path.join(self.dataset_dir, selected_tissue_sample, selected_phase, selected_demo_name)
            
            # Sample the current timestep in the correction range
            demo_len = get_valid_demo_start_end_indices(demo_folder_path, self.before_phase_offset, self.after_phase_offset, self.use_kinematic_indices_flag)[2]
            max_corr_len = int(demo_len * self.rel_corr_idx_threshold)
            # curr_ts = np.random.randint(0, max_corr_len) # TODO: Check if Gaussian works better
            curr_ts = randintgaussian(0, max_corr_len, mean=max_corr_len//4, std_dev=2*max_corr_len) # Sampling more at the beginning of the correction
            scaled_history_step_size = int(self.history_step_size * random.uniform(0.5, 1.0)) # Add variability to the history step size for corrections
            start_ts = curr_ts - self.history_len * scaled_history_step_size
            selected_phase_demo_dict = self.create_phase_demo_metadata_dict(selected_tissue_sample, [selected_phase], [selected_demo_name])
            
            # Retrieve the language embedding for the target_ts
            command_gt, command_embedding = self.command_embeddings_dict[selected_phase]
            command_embedding = torch.tensor(command_embedding).squeeze()
            
            if self.use_jaw_values_flag or self.selected_multitasks:
                # Read out the jaw values from start to end (when a certain flag is set) - either as further input or for generating multitask labels
                jaw_psm2_psm1_data_sequence, ee_csv_array = self.get_jaw_psm2_psm1_data_sequence(selected_tissue_sample, selected_phase_demo_dict, start_ts, curr_ts, scaled_history_step_size) 
            
            # History information of the last six phases (with padding if needed)
            sorted_phases = []
            for phase in list(self.command_embeddings_dict):
                phase_wo_recovery = phase.replace("_recovery", "")
                if phase_wo_recovery not in sorted_phases:
                    sorted_phases.append(phase_wo_recovery)
            if self.use_phase_history_flag:
                phase_history = self.get_phase_history(selected_phase_demo_dict, sorted_phases, curr_ts, correction_flag=True)
            else:
                self.phase_history_len = 1 # Required for eval of phase transitions
                phase_history = self.get_phase_history(selected_phase_demo_dict, sorted_phases, curr_ts, correction_flag=True)
            
            # Generate multitask labels for the selected multitasks
            if self.selected_multitasks:
                selected_demo_name = os.path.basename(demo_folder_path)
                ts_demo_frame_idx = self.get_current_phase_demo_folder_and_demo_frame_idx(selected_phase_demo_dict, curr_ts)[2]
                abs_phase_start_idx, abs_phase_end_idx = 0, get_valid_demo_start_end_indices(demo_folder_path, self.before_phase_offset, self.after_phase_offset, self.use_kinematic_indices_flag)[2]
                multitask_label_indices_dict = self.get_multitask_labels_dict(self.selected_multitasks, curr_ts, selected_phase, abs_phase_start_idx, abs_phase_end_idx, jaw_psm2_psm1_data_sequence[-1],
                                                                              selected_tissue_sample, selected_demo_name, ee_csv_data=ee_csv_array, ts_demo_frame_idx=ts_demo_frame_idx,
                                                                              corrections_dict=self.corrections_dict, rel_corr_idx_thresh=self.rel_corr_idx_threshold)
            else:
                multitask_label_indices_dict = {}
            
            # Save a video from start ts to curr_ts
            if self.record_debug_episode_video_flag:
                output_dir = os.path.join(os.getenv('PATH_TO_YAY_ROBOT'), "examples_plots", "dataset")
                self.save_debug_episode_video(selected_tissue_sample, selected_phase_demo_dict, curr_ts, curr_ts, output_dir, command_gt)
            
            # Construct the image sequences for the desired correction timesteps
            phase_end_timesteps = range(start_ts, curr_ts + 1, scaled_history_step_size)
            image_sequence = self.get_image_sequence(selected_tissue_sample, selected_phase_demo_dict, phase_end_timesteps)
                   
        elif self.extra_repeated_phase_last_frame_sampling_flag and choose_repeated_phase_last_frame_sample_flag and not self.incomplete_demos_flag:
            # Select a demo from available tissue and corresponding phase
            selected_tissue_sample = np.random.choice(list(self.tissue_phase_demo_dict.keys()))
            
            # Go through the phases in fixed order of execution (ignore recovery phases and last phase (as no transition))
            phases = list(self.tissue_phase_demo_dict[selected_tissue_sample].keys())
            last_phase_idx_of_tissue = int(phases[-1].split("_")[0])
            valid_transition_phases = [phase for phase in phases if int(phase.split("_")[0]) != last_phase_idx_of_tissue and "recovery" not in phase and "cutting" not in phase]    
            if "1_grabbing_gallbladder" in valid_transition_phases:
                valid_transition_phases += ["1_grabbing_gallbladder"] # Increase the probability of selecting the grabbing gallbladder phase
            selected_phase = np.random.choice(valid_transition_phases)        
            selected_demo_name = np.random.choice(self.tissue_phase_demo_dict[selected_tissue_sample][selected_phase])
            demo_folder_path = os.path.join(self.dataset_dir, selected_tissue_sample, selected_phase, selected_demo_name)
            
            # Take last frame of the phase
            demo_len = get_valid_demo_start_end_indices(demo_folder_path, self.before_phase_offset, self.after_phase_offset, self.use_kinematic_indices_flag)[2]
            abs_phase_start_idx, abs_phase_end_idx = 0, demo_len - 1
            curr_ts = start_ts = abs_phase_end_idx  
            selected_phase_demo_dict = self.create_phase_demo_metadata_dict(selected_tissue_sample, [selected_phase], [selected_demo_name])
            
            # Retrieve the language embedding for the next phase
            sorted_phases = [phase for phase in list(self.command_embeddings_dict) if "recovery" not in phase]
            curr_phase_idx = sorted_phases.index(selected_phase)
            next_phase_folder = sorted_phases[curr_phase_idx+1]
            command_gt, command_embedding = self.command_embeddings_dict[next_phase_folder]
            command_embedding = torch.tensor(command_embedding).squeeze()
            
            # Generate the repeated jaw values for the last timestep of the phase
            if self.use_jaw_values_flag or self.selected_multitasks:
                # Read out the jaw values from start to end (when a certain flag is set) - either as further input or for generating multitask labels
                curr_jaw_psm2_psm1_values, ee_csv_data = self.get_jaw_psm2_psm1_data_sequence(selected_tissue_sample, selected_phase_demo_dict, curr_ts, curr_ts)
                jaw_psm2_psm1_data_sequence = curr_jaw_psm2_psm1_values.repeat(self.history_len+1, 1) # Repeat the first dim
            
            # History information of the last six phases (with padding if needed)
            if self.use_phase_history_flag:
                phase_history = self.get_phase_history(selected_phase_demo_dict, sorted_phases, curr_ts, correction_flag=True)
            else:
                self.phase_history_len = 1 # Required for eval of phase transitions
                phase_history = self.get_phase_history(selected_phase_demo_dict, sorted_phases, curr_ts, correction_flag=True)
            
            # Generate multitask labels for the selected multitasks
            if self.selected_multitasks:
                if not "back" in selected_phase: # Don't use the kinematics from the beginning of the next phase when clip needs to be loaded or tool exchanged
                    # Get the dominant moving direction from the next phase
                    next_phase_folder_path = os.path.join(self.dataset_dir, selected_tissue_sample, next_phase_folder)
                    next_demo = np.random.choice(os.listdir(next_phase_folder_path))
                    next_phase_demo_folder_path = os.path.join(next_phase_folder_path, next_demo)
                    kinematics_file_path = os.path.join(next_phase_demo_folder_path, "ee_csv.csv")
                    next_phase_ee_csv_array = pd.read_csv(kinematics_file_path)
                    next_phase_valid_start_idx = get_valid_demo_start_end_indices(next_phase_demo_folder_path, self.before_phase_offset, self.after_phase_offset, self.use_kinematic_indices_flag)[0]
                    ts_demo_frame_idx = next_phase_valid_start_idx
                    ee_csv_data = next_phase_ee_csv_array
                else:
                    ts_demo_frame_idx = get_valid_demo_start_end_indices(demo_folder_path, self.before_phase_offset, self.after_phase_offset, self.use_kinematic_indices_flag)[0]
                multitask_label_indices_dict = self.get_multitask_labels_dict(self.selected_multitasks, curr_ts, selected_phase, abs_phase_start_idx, abs_phase_end_idx, jaw_psm2_psm1_data_sequence[-1],
                                                                              selected_tissue_sample, selected_demo_name, ee_csv_data=ee_csv_data, ts_demo_frame_idx=ts_demo_frame_idx,
                                                                              corrections_dict=self.corrections_dict, rel_corr_idx_thresh=self.rel_corr_idx_threshold)
            else:
                multitask_label_indices_dict = {}
            
            # Save a video from start ts to curr_ts
            if self.record_debug_episode_video_flag:
                output_dir = os.path.join(os.getenv('PATH_TO_YAY_ROBOT'), "examples_plots", "dataset")
                self.save_debug_episode_video(selected_tissue_sample, selected_phase_demo_dict, curr_ts, curr_ts, output_dir, command_gt)
            
            # Construct the repeated image sequences for the last timestep of the phase 
            phase_start_timesteps = [curr_ts]*(self.history_len+1)
            image_sequence = self.get_image_sequence(selected_tissue_sample, selected_phase_demo_dict, phase_start_timesteps) 
                
        else:
            # Select a random tissue sample to generate the episode from
            selected_tissue_sample = np.random.choice(list(self.tissue_phase_demo_dict.keys()))
            
            # Go through the phases in fixed order of execution (ignore recovery phases here - included in next paragraph)
            phases = list(self.tissue_phase_demo_dict[selected_tissue_sample].keys())
            phases_wo_recovery = [phase for phase in phases if "recovery" not in phase]
            sorted_phases = sorted(phases_wo_recovery, key=lambda x: int(x.split('_')[0]))
            
            # Choose first phase as either the recovery or normal phase and the second phase as the next phase
            phases_to_consider = sorted_phases.copy()
            if "1_grabbing_gallbladder" in phases_to_consider:
                phases_to_consider += ["1_grabbing_gallbladder"] # Increase the probability of selecting the grabbing gallbladder phase
            phase_1_wo_suffix = np.random.choice(phases_to_consider) # Exclude the last phase
            recovery_phase_flag = np.random.rand() <= self.recovery_probability
            recovery_available_flag = phase_1_wo_suffix + "_recovery" in phases
            phase_1_suffix = "_recovery" if recovery_phase_flag and recovery_available_flag else ""
            phase_1 = phase_1_wo_suffix + phase_1_suffix
            
            # If possible add a successor phase for training on transitions
            old_grab_pull_criterium = "grabbing" in phase_1 and selected_tissue_sample in self.tissue_samples_old_grab_pull_separation # Don't sample transitions for grabbing gallbladder for these tissue samples
            phase_1_idx = sorted_phases.index(phase_1_wo_suffix)
            last_phase_flag = phase_1_idx == len(sorted_phases) - 1
            no_direct_successor_flag = False
            if not self.incomplete_demos_flag and not old_grab_pull_criterium and not last_phase_flag:
                phase_2 = sorted_phases[phase_1_idx + 1]
                phase_1_id, phase_2_id = int(phase_1.split('_')[0]), int(phase_2.split('_')[0])
                if phase_1_id + 1 != phase_2_id:
                    no_direct_successor_flag = True
                    del phase_2
            add_successor_flag = not(no_direct_successor_flag or old_grab_pull_criterium or self.incomplete_demos_flag or last_phase_flag) # If possible add successor for training on transitions
            
            # Sample the demos for the first and second phase (go back to normal phase if no demos available for recovery phase)
            if not self.tissue_phase_demo_dict[selected_tissue_sample][phase_1] and "recovery" in phase_1:
                phase_1 = phase_1.replace("_recovery", "")
                demo_phase_1 = np.random.choice(self.tissue_phase_demo_dict[selected_tissue_sample][phase_1])
            else:
                demo_phase_1 = np.random.choice(self.tissue_phase_demo_dict[selected_tissue_sample][phase_1])
            
            # Store the selected phase demo and the start and end timestep
            if add_successor_flag: 
                demo_phase_2 = np.random.choice(self.tissue_phase_demo_dict[selected_tissue_sample][phase_2])
                phase_1_demo_folder_path = os.path.join(self.dataset_dir, selected_tissue_sample, phase_1, demo_phase_1)
                len_phase_1 = get_valid_demo_start_end_indices(phase_1_demo_folder_path, self.before_phase_offset, self.after_phase_offset, self.use_kinematic_indices_flag)[2]
                total_len_wo_p2 = len_phase_1 # Just use phase 1 as possible input (or potentially precessor phases)
                precessor_phases, precessor_phase_demos = deque(), deque() 
                if not "grabbing" in phase_1 and not "recovery" in phase_1: # If not recovery or grabbing phase, then add precessor phases (else repeat first frame)
                    required_num_samples = self.history_len * self.history_step_size + self.prediction_offset
                    if total_len_wo_p2 < required_num_samples: # Check that the stitched partial episode is long enough (for all input data)
                        if self.verbose:
                            print(f"Total length {len_phase_1} of phase 1 ({phase_1}) is too short --> Adding precessor phases in front of phase 1.")
                        demo_phase_1 = np.random.choice(self.tissue_phase_demo_dict[selected_tissue_sample][phase_1])
                        phase_1_demo_folder_path = os.path.join(self.dataset_dir, selected_tissue_sample, phase_1, demo_phase_1)
                        total_len_wo_p2 = len_phase_1 = get_valid_demo_start_end_indices(phase_1_demo_folder_path, self.before_phase_offset, self.after_phase_offset, self.use_kinematic_indices_flag)[2]
                    
                        for phase_idx in range(phase_1_idx - 1, -1, -1): 
                            if total_len_wo_p2 >= required_num_samples:
                                break
                            precessor_phase = sorted_phases[phase_idx]
                            precessor_phase_demo = np.random.choice(self.tissue_phase_demo_dict[selected_tissue_sample][precessor_phase])
                            precessor_phase_folder_path = os.path.join(self.dataset_dir, selected_tissue_sample, precessor_phase, precessor_phase_demo)
                            precessor_phase_demo_len = get_valid_demo_start_end_indices(precessor_phase_folder_path, self.before_phase_offset, self.after_phase_offset, self.use_kinematic_indices_flag)[2]
                            total_len_wo_p2 += precessor_phase_demo_len
                            precessor_phases.appendleft(precessor_phase)
                            precessor_phase_demos.appendleft(precessor_phase_demo)
                precessor_phases, precessor_phase_demos = list(precessor_phases), list(precessor_phase_demos)
                
                # Create a dictionary to store the selected phase demos and the start and end timestep, and further metadata
                selected_phases = precessor_phases + [phase_1, phase_2]
                selected_phase_demos = precessor_phase_demos + [demo_phase_1, demo_phase_2]
            else:
                selected_phases, selected_phase_demos = [phase_1], [demo_phase_1]
            selected_phase_demo_dict = self.create_phase_demo_metadata_dict(selected_tissue_sample, selected_phases, selected_phase_demos)
            
            if self.verbose:
                print(f"Selected tissue sample: {selected_tissue_sample}")
                if add_successor_flag:
                    print(f"Selected phases: {phase_1}, {phase_2}")
                    print(f"Selected demos: {demo_phase_1}, {demo_phase_2}")
                else:
                    print(f"Selected phase: {phase_1}")
                    print(f"Selected demo: {demo_phase_1}")
            
            # Sample a random curr_ts and compute the start_ts and target_ts from concatinated phase 1+2 (via Gaussian distribution)
            if "grabbing" in phase_1 or "recovery" in phase_1 or not add_successor_flag:
                min_phase_1_phase_2_idx = 0 # To sample more of grabbing gallbladder, allow repeated frames at the beginning
            elif selected_phase_demo_dict[phase_1]["phases_demo_end_idx"] >= self.history_len * self.history_step_size:
                min_phase_1_phase_2_idx = self.history_len * self.history_step_size # If enough frames from previous demos 
            else: 
                min_phase_1_phase_2_idx = selected_phase_demo_dict[phase_1]["phases_demo_end_idx"] # If frames need to be padded (and therefore the num actual frames would be greater than phase 1)
            if add_successor_flag:
                max_phase_1_phase_2_idx = selected_phase_demo_dict[phase_2]["phases_demo_end_idx"] - self.prediction_offset
            elif self.incomplete_demos_flag or old_grab_pull_criterium:
                max_phase_1_phase_2_idx = selected_phase_demo_dict[phase_1]["phases_demo_end_idx"]
            else:
                max_phase_1_phase_2_idx = selected_phase_demo_dict[phase_1]["phases_demo_end_idx"] - self.prediction_offset # If no successor phase is available
            if self.uniform_sampling_flag:
                curr_ts = np.random.randint(min_phase_1_phase_2_idx, max_phase_1_phase_2_idx)
            else:
                phase_1_phase_2_sampling_mean = selected_phase_demo_dict[phase_1]["phases_demo_end_idx"]
                phase_1_phase_2_sampling_std = (max_phase_1_phase_2_idx - min_phase_1_phase_2_idx) / (2*3)
                curr_ts = randintgaussian(min_phase_1_phase_2_idx, max_phase_1_phase_2_idx, mean=phase_1_phase_2_sampling_mean, std_dev=phase_1_phase_2_sampling_std)
            start_ts = curr_ts - self.history_len * self.history_step_size
            
            # -----
            
            # Read out the jaw values from start to end (when a certain flag is set) - either as further input or for generating multitask labels
            jaw_psm2_psm1_data_sequence, ee_csv_array = self.get_jaw_psm2_psm1_data_sequence(selected_tissue_sample, selected_phase_demo_dict, start_ts, curr_ts)
            
            # Retrieve the language embedding for the target_ts - for clipping and cutting only predict the next instruction if already clipped/cut
            psm1_not_closed_criterium = bool(("cutting" in phase_1 or "clipping" in phase_1) and jaw_psm2_psm1_data_sequence[-1][1] >= INSTRUMENT_CLOSED_THRESHOLD)
            psm2_not_closed_criterium = bool("grabbing" in phase_1 and jaw_psm2_psm1_data_sequence[-1][0] >= INSTRUMENT_CLOSED_THRESHOLD)
            if psm1_not_closed_criterium or psm2_not_closed_criterium or self.incomplete_demos_flag or old_grab_pull_criterium: 
                target_ts = curr_ts
            else:
                target_ts = curr_ts + self.prediction_offset
            command_embedding, command_gt, curr_phase = self.get_embedding_command_phase_for_ts(selected_phase_demo_dict, target_ts)        
            if command_embedding is None:
                raise ValueError(f"Could not find embedding for target_ts {target_ts}.")
            
            # Generate multitask labels for the selected multitasks
            if self.selected_multitasks:
                _, selected_demo_name, ts_demo_frame_idx = self.get_current_phase_demo_folder_and_demo_frame_idx(selected_phase_demo_dict, curr_ts)
                abs_phase_start_idx = selected_phase_demo_dict[curr_phase]["phases_demo_start_idx"]
                abs_phase_end_idx = selected_phase_demo_dict[curr_phase]["phases_demo_end_idx"]
                multitask_label_indices_dict = self.get_multitask_labels_dict(self.selected_multitasks, curr_ts, curr_phase, abs_phase_start_idx, abs_phase_end_idx, jaw_psm2_psm1_data_sequence[-1],
                                                                              selected_tissue_sample, selected_demo_name, ee_csv_data=ee_csv_array, ts_demo_frame_idx=ts_demo_frame_idx,
                                                                              corrections_dict=self.corrections_dict, rel_corr_idx_thresh=self.rel_corr_idx_threshold)
            else:
                multitask_label_indices_dict = {}
            
            # History information of the last six phases (with padding if needed)
            if "is_correction" in self.selected_multitasks:
                is_correction = multitask_label_indices_dict["is_correction"] == "correction"
            else:
                is_correction = False
            if self.use_phase_history_flag:
                phase_history = self.get_phase_history(selected_phase_demo_dict, sorted_phases, curr_ts, correction_flag=is_correction)
            else:
                self.phase_history_len = 1 # Required for eval of phase transitions
                phase_history = self.get_phase_history(selected_phase_demo_dict, sorted_phases, curr_ts, correction_flag=is_correction)
            
            # TODO: Check if this works - # TODO: Apply the same augmentations also here
            # Future frame
            if self.future_frame_prediction_flag:
                future_ts = min(curr_ts + self.future_frame_delta_t, selected_phase_demo_dict[phase_2]["phases_demo_end_idx"])
                future_frame = self.get_future_frame(selected_tissue_sample, selected_phase_demo_dict, future_ts)        
            
            # Save a video from start ts to curr_ts
            if self.record_debug_episode_video_flag:
                output_dir = os.path.join(os.getenv('PATH_TO_YAY_ROBOT'), "examples_plots", "dataset")
                self.save_debug_episode_video(selected_tissue_sample, selected_phase_demo_dict, start_ts, curr_ts, output_dir, command_gt)
            
            # Get the image sequence for the history_len
            image_timesteps = range(start_ts, curr_ts + 1, self.history_step_size)
            image_sequence = self.get_image_sequence(selected_tissue_sample, selected_phase_demo_dict, image_timesteps)     

        if self.use_jaw_values_flag:
            return image_sequence, command_embedding, command_gt, jaw_psm2_psm1_data_sequence, phase_history, multitask_label_indices_dict
        else:
            return image_sequence, command_embedding, command_gt, phase_history, multitask_label_indices_dict
    

def load_merged_data(
    dataset_dirs,
    num_episodes_list,
    camera_names,
    camera_file_suffixes,
    batch_size_train=32,
    batch_size_val=32,
    history_len=1,
    prediction_offset=10,
    history_step_size=1,
    test_only=False,
    input_transforms=None,
    reduced_base_instruction_set_flag=False,
    use_phase_history_flag=False,
    use_jaw_values_flag=False,
    phase_history_len=6,
    prediction_step_size=30,
    recovery_probability = 0.25,
    phase_history_only_phase_switches_flag = False,
    image_dim=(224, 224),
    llava_anyres_flag=False,
    no_llava_anyres_global_image_flag=False,
    wrist_images_rel_width=3/4,
    llava_anyres_rel_width=1/2,
    uniform_sampling_flag=False,
    selected_multitasks=[],
    use_seg_masks_input_flag=False,
    merge_seg_masks_flag=False,
    seg_mask_objs=["clip", "gallbladder", "tissue"], # NOTE: When merging order is expressing the priority, so what will be chosen when two masks overlap.
    use_kinematic_indices_flag=True,
    extra_corrections_sampling_flag=False,
    extra_corrections_sampling_probability=0.1,    
    extra_repeated_phase_last_frame_sampling_flag=False,
    extra_repeated_phase_last_frame_sampling_probability=0.1,
    add_center_crop_view_flag=False,
    distance_from_border_y=0.1,
    distance_from_border_x=0.25,
    y_offset=-0.1,
    train_on_all_data_flag=False,
    val_split_number=0
):
    
    print(f"{history_len=}, {history_step_size=}, {prediction_offset=}")
    
    ds_metadata_dict = {}

    # Save the metadata
    ds_metadata_dict["train_tissues"] = {}
    ds_metadata_dict["val_tissues"] = {}
    ds_metadata_dict["test_tissues"] = {}
    ds_metadata_dict["train_ds_statistics"] = {}
    ds_metadata_dict["val_ds_statistics"] = {}
    ds_metadata_dict["test_ds_statistics"] = {}
    ds_metadata_dict["history_len"] = history_len
    ds_metadata_dict["history_step_size"] = history_step_size
    ds_metadata_dict["prediction_offset"] = prediction_offset
    ds_metadata_dict["camera_names"] = camera_names
    ds_metadata_dict["test_only"] = test_only
    ds_metadata_dict["input_transforms"] = input_transforms
    ds_metadata_dict["dataset_dirs"] = dataset_dirs
    ds_metadata_dict["num_episodes_list"] = num_episodes_list 
    ds_metadata_dict["reduced_base_instruction_set_flag"] = reduced_base_instruction_set_flag
    ds_metadata_dict["use_phase_history_flag"] = use_phase_history_flag
    ds_metadata_dict["use_jaw_values_flag"] = use_jaw_values_flag
    ds_metadata_dict["phase_history_len"] = phase_history_len
    ds_metadata_dict["prediction_step_size"] = prediction_step_size
    ds_metadata_dict["recovery_probability"] = recovery_probability
    ds_metadata_dict["phase_history_only_phase_switches_flag"] = phase_history_only_phase_switches_flag
    ds_metadata_dict["image_dim"] = image_dim
    ds_metadata_dict["llava_anyres_flag"] = llava_anyres_flag
    ds_metadata_dict["no_llava_anyres_global_image_flag"] = no_llava_anyres_global_image_flag
    ds_metadata_dict["wrist_images_rel_width"] = wrist_images_rel_width
    ds_metadata_dict["llava_anyres_rel_width"] = llava_anyres_rel_width
    ds_metadata_dict["uniform_sampling_flag"] = uniform_sampling_flag
    ds_metadata_dict["selected_multitasks"] = selected_multitasks
    ds_metadata_dict["use_seg_masks_input_flag"] = use_seg_masks_input_flag
    ds_metadata_dict["merge_seg_masks_flag"] = merge_seg_masks_flag
    ds_metadata_dict["seg_mask_objs"] = seg_mask_objs
    ds_metadata_dict["use_kinematic_indices_flag"] = use_kinematic_indices_flag
    ds_metadata_dict["extra_corrections_sampling_flag"] = extra_corrections_sampling_flag
    ds_metadata_dict["extra_corrections_sampling_probability"] = extra_corrections_sampling_probability
    ds_metadata_dict["extra_repeated_phase_last_frame_sampling_flag"] = extra_repeated_phase_last_frame_sampling_flag
    ds_metadata_dict["extra_repeated_phase_last_frame_sampling_probability"] = extra_repeated_phase_last_frame_sampling_probability
    ds_metadata_dict["add_center_crop_view_flag"] = add_center_crop_view_flag
    ds_metadata_dict["distance_from_border_y"] = distance_from_border_y
    ds_metadata_dict["distance_from_border_x"] = distance_from_border_x
    ds_metadata_dict["y_offset"] = y_offset
    ds_metadata_dict["train_on_all_data_flag"] = train_on_all_data_flag

    # Construct the datasets and the dataset embeddings
    train_datasets, val_datasets, test_datasets = [], [], []
    command_embeddings_dict = {}
    val_command_embeddings_dict_add_datasets = {}
    class_occ_cnt_dict = defaultdict(lambda: 0)
    all_val_tissues = []
    for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list):
        # Load dataset dir and count number of tissue samples
        dataset_file_names = os.listdir(dataset_dir)
        dataset_name = os.path.basename(dataset_dir)
        dataset_config = DATASET_CONFIGS[dataset_name]
        tissue_samples_to_exclude = dataset_config["tissue_samples_to_exclude"] if "tissue_samples_to_exclude" in dataset_config else []
        if "endo_psm2" in camera_names or "endo_psm1" in camera_names: # If endo_psm2 or endo_psm1 is used, exclude tissue samples where wrist cameras are not set correctly (e.g., tissue_4)
            tissue_samples_to_exclude += dataset_config["tissue_samples_wrist_cameras_to_exclude"]      
        tissue_names = [tissue_name for tissue_name in dataset_file_names if tissue_name.startswith(("tissue", "phantom")) and tissue_name not in tissue_samples_to_exclude]
        
        # Split the tissue samples into train, val, test by randomly sampling until the ratios are fulfilled
        if not train_on_all_data_flag:
            val_tissues = dataset_config["val_tissues"][val_split_number] if "val_tissues" in dataset_config else []
            test_tissues = dataset_config["test_tissues"] if "test_tissues" in dataset_config else []
            train_tissues = [tissue_name for tissue_name in tissue_names if tissue_name not in val_tissues and tissue_name not in test_tissues]
        else:
            train_tissues = tissue_names
            val_tissues = test_tissues = []
        all_val_tissues += val_tissues            
            
        print(f"\nDataset: {dataset_dir}")
        print(f"Train tissues: {train_tissues}")
        print(f"Val tissues: {val_tissues}")
        print(f"Test tissues: {test_tissues}")
        
        ds_metadata_dict["train_tissues"][dataset_dir] = train_tissues
        ds_metadata_dict["val_tissues"][dataset_dir] = val_tissues
        ds_metadata_dict["test_tissues"][dataset_dir] = test_tissues
        
        # ---------------------- Construct datasets -----------------------
        
        if not test_only:
            # Construct dataset and dataloader for each dataset dir and merge them
            train_datasets.append(SequenceDataset(
                        "train",
                        [tissue_name for tissue_name in train_tissues],
                        dataset_dir,
                        camera_names,
                        camera_file_suffixes,
                        history_len,
                        prediction_offset,
                        history_step_size,
                        num_episodes,
                        input_transforms,
                        reduced_base_instruction_set_flag,
                        use_phase_history_flag,
                        use_jaw_values_flag,
                        phase_history_len,
                        prediction_step_size,
                        recovery_probability,
                        phase_history_only_phase_switches_flag,
                        image_dim=image_dim,
                        llava_anyres_flag=llava_anyres_flag,
                        no_llava_anyres_global_image_flag=no_llava_anyres_global_image_flag,
                        wrist_images_rel_width=wrist_images_rel_width,
                        llava_anyres_rel_width=llava_anyres_rel_width,
                        uniform_sampling_flag=uniform_sampling_flag,
                        selected_multitasks=selected_multitasks,
                        use_seg_masks_input_flag=use_seg_masks_input_flag,
                        merge_seg_masks_flag=merge_seg_masks_flag,
                        seg_mask_objs=seg_mask_objs,
                        use_kinematic_indices_flag=use_kinematic_indices_flag,
                        extra_corrections_sampling_flag=extra_corrections_sampling_flag,
                        extra_corrections_sampling_probability=extra_corrections_sampling_probability,
                        extra_repeated_phase_last_frame_sampling_flag=extra_repeated_phase_last_frame_sampling_flag,
                        extra_repeated_phase_last_frame_sampling_probability=extra_repeated_phase_last_frame_sampling_probability,
                        add_center_crop_view_flag=add_center_crop_view_flag,
                        distance_from_border_y=distance_from_border_y,
                        distance_from_border_x=distance_from_border_x,
                        y_offset=y_offset))
            if val_tissues:
                num_val_episodes = num_episodes // 2
                val_datasets.append(SequenceDataset(
                        "val",
                        [tissue_name for tissue_name in val_tissues],
                        dataset_dir,
                        camera_names,
                        camera_file_suffixes,
                        history_len,
                        prediction_offset,
                        history_step_size,
                        num_val_episodes,
                        input_transforms,
                        reduced_base_instruction_set_flag,
                        use_phase_history_flag,
                        use_jaw_values_flag,
                        phase_history_len,
                        prediction_step_size,
                        recovery_probability,
                        phase_history_only_phase_switches_flag,
                        image_dim=image_dim,
                        llava_anyres_flag=llava_anyres_flag,
                        no_llava_anyres_global_image_flag=no_llava_anyres_global_image_flag,
                        wrist_images_rel_width=wrist_images_rel_width,
                        llava_anyres_rel_width=llava_anyres_rel_width,
                        uniform_sampling_flag=uniform_sampling_flag,
                        selected_multitasks=selected_multitasks,
                        use_seg_masks_input_flag=use_seg_masks_input_flag,
                        merge_seg_masks_flag=merge_seg_masks_flag,
                        seg_mask_objs=seg_mask_objs,
                        use_kinematic_indices_flag=use_kinematic_indices_flag,
                        extra_corrections_sampling_flag=extra_corrections_sampling_flag,
                        extra_corrections_sampling_probability=extra_corrections_sampling_probability,
                        extra_repeated_phase_last_frame_sampling_flag=extra_repeated_phase_last_frame_sampling_flag,
                        extra_repeated_phase_last_frame_sampling_probability=extra_repeated_phase_last_frame_sampling_probability,
                        add_center_crop_view_flag=add_center_crop_view_flag,
                        distance_from_border_y=distance_from_border_y,
                        distance_from_border_x=distance_from_border_x,
                        y_offset=y_offset))
                val_ds_statistics_dict = val_datasets[-1].ds_statistics_dict
                ds_metadata_dict["val_ds_statistics"][dataset_dir] = val_ds_statistics_dict
                val_command_embeddings_dict = val_datasets[-1].command_embeddings_dict
                val_command_embeddings_dict_add_datasets.update(val_command_embeddings_dict)
            
            # Get dataset statistics
            train_ds_statistics_dict = train_datasets[-1].ds_statistics_dict
            ds_metadata_dict["train_ds_statistics"][dataset_dir] = train_ds_statistics_dict
            
            
            # Get the command embeddings for the train and val datasets
            train_command_embeddings_dict = train_datasets[-1].command_embeddings_dict
            
            # Update the command embeddings dictionary
            command_embeddings_dict.update(train_command_embeddings_dict)
            
            # Add the class occurence ratio to the class_occ_ratio_dict
            for command, _ in train_datasets[-1].command_embeddings_dict.values():
                class_occ_cnt_dict[command] += 1
                class_occ_cnt_dict["in_total"] += 1
                
        else: 
            test_datasets.append(SequenceDataset(
                        "test",
                        [tissue_name for tissue_name in test_tissues],
                        dataset_dir,
                        camera_names,
                        camera_file_suffixes,
                        history_len,
                        prediction_offset,
                        history_step_size,
                        num_episodes,
                        input_transforms,
                        reduced_base_instruction_set_flag,
                        use_phase_history_flag,
                        use_jaw_values_flag,
                        phase_history_len,
                        prediction_step_size,
                        recovery_probability,
                        phase_history_only_phase_switches_flag,
                        image_dim=image_dim,
                        llava_anyres_flag=llava_anyres_flag,
                        no_llava_anyres_global_image_flag=no_llava_anyres_global_image_flag,
                        wrist_images_rel_width=wrist_images_rel_width,
                        llava_anyres_rel_width=llava_anyres_rel_width,
                        uniform_sampling_flag=uniform_sampling_flag,
                        selected_multitasks=selected_multitasks,
                        use_seg_masks_input_flag=use_seg_masks_input_flag,
                        merge_seg_masks_flag=merge_seg_masks_flag,
                        seg_mask_objs=seg_mask_objs,
                        use_kinematic_indices_flag=use_kinematic_indices_flag,
                        extra_corrections_sampling_flag=extra_corrections_sampling_flag,
                        extra_corrections_sampling_probability=extra_corrections_sampling_probability,
                        extra_repeated_phase_last_frame_sampling_flag=extra_repeated_phase_last_frame_sampling_flag,
                        extra_repeated_phase_last_frame_sampling_probability=extra_repeated_phase_last_frame_sampling_probability,
                        add_center_crop_view_flag=add_center_crop_view_flag,
                        distance_from_border_y=distance_from_border_y,
                        distance_from_border_x=distance_from_border_x,
                        y_offset=y_offset))
            
            # Get dataset statistics
            test_ds_statistics_dict = test_datasets[-1].ds_statistics_dict
            ds_metadata_dict["test_ds_statistics"][dataset_dir] = test_ds_statistics_dict
            
            # Get the command embeddings for the test datasets (should be the same as for train and val datasets)
            test_command_embeddings_dict = test_datasets[-1].command_embeddings_dict
            command_embeddings_dict.update(test_command_embeddings_dict)

    # ----------------------------- Construct the dataloaders -------------------------------
    
    if not test_only:
        # Check for if all val commands are in the training commands
        train_commands = set([command for command, _ in command_embeddings_dict.values()])
        if all_val_tissues:
            val_commands = set([command for command, _ in val_command_embeddings_dict_add_datasets.values()])
            if not val_commands.issubset(train_commands):
                raise ValueError("Val commands are not subset of train commands.")
        
        
        # Merge all datasets (e.g., base dataset + fine tuning (correction) datasets) into one big dataset
        merged_train_dataset = ConcatDataset(train_datasets)
        if all_val_tissues:
            merged_val_dataset = ConcatDataset(val_datasets)
        
        train_dataloader = DataLoader(
            merged_train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=16,
            persistent_workers=True,
            collate_fn=multitask_collate_fn
        )
        if all_val_tissues:
            val_dataloader = DataLoader(
                merged_val_dataset,
                batch_size=batch_size_val,
                shuffle=False,
                pin_memory=True,
                num_workers=8,
                prefetch_factor=16,
                persistent_workers=True,
                collate_fn=multitask_collate_fn
            )
        else:
            val_dataloader = None
        
        # Extract the candidate embeddings and commands
        candidate_embeddings, candidate_texts = extract_candidate_embeddings_and_commands(command_embeddings_dict)
        phase_to_instruction_mapping = extract_phase_idx_to_instruction_mapping(command_embeddings_dict)
        ds_metadata_dict["phase_to_instruction_mapping"] = phase_to_instruction_mapping
        ds_metadata_dict["candidate_texts"] = candidate_texts
        ds_metadata_dict["candidate_embeddings"] = candidate_embeddings
    
        # Add class weights to the metadata (balanced class weights)
        total_samples = class_occ_cnt_dict["in_total"]
        del class_occ_cnt_dict["in_total"]
        num_classes = len(class_occ_cnt_dict)
        class_weights = {cls: total_samples / (num_classes * cnt) for cls, cnt in class_occ_cnt_dict.items()}
        class_weights_tensor = torch.tensor([class_weights[cls] for cls in candidate_texts], dtype=torch.float) # sort the class weights according to the candidate_texts (order for the model labels)
        ds_metadata_dict["class_weights"] = class_weights_tensor
        
        return train_dataloader, val_dataloader, ds_metadata_dict
    
    else:
        # Merge all datasets (e.g., base dataset + fine tuning (correction) datasets) into one big dataset
        merged_test_dataset = ConcatDataset(test_datasets) 

        test_dataloader = DataLoader(
            merged_test_dataset,
            batch_size=batch_size_val,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
            prefetch_factor=1,
            collate_fn=multitask_collate_fn
        )

        # Extract the candidate embeddings and commands
        candidate_embeddings, candidate_texts = extract_candidate_embeddings_and_commands(command_embeddings_dict)
        phase_to_instruction_mapping = extract_phase_idx_to_instruction_mapping(command_embeddings_dict)
        ds_metadata_dict["phase_to_instruction_mapping"] = phase_to_instruction_mapping
        ds_metadata_dict["candidate_texts"] = candidate_texts
        ds_metadata_dict["candidate_embeddings"] = candidate_embeddings

        return test_dataloader, ds_metadata_dict


def multitask_collate_fn(batch):
    """
    Custom collate function for handling multitask output dictionaries.
    """
    # Initialize lists for storing batched data
    image_sequences = []
    command_embeddings = []
    command_gts = []
    phase_histories = []
    multitask_labels_dicts = []
    
    # Optional list for jaw values
    jaw_data_sequences = []

    # Determine if jaw values are included by checking the length of the first item
    include_jaw_values = len(batch[0]) == 6

    for item in batch:
        # Unpack items based on whether jaw values are included
        if include_jaw_values:
            image_sequence, command_embedding, command_gt, jaw_data_sequence, phase_history, multitask_labels_dict = item
            jaw_data_sequences.append(jaw_data_sequence)
        else:
            image_sequence, command_embedding, command_gt, phase_history, multitask_labels_dict = item

        image_sequences.append(image_sequence)
        command_embeddings.append(command_embedding)
        command_gts.append(command_gt)
        phase_histories.append(phase_history)
        multitask_labels_dicts.append(multitask_labels_dict)

    # Stack all the tensors in the lists to create batched tensors
    image_sequences = torch.stack(image_sequences, dim=0)
    command_embeddings = torch.stack(command_embeddings, dim=0)

    # Create a dictionary for batched multitask labels
    batched_multitask_labels_dict = {}
    for key in multitask_labels_dicts[0].keys():
        batched_multitask_labels_dict[key] = torch.tensor([d[key] for d in multitask_labels_dicts])

    if include_jaw_values:
        jaw_data_sequences = torch.stack(jaw_data_sequences, dim=0)
        return image_sequences, command_embeddings, command_gts, jaw_data_sequences, phase_histories, batched_multitask_labels_dict
    else:
        return image_sequences, command_embeddings, command_gts, phase_histories, batched_multitask_labels_dict


"""
Test the SequenceDataset class.
"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from instructor.utils import set_seed

    # Parameters for the test
    dataset_name = "base_chole_clipping_cutting"  # "experiments" # "base_chole_clipping_cutting" # "base_chole_clipping_cutting" "base_chole_clipping_cutting_amos" "phantom_chole" 
    dataset_dir = os.path.join(os.getenv("PATH_TO_DATASET"), dataset_name) 
    incomplete_folders = ["tissue_1", "tissue_4", "tissue_23", "tissue_39", "tissue_19"]
    extra_repeated_phase_last_frame_sampling_flag = False
    all_dataset_tissues = [tissue for tissue in os.listdir(dataset_dir) if "tissue" in tissue and not (tissue in incomplete_folders and extra_repeated_phase_last_frame_sampling_flag)]
    for tissue_id in all_dataset_tissues:
        tissue_samples_ids =  [tissue_id] # "phantom_1" "tissue_12"
        camera_names = ["left_img_dir"] # ["endo_psm2", "left_img_dir", "endo_psm1"] # ["left_img_dir"] # ["endo_psm2", "left_img_dir", "endo_psm1"] # "right_img_dir"
        camera_file_suffixes = ["_left.jpg"] # ["_psm2.jpg", "_left.jpg", "_psm1.jpg"] #  ["_left.jpg"] # ["_psm2.jpg", "_left.jpg", "_psm1.jpg"] # "_right.jpg"
        history_len = 4
        prediction_offset = 15 # Get command for the current timestep
        history_step_size = 20
        num_episodes = 200 # Number of randlomy generated stitched episodes
        reduced_base_instruction_set_flag = False
        phase_history_len = 1
        prediction_step_size = 30
        recovery_probability = 1
        use_phase_history_flag=True,
        use_jaw_values_flag=True,
        phase_history_only_phase_switches_flag = True
        verbose_flag = True
        image_dim = (224, 224) # (H, W)
        llava_anyres_flag = True
        no_llava_anyres_global_image_flag = False
        wrist_images_rel_width = 3/4
        llava_anyres_rel_width = 2/3
        uniform_sampling_flag = True
        selected_multitasks = ["psm1_instrument", "curr_tube", "total_number_clips", "clip_loaded", "gallbladder_grabbed", "clips_left_tube", "clips_right_tube", "psm2_instrument_closed", "psm1_instrument_closed", "dominant_moving_direction", "is_correction"]
        use_seg_masks_input_flag = False
        merge_seg_masks_flag = True
        seg_mask_objs = ["clips", "left_tube", "right_tube"] # "clips", "left_tube", "right_tube", "flap"
        use_kinematic_indices_flag = True
        record_debug_episode_video_flag = True
        extra_corrections_sampling_flag = True
        extra_corrections_sampling_probability = 0
        extra_repeated_phase_last_frame_sampling_probability=0.1
        add_center_crop_view_flag = True
        distance_from_border_y=0.1
        distance_from_border_x=0.25
        y_offset=-0.1
        
        cmap = plt.get_cmap("tab10") # Color map to use

        # Define transforms/augmentations (resize transformation already applied in __getitem__ method)
        torch_input_transforms = []
        
        # NOTE: Automatic augmentations
        # torch_input_transforms.append(transforms.RandAugment())
        # torch_input_transforms.append(transforms.TrivialAugmentWide())
        # torch_input_transforms.append(transforms.AugMix())
        
        # NOTE: Manual augmentations
        # Torch augmentations
        torch_input_transforms.append(transforms.RandomResizedCrop(image_dim, scale=(0.8, 1.0)))
        torch_input_transforms.append(transforms.RandomRotation(degrees=[-5.0, 5.0]))
        torch_input_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)))
        # torch_color_input_transforms.append(v2.RandomPerspective(p=0.5))
        
        # Torch Color augmentations
        torch_color_input_transforms = []
        torch_color_input_transforms.append(transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.5, hue=0.08))
        # torch_color_input_transforms.append(v2.RandomPosterize(bits=7, p=0.25))
        # torch_color_input_transforms.append(v2.RandomAdjustSharpness(2, p=0.25))
        # torch_color_input_transforms.append(transforms.RandomApply([v2.GaussianBlur(kernel_size=5)], p=0.75))
        # torch_color_input_transforms.append(v2.RandomPhotometricDistort(p=0.8))
        # torch_color_input_transforms.append(transforms.RandomGrayscale(p=0.2))
        
        # Albumentations augmentations
        albumentation_input_transforms = []
        min_height, min_width = max(1, image_dim[0] // 40), max(1, image_dim[1] // 40)
        max_height, max_width = min(image_dim[0] // 30, image_dim[0]), min(image_dim[1] // 30, image_dim[1]) 
        albumentation_input_transforms.append(A.CoarseDropout(max_holes=128, max_height=max_height, max_width=max_width, min_holes=1, min_height=min_height, 
                        min_width=min_width, fill_value=0, p=0.5))
        
        # Store the transforms in a dictionary
        torch_input_transforms = transforms.Compose(torch_input_transforms)
        torch_color_input_transforms = transforms.Compose(torch_color_input_transforms)
        num_patches = SequenceDataset.get_num_patches(camera_names, add_center_crop_view_flag, llava_anyres_flag, no_llava_anyres_global_image_flag)
        num_input_images = num_patches * (history_len + 1)
        albumentations_additional_targets = dict(zip([f"image{i}" for i in range(num_input_images)], ["image"] * num_input_images))    
        albumentation_input_transforms = A.Compose(albumentation_input_transforms, additional_targets=albumentations_additional_targets)
        input_transforms = {"torch": torch_input_transforms, "albumentations": albumentation_input_transforms, "torch_color": torch_color_input_transforms}

        # Create a SequenceDataset instance
        dataset = SequenceDataset(
            "train",
            tissue_samples_ids,
            dataset_dir,
            camera_names,
            camera_file_suffixes,
            history_len,
            prediction_offset,
            history_step_size,
            num_episodes,
            input_transforms,
            reduced_base_instruction_set_flag=reduced_base_instruction_set_flag,
            use_phase_history_flag=use_phase_history_flag,
            use_jaw_values_flag=use_jaw_values_flag,
            phase_history_len=phase_history_len,
            prediction_step_size=prediction_step_size,
            recovery_probability=recovery_probability,
            phase_history_only_phase_switches_flag=phase_history_only_phase_switches_flag,
            image_dim=image_dim,
            llava_anyres_flag=llava_anyres_flag,
            no_llava_anyres_global_image_flag=no_llava_anyres_global_image_flag,
            wrist_images_rel_width=wrist_images_rel_width,
            llava_anyres_rel_width=llava_anyres_rel_width,
            uniform_sampling_flag=uniform_sampling_flag,
            selected_multitasks=selected_multitasks,
            use_seg_masks_input_flag=use_seg_masks_input_flag,
            merge_seg_masks_flag=merge_seg_masks_flag,
            seg_mask_objs=seg_mask_objs,
            use_kinematic_indices_flag=use_kinematic_indices_flag,
            record_debug_episode_video_flag=record_debug_episode_video_flag,
            extra_corrections_sampling_flag=extra_corrections_sampling_flag,
            extra_corrections_sampling_probability=extra_corrections_sampling_probability,
            extra_repeated_phase_last_frame_sampling_flag=extra_repeated_phase_last_frame_sampling_flag,
            extra_repeated_phase_last_frame_sampling_probability=extra_repeated_phase_last_frame_sampling_probability,
            add_center_crop_view_flag=add_center_crop_view_flag,
            distance_from_border_y=distance_from_border_y,
            distance_from_border_x=distance_from_border_x,
            y_offset=y_offset,
            verbose=verbose_flag)

        # Sample a random item from the dataset
        rdm_idx = np.random.randint(0, len(dataset))
        if use_jaw_values_flag:
            image_sequence, command_embedding, command, jaw_values, phase_history, multitask_label_indices_dict = dataset[rdm_idx]
        else:
            image_sequence, command_embedding, command, phase_history, multitask_label_indices_dict = dataset[rdm_idx]

        print(f"\nImage sequence shape: {image_sequence.shape}")
        print(f"Language embedding shape: {command_embedding.shape}")
        print(f"Command: {command}")
        print(f"Phase history ({phase_history_len=}): {phase_history}")
        if use_jaw_values_flag:
            print(f"Jaw values ({history_len=}):\n{jaw_values}")
        if multitask_label_indices_dict:
            multitask_label_dict = SequenceDataset.get_multitask_labels_from_label_indices_or_logits(multitask_label_indices_dict)
            print("Multitask labels:")
            for k, v in multitask_label_dict.items():
                print(f"{k}: {v}")

        # Create a figure with subplots: one row per timestamp, one column per camera
        fig_height = 4 * (history_len + 1)
        fig_width = len(dataset.camera_patch_names) * 3
        fig, axes = plt.subplots(history_len + 1, len(dataset.camera_patch_names), figsize=(fig_width, fig_height))
        if history_len == 0:
            axes = axes[np.newaxis, :]
        if len(dataset.camera_patch_names) == 1:
            axes = axes[:, np.newaxis]

        # Loop over each timestamp and camera to plot the images
        for t in range(history_len + 1):
            for cam_idx, cam_patch_name in enumerate(dataset.camera_patch_names):
                ax = axes[t, cam_idx]  # Get the specific subplot axis
                img = image_sequence[t, cam_idx].permute(1, 2, 0).numpy()
                
                # If the image is stacked with segmentation masks, extract the contours of the masks and put them on the image
                if img.shape[2] > 3:
                    img_bgr = cv2.cvtColor(img[:, :, :3]*255, cv2.COLOR_RGB2BGR).astype(np.uint8)  # Convert to BGR for OpenCV
                    seg_masks = image_sequence[t, cam_idx, 3:]  # The remaining channels are segmentation masks
                    if merge_seg_masks_flag:
                        class_ids = torch.unique(seg_masks)
                        seg_masks_per_class = [(seg_masks == class_id).numpy().astype(np.uint8).squeeze() for class_id in class_ids if class_id != 0]  # Skip the background class (obj_id=0)
                    else:
                        seg_masks_per_class = [seg_mask.squeeze().numpy().astype(np.uint8) for seg_mask in seg_masks]
                    for class_id, seg_mask in enumerate(seg_masks_per_class, start=1):
                        if seg_mask.sum() == 0:
                            print(f"Segmentation mask for class {class_id} is empty.")
                            continue
                        contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        class_color = np.array(cmap(class_id-1)[:3])  # Extract only RGB values
                        class_color_bgr = tuple((class_color * 255).astype(int)[::-1].tolist())  # Convert to BGR and ensure it's a tuple
                        cv2.drawContours(img_bgr, contours, -1, class_color_bgr, 2)  # Draw the contours on the BGR image
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert back to RGB for Matplotlib
                
                ax.imshow(img)
                ax.set_title(f"{cam_patch_name} at timestep {t}")
                ax.axis('off')  # Optionally turn off the axis

        # Set title to command
        dominant_moving_direction = multitask_label_dict["dominant_moving_direction"] if "dominant_moving_direction" in multitask_label_dict else None
        is_correction = multitask_label_dict["is_correction"] if "is_correction" in multitask_label_dict else None
        if use_jaw_values_flag and use_phase_history_flag:
            fig.suptitle(f"Command: {command}\nMovement: {dominant_moving_direction}\nCorrection: {is_correction}\nJaw values: \n{jaw_values}\nPhase history: {phase_history}", fontsize=16)
        elif use_phase_history_flag:
            fig.suptitle(f"Command: {command}\nMovement: {dominant_moving_direction}\nCorrection: {is_correction}\nPhase history: {phase_history}", fontsize=16)
        else:
            fig.suptitle(f"Command: {command}\nMovement: {dominant_moving_direction}\nCorrection: {is_correction}", fontsize=16)
        plt.tight_layout()
        example_dataset_plots_folder_path = os.path.join(path_to_yay_robot, "examples_plots", "dataset")
        if not os.path.exists(example_dataset_plots_folder_path):
            os.makedirs(example_dataset_plots_folder_path)
        file_name = os.path.join(example_dataset_plots_folder_path, f"{tissue_samples_ids=}_{history_len=}_{history_step_size=}_{image_dim=}_{use_seg_masks_input_flag=}.png")
        file_path = os.path.join(example_dataset_plots_folder_path, file_name)
        plt.savefig(file_path)
        print(f"\nSaved {file_name}.")
        plt.close(fig)  # Close the figure to free memory

        print("-"*40)
