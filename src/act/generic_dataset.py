import numpy as np
import torch
import os
import random

import h5py
import sys
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations, batch_rotations, transformations, trajectories
from torchvision import transforms, utils
import albumentations as A
from albumentations.pytorch import ToTensorV2

import seaborn as sns
from tqdm import tqdm
import json
import time

path_to_yay_robot = os.getenv('PATH_TO_SKAY_ROBOT')

if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
from aloha_pro.aloha_scripts.utils import initialize_model_and_tokenizer, encode_text
from auto_label_func import get_auto_label

from img_aug import DataAug


import IPython
e = IPython.embed

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def shift_image(image, shift_x, shift_y):
    """Shift the image by the given x and y offsets."""
    (h, w) = image.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, M, (w, h))
    return shifted_image

def rotate_image(image, angle):
    """Rotate the image by the given angle."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated



class EpisodicDatasetDvrkGeneric(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        tissue_sample_ids, 
        dataset_dir, 
        camera_names, 
        camera_file_suffixes, 
        # num_episodes,
        task_config,
        chunk_size=100,
        norm_stats=None,
        max_len=None,
        command_list=None,
        use_language=False,
        language_encoder="distilbert",
        ):

        super(EpisodicDatasetDvrkGeneric).__init__()

        if len(tissue_sample_ids) == 0:
            raise ValueError("No tissue samples found in the dataset directory.")
        
        # self.episode_ids = episode_ids
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0]
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.camera_file_suffixes = camera_file_suffixes
        self.norm_stats = norm_stats
        self.max_len = max_len
        if command_list is not None:
            self.command_list = [cmd.strip("'\"") for cmd in command_list]
        self.total_items = 0
        self.use_language = use_language
        self.chunk_size = chunk_size
        # self.num_episodes = num_episodes
        self.task_config = task_config
        self.action_mode = task_config['action_mode'][0]
        self.norm_scheme = task_config['norm_scheme']
        self.phantom = task_config['phantom']
        self.recovery_ratio = task_config['recovery_ratio']

        self.camera_names = self.task_config['camera_names']
        self.camera_suffixes = self.task_config['camera_file_suffixes']
        # assert len(self.camera_names) == len(self.camera_suffixes), "Camera names and suffixes must align"


        if task_config.get('use_sketch'):
            self.use_sketch = task_config['use_sketch']
        else:
            self.use_sketch = False

        if task_config.get('use_history'):
            self.use_history = task_config['use_history']
            self.history_len = task_config['history_len']
        else:
            self.use_history = False

        if task_config.get('segmentation_ratio'):
            self.segmentation_ratio = task_config['segmentation_ratio']
        else:
            self.segmentation_ratio = 0.0
        if task_config.get('use_auto_label'):
            self.use_auto_label = task_config['use_auto_label']
        else:
            self.use_auto_label = False

        if task_config.get('no_qpos'):
            self.no_qpos = task_config['no_qpos']
        else:
            self.no_qpos = False
        
        if task_config.get('estimation'):
            print("using estimation")
            self.estimation = task_config['estimation']
        else:
            self.estimation = False
            
        if task_config.get('stereo'):
            self.stereo = task_config['stereo']
            print("using stereo")
        else:
            self.stereo = False

        if task_config.get('goal_condition_style'):
            self.goal_condition_style = task_config['goal_condition_style']
        else:
            self.goal_condition_style = None


        self.goal_circle_size = 10
        self.is_sim = None
        self.cutting_action_pad_size = task_config['cutting_action_pad_size']
        self.img_height, self.img_width = [360, 480]
        self.dropout_correction_prob = 0.5
        # self.img_height, self.img_width = [224, 224]
        self.num_samples = task_config['num_episodes']
        if task_config.get('merging_subtasks'):
            self.merging_subtasks = task_config['merging_subtasks']
        else:
            self.merging_subtasks = False
        if self.merging_subtasks:
            if task_config.get('available_phase_commands'):
                self.available_phase_commands = task_config['available_phase_commands']
                print("Using custom phase commands for merging subtasks:", self.available_phase_commands)
            else:
                self.available_phase_commands = {
                    (1, 3): "apply first clip on the left tube", 
                    (4, 5):  "apply second clip on the left tube", 
                    (6, 7): "apply third clip on the left tube",
                    (8, 9): "cut the left tube",
                    (10, 11): "apply first clip on the right tube",
                    (12, 13): "apply second clip on the right tube",
                    (14, 15): "apply third clip on the right tube",
                    (16, 17): "cut the right tube"
                }
                print("Using default phase commands for merging subtasks:", self.available_phase_commands)

        self.arm_command_labels = ["move left arm to the left", "move left arm higher", "move left arm away from me", 
                    "move left arm to the right", "move left arm lower", "move left arm towards me", 
                    "move right arm to the left", "move right arm higher", "move right arm away from me",
                    "move right arm to the right", "move right arm lower", "move right arm towards me",
                    "close both grippers", "close left gripper", "close right gripper",
                    "open both grippers", "open left gripper", "open right gripper",
                    "do not move"]
        # Load the tissue samples and their phases and demos (for later stitching of the episodes)        
        self.tissue_phase_demo_dict = {}
        self.command_embeddings_dict = {}
        # self.command_embeddings_dict_json = {}

        for tissue_sample_id in tissue_sample_ids:
            if self.phantom:
                tissue_sample_name = f"phantom_{tissue_sample_id}"
            else:
                tissue_sample_name = f"tissue_{tissue_sample_id}"
            tissue_sample_dir_path = os.path.join(dataset_dir, tissue_sample_name)
            phases = os.listdir(tissue_sample_dir_path)
            self.tissue_phase_demo_dict[tissue_sample_name] = {}

            for phase_sample in phases:
                demo_samples_path = os.path.join(tissue_sample_dir_path, phase_sample)

                if os.path.isfile(demo_samples_path):
                    continue  # Skip if the tissue sample path is not a directory

                demo_samples = os.listdir(demo_samples_path)

                ## remove corrections folder
                for demo_sample in demo_samples:
                    if demo_sample == "Corrections" or demo_sample.endswith(".json"):
                        demo_samples.remove(demo_sample)

                ## initialize the dictionary for the tissue sample
                if tissue_sample_name not in self.tissue_phase_demo_dict:
                    self.tissue_phase_demo_dict[tissue_sample_name] = {}

                # ## adjust the number of demos for the recovery phase
                # if phase_sample.endswith("_recovery"):
                #     num_of_perfect_demos = len(os.listdir(os.path.join(tissue_sample_dir_path, phase_sample[:-9])))
                #     num_of_recovery_demos = int(num_of_perfect_demos * self.recovery_ratio)
                    
                #     print(f"tissue: {tissue_sample_id}, Recovery phase: {phase_sample}, num of perfect demos: {num_of_perfect_demos}, num of recovery demos: {num_of_recovery_demos}")
                #     demo_samples = demo_samples[:num_of_recovery_demos]

                # Add or update the demo samples in the dictionary
                self.tissue_phase_demo_dict[tissue_sample_name].setdefault(phase_sample, []).extend(demo_samples)


        print("num of tissues:", len(self.tissue_phase_demo_dict.keys()))
        print(self.tissue_phase_demo_dict.keys())
        print("phases:", self.tissue_phase_demo_dict[tissue_sample_name].keys())
        print("num of demos per phase:", {phase: len(demo_samples) for phase, demo_samples in self.tissue_phase_demo_dict[tissue_sample_name].items()})
        
        # print("num of samples:", sum(len(samples) for samples in self.tissue_phase_demo_dict.values()))
        total_count = 0
        for phase_dict in self.tissue_phase_demo_dict.values():
            for demo_samples in phase_dict.values():
                total_count += len(demo_samples)
        self.num_samples = total_count
        print("total count:", total_count)
        # print("self.command_embeddings_dict: ", self.command_embeddings_dict.keys())
        ## create language embeddings
        if self.use_language:

            self.language_encoder = language_encoder
            # tokenizer, model = initialize_model_and_tokenizer(self.language_encoder)
            unique_phase_folder_names = np.unique([phase_folder_name for tissue_sample in self.tissue_phase_demo_dict.values() for phase_folder_name in tissue_sample.keys()])

            # print("phase:", unique_phase_folder_names)
            print("\ngenerating command embeddings...\n")
            # self.command_embeddings_dict[tissue_sample_name] = self.generate_command_embeddings(unique_phase_folder_names, self.language_encoder, tokenizer, model)
            #if self.use_auto_label:
            #    json_name = f"candidate_embeddings_corrections_{self.language_encoder}.json"
            #else:
            json_name = f"candidate_embeddings_{self.language_encoder}.json"
            json_path = os.path.join(dataset_dir, json_name)

            self.command_embeddings_dict = self.get_command_embeddings_from_json(unique_phase_folder_names, json_path)
            print(self.command_embeddings_dict.keys())
            # print(self.command_embeddings_dict["1_needle_pickup"].keys())
            # print("embeddings are the same:", self.command_embeddings_dict[tissue_sample_name] == self.command_embeddings_dict_json[tissue_sample_name])

            # del tokenizer, model
            # print(f"   {phase_sample}, {demo_samples}\n")
        self.all_samples = [(tissue_sample, phase, sample) 
                            for tissue_sample in self.tissue_phase_demo_dict
                            for phase in self.tissue_phase_demo_dict[tissue_sample]
                            for sample in self.tissue_phase_demo_dict[tissue_sample][phase]]
        
        ## for weighted random sampler
        self.sample_task_labels = []
        for sample in self.all_samples:
            _, phase, _ = sample
            task_label = phase.split("_")[0]  # "1", "2", or "3"
            self.sample_task_labels.append(task_label)

        self.header_name_qpos_psm1 = ["psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
                                "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
                                "psm1_jaw"]
        
        self.header_name_qpos_psm2 = ["psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
                                "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
                                "psm2_jaw"]

        self.header_name_actions_psm1 = ["psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
                                    "psm1_sp.orientation.x", "psm1_sp.orientation.y", "psm1_sp.orientation.z", "psm1_sp.orientation.w",
                                    "psm1_jaw_sp"]

        self.header_name_actions_psm2 = ["psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
                                    "psm2_sp.orientation.x", "psm2_sp.orientation.y", "psm2_sp.orientation.z", "psm2_sp.orientation.w",
                                    "psm2_jaw_sp"]
        
        self.header_ecm = ["ecm_pose.position.x", "ecm_pose.position.y", "ecm_pose.position.z",
                            "ecm_pose.orientation.x", "ecm_pose.orientation.y", 
                            "ecm_pose.orientation.z", "ecm_pose.orientation.w"]
        
        self.quat_cp_psm1 = ["psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w"]
        self.quat_cp_psm2 = ["psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w"]

        # self.transforms = DataAug([224, 224])
        self.transforms = DataAug([self.img_height, self.img_width], use_history=(self.use_history or self.use_sketch), stereo=self.stereo)

        # self.__getitem__(0) # initialize self.is_sim

    def generate_command_embeddings(self, unique_phase_folder_names, encoder, tokenizer, model):
        if self.merging_subtasks:
            # Returns a dictionary containing the phase command as key and a tuple of the phase command and phase embedding as value
            phase_command_embeddings_dict = {}
            for phase_folder_name in tqdm(unique_phase_folder_names, desc="Embedding phase commands"):
                if phase_folder_name.endswith("_recovery"):
                    phase_folder_name = phase_folder_name[:-9]
                elif phase_folder_name.startswith("ACTUAL_CUTTING"):
                    if phase_folder_name.endswith("_left"):
                        phase_folder_name = "8_go_to_the_cutting_position_left_tube"
                    elif phase_folder_name.endswith("_right"):
                        phase_folder_name = "16_go_to_the_cutting_position_right_tube"

                phase_num, phase_command = phase_folder_name.split("_")[0], " ".join(phase_folder_name.split("_")[1:])
                # print("phase_num:", phase_num)
                if phase_num.isdigit():
                    phase_num = int(phase_num)
                    for phase_range, command in self.available_phase_commands.items():
                        if phase_range[0] <= phase_num <= phase_range[1]:
                            phase_command = command
                            break

                    embedding = encode_text(phase_command, encoder, tokenizer, model)
                    phase_command_embeddings_dict[phase_folder_name]= (phase_command, embedding)

            return phase_command_embeddings_dict
        else:
            # Returns a dictionary containing the phase command as key and a tuple of the phase command and phase embedding as value
            phase_command_embeddings_dict = {}
            for phase_folder_name in tqdm(unique_phase_folder_names, desc="Embedding phase commands"):
                if phase_folder_name.endswith("_recovery"):
                    phase_folder_name = phase_folder_name[:-9]
                elif phase_folder_name.startswith("ACTUAL_CUTTING"):
                    if phase_folder_name.endswith("_left"):
                        phase_folder_name = "8_go_to_the_cutting_position_left_tube"
                    elif phase_folder_name.endswith("_right"):
                        phase_folder_name = "16_go_to_the_cutting_position_right_tube"
                # Extract the phase command from the folder name (removing the phase idx and the "_" in between the words)
                _, phase_command = phase_folder_name.split("_")[0], " ".join(phase_folder_name.split("_")[1:])
                embedding = encode_text(phase_command, encoder, tokenizer, model)
                phase_command_embeddings_dict[phase_folder_name]= (phase_command, embedding)

            return phase_command_embeddings_dict

    def get_command_embeddings_from_json(self, unique_phase_folder_names, json_file_name):
        phase_command_embeddings_dict = {}

        try:
            with open(json_file_name, "r") as f:
                episode_data = json.load(f)
        except FileNotFoundError:
            print(f"File {json_file_name} not found.")
            return phase_command_embeddings_dict
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file {json_file_name}.")
            return phase_command_embeddings_dict

        for phase_folder_name in tqdm(unique_phase_folder_names, desc="Embedding phase commands"):
            if phase_folder_name.endswith("_recovery"):
                phase_folder_name = phase_folder_name[:-9]
            elif phase_folder_name.startswith("ACTUAL_CUTTING"):
                if phase_folder_name.endswith("_left"):
                    phase_folder_name = "8_go_to_the_cutting_position_left_tube"
                elif phase_folder_name.endswith("_right"):
                    phase_folder_name = "16_go_to_the_cutting_position_right_tube"
            # Extract the phase command from the folder name (removing the phase idx and the "_" in between the words)
            _, phase_command = phase_folder_name.split("_")[0], " ".join(phase_folder_name.split("_")[1:])

            if self.use_auto_label:
                for label in self.arm_command_labels:
                    phase_command_embeddings_dict.setdefault(phase_folder_name, {})
                    if label != "do not move":
                        combined_command = label
                    else:
                        combined_command = phase_command
                    # Search for the command in the JSON data
                    found_embedding = None
                    # print(f"Searching for command: {combined_command}")
                    for item in episode_data:
                        # print(f"command: {item.get('command')}")
                        # input("Press Enter to continue...")
                        if isinstance(item, dict) and item.get('command') == combined_command:
                            found_embedding = item.get('embedding')
                            # print(f"Embedding found for command: {phase_command}")
                            break
                
                    # Store the found embedding (if any)
                    if found_embedding is not None:
                        phase_command_embeddings_dict[phase_folder_name][label] = (combined_command, found_embedding)

                    else:
                        print(f"Embedding not found for command: {combined_command}")
            else:
                phase_command_embeddings_dict.setdefault(phase_folder_name, {})
                found_embedding = None
                for item in episode_data:
                    if isinstance(item, dict) and item.get('command') == phase_command:
                        found_embedding = item.get('embedding')
                        break
            
                if found_embedding is not None:
                    phase_command_embeddings_dict[phase_folder_name] = (phase_command, found_embedding)

                else:
                    print(f"Embedding not found for command: {phase_command}")

        return phase_command_embeddings_dict


    #### TODO:
    # def append_point_to_text(self, points, language_embeddeings):

    #     return new_embeddings

    def compute_diff_actions(self, qpos, action):
        """
        qpos: current position [9]
        action: actions commanded by the user [n_actions x 9]
        returns: relative actions w.r.t qpos
        """
        # find diff first and then fill-in the quaternion differences properly
        diff = action - qpos

        quat_init = qpos[3:7]
        quat_actions = action[:, 3:7]

        # convert quaternions to rotation matrices
        r_init = R.from_quat(quat_init)
        r_actions = R.from_quat(quat_actions)
        # find their diff
        diff_rs = r_init.inv()*r_actions 
        # extract their first two columns
        diff_6d = diff_rs.as_matrix()[:,:,:2]
        diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
        
        diff_expand = np.zeros((diff.shape[0], 10)) # TODO: hard-coded dim (10) for a single arm
        diff_expand[:diff.shape[0], 0:diff.shape[1]] = diff 
        diff = diff_expand

        diff[:, 3:9] = diff_6d
        diff[:, 9] = action[:, -1] # fill in the jaw angle (note: jaw angle is not relative)
        return diff
    
    def compute_diff_actions_relative_endoscope(self, qpos, action):
        """
        qpos: current position [9]
        action: actions commanded by the user [n_actions x 9]
        returns: relative actions w.r.t qpos
        """
        # find diff first and then fill-in the quaternion differences properly
        diff = action - qpos
        quat_actions = action[:, 3:7]

        r_actions = R.from_quat(quat_actions)
        diff_rs = r_actions 
        # extract their first two columns
        diff_6d = diff_rs.as_matrix()[:,:,:2]
        diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
        
        diff_expand = np.zeros((diff.shape[0], 10)) # TODO: hard-coded dim (10) for a single arm
        diff_expand[:diff.shape[0], 0:diff.shape[1]] = diff 
        diff = diff_expand

        diff[:, 3:9] = diff_6d
        diff[:, 9] = action[:, -1] # fill in the jaw angle (note: jaw angle is not relative)
        return diff
    
    def compute_relative_actions_in_SE3(self, qpos, action):
        """
        Note: this is the proper implementation
        qpos: current position (measured_cp), xyz, xyzw, jaw angle (8-dim vector)
        action: set point on the dvrk (action_horizon x 8)
        
        returns: relative position and rotation w.r.t qpos
        """
        
        diff = np.zeros((action.shape[0], 10)) # TODO: hard-coded dim (10) for a single arm

        # convert current pose to SE(3)
        qpos_wxyz = rotations.quaternion_wxyz_from_xyzw(qpos[3:7])
        qpos_py3d = np.concatenate((qpos[0:3], qpos_wxyz))
        g_qpos = transformations.transform_from_pq(qpos_py3d) # no jaw angle!

        # convert actions to SE(3)
        action_wxyz = batch_rotations.batch_quaternion_wxyz_from_xyzw(action[:, 3:7]) 
        action_py3d = np.concatenate((action[:, 0:3], action_wxyz), axis = 1)
        g_action = trajectories.transforms_from_pqs(action_py3d)

        # invert current pose
        g_qpos_inv = transformations.invert_transform(g_qpos)
        diff_SE3 = trajectories.concat_one_to_many(g_qpos_inv, g_action)

        # construct 6d rot
        diff_6d = diff_SE3[:,0:3,:2]
        diff_6d = diff_6d.transpose(0,2,1).reshape(-1, 6) # first column then second column
        
        # fill in translation elements
        diff[:, 0:3] = diff_SE3[:, 0:3, 3] # replace the translations with the last column first three rows of SE3
        # fill in 6d rot
        diff[:, 3:9] = diff_6d
        # fill in jaw angle (note: jaw angle is absolute, not relative)
        diff[:, 9] = action[:, 7]
        return diff

    # misnomer: jaw angles are also being normalized
    def min_max_scale_positions_only(self, diffs):
        """
        diffs: n_actions x 20
        return: normalized n_actions x 20
        Note: BOTH POSITIONS AND JAW ANGLES ARE NORMALIZED (orientations remain original)
        """
        max_ = self.task_config['action_mode'][1]['max_']
        min_ = self.task_config['action_mode'][1]['min_']
        normalized = (diffs - min_) / (max_ - min_) * 2 - 1

        # replace w/ originals for 6D rot
        normalized[:, 3:9] = diffs[:, 3:9]
        normalized[:, 13:19] = diffs[:, 13:19]

        return normalized
    
    def standardize_positions_only(self, diffs):
        """
        diffs: n_actions x 20
        return: normalized n_actions x 20 (zero mean unit variance)
        Note: BOTH POSITIONS AND JAW ANGLES ARE NORMALIZED (orientations remain original)
        """
        mean = self.task_config['action_mode'][1]['mean']
        std = self.task_config['action_mode'][1]['std']
        # print("mean shape", mean.shape)
        # print("std shape", std.shape)
        normalized = (diffs - mean) / std

        # replace w/ originals for 6D rot
        normalized[:, 3:9] = diffs[:, 3:9]
        normalized[:, 13:19] = diffs[:, 13:19]

        return normalized


    def preprocess_img(self, img, start_ts):
        if img is None:
            print("Image is None:", start_ts)
        img = cv2.resize(img, [self.img_width, self.img_height])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # construct observations
        img = torch.from_numpy(img).float() # channel last

        # bring channel to to the third
        img = torch.einsum('h w c -> c h w', img)

        # normalize image and change dtype to float
        img = img / 255.0

        return img

    def create_offset_map_with_gradient(self, image_shape, insert_point, exit_point, normalize_size=224.0, device='cpu', eps=1e-6):
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


    def offset_map_to_rgb_visual(self, offset_map):
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

    def __len__(self):
     
        return len(self.episode_ids)


    def __getitem__(self, index):
        
        try:
            # Get the tissue sample, phase, and sample based on the index
            episode_id = self.episode_ids[index]
            if episode_id < self.num_samples:
                tissue_sample, phase, sample = self.all_samples[episode_id]
            else:
                print("episode_id out of range")
                tissue_sample, phase, sample = self.all_samples[episode_id % self.num_samples]

            dataset_path = os.path.join(self.dataset_dir, f"{tissue_sample}/{phase}/{sample}")
            csv_path = os.path.join(dataset_path, "ee_estimate.csv" if self.estimation else "ee_csv.csv")
            csv = pd.read_csv(csv_path)
            episode_len = len(csv)
            start_ts = np.random.choice(episode_len)
            use_segmentation = random.random() < self.segmentation_ratio
            endo_img_dir, suffix = ("contours_on_image", "_contours.jpg") if use_segmentation else ("left_img_dir", "_left.jpg")

            # Handle cutting task padding
            if (phase.startswith("8_go_to_the_cutting_position_left_tube") or phase.startswith("16_go_to_the_cutting_position_right_tube")) and start_ts >= episode_len - self.cutting_action_pad_size:
                img_idx = episode_len - self.cutting_action_pad_size - 1
            else:
                img_idx = start_ts

            # -------------------------------
            # 1. Load raw image
            # -------------------------------        
            
            img_dict_raw = {}
            for cam_name, cam_suffix in zip(self.camera_names, self.camera_suffixes):
                subdir = {
                    '_left.jpg': 'left_img_dir',
                    '_right.jpg': 'right_img_dir',
                    '_psm1.jpg': 'endo_psm1',
                    '_psm2.jpg': 'endo_psm2',
                }.get(cam_suffix, 'left_img_dir')
                path = os.path.join(dataset_path, subdir, f"frame{img_idx:06d}{cam_suffix}")
                img = cv2.imread(path)
                if img is None:
                    raise FileNotFoundError(f"Image not found at: {path}")
                img_dict_raw[cam_name] = img

            # -------------------------------
            # 2. Plot goal points if needed (needle throw only)
            # -------------------------------
            if self.goal_condition_style == "plot" and phase.startswith("2_needle_throw"):
                # print("plotting goal points")
                clicked_csv_path = os.path.join(dataset_path, "clicked_point.csv")
                if os.path.exists(clicked_csv_path):
                    clicked = pd.read_csv(clicked_csv_path)
                    if not clicked.empty:
                        for _, row in clicked.iterrows():
                            x, y = int(row['x']), int(row['y'])
                            cv2.circle(img_dict_raw['left'], (x, y), self.goal_circle_size, (0, 255, 0), -1)

            if self.goal_condition_style == "mask":
                if phase.startswith("2_needle_throw"):
                    clicked_csv_path = os.path.join(dataset_path, "clicked_point.csv")
                    if os.path.exists(clicked_csv_path):
                        clicked = pd.read_csv(clicked_csv_path)
                        if not clicked.empty and len(clicked) >= 2:
                            # Create a 3-channel mask (H, W, 3) with all zeros
                            h, w = img_dict_raw["left"].shape[:2]
                            clicked_points_mask = np.zeros((h, w, 3), dtype=np.uint8)

                            # Draw insertion point (first point) as red
                            insert_x, insert_y = int(clicked.iloc[0]['x']), int(clicked.iloc[0]['y'])
                            cv2.circle(clicked_points_mask, (insert_x, insert_y), radius=10, color=(255, 0, 0), thickness=-1)  # Red in BGR

                            # Draw exit point (second point) as green
                            exit_x, exit_y = int(clicked.iloc[1]['x']), int(clicked.iloc[1]['y'])
                            cv2.circle(clicked_points_mask, (exit_x, exit_y), radius=10, color=(0, 255, 0), thickness=-1)  # Green in BGR

                            img_dict_raw["mask"] = clicked_points_mask
                        else:
                            print("clicked_point.csv has fewer than 2 points, skipping mask")
                            img_dict_raw["mask"] = np.zeros_like(img_dict_raw["left"])
                    else:
                        print("clicked_point.csv not found")
                        img_dict_raw["mask"] = np.zeros_like(img_dict_raw["left"])
                else:
                    img_dict_raw["mask"] = np.zeros_like(img_dict_raw["left"])

            if self.goal_condition_style == "dot" and phase.startswith("2_needle_throw"):
                # print("plotting goal points")
                clicked_csv_path = os.path.join(dataset_path, "clicked_point.csv")
                if os.path.exists(clicked_csv_path):
                    clicked_csv_path = os.path.join(dataset_path, "clicked_point.csv")
                    if os.path.exists(clicked_csv_path):
                        clicked = pd.read_csv(clicked_csv_path)
                        if not clicked.empty and len(clicked) >= 2:
                            # Create a 3-channel mask (H, W, 3) with all zeros
                            h, w = img_dict_raw["left"].shape[:2]
                            clicked_points_mask = np.zeros((h, w, 3), dtype=np.uint8)

                            # Draw insertion point (first point) as red
                            insert_x, insert_y = int(clicked.iloc[0]['x']), int(clicked.iloc[0]['y'])
                            cv2.circle(clicked_points_mask, (insert_x, insert_y), radius=10, color=(255, 0, 0), thickness=-1)  # Red in BGR

                            # Draw exit point (second point) as green
                            exit_x, exit_y = int(clicked.iloc[1]['x']), int(clicked.iloc[1]['y'])
                            cv2.circle(clicked_points_mask, (exit_x, exit_y), radius=10, color=(0, 255, 0), thickness=-1)  # Green in BGR

                            ## overlay the mask on the image
                            # Only blend where the mask has non-zero content
                            nonzero_mask = np.any(clicked_points_mask != 0, axis=-1)
                            overlay = img_dict_raw["left"].copy()
                            overlay[nonzero_mask] = cv2.addWeighted(
                                img_dict_raw["left"], 0.5, clicked_points_mask, 0.5, 0
                            )[nonzero_mask]
                            # overlay = cv2.addWeighted(img_dict_raw["left"], 0.5, clicked_points_mask, 0.5, 0)
                            img_dict_raw["left"] = overlay
                    else:
                        print("clicked_point.csv not found")

            if self.goal_condition_style == "map":
                if phase.startswith("2_needle_throw"):
                    # print("plotting goal points")
                    clicked_csv_path = os.path.join(dataset_path, "clicked_point.csv")
                    if os.path.exists(clicked_csv_path):
                        clicked = pd.read_csv(clicked_csv_path)
                        if not clicked.empty and len(clicked) >= 2:
                            # Create a 3-channel mask (H, W, 3) with all zeros
                            h, w = img_dict_raw["left"].shape[:2]

                            # Load clicked points
                            insert_x = int(clicked.iloc[0, 0])
                            insert_y = int(clicked.iloc[0, 1])
                            exit_x = int(clicked.iloc[1, 0])
                            exit_y = int(clicked.iloc[1, 1])
                            insert_point = (insert_x, insert_y)
                            exit_point = (exit_x, exit_y)
                            # Create offset map
                            offset_map = self.create_offset_map_with_gradient(
                                image_shape=(h, w),
                                insert_point=insert_point,
                                exit_point=exit_point,
                                device='cpu'
                            )

                            rgb_offset_viz = self.offset_map_to_rgb_visual(offset_map)
                            # img_dict_raw["left"] = cv2.addWeighted(img_dict_raw["left"], 0.5, rgb_offset_viz, 0.5, 0)
                            img_dict_raw["mask"] = rgb_offset_viz
                        else:
                            print("clicked_point.csv has fewer than 2 points, skipping mask")
                            img_dict_raw["mask"] = np.zeros_like(img_dict_raw["left"])
                    else:
                        print("clicked_point.csv not found")
                        img_dict_raw["mask"] = np.zeros_like(img_dict_raw["left"])
                else:
                    img_dict_raw["mask"] = np.zeros_like(img_dict_raw["left"])

                    # img_dict_raw["mask"] = clicked_points_mask
                    # fig = plt.figure(figsize=(10, 5))
                    # plt.subplot(1, 2, 1)
                    # plt.title("Image")
                    # plt.imshow(img_dict_raw["left"])
                    # plt.subplot(1, 2, 2)
                    # plt.title("Mask")
                    # plt.imshow(clicked_points_mask)
                    # plt.show()

            

            # --------------- (Chole only) ----------------
            # ## rectify rotation of the right wrist cam 
            # rotate_ids = [5, 6, 8, 12, 13, 14, 18]
            # for rotate_id in rotate_ids:
            #     if not self.phantom and tissue_sample.endswith(f"{rotate_id}"):
            #         angle = -52.0
            #         img_rw = rotate_image(img_rw, angle)
            #         shift_x, shift_y = 10, 0 
            #         img_rw = shift_image(img_rw, shift_x, shift_y)
            #         break  # Exit the loop after the first match

            # -------------------------------
            # 3. Preprocess and augment images
            # -------------------------------
            img_dict = {k: self.preprocess_img(v, start_ts) for k, v in img_dict_raw.items()}


            # -------------------------------
            # 4. Optional history/sketch
            # -------------------------------
            if self.use_history or self.use_sketch:
                hist_idx = img_idx - self.history_len
                if hist_idx >= 0:
                    hist_path = os.path.join(dataset_path, "labelled_img_depth_relative", f"frame{hist_idx:06d}_left.jpg")
                    hist_img = cv2.imread(hist_path)
                    if hist_img is not None:
                        img_dict['img_l_hist'] = self.preprocess_img(hist_img, start_ts)
                    else:
                        img_dict['img_l_hist'] = np.zeros_like(next(iter(img_dict.values())))
                else:
                    img_dict['img_l_hist'] = np.zeros_like(next(iter(img_dict.values())))



            # -------------------------------
            #  5. Apply data augmentation
            # -------------------------------

            tfmed = self.transforms(img_dict)
            image_data = np.stack([tfmed[k] for k in sorted(tfmed.keys())], axis=0)


            # -------------------------------
            #  6. Load and compute action data
            # -------------------------------

            # get current position and actions
            # qpos_psm1 = csv[self.header_name_qpos_psm1].iloc[start_ts, :].to_numpy()
            qpos_psm1 = csv[self.header_name_actions_psm1].iloc[start_ts, :].to_numpy()
            action_psm1 = csv[self.header_name_actions_psm1].iloc[start_ts:start_ts+400].to_numpy() # note 400 added here
            # qpos_psm2 = csv[self.header_name_qpos_psm2].iloc[start_ts, :].to_numpy()
            qpos_psm2 = csv[self.header_name_actions_psm2].iloc[start_ts, :].to_numpy()
            action_psm2 = csv[self.header_name_actions_psm2].iloc[start_ts:start_ts+400].to_numpy() # note 400 added here

            # compute relative actions TODO: make it work for SE3 scenarios
            if self.action_mode == 'hybrid':
                diff_psm1 = self.compute_diff_actions(qpos_psm1, action_psm1)
                diff_psm2 = self.compute_diff_actions(qpos_psm2, action_psm2)
            elif self.action_mode == 'ego':
                diff_psm1 = self.compute_relative_actions_in_SE3(qpos_psm1, action_psm1)
                diff_psm2 = self.compute_relative_actions_in_SE3(qpos_psm2, action_psm2)
            elif self.action_mode == 'relative_endoscope':
                diff_psm1 = self.compute_diff_actions_relative_endoscope(qpos_psm1, action_psm1)
                diff_psm2 = self.compute_diff_actions_relative_endoscope(qpos_psm2, action_psm2)

            else:
                raise(NotImplementedError) 

            # stack the actions along column dim
            action = np.column_stack((diff_psm1, diff_psm2))

            # normalize data
            if self.norm_scheme == 'min_max': 
                action = self.min_max_scale_positions_only(action)
            elif self.norm_scheme == 'std':
                action = self.standardize_positions_only(action)
            else:
                raise NotImplementedError

            action_len = min(episode_len - start_ts, 400) # TODO: a bit messy code
            padded_action = np.zeros((400, 20), dtype=np.float32) # TODO: this is hardcoded to be 400 
            # timesteps by default, but you will be taking a subset anyway later on i.e. chunk size, so it's probably ok and also
            # you will never be making predictions beyond 400 timestep horizon
            # also hardcoded to 10 dim per arm
            padded_action[:action_len] = action
            is_pad = np.zeros(400)
            is_pad[action_len:] = 1

            # set current poses to zeros (dvrk kinematics unreliable)
            qpos = np.zeros(20)

            # construct observations
            # image_data = torch.from_numpy(all_cam_images)
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            # -------------------------------
            #  7. Command Embedding
            # -------------------------------
            if self.use_language:
                directional_label = None
                if phase.endswith("_recovery"):
                    directional_label = get_auto_label(csv, start_ts)
                    phase = phase[:-9]

                if self.use_auto_label:
                    command_tuple = self.command_embeddings_dict[phase].get(directional_label, self.command_embeddings_dict[phase]["do not move"])
                else:
                    command_tuple = self.command_embeddings_dict[phase]

                _, embedding = command_tuple
                ### TODO: add point coordinates to the text embedding
                # if not self.goal_condition_style and self.goal_condition_style == "text":
                #     clicked_point_csv = pd.read_csv(os.path.join(dataset_path, "clicked_point.csv"))
                #     points = []
                #     if clicked_point_csv.shape[0] > 0:
                #         for i in range(clicked_point_csv.shape[0]):
                #             # print("clicked point:", clicked_point_csv.iloc[i])
                #             points.append((clicked_point_csv.iloc[i]['x'], clicked_point_csv.iloc[i]['y']))
                #             # x, y = int(clicked_point_csv.iloc[i]['x']), int(clicked_point_csv.iloc[i]['y'])
                #     embedding = self.append_point_to_text(points, embedding)
                command_embedding = torch.tensor(embedding).squeeze()

                if self.no_qpos:
                    return image_data, action_data, is_pad, command_embedding
                return image_data, qpos_data, action_data, is_pad, command_embedding

            return image_data, qpos_data, action_data, is_pad
        
        # Handle exceptions
        except FileNotFoundError as e:
            print(f"File not found at index {index}: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            print(f"Empty data error at index {index}: {e}")
            raise
        except KeyError as e:
            print(f"Key error at index {index}: {e}")
            raise
        except ValueError as e:
            print(f"Value error at index {index}: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error at index {index}: {e}")                



        
"""
Test the EpisodicDatasetDvrkGeneric class.
"""
if __name__ == "__main__":
    # seed = random.randint(0, 1000)
    set_seed(0)
    for i in range(10):
        # seed = random.randint(0, 1000)
        # set_seed(seed)
        # Parameters for the test
        path_to_dataset = os.getenv("PATH_TO_DATASET")
        # path_to_dataset = "/home/imerse/chole_ws/data"

        dataset_dir = os.path.join(path_to_dataset, "invivo_chole")
        use_language_flag = True
        from dvrk_scripts.constants_dvrk import TASK_CONFIGS
        task_config = TASK_CONFIGS['invivo_test']
        camera_names = task_config['camera_names']
        tissue_samples_ids = task_config["tissue_samples_ids"]
        num_episodes = task_config["num_episodes"]
        camera_file_suffixes = task_config['camera_file_suffixes']
        episode_ids = [i for i in range(num_episodes)]
        no_qpos = task_config.get('no_qpos', False)
        dataset = EpisodicDatasetDvrkGeneric(
                    episode_ids,
                    tissue_samples_ids,
                    dataset_dir,
                    camera_names,
                    camera_file_suffixes,
                    # num_episodes,
                    task_config,
                    chunk_size=60,
                    use_language=use_language_flag
                    )

        # Sample a random item from the dataset
        rdm_idx = np.random.randint(0, len(dataset))
        print("idx:", rdm_idx)
        if use_language_flag:
            if no_qpos:
                image_data, action_data, is_pad, command_embedding = dataset[rdm_idx]
            image_data, qpos_data, action_data, is_pad, command_embedding = dataset[rdm_idx]
        else:
            if no_qpos:
                image_data, action_data, is_pad = dataset[rdm_idx]
            image_data, qpos_data, action_data, is_pad = dataset[rdm_idx]   


        # Create a figure with subplots: one row per timestamp, one column per camera
        
        fig, axes = plt.subplots(1, len(image_data), figsize=(15, 10))
        for cam_idx, img in enumerate(image_data):

            # Check and possibly transpose the shape if needed
            if img.shape[0] == 3 and len(img.shape) == 3:
                img = np.transpose(img, (1, 2, 0))  # Transpose to (height, width, channels)

            axes[cam_idx].imshow(img)
            axes[cam_idx].axis('off')  # Optionally turn off the axis

        # set the title of the figure
        # fig.suptitle(f"{command}")
        # plt.show()
        plt.savefig(f"./visualization_{i}.png")
        
        # fig, axes = plt.subplots(1, 1, figsize=(15, 10))
        # for cam_idx, cam_name in enumerate(camera_names):
        #     img = image_data[cam_idx]  # Assuming image_data is a numpy array or compatible type

        # # Check and possibly transpose the shape if needed
        # if img.shape[0] == 3 and len(img.shape) == 3:
        #     img = np.transpose(img, (1, 2, 0))  # Transpose to (height, width, channels)

        # axes.imshow(img)
        # axes.set_title("left_img")
        # axes.axis('off')  # Optionally turn off the axis
        # plt.show()
        # plt.savefig(f"./visualization_{i}.png")
