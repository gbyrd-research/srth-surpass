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
import cv2
import random 
import shutil

class KalmanFilter:
    def __init__(self, init_pose, dt, state_variance, measurement_variance, initial_covariance=1.0):
        """
        Initialize the Kalman Filter.
        
        Args:
        - dt: Time step.
        - state_variance: Variance of the state (process noise).
        - measurement_variance: Variance of the measurement (measurement noise).
        """
        # State vector: [x, y, vx, vy]
        self.x = np.array([*init_pose, 0, 0])
        
        # State transition matrix (constant velocity model)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        # Observation matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        # Process noise covariance
        self.Q = state_variance * np.eye(4)
        
        # Measurement noise covariance
        self.R = measurement_variance * np.eye(2)
        
        # Error covariance
        self.P = initial_covariance * np.eye(4)

    def predict(self):
        """Predict the state and error covariance."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z):
        """Update the state with the latest measurement."""
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P

    def get_state(self):
        """Return the current state estimate."""
        return self.x[:2]

def kalman_filter_2d(predictions, dt, state_variance=1.0, measurement_variance=10.0):
    """
    Apply a Kalman Filter to 2D predictions using constant velocity model.
    
    Args:
    - predictions (list of tuples or list of np.array): Predicted end-effector positions.
    - dt (float): Time step between predictions.
    - state_variance (float): Variance of the state (process noise).
    - measurement_variance (float): Variance of the measurement (measurement noise).
    
    Returns:
    - filtered_predictions (list of np.array): Smoothed and filtered end-effector positions.
    """
    init_pose = predictions[0]
    kf = KalmanFilter(init_pose, dt, state_variance, measurement_variance)
    filtered_predictions = []
    threshold = 80  # Maximum allowed distance from the moving average before a point is considered an outlier
    for i, z in enumerate(predictions):
        distance = np.linalg.norm(z - kf.get_state())
        if distance > threshold:
            # print(f"Outlier detected at index {i}: {z} (distance {distance:.2f})")
            z = kf.get_state()  # Replace outlier with the current state estimate
            
        kf.predict()
        kf.update(np.array(z))
        filtered_predictions.append(kf.get_state())
    
    return filtered_predictions


# Create resnet
class Resnet_tool_tip_labelling_large(torch.nn.Module):
  """
  output size is 480 x 640
  """
  def __init__(self, orig_resnet):
    super().__init__()
    self.orig_resnet = orig_resnet
    
    self.dec4 = nn.ConvTranspose2d(512, 64, 2, 2)
    self.conv4_1 = nn.Conv2d(320, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_4_1 = torch.nn.BatchNorm2d(64)
    self.conv4_2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_4_2 = torch.nn.BatchNorm2d(64)
    
    
    # decoder 
    self.dec1 = nn.ConvTranspose2d(64, 64, kernel_size=(4, 4), stride=(4,4))
    # dimension preserving convolution
    self.conv1_1 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_1_1 = torch.nn.BatchNorm2d(64)
    self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_1_2 = torch.nn.BatchNorm2d(64)
    
    # another decoder layer
    self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size = (2,2), stride=(2,2))
    self.conv2_1 = nn.Conv2d(96, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_2_1 = torch.nn.BatchNorm2d(32)
    self.conv2_2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_2_2 = torch.nn.BatchNorm2d(32) 
    
    # second decoder layer
    self.dec3 = nn.ConvTranspose2d(32, 16, 2, 2)
    self.conv3_1 = nn.Conv2d(16, 16, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_3_1 = torch.nn.BatchNorm2d(16)
    self.conv3_2 = nn.Conv2d(16, 2, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    self.batchnorm_3_2 = torch.nn.BatchNorm2d(2)
    self.conv3_3 = nn.Conv2d(2, 1, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False)
    
  def forward(self, x):
    # encoder
    x1 = self.orig_resnet.conv1(x) # 64 x 240 x 320
    # print("x1", x1.shape)
    x2 = self.orig_resnet.bn1(x1) 
    # print("x2", x2.shape)
    x3 = self.orig_resnet.relu(x2)
    # print("x3", x3.shape)
    x4 = self.orig_resnet.maxpool(x3) # 64 x 120 x 160
    # print("x4", x4.shape)
    x5 = self.orig_resnet.layer1(x4) # 64 x 120 x 160
    # print("x5", x5.shape)
    x6 = self.orig_resnet.layer2(x5) # 128 x 60 x 80
    # print("x6", x6.shape)
    x7 = self.orig_resnet.layer3(x6) # 256 x 30 x 40
    # print("x7", x7.shape)
    x8 = self.orig_resnet.layer4(x7) # 512 x 15 x 20
    
    x13 = self.dec4(x8) # 64 x 30 x 40
    x13 = torch.cat((x13, x7), dim=1) # 320
    x13 = self.batchnorm_4_1( self.conv4_1(x13) ) # 64
    x13 = F.relu(x13)
    x13 = self.batchnorm_4_2(self.conv4_2(x13))
    x13 = F.relu(x13) # 64
    
    # decoder
    x13 = self.dec1(x13) # 64 x 120 x 160
    x13 = torch.cat((x13, x5), dim=1) # 128
    x13 = self.batchnorm_1_1( self.conv1_1(x13) ) # 32
    x13 = F.relu(x13)
    x13 = self.batchnorm_1_2(self.conv1_2(x13))
    x13 = F.relu(x13) # 32

    x14 = self.dec2(x13) # 16 x 240 x 320
    x14 = torch.cat((x14, x3), dim=1) # 80
    x14 = self.batchnorm_2_1( self.conv2_1(x14) ) # 16
    x14 = F.relu(x14)
    x14 = self.batchnorm_2_2(self.conv2_2(x14))
    x14 = F.relu(x14)
    
    x15 = self.dec3(x14) # 8 x 480 x 640
    x15 = self.batchnorm_3_1(self.conv3_1(x15)) # 8
    x15 = F.relu(x15)
    x15 = self.batchnorm_3_2( self.conv3_2(x15) )
    x15 = F.relu(x15)
    x15 = self.conv3_3(x15)
#     print(x14.shape)
    return x15


def process_imgs(base_dir, model, side, batch_size=8):
    if side == "left":
        obj_id_side = [0]
        csv_column = ['ee_PSM2_x', 'ee_PSM2_y']
    elif side == "right":
        obj_id_side = [1, 2]
        csv_column = ['ee_PSM1_x', 'ee_PSM1_y']

    cropped_img_path = os.path.join(base_dir, "cropped_imgs")
    images = [item for item in os.listdir(cropped_img_path) if os.path.isfile(os.path.join(cropped_img_path, item))]
    images = natsorted(images)

    info = {}
    model_outputs = []

    batch_imgs = []
    batch_info = []
    print(f"Processing {len(images)} images...")
    for img in images:
        # Construct image filename
        img_path = os.path.join(cropped_img_path, img)
        parts = img.split('_')
        if not parts[-1].endswith(".jpg"):
            continue

        # Parse the filename
        idx = 1
        frame_number = int(parts[idx])
        obj_id = int(parts[idx+2])
        x = int(parts[idx+4])
        y = int(parts[idx+6])
        w = int(parts[idx+8][:-4])

        if img_path is not None and obj_id in obj_id_side:
            image = io.imread(img_path)
            image = cv2.resize(image, (640, 480))

            # Convert to tensor and prepare batch
            image = TF.to_tensor(image)  # C x H x W
            batch_imgs.append(image)
            batch_info.append([frame_number, obj_id, x, y, w])

            # If batch size is reached, process the batch
            if len(batch_imgs) == batch_size:
                batch_tensor = torch.stack(batch_imgs).float().to(device)  # N x C x H x W

                # Perform inference
                outputs = model(batch_tensor)
                outputs_flatten = outputs.view(outputs.size(0), outputs.size(1), -1).cpu().detach().numpy()
                output_task1 = outputs_flatten[:, 0, :]  # Outputs for the first task [B, D]

                for i in range(len(batch_imgs)):
                    a1 = output_task1[i, :]  # Get output for the i-th image
                    a1_argmax = np.unravel_index(a1.argmax(), (480, 640))
                    ee = np.array((a1_argmax[1], a1_argmax[0]))

                    model_outputs.append([ee[0], ee[1]])
                    # print(batch_info[i][0])
                    info[batch_info[i][0]] = batch_info[i][1:]

                # Clear batch lists
                batch_imgs = []
                batch_info = []

    # Process any remaining images in the last batch
    if len(batch_imgs) > 0:
        batch_tensor = torch.stack(batch_imgs).float().to(device)  # N x C x H x W

        # Perform inference
        outputs = model(batch_tensor)
        outputs_flatten = outputs.view(outputs.size(0), outputs.size(1), -1).cpu().detach().numpy()
        output_task1 = outputs_flatten[:, 0, :]  # Outputs for the first task [B, D]

        for i in range(len(batch_imgs)):
            a1 = output_task1[i, :]  # Get output for the i-th image
            a1_argmax = np.unravel_index(a1.argmax(), (480, 640))
            ee = np.array((a1_argmax[1], a1_argmax[0]))

            model_outputs.append([ee[0], ee[1]])
            info[batch_info[i][0]] = batch_info[i][1:]

    print(f"Finished processing {len(model_outputs)} images.")
    return info, model_outputs


def make_video(base_dir, side):
    if side == "left":
        obj_id_side = [0]

    elif side == "right":
        obj_id_side = [1, 2]
    print(base_dir)
    labelled_img_path = os.path.join(base_dir, "labelled_img_"+side)
    output_video_path = os.path.join(base_dir, 'output_video_'+side+'.mp4')
    frames = []
    fps = 30
    # Ensure the labelled_img directory exists
    os.makedirs(labelled_img_path, exist_ok=True)
    images = [item for item in os.listdir(labelled_img_path) if os.path.isfile(os.path.join(labelled_img_path, item))]
    images = natsorted(images)
    testing = False
    for img in images:
        plot_path = os.path.join(labelled_img_path, img)
        parts = img.split('_')
#         print(parts)
        if not parts[-1].endswith(".jpg"):
            continue

        # parse the filename
        idx = 1
        frame_number = int(parts[idx])
        obj_id = int(parts[idx+2])
        x = int(parts[idx+4])
        y = int(parts[idx+6])
        w = int(parts[idx+8][:-4])
        # check if the object id is in the list

        if obj_id in obj_id_side:
            frame = cv2.imread(plot_path)
            frames.append(frame)
            # print(f"Frame {plot_path} added.")

        else:
            continue

    if len(frames) > 0:
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for frame in frames:
            video.write(frame)

        video.release()
        print(f"Video saved at {output_video_path}")
    else:
        print("No frames were captured.")

def label_orig_imgs(base_dir, info_l, info_r, ee_l, ee_r, output_vid=False, output_img=False):
    original_img_path = os.path.join(base_dir, 'left_img_dir')
    kalman_img_path = os.path.join(base_dir, 'kalman_img_path')
    output_video_path = os.path.join(base_dir, 'labelled_imgs_kalman.mp4')
    fps = 30
    images = [item for item in os.listdir(original_img_path) if os.path.isfile(os.path.join(original_img_path, item))]

    images = natsorted(images)
    frames = []

    predicted_ee_l = []
    predicted_ee_r = []

    # Initialize last known points
    last_known_ee_l = None
    last_known_ee_r = None

    for i, img in enumerate(images):
        # Construct image filename
        img_path = os.path.join(original_img_path, img)

        if not img_path.endswith(".jpg"):
            continue
        
        # Get object IDs and bounding box coordinates
        obj_id_l, x_l, y_l, w_l = info_l.get(i, [None, None, None, None])
        obj_id_r, x_r, y_r, w_r = info_r.get(i, [None, None, None, None])

        # Load the image
        image = cv2.imread(img_path)
        height, width, layers = image.shape

        # Process left EE point
        if obj_id_l is not None and i < len(ee_l):
            ee_l_point = ee_l[i]
            ee_l_point[0] = (ee_l_point[0] * w_l / 640) + x_l
            ee_l_point[1] = (ee_l_point[1] * w_l / 480) + y_l
            last_known_ee_l = ee_l_point
        else:
            if last_known_ee_l is not None:
                ee_l_point = last_known_ee_l
            else:
                print(f"No previous EE left point to use for frame {i}")
                ee_l_point = [0, 0]  # Or some default value
        predicted_ee_l.append(ee_l_point)

        # Process right EE point
        if obj_id_r is not None and i < len(ee_r):
            ee_r_point = ee_r[i]
            ee_r_point[0] = (ee_r_point[0] * w_r / 640) + x_r
            ee_r_point[1] = (ee_r_point[1] * w_r / 480) + y_r
            last_known_ee_r = ee_r_point
        else:
            if last_known_ee_r is not None:
                ee_r_point = last_known_ee_r
            else:
                print(f"No previous EE right point to use for frame {i}")
                ee_r_point = [0, 0]  # Or some default value
        predicted_ee_r.append(ee_r_point)

    # Apply Kalman filter
    predicted_ee_l = kalman_filter_2d(predicted_ee_l, dt=1/30)
    predicted_ee_r = kalman_filter_2d(predicted_ee_r, dt=1/30)

    if output_img:
        os.makedirs(kalman_img_path, exist_ok=True)
        for i, img in enumerate(images):
            img_path = os.path.join(original_img_path, img)
            image = cv2.imread(img_path)

            cv2.circle(image, (int(predicted_ee_l[i][0]), int(predicted_ee_l[i][1])), 5, (0, 0, 255), -1)
            cv2.circle(image, (int(predicted_ee_r[i][0]), int(predicted_ee_r[i][1])), 5, (0, 0, 255), -1)
            # Save the image
            cv2.imwrite(os.path.join(kalman_img_path, img), image)
            frames.append(image)

    if output_vid:
        if len(frames) > 0:
            height, width, layers = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            for frame in frames:
                video.write(frame)

            video.release()
            print(f"Video saved at {output_video_path}")
        else:
            print("No frames were captured.")

    return predicted_ee_l, predicted_ee_r

def draw_square_on_image(image, position, size=30, color=(0, 255, 0), thickness=5):
    top_left = (position[0] - size // 2, position[1] - size // 2)
    bottom_right = (position[0] + size // 2, position[1] + size // 2)
    cv2.rectangle(image, top_left, bottom_right, color, thickness)

def draw_traj_on_img(base_dir, draw_depth_traj=False, draw_jaw_info=False, output_vid=False):
    ee_l_header = ['ee_PSM2_x', 'ee_PSM2_y']
    ee_r_header = ['ee_PSM1_x', 'ee_PSM1_y']
    original_img_path = os.path.join(base_dir, 'left_img_dir')
    labelled_img_path = os.path.join(base_dir, 'labelled_img_depth_relative')
    output_video_path = os.path.join(base_dir, 'sketch_depth_relative.mp4')
    fps = 30
    # output_video_path = os.path.join(base_dir, 'labelled_imgs.mp4')
    os.makedirs(labelled_img_path, exist_ok=True)
    images = [item for item in os.listdir(original_img_path) if os.path.isfile(os.path.join(original_img_path, item))]
    images = natsorted(images)
    csv_path = os.path.join(base_dir, "ee_labels.csv")
    csv = pd.read_csv(csv_path)

    ## read ee csv if draw depth trajectory
    if draw_depth_traj:
        ee_csv_path = os.path.join(base_dir, "ee_csv.csv")
        ee_csv = pd.read_csv(ee_csv_path)
        header_name_qpos_psm1 = ["psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
                                "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
                                "psm1_jaw"]
        
        header_name_qpos_psm2 = ["psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
                                "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
                                "psm2_jaw"]
        
        ## find min and max depth
        ee_l_qpos = ee_csv[header_name_qpos_psm2].to_numpy()
        ee_r_qpos = ee_csv[header_name_qpos_psm1].to_numpy()

        ## from constants calculated from whole the dataset (delta z)
        min_depth_l = -0.022559415870363093
        max_depth_l =  0.017716835525270896
        min_depth_r = -0.022949239522599592
        max_depth_r =  0.02853797792824251

    if draw_jaw_info:
        ee_csv_path = os.path.join(base_dir, "ee_csv.csv")
        ee_csv = pd.read_csv(ee_csv_path)

    frames = []

    for i, img in enumerate(images):
        jaw_close_l_drawn = False
        jaw_close_r_drawn = False
        jaw_open_l_drawn = False
        jaw_open_r_drawn = False


        # Construct image filename
        img_path = os.path.join(original_img_path, img)

        if not img_path.endswith(".jpg"):
            continue

        image = cv2.imread(img_path)
        ee_l_points = csv[ee_l_header].iloc[i:i+100].to_numpy() # note 400 added here
        ee_r_points = csv[ee_r_header].iloc[i:i+100].to_numpy() # note 400 added here

        if draw_depth_traj:
            ee_l_qpos = ee_csv[header_name_qpos_psm2].iloc[i:i+100].to_numpy()
            ee_r_qpos = ee_csv[header_name_qpos_psm1].iloc[i:i+100].to_numpy()
            
            ## z is the depth, calculate delta depth wrt to the first frame
            ee_l_qpos[:, 2] = ee_l_qpos[:, 2] - ee_l_qpos[0, 2]
            ee_r_qpos[:, 2] = ee_r_qpos[:, 2] - ee_r_qpos[0, 2]

            ## normalize the depth
            norm_depth_l = ((ee_l_qpos[:, 2] - min_depth_l) * 255) / (max_depth_l - min_depth_l)
            norm_depth_r = ((ee_r_qpos[:, 2] - min_depth_r) * 255) / (max_depth_r - min_depth_r)

        if draw_jaw_info:
            ee_l_jaw = ee_csv[header_name_qpos_psm2].iloc[i:i+100, -1].to_numpy()
            ee_r_jaw = ee_csv[header_name_qpos_psm1].iloc[i:i+100, -1].to_numpy()

            if ee_l_jaw[0] < -0.15:
                jaw_already_closed_l = True
            else:
                jaw_already_closed_l = False

            if ee_r_jaw[0] < -0.15:
                jaw_already_closed_r = True
            else:
                jaw_already_closed_r = False


            ## find jaw closing instant if jaw is not closed (jaw closing value < -0.1)

            jaw_close_l = np.where(ee_l_jaw < -0.15)
            jaw_close_r = np.where(ee_r_jaw < -0.15)

            jaw_open_l = np.where(ee_l_jaw > -0.15)
            jaw_open_r = np.where(ee_r_jaw > -0.15)
            ## return the first jaw closing instant

            jaw_close_l = jaw_close_l[0]
            jaw_close_r = jaw_close_r[0]
            jaw_closing_index_l = jaw_close_l[0] if len(jaw_close_l) > 0 else None
            jaw_closing_index_r = jaw_close_r[0] if len(jaw_close_r) > 0 else None

            jaw_open_l = jaw_open_l[0]
            jaw_open_r = jaw_open_r[0]
            jaw_opening_index_l = jaw_open_l[0] if len(jaw_open_l) > 0 else None
            jaw_opening_index_r = jaw_open_r[0] if len(jaw_open_r) > 0 else None


        for n in range(len(ee_l_points)):
            ee_l_point = ee_l_points[n]
            ee_r_point = ee_r_points[n]
            ## use the normalized depth to color the points (0 -> red, 255 -> blue)
            if draw_depth_traj:
                color = (0, 0, int(norm_depth_l[n]))
                cv2.circle(image, (int(ee_l_point[0]), int(ee_l_point[1])), 5, color, -1)
                color = (0, 0, int(norm_depth_r[n]))
                cv2.circle(image, (int(ee_r_point[0]), int(ee_r_point[1])), 5, color, -1)
            else:
                cv2.circle(image, (int(ee_l_point[0]), int(ee_l_point[1])), 5, (0, 0, 255), -1)
                cv2.circle(image, (int(ee_r_point[0]), int(ee_r_point[1])), 5, (0, 0, 255), -1)

            if draw_jaw_info:
                if jaw_closing_index_l is not None and not jaw_already_closed_l:
                    if n == jaw_closing_index_l and not jaw_close_l_drawn:
                        pos = (int(ee_l_point[0]), int(ee_l_point[1]))

                        # Draw square on the image
                        draw_square_on_image(image, pos)
                        # cv2.circle(image, (int(ee_l_point[0]), int(ee_l_point[1])), 15, (0, 255, 0), -1)
                        jaw_close_l_drawn = True

                if jaw_closing_index_r is not None and not jaw_already_closed_r:
                    if n == jaw_closing_index_r and not jaw_close_r_drawn:
                        pos = (int(ee_r_point[0]), int(ee_r_point[1]))
                        draw_square_on_image(image, pos)
                        # cv2.circle(image, (int(ee_r_point[0]), int(ee_r_point[1])), 15, (0, 255, 0), -1)

                        jaw_close_r_drawn = True
                if jaw_opening_index_l is not None and jaw_already_closed_l:
                    if n == jaw_opening_index_l and not jaw_open_l_drawn:
                        pos = (int(ee_l_point[0]), int(ee_l_point[1]))
                        draw_square_on_image(image, pos, color=(255, 0, 0))
                        # cv2.circle(image, (int(ee_l_point[0]), int(ee_l_point[1])), 15, (0, 255, 0), -1)
                        jaw_open_l_drawn = True

                if jaw_opening_index_r is not None and jaw_already_closed_r:
                    if n == jaw_opening_index_r and not jaw_open_r_drawn:
                        pos = (int(ee_r_point[0]), int(ee_r_point[1]))
                        draw_square_on_image(image, pos, color=(255, 0, 0))
                        # cv2.circle(image, (int(ee_r_point[0]), int(ee_r_point[1])), 15, (0, 255, 0), -1)

                        jaw_open_r_drawn = True

        cv2.imwrite(os.path.join(labelled_img_path, img), image)
        frames.append(image)
        
    if output_vid:
        if len(frames) > 0:
            height, width, layers = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            for frame in frames:
                video.write(frame)

            video.release()
            print(f"Video saved at {output_video_path}")
        else:
            print("No frames were captured.")



## ----------------- Main ----------------- ##
# setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using", device)
# DELETE! Temporary for error trace
# device = "cpu"

resnet = models.resnet18(pretrained=False)

# choose resnet of your choice
resnet_enc_dec_concat_more = Resnet_tool_tip_labelling_large(resnet) 
model_left = resnet_enc_dec_concat_more.to(device)
model_right = resnet_enc_dec_concat_more.to(device)

epoch_num = 111
# load previous model
# checkpoint = torch.load(f"model_parameters/right_epoch_{epoch_num}val_checkpoint.pth.tar")


data_dir = os.getenv('PATH_TO_DATASET')
data_dir = os.path.join(data_dir, "base_chole_clipping_cutting")
tissue_ids = [71, 72, 73, 75, 77, 80]
# tissue_ids = [4]
# tissue_ids = [49, 50, 53, 54]
# tissue_ids = [1, 4, 5, 6, 8, 12, 13, 14, 18, 19, 22, 23, 30, 32, 35, 39, 40, 41, 47, 49, 50, 53, 54]
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
        for item in os.listdir(phase):
            img_dir = os.path.join(root, phase, item)
            # img_dir = data_dir + "/tissue_5/4_clipping_second_clip_left_tube/20240710-184558-875384"
            print(img_dir)

            ## if the labelled images are already present and is not empty, skip the processing
            if os.path.exists(os.path.join(img_dir, 'labelled_img_depth_relative')) and len(os.listdir(os.path.join(img_dir, 'labelled_img_depth_relative'))) == len(os.listdir(os.path.join(img_dir, 'left_img_dir'))):
                print("labelled images already present")
                continue

            ## load the left model and inference on the left images
            checkpoint_left = torch.load(f"model_parameters/epoch_{epoch_num}val_checkpoint.pth.tar", map_location='cuda:0')
            model_left.load_state_dict(checkpoint_left['state_dict'])
            model_left = model_left.to(device).eval()
            info_l, ee_l = process_imgs(img_dir, model_left, side="left", batch_size=32)
            print("finished left", len(info_l))

            ## load the right model and inference on the right images
            checkpoint_right = torch.load(f"model_parameters/right_epoch_{epoch_num}val_checkpoint.pth.tar", map_location='cuda:0')
            model_right.load_state_dict(checkpoint_right['state_dict'])
            model_right = model_right.to(device).eval()
            info_r, ee_r = process_imgs(img_dir, model_right, side="right", batch_size=32)
            print("finished right", len(info_r))
            
            ## label the original images
            pred_ee_l, pred_ee_r = label_orig_imgs(img_dir, info_l, info_r, ee_l, ee_r)

            ## save the predicted ee labels
            output_df = pd.DataFrame(pred_ee_l, columns=['ee_PSM2_x', 'ee_PSM2_y'])
            output_df['ee_PSM1_x'] = [x[0] for x in pred_ee_r]
            output_df['ee_PSM1_y'] = [x[1] for x in pred_ee_r]
            output_df.to_csv(os.path.join(img_dir, 'ee_labels.csv'), index=False)
            
            draw_traj_on_img(img_dir, draw_depth_traj=True, draw_jaw_info=True, output_vid=False)

            # input("Press Enter to continue...")

        phase_time_taken = time.time() - phase_start_t
        print(f"Time taken for phase {dir}: {phase_time_taken} seconds")

    tissue_time_taken = time.time() - tissue_start_t
    print(f"Time taken for tissue {tissue_id}: {tissue_time_taken} seconds")
                
                
            ## make videos
            # make_video(img_dir, side="left")
            # make_video(img_dir, side="right")
            # print("processed", img_dir)
            # input("Press Enter to continue...")


