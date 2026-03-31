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
import sys
import os
path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')

if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
from aloha_pro.aloha_scripts.utils import initialize_model_and_tokenizer, encode_text
from auto_label.auto_label_func import get_auto_label



## ----------------- Main ----------------- ##
# setup model

data_dir = os.getenv('PATH_TO_DATASET')
data_dir = os.path.join(data_dir, "base_chole_clipping_cutting")
tissue_ids = [50]
# tissue_ids = [49, 50, 53, 54]
# tissue_ids = [1, 4, 5, 6, 8, 12, 13, 14, 18, 19, 22, 23, 30, 32, 35, 39, 40, 41, 47, 49, 50, 53, 54]
crop_coords = []
output_vid = True
fps = 10
for tissue_id in tissue_ids:
    ## calculate time taken for each tissue
    tissue_start_t = time.time()
    root = os.path.join(data_dir, f"tissue_{tissue_id}")
    dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
    dirlist = natsorted(dirlist)
    for dir in dirlist:
        phase_start_t = time.time()
        # rand_int = random.randint(1, 17)
        rand_int = 8
        # print(f"rand_int: {rand_int}")

        if dir.startswith(str(rand_int)):
            phase = os.path.join(root, dir)
            frames = []

            for item in os.listdir(phase):
                if item.endswith(".json"):
                    continue
                img_dir = os.path.join(root, phase, item)
                # img_dir = data_dir + "/tissue_5/4_clipping_second_clip_left_tube/20240710-184558-875384"
                print(img_dir)

                # read csv
                csv_file = os.path.join(img_dir, "ee_csv.csv")
                ee_csv = pd.read_csv(csv_file)
                output_dir = os.path.join(data_dir, "kinematic_labelled_img")
                output_video_path = os.path.join(output_dir, f"{dir}_video_40.mp4")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                for i in range(0, len(ee_csv)):
                    start_ts = i
                    auto_label = get_auto_label(ee_csv, start_ts, chunk_size=40)
                    ## cv2 put text on the image
                    image_path_l = os.path.join(img_dir, "labelled_img_depth_relative",
                                    "frame{:06d}".format(start_ts) + "_left.jpg")
                    img = cv2.imread(image_path_l)
                    cv2.putText(img, auto_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # cv2.imwrite(os.path.join(output_dir, "frame{:06d}".format(start_ts) + "_left.jpg"), img)
                    frames.append(img)
                    
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

            # input("Press Enter to continue...")

            phase_time_taken = time.time() - phase_start_t
            print(f"Time taken for phase {dir}: {phase_time_taken} seconds")

        tissue_time_taken = time.time() - tissue_start_t
        print(f"Time taken for tissue {tissue_id}: {tissue_time_taken} seconds")
                    
                    



