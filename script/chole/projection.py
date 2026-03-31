import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image

# Intrinsic camera matrix from your parameters
original_camera_matrix = np.array([
    [1559.813876107918, 0, 598.821109464468],
    [0, 1556.615680161757, 573.3617661487762],
    [0, 0, 1]
])

# Scale factors for resizing
scale_x = 1300 / 960 
scale_y = 1024 / 540 
image_width = 960
image_height = 540

# Adjusted camera matrix for the resized image
camera_matrix_resized = original_camera_matrix.copy()
camera_matrix_resized[0, 0] *= scale_y  # Scale fx
camera_matrix_resized[1, 1] *= scale_x  # Scale fy
camera_matrix_resized[0, 2] *= scale_y  # Scale cx
camera_matrix_resized[1, 2] *= scale_x  # Scale cy

# Distortion coefficients
distortion_coeffs = np.array([-0.4501281679949737, 0.8450018429832337, 
                              -0.002413339292911583, -0.002976086845065323, 0])

# Load your image
base_dir = "/home/iulian/chole_ws/data/base_chole_clipping_cutting/tissue_18/1_grabbing_gallbladder_recovery/20240719-160046-708534_recovery"
ee_csv_path = os.path.join(base_dir, "ee_csv.csv")
ee_csv = pd.read_csv(ee_csv_path)
header_name_qpos_psm1 = ["psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
                        "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
                        "psm1_jaw"]

header_name_qpos_psm2 = ["psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
                        "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
                        "psm2_jaw"]

# Find min and max depth
ee_l_qpos = ee_csv[header_name_qpos_psm2].to_numpy()
ee_r_qpos = ee_csv[header_name_qpos_psm1].to_numpy()

# 3D position of the end-effector in the camera frame
position_3d = ee_l_qpos[:, :3]

R_z_180 = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0]
])

# Convert the rotation matrix to a rotation vector (rvec)
rvec, _ = cv2.Rodrigues(R_z_180)
tvec = np.zeros((3, 1))

# Project the 3D position to 2D image coordinates
image_points, _ = cv2.projectPoints(position_3d, rvec, tvec, camera_matrix_resized, distortion_coeffs)

# Iterate through the images in the base_dir
image_dir = os.path.join(base_dir, "left_img_dir")
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

# Create a list to store the images for the GIF
gif_images = []

starting_point = [298.73125, 398.9125]
init = image_points[0].flatten()

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    
    points = []
    for num in range(len(image_points)):
        u, v = image_points[num].flatten()
        u = u - init[0] + starting_point[0]
        v = v - init[1] + starting_point[1]
        points.append([u, v])
        cv2.circle(image, (int(u), int(v)), 5, (0, 255-num, 0), -1)  # Green dot at the position

    # Convert the image to RGB (from BGR) and add it to the list
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gif_images.append(Image.fromarray(rgb_image))

# Save the images as a GIF
# gif_images[0].save('trajectory.gif', save_all=True, append_images=gif_images[1:], duration=100, loop=0)
