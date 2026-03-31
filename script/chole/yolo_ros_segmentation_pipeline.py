#!/usr/bin/env python
import os
import rospy
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
import torch
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np

# Global variable to store the latest image from the left camera
image_left = None

# ROS Image -> OpenCV image converter
ros_cv2_bridge = CvBridge()

# Callback function for the left camera
def left_camera_callback(data):
    global image_left
    image_left = ros_cv2_bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

# Initialize ROS node
rospy.init_node('yolo_segmentation_node')

# Subscriber to the left camera
left_img_dir_sub = rospy.Subscriber("/jhu_daVinci/left/image_raw", Image, left_camera_callback, queue_size=1)

# Publisher for the segmentation masks
segmentation_pub = rospy.Publisher('/yolo_segmentation_masks', Image, queue_size=1)
contour_image_pub = rospy.Publisher('/yolo_contour_image', Image, queue_size=1)



def process_and_publish_segmentation(fps=30, gpu=0):
    cmap = plt.get_cmap("tab10")
    global image_left

    rate = rospy.Rate(fps)  # Process at the specified FPS

    # Load the YOLO model
    model_ckpts_folder = os.getenv("YOUR_CKPT_PATH")
    yolo_model_path = os.path.join(model_ckpts_folder, "yolo", "best.pt")
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model = YOLO(yolo_model_path).to(device)
    print("YOLO model loaded successfully")

    while not rospy.is_shutdown():
        if image_left is not None:
            # ---------- Preprocess the image for the YOLO model ----------
            
            # Resize the image to the YOLO model input size
            yolo_resize_dim = (640, 640) # 1080,1920) # 
            orig_left_image = image_left.copy()
            # orig_left_image = cv2.cvtColor(orig_left_image, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(orig_left_image, yolo_resize_dim, interpolation=cv2.INTER_LINEAR)

            # Convert the image to a tensor
            frame_tensor = torch.from_numpy(resized_frame).to(torch.float32) / 255.0 
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

            height, width = orig_left_image.shape[:2]
            img_size = (height, width)

            ## rescale the coordinates to the original image size
            scaling_factor_x = width / yolo_resize_dim[1]
            scaling_factor_y = height / yolo_resize_dim[0]

            # ------------- Run the YOLO model -------------
                
            # Run the YOLO model
            results = model(frame_tensor, conf=0.3)
            # --------- Process the segmentation masks and contour image ---------

            if results[0].masks is not None:
                # Extract segmentation masks and class IDs
                masks_coords = results[0].masks.xy
                class_ids = results[0].boxes.cpu().numpy().cls

                # Prepare fill mask for blended contour image
                fill_mask = np.zeros_like(orig_left_image, dtype=np.uint8) 

                # Initialize segmentation masks with zeros by default
                masks = {
                    "clips": np.zeros((height, width), dtype=np.uint8),
                    "left_tube": np.zeros((height, width), dtype=np.uint8),
                    "right_tube": np.zeros((height, width), dtype=np.uint8),
                    "flap": np.zeros((height, width), dtype=np.uint8)
                }

                # -------

                # Iterate through detected objects and fill in masks
                for i, coords in enumerate(masks_coords):
                    coords = np.array(coords, dtype=np.int32)
                    class_id = int(class_ids[i])

                    # Rescale the coordinates to the original image size
                    coords[:, 0] = (coords[:, 0] * scaling_factor_x).astype(np.int32)  # X coordinates
                    coords[:, 1] = (coords[:, 1] * scaling_factor_y).astype(np.int32)  # Y coordinates

                    if class_id == 0:
                        masks["clips"] = cv2.fillPoly(masks["clips"], [coords], color=255)
                    elif class_id == 1:
                        masks["left_tube"] = cv2.fillPoly(masks["left_tube"], [coords], color=255)
                    elif class_id == 2:
                        masks["right_tube"] = cv2.fillPoly(masks["right_tube"], [coords], color=255)
                    elif class_id == 3:
                        masks["flap"] = cv2.fillPoly(masks["flap"], [coords], color=255)

                    # Draw the contours on the original image
                    class_color = np.array(cmap(class_id)[:3])  # Extract only RGB values
                    class_color_scaled = tuple((class_color * 255).astype(int).tolist())  # Convert to BGR and ensure it's a tuple
                    cv2.drawContours(orig_left_image, [coords], -1, class_color_scaled, 2)
                    semi_transparent_color = (class_color_scaled[0], class_color_scaled[1], class_color_scaled[2], 100)  # BGR + Alpha

                    # Fill the polygon on the fill mask
                    cv2.fillPoly(fill_mask, [coords], color=semi_transparent_color[:-1])  # Fill without alpha channel

                # Blend the fill mask with the original image
                alpha = 0.5  # Transparency factor (0.0 to 1.0)
                blended_image = cv2.addWeighted(orig_left_image, 1.0, fill_mask, alpha, 0)

                
                # Merge the segmentation masks on one channel
                merged_seg_mask = np.zeros(img_size, dtype=np.uint8)
                for class_id, class_seg_mask_w_factor in enumerate(masks.values(), start=1): 
                    class_seg_mask_w_factor = np.where(class_seg_mask_w_factor / 255 > 0.5, 1, 0).astype(np.uint8)
                    class_seg_mask_w_factor = class_seg_mask_w_factor * class_id
                    # Merge the segmentation masks based on the priority of the objects (lower the higher, except for 0)
                    merged_seg_mask = np.where((merged_seg_mask == 0) & (class_seg_mask_w_factor != 0), class_seg_mask_w_factor, merged_seg_mask)
                    merged_seg_mask = np.where((merged_seg_mask != 0) & (class_seg_mask_w_factor != 0) & (class_seg_mask_w_factor < merged_seg_mask), class_seg_mask_w_factor, merged_seg_mask)            

                # --------- Publish the segmentation masks and contour image ---------

                # Convert the tensor to a ROS message and publish it
                seg_mask_msg = ros_cv2_bridge.cv2_to_imgmsg(merged_seg_mask, encoding='mono8')
                segmentation_pub.publish(seg_mask_msg)
                
                # Convert the contour image to a ROS message and publish it
                contour_image_msg = ros_cv2_bridge.cv2_to_imgmsg(blended_image, encoding="rgb8")
                contour_image_pub.publish(contour_image_msg)

        rate.sleep()


if __name__ == '__main__':
    fps = 30 # Frames per second to apply the YOLO model
    gpu = 0  # GPU index to use for inference
    process_and_publish_segmentation(fps, gpu)
