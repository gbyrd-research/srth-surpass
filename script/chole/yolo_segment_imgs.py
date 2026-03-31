from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import cv2
from natsort import natsorted
import shutil
import torch
import matplotlib.pyplot as plt
import time
torch.cuda.set_device(0)
# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("./best.pt")  # load a trained model (recommended for training)
model = YOLO("/home/imerse/yolo/runs/segment/train8/weights/best.pt")  # load a pretrained model (recommended for training)


def postprocess_mask(mask_image, min_size=500, kernel_size=(5, 5)):
    """
    Post-process the binary mask image by filling small holes and removing small objects.
    
    Args:
        mask_image (np.ndarray): The binary mask image (2D array) where the object is represented by non-zero pixels.
        min_size (int): The minimum size of connected components to keep. Smaller components will be removed.
        kernel_size (tuple): The size of the kernel used for morphological closing to fill small holes.
        
    Returns:
        np.ndarray: The processed binary mask image.
    """
    
    # Fill small holes using morphological closing
    kernel = np.ones(kernel_size, np.uint8)
    mask_filled = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel)

    # Remove small objects using connected component analysis
    num_labels, labels_im = cv2.connectedComponents(mask_filled)
    
    # Create an empty mask to hold the filtered result
    final_mask = np.zeros_like(mask_filled)
    
    # Iterate through the labeled components and keep only large enough components
    for label in range(1, num_labels):  # Skip the background label 0
        component = (labels_im == label)
        if np.sum(component) > min_size:
            final_mask[component] = 255            
    
    return final_mask

# Use the model
data_dir = os.getenv('PATH_TO_DATASET')
data_dir = os.path.join(data_dir, "base_chole_clipping_cutting_amos")
tissue_ids = [62, 68, 55]
# tissue_ids = [4, 5, 6, 8, 12, 13, 14, 18, 19, 22, 23, 30, 32, 35, 39, 40, 41, 47, 49, 50, 53, 54]
crop_coords = []
cmap = plt.get_cmap("tab10") # Color map to use
output_vid = False
fps = 30
for tissue_id in tissue_ids:
    tissue_start_t = time.time()

    root = os.path.join(data_dir, f"tissue_{tissue_id}")
    dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
    dirlist = natsorted(dirlist)
    
    for dir in dirlist:
        phase_start_t = time.time()

        # if dir.startswith("13"):
        phase = os.path.join(root, dir)
        
        for item in os.listdir(phase):
            output_video_path = os.path.join(root, phase, item, "yolo_seg_output.mp4")
            img_dir = os.path.join(root, phase, item, "left_img_dir")
            images = os.listdir(img_dir)
            print(img_dir)

            if os.path.exists(os.path.join(root, phase, item,'contours_on_image')) and len(os.listdir(os.path.join(root, phase, item, 'contours_on_image'))) == len(os.listdir(img_dir)):
                print("labelled images already present")
                continue

            results = model.predict(img_dir, vid_stride=10, half=True)
            seg_output_dir = os.path.join(root, phase, item, "seg_masks")
            contour_output_dir = os.path.join(root, phase, item, "contours_on_image")
            # yolo_output_dir = os.path.join(root, phase, item, "yolo_seg_output")
            if os.path.exists(seg_output_dir):
                shutil.rmtree(seg_output_dir)  # Remove directory
            if os.path.exists(contour_output_dir):
                shutil.rmtree(contour_output_dir)  # Remove directory
            os.makedirs(seg_output_dir, exist_ok=True)
            os.makedirs(contour_output_dir, exist_ok=True)
            # os.makedirs(yolo_output_dir, exist_ok=True)

            # Visualize the results
            frames = []
            for i, r in enumerate(results):

                # Plot result image with segmentation contours
                # im_bgr = r.plot(boxes=True, conf=False, color_mode='class')  # BGR-order numpy array
                # im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
                # cv2.imwrite(yolo_output_dir + "/frame_" + str(i) + ".jpg", im_bgr)

                original_image = r.orig_img
                # Save the segmentation masks
                if r.masks is None:
                    continue
                masks_coords = r.masks.xy
                class_ids = r.boxes.cpu().numpy().cls
                # Create a mask image for filling
                fill_mask = np.zeros_like(original_image, dtype=np.uint8)

                height, width = original_image.shape[:2]

                # Initialize a dictionary to store mask images for each class
                class_masks = {0: np.zeros((height, width), dtype=np.uint8),
                            1: np.zeros((height, width), dtype=np.uint8),
                            2: np.zeros((height, width), dtype=np.uint8),
                            3: np.zeros((height, width), dtype=np.uint8)}

                # Iterate through all detected objects
                for j, coords in enumerate(masks_coords):
                    # Convert coordinates to integer type
                    coords = np.array(coords, dtype=np.int32)

                    # Draw the contour on the corresponding class mask
                    class_id = int(class_ids[j])
                    cv2.fillPoly(class_masks[class_id], [coords], color=255)

                    # Draw the contours on the original image - check if this works
                    class_color = np.array(cmap(class_id)[:3])  # Extract only RGB values
                    class_color_bgr = tuple((class_color * 255).astype(int)[::-1].tolist())  # Convert to BGR and ensure it's a tuple
                    cv2.drawContours(original_image, [coords], -1, class_color_bgr, 2)
                    semi_transparent_color = (class_color_bgr[0], class_color_bgr[1], class_color_bgr[2], 100)  # BGR + Alpha

                    # Fill the polygon on the fill mask
                    cv2.fillPoly(fill_mask, [coords], color=semi_transparent_color[:-1])  # Fill without alpha channel

                # Blend the fill mask with the original image
                # Save the masks for each class
                for class_id, mask_img in class_masks.items():
                    if class_id == 0:
                        obj_name = "clips"
                    elif class_id == 1:
                        obj_name = "left_tube"
                    elif class_id == 2:
                        obj_name = "right_tube"
                    elif class_id == 3:
                        obj_name = "flap"

                    # Save the mask image only if it has non-zero pixels
                    if np.any(mask_img):
                        mask_file_path = os.path.join(seg_output_dir, f"frame{i:06d}_{obj_name}.jpg")
                        cv2.imwrite(mask_file_path, mask_img)


                alpha = 0.5  # Transparency factor (0.0 to 1.0)
                blended_image = cv2.addWeighted(original_image, 1.0, fill_mask, alpha, 0)

                contour_file_name = f"frame{i:06d}_contours.jpg"
                cv2.imwrite(os.path.join(contour_output_dir, contour_file_name), blended_image)
                frames.append(blended_image)

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
                

