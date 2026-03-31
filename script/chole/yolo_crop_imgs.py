from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import cv2
from natsort import natsorted
import shutil
import torch

torch.cuda.set_device(0)
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("./model_parameters/bb_best.pt")  # load a trained model (recommended for training)


# Use the model
data_dir = os.getenv('PATH_TO_DATASET')
data_dir = os.path.join(data_dir, "base_chole_clipping_cutting")
tissue_ids = [71, 72, 73, 75, 77, 80]
# tissue_ids = [4, 5, 6, 8, 12, 13, 14, 18, 19, 22, 23, 30, 32, 35, 39, 40, 41, 47, 49, 50, 53, 54]
crop_coords = []

for tissue_id in tissue_ids:
    root = os.path.join(data_dir, f"tissue_{tissue_id}")
    dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
    dirlist = natsorted(dirlist)
    for dir in dirlist:
        phase = os.path.join(root, dir)
        for item in os.listdir(phase):
            img_dir = os.path.join(root, phase, item, "left_img_dir")
            print(img_dir)
            results = model.predict(img_dir)
            output_dir = os.path.join(root, phase, item, "cropped_imgs")
            # shutil.rmtree(output_dir)  # Remove directory
            os.makedirs(output_dir, exist_ok=True)

            # Visualize the results

            for i, r in enumerate(results):

                # Plot results image
                im_bgr = r.plot(boxes=False, conf=False)  # BGR-order numpy array
                im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
                # print(r.boxes.cpu().numpy().xywh)
                bboxs = r.boxes.cpu().numpy().xywh
                classes = r.boxes.cpu().numpy().cls
                psm1 = False
                psm2 = False
                for id, bbox in enumerate(bboxs):
                    if psm1 and psm2:
                        break
                    x_center, y_center, w, h = bbox
                    x = x_center - w/2
                    y = y_center - h/2
                    s = int(max(w, h)) + 20
                    if classes[id] == 0:
                        crop_coords = [int(x)+20, int(y + h/2- s/2 + 20), int(x)+s, int(y + h/2 + s/2)]
                        psm2 = True
                    elif classes[id] == 1 or classes[id] == 2:
                        crop_coords = [int(x)-20, int(y + h/2 - s/2 + 20), int(x)+s-40, int(y + h/2 +s/2)]
                        psm1 = True
                    else:
                        continue
                    cropped_image = im_rgb.crop((crop_coords[0], crop_coords[1], crop_coords[2], crop_coords[3]))

                    # Resize the cropped image to 224x224
                    resized_image = cropped_image.resize((224, 224))
                    # Convert the image to an array and display it
                    resized_image_np = np.array(resized_image)
                    resized_image_np = cv2.cvtColor(resized_image_np, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(output_dir + "/frame_" + str(i) + "_obj_" +str(int(classes[id])) +
                                "_x_" + str(crop_coords[0]) + "_y_" + str(crop_coords[1]) +  "_w_" + str(crop_coords[2] - crop_coords[0]) +".jpg", resized_image_np)
            
            # print("cropped", item)
