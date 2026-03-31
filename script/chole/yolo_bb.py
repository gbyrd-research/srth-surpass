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
# tissue_ids = [71, 72, 73, 75, 77, 80]
tissue_ids = [71]
# tissue_ids = [4, 5, 6, 8, 12, 13, 14, 18, 19, 22, 23, 30, 32, 35, 39, 40, 41, 47, 49, 50, 53, 54]
crop_coords = []

for tissue_id in tissue_ids:
    root = os.path.join(data_dir, f"tissue_{tissue_id}")
    dirlist = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
    dirlist = natsorted(dirlist)
    for dir in dirlist:
        if dir.startswith("1"):

            phase = os.path.join(root, dir)
            for item in os.listdir(phase):
                img_dir = os.path.join(root, phase, item, "left_img_dir")
                print(img_dir)
                results = model.predict(img_dir)
                output_dir = os.path.join(img_dir, "inputs")
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)  # Remove directory
                os.makedirs(output_dir, exist_ok=True)

                # Visualize the results

                for i, r in enumerate(results):

                    # Plot results image
                    im_bgr = r.plot(boxes=True, conf=False)  # BGR-order numpy array
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

                        if classes[id] == 0:
                            crop_coords = [int(x), int(y), int(x + w), int(y + h)]

                            ## write to json file as the following format
                            """[
                                {
                                    "label": "dvrk-tool",
                                    "bbox_modal": [
                                        550,
                                        210,
                                        750,
                                        330
                                    ]
                                }
                            ]"""

                            json_file = os.path.join(output_dir, f"object_data_{i}.json")
                            with open(json_file, 'w') as f:
                                f.write("[\n")
                                f.write("\t{\n")
                                f.write(f'\t\t"label": "dvrk-tool",\n')
                                f.write("\t\t\"bbox_modal\": [\n")
                                f.write(f'\t\t\t{crop_coords[0]},\n')
                                f.write(f'\t\t\t{crop_coords[1]},\n')
                                f.write(f'\t\t\t{crop_coords[2]},\n')
                                f.write(f'\t\t\t{crop_coords[3]}\n')
                                f.write("\t\t]\n")
                                f.write("\t}\n")
                                f.write("]\n")

                            psm2 = True


                        elif classes[id] == 1 or classes[id] == 2:
                            # crop_coords = [int(x)-20, int(y + h/2 - s/2 + 20), int(x)+s-40, int(y + h/2 +s/2)]
                            if h < 270:
                                crop_coords = [int(x + w/4), int(y + h/4), int(x+ 3*w/4), int(y + 3*h/4)]
                            else:
                                crop_coords = [int(x + w/4), int(y_center), int(x+ 2*w/3), int(y + h)]
                            ## write to json file as the following format
                            """[
                                {
                                    "label": "dvrk-tool",
                                    "bbox_modal": [
                                        550,
                                        210,
                                        750,
                                        330
                                    ]
                                }
                            ]"""
                            # json_file = os.path.join(output_dir, f"object_data_{i}.json")
                            # with open(json_file, 'w') as f:
                            #     f.write("[\n")
                            #     f.write("\t{\n")
                            #     f.write(f'\t\t"label": "dvrk-tool",\n')
                            #     f.write("\t\t\"bbox_modal\": [\n")
                            #     f.write(f'\t\t\t{crop_coords[0]},\n')
                            #     f.write(f'\t\t\t{crop_coords[1]},\n')
                            #     f.write(f'\t\t\t{crop_coords[2]},\n')
                            #     f.write(f'\t\t\t{crop_coords[3]}\n')
                            #     f.write("\t\t]\n")
                            #     f.write("\t}\n")
                            #     f.write("]\n")
                            


                            psm1 = True
                        else:
                            continue



                input("press any key to continue")

