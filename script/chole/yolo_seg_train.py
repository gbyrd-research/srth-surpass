from ultralytics import YOLO
import torch

torch.cuda.set_device(0)
# Load a model
# model = YOLO("yolov8s-seg.yaml")  # build a new model from scratch
# model = YOLO("yolov8s-seg.pt")  # load a pretrained model (recommended for training)
model = YOLO("./model_parameters/seg_best.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="seg.yaml", epochs=80)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model(data="./model_parameters/seg.yaml")  # predict on an image

print("Mean Average Precision for boxes:", metrics.box.map)
print("Mean Average Precision for masks:", metrics.seg.map)
path = model.export(format="onnx")  # export the model to ONNX format