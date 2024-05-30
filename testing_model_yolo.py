import torch
import pathlib
import cv2
from PIL import Image
from ultralytics import YOLO

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

img_path = 'C:/Users/acer/Documents/coolyeah/skripsi/Revisi/real time code/00d1778c-veteran_in_20230522T113003.mkv_frame_14140.jpg'
weights_path='C:/Users/acer/Documents/coolyeah/skripsi/Revisi/real time code/no_preprocessing_model_skripsi.pt'
# model = torch.hub.load('uji_ablasi/yolov5', 'custom', path=weights_path)
model = torch.hub.load("ultralytics/yolov5", "custom", path = weights_path, force_reload=True)
# results = model(img_path)
image = cv2.imread(img_path)
# Convert the image to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize the image to the model's expected size
image = cv2.resize(image, (640, 640))

# Convert the image to a PIL Image object
image = Image.fromarray(image)

# Predict with the trained model
results = model(image)
# r_img = results.render() # returns a list with the images as np.arra
