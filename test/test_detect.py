from ultralytics import YOLO
import os
import cv2
from matplotlib import pyplot as plt

model = YOLO("model/detection_model.pt")

folder_name = "test/actual_test"

for filename in os.listdir(folder_name):
    result = model.predict(f"{folder_name}/{filename}")

    annotated_img = result[0].plot()

    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    plt.show()