import cv2
import json
import torch
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image
import logging
import torch.nn as nn
from classification import EmbeddingClassifier, YOLOInference, SegmentationInference
import os


detector = YOLOInference("models/bb_model.ts")
classifier = EmbeddingClassifier("models/class_model.ts", "database.pt", detector, device="cpu")
segmentator = SegmentationInference("models/segmentation_21_08_2023.ts")

image_path = "images/marlin.jpeg"
out_image_path = f"out_{image_path}"
print(classifier.classify_image(image_path, out_image_path, ))
segmentator.segmentate_img(out_image_path, out_image_path, (255, 0, 0, 255))


# folder_path = "images/own"
# out_folder_path = "out_images/own"
# for image_name in os.listdir(folder_path):
#     if image_name[0] != ".":
#         image_path = os.path.join(folder_path, image_name)
#         out_path = os.path.join(out_folder_path, image_name)
#         classifier.classify_image(image_path, out_path)
#         segmentator.segmentate_img(out_path, out_path, (255, 0, 0, 255))

    