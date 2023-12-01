import os, sys

from pip import main
import numpy as np
import cv2
from scipy.special import softmax


class OpenCV_Classify(object):
    def __init__(
        self,
        model_path,
        input_size=(224, 224),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        self.opencv_net = cv2.dnn.readNetFromONNX(model_path)
        self.mean = np.array([0.485, 0.456, 0.406]) * 255.0
        self.scale = 1 / 255.0
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image, top_k=5):
        input_img = image.astype(np.float32)
        input_img = cv2.resize(image, (224, 224))

        input_blob = cv2.dnn.blobFromImage(
            image=input_img,
            scalefactor=self.scale,
            size=(224, 224),
            mean=self.mean,
            swapRB=True,
            crop=True,
        )
        input_blob[0] /= np.asarray(self.std, dtype=np.float32).reshape(
            3, 1, 1
        )
        self.opencv_net.setInput(input_blob)
        out = self.opencv_net.forward()
        pred = softmax(out[0], axis=0)
        class_ids = pred.argmax(axis=0)
        class_scores = pred[class_ids]
        return np.array([class_scores]), np.array([class_ids])
