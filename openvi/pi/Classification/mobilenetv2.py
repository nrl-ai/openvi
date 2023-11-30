import os, sys
import numpy as np
from numpy import exp
import cv2
from scipy.special import softmax
import copy

base_path = os.path.dirname(__file__)


def run_mobilenetv2(image):
    onnx_path = os.path.join(
        base_path, "out_snapshot/r18_classify/mbv2_classify_best_model.onnx"
    )
    opencv_net = cv2.dnn.readNetFromONNX(onnx_path)

    input_img = image.astype(np.float32)
    input_img = cv2.resize(input_img, (224, 224))
    # img_res = copy.deepcopy(input_img)

    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    scale = 1 / 255.0
    std = [0.229, 0.224, 0.225]

    input_blob = cv2.dnn.blobFromImage(
        image=input_img,
        scalefactor=scale,
        size=(224, 224),
        mean=mean,
        swapRB=True,
        crop=True,
    )

    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    opencv_net.setInput(input_blob)
    out = opencv_net.forward()
    class_id = np.argmax(out)
    pred = softmax(out[0], axis=0)
    index = pred.argmax(axis=0)
    print(pred, index)
    label = f"{pred} - {index}"
    cv2.putText(
        input_img,
        label,
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        thickness=2,
    )

    return input_img


if __name__ == "__main__":
    image = cv2.imread("test/screw/2023-03-06_14-49-49_10.png")
    run_mobilenetv2(image)
