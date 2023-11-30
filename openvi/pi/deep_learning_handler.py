import cv2
from pi.Classification.resnet18 import run_resnet18
from pi.Classification.mobilenetv2 import run_mobilenetv2
from pi.ObjectDetection import nanodet_416
from pi.ObjectDetection import nanodet_320


def deep_learning(image, method_name, params):
    if method_name == "Classification":
        model_name = list(params.values())[-1]
        if model_name == "Resnet18":
            res_image = run_resnet18(image)
        elif model_name == "MobileNetV2":
            res_image = run_resnet18(image)
        else:
            res_image = None
            pass
    elif method_name == "ObjectDetection":
        thresh = list(params.values())[-1]
        model_name = list(params.values())[-2]
        if model_name == "NanoDet-Plus-m (416x416)":
            net = nanodet_416.my_nanodet(
                input_shape=416, prob_threshold=0.5, iou_threshold=0.6
            )
            res_image = net.detect(image)

            # winName = 'Deep learning object detection in OpenCV'
            # cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
            # cv2.imshow(winName, res_image)
        elif model_name == "NanoDet-Plus-m (320x320)":
            net = nanodet_320.my_nanodet(
                input_shape=320, prob_threshold=0.5, iou_threshold=0.6
            )
            res_image = net.detect(image)

            # winName = 'Deep learning object detection in OpenCV'
            # cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
            # cv2.imshow(winName, res_image)
        else:
            res_image = None

    return res_image
