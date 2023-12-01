import cv2
import json
from pi.image_processing_handler import image_processing
from pi.deep_learning_handler import deep_learning
from time import sleep
from picamera import PiCamera
import numpy as np


# export_file = 'node_editor/setting/export.json'
# with open(export_file) as fp:
#    work_flow = json.load(fp)

input_nodes = ["WebCam"]

image_processing_nodes = [
    "ApplyColorMap",
    "Blur",
    "Brightness",
    "Canny",
    "Contrast",
    "Crop",
    "EqualizeHist",
    "Flip",
    "GammaCorrection",
    "Grayscale",
    "OmnidirectionalViewer",
    "Resize",
    "SimpleFilter",
    "Threshold",
]

deep_learning_nodes = ["Classification", "ObjectDetection"]

camera = PiCamera()
camera.resolution = (320, 320)
camera.framerate = 24
camera.rotation = 180
sleep(2)
image = np.empty((320, 320, 3), dtype=np.uint8)

while True:  # Loop to capture webcam.
    try:
        camera.capture(image, "bgr")
        export_file = "node_editor/setting/export.json"
        with open(export_file) as fp:
            work_flow = json.load(fp)

        link_list = work_flow["link_list"]
        for node in link_list:
            start_node_idx = node[0].split(":")[0]
            start_node_name = node[0].split(":")[1]
            end_node_idx = node[1].split(":")[0]
            end_node_name = node[1].split(":")[1]
            print(f"{start_node_name}-{end_node_name}")

            # Handle start nodes
            if start_node_name in input_nodes:
                start_node_img = image
            elif start_node_name in image_processing_nodes:
                start_node_cfg = work_flow[
                    f"{start_node_idx}:{start_node_name}"
                ]["setting"]
                start_node_img = image_processing(
                    image, start_node_name, start_node_cfg
                )
            elif start_node_name in deep_learning_nodes:
                start_node_cfg = work_flow[
                    f"{start_node_idx}:{start_node_name}"
                ]["setting"]
                img = deep_learning(image, start_node_name, start_node_cfg)

            # Handle end nodes
            if end_node_name in image_processing_nodes:
                end_node_cfg = work_flow[f"{end_node_idx}:{end_node_name}"][
                    "setting"
                ]
                img = image_processing(
                    start_node_img, end_node_name, end_node_cfg
                )
                # cv2.imshow(f'{end_node_name}', img)
                # cv2.waitKey()
                # break
            elif end_node_name in deep_learning_nodes:
                end_node_cfg = work_flow[f"{end_node_idx}:{end_node_name}"][
                    "setting"
                ]
                img = deep_learning(
                    start_node_img, end_node_name, end_node_cfg
                )
                cv2.imshow(f"{end_node_name}", img)
        cv2.waitKey()
    except:
        traceback.print_exc()

    cv2.destroyAllWindows()
