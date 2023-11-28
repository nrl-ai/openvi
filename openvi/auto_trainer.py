import uuid

import dearpygui.dearpygui as dpg


class AutoTrainer():
    def __init__(self) -> None:
        dpg.add_text("AI AutoTrain - No-code Trainer for Computer Vision")
        dpg.add_text("Training name")
        dpg.add_input_text(width=500, tag="training_name")
        dpg.set_value("training_name", uuid.uuid4().hex)
        # Model selection
        dpg.add_text("Model")
        dpg.add_combo(
            items=["Object Detection", "Image Classification"],
            default_value="Object Detection",
            width=500,
            tag="model",
        )
        # Dataset input: Robotflow API Key, Roboflow Workspace, Roboflow Project, Roboflow Version
        dpg.add_text("Dataset")
        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False):
                dpg.add_text("Robotflow API Key")
                dpg.add_text("Robotflow Workspace")
                dpg.add_text("Robotflow Project")
                dpg.add_text("Robotflow Version")
            with dpg.group(horizontal=False):
                dpg.add_input_text(width=300, tag="robotflow_api_key")
                dpg.add_input_text(width=300, tag="robotflow_workspace")
                dpg.add_input_text(width=300, tag="robotflow_project")
                dpg.add_input_text(width=300, tag="robotflow_version")
        # Training input: Epoch, Batch size, Learning rate, Image size, Augmentation
        dpg.add_text("Training")
        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False):
                dpg.add_text("Epoch")
                dpg.add_text("Batch size")
                dpg.add_text("Learning rate")
                dpg.add_text("Image size")
                dpg.add_text("Augmentation")
            with dpg.group(horizontal=False):
                dpg.add_input_int(width=300, tag="epoch", default_value=100)
                dpg.add_input_int(width=300, tag="batch_size", default_value=16)
                dpg.add_input_float(width=300, tag="learning_rate", default_value=0.001)
                dpg.add_input_int(width=300, tag="image_size", default_value=640)
        # Training buttons
        with dpg.group(horizontal=True):
            dpg.add_button(label="Start Training", width=200, height=100)
            dpg.add_button(label="Stop Training", width=200, height=100)
            dpg.add_spacer()
        # Training log
        dpg.add_text("Training log")
        dpg.add_input_text(width=500, multiline=True, height=300, tag="training_log")
