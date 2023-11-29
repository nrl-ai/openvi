import uuid

import dearpygui.dearpygui as dpg

from openvi.utils import show_confirm

class AutoTrainer():
    def __init__(self) -> None:
        self.is_training = False
        dpg.add_text("AI AutoTrain - No-code Trainer for Computer Vision")
        dpg.add_text("Training name")
        dpg.add_input_text(width=500, tag="training_name")
        dpg.set_value("training_name", uuid.uuid4().hex)
        dpg.add_text("Model")
        dpg.add_combo(
            items=["Object Detection", "Image Classification"],
            default_value="Object Detection",
            width=500,
            tag="model",
        )
        dpg.add_text("Dataset")
        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False):
                dpg.add_text("Select dataset folder")
            with dpg.group(horizontal=False):
                dpg.add_input_text(width=260, tag="dataset_folder")
                dpg.add_button(label="Browse", callback=self.browse_dataset_folder)
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
            dpg.add_button(label="Start Training", tag="start_stop_training", width=500, height=80, callback=self.start_training)
            dpg.add_spacer()
        # Training log
        dpg.add_text("Training log")
        dpg.add_input_text(width=500, multiline=True, height=300, tag="training_log", readonly=True)

    def set_dataset_folder(self, user_data):
        dpg.set_value("dataset_folder", user_data["file_path_name"])
        dpg.set_value("training_log", "Dataset folder set.")

    def browse_dataset_folder(self):
        with dpg.file_dialog(
                label="Select Dataset Folder",
                default_path="./",
                file_count=0,
                directory_selector=True,
                show=True,
                modal=True,
                width=500,
                height=500,
                callback=lambda _, user_data: self.set_dataset_folder(user_data),
            ):
            pass

    def stop_training(self, sender, app_data, user_data):
        dpg.set_value("training_log", "Stopping training...")
        dpg.set_item_label("start_stop_training", "Start Training")

    def start_training(self):
        if self.is_training:
            show_confirm("Stop Training", "Are you sure you want to stop training?", ok_callback=self.stop_training)
            return
        if self.validate_fields():
            dpg.set_value("training_log", "Starting training...")
            dpg.set_item_label("start_stop_training", "Stop Training")
            self.is_training = True

    def validate_fields(self):
        dataset_folder = dpg.get_value("dataset_folder")
        epoch = dpg.get_value("epoch")
        batch_size = dpg.get_value("batch_size")
        learning_rate = dpg.get_value("learning_rate")
        image_size = dpg.get_value("image_size")
        if dataset_folder == "":
            dpg.set_value("training_log", "Dataset folder is required.")
            return False
        if epoch == "":
            dpg.set_value("training_log", "Epoch is required.")
            try:
                epoch = int(epoch)
            except ValueError:
                dpg.set_value("training_log", "Epoch must be an integer.")
                return False
            if epoch < 1:
                dpg.set_value("training_log", "Epoch must be greater than 0.")
                return False
        if batch_size == "":
            dpg.set_value("training_log", "Batch size is required.")
            try:
                batch_size = int(batch_size)
            except ValueError:
                dpg.set_value("training_log", "Batch size must be an integer.")
                return False
            if batch_size < 1:
                dpg.set_value("training_log", "Batch size must be greater than 0.")
                return False
        if learning_rate == "":
            dpg.set_value("training_log", "Learning rate is required.")
            try:
                learning_rate = float(learning_rate)
            except ValueError:
                dpg.set_value("training_log", "Learning rate must be a float.")
                return False
            if learning_rate < 0:
                dpg.set_value("training_log", "Learning rate must be greater than 0.")
                return False
        if image_size == "":
            dpg.set_value("training_log", "Image size is required.")
            try:
                image_size = int(image_size)
            except ValueError:
                dpg.set_value("training_log", "Image size must be an integer.")
                return False
            if image_size < 1:
                dpg.set_value("training_log", "Image size must be greater than 0.")
                return False
        return True
