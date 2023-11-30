import uuid
import subprocess
import threading
import os
import json
import pathlib
import time
import shutil

import dearpygui.dearpygui as dpg

from openvi.utils import show_confirm, show_error


class AutoTrainer:
    def __init__(self) -> None:
        self.is_training = False
        self.training_process = None
        self.project_path = None
        self.last_graph_update = time.time()

        dpg.add_text("AI AutoTrain - No-code Trainer for Computer Vision")
        dpg.add_text("Training name")
        dpg.add_input_text(width=500, tag="training_name")
        dpg.set_value("training_name", uuid.uuid4().hex)
        dpg.add_text("Model")
        dpg.add_combo(
            items=["Image Classification"],
            default_value="Image Classification",
            width=500,
            tag="model",
        )
        dpg.add_text("Dataset")
        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False):
                dpg.add_text("Select dataset folder")
            with dpg.group(horizontal=False):
                dpg.add_input_text(width=260, tag="dataset_folder")
                dpg.set_value(
                    "dataset_folder",
                    "/home/vietanhdev/Workspaces/openvi/image-classification/dataset/mvtec",
                )
                dpg.add_button(
                    label="Browse", callback=self.browse_dataset_folder
                )
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
                dpg.add_input_int(width=300, tag="epoch", default_value=1)
                dpg.add_input_int(width=300, tag="batch_size", default_value=8)
                dpg.add_input_float(
                    width=300, tag="learning_rate", default_value=0.001
                )
                dpg.add_input_int(
                    width=300, tag="image_size", default_value=224
                )
        # Training buttons
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Start Training",
                tag="start_stop_training",
                width=500,
                height=80,
                callback=self.start_training,
            )
            dpg.add_spacer()
        # Draw accuracy chart
        dpg.add_text("Accuracy chart")
        with dpg.plot(width=500, height=200, tag="accuracy_chart"):
            dpg.add_plot_legend(tag="accuracy_chart_legend")
            dpg.add_plot_axis("x", label="Epoch", tag="accuracy_chart_x_axis")
            dpg.add_plot_axis(
                "y", label="Accuracy", tag="accuracy_chart_y_axis"
            )
            dpg.add_line_series(
                [],
                [],
                label="Train",
                parent="accuracy_chart_y_axis",
                tag="accuracy_chart_series",
            )
            dpg.add_line_series(
                [],
                [],
                label="Valid",
                parent="accuracy_chart_y_axis",
                tag="val_accuracy_chart_series",
            )
        # Training log
        dpg.add_text("Training log")
        dpg.add_input_text(
            width=500,
            multiline=True,
            height=300,
            tag="training_log",
            readonly=True,
        )

    def set_project_path(self, project_path):
        self.project_path = project_path

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
        self.training_process.terminate()
        self.training_process = None
        self.is_training = False
        dpg.set_value("training_log", "")
        dpg.set_item_label("start_stop_training", "Start Training")
        dpg.set_value("training_log", "Training stopped.")

    def start_training(self):
        if self.is_training:
            show_confirm(
                "Stop Training",
                "Are you sure you want to stop training?",
                ok_callback=self.stop_training,
            )
            return
        if self.training_process is not None:
            show_error("Error", "Training is already running.")
        if self.validate_fields():
            dpg.set_value("training_log", "Starting training...")
            dpg.set_item_label("start_stop_training", "Stop Training")
            self.is_training = True
            self.training_thread = threading.Thread(
                target=self.training_process_func
            )
            self.training_thread.start()

    def get_next_model_name(self):
        models_path = pathlib.Path(self.project_path) / "models"
        if not os.path.exists(models_path):
            pathlib.Path(models_path).mkdir(parents=True, exist_ok=True)
        # Model name format: 000001, 000002, ...
        model_names = []
        for model_name in os.listdir(models_path):
            if model_name.isdigit():
                model_names.append(int(model_name))
        model_names.sort()
        if len(model_names) == 0:
            return "000001"
        return f"{model_names[-1] + 1:06}"

    def training_process_func(self):
        data_folder = dpg.get_value("dataset_folder")
        num_classes = len(os.listdir(f"{data_folder}/train"))
        self.current_training_path = (
            pathlib.Path(self.project_path) / "training" / uuid.uuid4().hex
        )
        if not os.path.exists(self.current_training_path):
            self.current_training_path.mkdir(parents=True, exist_ok=True)
        command = [
            "docker",
            "run",
            "-it",
            "--shm-size=32GB",
            "--gpus=all",
            "-v",
            f"{dpg.get_value('dataset_folder')}:/workspace/dataset",
            "-v",
            f"{self.current_training_path}:/workspace/out_snapshot",
            "vietanhdev/openvi-image-classification:latest",
            "python",
            "train.py",
            "--network",
            "resnet18",
            "--num_classes",
            str(num_classes),
            "--input_size",
            str(dpg.get_value("image_size")),
            "--epochs",
            str(dpg.get_value("epoch")),
            "--learning_rate",
            str(dpg.get_value("learning_rate")),
            "--batch_size",
            str(dpg.get_value("batch_size")),
            "--dataset",
            "/workspace/dataset",
            "--path_pretrain",
            "pretrains/resnet18-5c106cde.pth",
            "--job_name",
            dpg.get_value("training_name"),
        ]
        dpg.set_value("training_log", " ".join(command))
        self.training_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        while True:
            # Check if training process is still running
            rc = self.training_process.poll()
            if rc is not None:
                break
            # Read output
            output = self.training_process.stdout.readline()
            if output:
                current_log = dpg.get_value("training_log")
                current_log = current_log[-10000:]
                new_log = output.decode("utf-8")
                current_log = (current_log + new_log)[-10000:]
                dpg.set_value("training_log", current_log)
            if self.last_graph_update + 1 < time.time():
                self.last_graph_update = time.time()
                if os.path.exists(self.current_training_path / "metrics.json"):
                    with open(
                        self.current_training_path / "metrics.json", "r"
                    ) as f:
                        metrics = json.load(f)
                        epochs = [
                            i for i in range(1, len(metrics["train_acc"]) + 1)
                        ]
                        accuracy = metrics["train_acc"]
                        dpg.set_value(
                            "accuracy_chart_series", [epochs, accuracy]
                        )
                        dpg.fit_axis_data("accuracy_chart_x_axis")
                        dpg.fit_axis_data("accuracy_chart_y_axis")
                        val_accuracy = metrics["val_acc"]
                        dpg.set_value(
                            "val_accuracy_chart_series", [epochs, val_accuracy]
                        )
                        dpg.fit_axis_data("accuracy_chart_x_axis")
                        dpg.fit_axis_data("accuracy_chart_y_axis")
                else:
                    dpg.set_value("accuracy_chart_series", [[], []])

        # Finish training
        # Copy model to models folder
        model_name = self.get_next_model_name()
        models_path = pathlib.Path(self.project_path) / "models"
        if not os.path.exists(models_path):
            pathlib.Path(models_path).mkdir(parents=True, exist_ok=True)
        model_path = models_path / model_name
        if not os.path.exists(model_path):
            pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)

        # Copy all files in current training path to model path
        for file_name in os.listdir(self.current_training_path):
            shutil.copyfile(
                self.current_training_path / file_name, model_path / file_name
            )
        # Write metadata.json
        with open(model_path / "metadata.json", "w") as f:
            metadata = {
                "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "name": model_name,
                "num_classes": num_classes,
                "image_size": dpg.get_value("image_size"),
                "epoch": dpg.get_value("epoch"),
                "batch_size": dpg.get_value("batch_size"),
                "learning_rate": dpg.get_value("learning_rate"),
                "accuracy": metrics["train_acc"][
                    -1
                ],  # TODO: Get best accuracy
                "val_accuracy": metrics["val_acc"][
                    -1
                ],  # TODO: Get best val accuracy
            }
            json.dump(metadata, f)
        # Delete current training path
        shutil.rmtree(self.current_training_path)

        dpg.set_value("training_log", "Training finished.")
        rc = self.training_process.poll()
        self.is_training = False
        self.training_process = None
        dpg.set_item_label("start_stop_training", "Start Training")
        return rc

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
                dpg.set_value(
                    "training_log", "Batch size must be greater than 0."
                )
                return False
        if learning_rate == "":
            dpg.set_value("training_log", "Learning rate is required.")
            try:
                learning_rate = float(learning_rate)
            except ValueError:
                dpg.set_value("training_log", "Learning rate must be a float.")
                return False
            if learning_rate < 0:
                dpg.set_value(
                    "training_log", "Learning rate must be greater than 0."
                )
                return False
        if image_size == "":
            dpg.set_value("training_log", "Image size is required.")
            try:
                image_size = int(image_size)
            except ValueError:
                dpg.set_value("training_log", "Image size must be an integer.")
                return False
            if image_size < 1:
                dpg.set_value(
                    "training_log", "Image size must be greater than 0."
                )
                return False
        data_folder = dpg.get_value("dataset_folder")
        if not os.path.exists(data_folder):
            dpg.set_value("training_log", "Dataset folder does not exist.")
            self.is_training = False
            return False
        if not os.path.exists(f"{data_folder}/train"):
            dpg.set_value("training_log", "Train folder does not exist.")
            self.is_training = False
            return False
        if not os.path.exists(f"{data_folder}/val"):
            dpg.set_value("training_log", "Val folder does not exist.")
            self.is_training = False
            return False
        return True
