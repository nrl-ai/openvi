import pathlib
import json
import os

import dearpygui.dearpygui as dpg


class ModelTable:
    def __init__(self) -> None:
        self.project_path = None
        self.models_path = None

        dpg.add_text("Trained models")
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Refresh", width=200, callback=self.rescan_models
            )
        with dpg.group(horizontal=True, tag="trained_models"):
            with dpg.table(
                header_row=True, reorderable=True, tag="trained_models_table"
            ):
                dpg.add_table_column(label="Training name")
                dpg.add_table_column(label="Dataset")
                dpg.add_table_column(label="Model")
                dpg.add_table_column(label="Epoch")
                dpg.add_table_column(label="Batch size")
                dpg.add_table_column(label="Learning rate")
                dpg.add_table_column(label="Image size")
                dpg.add_table_column(label="Accuracy")

    def delete_model(self):
        if self.models_path is None:
            return
        selected_models = dpg.get_table_selections("Trained models")
        for selected_model in selected_models:
            metadata_path = self.models_path / "metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
            model_path = self.models_path / "model.pt"
            if model_path.exists():
                model_path.unlink()
            dpg.delete_item(selected_model)

    def rescan_models(self):
        if self.models_path is None:
            return
        # Clear table
        dpg.delete_item("trained_models_table")
        with dpg.table(
            header_row=True,
            reorderable=True,
            tag="trained_models_table",
            parent="trained_models",
        ):
            dpg.add_table_column(label="Name")
            dpg.add_table_column(label="Created At")
            dpg.add_table_column(label="Epoch")
            dpg.add_table_column(label="Batch size")
            dpg.add_table_column(label="Learning rate")
            dpg.add_table_column(label="Image size")
            dpg.add_table_column(label="Accuracy")
            dpg.add_table_column(label="Val Accuracy")
            for model_folder in sorted(
                os.listdir(self.models_path), reverse=True
            ):
                metadata_path = (
                    pathlib.Path(self.models_path)
                    / model_folder
                    / "metadata.json"
                )
                if not metadata_path.exists():
                    print(f"Metadata file {metadata_path} does not exist")
                    continue
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    with dpg.table_row():
                        dpg.add_text(metadata["name"])
                        dpg.add_text(metadata["created_time"])
                        dpg.add_text(metadata["epoch"])
                        dpg.add_text(metadata["batch_size"])
                        dpg.add_text(metadata["learning_rate"])
                        dpg.add_text(metadata["image_size"])
                        dpg.add_text(metadata["accuracy"])
                        dpg.add_text(metadata["val_accuracy"])

    def set_project_path(self, project_path):
        self.project_path = project_path
        self.models_path = pathlib.Path(self.project_path) / "models"
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.rescan_models()
