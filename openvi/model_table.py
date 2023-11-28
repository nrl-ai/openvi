import dearpygui.dearpygui as dpg


class ModelTable():
    def __init__(self) -> None:
        dpg.add_text("Trained models")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Refresh", width=200)
            dpg.add_button(label="Delete", width=200)
        with dpg.table(header_row=True):
            dpg.add_table_column(label="Training name")
            dpg.add_table_column(label="Dataset")
            dpg.add_table_column(label="Model")
            dpg.add_table_column(label="Epoch")
            dpg.add_table_column(label="Batch size")
            dpg.add_table_column(label="Learning rate")
            dpg.add_table_column(label="Image size")
            dpg.add_table_column(label="Accuracy")