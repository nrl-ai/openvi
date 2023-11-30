import dearpygui.dearpygui as dpg
from openvi.pi.test_socket_client import socket_client, check_conn
from zipfile import ZipFile
import shutil
import os
from os.path import basename


class Deployment:
    def __init__(self) -> None:
        with dpg.group(horizontal=True):
            # SSH information to deployment device
            with dpg.group(horizontal=False):
                dpg.add_text("SSH information")
                dpg.add_text("IP address")
                dpg.add_input_text(width=500, tag="ip_address")
                dpg.add_text("Port")
                dpg.add_input_text(width=500, tag="port")
                dpg.add_text("Browse flow file")
                dpg.add_input_text(width=500, tag="export_file_path")
                # with dpg.group(horizontal=True):
                #     with dpg.group(horizontal=False):
                #         dpg.add_text("Select export file")
                #     with dpg.group(horizontal=False):
                #         dpg.add_input_text(width=260, tag="export_file_path")
                #         dpg.add_button(label="Browse", callback=self.browse_dataset_folder)
                # Add space
                dpg.add_text("")
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Deploy",
                        width=200,
                        height=100,
                        callback=self._callback_deploy_,
                    )
                    dpg.add_button(
                        label="Try connection",
                        width=200,
                        height=100,
                        callback=self._callback_connect_,
                    )
            with dpg.group(horizontal=False):
                dpg.add_text("Logs")
                dpg.add_input_text(
                    width=500, multiline=True, height=300, tag="deployment_log"
                )

    def _callback_deploy_(self):
        # logs = " ".join("Deploy to Raspberry Pi ..." )

        ip_addr = dpg.get_value("ip_address")
        port = int(dpg.get_value("port"))
        export_file_path = dpg.get_value("export_file_path")
        dst_name = "export.json"
        # dst_file_path = os.path.join(os.path.dirname(export_file_path), dst_name)
        dst_file_path = os.path.join("/tmp", dst_name)
        print(ip_addr, port)
        print(export_file_path)
        shutil.copyfile(export_file_path, dst_file_path)

        zip_file = "socket_client.zip"
        zip_path = os.path.join(os.path.dirname(__file__), "pi", zip_file)
        with ZipFile(zip_path, "w") as myzip:
            myzip.write(dst_file_path, basename(dst_file_path))

        logs = socket_client(ip_addr, port, zip_path)
        logs = "\n".join(logs)
        dpg.set_value("deployment_log", logs)

    def _callback_connect_(self):
        ip_addr = dpg.get_value("ip_address")
        port = int(dpg.get_value("port"))
        ret, logs = check_conn(ip_addr, port)
        logs = "\n".join(logs)
        dpg.set_value("deployment_log", logs)

    def browse_dataset_folder(self, sender, data):
        with dpg.file_dialog(
            label="Select Dataset Folder",
            default_path="./",
            file_count=0,
            directory_selector=False,
            show=True,
            modal=True,
            width=500,
            height=500,
            callback=lambda _, user_data: self.set_dataset_folder(user_data),
            extensions=".*,.json",
        ):
            # directory = data[0]
            # file_directory = data[1]
            # print (directory, file_directory)

            # dpg.add_data("directory", directory)
            # dpg.add_data("file_directory", file_directory)
            pass

    # def set_dataset_folder(self, user_data):
    #     dpg.set_value("dataset_folder", user_data["file_path_name"])
    #     dpg.set_value("training_log", "Dataset folder set.")
