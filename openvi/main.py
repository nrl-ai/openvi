#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import copy
import json
import asyncio
import argparse
from collections import OrderedDict
import webbrowser
import os

sys.path.append(".")

import cv2
import dearpygui.dearpygui as dpg
from dearpygui_ext.themes import create_theme_imgui_light

import openvi.global_data as global_data


from openvi.node_editor.util import check_camera_connection
from openvi.node_editor.node_editor import DpgNodeEditor


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--setting",
        type=str,
        # get abs
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "node_editor/setting/setting.json"
            )
        ),
    )
    parser.add_argument("--unuse_async_draw", action="store_true")
    parser.add_argument("--use_debug_print", action="store_true")

    args = parser.parse_args()

    return args


def async_main(node_editor):
    node_image_dict = {}
    node_result_dict = {}
    while not node_editor.get_terminate_flag():
        update_node_info(node_editor, node_image_dict, node_result_dict)


def update_node_info(
    node_editor,
    node_image_dict,
    node_result_dict,
    mode_async=True,
):
    # Get node list
    node_list = node_editor.get_node_list()

    # Get node connection information
    sorted_node_connection_dict = node_editor.get_sorted_node_connection()

    # Update information of each node
    for node_id_name in node_list:
        if node_id_name not in node_image_dict:
            node_image_dict[node_id_name] = None

        node_id, node_name = node_id_name.split(":")
        connection_list = sorted_node_connection_dict.get(node_id_name, [])

        # Get instance from openvi.node name
        node_instance = node_editor.get_node_instance(node_name)

        # Update information of specified node
        if mode_async:
            try:
                image, result = node_instance.update(
                    node_id,
                    connection_list,
                    node_image_dict,
                    node_result_dict,
                )
            except Exception as e:
                print(e)
                sys.exit()
        else:
            image, result = node_instance.update(
                node_id,
                connection_list,
                node_image_dict,
                node_result_dict,
            )
        node_image_dict[node_id_name] = copy.deepcopy(image)
        node_result_dict[node_id_name] = copy.deepcopy(result)


def main():
    args = get_args()
    setting = args.setting
    unuse_async_draw = args.unuse_async_draw
    use_debug_print = args.use_debug_print

    # Load setting file
    print("**** Load Config ********")
    opencv_setting_dict = None
    with open(setting) as fp:
        opencv_setting_dict = json.load(fp)
    webcam_width = opencv_setting_dict["webcam_width"]
    webcam_height = opencv_setting_dict["webcam_height"]

    # Check camera connection
    print("**** Check Camera Connection ********")
    device_no_list = check_camera_connection()
    camera_capture_list = []
    for device_no in device_no_list:
        video_capture = cv2.VideoCapture(device_no)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
        camera_capture_list.append(video_capture)

    # Hold camera settings
    opencv_setting_dict["device_no_list"] = device_no_list
    opencv_setting_dict["camera_capture_list"] = camera_capture_list

    # DearPyGui preparation (context generation, setup, viewport generation)
    editor_width = opencv_setting_dict["editor_width"]
    editor_height = opencv_setting_dict["editor_height"]

    # Serial connection device check
    serial_device_no_list = []
    serial_connection_list = []
    use_serial = opencv_setting_dict["use_serial"]
    if use_serial:
        import serial

        try:
            from .node_editor.util import check_serial_connection
        except:
            from openvi.node_editor.util import check_serial_connection
        print("**** Check Serial Device Connection ********")
        serial_device_no_list = check_serial_connection()
        for serial_device_no in serial_device_no_list:
            ser = serial.Serial(serial_device_no, 115200)
            serial_connection_list.append(ser)

    # Keep serial connection device settings
    opencv_setting_dict["serial_device_no_list"] = serial_device_no_list
    opencv_setting_dict["serial_connection_list"] = serial_connection_list

    print("**** DearPyGui Setup ********")
    dpg.create_context()
    dpg.setup_dearpygui()
    dpg.create_viewport(
        title="OpenVI - Open Vision Intelligence",
        width=editor_width,
        height=editor_height,
    )

    # Change default font get the path of this file
    current_path = os.path.dirname(os.path.abspath(__file__))
    with dpg.font_registry():
        with dpg.font(
            current_path + "/font/Roboto/Roboto-Regular.ttf",
            16,
        ) as default_font:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese)
    dpg.bind_font(default_font)

    # Node editor generation
    print("**** Create NodeEditor ********")
    menu_dict = OrderedDict(
        {
            "InputNode": "input_node",
            "ProcessNode": "process_node",
            "DeepLearningNode": "deep_learning_node",
            "AnalysisNode": "analysis_node",
            "DrawNode": "draw_node",
            "OtherNode": "other_node",
            "PreviewReleaseNode": "preview_release_node",
        }
    )

    # Create main window
    width = 1280
    height = 720
    pos = [0, 0]
    with dpg.window(
        tag="OpenVI Window",
        label="OpenVI Window",
        width=width,
        height=height,
        pos=pos,
        menubar=True,
        no_title_bar=True,
    ):
        with dpg.viewport_menu_bar():
            with dpg.menu(label="Project"):
                dpg.add_menu_item(
                    label="New Project",
                )
                dpg.add_menu_item(
                    label="Open Project",
                )
                dpg.add_menu_item(
                    label="Export Project",
                )
                dpg.add_menu_item(
                    label="Exit", callback=lambda: dpg.stop_dearpygui()
                )
            with dpg.menu(label="About"):
                dpg.add_menu_item(
                    label="Open Github",
                    callback=lambda: webbrowser.open(
                        "https://github.com/openvi-team/openvi"
                    ),
                )

        def open_project():
            global_data.project_path = True
            print("**** Open Project ********")
            dpg.configure_item("No Project Opened", show=False)
            dpg.configure_item("Main Tab Bar", show=True)

        def close_project():
            global_data.project_path = None
            print("**** Close Project ********")
            dpg.configure_item("No Project Opened", show=True)
            dpg.configure_item("Main Tab Bar", show=False)

        # Add tabs
        with dpg.group(horizontal=True):
            with dpg.child_window(width=250, autosize_y=True):
                # Show project information
                with dpg.group(horizontal=False):
                    dpg.add_text("Project Name:")
                    dpg.add_input_text(width=230)
                    dpg.add_text("Description:")
                    dpg.add_input_text(width=230, multiline=True, height=100)
                # Show project operation
                with dpg.group(horizontal=False):
                    dpg.add_button(label="New Project", width=230)
                    dpg.add_button(label="Open Project", callback=open_project, width=230)
                    dpg.add_button(label="Close Project", callback=close_project, width=230)

            with dpg.child_window(autosize_y=True, tag="No Project Opened"):
                with dpg.group(horizontal=False):
                    dpg.add_text("No Project Opened.")
                    dpg.add_text("Please create or open project.")

            with dpg.tab_bar(tag="Main Tab Bar"):
                with dpg.tab(label="Data Import", show=True):
                    dpg.add_text("Data Import")
                with dpg.tab(label="Data Preparation", show=True):
                    dpg.add_text("Data Preparation")
                with dpg.tab(label="Training", show=True):
                    with dpg.group(horizontal=True):
                        dpg.add_text("Feature place holder:")
                        dpg.add_text(
                            "Manage AI training: Training and manage models"
                        )
                with dpg.tab(label="Inference Pipeline", show=True):
                    node_editor = DpgNodeEditor(
                        height=editor_height,
                        opencv_setting_dict=opencv_setting_dict,
                        menu_dict=menu_dict,
                        use_debug_print=use_debug_print,
                        node_dir=current_path + "/node",
                    )
                with dpg.tab(label="Edge Deployment", show=True):
                    with dpg.group(horizontal=True):
                        dpg.add_text("Feature place holder:")
                        dpg.add_text(
                            "Manage edge device: Deploy models to edge device"
                        )

    dpg.set_primary_window("OpenVI Window", True)
    dpg.show_viewport()

    light_theme = create_theme_imgui_light()
    dpg.bind_theme(light_theme)

    print("**** Start Main Event Loop ********")
    if not unuse_async_draw:
        event_loop = asyncio.get_event_loop()
        event_loop.run_in_executor(None, async_main, node_editor)
        dpg.start_dearpygui()
    else:
        node_image_dict = {}
        node_result_dict = {}
        while dpg.is_dearpygui_running():
            update_node_info(
                node_editor,
                node_image_dict,
                node_result_dict,
                mode_async=False,
            )
            dpg.render_dearpygui_frame()

    print("**** Terminate process ********")
    print("**** Close All Node ********")
    node_list = node_editor.get_node_list()
    for node_id_name in node_list:
        node_id, node_name = node_id_name.split(":")
        node_instance = node_editor.get_node_instance(node_name)
        node_instance.close(node_id)
    print("**** Release All VideoCapture ********")
    for camera_capture in camera_capture_list:
        camera_capture.release()
    print("**** Stop Event Loop ********")
    node_editor.set_terminate_flag()
    event_loop.stop()
    print("**** Destroy DearPyGui Context ********")
    dpg.destroy_context()


if __name__ == "__main__":
    main()
