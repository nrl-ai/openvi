import dearpygui.dearpygui as dpg


class Deployment():
    def __init__(self) -> None:
        with dpg.group(horizontal=True):
            # SSH information to deployment device
            with dpg.group(horizontal=False):
                dpg.add_text("SSH information")
                dpg.add_text("IP address")
                dpg.add_input_text(width=500, tag="ip_address")
                dpg.add_text("Username")
                dpg.add_input_text(width=500, tag="username")
                dpg.add_text("Password")
                dpg.add_input_text(width=500, tag="password")
                # Add space
                dpg.add_text("")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Deploy", width=200, height=100)
                    dpg.add_button(label="Try connection", width=200, height=100)
            with dpg.group(horizontal=False):
                dpg.add_text("Logs")
                dpg.add_input_text(width=500, multiline=True, height=300, tag="deployment_log")