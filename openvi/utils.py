import dearpygui.dearpygui as dpg


def close_then_call(callback=None):
    def wrapper(sender, app_data, user_data):
        modal_id, is_ok = user_data
        dpg.configure_item(modal_id, show=False)
        if callback is not None:
            callback(sender, app_data, user_data)

    return wrapper


def show_confirm(title, message, ok_callback=None, cancel_callback=None):
    if ok_callback is None:

        def ok_callback(sender, app_data, user_data):
            modal_id, is_ok = user_data
            dpg.configure_item(modal_id, show=False)

    with dpg.mutex():
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
        with dpg.window(label=title, modal=True, no_close=True) as modal_id:
            dpg.add_text(message)
            dpg.add_button(
                label="Ok",
                width=75,
                user_data=(modal_id, True),
                callback=close_then_call(callback=ok_callback),
            )
            dpg.group(horizontal=True)
            dpg.add_button(
                label="Cancel",
                width=75,
                user_data=(modal_id, False),
                callback=close_then_call(callback=cancel_callback),
            )
    dpg.split_frame()
    width = dpg.get_item_width(modal_id)
    height = dpg.get_item_height(modal_id)
    dpg.set_item_pos(
        modal_id,
        [viewport_width // 2 - width // 2, viewport_height // 2 - height // 2],
    )


def show_error(title, message):
    with dpg.mutex():
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
        with dpg.window(label=title, modal=True, no_close=True) as modal_id:
            dpg.add_text(message)
            dpg.add_button(
                label="Ok",
                width=75,
                user_data=(modal_id, True),
                callback=close_then_call(),
            )
    dpg.split_frame()
    width = dpg.get_item_width(modal_id)
    height = dpg.get_item_height(modal_id)
    dpg.set_item_pos(
        modal_id,
        [viewport_width // 2 - width // 2, viewport_height // 2 - height // 2],
    )
