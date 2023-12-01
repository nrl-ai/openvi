import cv2
import numpy as np


def image_processing(image, method_name, params):
    # print ("image processing")
    # print (method_name, params)

    if method_name == "ApplyColorMap":
        _colormap_types = {
            "COLORMAP_AUTUMN": cv2.COLORMAP_AUTUMN,
            "COLORMAP_BONE": cv2.COLORMAP_BONE,
            "COLORMAP_JET": cv2.COLORMAP_JET,
            "COLORMAP_WINTER": cv2.COLORMAP_WINTER,
            "COLORMAP_RAINBOW": cv2.COLORMAP_RAINBOW,
            "COLORMAP_OCEAN": cv2.COLORMAP_OCEAN,
            "COLORMAP_SUMMER": cv2.COLORMAP_SUMMER,
            "COLORMAP_SPRING": cv2.COLORMAP_SPRING,
            "COLORMAP_COOL": cv2.COLORMAP_COOL,
            "COLORMAP_HSV": cv2.COLORMAP_HSV,
            "COLORMAP_PINK": cv2.COLORMAP_PINK,
            "COLORMAP_HOT": cv2.COLORMAP_HOT,
            "COLORMAP_PARULA": cv2.COLORMAP_PARULA,
            "COLORMAP_MAGMA": cv2.COLORMAP_MAGMA,
            "COLORMAP_INFERNO": cv2.COLORMAP_INFERNO,
            "COLORMAP_PLASMA": cv2.COLORMAP_PLASMA,
            "COLORMAP_VIRIDIS": cv2.COLORMAP_VIRIDIS,
            "COLORMAP_CIVIDIS": cv2.COLORMAP_CIVIDIS,
            "COLORMAP_TWILIGHT": cv2.COLORMAP_TWILIGHT,
            "COLORMAP_TWILIGHT_SHIFTED": cv2.COLORMAP_TWILIGHT_SHIFTED,
            "COLORMAP_TURBO": cv2.COLORMAP_TURBO,
            "COLORMAP_DEEPGREEN": cv2.COLORMAP_DEEPGREEN,
        }
        colormap_type = list(params.values())[-1]
        colormap_type = _colormap_types[colormap_type]
        image = cv2.applyColorMap(image, colormap_type)
    elif method_name == "Blur":
        kernel_size = list(params.values())[-1]
        image = cv2.blur(image, (kernel_size, kernel_size))
    elif method_name == "Brightness":
        beta = list(params.values())[-1]
        image = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
    elif method_name == "Canny":
        min_val = list(params.values())[-1]
        max_val = 200
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.Canny(image, min_val, max_val)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif method_name == "Contrast":
        alpha = list(params.values())[-1]
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    elif method_name == "Crop":
        max_y = list(params.values())[-1]
        min_y = list(params.values())[-2]
        max_x = list(params.values())[-3]
        min_x = list(params.values())[-4]

        if max_x < min_x:
            max_x = min_x + 0.01
        if max_y < min_y:
            max_y = min_y + 0.01

        image_height, image_width = image.shape[0], image.shape[1]
        min_x_ = int(min_x * image_width)
        max_x_ = int(max_x * image_width)
        min_y_ = int(min_y * image_height)
        max_y_ = int(max_y * image_height)
        image = image[min_y_:max_y_, min_x_:max_x_]
    elif method_name == "Flip":
        vflip_flag = list(params.values())[-1]
        hflip_flag = list(params.values())[-1]

        flipcode = None
        if hflip_flag and vflip_flag:
            flipcode = 0
        elif hflip_flag:
            flipcode = 1
        elif vflip_flag:
            flipcode = -1

        if flipcode is not None:
            image = cv2.flip(image, flipcode)
    elif method_name == "GammaCorrection":
        gamma = list(params.values())[-1]
        table = (np.arange(256) / 255) ** gamma * 255
        table = np.clip(table, 0, 255).astype(np.uint8)
        image = cv2.LUT(image, table)
    elif method_name == "Grayscale":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif method_name == "OmnidirectionalViewer":
        imagepoint = list(params.values())[-1]
        roll = list(params.values())[-2]
        yaw = list(params.values())[-3]
        pitch = list(params.values())[-4]

        _output_width = 960
        _output_height = 540
        _sensor_size = 0.561
        sensor_width = _sensor_size
        sensor_height = _sensor_size
        sensor_height *= _output_height / _output_width

        rotation_matrix = create_rotation_matrix(
            roll,
            pitch,
            yaw,
        )

        phi, theta = calculate_phi_and_theta(
            -1.0,
            imagepoint,
            sensor_width,
            sensor_height,
            _output_width,
            _output_height,
            rotation_matrix,
        )

        image = remap_image(image, phi, theta)

    elif method_name == "Resize":
        _interpolation = {
            "INTER_LINEAR": cv2.INTER_LINEAR,
            "INTER_NEAREST": cv2.INTER_NEAREST,
            "INTER_AREA": cv2.INTER_AREA,
            "INTER_CUBIC": cv2.INTER_CUBIC,
            "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
            "INTER_NEAREST_EXACT": cv2.INTER_NEAREST_EXACT,
        }

        interpolation_text = list(params.values())[-1]
        interpolation_flag = _interpolation[interpolation_text]
        width = list(params.values())[-2]
        height = list(params.values())[-3]

        image = cv2.resize(
            image,
            dsize=(width, height),
            interpolation=interpolation_flag,
        )
    elif method_name == "SimpleFilter":
        x0y1 = list(params.values())[-1]
        x2y0 = list(params.values())[-2]
        x1y0 = list(params.values())[-3]
        x0y0 = list(params.values())[-4]
        x1y1 = 1.0
        x2y1 = 0.0
        x0y2 = 0.0
        x1y2 = 0.0
        x2y2 = 0.0
        k = 1.0

        kernel = (
            np.array(
                [[x0y0, x1y0, x2y0], [x0y1, x1y1, x2y1], [x0y2, x1y2, x2y2]]
            )
            * k
        )
        image = cv2.filter2D(image, -1, kernel)
    elif method_name == "Threshold":
        _threshold_types = {
            "THRESH_BINARY": cv2.THRESH_BINARY,
            "THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,
            "THRESH_TRUNC": cv2.THRESH_TRUNC,
            "THRESH_TOZERO": cv2.THRESH_TOZERO,
            "THRESH_TOZERO_INV": cv2.THRESH_TOZERO_INV,
            "THRESH_OTSU": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        }

        binary_threshold = list(params.values())[-1]
        threshold_type = list(params.values())[-2]
        threshold_type = _threshold_types[threshold_type]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(
            image,
            binary_threshold,
            255,
            threshold_type,
        )
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def remap_image(image, phi, theta):
    input_height, input_width = image.shape[:2]

    phi = phi * input_height / np.pi + input_height / 2
    phi = phi.astype(np.float32)
    theta = theta * input_width / (2 * np.pi) + input_width / 2
    theta = theta.astype(np.float32)

    output_image = cv2.remap(image, theta, phi, cv2.INTER_CUBIC)

    return output_image


def calculate_phi_and_theta(
    viewpoint,
    imagepoint,
    sensor_width,
    sensor_height,
    output_width,
    output_height,
    rotation_matrix,
):
    width = np.arange(
        (-1) * sensor_width,
        sensor_width,
        sensor_width * 2 / output_width,
    )
    height = np.arange(
        (-1) * sensor_height,
        sensor_height,
        sensor_height * 2 / output_height,
    )

    ww, hh = np.meshgrid(width, height)

    point_distance = imagepoint - viewpoint
    if point_distance == 0:
        point_distance = 0.1

    a1 = ww / point_distance
    a2 = hh / point_distance
    b1 = -a1 * viewpoint
    b2 = -a2 * viewpoint

    a = 1 + (a1**2) + (a2**2)
    b = 2 * ((a1 * b1) + (a2 * b2))
    c = (b1**2) + (b2**2) - 1

    d = ((b**2) - (4 * a * c)) ** (1 / 2)

    x = (-b + d) / (2 * a)
    y = (a1 * x) + b1
    z = (a2 * x) + b2

    xd = (
        rotation_matrix[0][0] * x
        + rotation_matrix[0][1] * y
        + rotation_matrix[0][2] * z
    )
    yd = (
        rotation_matrix[1][0] * x
        + rotation_matrix[1][1] * y
        + rotation_matrix[1][2] * z
    )
    zd = (
        rotation_matrix[2][0] * x
        + rotation_matrix[2][1] * y
        + rotation_matrix[2][2] * z
    )

    phi = np.arcsin(zd)
    theta = np.arcsin(yd / np.cos(phi))

    xd[xd > 0] = 0
    xd[xd < 0] = 1
    yd[yd > 0] = np.pi
    yd[yd < 0] = -np.pi

    offset = yd * xd
    gain = -2 * xd + 1
    theta = gain * theta + offset

    return phi, theta


def create_rotation_matrix(roll, pitch, yaw):
    roll = roll * np.pi / 180
    pitch = pitch * np.pi / 180
    yaw = yaw * np.pi / 180

    matrix01 = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), np.sin(roll)],
            [0, -np.sin(roll), np.cos(roll)],
        ]
    )

    matrix02 = np.array(
        [
            [np.cos(pitch), 0, -np.sin(pitch)],
            [0, 1, 0],
            [np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    matrix03 = np.array(
        [
            [np.cos(yaw), np.sin(yaw), 0],
            [-np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    matrix = np.dot(matrix03, np.dot(matrix02, matrix01))

    return matrix
