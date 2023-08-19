from setuptools import setup, find_packages
import re
import os
from os import path
from os.path import splitext
from os.path import basename
from glob import glob

package_keyword = "openvi"


def get_version():
    with open("__init__.py", "r") as f:
        version = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE
        ).group(1)
    return version


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


extra_py_files = package_files("openvi")

readme_path = path.abspath(path.dirname(__file__))
with open(path.join(readme_path, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="openvi",
    packages=find_packages(),
    py_modules=[splitext(basename(path))[0] for path in glob("./*")],
    package_data={"node_editor": extra_py_files},
    include_package_data=True,
    version=get_version(),
    author="OpenVI Team",
    url="https://github.com/openvi-team/openvi",
    description="No-code platform for computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="opencv node-editor onnx onnxruntime dearpygui",
    license="Apache-2.0 license",
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.23.0",
        "Cython==0.29.30",
        "opencv-contrib-python==4.5.5.64",
        "onnxruntime==1.12.0",
        "dearpygui==1.6.2",
        "mediapipe==0.10.3",
        "protobuf==3.20.0",
        "filterpy==1.4.5",
        "lap==0.4.0",
        "cython-bbox==0.1.3",
        "rich==12.4.4",
    ],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={
        "console_scripts": [
            "openvi=main:main",
        ],
    },
)
