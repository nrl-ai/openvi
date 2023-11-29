# OpenVI - Open toolkit for Computer Vision

This project aims to create a toolkit for computer vision, with image processing and computer vision training with no code. The main features are:

- Image processing graph: Everyone can create image processing graphs with OpenCV. Try the result in real time.
- Training machine learning models for object detection, image classification, and other computer vision tasks.
- Deploy machine learning models to devices.

![OpenVI](openvi.png)

- Python 3.9

```bash

pip install Cython numpy==1.23.0
pip install -e .
```

## Run

```bash
python -m openvi.main
```

## References

We used some code from the following repositories:

- Image processing graph: [Image-Processing-Node-Editor](https://github.com/Kazuhito00/Image-Processing-Node-Editor) - License: Apache 2.0.
