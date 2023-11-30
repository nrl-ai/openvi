# OpenVI - Open toolkit for Computer Vision

This project aims to create a toolkit for computer vision, with image processing and computer vision training with no code. The main features are:

- Image processing graph: Everyone can create image processing graphs with OpenCV. Try the result in real time.
- Training machine learning models for object detection, image classification, and other computer vision tasks.
- Deploy machine learning models to devices.

![OpenVI](https://raw.githubusercontent.com/openvi-team/openvi/main/openvi.png)

## Environment

- Python 3.9

```bash

pip install Cython numpy==1.23.0
pip install -e .
```

## Run

```bash
python -m openvi.main
```

[OpenVI - Screencast](https://github.com/openvi-team/openvi/assets/18329471/db9047e2-3b0b-4052-bdb2-ea550d481921)

![OpenVI - Screenshot](https://github.com/openvi-team/openvi/assets/18329471/96b6711b-e85f-4429-909b-e80cefaabef1)

![OpenVI - Screenshot](https://github.com/openvi-team/openvi/assets/18329471/5f3a4008-ab71-44d4-9f0f-1a62e0c6cfc7)

![OpenVI - Screenshot](https://github.com/openvi-team/openvi/assets/18329471/d9445a3b-3b43-4344-b633-a3f0ada9ff9e)


## Build and Publish

- Build and publish to PyPI:

```bash
python setup.py build twine
bash build_and_publish.sh
```

## References

We used some code and media from the following sources:

- Image processing graph: [Image-Processing-Node-Editor](https://github.com/Kazuhito00/Image-Processing-Node-Editor) - License: Apache 2.0.
- Video of <a href="https://pixabay.com/vi/users/robert_arangol-17277286/?utm_source=link-attribution&utm_medium=referral&utm_campaign=video&utm_content=46026">Robert Arango Lopez</a> from <a href="https://pixabay.com/vi//?utm_source=link-attribution&utm_medium=referral&utm_campaign=video&utm_content=46026">Pixabay</a>