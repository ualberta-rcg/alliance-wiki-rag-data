# OpenCV (Open Source Computer Vision Library)

OpenCV (Open Source Computer Vision Library) is a library of programming functions mainly aimed at real-time computer vision.

## Contents

* [CUDA](#cuda)
* [Extra modules](#extra-modules)
* [Python bindings](#python-bindings)
    * [Usage](#usage)
    * [Available Python packages](#available-python-packages)
* [Use with OpenEXR](#use-with-openexr)

## CUDA

OpenCV is also available with CUDA.

```bash
[name@server ~]$ module load gcc cuda opencv/X.Y.Z
```

where `X.Y.Z` represent the desired version.

## Extra modules

The module also contains the extra modules (contrib).

## Python bindings

The module contains bindings for multiple Python versions. To discover which are the compatible Python versions, run:

```bash
[name@server ~]$ module spider opencv/X.Y.Z
```

Or search directly `opencv_python`, by running:

```bash
[name@server ~]$ module spider opencv_python/X.Y.Z
```

where `X.Y.Z` represent the desired version.

### Usage

1. Load the required modules.

```bash
[name@server ~]$ module load gcc opencv/X.Y.Z python scipy-stack
```

where `X.Y.Z` represent the desired version.

2. Import OpenCV.

```bash
[name@server ~]$ python -c "import cv2"
```

If the command displays nothing, the import was successful.

### Available Python packages

Other Python packages depend on OpenCV bindings in order to be installed. OpenCV provides four different packages:

* `opencv_python`
* `opencv_contrib_python`
* `opencv_python_headless`
* `opencv_contrib_python_headless`

```bash
[name@server ~]$ pip list | grep opencv
opencv-contrib-python       4.5.5
opencv-contrib-python-headless 4.5.5
opencv-python               4.5.5
opencv-python-headless      4.5.5
```

With the `opencv` module loaded, your package dependency for one of the OpenCV named will be satisfied.

## Use with OpenEXR

In order to read EXR files with OpenCV, the module must be activated through an environment variable.

```bash
[name@server ~]$ OPENCV_IO_ENABLE_OPENEXR=1 python <file>
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=OpenCV&oldid=152030](https://docs.alliancecan.ca/mediawiki/index.php?title=OpenCV&oldid=152030)"
