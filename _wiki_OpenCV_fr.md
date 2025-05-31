# OpenCV

The OpenCV (Open Source Computer Vision) library is specialized in real-time image processing.

## CUDA

OpenCV is also available with CUDA.

```bash
[name@server ~]$ module load gcc cuda opencv/X.Y.Z
```

Where `X.Y.Z` designates the chosen version.


## Additional Modules

The module also contains the `contrib` modules.


## Python Interfaces

The module contains interfaces for several Python versions. To find out which interfaces are compatible with your version, run:

```bash
[name@server ~]$ module spider opencv/X.Y.Z
```

or search directly for `opencv_python` with:

```bash
[name@server ~]$ module spider opencv_python/X.Y.Z
```

Where `X.Y.Z` designates the chosen version.


### Usage

1. Load the required modules.

```bash
[name@server ~]$ module load gcc opencv/X.Y.Z python scipy-stack
```

Where `X.Y.Z` designates the chosen version.

2. Import OpenCV.

```bash
[name@server ~]$ python -c "import cv2"
```

The import is successful if nothing is displayed.


### Available Python Packages

Some Python packages require an OpenCV interface to be installed. The module offers the following OpenCV packages:

*   `opencv_python`
*   `opencv_contrib_python`
*   `opencv_python_headless`
*   `opencv_contrib_python_headless`

```bash
[name@server ~]$ pip list | grep opencv
opencv-contrib-python       4.5.5
opencv-contrib-python-headless 4.5.5
opencv-python               4.5.5
opencv-python-headless      4.5.5
```

When the `opencv` module is loaded, the dependency on OpenCV is satisfied.


## Usage with OpenEXR

For OpenCV to be able to read EXR files, the module must be activated via an environment variable.

```bash
[name@server ~]$ OPENCV_IO_ENABLE_OPENEXR=1 python <file>
```
