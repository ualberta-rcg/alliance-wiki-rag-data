# Keras

**Other languages:** English, français

"Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano."<sup>[1]</sup>

If you are porting a Keras program to one of our clusters, you should follow [our tutorial on the subject](link_to_tutorial_here).


## Installing

### Install TensorFlow, CNTK, or Theano

Install TensorFlow, CNTK, or Theano in a Python virtual environment.

Activate the Python virtual environment (named `$HOME/tensorflow` in our example):

```bash
[name@server ~]$ source $HOME/tensorflow/bin/activate
```

Install Keras in your virtual environment:

```bash
(tensorflow) [name@server ~]$ pip install keras
```

### R Package

This section details how to install Keras for R and use TensorFlow as the backend.

Install TensorFlow for R by following [these instructions](link_to_tensorflow_r_instructions_here).

Follow the instructions from the parent section.

Load the required modules:

```bash
[name@server ~]$ module load gcc/7.3.0 r/3.5.2
```

Launch R:

```bash
[name@server ~]$ R
```

In R, install the Keras package with `devtools`:

```R
devtools::install_github('rstudio/keras')
```

You are then good to go. Do not call `install_keras()` in R, as Keras and TensorFlow have already been installed in your virtual environment with `pip`. To use the Keras package installed in your virtual environment, enter the following commands in R after the environment has been activated:

```R
library(keras)
use_virtualenv(Sys.getenv('VIRTUAL_ENV'))
```

## References

[1] https://keras.io/


**(Retrieved from "https://docs.alliancecan.ca/mediawiki/index.php?title=Keras&oldid=139585")**
