# Keras

Keras is an open-source library written in Python that allows interaction with deep neural network and machine learning algorithms, including TensorFlow, CNTK, and Theano.

If you want to run a Keras program on one of our clusters, it would be beneficial to consult the [tutorial on the subject](link_to_tutorial_here).  (Please replace `link_to_tutorial_here` with the actual link)


## Installation

1. Install TensorFlow, CNTK, or Theano in a Python virtual environment.

2. Activate the virtual environment (in our example, `$HOME/tensorflow`).

   ```bash
   [name@server ~]$ source $HOME/tensorflow/bin/activate
   ```

3. Install Keras in the virtual environment.

   ```bash
   (tensorflow) [name@server ~]$ pip install keras
   ```


## Usage with R

To install Keras for R with TensorFlow as the backend:

1. Install TensorFlow following [these instructions](link_to_tensorflow_instructions_here). (Please replace `link_to_tensorflow_instructions_here` with the actual link)

2. Follow the instructions in the parent section.

3. Load the necessary modules.

   ```bash
   [name@server ~]$ module load gcc/7.3.0 r/3.5.2
   ```

4. Launch R.

   ```bash
   [name@server ~]$ R
   ```

5. Using `devtools`, install Keras in R.

   ```R
   devtools::install_github('rstudio/keras')
   ```

   Since Keras and TensorFlow are installed in the virtual environment with `pip`, do not use `install_keras()`.

6. To use Keras, activate the virtual environment and run the commands:

   ```R
   library(keras)
   use_virtualenv(Sys.getenv('VIRTUAL_ENV'))
   ```


## References

* [Wikipedia page on Keras](link_to_wikipedia_page_here) (Please replace `link_to_wikipedia_page_here` with the actual link)

