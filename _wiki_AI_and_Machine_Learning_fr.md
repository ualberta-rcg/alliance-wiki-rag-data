# AI and Machine Learning

To get the most out of your machine learning applications, you need to be aware of some specifics of our clusters. These machines are far more complex than the local computer you use for prototyping. Among other things, a cluster has distributed file systems that transparently switch between different storage types. Although accessing a file in `/project` may feel the same as if it were located on the current node, under the hood the performance implications are very different.  It is therefore important to read the section [Managing your datasets](#managing-your-datasets) below.

This page describes best practices for using the clusters, as well as references to useful information.


## Tutorials

SHARCNET offers a self-paced training tutorial; click on [Introduction to Machine Learning](link-to-tutorial).

If your program is ready to run on our clusters, see [our tutorial](link-to-tutorial).

See also this [user-prepared tutorial](link-to-tutorial), which describes the steps to set up your environment and the Alliance's with Python.


## Python

Python is a popular software in machine learning.  Refer to [our wiki page](link-to-wiki-page) for important information on versions, virtual environments, connecting and computing nodes, `multiprocessing`, Anaconda, Jupyter, etc.


### Avoiding Anaconda

We recommend using `virtualenv` to avoid the following problems caused by Anaconda and discussed on [this page](link-to-page).

In most cases, switching to `virtualenv` is easy. You just need to install the same packages, except for CUDA, CuDNN and other low-level libraries that are already on our clusters.


## Information on Available Software Packages

For information on installation and frequent problems, see the wiki page for each of the following packages:

* TensorFlow
* PyTorch
* Keras
* Torch
* SpaCy
* XGBoost
* Scikit-Learn
* SnapML


## Managing your datasets

### Storage and File Management

Research needs are diverse; therefore, we offer several solutions ranging from high-speed temporary local storage to long-term storage on different media. For more information, see [Storage and File Management](link-to-storage-page).


### Choosing the Storage Type According to the Size of Your Dataset

* If your dataset is about 10GB or less, it will probably fit in memory, depending on the amount of memory your job has. Your machine learning jobs should not read data from disk.

* If your dataset is about 100GB or less, it fits in the local storage space of the compute node; transfer it to this space at the beginning of the job since it is much faster and more reliable than the shared spaces `/home`, `/project`, and `/scratch`. For each job, a temporary directory is available at `$SLURM_TMPDIR`; see the example in [our tutorial](link-to-tutorial). However, be aware that a job from another user can fully occupy the node's storage space and leave you no room (we are looking for a solution to this problem); however, if it's your lucky day, you might have a terabyte just for yourself.

* If your dataset is larger, you may need to leave it in a shared space. You can permanently store data in your `/project` space; the `/scratch` space is sometimes faster, but is not designed for permanent storage. All shared storage spaces (`/home`, `/project`, and `/scratch`) are used to read and store data at a low frequency (e.g., 1 large block per 10 seconds rather than 10 small blocks per second).


### Datasets Composed of Several Small Files

In machine learning, it is common to have datasets composed of hundreds or even thousands of files, for example in the case of image datasets. Each file can be small, often less than a few hundred kilobytes, and in these cases, some problems can occur:

* The file system imposes a quota that restricts the number of files.
* The application could be significantly slowed down by transferring files from `/project` or `/scratch` to a compute node.

With a distributed file system, the data should be gathered into a single archive file; see [Working with a large number of files](link-to-large-files-page).


## Long Calculations

If your calculations are time-consuming, it is recommended to use checkpoints; for example, instead of three days of training, you could have three 24-hour blocks. This way, your work would not be lost in case of failure and you could benefit from better prioritization of your jobs since more nodes are reserved for short jobs.

Your favorite library probably supports checkpoints; see the typical case presented in [our tutorial](link-to-tutorial). If your program does not allow it, see the [generic solution](link-to-generic-solution).

See other examples in:

* [PyTorch Checkpoints](link-to-pytorch-checkpoints)
* [TensorFlow Checkpoints](link-to-tensorflow-checkpoints)


## Running Multiple Similar Tasks

In any of the following cases:

* Hyperparameter search
* Training several variants of the same method
* Running several optimization processes of the same duration

You should group several tasks into one using a tool like `META`, `GLOST`, or `GNU Parallel`.


## Experiment Tracking and Hyperparameter Optimization

Weights & Biases (wandb) and Comet.ml can help you optimize your compute allocation by:

* Facilitating the tracking and analysis of learning processes.
* Allowing Bayesian hyperparameter optimization.

Comet and Wandb are not currently available on Graham.


## Large-Scale Machine Learning (Big Data)

Modern deep learning packages like PyTorch and TensorFlow offer utilities for native large-scale learning work and tutorials are numerous. A topic that is little discussed, however, is the scalability of classical machine learning methods (and not deep learning) for working with large datasets; on this subject, see the wiki page [Large-Scale Machine Learning (Big Data)](link-to-big-data-page).


## Troubleshooting

### Determinism in Recurrent Neural Networks with CUDA

When the cuDNN library is present in CUDA Toolkit versions 10.2 and later, non-deterministic behavior may be observed in recurrent neural networks (RNNs) and calls to the multi-head self-attention API.

To avoid this problem, you can configure the environment variable `CUBLAS_WORKSPACE_CONFIG` with a single size for the buffer, for example `:16:8` or `:4096:2`.  This way, cuBLAS fixes the GPU memory to 8 buffers of 16KB each or 2 buffers of 4MB each.


**(Remember to replace the bracketed `link-to-…` placeholders with the actual links.)**
