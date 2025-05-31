# TensorFlow

TensorFlow is an open-source machine learning software library.  If you want to run a TensorFlow program on one of our clusters, we recommend that you read the [machine learning tutorial](link-to-tutorial-here).


## Contents

* [Installation](#installation)
    * [The R Package](#the-r-package)
* [Submitting a TensorFlow Task with a GPU](#submitting-a-tensorflow-task-with-a-gpu)
* [Monitoring](#monitoring)
    * [TensorBoard](#tensorboard)
* [Using Multiple GPUs](#using-multiple-gpus)
    * [TensorFlow 1.x](#tensorflow-1x)
        * [Parameter Server](#parameter-server)
        * [Replicated](#replicated)
        * [Benchmarking](#benchmarking)
    * [TensorFlow 2.x](#tensorflow-2x)
        * [Mirrored Strategy](#mirrored-strategy)
            * [Single Node](#single-node)
            * [Multiple Nodes](#multiple-nodes)
        * [Horovod](#horovod)
* [Creating Checkpoints](#creating-checkpoints)
    * [With Keras](#with-keras)
    * [With a Custom Training Loop](#with-a-custom-training-loop)
* [Custom Operators](#custom-operators)
    * [TensorFlow <= 1.4.x](#tensorflow-14x)
    * [TensorFlow > 1.4.x](#tensorflow-14x)
* [Troubleshooting](#troubleshooting)
    * [scikit-image](#scikit-image)
    * [libcupti.so](#libcuptiso)
    * [libiomp5.so invalid ELF header](#libiomp5so-invalid-elf-header)
* [Controlling the Number of CPUs and Threads](#controlling-the-number-of-cpus-and-threads)
    * [TensorFlow 1.x](#tensorflow-1x)
    * [TensorFlow 2.x](#tensorflow-2x)
* [Known Issues](#known-issues)


## Installation

The following instructions install TensorFlow in your home directory using the Python wheels located in `/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/`.  The TensorFlow wheel will be installed in a Python virtual environment using the `pip` command.

### TF 2.x

1. Load the modules required by TensorFlow; in some cases, other modules may be required (e.g., CUDA).
   ```bash
   [name@server ~]$ module load python/3
   ```

2. Create a new Python environment.
   ```bash
   [name@server ~]$ virtualenv --no-download tensorflow
   ```

3. Activate the new environment.
   ```bash
   [name@server ~]$ source tensorflow/bin/activate
   ```

4. Install TensorFlow in your new virtual environment using the following command.
   ```bash
   (tensorflow) [name@server ~]$ pip install --no-index tensorflow==2.8
   ```

### TF 1.x

1. Load the modules required by TensorFlow. TF 1.x requires StdEnv/2018.  Note: TF 1.x is not available on Narval, as this cluster does not offer StdEnv/2018.
   ```bash
   [name@server ~]$ module load StdEnv/2018 python/3
   ```

2. Create a new Python environment.
   ```bash
   [name@server ~]$ virtualenv --no-download tensorflow
   ```

3. Activate the new environment.
   ```bash
   [name@server ~]$ source tensorflow/bin/activate
   ```

4. Install TensorFlow in your new virtual environment using one of the following commands, depending on whether you need to use a GPU. Do not install the `tensorflow` package without the `_cpu` or `_gpu` suffix as there are compatibility issues with other libraries.

    * **CPU only:**
      ```bash
      (tensorflow) [name@server ~]$ pip install --no-index tensorflow_cpu==1.15.0
      ```
    * **GPU:**
      ```bash
      (tensorflow) [name@server ~]$ pip install --no-index tensorflow_gpu==1.15.0
      ```


### The R Package

To use TensorFlow in R, follow the instructions above to create a virtual environment and install TensorFlow within it. Then, proceed as follows:

1. Load the required modules.
   ```bash
   [name@server ~]$ module load gcc r
   ```

2. Activate your Python virtual environment.
   ```bash
   [name@server ~]$ source tensorflow/bin/activate
   ```

3. Launch R.
   ```bash
   (tensorflow) [name@server ~]$ R
   ```

4. In R, install the `devtools` package, then TensorFlow.
   ```r
   install.packages('devtools', repos='https://cloud.r-project.org')
   devtools::install_github('rstudio/tensorflow')
   ```

You can now proceed. Do not call `install_tensorflow()` in R since TensorFlow is already installed in your virtual environment with `pip`. To use TensorFlow as installed in your virtual environment, enter the following commands in R after the environment is activated:

```r
library(tensorflow)
use_virtualenv(Sys.getenv('VIRTUAL_ENV'))
```


## Submitting a TensorFlow Task with a GPU

Submit a TensorFlow task like this:

```bash
[name@server ~]$ sbatch tensorflow-test.sh
```

The script contains:

**File: `tensorflow-test.sh`**

```bash
#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-03:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
module load cuda cudnn
source tensorflow/bin/activate
python ./tensorflow-test.py
```

The Python script reads:

**TF 2.x**

**File: `tensorflow-test.py`**

```python
import tensorflow as tf
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
print(node1, node2)
print(node1 + node2)
```

**TF 1.x**

**File: `tensorflow-test.py`**

```python
import tensorflow as tf
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
print(node1, node2)
sess = tf.Session()
print(sess.run(node1 + node2))
```

Once the task is finished (which should take less than a minute), an output file with a name similar to `cdr116-122907.out` should be generated. The content of this file would be similar to the following; these are example TensorFlow messages and you may have others.


**TF 2.x**

**File: `cdr116-122907.out`**

```
2017-07-10 12:35:19.491097: I tensorflow/core/common_runtime/gpu/gpu_device.cc:961] DMA: 0
2017-07-10 12:35:19.491156: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   Y
2017-07-10 12:35:19.520737: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla P100-PCIE-12GB, pci bus id: 0000:82:00.0)
tf.Tensor(3.0, shape=(), dtype=float32) tf.Tensor(4.0, shape=(), dtype=float32)
tf.Tensor(7.0, shape=(), dtype=float32)
```

**TF 1.x**

**File: `cdr116-122907.out`**

```
2017-07-10 12:35:19.491097: I tensorflow/core/common_runtime/gpu/gpu_device.cc:961] DMA: 0
2017-07-10 12:35:19.491156: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   Y
2017-07-10 12:35:19.520737: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla P100-PCIE-12GB, pci bus id: 0000:82:00.0)
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
7.0
```

TensorFlow works on all types of GPU nodes. For large-scale deep learning or machine learning research, it is strongly recommended to use Cedar's Large GPU nodes. These nodes are equipped with 4 x P100-PCIE-16GB with GPUDirect P2P between each pair. For more information, see [this page](link-to-page-here).


## Monitoring

It is possible to connect to a node on which a task is running to execute processes. This allows you to monitor the resources used by TensorFlow and visualize the progress of the training. For examples, see [Monitoring a Running Task](link-to-monitoring-page-here).


## TensorBoard

TensorFlow offers the TensorBoard visualization suite, which reads TensorFlow events and summary files. To learn how to create these files, see [TensorBoard tutorial on summaries](link-to-tensorboard-tutorial-here).

However, be aware that TensorBoard is too computationally intensive to run on a login node. We recommend running it in the same task as the TensorFlow process. To do this, launch TensorBoard in the background by calling it before the Python script, adding the ampersand (&) character.

```bash
# Your SBATCH arguments here

tensorboard --logdir=/tmp/your_log_dir --host 0.0.0.0 --load_fast false &
python train.py  # example
```

To access TensorBoard with a browser once the task is running, you need to create a link between your computer and the node on which TensorFlow and TensorBoard are running. To do this, you need the hostname of the compute node where the TensorFlow server is located. To find it, display the list of your tasks with the `sq` command and locate the task; the hostname is the value in the NODELIST column.

To create the connection, run the following command on your local computer:

```bash
[name@my_computer ~]$ ssh -N -f -L localhost:6006:computenode:6006 userid@cluster.computecanada.ca
```

Replace `computenode` with the hostname obtained in the previous step; `userid` with your Alliance username; and `cluster` with the hostname of the cluster, i.e., `beluga`, `cedar`, `graham`, etc. If port 6006 was already in use, tensorboard will use another one (e.g., 6007, 6008...).

Once the connection is established, go to `http://localhost:6006`.


## Using Multiple GPUs

### TensorFlow 1.x

There are several methods for managing variables, the most common being Parameter Server and Replicated. We will use [this code](link-to-code-here) to illustrate the various methods; you can adapt it to your specific needs.

#### Parameter Server

The master copy of the variables is stored on a parameter server. In distributed training, the parameter servers are separate processes in each device. At each step, each tower gets a copy of the variables from the parameter server and returns its gradients.

Parameters can be stored on a CPU:

```bash
python tf_cnn_benchmarks.py --variable_update=parameter_server --local_parameter_device=cpu
```

or on a GPU:

```bash
python tf_cnn_benchmarks.py --variable_update=parameter_server --local_parameter_device=gpu
```

#### Replicated

Each GPU has its own copy of the variables. Gradients are copied to all towers by aggregating the contents of the devices or by an all-reduce algorithm (depending on the value of the `all_reduce_spec` parameter).

* With the default all-reduce method:
  ```bash
  python tf_cnn_benchmarks.py --variable_update=replicated
  ```
* Xring --- use a global ring reduction for all tensors:
  ```bash
  python tf_cnn_benchmarks.py --variable_update=replicated --all_reduce_spec=xring
  ```
* Pscpu --- use CPU at worker 0 to reduce all tensors:
  ```bash
  python tf_cnn_benchmarks.py --variable_update=replicated --all_reduce_spec=pscpu
  ```
* NCCL --- use NCCL to locally reduce all tensors:
  ```bash
  python tf_cnn_benchmarks.py --variable_update=replicated --all_reduce_spec=nccl
  ```

The methods behave differently depending on the models; we strongly recommend testing your models with all methods on different types of GPU nodes.


#### Benchmarking

Results were obtained with TensorFlow v1.5 (CUDA9 and cuDNN 7) on Graham and Cedar with single and multiple GPUs and different variable management methods; see [TensorFlow Benchmarks](link-to-benchmarks-here).

**ResNet-50**

Batches of 32 per GPU and data parallelism (results are in images per second).

| Node Type             | 1 GPU | # GPUs | ps,cpu | ps,gpu | replicated | replicated, xring | replicated, pscpu | replicated, nccl |
|-----------------------|-------|--------|--------|--------|-------------|-------------------|-------------------|-------------------|
| Graham, GPU base      | 171.23 | 2      | 93.31  | 324.04 | 318.33      | 316.01            | 109.82            | 315.99            |
| Cedar, GPU Base       | 172.99 | 4      | 662.65 | 595.43 | 616.02      | 490.03            | 645.04            | 608.95            |
| Cedar, GPU Large      | 205.71 | 4      | 673.47 | 721.98 | 754.35      | 574.91            | 664.72            | 692.25            |

**VGG-16**

Batches of 32 per GPU and data parallelism (results are in images per second).

| Node Type             | 1 GPU | # GPUs | ps,cpu | ps,gpu | replicated | replicated, xring | replicated, pscpu | replicated, nccl |
|-----------------------|-------|--------|--------|--------|-------------|-------------------|-------------------|-------------------|
| Graham, GPU base      | 115.89 | 2      | 91.29  | 194.46 | 194.43      | 203.83            | 132.19            | 219.72            |
| Cedar, GPU Base       | 114.77 | 4      | 232.85 | 280.69 | 274.41      | 341.29            | 330.04            | 388.53            |
| Cedar, GPU Large      | 137.16 | 4      | 175.20 | 379.80 | 336.72      | 417.46            | 225.37            | 490.52            |


### TensorFlow 2.x

Similar to TensorFlow 1.x, TensorFlow 2.x offers different strategies for using multiple GPUs with the high-level `tf.distribute` API. In the following sections, we show code examples for each strategy with Keras. For more information, see the [official TensorFlow documentation](link-to-tensorflow-docs-here).


#### Mirrored Strategy

##### Single Node

**File: `tensorflow-singleworker.sh`**

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:4
#SBATCH --mem=8G
#SBATCH --time=0-00:30
#SBATCH --output=%N-%j.out
module load python/3
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index tensorflow
export NCCL_BLOCKING_WAIT=1
#Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
python tensorflow-singleworker.py
```

**File: `tensorflow-singleworker.py`**

```python
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, tensorflow MirroredStrategy test')
parser.add_argument('--lr', default=0.001, help='')
parser.add_argument('--batch_size', type=int, default=256, help='')
args = parser.parse_args()

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.SGD(learning_rate=args.lr),
                  metrics=['accuracy'])

    # This next line will attempt to download the CIFAR10 dataset from the internet if you don't already have it stored in ~/.keras/datasets.
    # Run this line on a login node prior to submitting your job, or manually download the data from
    # https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz, rename to "cifar-10-batches-py.tar.gz" and place it under ~/.keras/datasets
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
    model.fit(dataset, epochs=2)
```

##### Multiple Nodes

The syntax for using distributed GPUs across multiple nodes is very similar to the single-node case; the main difference is the use of `MultiWorkerMirroredStrategy()`. Here, we use `SlurmClusterResolver()` to tell TensorFlow to get the task information from Slurm instead of manually assigning a chief node and worker nodes, for example. We also need to add `CommunicationImplementation.NCCL` to the distribution strategy to indicate that we want to use NVIDIA's NCCL library for inter-GPU communication. This was not necessarily the case for a single node since NCCL is the default with `MirroredStrategy()`.

**File: `tensorflow-multiworker.sh`**

```bash
#!/bin/bash
#SBATCH --nodes 2              # Request 2 nodes so all resources are in two nodes.
#SBATCH --gres=gpu:2          # Request 2 GPU "generic resources”. You will get 2 per node.
#SBATCH --ntasks-per-node=2   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter if your input pipeline can handle parallel data-loading/data-transforms
#SBATCH --mem=8G
#SBATCH --time=0-00:30
#SBATCH --output=%N-%j.out
srun -N $SLURM_NNODES -n $SLURM_NNODES config_env.sh

module load gcc/9.3.0 cuda/11.8
export NCCL_BLOCKING_WAIT=1
#Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
srun launch_training.sh
```

**File: `config_env.sh`**

```bash
#!/bin/bash
module load python

virtualenv --no-download $SLURM_TMPDIR/ENV
source $SLURM_TMPDIR/ENV/bin/activate

pip install --upgrade pip --no-index

pip install --no-index tensorflow
echo "Done installing virtualenv!"
```

**File: `launch_training.sh`**

```bash
#!/bin/bash
source $SLURM_TMPDIR/ENV/bin/activate

python tensorflow-multiworker.py
```

**File: `tensorflow-multiworker.py`**

```python
import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, tensorflow MultiWorkerMirrored test')
parser.add_argument('--lr', default=0.001, help='')
parser.add_argument('--batch_size', type=int, default=256, help='')
args = parser.parse_args()

cluster_config = tf.distribute.cluster_resolver.SlurmClusterResolver()
comm_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    cluster_resolver=cluster_config, communication_options=comm_options)
with strategy.scope():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.SGD(learning_rate=args.lr),
                  metrics=['accuracy'])

    # This next line will attempt to download the CIFAR10 dataset from the internet if you don't already have it stored in ~/.keras/datasets.
    # Run this line on a login node prior to submitting your job, or manually download the data from
    # https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz, rename to "cifar-10-batches-py.tar.gz" and place it under ~/.keras/datasets
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
    model.fit(dataset, epochs=2)
```


#### Horovod

Horovod is a distributed deep learning library for TensorFlow, Keras, PyTorch, and Apache MXNet. We will repeat the same tutorial as above, this time using Horovod.

**File: `tensorflow-horovod.sh`**

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:2          # Request 2 GPU "generic resources”. You will get 2 per node.
#SBATCH --ntasks-per-node=2    # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter if your input pipeline can handle parallel data-loading/data-transforms
#SBATCH --mem=8G
#SBATCH --time=0-00:30
#SBATCH --output=%N-%j.out
module load StdEnv/2020
module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index tensorflow==2.5.0 horovod
export NCCL_BLOCKING_WAIT=1
#Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
srun python tensorflow-horovod.py
```

**File: `tensorflow-horovod.py`**

```python
import tensorflow as tf
import numpy as np
import horovod.tensorflow.keras as hvd
import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, tensorflow horovod test')
parser.add_argument('--lr', default=0.001, help='')
parser.add_argument('--batch_size', type=int, default=256, help='')
args = parser.parse_args()

hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10))

optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
optimizer = hvd.DistributedOptimizer(optimizer)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=optimizer,
              metrics=['accuracy'])

callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),]

# This next line will attempt to download the CIFAR10 dataset from the internet if you don't already have it stored in ~/.keras/datasets.
# Run this line on a login node prior to submitting your job, or manually download the data from
# https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz, rename to "cifar-10-batches-py.tar.gz" and place it under ~/.keras/datasets
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
model.fit(dataset, epochs=2, callbacks=callbacks, verbose=2) # verbose=2 to avoid printing a progress bar to *.out files.
```


## Creating Checkpoints

Regardless of how long your code takes to run, a good habit to adopt is to create checkpoints during training. A checkpoint gives you a snapshot of your model at a specific point in the training process (after a certain number of iterations or epochs); the snapshot is saved to disk and you can retrieve it later. This is useful for breaking down a long-running task into smaller tasks, which may be allocated to a cluster more quickly. It's also a good way to avoid losing your work in case of unexpected errors or hardware failure.


### With Keras

To create a checkpoint in a Keras training, we recommend the `callbacks` parameter of the `model.fit()` method. In the following example, we ask TensorFlow to create a checkpoint at the end of each training epoch.

```python
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath="./ckpt",save_freq="epoch")] # Make sure the path where you want to create the checkpoint exists

model.fit(dataset, epochs=10 , callbacks=callbacks)
```

For more information, see the [official TensorFlow documentation](link-to-tensorflow-docs-here).


### With a Custom Training Loop

See the [official TensorFlow documentation](link-to-tensorflow-docs-here).


## Custom Operators

As part of your research, you may need to use [code to take advantage of custom operators](link-to-custom-operators-here) that are not included in the TensorFlow distributions, or even want to [create your own custom operators](link-to-creating-custom-operators-here). In both cases, your custom operators must be compiled before you submit the task. Follow the steps below.

First, create a Python virtual environment and install a version of TensorFlow compatible with your custom operators. Then, go to the directory containing the source code of the operators and use the following commands depending on the version you have installed.


### TensorFlow <= 1.4.x

If your custom operator can support a GPU:

```bash
[name@server ~]$ module load cuda/<version>
[name@server ~]$ nvcc <operator>.cu -o <operator>.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
[name@server ~]$ g++ -std=c++11 <operator>.cpp <operator>.cu.o -o <operator>.so -shared -fPIC -I /<path to python virtual env>/lib/python<version>/site-packages/tensorflow/include -I/<path to python virtual env>/lib/python<version>/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-<version>/include -lcudart -L /usr/local/cuda-<version>/lib64/
```

If your custom operator cannot support a GPU:

```bash
[name@server ~]$ g++ -std=c++11 <operator>.cpp -o <operator>.so -shared -fPIC -I /<path to python virtual env>/lib/python<version>/site-packages/tensorflow/include -I/<path to python virtual env>/lib/python<version>/site-packages/tensorflow/include/external/nsync/public
```


### TensorFlow > 1.4.x

If your custom operator can support a GPU:

```bash
[name@server ~]$ module load cuda/<version>
[name@server ~]$ nvcc <operator>.cu -o <operator>.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
[name@server ~]$ g++ -std=c++11 <operator>.cpp <operator>.cu.o -o <operator>.so -shared -fPIC -I /<path to python virtual env>/lib/python<version>/site-packages/tensorflow/include -I /usr/local/cuda-<version>/include -I /<path to python virtual env>/lib/python<version>/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-<version>/lib64/ -L /<path to python virtual env>/lib/python<version>/site-packages/tensorflow -ltensorflow_framework
```

If your custom operator cannot support a GPU:

```bash
[name@server