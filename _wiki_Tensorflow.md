# TensorFlow

TensorFlow is an open-source software library for Machine Intelligence.

If you are porting a TensorFlow program to an Alliance cluster, you should follow our [tutorial on machine learning](link-to-tutorial).


## Contents

* [Installing TensorFlow](#installing-tensorflow)
    * [R package](#r-package)
* [Submitting a TensorFlow job with a GPU](#submitting-a-tensorflow-job-with-a-gpu)
* [Monitoring](#monitoring)
    * [TensorBoard](#tensorboard)
* [TensorFlow with multi-GPUs](#tensorflow-with-multi-gpus)
    * [TensorFlow 1.x](#tensorflow-1x)
        * [Parameter server](#parameter-server)
        * [Replicated](#replicated)
        * [Benchmarks](#benchmarks)
    * [TensorFlow 2.x](#tensorflow-2x)
        * [Mirrored strategy](#mirrored-strategy)
            * [Single node](#single-node)
            * [Multiple nodes](#multiple-nodes)
        * [Horovod](#horovod)
* [Creating model checkpoints](#creating-model-checkpoints)
    * [With Keras](#with-keras)
    * [With a custom training loop](#with-a-custom-training-loop)
* [Custom TensorFlow operators](#custom-tensorflow-operators)
    * [TensorFlow <= 1.4.x](#tensorflow-14x)
    * [TensorFlow > 1.4.x](#tensorflow-14x)
* [Troubleshooting](#troubleshooting)
    * [scikit image](#scikit-image)
    * [libcupti.so](#libcuptiso)
    * [libiomp5.so invalid ELF header](#libiomp5so-invalid-elf-header)
* [Controlling the number of CPUs and threads](#controlling-the-number-of-cpus-and-threads)
    * [TensorFlow 1.x](#tensorflow-1x)
    * [TensorFlow 2.x](#tensorflow-2x)
* [Known issues](#known-issues)


## Installing TensorFlow

These instructions install TensorFlow in your `/home` directory using Alliance's prebuilt Python wheels. Custom Python wheels are stored in `/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/`. To install a TensorFlow wheel, we will use the `pip` command and install it into a Python virtual environment.

### TF 2.x

1. Load modules required by TensorFlow. In some cases, other modules may be required (e.g., CUDA).
   ```bash
   module load python/3
   ```

2. Create a new Python virtual environment.
   ```bash
   virtualenv --no-download tensorflow
   ```

3. Activate your newly created Python virtual environment.
   ```bash
   source tensorflow/bin/activate
   ```

4. Install TensorFlow in your newly created virtual environment using the following command.
   ```bash
   pip install --no-index tensorflow==2.8
   ```

### TF 1.x

1. Load modules required by TensorFlow. TF 1.x requires `StdEnv/2018`.

   Note: TF 1.x is not available on Narval, since `StdEnv/2018` is not available on this cluster.
   ```bash
   module load StdEnv/2018 python/3
   ```

2. Create a new Python virtual environment.
   ```bash
   virtualenv --no-download tensorflow
   ```

3. Activate your newly created Python virtual environment.
   ```bash
   source tensorflow/bin/activate
   ```

4. Install TensorFlow in your newly created virtual environment using one of the commands below, depending on whether you need to use a GPU. Do not install the `tensorflow` package (without the `_cpu` or `_gpu` suffixes) as it has compatibility issues with other libraries.

   **CPU-only:**
   ```bash
   pip install --no-index tensorflow_cpu==1.15.0
   ```

   **GPU:**
   ```bash
   pip install --no-index tensorflow_gpu==1.15.0
   ```

### R package

To use TensorFlow in R, you will need to first follow the preceding instructions on creating a virtual environment and installing TensorFlow in it. Once this is done, follow these instructions.

1. Load the required modules.
   ```bash
   module load gcc r
   ```

2. Activate your Python virtual environment.
   ```bash
   source tensorflow/bin/activate
   ```

3. Launch R.
   ```bash
   R
   ```

4. In R, install package `devtools`, then `tensorflow`:
   ```R
   install.packages('devtools', repos='https://cloud.r-project.org')
   devtools::install_github('rstudio/tensorflow')
   ```

You are then good to go. Do not call `install_tensorflow()` in R, as TensorFlow has already been installed in your virtual environment with `pip`. To use the TensorFlow installed in your virtual environment, enter the following commands in R after the environment has been activated.

```R
library(tensorflow)
use_virtualenv(Sys.getenv('VIRTUAL_ENV'))
```

## Submitting a TensorFlow job with a GPU

Once you have the above setup completed, you can submit a TensorFlow job.

```bash
sbatch tensorflow-test.sh
```

The job submission script contains:

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

The Python script has the form:

### TF 2.x

**File: `tensorflow-test.py`**

```python
import tensorflow as tf
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
print(node1, node2)
print(node1 + node2)
```

### TF 1.x

**File: `tensorflow-test.py`**

```python
import tensorflow as tf
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
print(node1, node2)
sess = tf.Session()
print(sess.run(node1 + node2))
```

Once the job has completed (should take less than a minute), you should see an output file called something like `cdr116-122907.out` with contents similar to the following (the logged messages from TensorFlow are only examples, expect different messages and more messages):

### TF 2.x

**File: `cdr116-122907.out`**

```
2017-07-10 12:35:19.491097: I tensorflow/core/common_runtime/gpu/gpu_device.cc:961] DMA: 0
2017-07-10 12:35:19.491156: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   Y
2017-07-10 12:35:19.520737: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla P100-PCIE-12GB, pci bus id: 0000:82:00.0)
tf.Tensor(3.0, shape=(), dtype=float32) tf.Tensor(4.0, shape=(), dtype=float32)
tf.Tensor(7.0, shape=(), dtype=float32)
```

### TF 1.x

**File: `cdr116-122907.out`**

```
2017-07-10 12:35:19.491097: I tensorflow/core/common_runtime/gpu/gpu_device.cc:961] DMA: 0
2017-07-10 12:35:19.491156: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   Y
2017-07-10 12:35:19.520737: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla P100-PCIE-12GB, pci bus id: 0000:82:00.0)
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
7.0
```

TensorFlow can run on all GPU node types. Cedar's GPU large node type, which is equipped with 4 x P100-PCIE-16GB with GPUDirect P2P enabled between each pair, is highly recommended for large-scale deep learning or machine learning research. See [Using GPUs with SLURM](link-to-gpu-slurm-doc) for more information.


## Monitoring

It is possible to connect to the node running a job and execute processes. This can be used to monitor resources used by TensorFlow and to visualize the progress of the training. See [Attaching to a running job](link-to-attaching-job) for examples.


### TensorBoard

TensorFlow comes with a suite of visualization tools called TensorBoard. TensorBoard operates by reading TensorFlow events and model files. To know how to create these files, read [TensorBoard tutorial on summaries](link-to-tensorboard-tutorial).

TensorBoard requires too much processing power to be run on a login node. Users are strongly encouraged to execute it in the same job as the Tensorflow process. To do so, launch TensorBoard in the background by calling it before your python script, and appending an ampersand (`&`) to the call:

```bash
# Your SBATCH arguments here

tensorboard --logdir=/tmp/your_log_dir --host 0.0.0.0 --load_fast false &
python train.py  # example
```

Once the job is running, to access TensorBoard with a web browser, you need to create a connection between your computer and the compute node running TensorFlow and TensorBoard. To do this you first need the hostname of the compute node running the Tensorboard server. Show the list of your jobs using the command `sq`; find the job, and note the value in the "NODELIST" column (this is the hostname).

To create the connection, use the following command on your local computer:

```bash
ssh -N -f -L localhost:6006:computenode:6006 userid@cluster.computecanada.ca
```

Replace `computenode` with the node hostname you retrieved from the preceding step, `userid` by your Alliance username, `cluster` by the cluster hostname (i.e.: `beluga`, `cedar`, `graham`, etc.). If port 6006 was already in use, tensorboard will be using another one (e.g., 6007, 6008...).

Once the connection is created, go to `http://localhost:6006`.


## TensorFlow with multi-GPUs

### TensorFlow 1.x

TensorFlow provides different methods of managing variables when training models on multiple GPUs. "Parameter Server" and "Replicated" are the most two common methods.

In this section, TensorFlow Benchmarks code will be used as an example to explain the different methods. Users can reference the TensorFlow Benchmarks code to implement their own.


#### Parameter server

Variables are stored on a parameter server that holds the master copy of the variable. In distributed training, the parameter servers are separate processes in the different devices. For each step, each tower gets a copy of the variables from the parameter server, and sends its gradients to the param server.

Parameters can be stored in a CPU:

```bash
python tf_cnn_benchmarks.py --variable_update=parameter_server --local_parameter_device=cpu
```

or a GPU:

```bash
python tf_cnn_benchmarks.py --variable_update=parameter_server --local_parameter_device=gpu
```

#### Replicated

With this method, each GPU has its own copy of the variables. To apply gradients, an all_reduce algorithm or or regular cross-device aggregation is used to replicate the combined gradients to all towers (depending on the `all_reduce_spec` parameter's setting).

* **All reduce method (default):**
   ```bash
   python tf_cnn_benchmarks.py --variable_update=replicated
   ```

* **Xring:** use one global ring reduction for all tensors:
   ```bash
   python tf_cnn_benchmarks.py --variable_update=replicated --all_reduce_spec=xring
   ```

* **Pscpu:** use CPU at worker 0 to reduce all tensors:
   ```bash
   python tf_cnn_benchmarks.py --variable_update=replicated --all_reduce_spec=pscpu
   ```

* **NCCL:** use NCCL to locally reduce all tensors:
   ```bash
   python tf_cnn_benchmarks.py --variable_update=replicated --all_reduce_spec=nccl
   ```

Different variable managing methods perform differently with different models. Users are highly recommended to test their own models with all methods on different types of GPU node.


#### Benchmarks

This section will give ResNet-50 and VGG-16 benchmarking results on both Graham and Cedar with single and multiple GPUs using different methods for managing variables. TensorFlow v1.5 (built with CUDA 9 and cuDNN 7) is used. The benchmark can be found on github at [TensorFlow Benchmarks](link-to-benchmarks).

**ResNet-50**

Batch size is 32 per GPU. Data parallelism is used. (Results in "images per second")

| Node type             | 1 GPU | Number of GPUs | ps,cpu | ps, gpu | replicated | replicated, xring | replicated, pscpu | replicated, nccl |
|-----------------------|-------|-----------------|--------|---------|-------------|-------------------|-------------------|-------------------|
| Graham  GPU node      | 171.23 | 2               | 93.31  | 324.04  | 318.33       | 316.01            | 109.82            | 315.99            |
| Cedar GPU Base        | 172.99 | 4               | 662.65 | 595.43  | 616.02       | 490.03            | 645.04            | 608.95            |
| Cedar GPU Large       | 205.71 | 4               | 673.47 | 721.98  | 754.35       | 574.91            | 664.72            | 692.25            |


**VGG-16**

Batch size is 32 per GPU. Data parallelism is used. (Results in images per second)

| Node type             | 1 GPU | Number of GPUs | ps,cpu | ps, gpu | replicated | replicated, xring | replicated, pscpu | replicated, nccl |
|-----------------------|-------|-----------------|--------|---------|-------------|-------------------|-------------------|-------------------|
| Graham  GPU node      | 115.89 | 2               | 91.29  | 194.46  | 194.43       | 203.83            | 132.19            | 219.72            |
| Cedar GPU Base        | 114.77 | 4               | 232.85 | 280.69  | 274.41       | 341.29            | 330.04            | 388.53            |
| Cedar GPU Large       | 137.16 | 4               | 175.20 | 379.80  | 336.72       | 417.46            | 225.37            | 490.52            |


### TensorFlow 2.x

Much like TensorFlow 1.x, TensorFlow 2.x offers a number of different strategies to make use of multiple GPUs through the high-level API `tf.distribute`. In the following sections, we provide code examples of each strategy using Keras for simplicity. For more details, please refer to the official [TensorFlow documentation](link-to-tf-docs).


#### Mirrored strategy

##### Single node

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

The Python script `tensorflow-singleworker.py` has the form:

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

    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
    model.fit(dataset, epochs=2)
```

##### Multiple nodes

The syntax to use multiple GPUs distributed across multiple nodes is very similar to the single node case, the most notable difference being the use of `MultiWorkerMirroredStrategy()`. Here, we use `SlurmClusterResolver()` to tell TensorFlow to acquire all the necessary job information from SLURM, instead of manually assigning master and worker nodes, for example. We also need to add `CommunicationImplementation.NCCL` to the distribution strategy to specify that we want to use Nvidia's NCCL backend for inter-GPU communications. This was not necessary in the single-node case, as NCCL is the default backend with `MirroredStrategy()`.

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

Where `config_env.sh` has the form:

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

The script `launch_training.sh` has the form:

**File: `launch_training.sh`**

```bash
#!/bin/bash
source $SLURM_TMPDIR/ENV/bin/activate

python tensorflow-multiworker.py
```

And the Python script `tensorflow-multiworker.py` has the form:

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

    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
    model.fit(dataset, epochs=2)
```

#### Horovod

[Horovod](link-to-horovod) is a distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. The following is the same tutorial from above reimplemented using Horovod:

**File: `tensorflow-horovod.sh`**

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:2          # Request 2 GPU "generic resources”. You will get 2 per node.
#SBATCH --ntasks-per-node=2   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter if your input pipeline can handle parallel data-loading/data-transforms
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

(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
model.fit(dataset, epochs=2, callbacks=callbacks, verbose=2) # verbose=2 to avoid printing a progress bar to *.out files.
```


## Creating model checkpoints

Whether or not you expect your code to run for long time periods, it is a good habit to create Checkpoints during training. A checkpoint is a snapshot of your model at a given point during the training process (after a certain number of iterations or after a number of epochs) that is saved to disk and can be loaded at a later time. It is a handy way of breaking jobs that are expected to run for a very long time, into multiple shorter jobs that may get allocated on the cluster more quickly. It is also a good way of avoiding losing progress in case of unexpected errors in your code or node failures.


### With Keras

To create a checkpoint when training with `keras`, we recommend using the `callbacks` parameter of the `model.fit()` method. The following example shows how to instruct TensorFlow to create a checkpoint at the end of every training epoch:

```python
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath="./ckpt",save_freq="epoch")] # Make sure the path where you want to create the checkpoint exists

model.fit(dataset, epochs=10 , callbacks=callbacks)
```

For more information, please refer to the [official TensorFlow documentation](link-to-tf-docs).


### With a custom training loop

Please refer to the [official TensorFlow documentation](link-to-tf-docs).


## Custom TensorFlow operators

In your research, you may come across code that leverages custom operators that are not part of the core tensorflow distribution, or you might want to create your own. In both cases, you will need to compile your custom operators before submitting your job. To ensure your code will run correctly, follow the steps below.

First, create a Python virtual environment and install a tensorflow version compatible with your custom operators. Then go to the directory containing the operators source code and follow the next steps according to the version you installed:


### TensorFlow <= 1.4.x

If your custom operator is GPU-enabled:

```bash
module load cuda/<version>
nvcc <operator>.cu -o <operator>.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 <operator>.cpp <operator>.cu.o -o <operator>.so -shared -fPIC -I /<path to python virtual env>/lib/python<version>/site-packages/tensorflow/include -I/<path to python virtual env>/lib/python<version>/site-packages/tensorflow/include/external/nsync/public -I /usr/local/cuda-<version>/include -lcudart -L /usr/local/cuda-<version>/lib64/
```

If your custom operator is not GPU-enabled:

```bash
g++ -std=c++11 <operator>.cpp -o <operator>.so -shared -fPIC -I /<path to python virtual env>/lib/python<version>/site-packages/tensorflow/include -I/<path to python virtual env>/lib/python<version>/site-packages/tensorflow/include/external/nsync/public
```


### TensorFlow > 1.4.x

If your custom operator is GPU-enabled:

```bash
module load cuda/<version>
nvcc <operator>.cu -o <operator>.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 <operator>.cpp <operator>.cu.o -o <operator>.so -shared -fPIC -I /<path to python virtual env>/lib/python<version>/site-packages/tensorflow/include -I /usr/local/cuda-<version>/include -I /<path to python virtual env>/lib/python<version>/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-<version>/lib64/ -L /<path to python virtual env>/lib/python<version>/site-packages/tensorflow -ltensorflow_framework
```

If your custom operator is not GPU-enabled:

```bash
g++ -std=c++11 <operator>.cpp -o <operator>.so -shared -fPIC -I /<path to python virtual env>/lib/python<version>/site-packages/tensorflow/include -I /<path to python virtual env>/lib/python<version>/site-packages/tensorflow/include/external/nsync/public -L /<path to python virtual env>/lib/python<version>/site-packages/tensorflow -ltensorflow_framework
```


## Troubleshooting

### scikit image

If you are using the scikit-image library, you may get the following error:

```
OMP: Error #15: Initializing libiomp5.so, but found libiomp5.so already initialized.
```

This is because the tensorflow library tries to load a bundled version of OMP which conflicts with the system version. The workaround is as follows:

```bash
(tf_skimage_venv) name@server$ cd tf_skimage_venv
(tf_skimage_venv) name@server$ export LIBIOMP_PATH=$(strace python -c