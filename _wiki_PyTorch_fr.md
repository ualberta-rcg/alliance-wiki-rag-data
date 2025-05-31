# PyTorch

PyTorch is a Python package offering two high-level features:

*   Tensor computation (similar to NumPy) with strong GPU acceleration.
*   Deep learning neural networks within a tape-based autograd system.

If you want to run a PyTorch program on one of our clusters, it's recommended to read this [tutorial](link-to-tutorial-here).


## Clarification

There is some similarity between PyTorch and Torch, but for practical purposes, you can consider them different projects.  PyTorch developers also offer LibTorch, which allows implementing PyTorch extensions using C++ and implementing pure C++ machine learning applications. Python models written with PyTorch can be converted and used in C++ with TorchScript.


## Installation

### Recently Added Wheels

To find the latest PyTorch version, use:

```bash
[name@server ~]$ avail_wheels torch
```

For more information, see [Available Wheels](link-to-available-wheels-here).


### Installing the Wheel

The best option is to install using Python wheels as follows:

1.  Load a Python module with `module load python`.
2.  Create and start a virtual environment.
3.  Install PyTorch in the virtual environment with `pip install`.

#### GPU and CPU

```bash
(venv)[name@server ~]$ pip install --no-index torch
```

With H100 GPUs, torch 2.3 and higher is required.

**Note:** PyTorch 1.10 has known issues on our clusters (except Narval). If distributed training produces errors or you get an error including `c10::Error`, we recommend installing PyTorch 1.9.1 with:

```bash
pip install --no-index torch==1.9.1
```

#### Additional Packages

In addition to `torch`, you can also install `torchvision`, `torchtext`, and `torchaudio`.

```bash
(venv)[name@server ~]$ pip install --no-index torch torchvision torchtext torchaudio
```


## Submitting a Job

The following script is an example of submitting a job using the Python wheel with a virtual environment.

**File: pytorch-test.sh**

```bash
#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out
module load python/<select version> # Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch --no-index

python pytorch-test.py
```

The Python script `pytorch-ddp-test.py` has the following form:

**File: pytorch-test.py**

```python
import torch
x = torch.Tensor(5, 3)
print(x)
y = torch.rand(5, 3)
print(y)
# let us run the following only if CUDA is available
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x + y)
```

You can then submit a PyTorch job with:

```bash
[name@server ~]$ sbatch pytorch-test.sh
```


## High Performance

### TF32: Performance vs. Precision

With version 1.7.0, PyTorch added support for Nvidia's TensorFloat-32 (TF32) mode, available only for Ampere and newer GPU architectures.  In versions 1.7.x to 1.11.x, this mode (enabled by default) made tensor operations up to 20x faster than equivalent single-precision (FP32) operations. However, this performance gain could reduce the precision of the results, problematic for deep learning models using ill-conditioned matrices or long sequences of tensor operations. Following user feedback, TF32 is disabled by default for matrix multiplications and enabled by default for convolutions starting from version 1.12.0.

As of October 2022, our only cluster offering Ampere GPUs is Narval. When using PyTorch on Narval:

* You might notice a significant slowdown in GPU execution of the same code with `torch < 1.12.0` and `torch >= 1.12.0`.
* You might get different results in GPU execution of the same code with `torch < 1.12.0` and `torch >= 1.12.0`.

To enable or disable TF32 for `torch >= 1.12.0`, set the following flags to `True` or `False`:

```python
torch.backends.cuda.matmul.allow_tf32 = False # Enable/disable TF32 for matrix multiplications
torch.backends.cudnn.allow_tf32 = False # Enable/disable TF32 for convolutions
```

For more information, see [this section of the PyTorch documentation](link-to-pytorch-doc-here).


### Working with Multiple CPUs

By default, PyTorch enables multi-CPU parallelism in two ways:

*   **Intra-op:** Parallel implementation of frequently used deep learning operators like matrix multiplication or convolution, using OpenMP directly or lower-level libraries like MKL and OneDNN. When PyTorch code needs to perform such operations, it automatically uses multiple threads across all available CPU cores.
*   **Inter-op:** Ability to execute different parts of the code concurrently. This usually requires the program to be designed to execute multiple parts in parallel, for example, using the just-in-time compiler `torch.jit` to execute asynchronous tasks in a TorchScript program.

For small models, we strongly recommend using multiple CPUs instead of a GPU. Training will likely be faster with a GPU (except for very small models), but if the model and dataset are not large enough, the speedup from the GPU might not be significant, and the job will only use a small fraction of its computing capacity. This might not be a problem on your own computer, but in a shared environment like our clusters, you would be blocking a resource that could be used for larger-scale computations by another project.  Furthermore, using a GPU would contribute to the decrease of your group's allocation and impact the priority given to your colleagues' jobs.

The following code has several opportunities for intra-op parallelism. By requesting more CPUs without changing the code, you can observe the performance impact.

**File: pytorch-multi-cpu.sh**

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1 # change this parameter to 2,4,6,... to see the effect on performance
#SBATCH --mem=8G
#SBATCH --time=0:05:00
#SBATCH --output=%N-%j.out
#SBATCH --account=<your account>
module load python
# Using Default Python version - Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision --no-index
echo "starting training..."
time python cifar10-cpu.py
```

**File: cifar10-cpu.py**

```python
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import argparse
import os

parser = argparse.ArgumentParser(description='cifar10 classification models, cpu performance test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

def main():
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ### This next line will attempt to download the CIFAR10 dataset from the internet if you don't already have it stored in ./data
    ### Run this line on a login node with "download=True" prior to submitting your job, or manually download the data from
    ### https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz and place it under ./data
    dataset_train = CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers)
    perf = []
    total_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time = time.time() - start
        images_per_sec = args.batch_size / batch_time
        perf.append(images_per_sec)
        total_time = time.time() - total_start

if __name__ == '__main__':
    main()
```


### Working with a Single GPU

It's often said that you should always train a model with a GPU if one is available. This is almost always true (training very small models is often faster with one or more CPUs) on a local workstation, but not on our clusters.  In other words, you should not request a GPU if your code cannot make reasonable use of its computing capacity.

The superior performance of GPUs for deep learning tasks comes from two sources:

*   Ability to parallelize the execution of key operations, such as the multiply-accumulate, across thousands of compute cores, compared to the much smaller number of cores available with most CPUs.
*   Much larger memory bandwidth than a CPU, allowing GPUs to efficiently use their large number of cores to process a larger amount of data per compute cycle.

As with multiple CPUs, PyTorch offers parallel implementations of frequently used deep learning operators like matrix multiplication and convolution and uses specialized libraries for GPUs like CUDNN or MIOpen, depending on the hardware platform. This means that for a GPU to be worthwhile for a deep learning task, it must be composed of elements that can be scaled to a massive application of parallelism either by the number of operations that can be parallelized, by the amount of data to be processed, or ideally both. A concrete example would be a large model that has a large number of units and layers or that has a lot of input data, and ideally both.

In the example below, we adapt the code from the previous section to use a GPU and examine the performance. We observe that two parameters play an important role: `batch_size` and `num_workers`. The first parameter improves performance by increasing the size of the inputs at each iteration and better utilizing the GPU's capacity. In the case of the second parameter, performance is improved by facilitating the movement of data between the host memory (the CPU) and the GPU memory, reducing the GPU's idle time waiting for data to process.

We can draw two conclusions:

*   Increasing the value of `batch_size` to the maximum possible for the GPU memory optimizes performance.
*   Using a `DataLoader` with as many workers as `cpus-per-task` facilitates the supply of data to the GPU.

Of course, the `batch_size` parameter also impacts a model's performance on a task (i.e., accuracy, error, etc.), and there are different schools of thought on using large batches. We do not address this here, but if you believe that a small batch would be better suited to your application, go to the section "Working with a Single GPU" to learn how to maximize GPU utilization with small data inputs.

**File: pytorch-single-gpu.sh**

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=8G
#SBATCH --time=0:05:00
#SBATCH --output=%N-%j.out
#SBATCH --account=<your account>
module load python
# Using Default Python version - Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision --no-index
echo "starting training..."
time python cifar10-gpu.py --batch_size=512 --num_workers=0
```

**File: cifar10-gpu.py**

```python
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, single gpu performance test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')

def main():
    args = parser.parse_args()

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net().cuda() # Load model on the GPU
    criterion = nn.CrossEntropyLoss().cuda() # Load the loss function on the GPU
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers)
    perf = []
    total_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time = time.time() - start
        images_per_sec = args.batch_size / batch_time
        perf.append(images_per_sec)
        total_time = time.time() - total_start

if __name__ == '__main__':
    main()
```


#### Data Parallelism with a Single GPU

It is not advisable to use a GPU with a relatively small model that does not use a large portion of the GPU's memory and a reasonable portion of its computing capacity; use one or more CPUs instead. However, taking advantage of GPU parallelism becomes a good option if you have such a model with a very large dataset and want to perform training with small batches.

In this context, data parallelization refers to methods for training multiple copies of a model in parallel, where each copy receives a piece of the training data at each iteration. At the end of an iteration, the gradients are aggregated, and the parameters of each copy are updated synchronously or asynchronously, depending on the method. This approach can significantly increase execution speed, with an iteration being approximately N times faster with a large dataset, where N is the number of model copies. To use this approach, a warning is in order: for the trained model to be equivalent to the same model trained without parallelism, you must adjust the learning rate or the desired batch size according to the number of copies. For more information, see [these discussions](link-to-discussions-here).

PyTorch offers implementations of data parallelism methods, with the `DistributedDataParallel` class being the one recommended by PyTorch developers for the best performance. Designed for working with multiple GPUs, it can also be used with a single GPU.

In the example below, we adapt the code for a single GPU to use data parallelism. The task is relatively small; the batch size is 512 images, the model occupies about 1GB of GPU memory, and training only uses about 6% of its computing capacity. This model should not be trained on a GPU on our clusters. However, by parallelizing the data, a V100 GPU with 16GB of memory can hold 14 or 15 copies of the model and increase resource utilization in addition to obtaining a good speed increase. We use NVIDIA's Multi-Process Service (MPS) with MPI to efficiently place multiple copies of the model on a GPU.

**File: pytorch-gpu-mps.sh**

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 # request a GPU
#SBATCH --tasks-per-node=8 # This is the number of model replicas we will place on the GPU. Change this to 10,12,14,... to see the effect on performance
#SBATCH --cpus-per-task=1 # increase this parameter and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=8G
#SBATCH --time=0:05:00
#SBATCH --output=%N-%j.out
#SBATCH --account=<your account>
module load python
# Using Default Python version - Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision --no-index
# Activate Nvidia MPS:
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d
echo "starting training..."
time srun --cpus-per-task=$SLURM_CPUS_PER_TASK python cifar10-gpu-mps.py --batch_size=512 --num_workers=0
```

**File: cifar10-gpu-mps.py**

```python
import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.utils.data.distributed
import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, distributed data parallel maps test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')

def main():
    print("Starting...")
    args = parser.parse_args()
    rank = os.environ.get("SLURM_LOCALID")
    current_device = 0
    torch.cuda.set_device(current_device)
    """ this block initializes a process group and initiate communications
    between all processes that will run a model replica """
    print('From Rank: {} , ==> Initializing Process Group...'.format(rank))
    dist.init_process_group(backend="mpi", init_method=args.init_method)
    # Use backend="mpi" or "gloo". NCCL does not work on a single GPU due to a hard-coded multi-GPU topology check.
    print("process group ready!")
    print('From Rank: {} , ==> Making model..'.format(rank))

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[current_device])
    # Wrap the model with DistributedDataParallel
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    print('From Rank: {} , ==> Preparing data..'.format(rank))
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = CIFAR10(root='~/data', train=True, download=False, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler)
    perf = []
    total_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time = time.time() - start
        images_per_sec = args.batch_size / batch_time
        perf.append(images_per_sec)
        total_time = time.time() - total_start

if __name__ == '__main__':
    main()
```


### Working with Multiple GPUs

#### Issue with DistributedDataParallel and PyTorch 1.10

With our PyTorch 1.10 wheel `torch-1.10.0+computecanada`, code using `DistributedDataParallel` to work with multiple GPUs might fail unpredictably if the backend is set to `'nccl'` or `'gloo'`. We recommend using the latest PyTorch version instead of 1.10 on all general-purpose clusters.


#### Data Parallelism with Multiple GPUs

In this context, data parallelization refers to methods for training multiple copies of a model in parallel, where each copy receives a portion of the training data at each iteration. At the end of an iteration, the gradients are aggregated, and the parameters of each copy are updated synchronously or asynchronously, depending on the method. This approach can significantly increase execution speed, with an iteration being approximately N times faster with a large dataset, where N is the number of model copies. To use this approach, a warning is in order: for the trained model to be equivalent to the same model trained without parallelism, you must adjust the learning rate or the desired batch size according to the number of copies. For more information, see [these discussions](link-to-discussions-here).

When multiple GPUs are used, each receives a copy of the model; therefore, it must be small enough to fit in a single GPU's memory. To train a model that exceeds the memory capacity of a single GPU, see the section "Model Parallelism with Multiple GPUs".

There are several ways to parallelize data with PyTorch. Here, we present tutorials with the `DistributedDataParallel` class, with the PyTorch Lightning package, and with the Horovod package.


##### DistributedDataParallel

With multiple GPUs, the `DistributedDataParallel` class is recommended by PyTorch developers, whether with a single node or multiple nodes. In the following case, multiple GPUs are distributed across two nodes.

**File: pytorch-ddp-test.sh**

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:2          # Request 2 GPU "generic resources”.
#SBATCH --tasks-per-node=2   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --mem=8G
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out
module load python
# Using Default Python version - Make sure to choose a version that suits your application
srun -N $SLURM_NNODES -n $SLURM_NNODES bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision --no-index
EOF
export TORCH_NCCL_ASYNC_HANDLING=1
export MASTER_ADDR=$(hostname)
#Store the master node’s IP address in the MASTER_ADDR environment variable.
echo "r $SLURM_NODEID master: $MASTER_ADDR"
echo "r $SLURM_NODEID Launching python script"
# The $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
source $SLURM_TMPDIR/env/bin/activate

srun python pytorch-ddp-test.py --init_method tcp://$MASTER_ADDR:3456 --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) --batch_size 256
```

The Python script `pytorch-ddp-test.py` has the following form:

**File: pytorch-ddp-test.py**

```python
import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.utils.data.distributed
import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, distributed data parallel test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--max_epochs', type=int, default=4, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')

def main():
    print("Starting...")
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    """ This next line is the key to getting DistributedDataParallel working on SLURM:
    SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the
    current process inside a node and is also 0 or 1 in this example."""
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank
    current_device = local_rank
    torch.cuda.set_device(current_device)
    """ this block initializes a process group and initiate communications
    between all processes running on all nodes """
    print('From Rank: {} , ==> Initializing Process Group...'.format(rank))
    #init the process group
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
    print("process group ready!")
    print('From Rank: {} , ==> Making model..'.format(rank))

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[current_device])
    print('From Rank: {} , ==> Preparing data..'.format(rank))
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    for epoch in range(args.max_epochs):
        train_sampler.set_epoch(epoch)
        train(epoch, net, criterion, optimizer, train_loader, rank)

def train(epoch, net, criterion, optimizer, train_loader, train_rank):
    train_loss = 0
    correct = 0
    total = 0
    epoch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100 * correct / total
        batch_time = time.time() - start
        elapse_time = time.time() - epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)
        print("From Rank: {} ,