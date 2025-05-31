# CUDA

**Other languages:** English, français

"CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs)."<sup>[1]</sup>

It is reasonable to think of CUDA as a set of libraries and associated C, C++, and Fortran compilers that enable you to write code for GPUs. See [OpenACC Tutorial](link-to-openacc-tutorial) for another set of GPU programming tools.


## Contents

1. Quick start guide
    * Compiling
    * Submitting jobs
    * Linking libraries
2. Troubleshooting
    * Compute capability


## Quick start guide

### Compiling

Here we show a simple example of how to use the CUDA C/C++ language compiler, `nvcc`, and run code created with it. For a longer tutorial in CUDA programming, see [CUDA tutorial](link-to-cuda-tutorial).

First, load a CUDA module.

```bash
$ module purge
$ module load cuda
```

The following program will add two numbers together on a GPU. Save the file as `add.cu`.  The `.cu` file extension is important!

**File: add.cu**

```cpp
#include <iostream>

__global__ void add(int *a, int *b, int *c){
    *c = *a + *b;
}

int main(void){
    int a, b, c;
    int *dev_a, *dev_b, *dev_c;
    int size = sizeof(int);

    // allocate device copies of a,b, c
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    a = 2;
    b = 7;

    // copy inputs to device
    cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);

    // launch add() kernel on GPU, passing parameters
    add<<<1, 1>>>(dev_a, dev_b, dev_c);

    // copy device result back to host
    cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);

    std::cout << a << "+" << b << "=" << c << std::endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
```

Compile the program with `nvcc` to create an executable named `add`.

```bash
$ nvcc add.cu -o add
```

### Submitting jobs

To run the program, create a Slurm job script as shown below. Be sure to replace `def-someuser` with your specific account (see [Accounts and projects](link-to-accounts-and-projects)). For options relating to scheduling jobs with GPUs see [Using GPUs with Slurm](link-to-using-gpus-with-slurm).

**File: gpu_job.sh**

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=400M                # memory (per node)
#SBATCH --time=0-00:10            # time (DD-HH:MM)
./add #name of your program
```

Submit your GPU job to the scheduler with

```bash
$ sbatch gpu_job.sh
```

Example output: `Submitted batch job 3127733`

For more information about the `sbatch` command and running and monitoring jobs, see [Running jobs](link-to-running-jobs).

Once your job has finished, you should see an output file similar to this:

```bash
$ cat slurm-3127733.out
2+7=9
```

If you run this without a GPU present, you might see output like `2+7=0`.


### Linking libraries

If you have a program that needs to link some libraries included with CUDA, for example `cuBLAS`, compile with the following flags:

```bash
nvcc -lcublas -Xlinker=-rpath,$CUDA_PATH/lib64
```

To learn more about how the above program works and how to make the use of GPU parallelism, see [CUDA tutorial](link-to-cuda-tutorial).


## Troubleshooting

### Compute capability

NVIDIA has created this technical term, which they describe as follows:

> "The compute capability of a device is represented by a version number, also sometimes called its "SM version". This version number identifies the features supported by the GPU hardware and is used by applications at runtime to determine which hardware features and/or instructions are available on the present GPU." ([CUDA Toolkit Documentation, section 2.6](link-to-cuda-toolkit-doc))

The following errors are connected with compute capability:

* `nvcc fatal : Unsupported gpu architecture 'compute_XX'`
* `no kernel image is available for execution on the device (209)`

If you encounter either of these errors, you may be able to fix it by adding the correct flag to the `nvcc` call:

`-gencode arch=compute_XX,code=[sm_XX,compute_XX]`

If you are using `cmake`, provide the following flag:

`cmake .. -DCMAKE_CUDA_ARCHITECTURES=XX`

where “XX” is the compute capability of the Nvidia GPU that you expect to run the application on. To find the value to replace “XX”, see the [Available GPUs table](link-to-available-gpus-table).

For example, if you will run your code on a Narval A100 node, its compute capability is 80. The correct flag to use when compiling with `nvcc` is `-gencode arch=compute_80,code=[sm_80,compute_80]`. The flag to supply to `cmake` is: `cmake .. -DCMAKE_CUDA_ARCHITECTURES=80`


<sup>[1]</sup> NVIDIA CUDA Home Page. CUDA is a registered trademark of NVIDIA.  Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=CUDA&oldid=137392](https://docs.alliancecan.ca/mediawiki/index.php?title=CUDA&oldid=137392)"


**(Note:  Replace bracketed links like `[link-to-cuda-tutorial]` with the actual URLs.)**
