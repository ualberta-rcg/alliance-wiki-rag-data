# CUDA

CUDA is a parallel computing platform and programming model developed by NVIDIA for general-purpose computations using GPUs.  CUDA can be viewed as a set of libraries and compilers (C, C++, and Fortran) that allow you to create programs for GPUs. For other GPU programming tools, see the [OpenACC Tutorial](link-to-openacc-tutorial).


## A Simple Example

### Compilation

Here, we'll run code created with the CUDA C/C++ compiler `nvcc`. A more detailed version of this example can be found on the [CUDA Tutorial](link-to-cuda-tutorial) page.

First, load the CUDA module:

```bash
$ module purge
$ module load cuda
```

In this example, we add two numbers. Save the file as `add.cu`; the `.cu` suffix is important.

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
    add<<<1,1>>>(dev_a, dev_b, dev_c);

    // copy device result back to host
    cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);

    std::cout << a << "+" << b << "=" << c << std::endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
```

Compile the program using `nvcc` to create the executable `add`:

```bash
$ nvcc add.cu -o add
```

### Submitting Jobs

To run the program, create the Slurm script below.  Make sure to replace `def-someuser` with your account name (see [Accounts and Projects](link-to-accounts-and-projects)). For details on scheduling, see [Slurm Scheduling of GPU Tasks](link-to-slurm-scheduling).

**File: gpu_job.sh**

```bash
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=400M                # memory (per node)
#SBATCH --time=0-00:10            # time (DD-HH:MM)
./add #name of your program
```

Submit the job to the scheduler:

```bash
$ sbatch gpu_job.sh
Submitted batch job 3127733
```

For more information on the `sbatch` command, job execution, and monitoring, see [Running Jobs](link-to-running-jobs).

The output file will look like this:

```bash
$ cat slurm-3127733.out
2+7=9
```

Without a GPU, the result would be similar to `2+7=0`.


### Linking Libraries

If your program needs to link against libraries included with CUDA, for example `cuBLAS`, compile with these flags:

```bash
nvcc -lcublas -Xlinker=-rpath,$CUDA_PATH/lib64
```

See the [CUDA Tutorial](link-to-cuda-tutorial) for more details on this example and how to use parallelism with GPUs.


## Troubleshooting

### Compute Capability Attribute

NVIDIA uses the term "compute capability" to refer to an attribute of GPU devices. Sometimes called "SM version" (SM for Streaming Multiprocessor), this is a version number that identifies certain features of a GPU. This attribute is used at application runtime to determine the hardware capabilities and/or instructions available for a particular GPU; for more information, see [CUDA Toolkit Documentation, section 2.6](link-to-cuda-toolkit-documentation).

The following error messages are caused by a problem related to this attribute:

```
nvcc fatal : Unsupported gpu architecture 'compute_XX'
no kernel image is available for execution on the device (209)
```

Adding a flag in the `nvcc` call might solve these problems:

```bash
-gencode arch=compute_XX,code=[sm_XX,compute_XX]
```

If you are using `cmake`, the flag would be:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=XX
```

where `XX` is the compute capability value for the NVIDIA GPU that will be used to run your application. To find these values, see the [table of available GPUs](link-to-gpu-table).

For example, if your code will be executed on a Narval A100 node, the compute capability has a value of 80, and the flag to use when compiling with `nvcc` is:

```bash
-gencode arch=compute_80,code=[sm_80,compute_80]
```

The flag for `cmake` is:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
```

↑ NVIDIA trademark.


**(Remember to replace the bracketed placeholders like `[link-to-openacc-tutorial]` with actual links.)**
