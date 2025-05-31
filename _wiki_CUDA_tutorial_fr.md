# CUDA Tutorial

This page is a translated version of the page CUDA tutorial and the translation is 100% complete.

Other languages: [English](link-to-english-version), français

## Contents

1. Introduction
2. What is a GPU?
3. What is CUDA?
4. GPU Architecture
5. Programming Model
6. Execution Model
7. Thread Blocks
8. Thread Scheduling
9. GPU Memory Types
10. Basic Operations
    * Memory Allocation
    * Data Transfer
11. Example of a Simple CUDA C Program
12. Advantages of Shared Memory
13. Basic Performance Factors
    * Memory Transfers
    * Bandwidth
    * Common Programming Strategies


## Introduction

This tutorial introduces the highly parallel computing component that is the Graphics Processing Unit (GPU); the parallel programming language CUDA; and some of the CUDA numerical libraries used in high-performance computing.

### Prerequisites

This tutorial shows how to use CUDA to accelerate C or C++ programs; a good knowledge of one of these languages will allow you to get the most out of it. While CUDA is also used for Fortran programs, we will limit ourselves here to CUDA for C/C++ and use the term CUDA C. This essentially involves producing C/C++ functions that can be executed by CPUs and GPUs.

### Learning Objectives

* Understand the architecture of a GPU
* Understand the execution of a CUDA program
* Understand and manage the different types of GPU memory
* Write and compile a CUDA code example


## What is a GPU?

A GPU (Graphics Processing Unit) is a single-chip processor capable of performing mathematical calculations quickly to produce image renderings. For several years, the power of the GPU has also been used to accelerate the execution of intensive calculations in several areas of cutting-edge scientific research.


## What is CUDA?

CUDA (Compute Unified Device Architecture) is a software environment and a scalable programming model for processing intensive parallel computations on GPUs.


## GPU Architecture

A GPU consists of:

* Global memory similar to CPU memory, accessible by both CPU and GPU
* Streaming Multiprocessors (SMs)
* Each SM is composed of several Streaming Processors (SPs) that perform the calculations
* Each SM has its own control unit, registers, execution pipelines, etc.


## Programming Model

Let's first look at some important terms:

* **Host:** Refers to the CPU and its memory (host memory).
* **Graphics Card:** Refers to the GPU and its memory (graphics card memory).

The CUDA model is a heterogeneous model where both the CPU and the GPU are used. CUDA code can manage both types of memory: host memory and graphics card memory. The code also executes GPU functions called kernels. These functions are executed in parallel by multiple GPU threads. The process involves five steps:

1. Declaration and allocation of host memory and graphics card memory.
2. Initialization of host memory.
3. Transfer of data from host memory to graphics card memory.
4. Execution of GPU functions (kernels).
5. Return of data to host memory.


## Execution Model

Simple CUDA code executed in a GPU is called a kernel.  We must ask:

* How to execute a kernel on a group of streaming multiprocessors?
* How to make this kernel execute intensively in parallel?

Here is the recipe in answer to these questions:

* Each GPU core (streaming processor) executes a sequential thread, which is the smallest discrete set of instructions managed by the operating system's scheduler.
* All GPU cores execute the kernel simultaneously according to the SIMT (Single Instruction, Multiple Threads) model.

The following procedure is recommended:

1. Copy the input data from CPU memory to GPU memory.
2. Load and launch the GPU program (the kernel).
3. Copy the results from GPU memory to CPU memory.


## Thread Blocks

Threads are grouped into blocks that form grids.

To obtain intensive parallelism, as many threads as possible must be used; since a CUDA kernel comprises a very large number of threads, they must be well organized. With CUDA, threads are grouped into blocks of threads, which themselves form a grid. Dividing the threads ensures that:

* Grouped threads cooperate via shared memory,
* Threads in one block do not cooperate with threads in other blocks.

According to this model, the threads in a block work on the same group of instructions (but perhaps with different datasets) and exchange data via shared memory. Threads in other blocks do the same (see figure).

**Intercommunication via shared memory of threads in a block.**

Each thread uses identifiers (IDs) to decide which data to use:

* Block IDs: 1D or 2D (blockIdx.x, blockIdx.y)
* Thread IDs: 1D, 2D, or 3D (threadIdx.x, threadIdx.y, threadIdx.z)

This model simplifies memory addressing when processing multidimensional data.


## Thread Scheduling

A Streaming Multiprocessor (SM) usually executes one block of threads at a time. The code is executed in groups of 32 threads (called warps). A physical scheduler is free to assign blocks to any SM at any time.  Furthermore, when an SM receives the block assigned to it, this does not mean that this particular block will be executed continuously. In fact, the scheduler can delay/suspend the execution of such blocks under certain conditions, for example if the data is no longer available (indeed, reading data from the GPU's global memory is very time-consuming). When this happens, the scheduler executes another block of threads that is ready to be executed. This is a kind of zero-overhead scheduling that promotes a smoother execution flow so that the SMs do not remain inactive.


## GPU Memory Types

Several types of memory are available to CUDA operations:

* **Global memory:** Off-chip, efficient for I/O operations, but relatively slow.
* **Shared memory:** On-chip, allows good collaboration between threads, very fast.
* **Registers and local memory:** Thread workspace, very fast.
* **Constant memory:**


## Basic Operations

### Memory Allocation

`cudaMalloc((void**)&array, size)`

Allocates an object in the graphics card's memory. Requires the address of a pointer to the allocated data and the size.

`cudaFree(array)`

Deallocates the object in memory. Only requires the pointer to the data.


### Data Transfer

`cudaMemcpy(array_dest, array_orig, size, direction)`

Copies data from the graphics card to the host or from the host to the graphics card. Requires pointers to the data, the size, and the direction type (cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, etc.).

`cudaMemcpyAsync`

Identical to cudaMemcpy, but transfers data asynchronously, meaning that the execution of other processes is not blocked.


## Example of a Simple CUDA C Program

In this example, we add two numbers. This is a very simple example and you shouldn't expect to see a large acceleration.

```c++
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

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}
```

Something is certainly missing; this code doesn't look parallel... As a solution, let's modify the content of the kernel between the triple chevrons (<<< >>>).

```c++
add<<<N, 1>>>(dev_a, dev_b, dev_c);
```

Here, we have replaced 1 with N so that N different CUDA blocks are executed at the same time. To parallelize, however, modifications must also be made to the kernel:

```c++
__global__ void add(int *a, int *b, int *c){
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
```

where `blockIdx.x` is the unique number identifying a CUDA block. In this way, each CUDA block adds a value of a[] to b[].

**Parallelization based on blocks.**

Let's modify the content between the triple chevrons again.

```c++
add<<<1, N>>>(dev_a, dev_b, dev_c);
```

The task is now distributed over parallel threads rather than blocks. What is the advantage of parallel threads? Unlike blocks, threads can communicate with each other; in other words, we parallelize over multiple threads in the block when communication is intense. Portions of code that can be executed independently, either with little or no communication, are distributed over parallel blocks.


## Advantages of Shared Memory

Until now, all memory transfers in the kernel have been via the regular (global) memory of the GPU, which is relatively slow. There is often so much communication between threads that performance is significantly reduced. To counter this problem, we can use shared memory, which can speed up memory transfers between threads. The secret, however, is that only threads in the same block can communicate. To illustrate the use of this memory, let's look at the example of the dot product where two vectors are multiplied element by element and then added, as follows:

```c++
__global__ void dot(int *a, int *b, int *c){
  int temp = a[threadIdx.x] * b[threadIdx.x];
}
```

After each thread has executed its portion, everything must be added; each thread must share its data. However, the problem is that each copy of the thread's temporary variable is private. The solution is to use shared memory with the following modifications to the kernel:

```c++
#define N 512
__global__ void dot(int *a, int *b, int *c){
  __shared__ int temp[N];
  temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0){
    int sum;
    for (int i = 0; i < N; i++)
      sum += temp[i];
    *c = sum;
  }
}
```


## Basic Performance Factors

### Memory Transfers

PCI-e is extremely slow (4-6Go/s) compared to host memory and graphics card memory.

* Minimize memory copies in both directions.
* Keep data on the graphics card as long as possible.
* It is sometimes not efficient to use the host (the CPU) for non-optimal tasks; it might be faster to execute them with the GPU than to copy to the CPU, execute, and return the result.
* Use memory times to analyze execution times.


### Bandwidth

Always consider bandwidth limitations when modifying your code.

* Know the theoretical peak bandwidth of the various data links.
* Count the bytes written/read and compare with the theoretical peak.
* Use the various types of memory as appropriate: global, shared, constant.


### Common Programming Strategies

Constant memory also resides in DRAM; access is much slower than for shared memory.

BUT, it is cached!!!

Highly efficient read-only access, transmission.

Distribute data well according to the access mode:

* Read-only: Constant memory (very fast if in cache)
* Read/write in the block: Shared memory (very fast)
* Read/write in the thread: Registers (very fast)
* Read/write in input/results: Global memory (very slow)


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=CUDA_tutorial/fr&oldid=110402](https://docs.alliancecan.ca/mediawiki/index.php?title=CUDA_tutorial/fr&oldid=110402)"
