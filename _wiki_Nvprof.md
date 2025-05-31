# Nvprof

This is a draft, a work in progress intended for publication as a full article.  Its accuracy and completeness are not guaranteed.

Nvprof is a lightweight, command-line, GUI-less profiler available for Linux, Windows, and macOS.  This tool allows you to collect and view profiling data of CUDA-related activities on both the CPU and GPU, including kernel execution and memory transfers. Profiling options are provided via command-line arguments.


## Strengths

Nvprof is capable of providing textual reports including:

* Summary of GPU and CPU activity
* Trace of GPU and CPU activity
* Event collection

Nvprof also features headless profile collection using the NVIDIA Visual Profiler:

1. Use Nvprof on a headless node to collect data.
2. Visualize the timeline with the Visual Profiler.


## Quickstart Guide

On Béluga and Narval, the NVIDIA Data Center GPU Manager (DCGM) needs to be disabled during job submission:

```bash
DISABLE_DCGM=1 salloc --gres=gpu:1 ...
```

After your job starts, DCGM will stop within a minute.  The following loop waits for the monitoring service to stop (when `grep` returns nothing):

```bash
while [ ! -z "$(dcgmi -v | grep 'Hostengine build info:')" ]; do sleep 5; done
```


### Environment Modules

Before using NVPROF, load the appropriate module. NVPROF is part of the CUDA package. Run the following to see available versions (considering your loaded compiler and MPI modules):

```bash
module avail cuda
```

For a comprehensive list of CUDA modules:

```bash
module -r spider '.*cuda.*'
```

At the time of writing, these were available:

* `cuda/10.0.130`
* `cuda/10.0`
* `cuda/9.0.176`
* `cuda/9.0`
* `cuda/8.0.44`
* `cuda/8.0`

Use `module load cuda/version` to select a version. For example, to load CUDA compiler version 10.0:

```bash
module load cuda/10.0
```

You also need to set the CUDA library path:

```bash
export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH
```


### Compile Your Code

To obtain useful information from Nvprof, compile your code with a CUDA compiler (e.g., `nvcc` for C).


### Profiling Modes

Nvprof operates in several modes.

#### Summary Mode

This is the default mode. It outputs a single result line for each instruction (kernel function, CUDA memory copy/set). For each kernel, it shows the total, average, minimum, and maximum execution times.

In this example, the application is `a.out`:

```bash
nvprof ./a.out
```

Example output:

```
Starting...
==27694== NVPROF is profiling process 27694, command: a.out
GPU Device 0: "GeForce GTX 580" with compute capability 2.0

MatrixA (320,320), MatrixB (640,320)
Computing result using CUDA Kernel...
done
Performance = 35.35 GFlop/s, Time = 3.708 msec, Size = 131072000 Ops, WorkgroupSize = 1024 threads/block
Checking computed result for correctness: OK
==27694== Profiling application: matrixMul
==27694== Profiling result:
Time (%)      Time             Calls       Avg       Min       Max      Name
99.94%     1.11524s           301      3.7051ms   3.6928ms   3.7174ms  void matrixMulCUDA<int=32>(float*, float*, float*, int, int)
0.04%       406.30us           2       203.15us   203.15us   203.15us  [CUDA memcpy HtoD]
0.02%       248.29us           1       248.29us   248.29us   248.29us  [CUDA memcpy DtoH]
```

Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=Nvprof&oldid=137815](https://docs.alliancecan.ca/mediawiki/index.php?title=Nvprof&oldid=137815)"
