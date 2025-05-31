# OpenACC Tutorial - Optimizing Loops

This page is a translated version of the page [OpenACC Tutorial - Optimizing loops](https://docs.alliancecan.ca/mediawiki/index.php?title=OpenACC_Tutorial_-_Optimizing_loops&oldid=14704) and the translation is 100% complete.

**Other languages:**

* [English](https://docs.alliancecan.ca/mediawiki/index.php?title=OpenACC_Tutorial_-_Optimizing_loops&oldid=14704)
* français


## Learning Objectives

* Understand the different levels of parallelism in a GPU.
* Understand compiler messages about parallelization.
* Know how to obtain optimization suggestions from the visual profiler.
* Know how to indicate parallelization parameters to the compiler.


## Contents

1. Outsmarting the Compiler
2. OpenACC Parallelism Levels
3. Controlling Parallelism
    * Specifying the Accelerator Type
    * Specifying the Size of Each Parallelism Level
4. Controlling Vector Length
5. Guided Analysis of the NVIDIA Visual Profiler
6. Adding Workers
7. Two Other Optimization Clauses
8. Exercise


## 1. Outsmarting the Compiler

Until now, the compiler has done a good job; in the previous steps, the performance gain tripled compared to that of the CPU. Let's now study how the compiler parallelized the code and, if possible, give it a helping hand. To do this, we need to understand the different levels of parallelism possible with OpenACC, particularly with an NVIDIA GPU. Let's first examine the feedback provided by the compiler during the compilation of the last version (in `steps2.kernels`).

```bash
[name@server ~]$ pgc++ -fast -ta=tesla,lineinfo -Minfo=all,intensity,ccff -c -o main.o main.cpp
...
initialize_vector (vector &, double) : ... 42, Intensity = 0.0 Loop is parallelizable Accelerator kernel generated Generating Tesla code
42, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
dot (const vector &, const vector &) : ... 29, Intensity = 1.00 Loop is parallelizable Accelerator kernel generated Generating Tesla code
29, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
30, Sum reduction generated for sum
waxpby (double, const vector &, double, const vector &, const vector &) : ... 44, Intensity = 1.00 Loop is parallelizable Accelerator kernel generated Generating Tesla code
44, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
```

Notice that each loop is parallelized with `vector(128)`; this means that the compiler generated instructions for a code block of length 128. This is where the programmer has an advantage. In fact, if you examine the contents of the `matrix.h` file, you will see that each row of the matrix has 27 elements; the compiler therefore generated instructions for the unnecessary calculation of 101 elements. We will see later how to handle this case.


## 2. OpenACC Parallelism Levels

The three possible levels of parallelism with OpenACC are `vector`, `worker`, and `gang`.

`vector` threads execute a single operation on multiple data (SIMD), in a single step. If there is less data than the vector length, the operation is still executed on NULL values and the result is discarded.

The `worker` clause calculates a `vector`.

The `gang` level includes one or more `worker`s, which share resources such as cache memory or the processor. Each `gang` works completely independently.


## OpenACC and CUDA Correspondence

Since OpenACC is designed for generic accelerators, there is no direct correspondence with CUDA threads, blocks, and warps. In OpenACC version 2.0, the levels are nested, starting from the outside with `gang` to the inside with `vector`. The OpenACC-CUDA correspondence is established by the compiler. We know that there could be exceptions, but the following correspondence is generally valid:

* OpenACC `vector` => CUDA threads
* OpenACC `worker` => CUDA warps
* OpenACC `gang` => CUDA thread blocks


## 3. Controlling Parallelism

`loop` can be used with certain clauses to control the level of parallelism that the compiler should produce for the loop. These clauses are:

* `gang`, which produces the `gang` level of parallelism
* `worker`, which produces the `worker` level of parallelism
* `vector`, which produces the `vector` level of parallelism
* `seq`, which executes the loop sequentially without parallelism

A loop can have multiple level clauses, but they must be placed from the outside to the inside (from `gang` to `vector`).


### 3.1 Specifying the Accelerator Type

Depending on how the parallelization is called to take place, different types of accelerators will not have the same performance. The OpenACC clause `device_type` allows you to specify the type of accelerator to which the clause that follows it in the code label applies. For example, `device_type(nvidia) vector` is only performed if the code is compiled for an NVIDIA GPU.


### 3.2 Specifying the Size of Each Parallelism Level

A size parameter can be added to the `vector`, `worker`, and `gang` clauses. For example, `worker(32) vector(32)` creates 32 workers to perform calculations on vectors of length 32.


#### Maximum Values

Some accelerators may have limited numbers of `vector`, `worker`, and `gang` to parallelize a loop. In the case of NVIDIA GPUs:

* The length of `vector` is a multiple of 32 (at most 1024);
* The size of `gang` is the product of the number of `worker`s multiplied by the size of `vector` (at most 1204).


## 4. Controlling Vector Length

Let's go back to our example; we noticed that the compiler had set the length of `vector` to 128. Since we know that the rows contain 27 elements, we can reduce the length of `vector` to 32. With the `kernels` directive, here's what the code looks like:

```c++
#pragma acc kernels present(row_offsets,cols,Acoefs,xcoefs,ycoefs)
{
    for (int i = 0; i < num_rows; i++) {
        double sum = 0;
        int row_start = row_offsets[i];
        int row_end = row_offsets[i + 1];
        #pragma acc loop device_type(nvidia) vector(32)
        for (int j = row_start; j < row_end; j++) {
            unsigned int Acol = cols[j];
            double Acoef = Acoefs[j];
            double xcoef = xcoefs[Acol];
            sum += Acoef * xcoef;
        }
        ycoefs[i] = sum;
    }
}
```

If you prefer the `parallel loop` directive, the length of `vector` is defined at the level of the outermost loop with the `vector_length` clause. The `vector` clause is then used to parallelize an inner loop via the `vector` level, giving:

```c++
#pragma acc parallel loop present(row_offsets,cols,Acoefs,xcoefs,ycoefs) \
    device_type(nvidia) vector_length(32)
for (int i = 0; i < num_rows; i++) {
    double sum = 0;
    int row_start = row_offsets[i];
    int row_end = row_offsets[i + 1];
    #pragma acc loop reduction(+:sum) \
        device_type(nvidia) vector
    for (int j = row_start; j < row_end; j++) {
        unsigned int Acol = cols[j];
        double Acoef = Acoefs[j];
        double xcoef = xcoefs[Acol];
        sum += Acoef * xcoef;
    }
    ycoefs[i] = sum;
}
```

If you make this change, you will see that on a K20, the execution time goes from 10 to about 15 seconds. The compiler demonstrates its skill here.


## 5. Guided Analysis of the NVIDIA Visual Profiler

**Guided Analysis, Step 1**

**Guided Analysis, Step 2**

**Guided Analysis, Step 3**

**Guided Analysis, Step 4**

As we did in the "Code Porting Performance" section of the "Adding Directives" lesson, open NVIDIA Visual Profiler and start a new session with the last executable we produced. Follow these steps (see images):

1. Under the "Analysis" tab, click "Examine GPU Usage". At the end of the analysis, the compiler produces a series of warnings indicating possible improvements.
2. Click "Examine Individual Kernels" to display the list of kernels.
3. Select the first one in the list and click "Perform Kernel Analysis". The profiler presents a detailed analysis of the kernel and indicates probable bottlenecks. In this case, performance is limited by memory latency.
4. Click "Perform Latency Analysis".

At the end of the procedure, the following information should be displayed:

We have several important pieces of information here:

* The text clearly indicates that performance is limited by the size of the blocks, which corresponds to the size of the gangs in OpenACC;
* The "Active Threads" line informs us that the GPU executes 512 threads out of the 2048 possible;
* The "Occupancy" line shows that the GPU is used at 25% of its capacity; this is the ratio of actual use to possible use of the GPU. Note that 100% occupancy does not necessarily give the best performance, but 25% is rather low;
* The most useful answers are in the "Warps" table.
* We learn that the GPU executes 32 threads per block (in OpenACC, vector threads per gang) while it could execute 1024.
* We also see that the GPU executes 1 warp per block (in OpenACC, 1 worker per gang) while it could execute 32.
* On the last line, we see that for the accelerator to operate at full efficiency, 64 gangs should be executed, but the accelerator can only process 16.

The conclusion is that we need larger gangs, which we will do by adding workers while keeping the vector size at 32.


## 6. Adding Workers

Since we know that for an NVIDIA GPU the size of a `gang` cannot exceed 1024 and that this size is the product of the length of `vector` multiplied by the number of `worker`s, we want to have 32 `worker`s per gang. With the `kernels` directive, the code reads:

```c++
#pragma acc kernels present(row_offsets,cols,Acoefs,xcoefs,ycoefs)
{
    #pragma acc loop device_type(nvidia) gang worker(32)
    for (int i = 0; i < num_rows; i++) {
        double sum = 0;
        int row_start = row_offsets[i];
        int row_end = row_offsets[i + 1];
        #pragma acc loop device_type(nvidia) vector(32)
        for (int j = row_start; j < row_end; j++) {
            unsigned int Acol = cols[j];
            double Acoef = Acoefs[j];
            double xcoef = xcoefs[Acol];
            sum += Acoef * xcoef;
        }
        ycoefs[i] = sum;
    }
}
```

Note that we parallelize the outer loop `workers` since the inner loop is already at the `vector` level of parallelism.

With the `parallel loop` directive, the code reads:

```c++
#pragma acc parallel loop present(row_offsets,cols,Acoefs,xcoefs,ycoefs) \
    device_type(nvidia) vector_length(32) \
    gang worker num_workers(32)
for (int i = 0; i < num_rows; i++) {
    double sum = 0;
    int row_start = row_offsets[i];
    int row_end = row_offsets[i + 1];
    #pragma acc loop reduction(+:sum) \
        device_type(nvidia) vector
    for (int j = row_start; j < row_end; j++) {
        unsigned int Acol = cols[j];
        double Acoef = Acoefs[j];
        double xcoef = xcoefs[Acol];
        sum += Acoef * xcoef;
    }
    ycoefs[i] = sum;
}
```

This additional step results in a performance nearly double that of what the compiler can do on its own. On a K20, the code took 10 seconds to execute and the duration is now 6 seconds.


## 7. Two Other Optimization Clauses

So far we have not mentioned two clauses that are very useful in optimizing loops.

The `collapse(N)` clause is used with a loop directive to collapse the next N loops into a single flat loop. It is used in cases of nested loops or when loops are very short.

The `tile(N,[M,...])` clause distributes the following loops into a tiled structure before parallelizing. It is useful in the case of an algorithm with strong locality because the accelerator can use the data from surrounding tiles.


## 8. Exercise

**Jacobi Iterations**

Put into practice what you have learned about OpenACC.

In the `bonus` directory is code that solves the Laplace equation with the Jacobi method. Port this code to a GPU and observe the performance gain you obtain.

[Return to the beginning of the tutorial](https://docs.alliancecan.ca/mediawiki/index.php?title=OpenACC_Tutorial_-_Introduction/fr&oldid=14702)
