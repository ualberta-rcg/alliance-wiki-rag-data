# OpenACC Tutorial - Profiling (French)

This page is a translated version of the page [OpenACC Tutorial - Profiling](https://docs.alliancecan.ca/mediawiki/index.php?title=OpenACC_Tutorial_-_Profiling&oldid=135820) and the translation is 100% complete.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=OpenACC_Tutorial_-_Profiling&oldid=135820), français

## Learning Objectives

* Understand what a profiler is
* Know how to use the NVPROF profiler
* Understand code performance
* Know how to focus your efforts and rewrite time-consuming routines


## Contents

1. Code Profiler
2. Preparing the Code for the Exercise
    *  NVIDIA nvprof command-line profiler
3. Compiler Information
4. Obtaining Compiler Information
5. Interpreting the Result
6. Understanding the Code


## Code Profiler

Why would you need to profile code? Because it's the only way to:

* Understand how time is spent at critical points (hotspots)
* Understand code performance
* Know how to best use your development time

Why is it important to know the critical points in the code? According to Amdahl's Law, parallelizing the routines that require the most execution time (the critical points) produces the most impact.


## Preparing the Code for the Exercise

For the following example, we use code from a [Git data repository](https://github.com/OpenACC-Standard/OpenACC-Examples/tree/main/cpp).

Download and extract the package and go to the `cpp` or `f90` directory. The goal of this example is to compile and link the code to obtain an executable to profile its source code with a profiler.


### Compiler Choice

Developed by Cray and NVIDIA via its Portland Group division until 2020, then via its HPC SDK, these two types of compilers offer the most advanced support for OpenACC.

As for GNU compilers, support for OpenACC 2.x continues to improve since version 6 of GCC. As of July 2022, GCC versions 10, 11, and 12 support OpenACC version 2.6.

In this tutorial, we use version 22.7 of the NVIDIA HPC SDK. Note that NVIDIA compilers are free for academic research purposes.

```bash
[name@server ~]$ module load nvhpc/22.7
Lmod is automatically replacing "intel/2020.1.217" with "nvhpc/22.7".

The following have been reloaded with a version change:
1) gcccore/.9.3.0 => gcccore/.11.3.0
2) libfabric/1.10.1 => libfabric/1.15.1
3) openmpi/4.0.3 => openmpi/4.1.4
4) ucx/1.8.0 => ucx/1.12.1
[name@server ~]$ make
nvc++ -c -o main.o main.cpp
nvc++ main.o -o cg.x
```

Once the executable `cg.x` is created, we will profile its source code. The profiler measures function calls by running and monitoring this program.

**Important:** This executable uses approximately 3GB of memory and almost 100% of a CPU core. The test environment should therefore have 4GB of available memory and at least two (2) CPU cores.


### Profiler Choice

In this tutorial, we use two profilers:

* `nvprof` from NVIDIA, a command-line profiler capable of analyzing non-GPU codes
* `nvvp` (NVIDIA Visual Profiler), a cross-platform analysis tool for programs written with OpenACC and CUDA C/C++.

Since `cg.x` that we built does not yet use the GPU, we will start the analysis with the `nvprof` profiler.


## NVIDIA nvprof Command-Line Profiler

In its high-performance computing development kit, NVIDIA usually provides `nvprof`, but the version to be used on our clusters is included in a CUDA module.

```bash
[name@server ~]$ module load cuda/11.7
```

To profile a pure CPU executable, we must add the `--cpu-profiling on` arguments to the command line.

```bash
[name@server ~]$ nvprof --cpu-profiling on ./cg.x
... <Program output> ...
======== CPU profiling result (bottom up) :
Time (%) Time  Name
83.54%  90.6757s matvec (matrix const &, vector const &, vector const &)
83.54%  90.6757s | main
7.94%   8.62146s waxpby (double, vector const &, double, vector const &, vector const &)
7.94%   8.62146s | main
5.86%   6.36584s dot (vector const &, vector const &)
5.86%   6.36584s | main
2.47%   2.67666s allocate_3d_poisson_matrix (matrix &, int)
2.47%   2.67666s | main
0.13%   140.35ms initialize_vector (vector &, double)
0.13%   140.35ms | main
...
======== Data collected at 100Hz frequency
```

In the result, the `matvec()` function uses 83.5% of the execution time; its call is in the `main()` function.


## Compiler Information

Before working on the routine, we need to understand what the compiler does; let's ask ourselves the following questions:

* What optimizations were automatically applied by the compiler?
* What prevented further optimization?
* Would performance be affected by small modifications?

The NVIDIA compiler offers the `-Minfo` flag with the following options:

* `all`, to print almost all types of information, including
    * `accel` for compiler operations related to the accelerator
    * `inline` for information on extracted and aligned functions
    * `loop,mp,par,stdpar,vect` for information on loop optimization and vectorization
    * `intensity`, to print information on loop intensity
* (no option) produces the same result as the `all` option, but without the information provided by `inline`.


## Obtaining Compiler Information

Modify the Makefile:

```makefile
CXX=nvc++
CXXFLAGS=-fast -Minfo=all,intensity
LDFLAGS=${CXXFLAGS}
```

Perform a new build:

```bash
[name@server ~]$ make clean; make
...
nvc++ -fast -Minfo=all,intensity -c -o main.o main.cpp
...
```

The output will contain detailed information about compiler optimizations.  (A large code block showing compiler output follows here.  Due to its length and formatting challenges in Markdown, it is omitted for brevity.  The original HTML contains this information.)


## Interpreting the Result

The computational intensity of a loop represents the amount of work done by the loop as a function of the memory operations performed, i.e.,  $ {\mbox{computational intensity}}={\frac {\mbox{calculation operations}}{\mbox{memory operations}}} $

In the result, a value greater than 1 for `Intensity` indicates that the loop would run well on a graphics processing unit (GPU).


## Understanding the Code

Let's take a close look at the main loop of the `matvec()` function implemented in `matrix_functions.h`:

```c++
for (int i = 0; i < num_rows; i++) {
    double sum = 0;
    int row_start = row_offsets[i];
    int row_end = row_offsets[i + 1];
    for (int j = row_start; j < row_end; j++) {
        unsigned int Acol = cols[j];
        double Acoef = Acoefs[j];
        double xcoef = xcoefs[Acol];
        sum += Acoef * xcoef;
    }
    ycoefs[i] = sum;
}
```

Data dependencies are found by asking the following questions:

* Does one iteration affect others? For example, when a Fibonacci sequence is generated, each new value depends on the two preceding values. It is therefore very difficult, if not impossible, to implement efficient parallelism.
* Is the accumulation of values in `sum` a dependency? No, it's a reduction! And modern compilers optimize this kind of reduction well.
* Do loop iterations write and read to the same vectors so that values are used or overwritten by other iterations? Fortunately, this does not occur in the above code.

Now that the code has been analyzed, we can add directives to the compiler.


<- Previous page, Introduction | ^- Back to the beginning of the tutorial | Next page, Adding directives ->
