# PGPROF: A Performance Analysis Tool for Parallel Programs

PGPROF is a powerful and simple tool for analyzing the performance of parallel programs written with OpenMP, MPI, OpenACC, or CUDA.  It offers two profiling modes: command-line and graphical.

## Quickstart Guide

Using PGPROF typically involves two steps:

1.  **Data Collection:** Run your application with profiling enabled.
2.  **Analysis:** Visualize the collected data.

Both steps can be performed using either the command-line or graphical mode.

### Environment Modules

Before using PGPROF, load the appropriate module.  PGPROF is part of the PGI compiler package.  Use the following commands:

```bash
module avail pgi
```

To see available versions. For a comprehensive list of PGI modules:

```bash
module -r spider '.*pgi.*'
```

Then load the desired version:

```bash
module load pgi/version 
```

For example, to load version 17.3:

```bash
module load pgi/17.3
```

### Compiling Your Code

To obtain useful information from PGPROF, compile your code using one of the PGI compilers (`pgcc` for C, `pgc++` for C++, `pgfortran` for Fortran). Fortran sources might require the `-g` flag.


### Command-Line Mode

#### Data Collection

Use PGPROF to run your application and save the performance data to a file.  For example, to profile `a.out` and save the data to `a.prof`:

```bash
pgprof -o a.prof ./a.out
```

The data file can then be analyzed in graphical mode (using *File | Import*) or in command-line mode.


#### Analysis

To analyze the data in command-line mode:

```bash
pgprof -i a.prof
```

The results are typically categorized (e.g., GPU kernel execution profile, CUDA API execution profile, OpenACC execution profile, CPU execution profile).  An example output might look like this:

```
======
Profiling result:
Time (%)   Time       Calls    Avg      Min      Max      Name
38.14%     1.41393s   20       70.696ms  70.666ms  70.731ms  calc2_198_gpu
31.11%     1.15312s   18       64.062ms  64.039ms  64.083ms  calc3_273_gpu
23.35%     865        0.68ms   20       43.284ms  43.244ms  43.325ms  calc1_142_gpu
5.17%      191        0.78ms   141      1.3602ms  1.3120us  1.6409ms  [CUDA memcpy HtoD]
...
========
API calls:
Time (%)   Time       Calls    Avg      Min      Max      Name
92.65%     3.49314s   62       56.341ms  1.8850us  70.771ms  cuStreamSynchronize
3.78%      142        0.36ms   1       142       0.36ms   142       0.36ms   cuDevicePrimaryCtxRetain
...
========
OpenACC (excl):
Time (%)   Time       Calls    Avg      Min      Max      Name
36.27%     1.41470s   20       70.735ms  70.704ms  70.773ms  acc_wait@swim-acc-data.f:223
63.3%      1.15449s   18       64.138ms  64.114ms  64.159ms  acc_wait@swim-acc-data.f:302
========
CPU profiling result (bottom up):
Time (%)   Time       Name
59.09%     8.55785s   cudbgGetAPIVersion
59.09%     8.55785s   start_thread
59.09%     8.55785s   clone
25.75%     3.73007s   cuStreamSynchronize
25.75%     3.73007s   __pgi_uacc_cuda_wait
25.75%     3.73007s   __pgi_uacc_computedone
10.38%     1.50269s   swim_mod_calc2_
```

#### Options

The output can be filtered. For example, `--cpu-profiling` shows only CPU results.  `--cpu-profiling-mode top-down` or `--cpu-profiling-mode bottom-up` control the call tree orientation.  Examples:

```bash
pgprof --cpu-profiling-mode top-down -i a.prof
pgprof --cpu-profiling-mode bottom-up -i a.prof
```


### Graphical Mode

The graphical mode allows for both data collection and analysis within a single session, or analysis of pre-saved data files.

#### Data Collection

1.  Launch the PGI profiler (on a compute node in an interactive session, not the login node, due to memory requirements).  Start an interactive session with `salloc --x11 ...` to enable X11 forwarding.
2.  In the *File* menu, click *New Session*.
3.  Select your executable and add any necessary arguments.
4.  Click *Next*, then *Finish*.

#### Analysis

In the *CPU Details* tab, click "Show the top-down (callers first) call tree view". The visualization window has four panes:

*   Upper-right: Timeline of events.
*   *GPU Details*: GPU kernel performance details.
*   *CPU Details*: CPU function performance details.
*   *Properties*: Details for a selected function in the timeline.


## References

PGPROF is a product of PGI, a subsidiary of NVIDIA Corporation.

**(Note:  The original MediaWiki source included links to a "Quick Start Guide" and "User's Guide," which are not included here as they were not provided in the input.)**
