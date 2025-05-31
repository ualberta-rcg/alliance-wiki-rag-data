# PGPROF

PGPROF is a simple yet powerful tool for analyzing parallel programs written with OpenMP, MPI, OpenACC, or CUDA. Profiling can be performed in command-line or graphical mode.

## Usage

In both modes, the work is generally done in two steps:

1. **Data collection:** Running the application with profiling enabled.
2. **Analysis:** Visualizing the data obtained in the first step.

### Environment Modules

To use PGPROF, you need to load the appropriate module.  Since PGPROF is part of the PGI compiler package, run the command `module avail pgi` to see the available versions for the compiler, MPI, and CUDA modules you have already loaded. For a list of available PGI modules, run `module -r spider '.*pgi.*'`.

As of December 2018, the available modules were:

*   `pgi/13.10`
*   `pgi/17.3`

Select a version with `module load pgi/version`; to load version 17.3 of the PGI compiler, the command is:

```bash
[name@server ~]$ module load pgi/17.3
```

### Code Compilation

To get useful information from PGPROF, you must first compile the code with one of the PGI compilers: `pgcc` for C, `pgc++` for C++, `pgfortran` for Fortran.  Fortran source code may need to be compiled with the `-g` flag.

### Command-Line Mode

**Data Collection:** The first step is to use PGPROF to run the application and save the performance data. In the following example, `a.out` is the application and `a.prof` is the file where the data is saved.

```bash
[name@server ~]$ pgprof -o a.prof ./a.out
```

The data file can be saved and then analyzed in graphical mode using the `File > Import` command (see **Graphical Mode** below) or in command-line mode as follows:

**Analysis:** For the visualization step, use:

```bash
[name@server ~]$ pgprof -i a.prof
```

The results are typically presented in several categories, for example:

*   GPU kernel performance profile
*   CUDA API execution profile
*   OpenACC execution profile
*   CPU function performance profile

Example output:

```
====== Profiling result:
Time (%) Time Calls Avg Min Max Name
38.14% 1.41393s 20 70.696ms 70.666ms 70.731ms calc2_198_gpu
31.11% 1.15312s 18 64.062ms 64.039ms 64.083ms calc3_273_gpu
23.35% 865 0.68ms 20 43.284ms 43.244ms 43.325ms calc1_142_gpu
5.17% 191 0.78ms 141 1.3602ms 1.3120us 1.6409ms [CUDA memcpy HtoD]
...
======== API calls:
Time (%) Time Calls Avg Min Max Name
92.65% 3.49314s 62 56.341ms 1.8850us 70.771ms cuStreamSynchronize
3.78% 142 0.36ms 1 142.36ms 142.36ms 142.36ms cuDevicePrimaryCtxRetain
...
======== OpenACC (excl):
Time (%) Time Calls Avg Min Max Name
36.27% 1.41470s 20 70.735ms 70.704ms 70.773ms acc_wait@swim-acc-data.f:223
63.3% 1.15449s 18 64.138ms 64.114ms 64.159ms acc_wait@swim-acc-data.f:302
======== CPU profiling result (bottom up):
Time (%) Time Name
59.09% 8.55785s cudbgGetAPIVersion
59.09% 8.55785s start_thread
59.09% 8.55785s clone
25.75% 3.73007s cuStreamSynchronize
25.75% 3.73007s __pgi_uacc_cuda_wait
25.75% 3.73007s __pgi_uacc_computedone
10.38% 1.50269s swim_mod_calc2_
```

#### Options

To display results for only one category, for example, CPU-related results: `--cpu-profiling`.

To display results for the main function first, followed by those for the subordinate functions: `--cpu-profiling-mode top-down`.

```bash
[name@server ~]$ pgprof --cpu-profiling-mode top-down -i a.prof
======== CPU profiling result (top down):
Time (%) Time Name
97.36% 35.2596s main
97.36% 35.2596s MAIN_
32.02% 11.5976s swim_mod_calc3_
29.98% 10.8578s swim_mod_calc2_
25.93% 9.38965s swim_mod_calc1_
6.82% 2.46976s swim_mod_inital_
1.76% 637.36ms __fvd_sin_vex_256
```

To find out which part of the application requires the most execution time: `--cpu-profiling-mode bottom-up`, where the results for each function are followed by those of the calling function, going up to the main function.

```bash
[name@server ~]$ pgprof --cpu-profiling-mode bottom-up -i a.prof
======== CPU profiling result (bottom up):
Time (%) Time Name
32.02% 11.5976s swim_mod_calc3_
32.02% 11.5976s MAIN_
32.02% 11.5976s main
29.98% 10.8578s swim_mod_calc2_
29.98% 10.8578s MAIN_
29.98% 10.8578s main
25.93% 9.38965s swim_mod_calc1_
25.93% 9.38965s MAIN_
25.93% 9.38965s main
3.43% 1.24057s swim_mod_inital_
```

### Graphical Mode

**Data Collection:** Launch the PGI profiler.  Because the PGPROF user interface is based on Java, it should be run on the compute node in an interactive session rather than on the login node, as the latter may not have enough memory (see Java for more information). To enable X11 forwarding, the interactive session can be started with `salloc --x11 ...` (see Interactive Jobs for more information).

Start a new session with `File > New Session`. Select the executable file to profile and add profiling arguments, if any. Click `Next`, then `Finish`.

**Analysis:** In the CPU Details pane, click the "Show the top-down (callers first) call tree view" button.

The data visualization window has four panes:

*   The top pane's right side shows all events according to their execution time.
*   **GPU Details:** Shows the performance of GPU kernels.
*   **CPU Details:** Shows the performance of CPU functions.
*   **Properties:** Detailed information for the function selected in the top pane.


## References

PGPROF is produced by PGI, a subsidiary of NVIDIA Corporation.

*   [Getting Started Guide](link_to_getting_started_guide)
*   [User's Guide](link_to_users_guide)


**(Note:  Please replace `link_to_getting_started_guide` and `link_to_users_guide` with actual links if available.)**
