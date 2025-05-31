# Debugging and Profiling Tools

An important step in the software development process, particularly for compiled languages like Fortran and C/C++, involves using a debugger to detect and identify the origin of runtime errors (e.g., memory leaks, floating-point exceptions). Once the program's correctness is assured, the next step is profiling the software. This uses a profiler to determine the percentage of total execution time each section of the source code is responsible for when run with a representative test case.  A profiler provides information such as how many times a particular function is called, which other functions call it, and the average time (in milliseconds) each invocation of that function takes.


## Debugging and Profiling Tools

Our national clusters offer various debugging and profiling tools, both command-line and GUI-based (requiring an X11 connection).  Debugging sessions should be conducted using an interactive job, not on a login node.


### GNU debugger (gdb)

Please see the [GDB page](link-to-gdb-page).


### PGI debugger (pgdb)

Please see the [Pgdbg page](link-to-pgdbg-page).


### ARM debugger (ddt)

Please see the [ARM software page](link-to-arm-software-page).


### GNU profiler (gprof)

Please see the [Gprof page](link-to-gprof-page).


### Scalasca profiler (scalasca, scorep, cube)

Scalasca is an open-source, GUI-driven parallel profiling toolset. It's currently available for `gcc 9.3.0` and `OpenMPI 4.0.3`, with AVX2 or AVX512 architecture.  Its environment can be loaded with:

```bash
module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 scalasca
```

The current version is 2.5. More information can be found in the 2.x user guide, which includes workflow examples [here](link-to-scalasca-guide).


### PGI profiler (pgprof)

Please see the [Pgprof page](link-to-pgprof-page).


### Nvidia command-line profiler (nvprof)

Please see the [nvprof page](link-to-nvprof-page).


### Valgrind

Please see the [Valgrind page](link-to-valgrind-page).


## External References

*   [Introduction to (Parallel) Performance from SciNet](link-to-scinet-performance-intro)
*   [Code profiling on Graham](link-to-graham-profiling-video), video, 54 minutes.


**(Note:  Replace bracketed placeholders like `[link-to-gdb-page]` with actual links.)**
