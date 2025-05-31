# Debugging and Profiling

An important step in software development, particularly in Fortran and C/C++, is using debugging software to detect and identify the origin of runtime errors (e.g., memory leaks, floating-point exceptions, etc.). After eliminating errors, the next step is to profile the program with profiling software to determine the percentage of execution time for each section of the source code with a representative test scenario. A profiler can provide information on the number of times a function is called, which functions call it, and how many milliseconds each call costs on average.


## Tools

Our clusters offer a choice of debuggers and profilers to perform the work in graphical mode via X11 connection or in command-line mode. Debugging should be done in an interactive task and not in a login node.

### GNU Debugger (gdb)

See [GDB](GDB_link_here).  *(Replace GDB_link_here with the actual link)*

### PGI Debugger (pgdb)

See [PGDBG](PGDBG_link_here). *(Replace PGDBG_link_here with the actual link)*

### ARM Debugger (ddt)

See [ARM](ARM_link_here). *(Replace ARM_link_here with the actual link)*

### GNU Profiler (gprof)

See [Gprof](Gprof_link_here). *(Replace Gprof_link_here with the actual link)*

### Scalasca Profiler (scalasca, scorep, cube)

Scalasca is an open-source suite of tools with a graphical interface for parallel profiling with GPUs. These tools are available for `gcc 9.3.0` and `OpenMPI 4.0.3`, in AVX2 and AVX512 architectures. Load the environment with:

```bash
module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 scalasca
```

The current version is 2.5.  More information and workflow examples can be found in the [Scalasca User Guide](Scalasca_User_Guide_link_here). *(Replace Scalasca_User_Guide_link_here with the actual link)*


### PGI Profiler (pgprof)

See [Pgprof](Pgprof_link_here). *(Replace Pgprof_link_here with the actual link)*

### Nvidia Command-Line Profiler (nvprof)

See [nvprof](nvprof_link_here). *(Replace nvprof_link_here with the actual link)*

### Valgrind

See [Valgrind](Valgrind_link_here). *(Replace Valgrind_link_here with the actual link)*


## Other References

* [Introduction to (Parallel) Performance](Introduction_to_Parallel_Performance_link_here), SciNet *(Replace Introduction_to_Parallel_Performance_link_here with the actual link)*
* [Code profiling on Graham](Code_profiling_on_Graham_link_here) (54-minute video), SHARCNET *(Replace Code_profiling_on_Graham_link_here with the actual link)*


**(Note:  All bracketed placeholders like `[GDB](GDB_link_here)` need to be replaced with the correct links.)**
