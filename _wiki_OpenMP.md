# OpenMP

**Other languages:** English, français

## Description

OpenMP (Open Multi-Processing) is an application programming interface (API) for shared memory parallel computing.  It's supported on numerous platforms, including Linux and Windows, and is available for C/C++ and Fortran programming languages. The API consists of a set of directives, a software library, and environment variables.

OpenMP allows the development of fine-grained parallel applications on a multicore machine while preserving the structure of the serial code. Although only one program instance runs, it can execute multiple subtasks in parallel. Directives inserted into the program control whether a section executes in parallel and, if so, the work distribution among subtasks.  These directives are usually non-intrusive; a compiler without OpenMP support can still compile the program, allowing serial execution.

OpenMP relies on the notion of **threads**. A thread is like a lightweight process or a "virtual processor, operating serially," and can be defined as the smallest unit of work/processing that an operating system can schedule.  For a programmer, five threads virtually correspond to five cores performing computations in parallel.  The number of threads is independent of the number of physical cores; for example, two cores can run a program with ten threads. The operating system manages core time sharing among threads.

Conversely, one thread cannot be executed by two processors simultaneously. To utilize four cores, at least four threads must be created. Using more threads than available cores might be advantageous in some cases, but matching thread and core counts is usual practice.

Another crucial aspect of threads is **synchronization**. When multiple threads perform computations concurrently, the execution order is unpredictable. If order matters for code correctness, the programmer must use OpenMP synchronization directives.  The precise thread distribution over cores is also unknown to the programmer (though thread affinity capabilities offer some control).

When parallelizing a program using OpenMP (or any other technique), consider the program's **scalability** (its ability to run efficiently in parallel). After parallelization and verification, perform a scaling analysis to understand its parallel performance.

[A tutorial for getting started with OpenMP under Linux](link_to_tutorial_needed)


## Compilation

For most compilers, compiling an OpenMP program involves adding a command-line option to the compilation flags. For GNU compilers (GCC), it's `-fopenmp`; for Intel, depending on the version, it might be `-qopenmp`, `-fopenmp`, or `-openmp`. Refer to your compiler's documentation.


## Directives

OpenMP directives are inserted into Fortran programs using sentinels (keywords placed immediately after a comment symbol):

```
!$OMP directive
c$OMP directive
C$OMP directive
*$OMP directive
```

In C, directives use a pragma construct:

```c
#pragma omp directive
```

### OpenMP Directives

| Fortran                     | C, C++                                      |
|-----------------------------|----------------------------------------------|
| `!$OMP PARALLEL [clause,...]` | `#pragma omp parallel [clause,...]`         |
| `block`                     | `structured-block`                           |
| `!$OMP END PARALLEL`        |                                              |
| `!$OMP DO [clause,...]`      | `#pragma omp for [clause,...]`              |
| `do_loop`                   | `for-loop`                                  |
| `!$OMP END DO`              |                                              |
| `!$OMP SECTIONS [clause,...]` | `#pragma omp sections [clause,...] { ... }` |
| `!$OMP SECTION`             | `#pragma omp section`                        |
| `block`                     | `structured-block`                           |
| `!$OMP END SECTIONS [NOWAIT]`|                                              |
| `!$OMP SINGLE [clause,...]`  | `#pragma omp single [clause,...]`           |
| `block`                     | `structured-block`                           |
| `!$OMP END SINGLE [NOWAIT]` |                                              |
| `!$OMP PARALLEL DO [clause,...]` | `#pragma omp parallel for [clause,...]`     |
| `DO_LOOP`                   | `for-loop`                                  |
| `!$OMP PARALLEL SECTIONS [clause,...]` | `#pragma omp parallel sections [clause,...] { ... }` |
| `!$OMP MASTER`              | `#pragma omp master`                         |
| `block`                     | `structured-block`                           |
| `!$OMP END MASTER`           |                                              |
| `!$OMP CRITICAL [(name)]`    | `#pragma omp critical [(name)]`             |
| `block`                     | `structured-block`                           |
| `!$OMP END CRITICAL [(name)]`|                                              |
| `!$OMP BARRIER`             | `#pragma omp barrier`                        |
| `!$OMP ATOMIC`              | `#pragma omp atomic`                         |
| `expression_statement`      | `expression-statement`                       |
| `!$OMP FLUSH [(list)]`      | `#pragma omp flush [(list)]`                |
| `!$OMP ORDERED`             | `#pragma omp ordered`                        |
| `block`                     | `structured-block`                           |
| `!$OMP END ORDERED`         |                                              |
| `!$OMP THREADPRIVATE(/cb/[,/cb/]...)` | `#pragma omp threadprivate (list)`         |


**Clauses:**

| Clause           | Fortran          | C, C++           |
|-------------------|-------------------|--------------------|
| PRIVATE (list)    | `PRIVATE (list)`   | `private (list)`    |
| SHARED (list)    | `SHARED (list)`   | `shared (list)`    |
| SHARED \| NONE   | `SHARED \| NONE` | `shared \| none`   |
| FIRSTPRIVATE (list)| `FIRSTPRIVATE (list)` | `firstprivate (list)` |
| LASTPRIVATE (list)| `LASTPRIVATE (list)` | `lastprivate (list)` |
| REDUCTION (op: list) | `REDUCTION (op: list)` | `reduction (op: list)` |
| IF (scalar_logical_expression) | `IF (scalar_logical_expression)` | `if (scalar-expression)` |
| COPYIN (list)     | `COPYIN (list)`    | `copyin (list)`     |
| NOWAIT            | `NOWAIT`          | `nowait`            |


## Environment

Several environment variables influence OpenMP program execution:

*   `OMP_NUM_THREADS`
*   `OMP_SCHEDULE`
*   `OMP_DYNAMIC`
*   `OMP_STACKSIZE`
*   `OMP_NESTED`

These can be set/modified using a UNIX command like:

```bash
export OMP_NUM_THREADS=12
```

`OMP_NUM_THREADS` is usually set to the number of reserved cores per machine, though this might differ for hybrid OpenMP/MPI applications.  `OMP_SCHEDULE` controls loop (and parallel section) distribution. The default value depends on the compiler and can be added to the source code. Possible values are `static,n`, `dynamic,n`, `guided,n`, or `auto`. For the first three, `n` is the number of iterations managed by each thread.  `static` distributes iterations at the beginning; `dynamic` distributes them during execution based on thread execution time; `guided` starts with a large iteration count, gradually shrinking it; `auto` lets the compiler/library decide.  `dynamic`, `guided`, and `auto` offer better load balancing but make it harder to predict thread core assignment and memory access, which can be problematic in NUMA architectures.

`OMP_STACKSIZE` specifies the stack size for each thread (the main thread's stack size comes from the execution shell).  The implied value is 4M if not set. Insufficient stack memory can cause crashes.

Other environment variables exist, some compiler-specific (Intel's start with `KMP_`, GNU's with `GOMP_`).  For optimal memory access, set `OMP_PROC_BIND` and affinity variables (`KMP_AFFINITY` for Intel, `GOMP_CPU_AFFINITY` for GNU) to prevent OS thread movement between processors, especially important in NUMA architectures.

[Intel compiler environment variables](intel_website_link_needed)
[GNU compiler environment variables](gnu_website_link_needed)


## Example

**Hello world** example:

**C:**

```c
#include <stdio.h>
#include <omp.h>

int main() {
  #pragma omp parallel
  {
    printf("Hello world from thread %d out of %d\n", omp_get_thread_num(), omp_get_num_threads());
  }
  return 0;
}
```

**Fortran:**

```fortran
program hello
  implicit none
  integer :: omp_get_thread_num, omp_get_num_threads
!$omp parallel
  print *, 'Hello world from thread', omp_get_thread_num(), &
         'out of', omp_get_num_threads()
!$omp end parallel
end program hello
```

**C Compilation and Execution:**

```bash
gcc -O3 -fopenmp ompHello.c -o ompHello
export OMP_NUM_THREADS=4
./ompHello
```

**Fortran Compilation and Execution:**

```bash
gfortran -O3 -fopenmp ompHello.f90 -o fomphello
export OMP_NUM_THREADS=4
./fomphello
```

[Example of submitting an OpenMP job](link_to_job_submission_example_needed)


## References

*   [Lawrence Livermore National Labs OpenMP tutorial](llnl_tutorial_link_needed)
*   [OpenMP.org specifications, reference cards, and examples](openmp_org_link_needed)

