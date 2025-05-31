# OpenMP

This page is a translated version of the page OpenMP and the translation is 100% complete.

Other languages: English, français


## Description

OpenMP (for Open Multi-Processing) is a programming interface (API) for parallel computing on shared memory architecture. The OpenMP interface is supported on many platforms including Unix and Windows, for the C/C++ and Fortran programming languages. It consists of a set of directives, a software library, and environment variables.

OpenMP allows for the rapid development of fine-grained parallel applications while remaining close to sequential code. With a single instance of the program, multiple subtasks can be executed in parallel. Directives inserted into the program determine whether a section of the program executes in parallel; in this case, the directives also handle the distribution of work across multiple threads. Thus, a compiler that does not understand the directives can still compile the program, which can then be executed serially.

The OpenMP interface is based on the concept of execution threads, well known in object-oriented programming. An execution thread is a bit like a virtual processor operating sequentially; it is the smallest unit of work/calculation that an operating system can program. From the programmer's point of view, five threads are virtually equivalent to five processors that can perform calculations in parallel. It is important to understand that the number of threads is not associated with the number of physical processors available: for example, two processors can execute a program with 10 threads. It is the operating system that is responsible for sharing the time of the available processors between the threads.

However, it is not possible to execute the same thread on multiple processors; if you have, for example, four processors, you will need to use at least four threads to take advantage of all the computing power. In some cases, it might be advantageous to use more threads than processors; however, the number of threads is usually equal to the number of processors.

Another important point concerning threads is synchronization. When several threads of the same program perform calculations at the same time, one cannot absolutely presume the order in which they will be performed. If a determined order is necessary to ensure the integrity of the code, the programmer will use the OpenMP synchronization directives. The exact method of distribution on the threads remains unknown to the programmer, but there are control features (see processor affinity).

To parallelize a program with OpenMP or any other technique, it is important to consider the program's ability to execute in parallel, which we will call its scalability. After parallelizing your software and you are satisfied with its quality, we recommend that you perform a scalability analysis to understand its performance.

For information on using OpenMP under Linux, see this tutorial.


## Compilation

For most compilers, compiling OpenMP code is simply done by adding a compilation option. For GNU compilers (GCC), this is the `-fopenmp` option; for Intel compilers, depending on the version, it may be `-qopenmp`, `-fopenmp`, or `-openmp`. For other compilers, check their respective documentation.


## Directives

OpenMP directives are inserted into Fortran programs using sentinels. A sentinel is a keyword placed immediately after the symbol indicating a comment. For example:

```
!$OMP directive
c$OMP directive
C$OMP directive
*$OMP directive
```

In C, directives are inserted using a pragma:

```
#pragma omp directive
```

### OpenMP Directives

| Fortran                     | C, C++                                      |
|-----------------------------|-----------------------------------------------|
| `!$OMP PARALLEL [clause, clause,…]` <br> `block` <br> `!$OMP END PARALLEL` | `#pragma omp parallel [clause, clause,…]` <br> `structured-block` |
| `!$OMP DO [ clause, clause,… ]` <br> `do_loop` <br> `!$OMP END DO`         | `#pragma omp for [ clause, clause,… ]` <br> `for-loop`             |
| `!$OMP SECTIONS [clause, clause,…]` <br> `!$OMP SECTION` <br> `block` <br> `!$OMP SECTION` <br> `block` <br> `!$OMP END SECTIONS [NOWAIT]` | `#pragma omp sections [clause, clause,…] {` <br> `[ #pragma omp section ]` <br> `structured-block` <br> `[ #pragma omp section ]` <br> `structured-block` <br> `}` |
| `!$OMP SINGLE [clause, clause,…]` <br> `block` <br> `!$OMP END SINGLE [NOWAIT]` | `#pragma omp single [clause, clause,…]` <br> `structured-block` |
| `!$OMP PARALLEL DO [clause, clause,…]` <br> `DO_LOOP` <br> `[ !$OMP END PARALLEL DO ]` | `#pragma omp parallel for [clause, clause,…]` <br> `for-loop`             |
| `!$OMP PARALLEL SECTIONS [clause, clause,…]` <br> `!$OMP SECTION` <br> `block` <br> `!$OMP SECTION` <br> `block` <br> `!$OMP END PARALLEL SECTIONS` | `#pragma omp parallel sections [clause, clause,…] {` <br> `[ #pragma omp section ]` <br> `structured-block` <br> `[ #pragma omp section ]` <br> `structured-block` <br> `}` |
| `!$OMP MASTER` <br> `block` <br> `!$OMP END MASTER` | `#pragma omp master` <br> `structured-block` |
| `!$OMP CRITICAL [(name)]` <br> `block` <br> `!$OMP END CRITICAL [(name)]` | `#pragma omp critical [(name)]` <br> `structured-block` |
| `!$OMP BARRIER` | `#pragma omp barrier` |
| `!$OMP ATOMIC` <br> `expresion_statement` | `#pragma omp atomic` <br> `expression-statement` |
| `!$OMP FLUSH [(list)]` | `#pragma omp flush [(list)]` |
| `!$OMP ORDERED` <br> `block` <br> `!$OMP END ORDERED` | `#pragma omp ordered` <br> `structured-block` |
| `!$OMP THREADPRIVATE( /cb/[, /cb/]…)` | `#pragma omp threadprivate ( list )` |


#### Clauses

| Fortran                     | C, C++          |
|-----------------------------|-------------------|
| `PRIVATE ( list )`          | `private ( list )` |
| `SHARED ( list )`           | `shared ( list )`  |
| `SHARED \| NONE )`          | `shared \| none )` |
| `FIRSTPRIVATE ( list )`     | `firstprivate ( list )` |
| `LASTPRIVATE ( list )`      | `lastprivate ( list )` |
| `reduction ( op : list )` | `reduction ( op : list )` |
| `IF ( scalar_logical_expression )` | `if ( scalar-expression )` |
| `COPYIN ( list )`          | `copyin ( list )` |
| `NOWAIT`                    | `nowait`          |


## Environment

Certain environment variables affect the execution of an OpenMP program:

*   `OMP_NUM_THREADS`
*   `OMP_SCHEDULE`
*   `OMP_DYNAMIC`
*   `OMP_STACKSIZE`
*   `OMP_NESTED`

Variables are defined or modified with a Unix command such as:

```bash
[name@server ~]$ export OMP_NUM_THREADS=12
```

In most cases, you will want to specify with `OMP_NUM_THREADS` the number of cores reserved per machine. However, this may be different for a hybrid OpenMP/MPI application.

Another important variable is `OMP_SCHEDULE`. This controls how loops (and more generally parallel sections) are distributed. The default value depends on the compiler and can be defined in the source code. Possible values are `static,n`, `dynamic,n`, `guided,n`, and `auto`, where `n` indicates the number of iterations managed by each execution thread.

*   In the case of `static`, the number of iterations is fixed and the iterations are distributed at the beginning of the parallel section.
*   In the case of `dynamic`, the number of iterations is fixed, but the iterations are distributed during execution depending on the time required by each thread to execute its iterations.
*   In the case of `guided`, `n` indicates the minimum number of iterations. The number of iterations is first chosen "large", then decreases dynamically as the remaining number of iterations decreases.
*   For `auto` mode, the compiler and library are free to make choices.

The advantage of `dynamic`, `guided`, and `auto` cases is that they theoretically allow for better balancing of execution threads, since they adjust dynamically according to the time required by each thread. On the other hand, the disadvantage is that you do not know in advance which processor will execute which thread, and which memory it must access. It is thus impossible with these types of scheduling to predict the affinity between memory and the processor executing the calculation. This can be particularly problematic in a NUMA architecture.

The environment variable `OMP_STACKSIZE` defines the stack size for each of the threads created during OpenMP execution. Note that the main OpenMP thread (the one that executes the sequential part of the program) gets its stack size from the interpreter (shell) while `OMP_STACKSIZE` affects each of the additional threads created during execution. If this variable is not defined, the value will be 4MB. If your code does not have enough memory for the stack, it may terminate abnormally due to a segmentation fault.

Other environment variables are also available: some are compiler-specific while others are more generic. See the list of variables for Intel compilers and for GNU compilers.

Intel compiler-specific environment variables begin with `KMP_` while GNU-specific ones begin with `GOMP_`. For optimal memory access performance, set the `OMP_PROC_BIND` variables and the affinity variables `KMP_AFFINITY` for Intel and `GOMP_CPU_AFFINITY` for GNU. This prevents OpenMP execution threads from moving from one processor to another, which is particularly important in a NUMA architecture.


## Example

Here is a "hello world" example that shows the use of OpenMP.

**C**

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

**Fortran**

```fortran
program hello
implicit none
integer omp_get_thread_num, omp_get_num_threads
!$omp parallel
print *, 'Hello world from thread', omp_get_thread_num(), &
'out of', omp_get_num_threads()
!$omp end parallel
end program hello
```

The C code is compiled and executed as follows:

```bash
litai10:~$ gcc -O3 -fopenmp ompHello.c -o ompHello
litai10:~$ export OMP_NUM_THREADS=4
litai10:~$ ./ompHello
Hello world from thread 0 out of 4
Hello world from thread 2 out of 4
Hello world from thread 1 out of 4
Hello world from thread 3 out of 4
```

The Fortran 90 code is compiled and executed as follows:

```bash
litai10:~$ gfortran -O3 -fopenmp ompHello.f90 -o fomphello
litai10:~$ export OMP_NUM_THREADS=4
litai10:~$ ./fomphello
Hello world from thread           0 out of           4
Hello world from thread           2 out of           4
Hello world from thread           1 out of           4
Hello world from thread           3 out of           4
```

To learn how to submit an OpenMP task, see the Multithreaded task or OpenMP task section of the Running tasks page.


## References

*   Lawrence Livermore National Laboratory: OpenMP documentation.
*   OpenMP.org: specifications, cheat sheets for C/C++ and Fortran interfaces, examples.

