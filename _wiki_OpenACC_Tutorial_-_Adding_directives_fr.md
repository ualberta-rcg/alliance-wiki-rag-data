# OpenACC Tutorial - Adding Directives (French)

This page is a translated version of the page [OpenACC Tutorial - Adding directives](https://docs.alliancecan.ca/mediawiki/index.php?title=OpenACC_Tutorial_-_Adding_directives&oldid=137998) and the translation is 100% complete.

Other languages: [English](https://docs.alliancecan.ca/mediawiki/index.php?title=OpenACC_Tutorial_-_Adding_directives&oldid=137998), [français](https://docs.alliancecan.ca/mediawiki/index.php?title=OpenACC_Tutorial_-_Adding_directives/fr&oldid=138004)


## Learning Objectives

* Understand the offloading process.
* Understand what an OpenACC directive is.
* Know the difference between `loop` and `kernels` directives.
* Know how to program with OpenACC.
* Understand the concept of aliasing in C/C++.
* Know how to use compiler feedback and avoid false aliasing.


## Contents

1. Transfer to a Graphics Processing Unit (GPU)
2. OpenACC Directives
    2.1 Loops and Kernels
3. The `kernels` Directive
    3.1 Example: Porting a Matrix-Vector Product
        3.1.1 Building with OpenACC
4. Fixing False Loop Dependencies
    4.1 `restrict` Keyword
    4.2 Loop with `independent` Clause
    4.3 Matrix-Vector Product
5. Performance of the Ported Code
    5.1 NVIDIA Visual Profiler
6. The `parallel loop` Directive
7. Differences between `parallel loop` and `kernels`


## Transfer to a Graphics Processing Unit (GPU)

Before porting code to a GPU, you should know that GPUs do not share the same memory as the host CPU.

* Host memory is generally larger but slower than GPU memory.
* A GPU does not have direct access to host memory.
* To use a GPU, data must pass through the PCI bus, whose bandwidth is lower than that of the CPU and GPU.

Therefore, it is extremely important to properly manage transfers between the source memory and the GPU. This process is called *offloading*.


## OpenACC Directives

OpenACC directives are similar to OpenMP directives. In C/C++, they are `pragma` statements, and in Fortran, they are comments. Using directives has several advantages:

First, since the code is minimally affected, modifications can be made incrementally, one `pragma` at a time. This is particularly useful for debugging, as it is easy to identify the precise change that creates the bug.

Second, OpenACC can be disabled at compile time. The `pragma` statements are then treated as comments and are not considered by the compiler, allowing you to compile an accelerated version and a normal version from the same source code.

Third, since the compiler does all the transfer work, the same code can be compiled for different types of accelerators, whether a GPU or SIMD instructions on a CPU. Thus, a change in hardware will simply require updating the compiler without modifying the code.

The code in our example contains two loops: the first initializes two vectors, and the second performs a level 1 vector addition operation.


```c++
#pragma acc kernels
{
  for (int i = 0; i < N; i++) {
    x[i] = 1.0;
    y[i] = 2.0;
  }
  for (int i = 0; i < N; i++) {
    y[i] = a * x[i] + y[i];
  }
}
```

```fortran
!$acc kernels
do i = 1, N
  x(i) = 1.0
  y(i) = 2.0
end do
y(:) = a * x(:) + y(:)
!$acc end kernels
```

In both cases, the compiler identifies two kernels:

* In C/C++, the two kernels are inside each loop.
* In Fortran, the kernels are inside the first loop and inside the implicit loop performed during an operation on arrays.

Note that the OpenACC block is delimited in C/C++ by curly braces; in Fortran, the comment is placed once at the beginning and once at the end, this time with the addition of `end`.


### Loops and Kernels

When the compiler reads the OpenACC `kernels` directive, it analyzes the code to identify sections that can be parallelized. This often corresponds to the body of a loop that has independent iterations. In this case, the compiler delimits the beginning and end of the code body with the `kernel` function. Calls to this function will not be affected by other calls. The function is compiled and can then be executed on an accelerator. Since each call is independent, each of the hundreds of cores of the accelerator can execute the function in parallel for a specific index.

```
LOOP                  KERNEL
for (int i = 0; i < N; i++) {
  C[i] = A[i] + B[i];
}
void kernelName(A, B, C, i) {
  C[i] = A[i] + B[i];
}
```

Calculates sequentially from `i=0` to `i=N-1`, inclusive. Each computing unit executes the function for a single value of `i`.


### The `kernels` Directive

This directive is *descriptive*. The programmer uses it to indicate to the compiler the portions that, in his opinion, can be parallelized. The compiler does what it wants with this information and adopts the strategy that it deems best to execute the code, *including* its sequential execution. In general, the compiler:

1. Analyzes the code to detect parallelism.
2. If it detects parallelism, identifies the data to transfer and decides when to do the transfer.
3. Creates a kernel.
4. Transfers the kernel to the GPU.


Here is an example of this directive:

```c++
#pragma acc kernels
{
  for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
  }
}
```

It is rare for the code to be so simple, and it is necessary to rely on *compiler feedback* to find the portions that it has neglected to parallelize.


### Description or Prescription

If you have already used OpenMP, you will find in OpenACC the principle of *directives*. However, there are important differences between OpenMP and OpenACC directives:

OpenMP directives are basically *prescriptive*. This means that the compiler is forced to perform the parallelization, regardless of whether the effect deteriorates or improves performance. The result is predictable for all compilers. In addition, the parallelization will be done in the same way, regardless of the hardware used to execute the code. On the other hand, the same code may have lower performance, depending on the architecture. It may therefore be preferable, for example, to change the order of the loops. To parallelize code with OpenMP and obtain optimal performance in different architectures, you would need a different set of directives for each architecture.

For their part, several OpenACC directives are *descriptive* in nature. Here, the compiler is free to compile the code in the way it deems best, according to the target architecture. In some cases, the code will not be parallelized at all. The *same code* executed on a GPU or on a CPU can give different binary code. This means that performance may vary depending on the compiler, and that newer generation compilers will be more efficient, especially in the presence of new hardware.


### Example: Porting a Matrix-Vector Product

For our example, we use code from the [Github repository](https://github.com/openacc-standard/OpenACC-Examples/tree/master/Tutorials/AddingDirectives), specifically a portion of code from the file `cpp/matrix_functions.h`. The equivalent Fortran code is in the `matvec` subroutine contained in the file `matrix.F90`. The C++ code is as follows:

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

The *first change* to make to the code is to add the `kernels` directive to try to make it run on the GPU. For now, we don't have to worry about data transfer or provide information to the compiler.

```c++
#pragma acc kernels
{
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
}
```


### Building with OpenACC

NVidia compilers use the `-acc` option to allow compilation for an accelerator. We use the `-gpu=managed` sub-option to tell the compiler that we want to use *managed memory* to simplify data transfer to and from the device; we will not use this option in a later example. We also use the `-fast` option for optimization.

```bash
[name@server ~]$ nvc++ -fast -Minfo=accel -acc -gpu=managed main.cpp -o challenge
...
matvec (const matrix &, const vector &, const vector &):
23, include "matrix_functions.h"
30, Generating implicit copyin (cols[:],row_offsets[:num_rows+1],Acoefs[:]) [if not already present]
Generating implicit copyout (ycoefs[:num_rows]) [if not already present]
Generating implicit copyin (xcoefs[:]) [if not already present]
31, Loop carried dependence of ycoefs-> prevents parallelization
Loop carried backward dependence of ycoefs-> prevents vectorization
Complex loop carried dependence of Acoefs->,xcoefs-> prevents parallelization
Generating NVIDIA GPU code
31, #pragma acc loop seq
35, #pragma acc loop vector(128) /* threadIdx.x */
Generating implicit reduction (+:sum)
35, Loop is parallelizable
```

The result shows that the outer loop on line 31 could not be parallelized by the compiler. In the next section, we explain how to handle these dependencies.


## Fixing False Loop Dependencies

Even when the programmer knows that a loop can be parallelized, the compiler may not notice it. A common case in C/C++ is known as *pointer aliasing*. Unlike Fortran, C/C++ does not have arrays as such, but rather pointers. The concept of aliasing applies to two pointers pointing to the same memory. If the compiler does not know that pointers are not aliases, it must assume so. In the previous example, it is clear why the compiler could not parallelize the loop. Assuming that the pointers are identical, there is necessarily a dependence of the loop iterations.


### `restrict` Keyword

One way to tell the compiler that pointers are *not* aliases is to use the `restrict` keyword, introduced for this purpose in C99. There is still no standard way to do this in C++, but each compiler has its own keyword. Depending on the compiler, you can use `__restrict` or `__restrict__`. Portland Group and NVidia compilers use `__restrict`. To find out why there is no standard in C++, see [this document](https://isocpp.org/wiki/faq/pointers#restrict). This concept is important for OpenACC as well as for all C/C++ programming, because compilers can perform several other optimizations if pointers are not aliases. Note that the keyword is placed *after* the pointer since it refers to the latter and not to the type; in other words, the declaration must read `float * __restrict A;` rather than `float __restrict * A;`.


### Using the `restrict` Keyword

By declaring a pointer as `restrict`, we ensure that only this pointer or a derived value (such as `ptr + 1`) can access the object it refers to, for the lifetime of the pointer. This is a guarantee that the programmer gives to the compiler; if the programmer fails to meet his obligation, the behavior is undefined. For more information, see the Wikipedia article on [restrict](https://en.wikipedia.org/wiki/Restrict).


### Loop with `independent` Clause

Another way to ensure that the compiler treats loops independently is to specify it explicitly with the `independent` clause. Like any other *prescriptive* directive, the compiler is obliged to use it, and the analysis it could do will not be considered. Taking the example from the "The `kernels` Directive" section above, we have:

```c++
#pragma acc kernels
{
  #pragma acc loop independent
  for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
  }
}
```


### Matrix-Vector Product

Let's go back to the matrix-vector product example above. Our recommendation to avoid false aliasing is to define the pointers as restricted by replacing the code in `matrix_functions.h`.

Replace:

```c++
double * Acoefs = A.coefs;
double * xcoefs = x.coefs;
double * ycoefs = y.coefs;
```

With:

```c++
double * __restrict Acoefs = A.coefs;
double * __restrict xcoefs = x.coefs;
double * __restrict ycoefs = y.coefs;
```

Note that the other pointers do not need to be restricted since the compiler does not report them as causing problems. By recompiling with the changes we have just made, the compiler issues the following message:

```bash
[name@server ~]$ nvc++ -fast -Minfo=accel -acc -gpu=managed main.cpp -o challenge
matvec (const matrix &, const vector &, const vector &):
23, include "matrix_functions.h"
27, Generating implicit copyout (ycoefs[:num_rows]) [if not already present]
Generating implicit copyin (xcoefs[:],row_offsets[:num_rows+1],Acoefs[:],cols[:]) [if not already present]
30, Loop is parallelizable
Generating Tesla code
30, #pragma acc loop gang /* blockIdx.x */
34, #pragma acc loop vector(128) /* threadIdx.x */
Generating implicit reduction (+:sum)
34, Loop is parallelizable
```


## Performance of the Ported Code

Now that the code is ported to the GPU, we can analyze its performance and verify that the results are correct. Executing the original code on a GPU node produces this:

```bash
[name@server ~]$ ./cg.x
Rows: 8120601, nnz: 218535025
Iteration: 0, Tolerance: 4.0067e+08
Iteration: 10, Tolerance: 1.8772e+07
Iteration: 20, Tolerance: 6.4359e+05
Iteration: 30, Tolerance: 2.3202e+04
Iteration: 40, Tolerance: 8.3565e+02
Iteration: 50, Tolerance: 3.0039e+01
Iteration: 60, Tolerance: 1.0764e+00
Iteration: 70, Tolerance: 3.8360e-02
Iteration: 80, Tolerance: 1.3515e-03
Iteration: 90, Tolerance: 4.6209e-05
Total Iterations: 100
Total Time: 29.894881s
```

Here is the result for the OpenACC version:

```bash
[name@server ~]$ ./challenge
Rows: 8120601, nnz: 218535025
Iteration: 0, Tolerance: 4.0067e+08
Iteration: 10, Tolerance: 1.8772e+07
Iteration: 20, Tolerance: 6.4359e+05
Iteration: 30, Tolerance: 2.3202e+04
Iteration: 40, Tolerance: 8.3565e+02
Iteration: 50, Tolerance: 3.0039e+01
Iteration: 60, Tolerance: 1.0764e+00
Iteration: 70, Tolerance: 3.8360e-02
Iteration: 80, Tolerance: 1.3515e-03
Iteration: 90, Tolerance: 4.6209e-05
Total Iterations: 100
Total Time: 115.068931s
```

The results are correct; however, far from gaining speed, the operation took almost four times longer! Let's use the NVidia Visual Profiler (`nvvp`) to see what's going on.


### NVIDIA Visual Profiler

NVIDIA Visual Profiler (NVVP) is a graphical profiler for OpenACC applications. It is an analysis tool for *code written with OpenACC and CUDA C/C++ directives*. Consequently, if the executable does not use the GPU, this profiler will not provide any results.

When X11 is redirected to an X-Server or when you are using a Linux desktop environment (also via JupyterHub with 2 CPU cores, 5000M of memory and 1 GPU), you can launch NVVP from a terminal:

```bash
[name@server ~]$ module load cuda/11.7 java/1.8
[name@server ~]$ nvvp
```

After displaying the NVVP launch window, you must enter the Workspace directory that will be used for temporary files. In the suggested path, replace `home` with `scratch` and click `OK`.

Select `File > New Session` or click the corresponding button in the toolbar. Click the `Browse` button to the right of the `File` field for the path. Change the directory if necessary. Select an executable built with code written with OpenACC and CUDA C/C++ directives. Under the `Arguments` field, select the `Profile current process only` option. Click `Next >` to see the other profiling options. Click `Finish` to start profiling the executable.

To do this, follow these steps:

1. Launch `nvvp` with the command `nvvp &` (the `&` symbol commands the launch in the background).
2. Select `File -> New Session`.
3. In the `File:` field, look for the executable (named `challenge` in our example).
4. Click `Next` until you can click `Finish`.

The program is executed, and we obtain a chronological table of the execution (see the image). We note that the data transfer between the start and the arrival occupies most of the execution time, which is frequent when code is ported from a CPU to a GPU. We will see how this can be improved in the next part, *Data Movement*.


## The `parallel loop` Directive

With the `kernels` directive, it is the compiler that does all the analysis; this is a *descriptive* approach to porting code. OpenACC also offers a *prescriptive* approach with the `parallel` directive, which can be combined with the `loop` directive as follows:

```c++
#pragma acc parallel loop
for (int i = 0; i < N; i++) {
  C[i] = A[i] + B[i];
}
```

Since `parallel loop` is a *prescriptive* directive, the compiler is forced to execute the loop in parallel. This means that the `independent` clause mentioned above is implicit within a parallel region.

To use this directive in our matrix-vector product example, we need the `private` and `reduction` clauses to manage the data flow in the parallel region.

With the `private` clause, a copy of the variable is made for each iteration of the loop; the value of the variable is thus independent of the other iterations.

With the `reduction` clause, the values of the variable in each iteration are *reduced* to a single value. The clause is used with addition (+), multiplication (*), maximum (max), and minimum (min) operations, among others.

These clauses are not necessary with the `kernels` directive since it does the work for you.

Let's take the matrix-vector product example again with the `parallel loop` directive:

```c++
#pragma acc parallel loop
for (int i = 0; i < num_rows; i++) {
  double sum = 0;
  int row_start = row_offsets[i];
  int row_end = row_offsets[i + 1];
  #pragma acc loop reduction(+:sum)
  for (int j = row_start; j < row_end; j++) {
    unsigned int Acol = cols[j];
    double Acoef = Acoefs[j];
    double xcoef = xcoefs[Acol];
    sum += Acoef * xcoef;
  }
  ycoefs[i] = sum;
}
```

Compilation produces the following message:

```bash
[name@server ~]$ nvc++ -fast -Minfo=accel -acc -gpu=managed main.cpp -o challenge
matvec (const matrix &, const vector &, const vector &):
23, include "matrix_functions.h"
27, Accelerator kernel generated
Generating Tesla code
29, #pragma acc loop gang /* blockIdx.x */
34, #pragma acc loop vector(128) /* threadIdx.x */
Sum reduction generated for sum
27, Generating copyout (ycoefs[:num_rows])
Generating copyin (xcoefs[:],Acoefs[:],cols[:],row_offsets[:num_rows+1])
34, Loop is parallelizable
```


## Differences between `parallel loop` and `kernels`

| Feature          | `parallel loop`           | `kernels`                |
|-----------------|---------------------------|---------------------------|
| Code Integrity   | Programmer's responsibility | Compiler's responsibility |
| Parallelization  | Explicit, prescriptive     | Implicit, descriptive      |
| OpenMP Similarity | Similar                    | Different                  |
| Code Optimization| Less compiler optimization | More compiler optimization |


Both approaches are valid, and their performance is comparable.


## Exercise: Using `kernels` or `parallel loop`

Modify the `matvec`, `waxpby`, and `dot` functions. You can use either `kernels` or `parallel loop`. The solution is in the `step1` directories of Github*.

Modify the Makefile by adding `-acc -gpu=managed` and `-Minfo=accel` to the compiler flags.

* Previous page, Profilers
* Back to the beginning of the tutorial
* Next page, Data Movement


Retrieved from "[https://docs.alliancecan.ca/mediawiki/index.php?title=OpenACC_Tutorial_-_Adding_directives/fr&oldid=138004](https://docs.alliancecan.ca/mediawiki/index.php?title=OpenACC_Tutorial_-_Adding_directives/fr&oldid=138004)"
