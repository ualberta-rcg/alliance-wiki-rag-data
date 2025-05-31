# BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra PACK)

BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra PACK) are two of the most widely used libraries in high-performance computing for research. They are used for vector and matrix operations that are common in many algorithms.  They are more than just libraries; they define a standard programming interface—a set of function definitions that can be called to perform calculations such as the dot product of two double-precision vectors or the product of two Hermitian matrices of complex numbers.

In addition to the Netlib reference implementation, several other implementations of these two standards exist. The performance of these implementations can vary significantly depending on the software used; for example, it is well established that the implementation provided by the Intel Math Kernel Library (Intel MKL) offers better performance with Intel processors in most situations. However, Intel owns this implementation, and in some cases, it is preferable to use the free OpenBLAS implementation or BLIS, which performs better with AMD processors. Two projects that are no longer maintained are gotoblas and ATLAS BLAS.

Unfortunately, it is usually necessary to recompile software to determine which implementation is the most efficient for a particular code and hardware configuration. This is problematic when creating a portable software environment that can run on multiple clusters. To remedy this, you can use FlexiBLAS, an abstraction layer that allows you to change the implementation of BLAS and LAPACK at runtime rather than during compilation.


## Contents

1. Choosing an Implementation
2. Compiling with FlexiBLAS
3. Changing the BLAS/LAPACK Implementation for Execution
4. Using Intel MKL Directly


## Choosing an Implementation

For several years, we recommended using Intel MKL as the reference implementation since our clusters only had Intel processors. Since the commissioning of Narval, which has AMD processors, we now recommend compiling the code with FlexiBLAS.  Our FlexiBLAS module configuration ensures that BLIS is used with AMD processors and Intel MKL with other processor types, which usually offers optimal performance.


## Compiling with FlexiBLAS

Since FlexiBLAS is relatively recent, not all systems will recognize it by default.  This can usually be overcome by configuring the compilation options to use `-lflexiblas` for BLAS and LAPACK. These options are usually in your Makefile; otherwise, you can pass them as parameters to `configure` or `cmake`. CMake versions from 3.19 can automatically find FlexiBLAS; to use one of these versions, load the module `cmake/3.20.1` or `cmake/3.21.4`.


## Changing the BLAS/LAPACK Implementation for Execution

The main advantage of FlexiBLAS is the ability to change the background implementation for execution by configuring the environment variable `FLEXIBLAS`. As of January 2022, the available implementations are `netlib`, `blis`, `imkl`, and `openblas`, but you can get the complete list with the command:

```bash
[name@server ~]$ flexiblas list
```

On Narval, we have configured `FLEXIBLAS=blis` to use BLIS by default. On other clusters, `FLEXIBLAS` is not defined, and Intel MKL is used by default.


## Using Intel MKL Directly

Even though we recommend using FlexiBLAS, it is still possible to use Intel MKL directly. With an Intel compiler (e.g., `ifort`, `icc`, `icpc`), the solution is to replace `-lblas` and `-llapack` in the compiler and linker options with `-mkl=sequential` to avoid using internal threads or `-mkl` to use internal threads.

This ensures that the MKL implementation of BLAS/LAPACK is used. See [the documentation on options](link_to_documentation_here).

With compilers other than Intel's, such as the GNU Compiler Collection (GCC), you must explicitly list the required MKL libraries during the linking stage. Intel provides the MKL Link Advisor tool to help you determine the compilation and linking options.

MKL Link Advisor is also useful if you get `undefined reference` errors with Intel compilers and `-mkl`.
