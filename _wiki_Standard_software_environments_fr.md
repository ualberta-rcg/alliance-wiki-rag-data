# Standard Software Environments

This document describes the standard software environments used by our technical team.  These environments are made available through a set of modules that allow switching between different versions of software packages.  The modules are organized in a tree structure, with the trunk containing the same utilities as those offered in standard Linux environments. The main branches are compiler versions, with sub-branches for each MPI or CUDA version.

A standard software environment is a specific combination of compiler and MPI modules grouped into a module called `StdEnv`. As of February 2023, four versions of standard environments were available: 2023, 2020, 2018.3, and 2016.4.  Each version includes significant improvements.  We only support versions 2023 and 2020.  This document details the differences between versions and explains why using the most recent version is recommended.  The latest versions of software packages are usually installed in the latest software environment.


## StdEnv/2023

This latest iteration of our software environment uses GCC 12.3.0, Intel 2023.1, and Open MPI 4.1.5 by default.

To activate this environment, run the command:

```bash
module load StdEnv/2023
```

### Performance Improvements

The minimum supported CPU instruction set is AVX2, generally `x86-64-v3`. Even the compatibility layer providing basic Linux commands is compiled with optimizations for these instructions.

### Changes to Default Modules

The default compiler is now GCC instead of Intel. We compile with Intel only applications that demonstrate better performance with Intel. CUDA is now an extension of OpenMPI rather than the other way around; MPI for CUDA is loaded at startup if CUDA is loaded. This allows sharing multiple MPI libraries across all branches (CUDA or not).

The default versions of the following modules have been updated:

* GCC 9.3 => GCC 12.3
* OpenMPI 4.0.3 => OpenMPI 4.1.5
* Intel Compilers 2020 => 2023
* Intel MKL 2020 => Flexiblas 3.3.1 (with MKL 2023 or BLIS 0.9.0)
* CUDA 11 => CUDA 12


## StdEnv/2020

This version of our software environment underwent the most significant changes. The default compilers were changed to GCC 9.3.0 and Intel 2020.1. The default MPI implementation was changed to Open MPI 4.0.3.

Activate this environment with the command:

```bash
module load StdEnv/2020
```

### Performance Improvements

Binaries generated with the Intel compiler automatically support AVX2 and AVX512 instruction sets. Technically, these are multi-architecture binaries, also called *fat binaries*. This means that when using a cluster like Cedar or Graham, which have seen multiple processor generations, you no longer need to manually load an `arch` module if you are using software packages generated with the Intel compiler.

Some software packages previously installed with GCC or Intel are now at a lower level in the hierarchy, making the same module visible regardless of the loaded compiler; this is the case, for example, for the `R` modules and several bioinformatics packages for which the `gcc` module previously had to be loaded. This was made possible by CPU architecture-specific optimizations performed below the compiler level.

We also installed a newer version of the GNU C Library, which offers optimized mathematical functions. This required a newer version of the Linux kernel (see below).

### Compatibility Layer

The compatibility layer is a level below the compilers and software packages to make them independent of the underlying operating system and to work on CentOS, Ubuntu, or Fedora. A major change in version 2020 was switching the compatibility layer tool from the Nix package manager to Gentoo Prefix.

### Linux Kernel

Versions 2016.4 and 2018.3 require a Linux kernel version 2.6.32 or higher, supported from CentOS 6. Version 2020 requires a Linux kernel 3.10 or higher, supported from CentOS 7. Other Linux distributions usually have a much newer kernel, so you won't need to change your Linux distribution if you're using this standard environment on a distribution other than CentOS.

### Module Extensions

With the 2020 environment, we started installing several Python extensions in the corresponding main modules. For example, PyQt5 was installed in the `qt/5.12.8` module to support multiple Python versions. The module system has been modified to allow you to easily find this type of extension. For example, with:

```bash
module spider pyqt5
```

you will know that you can obtain the `qt/5.12.8` module.


## StdEnv/2018.3

**Obsolete** This environment is no longer supported.

This second version of our software environment was installed in 2018, with the commissioning of the Béluga cluster, shortly after the deployment of Niagara. The default compilers were changed to GCC 7.3.0 and Intel 2018.3. The default MPI implementation was changed to Open MPI 3.1.2. This is the first version to offer support for AVX512 instructions.

Activate this environment with the command:

```bash
module load StdEnv/2018.3
```


## StdEnv/2016.4

**Obsolete** This environment is no longer supported.

This first version of our software environment was installed in 2016 with the commissioning of the Cedar and Graham clusters. The default compilers are GCC 5.4.0 and Intel 2016.4. The default MPI implementation is Open MPI 2.1.1. Most software compiled in this environment does not support AVX512 instructions, unlike the Skylake processors of Béluga, Niagara, and recent additions to Cedar and Graham.

Activate this environment with the command:

```bash
module load StdEnv/2016.4
```
