# Autotools

Other languages: English français

## Description

`autoconf` is a tool belonging to the `autotools` suite, also known as the GNU build system.  It automates generating custom Makefiles for building programs on different systems, potentially with different compilers.

When building a program using `autoconf`, the first step is to call the `configure` script:

```bash
[name@server ~]$ ./configure
```

This verifies the presence and appropriate versions of compilers and other software, generating a system-specific Makefile.  Next, call `make` as usual:

```bash
[name@server ~]$ make
```

Finally, `make install` installs the files. To install only for yourself, not all server users, specify the installation location:

```bash
[name@server ~]$ mkdir $HOME/SOFTWARE
[name@server ~]$ make install --prefix=$HOME/SOFTWARE
```

In some cases, supply the `--prefix` option to `./configure` instead of `make`; consult the specific software's documentation. You might also need to [create a module](https://docs.alliancecan.ca/mediawiki/index.php?title=Autotools&oldid=9100) to show the system the paths to your newly installed software.

A basic compilation using `autoconf` is:

```bash
[name@server ~]$ ./configure && make && make install --prefix=$HOME/SOFTWARE
```

## Frequently Used Options for `configure` Scripts

`configure` scripts accept many options, varying by project. However, some are common:

Always run:

```bash
[name@server ~]$ ./configure --help
```

to get a detailed list of supported options.

### Installation Directory

The `--prefix` option always exists. It defines the `make install` installation directory. For example, to install to the `programs` subdirectory within your home directory:

```bash
[name@server ~]$ ./configure --prefix=$HOME/programs/
```

### Feature Options

Most scripts enable/disable features using `--enable-feature` or `--disable-feature`.  Advanced computing options often include parallelization (threads or MPI):

```bash
[name@server ~]$ ./configure --enable-mpi
```

or

```bash
[name@server ~]$ ./configure --enable-threads
```

Options like `--with-...` configure specific features.  Generally, avoid these and let `autoconf` find parameters automatically, but sometimes `--with-...` options are necessary. For example:

```bash
[name@server ~]$ ./configure --enable-mpi --with-mpi-dir=$MPIDIR
```

### Options Defined by Variables

Specify the compiler and its options by declaring variables after the `./configure` command. To define the C compiler and its options:

```bash
[name@server ~]$ ./configure CC=icc CFLAGS="-O3 -xHost"
```

Commonly used variables:

| Option     | Description                                                              |
|-------------|--------------------------------------------------------------------------|
| `CFLAGS`    | Options to pass to the C compiler                                        |
| `CPPFLAGS`  | Options to pass to the preprocessor and C, C++, Objective C, Objective C++ compilers |
| `CXXFLAGS`  | Options to pass to the C++ compiler                                      |
| `DEFS`      | Define a preprocessor macro                                              |
| `FCFLAGS`   | Options to pass to the Fortran compiler                                  |
| `FFLAGS`    | Options to pass to the Fortran 77 compiler                               |
| `LDFLAGS`   | Options to pass to the linker                                            |
| `LIBS`      | Libraries to link                                                        |

A more exhaustive list is available in the [autoconf documentation](https://docs.alliancecan.ca/mediawiki/index.php?title=Autotools&oldid=9100).
